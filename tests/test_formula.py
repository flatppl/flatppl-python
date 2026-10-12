"""Formula imports keep the Rust recipe through host-table preparation."""

import gc
import json
import math
import re
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from flatppl import CompilationError, Module, flatppl, from_formula

PRIORS = [
    {"class": name, "prior": "Normal(0, 2)"} for name in ("b", "Intercept", "sigma")
]


def _training(producer: str) -> Any:
    columns = {
        "x": [1.0, 3.0, None, 5.0, 9.0],
        "y": [2.0, 4.0, 8.0, 5.0, None],
        "f": ["low", "high", "low", "low", "high"],
        "unused": [None, 0.0, None, 0.0, None],
    }
    levels = ["high", "low"]
    if producer == "pandas":
        table = pd.DataFrame(columns, index=[10, 20, 30, 40, 50])
        table["f"] = pd.Categorical(columns["f"], categories=levels)
        return table
    if producer == "polars":
        return pl.DataFrame(columns).with_columns(pl.col("f").cast(pl.Enum(levels)))
    return pa.table(
        {
            **columns,
            "f": pa.DictionaryArray.from_arrays([1, 0, 1, 1, 0], levels),
        }
    )


def _posterior(model: Module) -> Any:
    model.context.register("training.flatppl", model)
    return flatppl(
        """
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,2), sigma=posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    ).compile(autodiff=False)


@pytest.mark.parametrize("producer", ["pandas", "polars", "arrow"])
def test_formula_tables_retain_categories_null_rows_and_normal_score(
    producer: str,
) -> None:
    model = from_formula(
        "y ~ x + f",
        _training(producer),
        priors=PRIORS,
        contrasts={"f": "treatment"},
    )
    assert model.formula.rows == (0, 1, 3)
    assert model.formula.info["design_columns"] == ("x", "flow")
    # Explicit treatment coding uses high as the reference for every producer,
    # including Polars Enum, whose Arrow dictionary is marked ordered.
    design = np.array([[-2, 1 / 3], [0, -2 / 3], [2, 1 / 3]])
    np.testing.assert_allclose(
        model.query("X").compile(autodiff=False)(), design, atol=1e-7
    )
    point = {"temp_Intercept": 1.0, "b": [0.5, -1.0], "sigma": 2.0}
    mean = 1 + design @ np.array([0.5, -1.0])
    # Three coefficient priors plus a positive half-Normal scale prior.
    normalizer = -math.log(2 * math.sqrt(2 * math.pi))
    prior = 4 * normalizer - (1 + 0.5**2 + 1 + 2**2) / 8 + math.log(2)
    likelihood = 3 * normalizer - np.sum((np.array([2, 4, 5]) - mean) ** 2) / 8
    score = _posterior(model)
    np.testing.assert_allclose(score(point), prior + likelihood, atol=3e-6)


def test_prediction_reuses_training_recipe_without_response_or_mutation() -> None:
    model = from_formula("y ~ x + f", _training("arrow"), priors=PRIORS)
    training_matrix = model.query("X").compile(autodiff=False)
    before = np.asarray(training_matrix())
    new_data = pa.table(
        {
            "x": [7.0, 11.0, 13.0],
            # The prediction dictionary reverses the training level order.
            "f": pa.DictionaryArray.from_arrays([0, None, 1], ["low", "high"]),
        }
    )
    prediction = model.formula.prepare(new_data)
    assert prediction.formula.rows == (0, 2)
    assert prediction.context is model.context
    expected = np.array([[4, 1 / 3], [10, -2 / 3]])
    np.testing.assert_allclose(
        prediction.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    bound = prediction.set(y_data=[3.0, 7.0])
    model.context.register("prediction.flatppl", bound)
    evaluate = flatppl(
        """
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,2), sigma=posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [3.0,7.0]))
        """,
        context=model.context,
    ).compile(autodiff=False)
    mean = 1 + expected @ np.array([0.5, -1.0])
    expected_score = -2 * math.log(2 * math.sqrt(2 * math.pi))
    expected_score -= np.sum((np.array([3, 7]) - mean) ** 2) / 8
    np.testing.assert_allclose(
        evaluate({"temp_Intercept": 1.0, "b": [0.5, -1.0], "sigma": 2.0}),
        expected_score,
        atol=2e-6,
    )
    # Binding observations and slicing a prepared module retain its recipe.
    repeated = bound.slice("K_model").formula.prepare(new_data)
    np.testing.assert_allclose(
        repeated.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    np.testing.assert_array_equal(training_matrix(), before)
    assert model.formula.rows == (0, 1, 3)
    with pytest.raises(CompilationError):
        model.formula.prepare({"x": [1.0], "f": ["unseen"]})


def test_ordered_factors_and_observed_preparation_keep_training_defaults() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 3.0, 2.0],
            "f": pd.Categorical(
                ["low", "high", "low"], categories=["low", "high"], ordered=True
            ),
        }
    )
    model = from_formula("y ~ f", data)
    # R's orthonormal linear contrast at two ordered levels is (-1,1)/sqrt(2).
    expected = np.array([[-2 / 3], [4 / 3], [-2 / 3]]) / math.sqrt(2)
    np.testing.assert_allclose(
        model.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    observed = model.formula.prepare(
        {"y": [100.0, None], "f": ["high", "low"]}, mode="training"
    )
    assert observed.formula.rows == (0,)
    assert observed.formula.recipe == model.formula.recipe
    np.testing.assert_array_equal(observed.constant("y_data"), [100.0])
    assert observed.formula.info["has_improper_prior"] is True


def test_formula_paths_retain_reloadable_data_and_immutable_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLATPPL_CACHEDIR", "cache")
    training = tmp_path / "training.csv"
    training.write_text("y,x,f\n2,1,low\n4,3,high\n8,,low\n5,5,low\n,9,high\n")
    model = from_formula("y ~ x + f", training, priors=PRIORS)
    source = model.source
    match = re.search(r'load_data\(\s*("[^"\n]*")', source)
    assert match is not None
    sidecar = Path(json.loads(match[1]))
    assert sidecar.is_absolute() and sidecar.is_relative_to(
        (tmp_path / "cache").resolve()
    )
    expected_training = np.array([[-2, 1 / 3], [0, -2 / 3], [2, 1 / 3]])
    before = model.query("X").compile(autodiff=False)
    np.testing.assert_allclose(before(), expected_training, atol=1e-7)
    snapshot = np.asarray(before()).copy()
    fresh = from_formula("y ~ x + f", str(training), priors=PRIORS)
    assert fresh.source == source

    prediction_data = pa.table({"x": [7.0, 11.0], "f": ["low", "high"]})
    prediction_path = tmp_path / "prediction.arrow"
    with pa.OSFile(str(prediction_path), "wb") as sink:
        with pa.ipc.new_file(sink, prediction_data.schema) as writer:
            writer.write_table(prediction_data)
    prediction = model.formula.prepare(prediction_path)
    expected_prediction = np.array([[4, 1 / 3], [8, -2 / 3]])
    np.testing.assert_allclose(
        prediction.query("X").compile(autodiff=False)(), expected_prediction, atol=1e-7
    )
    assert prediction.formula.recipe == model.formula.recipe
    assert prediction.context is model.context
    in_memory = model.formula.prepare(prediction_data)
    assert "load_data(" not in in_memory.source
    np.testing.assert_allclose(
        in_memory.query("X").compile(autodiff=False)(), expected_prediction, atol=1e-7
    )

    observed_path = tmp_path / "observed.json"
    observed_path.write_text('{"x":[7.0],"f":["low"],"y":[10.0]}')
    observed = model.formula.prepare(observed_path, mode="training")
    np.testing.assert_array_equal(
        observed.query("y_data").compile(autodiff=False)(), [10]
    )
    np.testing.assert_array_equal(before(), snapshot)
    # Compiled programs retain their source modules for later batching. The
    # uncompiled owner verifies that the cached file survives its Context.
    context = weakref.ref(fresh.context)
    training.unlink()
    del model, prediction, observed, in_memory, before, fresh
    gc.collect()
    assert context() is None
    monkeypatch.chdir(tmp_path.parent)
    np.testing.assert_allclose(
        flatppl(source).query("X").compile(autodiff=False)(),
        expected_training,
        atol=1e-7,
    )
    assert sidecar.is_file()
