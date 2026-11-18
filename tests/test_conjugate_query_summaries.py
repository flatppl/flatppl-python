import json
import math
import re
from pathlib import Path

import jax
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


@pytest.mark.parametrize(
    "domain,prior,law,cell,point,events",
    [
        (
            "unitinterval",
            "Beta(h,2.)",
            "Bernoulli(t)",
            "booleans",
            "0.5",
            (True, False),
        ),
        ("unitinterval", "Beta(h,2.)", "Binomial(2,t)", "integers", "0.5", (0, 2)),
        ("posreals", "Gamma(h,3.)", "Poisson(t)", "integers", "1.", (0, 2)),
        ("reals", "Normal(0.5,h)", "Normal(t,h)", "reals", "0.25", (-1.0, 1.5)),
        (
            "posreals",
            "InverseGamma(h,3.)",
            "Normal(h,sqrt(t))",
            "reals",
            "3.",
            (-1.0, 2.0),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([h,3.,4.])",
            "Categorical(t)",
            "integers",
            "[0.45,0.1,0.45]",
            (1, 3),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([h,3.,4.])",
            "Categorical0(t)",
            "integers",
            "[0.45,0.1,0.45]",
            (0, 2),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([h,3.,4.])",
            "Multinomial(2,t)",
            "cartpow(integers,3)",
            "[0.6,0.2,0.2]",
            ((2, 0, 0), (0, 1, 1)),
        ),
    ],
    ids=[
        "bernoulli",
        "binomial",
        "poisson",
        "normal-mean",
        "normal-variance",
        "categorical",
        "categorical0",
        "multinomial",
    ],
)
def test_prepared_posterior_scores_and_draw_use_fixed_size_summaries(
    domain: str,
    prior: str,
    law: str,
    cell: str,
    point: str,
    events: tuple[int | float | tuple[int, int, int], ...],
) -> None:
    layouts: list[tuple[tuple[tuple[int, ...], ...], int]] = []
    with jax.enable_x64():
        for size in (31, 257):
            query = flatppl(f"""
                t = elementof({domain})
                h = external(posreals)
                y = external(cartpow({cell},{size}))
                seed = external(nonnegintegers)
                L = likelihoodof(functionof(iid({law},{size}),t=t),y)
                Q = normalize(bayesupdate(L,joint(t={prior})))
                point = record(t={point})
                inputs = (h,y,seed)
                outputs = (densityof(Q,point),logdensityof(Q,point),rand(Q,rnginit(seed)))
            """)
            preparation, online, names = query.prepare_likelihood("y")
            score = online.compile(dtype="float64", autodiff=False)
            fields = score.schema["inputs"]
            assert names and [field["name"] for field in fields] == [
                "h",
                "seed",
                *names,
            ]
            shapes = tuple(tuple(field["value"]["shape"]) for field in fields[2:])
            layouts.append((shapes, score.stats()["total"]))
            assert not re.search(rf"tensor<(?:\d+x)*{size}x", score.stablehlo)

            if size == 31:
                prepare = preparation.compile(dtype="float64", autodiff=False)
                assert [field["name"] for field in prepare.schema["inputs"]] == ["y"]
                direct = query.compile(dtype="float64", autodiff=False)
                data = np.take(
                    np.asarray(events), np.arange(size) % len(events), axis=0
                )
                result = prepare(data)
                summaries = result if isinstance(result, tuple) else (result,)
                density, log_density, (sample, state) = score(2.0, 42, *summaries)
                expected, log_expected, (draw, next_state) = direct(2.0, data, 42)
                np.testing.assert_allclose(
                    [density, log_density],
                    [expected, log_expected],
                    rtol=2e-12,
                    atol=2e-12,
                )
                assert float(density) == pytest.approx(
                    math.exp(float(log_density)), rel=2e-12
                )
                np.testing.assert_allclose(
                    sample["t"], draw["t"], rtol=2e-12, atol=2e-12
                )
                np.testing.assert_array_equal(state, next_state)
    assert layouts[0] == layouts[1]


def test_bound_and_runtime_observations_in_ordinary_density_queries() -> None:
    model = flatppl("""
        t = elementof(unitinterval)
        x = elementof(unitinterval)
        y = external(cartpow(integers,2))
        L = likelihoodof(functionof(iid(Binomial(2,t),2),t=t),y)
        P = bayesupdate(L,joint(t=Beta(1.,1.)))
        Q = normalize(P)
        inputs = (x,y)
        outputs = (densityof(P,record(t=x)),logdensityof(P,record(t=x)),
                   densityof(Q,record(t=x)),logdensityof(Q,record(t=x)))
    """)
    data = np.array([1, 1])
    bound = model.set(y=data)
    with jax.enable_x64():
        runtime = model.compile(dtype="float64")
        np.testing.assert_allclose(
            runtime(0.5, data),
            [0.25, math.log(0.25), 15 / 8, math.log(15 / 8)],
            atol=2e-12,
        )
        data[:] = 0
        np.testing.assert_allclose(
            runtime(0.5, data),
            [1 / 16, math.log(1 / 16), 5 / 16, math.log(5 / 16)],
            atol=2e-12,
        )
        for module, probability, slope in (
            (bound, 15 / 8, 0.0),
            (model.set(y=data), 5 / 16, -8.0),
        ):
            for enabled in (True, False):
                score = module.logdensity(
                    "Q",
                    unconstrained=False,
                    dtype="float64",
                    sufficient_statistics=enabled,
                )
                assert float(score({"t": 0.5})) == pytest.approx(
                    math.log(probability), abs=2e-12
                )
                assert float(score.grad({"t": 0.5})["t"]) == pytest.approx(
                    slope, abs=2e-12
                )


def test_prepared_binomial_likelihood_retains_allocation_support_and_derivatives() -> (
    None
):
    query = flatppl("""
        p = elementof(reals)
        y = external(cartpow(reals,2))
        inputs = (p,y)
        outputs = logdensityof(iid(Binomial(2,p),2),y)
    """)
    preparation, online, _ = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64")
        score = online.compile(dtype="float64", checks="raise")
        for point, observations, value, slope in [
            (0.5, [0.0, 2.0], math.log(1 / 16), 0.0),
            (0.5, [1.0, 1.0], math.log(1 / 4), 0.0),
            (0.0, [0.0, 0.0], 0.0, -4.0),
            (1.0, [2.0, 2.0], 0.0, 4.0),
            (0.25, [-1.0, 2.0], -math.inf, 8.0),
            (0.25, [0.5, 3.0], -math.inf, 0.0),
        ]:
            result = prepare(np.asarray(observations))
            summaries = result if isinstance(result, tuple) else (result,)
            value_at, grad = jax.value_and_grad(lambda p: score(p, *summaries))(point)
            assert float(value_at) == pytest.approx(value, abs=2e-12)
            assert float(grad) == pytest.approx(slope, abs=2e-12)
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            score(2.0, *summaries)


def test_binomial_allocation_depends_on_each_trial_count() -> None:
    query = flatppl("""
        t = elementof(unitinterval)
        p = elementof(unitinterval)
        n = external(posintegers)
        y = external(cartpow(integers,2))
        prior = joint(t=Beta(1.,1.))
        P2 = bayesupdate(likelihoodof(functionof(iid(Binomial(2,t),2),t=t),y),prior)
        P3 = bayesupdate(likelihoodof(functionof(iid(Binomial(3,t),2),t=t),y),prior)
        Pn = bayesupdate(likelihoodof(functionof(iid(Binomial(n,t),2),t=t),y),prior)
        inputs = (p,n,y)
        point = record(t=p)
        outputs = (densityof(P2,point), logdensityof(P3,point), densityof(Pn,point))
    """)
    preparation, online, names = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64")
        score = online.compile(dtype="float64")
        # The live trial count still needs each row for its binomial coefficient.
        assert [f["name"] for f in score.schema["inputs"]] == ["p", "n", "y", *names]
        for counts in ((0, 2), (1, 1), (0, 3)):
            data = np.array(counts)
            result = prepare(data)
            summaries = result if isinstance(result, tuple) else (result,)
            for n in (3, 4):
                expected = [
                    math.prod(math.comb(trials, k) for k in counts) / 4**trials
                    for trials in (2, 3, n)
                ]
                expected[1] = math.log(expected[1])
                np.testing.assert_allclose(
                    score(0.5, n, data, *summaries), expected, rtol=2e-12, atol=2e-12
                )


@pytest.mark.parametrize("law,base", [("Categorical", 1), ("Categorical0", 0)])
def test_prepared_categorical_keeps_ordered_scores_and_unused_zero_weights(
    law: str, base: int
) -> None:
    query = flatppl(f"""
        t = elementof(reals)
        y = external(cartpow(reals,6))
        p = [t,0.,1.-t]
        inputs = (t,y)
        outputs = logdensityof(iid({law}(p),6),y)
    """)
    preparation, online, names = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64", checks="raise")
        native = query.compile(
            dtype="float64", checks="raise", sufficient_statistics=False
        )
        assert [f["name"] for f in score.schema["inputs"]] == ["t", *names]
        for point, labels, expected, slope in [
            (
                0.25,
                [0, 2, 0, 2, 0, 0],
                4 * math.log(0.25) + 2 * math.log(0.75),
                4 / 0.25 - 2 / 0.75,
            ),
            (0.25, [0, 2, 0.5, 3, -1, math.nan], -math.inf, 1 / 0.25 - 1 / 0.75),
            (0.0, [2] * 6, 0.0, -6.0),
            (1.0, [0] * 6, 0.0, 6.0),
        ]:
            data = np.asarray(labels, dtype=float) + base
            result = prepare(data)
            summaries = result if isinstance(result, tuple) else (result,)
            for f in (lambda t: score(t, *summaries), lambda t: native(t, data)):
                value, derivative = jax.value_and_grad(f)(point)
                assert float(value) == pytest.approx(expected, abs=2e-12)
                assert float(derivative) == pytest.approx(slope, abs=2e-12)
        for label in (base, base - 1):
            result = prepare(np.full(6, label, dtype=float))
            summaries = result if isinstance(result, tuple) else (result,)
            with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
                score(-0.5, *summaries)


def test_bound_category_histogram_folds_only_when_precisions_agree() -> None:
    # More than 64 labels bypass the repeated-value rule.
    categories, size = 80, 4099
    query = flatppl(f"""
        p = elementof(stdsimplex({categories}))
        y = external(cartpow(reals,{size}))
        inputs = (p,y)
        outputs = logdensityof(iid(Categorical(p),{size}),y)
    """)
    labels = (1 + np.arange(size) % categories).astype(float)
    probabilities = np.full(categories, 1 / categories)
    expected = -size * math.log(categories)
    with jax.enable_x64():
        bound = query.set(y=labels)
        score = bound.compile(dtype="float64", autodiff=False)
        assert any(row.rule == "category counts" for row in bound.analyze().reductions)
        assert "stablehlo.scatter" not in score.stablehlo
        assert float(score(probabilities)) == pytest.approx(expected, rel=2e-12)

        labels[0] += 2**-30
        rounded = query.set(y=labels)
        assert (
            float(rounded.compile(dtype="float64", autodiff=False)(probabilities))
            == -math.inf
        )
        assert float(
            rounded.compile(dtype="float32", autodiff=False)(probabilities)
        ) == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("law,base", [("Categorical", 1), ("Categorical0", 0)])
def test_file_labels_fold_for_ordinary_and_normalized_scores(
    tmp_path: Path, law: str, base: int
) -> None:
    size, categories = 4099, 80
    labels = (np.arange(size) % categories + base).tolist()
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    query = flatppl(
        f"""
        p = elementof(stdsimplex({categories}))
        t = elementof(stdsimplex({categories}))
        y = load_data("labels.json",cartpow(integers,{size}))
        L = likelihoodof(functionof(iid({law}(t),{size}),t=t),y)
        P = bayesupdate(L,joint(t=Dirichlet(fill(1.,{categories}))))
        inputs = p
        outputs = (logdensityof(P,record(t=p)),logdensityof(normalize(P),record(t=p)))
    """,
        source_path=tmp_path / "query.flatppl",
    )
    point = np.full(categories, 1 / categories)
    counts = np.bincount(np.asarray(labels) - base, minlength=categories)
    with jax.enable_x64():
        score = query.compile(dtype="float64", autodiff=False)
        assert "stablehlo.scatter" not in score.stablehlo
        assert f"tensor<{size}x" not in score.stablehlo
        raw, normalized = score(point)
        likelihood = -size * math.log(categories)
        assert float(raw) == pytest.approx(
            math.lgamma(categories) + likelihood, abs=2e-9
        )
        assert float(normalized) == pytest.approx(
            math.lgamma(size + categories)
            - sum(math.lgamma(k + 1) for k in counts)
            + likelihood,
            abs=2e-9,
        )
