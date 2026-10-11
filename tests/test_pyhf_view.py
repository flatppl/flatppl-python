"""`flatppl.pyhf` against pyhf 0.7.6 on a two-channel, two-measurement workspace.

The fixture covers every modifier type, a normsys and histosys shared name,
staterror sorting, lumi, a shapesys bin and a staterror bin without
uncertainty, and every measurement override field.
"""

import json
import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pyhf
import pytest

from flatppl import CompilationError, flatppl, from_pyhf
from flatppl.pyhf import Channel, ParameterSlice, PyhfView

SOURCE = (Path(__file__).parent / "data" / "pyhf_measurements.json").read_text()
MEASUREMENTS = ["first", "second set"]
MODES = ["bound", "external"]


def reference(measurement: str) -> Any:
    # pyhf logs a warning when a workspace has several measurements.
    logging.getLogger("pyhf").setLevel(logging.ERROR)
    workspace = pyhf.Workspace(json.loads(SOURCE))
    return workspace, workspace.model(measurement_name=measurement)


def point(model: Any) -> np.ndarray:
    # Away from the suggested values, so every parameter moves the density.
    init = np.asarray(model.config.suggested_init(), dtype=np.float64)
    return init + 0.05 * np.arange(1, init.size + 1) / init.size


@pytest.mark.parametrize("measurement", MEASUREMENTS)
def test_view_reads_pyhf_layout_and_settings(measurement: str) -> None:
    _, model = reference(measurement)
    view = from_pyhf(SOURCE, measurement=measurement).pyhf
    config = model.config
    assert isinstance(view, PyhfView)
    assert view.measurement == measurement
    assert view.par_order == tuple(config.par_order)
    assert view.npars == config.npars
    assert view.poi == config.poi_name
    assert view.poi_index == config.poi_index
    for name in config.par_order:
        types = {kind for n, kind in config.modifiers if n == name}
        # A name that normsys and histosys share is one histosys paramset.
        kind = "histosys" if types == {"normsys", "histosys"} else types.pop()
        assert view.par_map[name] == ParameterSlice(
            name, config.par_map[name]["slice"], kind
        )
    np.testing.assert_array_equal(view.suggested_init(), config.suggested_init())
    np.testing.assert_array_equal(view.suggested_bounds(), config.suggested_bounds())
    assert view.suggested_fixed() == list(config.suggested_fixed())
    assert view.auxdata_order == tuple(config.auxdata_order)
    np.testing.assert_allclose(view.auxdata, config.auxdata, rtol=1e-15)
    assert view.channels == tuple(
        Channel(
            name, config.channel_nbins[name], f"{name}_observed", f"{name}_expected"
        )
        for name in config.channels
    )
    assert dict(view.channel_slices) == {
        name: config.channel_slices[name] for name in config.channels
    }


@pytest.mark.parametrize("measurement", MEASUREMENTS)
@pytest.mark.parametrize("observations", MODES)
@pytest.mark.parametrize("auxiliary", MODES)
def test_logpdf_and_expected_data_match_pyhf(
    measurement: str, observations: str, auxiliary: str
) -> None:
    workspace, model = reference(measurement)
    data = np.asarray(workspace.data(model), dtype=np.float64)
    pars = point(model)
    main = sum(model.config.channel_nbins.values())
    passed = {
        ("bound", "bound"): (),
        ("external", "bound"): (data[:main],),
        ("bound", "external"): (data[main:],),
        ("external", "external"): (data,),
    }[observations, auxiliary]
    module = from_pyhf(
        SOURCE,
        measurement=measurement,
        observations=observations,  # pyright: ignore[reportArgumentType]
        auxiliary=auxiliary,  # pyright: ignore[reportArgumentType]
    )
    with jax.enable_x64():
        logpdf = module.pyhf.logpdf(dtype="float64")
        expected = module.pyhf.expected_data(dtype="float64")
        np.testing.assert_allclose(
            logpdf(pars, *passed), model.logpdf(pars, data)[0], rtol=1e-12
        )
        np.testing.assert_allclose(
            expected(pars),
            model.expected_data(pars, include_auxdata=False),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            module.pyhf.expected_data(dtype="float64", include_auxdata=True)(pars),
            model.expected_data(pars),
            rtol=1e-12,
        )
        gradient = jax.grad(logpdf)(pars, *passed)
        batch = np.stack([pars, 1.01 * pars])
        batched = jax.vmap(lambda p: logpdf(p, *passed))(batch)
        np.testing.assert_allclose(
            batched, [logpdf(p, *passed) for p in batch], rtol=1e-12
        )
    assert gradient.shape == pars.shape


def test_set_external_observations_and_centres_match_the_literal_import() -> None:
    _, model = reference("first")
    data = np.asarray(pyhf.Workspace(json.loads(SOURCE)).data(model))
    literal = from_pyhf(SOURCE)
    external = from_pyhf(SOURCE, observations="external", auxiliary="external")
    view = external.pyhf
    observed = {
        channel.observed: data[view.channel_slices[channel.name]]
        for channel in view.channels
    }
    # The nominal centres come from the settings record through Parameters.
    centres = {
        p.name: p.centre for p in external.parameters.values() if p.centre is not None
    }
    assert sorted(centres) == sorted(view.par_map[n].name for n in view.auxdata_order)
    bound = external.set(**observed, pyhf_auxdata=centres)
    pars = point(model)
    with jax.enable_x64():
        mine = bound.pyhf.logpdf(dtype="float64")
        theirs = literal.pyhf.logpdf(dtype="float64")
        np.testing.assert_allclose(mine(pars), theirs(pars), rtol=1e-14)


def test_parameters_carry_converter_settings_and_original_names() -> None:
    renamed = json.loads(SOURCE)
    renamed["channels"][0]["samples"][1]["modifiers"][2]["name"] = "k-factor"
    renamed["measurements"][1]["config"]["poi"] = "k-factor"
    module = from_pyhf(json.dumps(renamed), measurement="second set")
    parameters = module.parameters
    k = parameters["k_factor"]
    assert (k.init, k.lo, k.hi, k.fixed) == (1.0, 0.0, 10.0, False)
    assert (k.constraint, k.width, k.centre) == (None, None, None)
    assert (k.original_name, k.role) == ("k-factor", "poi")
    mu = parameters["mu"]
    assert (mu.init, mu.lo, mu.hi, mu.fixed, mu.role) == (2.0, -1.0, 5.0, True, None)
    sh = parameters["sh"]
    np.testing.assert_array_equal(sh.fixed, [False, True])
    np.testing.assert_allclose(np.asarray(sh.width), [0.1, 1.0])
    np.testing.assert_allclose(np.asarray(sh.centre), [25.0, 1.0])
    assert (sh.constraint, parameters["lumi"].constraint) == ("poisson", "normal")
    assert module.pyhf.par_map["sh"].type == "shapesys"
    assert module.pyhf.poi == "k-factor"
    assert module.pyhf.par_map["k-factor"].name == "k_factor"
    plain = flatppl("x = elementof(reals)").parameters["x"]
    assert (plain.init, plain.original_name, plain.role) == (None, None, None)


def test_import_options_are_checked() -> None:
    with pytest.raises(
        CompilationError, match="no measurement named `third`"
    ) as caught:
        from_pyhf(SOURCE, measurement="third")
    assert caught.value.stage == "pyhf"
    with pytest.raises(ValueError, match="observations"):
        from_pyhf(SOURCE, observations="literal")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="auxiliary"):
        from_pyhf(SOURCE, auxiliary="literal")  # pyright: ignore[reportArgumentType]


def test_logpdf_checks_its_data_and_modules_without_a_record_have_no_view() -> None:
    _, model = reference("first")
    pars = point(model)
    with pytest.raises(ValueError, match="no external data"):
        from_pyhf(SOURCE).pyhf.logpdf()(pars, [1.0])
    external = from_pyhf(SOURCE, observations="external").pyhf.logpdf()
    with pytest.raises(ValueError, match="pass data with 3 values"):
        external(pars)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        external(pars, [1.0, 2.0])
    with pytest.raises(AttributeError, match="no pyhf settings record"):
        _ = flatppl("x = elementof(reals)").pyhf
    assert not hasattr(flatppl("pyhf = 1.0"), "pyhf")
