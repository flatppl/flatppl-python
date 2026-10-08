import io
import json
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import pytest

from flatppl import CompilationError, Context, flatppl, from_pyhf


def test_pyhf_input_forms_keep_streams_open(tmp_path):
    spec = pyhf.simplemodels.uncorrelated_background([5, 10], [50, 60], [5, 12]).spec
    text = json.dumps(spec)
    path = tmp_path / "model.json"
    path.write_text(text, encoding="utf-8-sig")
    stream = io.StringIO("prefix" + text)
    stream.seek(len("prefix"))
    with path.open() as file:
        sources = [text, path, str(path), stream, io.BytesIO(text.encode("utf-8-sig")), file]
        for source in sources:
            model = from_pyhf(source)
            model.context.register("model.flatppl", model)
            evaluate = flatppl('''
                mu = elementof(reals)
                gamma = elementof(cartpow(posreals, 2))
                m = load_module("model.flatppl", mu = mu, uncorr_bkguncrt = gamma)
                inputs = (mu, gamma)
                outputs = m.singlechannel_expected
            ''', context=model.context).compile()
            np.testing.assert_allclose(evaluate(2., [1., 1.]), [60, 80])
            if hasattr(source, "read"):
                assert not source.closed


def test_pyhf_model_runtime_observations_and_workspace_match_full_logpdf():
    reference = pyhf.simplemodels.uncorrelated_background([5, 10], [50, 60], [5, 12])
    context = Context()
    model = from_pyhf(json.dumps(reference.spec), context=context)
    context.register("model.flatppl", model)
    evaluate = flatppl('''
        p = elementof(cartprod(mu = reals, uncorr_bkguncrt = cartpow(posreals, 2)))
        data = external(cartpow(nonnegreals, 2))
        m = load_module("model.flatppl", singlechannel_observed = data)
        forward = load_module("model.flatppl", mu = p.mu, uncorr_bkguncrt = p.uncorr_bkguncrt)
        inputs = (p, data)
        outputs = record(expected = forward.singlechannel_expected,
                         logpdf = logdensityof(m.likelihood, p))
    ''', context=context).compile()
    workspace = dict(reference.spec, observations=[{"name": "singlechannel", "data": [50, 60]}],
                     measurements=[{"name": "measurement", "config": {"poi": "mu", "parameters": []}}],
                     version="1.0.0")
    context.register("workspace.flatppl", from_pyhf(json.dumps(workspace), context=context))
    fixed = flatppl('''
        m = load_module("workspace.flatppl")
        p = elementof(cartprod(mu = reals, uncorr_bkguncrt = cartpow(posreals, 2)))
        inputs = p
        outputs = logdensityof(m.likelihood, p)
    ''', context=context).compile()
    for mu, gamma, data in [(1., [1., 1.], [50., 60.]), (1.7, [.8, 1.2], [52.5, 63.25])]:
        p = {"mu": jnp.float32(mu), "uncorr_bkguncrt": jnp.array(gamma, dtype=jnp.float32)}
        actual = jax.jit(evaluate)(p, data)
        pars = np.asarray([mu, *gamma])
        expected = np.asarray(reference.expected_actualdata(pars))
        score = reference.logpdf(pars, [*data, *reference.config.auxdata])[0]
        np.testing.assert_allclose(actual["expected"], expected, rtol=2e-6)
        np.testing.assert_allclose(actual["logpdf"], score, atol=1e-4, rtol=0)
        analytic_counts = mu * np.array([5., 10.]) + np.array(gamma) * [50., 60.]
        rates = np.concatenate([analytic_counts, np.array(gamma) * [100., 25.]])
        counts = np.array([*data, 100., 25.])
        analytic = sum(
            k * math.log(rate) - rate - math.lgamma(k + 1)
            for k, rate in zip(counts, rates)
        )
        np.testing.assert_allclose(actual["logpdf"], analytic, atol=1e-4, rtol=0)
        np.testing.assert_allclose(
            fixed(p), reference.logpdf(pars, [50., 60., 100., 25.])[0], atol=1e-4, rtol=0
        )
        grad = jax.grad(lambda point: evaluate(point, data)["logpdf"])(p)
        residual = np.array(data) / expected - 1
        np.testing.assert_allclose(grad["mu"], residual @ [5., 10.], atol=2e-5)
        np.testing.assert_allclose(
            grad["uncorr_bkguncrt"],
            residual * [50., 60.] + np.array([100., 25.]) * (1 / np.array(gamma) - 1),
            atol=3e-5,
        )


def test_pyhf_diagnostics_retain_input_origin(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"channels":')
    with pytest.raises(CompilationError) as caught:
        from_pyhf(path)
    assert caught.value.stage == "pyhf"
    assert caught.value.source == str(path)


def test_sliced_reduction_counts_support_vmap():
    evaluate = flatppl('''
        x = elementof(cartpow(reals, [4, 3]))
        inputs = x
        rows = x[[1, 3], all]
        outputs = aggregate(sum, [.col], rows[.row, .col])
    ''').compile()
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    compiled = jax.jit(jax.vmap(evaluate)).lower(values).compile()
    actual = compiled(values)
    np.testing.assert_allclose(actual, np.asarray(values)[:, [0, 2], :].sum(axis=1))
    assert not re.search(r"\bwhile\(", compiled.as_text())
