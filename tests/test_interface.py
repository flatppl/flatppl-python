import gc
import math
import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from flatppl import CompilationError, Context, flatppl


def test_inline_posterior_values_and_gradients():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "nuts.py"))
    density = example["make_logdensity"]()
    evaluate = jax.jit(jax.value_and_grad(density))
    for theta in (-2.0, -0.25, 0.5, 2.0):
        value, gradient = evaluate({"theta": jnp.float32(theta)})
        expected = -math.log(2 * math.pi) - (theta**2 + (1 - theta) ** 2) / 2
        np.testing.assert_allclose(value, expected, atol=2e-6)
        np.testing.assert_allclose(gradient["theta"], 1 - 2 * theta, atol=2e-6)


def test_product_density_gradient_at_zero_and_value_only_calls():
    module = flatppl(r"""
        x = elementof(cartpow(reals, 6))
        inputs = x
        mu = prod(x[[1,3,5]]) + 2.0 * prod(x[[2,4,6]]) + x[1]*x[3]*x[5]
        outputs = logdensityof(Normal(mu = mu, sigma = 1.0), 1.0)
    """)
    x = jnp.array([0, 2, 3, 0, 5, 6], dtype=jnp.float32)
    value, gradient = jax.jit(jax.value_and_grad(module.compile()))(x)
    np.testing.assert_allclose(value, -math.log(2 * math.pi) / 2 - 0.5, atol=2e-6)
    np.testing.assert_allclose(gradient, [30, 0, 0, 24, 0, 0], atol=2e-6)
    value_only = module.compile(autodiff=False)
    np.testing.assert_allclose(jax.jit(value_only)(x), value, atol=2e-6)
    with pytest.raises(TypeError):
        jax.grad(value_only)(x)


def test_nested_inputs_use_schema_order_and_support_gradients():
    function = flatppl(r"""
        p = elementof(cartprod(z = reals, a = cartprod(beta = reals, v = cartpow(reals, 2))))
        inputs = p
        outputs = p.z + 10.0 * p.a.beta + 100.0 * sum(p.a.v)
    """).compile()
    point = {
        "a": {"v": jnp.array([3.0, 5.0]), "beta": jnp.float32(2)},
        "z": jnp.float32(7),
    }
    value, gradient = jax.jit(jax.value_and_grad(function))(point)
    np.testing.assert_allclose(value, 827)
    np.testing.assert_allclose(gradient["a"]["v"], [100, 100])
    np.testing.assert_allclose(gradient["a"]["beta"], 10)
    np.testing.assert_allclose(gradient["z"], 1)


def test_declared_order_unused_input_and_nested_outputs():
    function = flatppl(r"""
        a = elementof(reals)
        b = elementof(reals)
        unused = elementof(reals)
        inputs = (unused, b, a)
        pair = (a, b)
        difference = b - a
        result = record(pair = pair, a = a)
        outputs = (difference, result)
    """).compile()
    difference, record = jax.jit(function)(100.0, 5.0, 2.0)
    np.testing.assert_allclose(difference, 3)
    np.testing.assert_allclose(record["pair"], [2, 5])
    np.testing.assert_allclose(record["a"], 2)
    named = function(a=2.0, unused=-1.0, b=5.0)
    np.testing.assert_allclose(named[0], difference)
    assert [item["name"] for item in function.schema["inputs"]] == ["unused", "b", "a"]
    assert function.schema["output_names"] == ("difference", "result")


def test_promoted_table_data_changes_without_source_io():
    function = flatppl(r"""
        data = load_data("absent.csv", cartpow(cartprod(x = reals, y = reals), 3))
        inputs = data
        outputs = record(total = sum(data.x), residual = data.y .- data.x, count = lengthof(data))
    """).compile()
    first = function({"y": jnp.array([3.0, 7.0, 5.0]), "x": jnp.array([1.0, 2.0, 3.0])})
    second = function(
        {"x": jnp.array([2.0, 4.0, 6.0]), "y": jnp.array([3.0, 7.0, 5.0])}
    )
    np.testing.assert_allclose(first["total"], 6)
    np.testing.assert_allclose(first["residual"], [2, 5, 2])
    np.testing.assert_allclose(second["total"], 12)
    np.testing.assert_allclose(second["residual"], [1, 3, -1])
    np.testing.assert_array_equal([first["count"], second["count"]], [3, 3])


def test_numpy_vectors_and_pandas_tables_use_query_precision():
    summarize = flatppl(r"""
        data = external(cartpow(cartprod(x = reals, y = reals), 3))
        inputs = data
        outputs = sum(data.y .- data.x)
    """).compile()
    columns = {"y": np.array([3.0, 7.0, 5.0]), "x": np.array([1.0, 2.0, 3.0])}
    table = pd.DataFrame(columns, index=[4, 2, 9])
    np.testing.assert_allclose(summarize(data=columns), 9)
    np.testing.assert_allclose(summarize(data=table), 9)
    table["x"] *= 2
    np.testing.assert_allclose(summarize(data=table), 3)
    assert summarize(data=table).dtype == jnp.float32


def test_bound_sizes_arrays_and_registered_models_keep_their_snapshots():
    context = Context()
    source = flatppl(r"""
        n = external(posintegers)
        weights = external(cartpow(reals, [2, n]))
        data = external(cartpow(reals, n))
        inputs = data
        outputs = sum(weights * data) / lengthof(data)
    """, context=context)
    weights = np.array([[1., 2., 3.], [4., 5., 6.]])
    bound = source.set(n=3, weights=weights)
    weights[:] = 99
    context.register("weighted.flatppl", bound)
    imported = flatppl(r"""
        data = external(cartpow(reals, 3))
        m = load_module("weighted.flatppl", data = data)
        inputs = data
        outputs = m.outputs
    """, context=context).compile()
    direct = bound.compile()
    other = source.set(**{"n": 2, "weights": [[2., 3.], [4., 5.]]}).compile()
    np.testing.assert_allclose(direct([1., 2., 3.]), 46 / 3)
    np.testing.assert_allclose(imported([1., 2., 3.]), 46 / 3)
    np.testing.assert_allclose(other([2., 4.]), 22)
    np.testing.assert_allclose(jax.grad(direct)(jnp.ones(3)), [5/3, 7/3, 3])
    np.testing.assert_allclose(direct([3., 2., 1.]), 38 / 3)


def test_imported_values_inside_expressions_preserve_parameters():
    context = Context()
    context.register("values.flatppl", flatppl("offset = 3.0\nx = elementof(reals)\nvalue = x + offset", context=context))
    function = flatppl(r"""
        p = elementof(cartprod(z = reals, a = cartpow(reals, 2)))
        left = load_module("values.flatppl", x = p.z)
        right = load_module("values.flatppl", x = sum(p.a))
        inputs = p
        outputs = left.value + 2.0 * right.value + left.offset
    """, context=context).compile()
    value, gradient = jax.jit(jax.value_and_grad(function))({"z": 1., "a": jnp.array([2., 4.])})
    np.testing.assert_allclose(value, 25)
    np.testing.assert_allclose(gradient["z"], 1)
    np.testing.assert_allclose(gradient["a"], [2, 2])


def test_context_snapshots_and_registration_lifecycle(tmp_path):
    model_path = tmp_path / "model.flatppl"
    model_path.write_text("value = 2.0\noutputs = value")
    query_source = 'm = load_module("model.flatppl")\noutputs = m.value'
    context = Context()
    model = context.load(model_path)
    context.register("model.flatppl", model)
    first = flatppl(query_source, context=context).compile()
    model_path.write_text("value = 7.0\noutputs = value")
    replacement = flatppl("value = 9.0", context=context)
    with pytest.raises(CompilationError):
        context.register("model.flatppl", replacement)
    np.testing.assert_allclose(first(), 2)
    np.testing.assert_allclose(context.load(model_path).compile()(), 2)
    fresh = Context()
    fresh.register("model.flatppl", fresh.load(model_path))
    np.testing.assert_allclose(flatppl(query_source, context=fresh).compile()(), 7)
    context.register("new.flatppl", replacement)
    np.testing.assert_allclose(
        flatppl(
            'm = load_module("new.flatppl")\noutputs = m.value', context=context
        ).compile()(),
        9,
    )


def test_registry_origins_file_origins_and_collision(tmp_path):
    context = Context()
    model = flatppl("value = 3.0", context=context)
    context.register("models/value.flatppl", model)
    query = 'm = load_module("value.flatppl")\noutputs = m.value'
    nested = flatppl(query, context=context, name="models/query.flatppl")
    context.register("alias.flatppl", nested)
    np.testing.assert_allclose(
        flatppl(
            'm = load_module("alias.flatppl")\noutputs = m.outputs', context=context
        ).compile()(),
        3,
    )
    (tmp_path / "value.flatppl").write_text("value = 11.0")
    np.testing.assert_allclose(
        flatppl(query, source_path=tmp_path / "host.py").compile()(), 11
    )
    with pytest.raises(CompilationError):
        flatppl(
            query,
            context=context,
            name="models/query.flatppl",
            source_path=tmp_path / "host.py",
        )
    np.testing.assert_allclose(nested.compile()(), 3)
    file_path = (tmp_path / "value.flatppl").resolve()
    labelled = Context()
    labelled.register(
        "value.flatppl",
        flatppl("value = 19.0", context=labelled, source_path=file_path),
    )
    with pytest.raises(CompilationError):
        flatppl(query, context=labelled, source_path=file_path.parent / "host.py")


def test_binding_collisions_and_native_ownership():
    context = Context()
    module = flatppl("compile = 5.0\noutputs = compile", context=context)
    handle = module.bindings["compile"]
    del context, module
    gc.collect()
    np.testing.assert_allclose(handle.module.compile()(), 5)
    assert handle.name == "compile"


def test_import_diagnostic_and_corrected_context(tmp_path):
    source = tmp_path / "bad.flatppl"
    source.write_text("value = )")
    query = 'm = load_module("bad.flatppl")\noutputs = m.value'
    with pytest.raises(CompilationError) as failure:
        flatppl(query, source_path=tmp_path / "host.py")
    assert failure.value.source == str(source)
    assert failure.value.span is not None
    assert failure.value.import_chain == (str(tmp_path / "host.py"),)
    source.write_text("value = 6.0")
    np.testing.assert_allclose(
        flatppl(query, source_path=tmp_path / "host.py").compile()(), 6
    )


def test_import_instances_preserve_independent_and_shared_laws():
    context = Context()
    context.register(
        "leaf.flatppl",
        flatppl(
            r"""
        z ~ Normal(0.0, 1.0)
        a ~ Normal(z, 1.0)
        b ~ Normal(z, 1.0)
        left = lawof(a)
        right = lawof(b)
    """,
            context=context,
        ),
    )
    shared = flatppl(
        r"""
        m = load_module("leaf.flatppl")
        left = m.left
        right = m.right
        outputs = logdensityof(joint(a = left, b = right), record(a = 0.0, b = 1.0))
    """,
        context=context,
    ).compile()
    independent = flatppl(
        r"""
        m = load_module("leaf.flatppl")
        n = load_module("leaf.flatppl")
        left = m.left
        right = n.right
        outputs = logdensityof(joint(a = left, b = right), record(a = 0.0, b = 1.0))
    """,
        context=context,
    ).compile()
    np.testing.assert_allclose(
        shared(), -math.log(2 * math.pi) - math.log(3) / 2 - 1 / 3
    )
    np.testing.assert_allclose(
        independent(), -math.log(2 * math.pi) - math.log(2) - 1 / 4
    )


def test_promoted_bindings_replace_only_their_named_values():
    function = flatppl(r"""
        origin = 1.0
        a = origin
        b = origin
        record_origin = record(x = 5.0)
        p = record_origin
        data = external(reals)
        derived = data + 1.0
        inputs = (a, b, p, derived)
        score = logdensityof(joint(x = Normal(0.0, 1.0)), p)
        outputs = (a, b, origin, p.x, record_origin.x, derived, score)
    """).compile()
    expected = [2, 7, 1, 11, 5, 13, -math.log(2 * math.pi) / 2 - 121 / 2]
    np.testing.assert_allclose(function(2.0, 7.0, {"x": 11.0}, 13.0), expected)


def test_nested_record_joint_density():
    density = flatppl(r"""
        p = elementof(cartprod(x = reals, nested = cartprod(y = reals)))
        inputs = p
        outputs = logdensityof(joint(x = Normal(0.0, 1.0), nested = joint(y = Normal(1.0, 2.0))), p)
    """).compile()
    point = {"nested": {"y": jnp.float32(3)}, "x": jnp.float32(0.5)}
    value, gradient = jax.value_and_grad(density)(point)
    np.testing.assert_allclose(value, -math.log(2 * math.pi) - math.log(2) - 0.625)
    np.testing.assert_allclose(gradient["x"], -0.5)
    np.testing.assert_allclose(gradient["nested"]["y"], -0.5)


def test_directory_bundle_alias_and_containment(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "leaf.flatppl").write_text("value = 3.0")
    (bundle / "main.flatppl").write_text(
        'm = load_module("leaf.flatppl")\noutputs = m.value'
    )
    context = Context()
    context.register("bundle", context.load(bundle))
    query = flatppl(
        'm = load_module("bundle")\noutputs = m.outputs',
        context=context,
        source_path=tmp_path / "host.py",
    )
    np.testing.assert_allclose(query.compile()(), 3)
    (tmp_path / "outside.flatppl").write_text("value = 42.0")
    (bundle / "main.flatppl").write_text(
        'm = load_module("../outside.flatppl")\noutputs = m.value'
    )
    fresh = Context()
    # A prior unrestricted file load must not bypass the bundle boundary.
    unrestricted = fresh.load(bundle / "main.flatppl")
    np.testing.assert_allclose(unrestricted.compile()(), 42)
    fresh.register("bundle", unrestricted)
    with pytest.raises(CompilationError):
        fresh.load(bundle)
    with pytest.raises(CompilationError):
        flatppl(
            'm = load_module("bundle")\noutputs = m.outputs',
            context=fresh,
            source_path=tmp_path / "host.py",
        )
    np.testing.assert_allclose(query.compile()(), 3)


def test_rng_pairs_chaining_and_output_order():
    source = r"""
        s = rnginit(0)
        a = rand(s, Normal(0.0, 1.0))
        b = rand(s, Normal(0.0, 1.0))
        c = rand(a[2], Normal(0.0, 1.0))
        score = logdensityof(Normal(0.0, 1.0), a[1])
        inputs = s
    """
    with jax.enable_x64():
        forward = flatppl(source + "outputs = (a, b, c, s, score)").compile(
            autodiff=False
        )
        reverse = flatppl(source + "outputs = (score, s, c, b, a)").compile(
            autodiff=False
        )
        state = jnp.array([1729, 0], dtype=jnp.uint64)
        a, b, c, original, score = forward(state)
        np.testing.assert_array_equal(original, state)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], [1729, 1])
        np.testing.assert_array_equal(b[1], a[1])
        np.testing.assert_array_equal(c[1], [1729, 2])
        next_a = forward(a[1])[0]
        np.testing.assert_array_equal(c[0], next_a[0])
        np.testing.assert_allclose(
            score, -math.log(2 * math.pi) / 2 - float(a[0]) ** 2 / 2
        )
        for actual, expected in zip(
            jax.tree.leaves(reverse(state)), jax.tree.leaves((score, original, c, b, a))
        ):
            np.testing.assert_array_equal(actual, expected)


def test_precision_and_repeated_device_calls():
    module = flatppl("x = elementof(reals)\ninputs = x\noutputs = (x + 1.0) - x")
    with jax.enable_x64(False):
        single = jax.jit(module.compile())
        point = jnp.float32(2**24)
        single(point).block_until_ready()
        with jax.transfer_guard("disallow"):
            first = single(point)
            second = single(point)
        np.testing.assert_array_equal([first, second], [0, 0])
        assert first.devices() == point.devices()
        assert not jax.config.x64_enabled
        with pytest.raises(ValueError):
            module.compile(dtype="float64")
        np.testing.assert_array_equal(single(point), 0)
    with jax.enable_x64():
        double = module.compile(dtype="float64")
        np.testing.assert_array_equal(double(jnp.float64(2**24)), 1)


def test_array_axis_order_matches_explicit_matrices():
    evaluate = flatppl(r"""
        data = external(cartpow(reals, 6))
        rows = array(data, [2, 3], [1, 2])
        columns = array(data, [2, 3], [2, 1])
        inputs = data
        outputs = (rows, columns, lengthof(transpose(data)))
    """).compile()
    data = np.arange(1, 7, dtype=np.float64)
    rows, columns, length = evaluate(data)
    np.testing.assert_array_equal(rows, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(columns, [[1, 3, 5], [2, 4, 6]])
    assert int(length) == 6
    np.testing.assert_array_equal(evaluate(data.tolist())[0], rows)
    with pytest.raises(TypeError):
        evaluate([jnp.asarray(1 + 2j)] * 6)
