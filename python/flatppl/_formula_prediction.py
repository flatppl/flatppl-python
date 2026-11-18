"""Compile and batch response queries against prepared formula modules."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from operator import index
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ._api import _domain_text, flatppl  # pyright: ignore[reportPrivateUsage]
from ._jax import CompiledFunction, float_dtype, pack_tensor

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ._api import Module
    from ._schema import RecordSchema
    from .formula import FormulaView

type Operation = Literal[
    "expected_response", "log_lik", "sample", "reported_parameters"
]


def _query(view: FormulaView, operation: Operation, seed_size: int) -> Module:
    queries = view._response_queries  # pyright: ignore[reportPrivateUsage]
    cache_key = (operation, seed_size)
    if cache_key not in queries:
        model = view.module
        domain = model.bindings["Pi_post"].domain
        assert domain is not None and domain["set"] == "record"
        arguments = ", ".join(
            f"{field['name']} = point.{field['name']}" for field in domain["fields"]
        )
        model_name = view._response_model_name  # pyright: ignore[reportPrivateUsage]
        lines = [
            f'_model = load_module("{model_name}")',
            f"point = elementof({_domain_text(domain)})",
        ]
        if operation == "sample":
            output = (
                "colstack([response.y, response.y2])"
                if "response_columns" in view.info
                else "response.y"
            )
            lines.extend(
                [
                    "_response_kernel = _model.K_model",
                    f"seed = external(cartpow(integers, {seed_size}))",
                    "inputs = (point, seed)",
                    f"response, _ = rand(_response_kernel({arguments}), rnginit(seed))",
                    f"outputs = {output}",
                ]
            )
        else:
            lines.extend(
                ["inputs = point", f"outputs = _model.{operation}({arguments})"]
            )
        queries[cache_key] = flatppl("\n".join(lines), context=model.context)
    return queries[cache_key]


def _batch_prefix(
    supplied: tuple[int, ...], shape: tuple[int, ...], name: str
) -> tuple[int, ...]:
    rank = len(supplied) - len(shape)
    if rank < 0 or supplied[rank:] != shape:
        raise ValueError(
            f"draws.{name} requires trailing shape {shape}, got {supplied}"
        )
    return supplied[:rank]


def _pack_draws(
    compiled: CompiledFunction,
    draws: Mapping[str, object],
    *,
    replaced: Collection[str] = (),
) -> tuple[dict[str, jax.Array], tuple[int, ...]]:
    schema = cast("RecordSchema", compiled.schema["inputs"][0]["value"])
    names = {field["name"] for field in schema["fields"]}.difference(replaced)
    if set(draws) != names:
        raise TypeError(f"draws requires fields {sorted(names)}")
    arrays: dict[str, jax.Array] = {}
    shape: tuple[int, ...] | None = None
    for field in schema["fields"]:
        name, value = field["name"], field["value"]
        if name not in names:
            continue
        assert value["kind"] == "tensor"
        array = pack_tensor(draws[name], value["dtype"], f"draws.{name}")
        prefix = _batch_prefix(array.shape, tuple(value["shape"]), name)
        if shape is not None and prefix != shape:
            raise ValueError(
                f"draws.{name} requires leading draw shape {shape}, got {prefix}"
            )
        arrays[name] = array
        shape = prefix
    return arrays, shape or ()


def evaluate(
    view: FormulaView,
    operation: Operation,
    draws: Mapping[str, object],
    *,
    dtype: DTypeLike,
    key: jax.Array | None = None,
) -> jax.Array:
    """Evaluate a prepared response query at one point or over leading draw axes."""
    word_count = 0 if key is None else jax.random.key_data(key).size
    compiled = _query(view, operation, word_count * 4).compile(
        dtype=dtype, autodiff=operation != "sample"
    )
    arrays, shape = _pack_draws(compiled, draws)
    inputs = (arrays,)
    if key is not None:
        words = jax.random.key_data(jax.random.split(key, shape))
        shifts = jnp.arange(0, 32, 8, dtype=jnp.uint32)  # pyright: ignore[reportUnknownMemberType]
        bytes_ = ((words[..., None] >> shifts) & jnp.uint32(255)).reshape(
            (*shape, word_count * 4)
        )
        seed_dtype = compiled.schema["inputs"][1]["value"]["dtype"]
        inputs = (arrays, bytes_.astype(seed_dtype))
    return cast(jax.Array, _batch_call(compiled, shape, *inputs))


def _batch_call(
    compiled: CompiledFunction, shape: tuple[int, ...], *inputs: Any
) -> Any:
    call = compiled.__call__
    for _ in shape:
        call = jax.vmap(call)  # pyright: ignore[reportUnknownMemberType]
    return call(*inputs)


def parameter_draws(
    view: FormulaView, draws: Mapping[str, object], *, dtype: DTypeLike
) -> dict[str, jax.Array]:
    """Evaluate parameter bindings and select the reported scalar components."""
    compiled = _query(view, "reported_parameters", 0).compile(dtype=dtype)
    arrays, shape = _pack_draws(compiled, draws)
    values = cast(dict[str, jax.Array], _batch_call(compiled, shape, arrays))
    result: dict[str, jax.Array] = {}
    for meta in view.info["parameters"]:
        indices = meta.get("indices")
        if indices is None:
            indices = (meta["index"],) if "index" in meta else ()
        value = values[meta["binding"]]
        result[meta["name"]] = value[(..., *(index - 1 for index in indices))]
    return result


def sample_response(
    view: FormulaView, draws: Mapping[str, object], *, key: jax.Array, dtype: DTypeLike
) -> jax.Array:
    """Draw from the prepared observation law with an explicit key."""
    family = view.info["family"]
    if family in ("categorical", "cumulative"):
        # Native broadcast sampling does not yet lower a vector of probabilities
        # per row. Each categorical row is independent given the model point.
        probs = evaluate(view, "expected_response", draws, dtype=dtype)
        valid = jnp.all(jnp.isfinite(probs) & (probs >= 0)) & jnp.all(
            probs.sum(axis=-1) > 0
        )
        checkify.check(  # pyright: ignore[reportUnknownMemberType]
            valid,
            "FlatPPL categorical probabilities need finite, nonnegative values with a positive total",
        )
        codes = jax.random.categorical(key, jnp.log(probs), axis=-1, mode="high")
        return (codes + 1).astype(
            jnp.int64 if probs.dtype == jnp.float64 else jnp.int32
        )
    if family == "binomial":
        return _sample_binomial(view, draws, key=key, dtype=dtype)
    return evaluate(view, "sample", draws, dtype=dtype, key=key)


def _sample_binomial(
    view: FormulaView, draws: Mapping[str, object], *, key: jax.Array, dtype: DTypeLike
) -> jax.Array:
    # Native broadcast sampling does not yet support per-row trial counts.
    trial_spec = view.recipe["formula"]["trials"]
    if trial_spec["kind"] == "constant":
        trials = jnp.asarray(trial_spec["value"])  # pyright: ignore[reportUnknownMemberType]
    else:
        queries = view._response_queries  # pyright: ignore[reportPrivateUsage]
        cache_key = ("trials_data", 0)
        if cache_key not in queries:
            queries[cache_key] = view.module.query("trials_data")
        trials = cast(
            jax.Array, queries[cache_key].compile(dtype=dtype, autodiff=False)()
        )
    expected = evaluate(view, "expected_response", draws, dtype=dtype)
    probability = expected / jnp.maximum(trials, 1)
    valid = jnp.all((probability >= 0) & (probability <= 1))
    checkify.check(valid, "FlatPPL binomial probabilities must be in [0, 1]")  # pyright: ignore[reportUnknownMemberType]
    wide = expected.dtype == jnp.float64
    count_limit = 2**53 if wide else 2**24
    checkify.check(  # pyright: ignore[reportUnknownMemberType]
        jnp.all((trials >= 0) & (trials <= count_limit)),
        "FlatPPL binomial trial counts exceed the exact range of the selected dtype",
    )
    counts = jax.random.binomial(key, trials, probability, dtype=expected.dtype)
    return counts.astype(jnp.int64 if wide else jnp.int32)


def sample_new_groups(
    view: FormulaView, draws: Mapping[str, object], *, key: jax.Array, dtype: DTypeLike
) -> dict[str, object]:
    """Append one standard Normal innovation per new level and coefficient."""
    expanded = dict(draws)
    keys = jax.random.split(key, len(view.new_group_shapes))
    inputs = view.module.bindings["K_model"].inputs
    assert inputs is not None
    dtype_name = float_dtype(dtype)
    for (name, new_shape), group_key in zip(
        view.new_group_shapes.items(), keys, strict=True
    ):
        if name not in draws:
            raise TypeError(f"draws is missing group coordinate {name!r}")
        original = pack_tensor(draws[name], dtype_name, f"draws.{name}")
        prepared_shape = inputs[name]["shape"]
        original_shape = (*prepared_shape[:-1], prepared_shape[-1] - new_shape[-1])
        prefix = _batch_prefix(original.shape, original_shape, name)
        innovations = jax.random.normal(
            group_key, (*prefix, *new_shape), dtype=original.dtype
        )
        expanded[name] = jnp.concatenate((original, innovations), axis=-1)
    return expanded


def _group_shapes(
    view: FormulaView,
    draws: Mapping[str, object],
    groups: Sequence[str] | None,
) -> dict[str, tuple[int, ...]]:
    """Resolve the prepared tensor shapes of the selected innovation blocks."""
    requested = set((groups,) if isinstance(groups, str) else groups or ())
    if unknown := requested.difference(view.group_terms):
        raise ValueError(f"unknown group innovation blocks {sorted(unknown)}")
    selected = {
        name for name in view.group_terms if groups is None or name in requested
    }
    if not selected:
        return {}
    bindings = view.module.bindings
    inputs = bindings["K_model"].inputs
    assert inputs is not None
    names = set(inputs)
    if set(draws).difference(names) or names.difference(draws, selected):
        raise TypeError(f"draws requires fields {sorted(names - selected)}")

    shapes: dict[str, tuple[int, ...]] = {}
    leading: tuple[int, ...] | None = None
    for name in inputs:
        shape = bindings[name].shape
        if shape is None or None in shape:
            raise ValueError(f"draws.{name} requires a known parameter shape")
        extent = cast("tuple[int, ...]", shape)
        if name in selected:
            shapes[name] = extent
        if name not in draws:
            continue
        supplied = jnp.shape(draws[name])  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
        if name in selected and name in view.new_group_shapes:
            fitted = (*extent[:-1], extent[-1] - view.new_group_shapes[name][-1])
            if supplied[-len(fitted) :] == fitted:
                extent = fitted
        prefix = _batch_prefix(supplied, extent, name)
        if leading is not None and prefix != leading:
            raise ValueError(
                f"draws.{name} requires leading draw shape {leading}, got {prefix}"
            )
        leading = prefix
    return {name: (*(leading or ()), *shape) for name, shape in shapes.items()}


def zero_group_effects(
    view: FormulaView,
    draws: Mapping[str, object],
    *,
    groups: Sequence[str] | None,
    dtype: DTypeLike,
) -> dict[str, object]:
    """Replace only selected group innovations with deterministic zeros."""
    shapes = _group_shapes(view, draws, groups)
    dtype_name = float_dtype(dtype)
    zeroed = {
        name: jnp.zeros(shape, dtype_name)  # pyright: ignore[reportUnknownMemberType]
        for name, shape in shapes.items()
    }
    return {**draws, **zeroed}


_REAL_LOCATIONS = {"gaussian", "student", "lognormal", "hurdle_lognormal"}
_PROBABILITY_LOCATIONS = {
    "bernoulli",
    "binomial",
    "zero_inflated_binomial",
    "beta",
    "zero_inflated_beta",
}
_MARGINAL_BATCH_SIZE = 256


def _validate_marginal_groups(
    view: FormulaView, blocks: Mapping[str, tuple[int, ...]]
) -> None:
    family = view.info["family"]
    nonlinear = view.recipe.get("nonlinear", {}).get("parameters", ())
    locations = {"mu", "mu2"} if "response_columns" in view.info else {"mu"}
    for name in blocks:
        for parameter, _ in view.group_terms[name]:
            link = view.recipe["predictors"][parameter]["link"]
            if (
                parameter in nonlinear
                or link == "inverse"
                or (
                    link == "identity"
                    and (parameter not in locations or family not in _REAL_LOCATIONS)
                )
                or (
                    parameter == "mu"
                    and link == "log"
                    and family in _PROBABILITY_LOCATIONS
                )
                or (
                    parameter == "sigma" and family in {"lognormal", "hurdle_lognormal"}
                )
            ):
                raise ValueError(
                    f"cannot marginalize group {name!r} in {parameter!r} "
                    f"for {family} with link {link!r}"
                )


def marginal_expected_response(
    view: FormulaView,
    draws: Mapping[str, object],
    *,
    groups: Sequence[str] | None,
    key: jax.Array,
    samples: int,
    dtype: DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    """Average the conditional means over selected joint Gaussian group effects."""
    samples = index(samples)
    if samples < 2:
        raise ValueError("marginal_expected_response requires at least two samples")
    shapes = _group_shapes(view, draws, groups)
    if not shapes:
        response = view.expected_response(draws, dtype=dtype)
        return response, jnp.zeros_like(response)  # pyright: ignore[reportUnknownMemberType]
    _validate_marginal_groups(view, shapes)
    dtype_name = float_dtype(dtype)
    if "expected_response" not in view.module.bindings:
        raise ValueError(f"{view.info['family']} has no expected-response query")
    compiled = _query(view, "expected_response", 0).compile(dtype=dtype)
    retained, _ = _pack_draws(
        compiled,
        {name: value for name, value in draws.items() if name not in shapes},
        replaced=shapes,
    )

    def conditional_response(replacements: dict[str, jax.Array]) -> jax.Array:
        return view.expected_response({**retained, **replacements}, dtype=dtype)

    responses = jax.vmap(conditional_response)  # pyright: ignore[reportUnknownMemberType]

    def moments(start: int, size: int) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(jax.random.fold_in(key, start), len(shapes))
        innovations = {
            name: jax.random.normal(block_key, (size, *shape), dtype=dtype_name)
            for (name, shape), block_key in zip(shapes.items(), keys, strict=True)
        }
        values = responses(innovations)
        mean = values.mean(axis=0)
        return mean, jnp.square(values - mean).sum(axis=0)

    # Combine batch means and centred sums of squares without keeping samples.
    size = min(samples, _MARGINAL_BATCH_SIZE)
    mean, squared = moments(0, size)
    for start in range(size, samples, size):
        count = min(size, samples - start)
        next_mean, next_squared = moments(start, count)
        difference = next_mean - mean
        ratio = count / (start + count)
        squared = squared + next_squared + jnp.square(difference) * (start * ratio)
        mean = mean + difference * ratio
    return mean, jnp.sqrt(squared / (samples * (samples - 1)))
