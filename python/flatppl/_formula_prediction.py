"""Compile and batch response queries against prepared formula modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

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

type Operation = Literal["expected_response", "log_lik", "sample"]


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
            lines.extend(
                [
                    "_response_kernel = _model.K_model",
                    f"seed = external(cartpow(integers, {seed_size}))",
                    "inputs = (point, seed)",
                    f"response, _ = rand(_response_kernel({arguments}), rnginit(seed))",
                    "outputs = response.y",
                ]
            )
        else:
            lines.extend(
                ["inputs = point", f"outputs = _model.{operation}({arguments})"]
            )
        queries[cache_key] = flatppl("\n".join(lines), context=model.context)
    return queries[cache_key]


def _batch_prefix(
    array: jax.Array, shape: tuple[int, ...], name: str
) -> tuple[int, ...]:
    rank = array.ndim - len(shape)
    if rank < 0 or array.shape[rank:] != shape:
        raise ValueError(
            f"draws.{name} requires trailing shape {shape}, got {array.shape}"
        )
    return array.shape[:rank]


def _pack_draws(
    compiled: CompiledFunction, draws: Mapping[str, object]
) -> tuple[dict[str, jax.Array], tuple[int, ...]]:
    schema = cast("RecordSchema", compiled.schema["inputs"][0]["value"])
    names = {field["name"] for field in schema["fields"]}
    if set(draws) != names:
        raise TypeError(f"draws requires fields {sorted(names)}")
    arrays: dict[str, jax.Array] = {}
    shape: tuple[int, ...] | None = None
    for field in schema["fields"]:
        name, value = field["name"], field["value"]
        assert value["kind"] == "tensor"
        array = pack_tensor(draws[name], value["dtype"], f"draws.{name}")
        prefix = _batch_prefix(array, tuple(value["shape"]), name)
        if shape is not None and prefix != shape:
            raise ValueError(
                f"draws.{name} requires leading draw shape {shape}, got {prefix}"
            )
        arrays[name] = array
        shape = prefix
    assert shape is not None
    return arrays, shape


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
    call = compiled.__call__
    for _ in shape:
        call = jax.vmap(call)  # pyright: ignore[reportUnknownMemberType]
    return cast(jax.Array, call(*inputs))


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
        prefix = _batch_prefix(original, original_shape, name)
        innovations = jax.random.normal(
            group_key, (*prefix, *new_shape), dtype=original.dtype
        )
        expanded[name] = jnp.concatenate((original, innovations), axis=-1)
    return expanded
