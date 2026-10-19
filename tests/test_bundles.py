import io
import math
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, Context, flatppl, load


def test_zip_paths_bytes_and_registered_slices_preserve_density(tmp_path: Path) -> None:
    archive = (Path(__file__).parent / "data" / "portable.flatppl.zip").read_bytes()
    path = tmp_path / "model.flatppl.zip"
    path.write_bytes(archive)
    context = Context()
    model = load(path, context=context)
    memory = context.load_bundle("memory.zip", archive)
    context.register("model.flatppl", model)
    query = flatppl(
        'x = elementof(reals)\nm = load_module("model.flatppl", theta = x)\n'
        "inputs = x\noutputs = m.outputs",
        context=context,
    )
    sliced = model.slice(["inputs", "outputs"])
    assert list(tmp_path.iterdir()) == [path]
    path.unlink()
    theta = jnp.float32(0.25)
    expected = -1.5 * math.log(2 * math.pi) - 0.5 * sum(
        (float(theta) - y) ** 2 for y in (0, 1, 2)
    )
    for candidate in (model, memory, query, sliced):
        value, gradient = jax.jit(jax.value_and_grad(candidate.compile()))(theta)
        np.testing.assert_allclose(value, expected, atol=2e-6)
        np.testing.assert_allclose(gradient, 3 - 3 * theta, atol=2e-6)


def test_pack_retains_bound_priors_cached_data_and_binary_bytes(tmp_path: Path) -> None:
    context = Context()
    prior = flatppl("mu = external(reals)\nPi = Normal(mu, 1)", context=context)
    context.register("prior.flatppl", prior.set(mu=2.0))
    data = tmp_path / "values.json"
    data.write_text("[1, 2]")
    (tmp_path / "raw.bin").write_bytes(b"\x00\xff\x80\x01")
    query = flatppl(
        '% Keep the model description.\npath = "prior.flatppl"\n'
        "prior = load_module(path)\n"
        'values = load_data("values.json", cartpow(reals, 2))\n'
        'raw = load_data("raw.bin", cartpow(reals, 4))\n'
        "x = elementof(reals)\ninputs = x\n"
        "outputs = logdensityof(prior.Pi, x) + sum(values)",
        context=context,
        source_path=tmp_path / "host.py",
    )
    expected = -0.5 * math.log(2 * math.pi) - 0.5 * (0.25 - 2) ** 2 + 3
    np.testing.assert_allclose(query.compile()(0.25), expected, atol=2e-6)
    data.write_text("[100, 200]")
    archive = query.pack()
    with ZipFile(io.BytesIO(archive)) as files:
        assert files.read("data/values.json") == b"[1, 2]"
        assert files.read("data/raw.bin") == b"\x00\xff\x80\x01"
        assert files.read("main.flatppl").startswith(b"% Keep the model description.")
    restored = Context().load_bundle("export.zip", archive)
    path = tmp_path / "export.flatppl.zip"
    query.save(path)
    saved = load(str(path))
    sliced = query.slice(["inputs", "outputs"])
    reopened_slice = Context().load_bundle("slice.zip", sliced.pack())
    for candidate in (restored, saved, reopened_slice):
        value, gradient = jax.jit(jax.value_and_grad(candidate.compile()))(
            jnp.float32(0.25)
        )
        np.testing.assert_allclose(value, expected, atol=2e-6)
        np.testing.assert_allclose(gradient, 2 - 0.25, atol=2e-6)


def test_failed_save_preserves_existing_file(tmp_path: Path) -> None:
    module = flatppl(
        'data = load_data("missing.json", reals)\noutputs = data',
        source_path=tmp_path / "model.flatppl",
    )
    path = tmp_path / "model.flatppl.zip"
    path.write_bytes(b"existing file")
    with pytest.raises(CompilationError):
        module.save(path)
    assert path.read_bytes() == b"existing file"
