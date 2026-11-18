import math
import re

import jax
import numpy as np
import pytest

from flatppl import Module, flatppl


def evidence_query(
    size: int, hyperparameter: str, latent: str, prior: str, law: str, cell: str
) -> Module:
    return flatppl(f"""
        h = external({hyperparameter})
        t = elementof({latent})
        y = external(cartpow({cell}, {size}))
        K = functionof(iid({law}, {size}), theta=t)
        P = bayesupdate(likelihoodof(K, y), joint(theta={prior}))
        inputs = (h, y)
        outputs = log(totalmass(P))
    """)


def summary_args(value: jax.Array | tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return value if isinstance(value, tuple) else (value,)


@pytest.mark.parametrize(
    "hyperparameter,latent,prior,law,cell,events",
    [
        ("posreals", "posreals", "Gamma(h,3)", "Poisson(t)", "integers", (0, 2)),
        ("posreals", "reals", "Normal(0.5,h)", "Normal(t,h)", "reals", (-1.0, 1.5)),
        (
            "posreals",
            "posreals",
            "InverseGamma(h,3)",
            "Normal(h,sqrt(t))",
            "reals",
            (-1.0, 2.0),
        ),
        (
            "posreals",
            "stdsimplex(3)",
            "Dirichlet([h,3.,4.])",
            "Categorical(t)",
            "integers",
            (1, 3),
        ),
        (
            "posreals",
            "stdsimplex(3)",
            "Dirichlet([h,3.,4.])",
            "Categorical0(t)",
            "integers",
            (0, 2),
        ),
        (
            "posintegers",
            "stdsimplex(3)",
            "Dirichlet([real(h),3.,4.])",
            "Multinomial(h,t)",
            "cartpow(integers,3)",
            ((2, 0, 0), (0, 1, 1)),
        ),
    ],
    ids=[
        "poisson",
        "normal-mean",
        "normal-variance",
        "categorical",
        "categorical0",
        "multinomial",
    ],
)
def test_prepared_evidence_score_has_fixed_size_summaries(
    hyperparameter: str,
    latent: str,
    prior: str,
    law: str,
    cell: str,
    events: tuple[int | float | tuple[int, int, int], ...],
) -> None:
    layouts: list[tuple[tuple[tuple[int, ...], ...], int]] = []
    with jax.enable_x64():
        for size in (31, 257):
            query = evidence_query(size, hyperparameter, latent, prior, law, cell)
            preparation, online, names = query.prepare_likelihood("y")
            score = online.compile(dtype="float64", autodiff=False)
            fields = score.schema["inputs"]
            assert names
            assert [field["name"] for field in fields] == ["h", *names]
            shapes = tuple(tuple(field["value"]["shape"]) for field in fields[1:])
            layouts.append((shapes, score.stats()["total"]))
            assert not re.search(rf"tensor<(?:\d+x)*{size}x", score.stablehlo)

            if size == 31:
                prepare = preparation.compile(dtype="float64", autodiff=False)
                assert [field["name"] for field in prepare.schema["inputs"]] == ["y"]
                data = np.take(
                    np.asarray(events), np.arange(size) % len(events), axis=0
                )
                summaries = summary_args(prepare(data))
                direct = query.compile(dtype="float64", autodiff=False)
                assert float(score(2, *summaries)) == pytest.approx(
                    float(direct(2, data)), rel=2e-12, abs=2e-11
                )
    assert layouts[0] == layouts[1]


def test_preparing_mutated_poisson_counts_recomputes_allocation_and_support() -> None:
    query = evidence_query(
        4, "posreals", "posreals", "Gamma(h,3)", "Poisson(t)", "integers"
    )
    preparation, online, _ = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64", autodiff=False)
        data = np.array([0, 2, 0, 0], dtype=np.int64)
        original = summary_args(jax.block_until_ready(prepare(data)))
        data[:2] = [1, 1]
        split = summary_args(jax.block_until_ready(prepare(data)))
        data[:2] = [-1, 3]
        impossible = summary_args(prepare(data))

        # Both valid allocations have two events. The double count contributes 1/2!.
        for shape in (2.0, 5.0):
            expected = shape * math.log(3 / 7) + math.log(shape * (shape + 1) / 49)
            assert float(score(shape, *original)) == pytest.approx(
                expected - math.log(2), abs=2e-12
            )
            assert float(score(shape, *split)) == pytest.approx(expected, abs=2e-12)
            assert float(score(shape, *impossible)) == -math.inf
