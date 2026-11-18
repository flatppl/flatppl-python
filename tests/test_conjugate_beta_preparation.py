import math

import jax
import numpy as np
import pytest

from flatppl import Module, flatppl


def beta_query(law: str, trials: str, cell: str, output: str) -> Module:
    return flatppl(f"""
        p = elementof(unitinterval)
        a = external(posreals)
        {trials}
        y = external(cartpow({cell},2))
        L = likelihoodof(functionof(iid({law},2),p=p),y)
        P = bayesupdate(L,joint(p=Beta(a,1.)))
        inputs = (a,{"n," if trials else ""}y)
        outputs = {output}
    """)


@pytest.mark.parametrize(
    "law,trials",
    [
        ("Bernoulli(p)", ""),
        ("Binomial(3,p)", ""),
        ("Binomial(n,p)", "n = external(posintegers)"),
    ],
    ids=["bernoulli", "fixed-binomial", "prepared-binomial"],
)
def test_beta_evidence_prepares_the_observed_allocation(law: str, trials: str) -> None:
    query = beta_query(law, trials, "reals", "log(totalmass(P))")
    data_names = ("n", "y") if trials else ("y",)
    preparation, online, names = query.prepare_likelihood(*data_names)
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64")
        assert [f["name"] for f in prepare.schema["inputs"]] == list(data_names)
        inputs = ["a", "n"] if trials else ["a"]
        assert [f["name"] for f in score.schema["inputs"]] == [*inputs, *names]
        for n in (2, 5) if trials else (1 if law == "Bernoulli(p)" else 3,):
            prefix = (n,) if trials else ()
            allocations = ((0, 1), (1, 1)) if n == 1 else ((0, 2), (1, 1))
            for counts in allocations:
                summary = prepare(*prefix, np.asarray(counts, dtype=float))
                assert isinstance(summary, tuple)
                k = sum(counts)
                failures = 2 * n - k
                coefficient = math.prod(math.comb(n, count) for count in counts)
                for a in (1.0, 2.5):
                    # Integrate the two likelihood factors against a Beta(a,1) prior.
                    total = 2 * n
                    expected = (
                        math.log(coefficient)
                        + sum(math.log(a + i) for i in range(k))
                        + math.lgamma(1 + failures)
                        - sum(math.log(a + 1 + i) for i in range(total))
                    )
                    derivative = sum(1 / (a + i) for i in range(k)) - sum(
                        1 / (a + 1 + i) for i in range(total)
                    )
                    value, grad = jax.value_and_grad(score)(a, *prefix, *summary)
                    assert float(value) == pytest.approx(expected, abs=2e-12)
                    assert float(grad) == pytest.approx(derivative, abs=2e-12)
            for invalid in ([0.5, 0.5], [0.0, float(n + 1)]):
                summary = prepare(*prefix, np.asarray(invalid))
                assert isinstance(summary, tuple)
                assert np.isneginf(score(1.0, *prefix, *summary))


def test_live_binomial_trials_preserve_integer_failures_and_support() -> None:
    # Adjacent source counts share a float32 representation at this boundary.
    limit = 2**24
    point = 1 - 2**-23
    query = beta_query(
        "Binomial(n,p)",
        "n = external(posintegers)",
        "integers",
        f"logdensityof(normalize(P),record(p={point}))",
    )
    preparation, online, names = query.prepare_likelihood("y")
    with jax.enable_x64(False):
        prepare = preparation.compile(dtype="float32", autodiff=False)
        score = online.compile(dtype="float32", autodiff=False)
        assert [f["name"] for f in score.schema["inputs"]] == ["a", "n", *names]
        summary = prepare(np.array([limit + 1, limit], dtype=np.int32))
        assert isinstance(summary, tuple)
        assert np.isnan(score(1.0, limit, *summary))
        for n in (limit + 1, limit + 2):
            a = 1 + (2 * limit + 1)
            b = 1 + (2 * n - (2 * limit + 1))
            # For integer b, 1/B(a,b) is a short rising factorial divided by (b-1)!.
            expected = (
                sum(math.log(a + i) for i in range(b))
                - math.lgamma(b)
                + (a - 1) * math.log(point)
                + (b - 1) * math.log1p(-point)
            )
            assert float(score(1.0, n, *summary)) == pytest.approx(expected, abs=4e-5)


def test_binomial_evidence_with_a_live_trial_count_keeps_row_allocations() -> None:
    query = beta_query(
        "Binomial(n,p)", "n = external(posintegers)", "integers", "log(totalmass(P))"
    )
    preparation, online, names = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = preparation.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64", autodiff=False)
        assert [f["name"] for f in score.schema["inputs"]] == ["a", "n", "y", *names]
        for counts in ((0, 2), (1, 1)):
            y = np.asarray(counts)
            summary = prepare(y)
            assert isinstance(summary, tuple)
            for n in (1, 2, 5):
                # A uniform prior gives a uniform pooled count. Conditional on
                # that count, the observed split follows a hypergeometric law.
                allocation = math.prod(math.comb(n, count) for count in counts)
                expected = allocation / ((2 * n + 1) * math.comb(2 * n, sum(counts)))
                value = float(score(1.0, n, y, *summary))
                if expected == 0:
                    assert value == -math.inf
                else:
                    assert value == pytest.approx(math.log(expected), abs=2e-12)


def test_large_balanced_binomial_allocation_in_float32() -> None:
    query = beta_query(
        "Binomial(n,p)", "n = external(posintegers)", "integers", "log(totalmass(P))"
    )
    preparation, online, _ = query.prepare_likelihood("n", "y")
    k = 1_000_000
    with jax.enable_x64(False):
        prepare = preparation.compile(dtype="float32", autodiff=False)
        score = online.compile(dtype="float32", autodiff=False)
        summary = prepare(2 * k, np.full(2, k, dtype=np.int32))
        assert isinstance(summary, tuple)
        # Under a uniform prior the total is uniform. Its division across the
        # two rows is hypergeometric, with mass choose(2k,k)^2 / choose(4k,2k).
        allocation = (
            4 * math.lgamma(2 * k + 1) - 4 * math.lgamma(k + 1) - math.lgamma(4 * k + 1)
        )
        expected = allocation - math.log(4 * k + 1)
        assert float(score(1.0, 2 * k, *summary)) == pytest.approx(expected, abs=3e-4)


@pytest.mark.parametrize("cells", ["integers", "reals"])
def test_ordinary_bernoulli_prepares_numeric_events(cells: str) -> None:
    query = flatppl(f"""
        p = elementof(unitinterval)
        y = external(cartpow({cells},3))
        inputs = (p,y)
        outputs = logdensityof(iid(Bernoulli(p),3),y)
    """)
    prep, online, names = query.prepare_likelihood("y")
    with jax.enable_x64():
        prepare = prep.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64")
        assert [f["name"] for f in score.schema["inputs"]] == ["p", *names]
        for invalid in (False, True):
            values = [1, 0, 2 if invalid else 1]
            packed = prepare(
                np.asarray(values, dtype=float if cells == "reals" else int)
            )
            summary = packed if isinstance(packed, tuple) else (packed,)
            value, grad = jax.value_and_grad(score)(0.25, *summary)
            if invalid:
                assert np.isneginf(value)
                assert float(grad) == pytest.approx(4 - 4 / 3)
            else:
                assert float(value) == pytest.approx(math.log(0.25**2 * 0.75))
                assert float(grad) == pytest.approx(8 - 4 / 3)
