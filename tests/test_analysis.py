import numpy as np
import pytest

from flatppl import CompilationError, InputCell, flatppl


def test_posterior_update_reports_explain_sampling_and_refusal():
    model = """
        p = elementof(unitinterval)
        seed = elementof(nonnegintegers)
        data = elementof(booleans)
        L = likelihoodof(functionof(Bernoulli(p), p=p), data)
        P = bayesupdate(L, joint(p=Beta(1, 1)))
        inputs = (data, seed)
        outputs = rand(normalize(P), rnginit(seed))
    """
    query = flatppl(model)
    (update,) = query.analyze().posterior_updates
    assert (update.binding, update.field, update.family) == ("P", "p", "Beta")
    assert update.backend == "rust-stablehlo"
    assert (update.likelihoods, update.parameters, update.sampling) == (
        ("Bernoulli",),
        None,
        "exact_iid",
    )
    fixed = model.replace("elementof(booleans)", "true").replace("(data, seed)", "seed")
    assert flatppl(fixed).analyze().posterior_updates[0].parameters == (2.0, 1.0)
    with pytest.raises(CompilationError, match="no matching rule") as caught:
        flatppl(model.replace("Bernoulli(p)", "Bernoulli(p*p)")).analyze()
    (refusal,) = caught.value.posterior_updates
    assert (refusal.binding, refusal.family, refusal.sampling) == (
        "P",
        None,
        "lowering_refused",
    )


def test_array_read_facts_normalize_coordinates_without_splitting_parameters():
    query = flatppl(
        "p=elementof(cartprod(v=cartpow(reals,[2,3]),spare=reals))\ninputs=p\noutputs=get(p.v,2,3)*get0(p.v,1,2)+sum(p.v)"
    )
    report = query.analyze()
    assert report.components == (("p",),)
    assert report.input_arrays[0].path == ("p", "v")
    assert report.input_arrays[0].shape == (2, 3)
    assert report.terms[0].input_cells == (InputCell(("p", "v"), (1, 2)),)
    assert not report.terms[0].input_paths
    assert report.terms[1].input_paths == (("p", "v"),)
    assert not report.terms[1].input_cells


def test_structural_report_keeps_signs_multiplicity_paths_and_unused_inputs():
    query = flatppl("""
point=elementof(cartprod(a=reals,b=reals,spare=reals))
y=external(reals)
inputs=(point,y)
f=logdensityof(Normal(point.a,1.),y)
outputs=f+f-logdensityof(Normal(point.b,2.),y)
""")
    report = query.analyze("y")
    assert report.parameters == ("point",)
    assert report.data == ("y",)
    assert [(term.sign, term.multiplicity) for term in report.terms] == [
        (1, 2),
        (-1, 1),
    ]
    assert report.terms[0].input_paths == (("point", "a"), ("y",))
    assert report.terms[1].input_paths == (("point", "b"), ("y",))
    # The root record remains one parameter block, even with static field paths.
    assert report.components == (("point",),)
    query = flatppl(
        "a=elementof(reals)\nb=elementof(reals)\nx=elementof(reals)\ny=external(reals)\ninputs=(a,b,x,y)\noutputs=logdensityof(Normal(a,1.),y)+logdensityof(Normal(b,1.),y)"
    )
    report = query.analyze("y")
    assert report.components == (("a",), ("b",), ("x",))
    assert report.unused_parameters == ("x",)


def test_report_matches_actual_preparation_and_preserves_original_query():
    query = flatppl(
        "mu=elementof(reals)\ny=external(cartpow(reals,4))\ninputs=(mu,y)\noutputs=logdensityof(iid(Normal(mu,1.),4),y)"
    )
    original = query.compile()
    ordinary = query.analyze("y")
    report = query.analyze("y", prepare=True)
    preparation, score, names = query.prepare_likelihood("y")
    assert report.summaries == names
    assert report.preparation_refusal is None
    assert any(item.outcome == "reduced" for item in report.reductions)
    assert any(item.outcome == "reduced" for item in ordinary.reductions)
    assert not ordinary.summaries
    assert not ordinary.preparing
    disabled = query.analyze("y", prepare=True, sufficient_statistics=False)
    assert all(item.outcome == "disabled" for item in disabled.reductions)
    assert disabled.reductions
    assert query.compile().stablehlo == original.stablehlo
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    summaries = preparation.compile()(y)
    value = score.compile()(0.5, *summaries)
    expected = -2 * np.log(2 * np.pi) - 0.5 * np.sum((y - 0.5) ** 2)
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    report = flatppl("a=elementof(reals)\ninputs=a\noutputs=exp(a)").analyze(
        prepare=True
    )
    assert report.preparation_refusal is not None
    assert not report.reductions


def test_normal_candidates_expose_parent_paths_proofs_and_cycles():
    query = flatppl(
        "p=elementof(cartprod(theta=reals,mu=reals,s=interval(.5,4)))\ninputs=p\noutputs=logdensityof(Normal(p.mu,p.s),p.theta)"
    )
    (candidate,) = query.analyze().normal_coordinates
    assert candidate.path == ("p", "theta")
    assert candidate.location_paths == (("p", "mu"),)
    assert candidate.scale_paths == (("p", "s"),)
    assert not candidate.positive_finite_scale_proven
    assert not candidate.finite_location_proven
    assert not candidate.cyclic
    query = flatppl(
        "a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\noutputs=logdensityof(Normal(b,1.),a)+logdensityof(Normal(a,1.),b)"
    )
    assert all(candidate.cyclic for candidate in query.analyze().normal_coordinates)
