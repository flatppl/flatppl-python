"""Build FlatPPL regression modules with the bundled Rust formula frontend.

Python transports tables and exposes the converter's metadata. Formula parsing,
contrasts, row selection and prior construction all run in Rust. See
:doc:`/formula` for training and prediction examples.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from . import _native  # pyright: ignore[reportMissingModuleSource]
from ._api import Context, Module, flatppl, native_call
from ._frozen import freeze
from ._tables import ipc

if TYPE_CHECKING:
    import jax
    from jax.typing import DTypeLike

    from ._api import Integration
    from .inference import LogDensity

__all__ = ["FormulaView", "from_formula"]


def _table_source(data: object) -> tuple[bytes | None, str | None]:
    if isinstance(data, (str, os.PathLike)):
        return None, os.fsdecode(
            cast("str | os.PathLike[str] | os.PathLike[bytes]", data)
        )
    return ipc(data), None


def from_formula(
    formula: str | tuple[str, str],
    data: object,
    family: str = "gaussian",
    priors: Sequence[Mapping[str, object]] | None = None,
    *,
    link: str | None = None,
    dpars: Mapping[str, Mapping[str, object]] | None = None,
    nlpars: Mapping[str, str] | None = None,
    context: Context | None = None,
    name: str | None = None,
    **options: object,
) -> Module:
    """Convert a formula and training table to an immutable FlatPPL module.

    The Rust frontend handles the supported brms formula subset and diagnoses
    unsupported families, terms, priors and options. The returned module has
    its training data bound; :attr:`flatppl.Module.formula` retains the template
    and recipe for preparing new data.

    Args:
        formula: An R-style formula, such as ``"y ~ x + group"``. A pair of
            formulas fits two Gaussian responses with correlated residuals.
        data: A CSV, JSON or Arrow IPC path, or a pandas or Polars dataframe,
            Arrow table or record batch, another dataframe interchange producer,
            or a mapping of column names to arrays. Categories, numeric
            precision and nulls reach Rust intact.
        family: The observation family. Rust selects its default link.
        priors: Prior entries, for example
            ``[{"class": "b", "prior": "Normal(0, 2)"}]``. Omitted entries use
            the frontend's brms defaults, including improper Lebesgue priors for
            ordinary regression coefficients.
        link: Override the family's default link.
        dpars: Formulas for family parameters, for example
            ``{"sigma": {"formula": "sigma ~ x", "link": "log"}}``.
            Rust validates each predictor and chooses any omitted link.
        nlpars: Formulas for named nonlinear parameters, for example
            ``{"a": "a ~ 1", "b": "b ~ 1"}`` with ``"y ~ a * exp(b * x)"``.
            Provide an explicit coefficient prior for each parameter.
        context: The owning context. The default creates a new one.
        name: The registry path that relative registry imports start from.
        **options: Rust converter options, such as ``center`` and ``contrasts``.

    Returns:
        The training module, with generated source and formula metadata.

    Raises:
        CompilationError: Rust cannot convert the specification or table.
        TypeError: The data cannot be transported as a table.
        ValueError: The table has duplicate column names.
    """
    spec: dict[str, object] = {"family": family, "options": options}
    if isinstance(formula, str):
        spec["formula"] = formula
    else:
        spec["formula"], spec["paired_formula"] = formula
    if priors is not None:
        spec["priors"] = list(priors)
    if link is not None:
        spec["link"] = link
    if dpars is not None:
        spec["dpars"] = {name: dict(predictor) for name, predictor in dpars.items()}
    if nlpars is not None:
        spec["nlpars"] = dict(nlpars)
    if context is None:
        context = Context()
    source = json.dumps(spec, allow_nan=False)
    if isinstance(data, (str, os.PathLike)):
        model = native_call(
            context._native.import_formula_path,  # pyright: ignore[reportPrivateUsage]
            source,
            os.fsdecode(cast("str | os.PathLike[str] | os.PathLike[bytes]", data)),
            name,
        )
    else:
        model = native_call(
            context._native.import_formula,  # pyright: ignore[reportPrivateUsage]
            source,
            ipc(data),
            name,
        )
    info = freeze(json.loads(model.info()))
    module = Module(model.module, context)
    module._formula = FormulaView(  # pyright: ignore[reportPrivateUsage]
        module,
        info,
        freeze(json.loads(model.recipe())),
        tuple(info["training_rows"]),
        model,
    )
    return module


@dataclass(frozen=True)
class FormulaView:
    """Read-only training metadata and a reusable data preparation recipe.

    Get this view from :attr:`flatppl.Module.formula`. The native converter owns
    the retained template; preparing new rows never changes the training data
    or refits its preprocessing state.

    Attributes:
        module: The module represented by this view.
        info: The converter's model metadata, including formula, family,
            design columns, parameter names and original training row indices.
        recipe: Training levels, contrasts, centring and prior defaults as a
            read-only tree. It is for inspection; preparation uses Rust's copy.
        rows: Zero-based row positions retained from this module's input table.
            These are positions, not pandas index labels.
        new_group_shapes: Shapes of new group innovations per parameter point,
            by sampled binding name. The last axis indexes new levels.
    """

    module: Module = field(repr=False)
    info: Mapping[str, Any]
    recipe: Mapping[str, Any] = field(repr=False)
    rows: tuple[int, ...]
    _native: _native.FormulaModel = field(repr=False, compare=False)

    new_group_shapes: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _response_queries: dict[tuple[str, int], Module] = field(
        default_factory=dict[tuple[str, int], Module],
        init=False,
        repr=False,
        compare=False,
    )

    @cached_property
    def _response_model_name(self) -> str:
        return self.module.context._register_query_model(self.module)  # pyright: ignore[reportPrivateUsage]

    @cached_property
    def group_terms(self) -> Mapping[str, tuple[tuple[str, str], ...]]:
        """Map sampled innovation blocks to their (predictor, grouping column) pairs."""
        terms: dict[str, list[tuple[str, str]]] = {}
        for predictor, recipe in self.recipe["predictors"].items():
            for group in recipe["groups"]:
                terms.setdefault(group["z"], []).append((predictor, group["column"]))
        return MappingProxyType({name: tuple(pairs) for name, pairs in terms.items()})

    def prepare(
        self,
        data: object,
        *,
        mode: Literal["prediction", "training"] = "prediction",
        allow_new_levels: bool = False,
        history: object | None = None,
    ) -> Module:
        """Bind a fresh module using this fit's training recipe.

        Prediction needs no response column. Rows missing required predictors
        are omitted by Rust and reported in the returned view's ``rows``.

        Args:
            data: A table or file path accepted by :func:`from_formula`.
            mode: ``"prediction"`` leaves observations external. ``"training"``
                also binds the supplied response for scoring, using the same
                recipe and prior defaults. Call :func:`from_formula` for a new
                fit or cross-validation fold that needs new defaults.
            allow_new_levels: In prediction mode, retain unseen group levels.
                Expand fitted draws with :meth:`sample_new_groups` before
                evaluating them. Unseen fixed-effect levels always raise.
            history: For Gaussian AR(1) prediction, a table or path with
                observed responses, predictors, group and time. Each group's
                forecast starts at its last complete historical observation.
                Groups absent from history start from the stationary law.

        Returns:
            A new module in the same context, with its own row selection.

        Raises:
            CompilationError: Data are incompatible with the training recipe.
        """
        context = self.module.context
        native_context = context._native  # pyright: ignore[reportPrivateUsage]
        options = {"allow_new_levels": True} if allow_new_levels else {}
        if history is not None:
            if mode != "prediction":
                raise ValueError("history requires prediction mode")
            native, metadata = native_call(
                native_context.prepare_formula_forecast,
                self._native,
                *_table_source(data),
                *_table_source(history),
                **options,
            )
        elif isinstance(data, (str, os.PathLike)):
            native, metadata = native_call(
                native_context.prepare_formula_path,
                self._native,
                os.fsdecode(cast("str | os.PathLike[str] | os.PathLike[bytes]", data)),
                mode,
                **options,
            )
        else:
            native, metadata = native_call(
                native_context.prepare_formula,
                self._native,
                ipc(data),
                mode,
                **options,
            )
        module = Module(native, context)
        decoded = json.loads(metadata)
        prediction_metadata = allow_new_levels or history is not None
        rows = decoded["rows"] if prediction_metadata else decoded
        group_shapes = (
            freeze(decoded["new_group_shapes"]) if prediction_metadata else freeze({})
        )
        module._formula = FormulaView(  # pyright: ignore[reportPrivateUsage]
            module, self.info, self.recipe, tuple(rows), self._native, group_shapes
        )
        return module

    @cached_property
    def _posterior_query(self) -> Module:
        return flatppl(
            f'''_model = load_module("{self._response_model_name}")
            point = elementof(_model.formula_parameter_domain)
            inputs = point
            outputs = logdensityof(_model.Pi_post, point)
            ''',
            context=self.module.context,
        )

    def logdensity(
        self,
        *,
        unconstrained: bool = True,
        jacobian: bool = True,
        dtype: DTypeLike = "float32",
        target: Literal["cpu", "gpu"] | None = None,
        integration: Integration | None = None,
        optimize_forward: bool = False,
    ) -> LogDensity:
        """Prepare a sampler log density over this formula's posterior coordinates.

        Options have the same meaning as in :meth:`flatppl.Module.logdensity`.
        Model points use the sampled binding names from ``Pi_post``. The density
        includes the prior and the likelihood of the bound observations. The
        model needs fixed continuous parameter supports. Score correlated group
        factors in model coordinates with ``unconstrained=False``.
        """
        if not self._has_observations():
            raise ValueError(
                'logdensity needs observations; use mode="training" or bind the encoded response columns'
            )
        if "formula_parameter_domain" not in self.module.bindings:
            raise ValueError(
                "formula logdensity requires fixed continuous parameter supports"
            )

        correlations = {
            name
            for predictor in self.recipe["predictors"].values()
            for group in predictor["groups"]
            if (name := group.get("correlation")) is not None
        }
        if correlations and unconstrained:
            names = ", ".join(sorted(correlations))
            raise ValueError(
                f"no unconstraining chart for correlation factors {names}; "
                "use unconstrained=False to score model coordinates"
            )

        from ._inference import build

        initial: dict[str, object] = {}
        if correlations:
            import numpy as np

            from ._jax import float_dtype

            dtype_name = float_dtype(dtype)
            for name in correlations:
                shape = self.module.bindings[name].shape
                assert shape and shape[0] is not None
                initial[name] = np.eye(shape[0], dtype=dtype_name)
        return build(
            self._posterior_query,
            None,
            unconstrained=unconstrained,
            jacobian=jacobian,
            bounds=True,
            dtype=dtype,
            target=target,
            integration=integration,
            optimize_forward=optimize_forward,
            initial_fields=initial,
        )

    def parameter_draws(
        self, draws: Mapping[str, object], *, dtype: DTypeLike = "float32"
    ) -> dict[str, jax.Array]:
        """Report draws by the coefficient names in ``info["parameters"]``.

        Pass sampled model-coordinate bindings, as for :meth:`expected_response`.
        Each reported array preserves the leading draw axes. Fixed parameters
        repeat their values across draws. Derived parameters use the model's
        bindings, including the intercept on the original predictor scale and
        group effects. Read each name's role from ``info["parameters"]``.
        """
        from ._formula_prediction import parameter_draws

        return parameter_draws(self, draws, dtype=dtype)

    def expected_response(
        self, draws: Mapping[str, object], *, dtype: DTypeLike = "float32"
    ) -> jax.Array:
        """Return conditional means for each posterior draw and retained row.

        Pass the named constrained coordinates returned by
        :attr:`flatppl.inference.LogDensity.constrain`. Coordinate arrays can share any
        leading draw axes, such as chain and draw. Results have the same
        leading axes followed by rows.

        Bernoulli returns success probabilities and Binomial returns expected
        counts. Categorical and ordinal families append a class axis in
        ``info["response_levels"]`` order. Censoring returns latent response
        means. Truncation and mixture families use their predictive laws.
        Paired Gaussian models append a response axis in
        ``info["response_columns"]`` order.
        """
        from ._formula_prediction import evaluate

        if "expected_response" not in self.module.bindings:
            raise ValueError(f"{self.info['family']} has no expected-response query")
        return evaluate(self, "expected_response", draws, dtype=dtype)

    def posterior_predict(
        self,
        draws: Mapping[str, object],
        *,
        key: jax.Array,
        dtype: DTypeLike = "float32",
    ) -> jax.Array:
        """Draw one response vector per parameter point from the fitted law.

        Draw coordinates and leading axes match :meth:`expected_response`.
        Supply a JAX PRNG key for reproducible sampling. Reusing a key
        reproduces the same draws. Censored models return latent responses.
        Categorical and ordinal outputs are one-based codes in
        ``info["response_levels"]`` order. Paired Gaussian models append a
        response axis with both correlated draws in ``info["response_columns"]`` order.

        Raises:
            CompilationError: The response kernel has no sample lowering.
        """
        from ._formula_prediction import sample_response

        return sample_response(self, draws, dtype=dtype, key=key)

    def log_likelihood(
        self, draws: Mapping[str, object], *, dtype: DTypeLike = "float32"
    ) -> jax.Array:
        """Return pointwise log likelihoods for each draw and retained row.

        Returns unweighted scores of the fitted observation law, including
        censoring and truncation. Prepare held-out observations with
        ``mode="training"``. Bind encoded ``y_data`` directly for uncensored
        responses, along with ``y2_data`` for paired Gaussian models. A paired
        model returns one joint score for each row. Draw coordinates and leading
        axes match :meth:`expected_response`.
        """
        from ._formula_prediction import evaluate

        if not self._has_observations():
            raise ValueError(
                'log_likelihood needs observations; use mode="training" or bind response data'
            )
        return evaluate(self, "log_lik", draws, dtype=dtype)

    def _has_observations(self) -> bool:
        names = (
            ("y_data", "y2_data") if "response_columns" in self.info else ("y_data",)
        )
        return all(self.module.bindings[name].kind != "observation" for name in names)

    def sample_new_groups(
        self,
        draws: Mapping[str, object],
        *,
        key: jax.Array,
        dtype: DTypeLike = "float32",
    ) -> dict[str, object]:
        """Expand fitted posterior draws for a prepared prediction with new groups.

        Use a JAX PRNG key distinct from the response sampling key. This adds
        independent standard Normal innovations per draw, level and coefficient.
        The fitted model applies group scales and correlations. Rows sharing
        a new level share its effects. Reuse the result across response queries.
        """
        from ._formula_prediction import sample_new_groups

        return sample_new_groups(self, draws, key=key, dtype=dtype)

    def zero_group_effects(
        self,
        draws: Mapping[str, object],
        *,
        groups: Sequence[str] | None = None,
        dtype: DTypeLike = "float32",
    ) -> dict[str, object]:
        """Return coordinates for prediction with selected group effects set to zero.

        Pass a name or a sequence of names from :attr:`group_terms`. The
        default selects all group blocks. Coordinates for selected blocks
        may be omitted. Other coordinates retain their supplied values and
        determine the draw axes. Pass a selected coordinate to retain its draw
        axes if the model has no other sampled parameters.

        Pass the returned coordinates to :meth:`expected_response`,
        :meth:`posterior_predict` or :meth:`log_likelihood`.
        """
        from ._formula_prediction import zero_group_effects

        return zero_group_effects(self, draws, groups=groups, dtype=dtype)

    def marginal_expected_response(
        self,
        draws: Mapping[str, object],
        *,
        key: jax.Array,
        groups: Sequence[str] | None = None,
        samples: int = 256,
        dtype: DTypeLike = "float32",
    ) -> tuple[jax.Array, jax.Array]:
        """Estimate the response mean averaged over selected fitted group distributions.

        Selection and draw coordinates follow :meth:`zero_group_effects`.
        For each posterior draw, integrate the conditional response mean over
        ``samples`` samples of the selected Gaussian effects given their
        fitted scales and correlations. Rows sharing a level use the same
        effect within each sample. Other coordinates stay fixed at their
        supplied value for each posterior draw.

        Return the estimate and its Monte Carlo standard error, both with the
        shape of :meth:`expected_response`. The standard error describes
        integration noise conditional on each draw. Use a JAX PRNG key and
        at least two samples.
        """
        from ._formula_prediction import marginal_expected_response

        return marginal_expected_response(
            self, draws, key=key, groups=groups, samples=samples, dtype=dtype
        )
