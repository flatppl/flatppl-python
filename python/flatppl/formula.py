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
from ._api import Context, Module, native_call
from ._frozen import freeze
from ._tables import ipc

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike

__all__ = ["FormulaView", "from_formula"]


def from_formula(
    formula: str,
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
        formula: An R-style formula, such as ``"y ~ x + group"``.
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
    spec: dict[str, object] = {
        "formula": formula,
        "family": family,
        "options": options,
    }
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

    def prepare(
        self,
        data: object,
        *,
        mode: Literal["prediction", "training"] = "prediction",
        allow_new_levels: bool = False,
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

        Returns:
            A new module in the same context, with its own row selection.

        Raises:
            CompilationError: Data are incompatible with the training recipe.
        """
        context = self.module.context
        native_context = context._native  # pyright: ignore[reportPrivateUsage]
        options = {"allow_new_levels": True} if allow_new_levels else {}
        if isinstance(data, (str, os.PathLike)):
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
        rows = decoded["rows"] if allow_new_levels else decoded
        group_shapes = (
            freeze(decoded["new_group_shapes"]) if allow_new_levels else freeze({})
        )
        module._formula = FormulaView(  # pyright: ignore[reportPrivateUsage]
            module, self.info, self.recipe, tuple(rows), self._native, group_shapes
        )
        return module

    def expected_response(
        self, draws: Mapping[str, object], *, dtype: DTypeLike = "float32"
    ) -> Array:
        """Return conditional means for each posterior draw and retained row.

        Pass the named constrained coordinates returned by
        :attr:`flatppl.inference.LogDensity.constrain`. Coordinate arrays can share any
        leading draw axes, such as chain and draw. Results have the same
        leading axes followed by rows.

        Bernoulli returns success probabilities and Binomial returns expected
        counts. Categorical and ordinal families append a class axis in
        ``info["response_levels"]`` order. Censoring returns latent response
        means. Truncation and mixture families use their predictive laws.
        """
        from ._formula_prediction import evaluate

        if "expected_response" not in self.module.bindings:
            raise ValueError(f"{self.info['family']} has no expected-response query")
        return evaluate(self, "expected_response", draws, dtype=dtype)

    def posterior_predict(
        self, draws: Mapping[str, object], *, key: Array, dtype: DTypeLike = "float32"
    ) -> Array:
        """Draw one response vector per parameter point from the fitted law.

        Draw coordinates and leading axes match :meth:`expected_response`.
        Supply a JAX PRNG key for reproducible sampling. Reusing a key
        reproduces the same draws. Censored models return latent responses.
        Categorical and ordinal outputs are one-based codes in
        ``info["response_levels"]`` order.

        Raises:
            CompilationError: The response kernel has no sample lowering.
        """
        from ._formula_prediction import sample_response

        return sample_response(self, draws, dtype=dtype, key=key)

    def log_likelihood(
        self, draws: Mapping[str, object], *, dtype: DTypeLike = "float32"
    ) -> Array:
        """Return pointwise log likelihoods for each draw and retained row.

        Returns unweighted scores of the fitted observation law, including
        censoring and truncation. Prepare held-out observations with
        ``mode="training"``. Bind encoded ``y_data`` directly for uncensored
        responses. Draw coordinates and leading axes match :meth:`expected_response`.
        """
        from ._formula_prediction import evaluate

        if self.module.bindings["y_data"].kind == "observation":
            raise ValueError(
                'log_likelihood needs observations; use mode="training" or bind y_data'
            )
        return evaluate(self, "log_lik", draws, dtype=dtype)

    def sample_new_groups(
        self, draws: Mapping[str, object], *, key: Array, dtype: DTypeLike = "float32"
    ) -> dict[str, object]:
        """Expand fitted posterior draws for a prepared prediction with new groups.

        Use a JAX PRNG key distinct from the response sampling key. This adds
        independent standard Normal innovations per draw, level and coefficient.
        The fitted model applies group scales and correlations. Rows sharing
        a new level share its effects. Reuse the result across response queries.
        """
        from ._formula_prediction import sample_new_groups

        return sample_new_groups(self, draws, key=key, dtype=dtype)
