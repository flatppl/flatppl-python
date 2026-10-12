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
from typing import Any, Literal, cast

from . import _native  # pyright: ignore[reportMissingModuleSource]
from ._api import Context, Module, native_call
from ._frozen import freeze
from ._tables import ipc

__all__ = ["FormulaView", "from_formula"]


def from_formula(
    formula: str,
    data: object,
    family: str = "gaussian",
    priors: Sequence[Mapping[str, object]] | None = None,
    *,
    link: str | None = None,
    dpars: Mapping[str, Mapping[str, object]] | None = None,
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
            the frontend's brms defaults, including improper Flat priors for
            ordinary regression coefficients.
        link: Override the family's default link.
        dpars: Formulas for family parameters, for example
            ``{"sigma": {"formula": "sigma ~ x", "link": "log"}}``.
            Rust validates each predictor and chooses any omitted link.
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
    """

    module: Module = field(repr=False)
    info: Mapping[str, Any]
    recipe: Mapping[str, Any] = field(repr=False)
    rows: tuple[int, ...]
    _native: _native.FormulaModel = field(repr=False, compare=False)

    def prepare(
        self, data: object, *, mode: Literal["prediction", "training"] = "prediction"
    ) -> Module:
        """Bind a fresh module using this fit's training recipe.

        Prediction needs no response column. Rows missing required predictors
        are omitted by Rust and reported in the returned view's ``rows``.
        Unseen factor/group levels are diagnosed instead of recoded.

        Args:
            data: A table or file path accepted by :func:`from_formula`.
            mode: ``"prediction"`` leaves observations external. ``"training"``
                also binds the supplied response for scoring, using the same
                recipe and prior defaults. Call :func:`from_formula` for a new
                fit or cross-validation fold that needs new defaults.

        Returns:
            A new module in the same context, with its own row selection.

        Raises:
            CompilationError: Data are incompatible with the training recipe.
        """
        if isinstance(data, (str, os.PathLike)):
            native, rows = native_call(
                self.module.context._native.prepare_formula_path,  # pyright: ignore[reportPrivateUsage]
                self._native,
                os.fsdecode(cast("str | os.PathLike[str] | os.PathLike[bytes]", data)),
                mode,
            )
        else:
            native, rows = native_call(
                self.module.context._native.prepare_formula,  # pyright: ignore[reportPrivateUsage]
                self._native,
                ipc(data),
                mode,
            )
        module = Module(native, self.module.context)
        module._formula = FormulaView(  # pyright: ignore[reportPrivateUsage]
            module, self.info, self.recipe, tuple(json.loads(rows)), self._native
        )
        return module
