# Structured values cross the boundary as JSON text. flatppl._schema types them.
from collections.abc import Sequence
from typing import final

@final
class Error(Exception): ...

@final
class Context:
    def __init__(self) -> None: ...
    def parse(
        self, source: str, name: str | None = None, source_path: str | None = None
    ) -> Module: ...
    def import_pyhf(
        self,
        source: str,
        name: str | None = None,
        source_path: str | None = None,
        measurement: str | None = None,
        observations: str = "bound",
        auxiliary: str = "bound",
    ) -> Module: ...
    def import_formula(
        self,
        spec: str,
        data_ipc: bytes | None = None,
        name: str | None = None,
        source_path: str | None = None,
    ) -> FormulaModel: ...
    def prepare_formula(
        self,
        model: FormulaModel,
        data_ipc: bytes,
        mode: str = "prediction",
    ) -> tuple[Module, str]: ...
    def import_formula_path(
        self,
        spec: str,
        path: str,
        name: str | None = None,
        source_path: str | None = None,
    ) -> FormulaModel: ...
    def prepare_formula_path(
        self,
        model: FormulaModel,
        path: str,
        mode: str = "prediction",
    ) -> tuple[Module, str]: ...
    def load(self, path: str) -> Module: ...
    def register(self, name: str, module: Module) -> None: ...
    def set(self, module: Module, constants: str) -> Module: ...
    def slice(self, module: Module, outputs: Sequence[str]) -> Module: ...
    def transform(
        self, module: Module, logdensity: bool
    ) -> tuple[Module, Module, Module]: ...

@final
class FormulaModel:
    @property
    def module(self) -> Module: ...
    def info(self) -> str: ...
    def recipe(self) -> str: ...

@final
class Module:
    @property
    def source(self) -> str: ...
    @property
    def source_name(self) -> str: ...
    def bindings(self) -> str: ...
    def constant(self, name: str) -> str: ...
    def math(self, format: str) -> str: ...
    def export(
        self,
        dtype: str,
        autodiff: bool,
        integration: tuple[float, float, int] | None = None,
        batch: tuple[Sequence[int], Sequence[Sequence[int | None]]] | None = None,
        target: str = "cpu",
    ) -> str: ...
