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
        *,
        allow_new_levels: bool = False,
    ) -> tuple[Module, str]: ...
    def prepare_formula_forecast(
        self,
        model: FormulaModel,
        data_ipc: bytes | None,
        path: str | None,
        history_ipc: bytes | None,
        history_path: str | None,
        *,
        allow_new_levels: bool = False,
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
        *,
        allow_new_levels: bool = False,
    ) -> tuple[Module, str]: ...
    def load(self, path: str) -> Module: ...
    def load_bundle(self, name: str, data: bytes) -> Module: ...
    def register(self, name: str, module: Module) -> None: ...
    def contains(self, name: str) -> bool: ...
    def pack(self, module: Module) -> bytes: ...
    def fingerprint(self) -> bytes: ...
    def set(self, module: Module, constants: str) -> Module: ...
    def slice(self, module: Module, outputs: Sequence[str]) -> Module: ...
    def analyze(
        self,
        module: Module,
        data_inputs: Sequence[str],
        preparing: bool,
        sufficient_statistics: bool,
    ) -> str: ...
    def prepare_likelihood(
        self, module: Module, data_inputs: Sequence[str], *, pack_summaries: bool = True
    ) -> tuple[Module, Module, list[str]]: ...
    def prepare_incremental_score(
        self, module: Module, data_inputs: Sequence[str]
    ) -> IncrementalPlan: ...
    def normal_coordinate(
        self,
        module: Module,
        data_inputs: Sequence[str],
        parameter: Sequence[str],
        logdensity: bool,
        dtype: str,
    ) -> tuple[Module, Module, Module, Module, str]: ...
    def transform(
        self,
        module: Module,
        logdensity: bool,
        correlation_factors: Sequence[str] | None = None,
    ) -> tuple[Module, Module, Module]: ...
    def normal_coordinates(
        self,
        module: Module,
        data_inputs: Sequence[str],
        parameters: Sequence[Sequence[str]],
        logdensity: bool,
        dtype: str,
    ) -> tuple[Module, Module, Module, Module, list[str], list[list[str]]]: ...

@final
class FormulaModel:
    @property
    def module(self) -> Module: ...
    def info(self) -> str: ...
    def recipe(self) -> str: ...

@final
class IncrementalPlan:
    def terms(
        self,
    ) -> list[
        tuple[
            Module, list[str], str, list[list[str]], list[tuple[list[str], list[int]]]
        ]
    ]: ...
    def combine(self) -> Module: ...
    def inputs(self) -> list[str]: ...
    def affected_terms(self, changed: Sequence[str]) -> list[int]: ...
    def affected_paths(self, changed: Sequence[Sequence[str]]) -> list[int]: ...
    def affected_cells(
        self, changed: Sequence[tuple[Sequence[str], Sequence[int]]]
    ) -> list[int]: ...
    def revisions(self) -> ScoreRevisions: ...

@final
class ScoreTicket: ...

@final
class ScoreRevisions:
    def invalidate(self, changed: Sequence[str]) -> list[int]: ...
    def invalidate_paths(self, changed: Sequence[Sequence[str]]) -> list[int]: ...
    def invalidate_cells(
        self, changed: Sequence[tuple[Sequence[str], Sequence[int]]]
    ) -> list[int]: ...
    def ticket(self, term: int) -> ScoreTicket: ...
    def accept(self, ticket: ScoreTicket) -> bool: ...
    def is_current(self, term: int) -> bool: ...

@final
class Module:
    def pack(self) -> bytes: ...
    def fingerprint(self) -> bytes: ...
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
        sufficient_statistics: bool = True,
        prepare_fixed_data: bool = False,
    ) -> str: ...
