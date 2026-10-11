"""Hooks for the executed example gallery: script arguments and runtime budgets."""

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

EXAMPLES = Path(__file__).parents[1] / "examples"
BUDGET = re.compile(r"^Runtime budget: (\d+) s\b", re.MULTILINE)
_started: dict[str, float] = {}


def reset_argv(gallery_conf: dict[str, Any], script_vars: dict[str, Any]) -> list[str]:
    """Run the corpus example on one target with its short sampling run."""
    if Path(script_vars["src_file"]).name != "corpus.py":
        return []
    output = Path(tempfile.mkdtemp()) / "result.json"
    directory = os.environ["FLATPPL_EXAMPLES_DIR"]
    return [
        "poisson-model",
        directory,
        str(output),
        "--warmup",
        "64",
        "--samples",
        "64",
    ]


def enforce_budget(gallery_conf: dict[str, Any], fname: str | None, when: str) -> None:
    """Fail the build when an example runs longer than the budget it states."""
    if fname is None:
        return
    if when == "before":
        _started[fname] = time.monotonic()
        return
    elapsed = time.monotonic() - _started.pop(fname)
    stated = BUDGET.search((EXAMPLES / fname).read_text())
    if stated is None:
        raise RuntimeError(f"{fname} states no runtime budget")
    if elapsed > int(stated[1]):
        raise RuntimeError(
            f"{fname} ran {elapsed:.0f} s, over its budget of {stated[1]} s"
        )
