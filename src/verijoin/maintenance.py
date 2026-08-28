from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .lineage import (
    LineageSnapshot,
    bind_lineage,
    verify_lineage,
    verify_result_binding,
)
from .schema import Example, Program
from .vm import execute

RefreshAction = Literal["reuse", "replay", "recompile"]


@dataclass(frozen=True, slots=True)
class RefreshResult:
    action: RefreshAction
    answer: str
    snapshot: LineageSnapshot | None
    errors: tuple[str, ...] = ()


def refresh_result(
    example: Example,
    program: Program,
    cached_answer: str,
    snapshot: LineageSnapshot,
) -> RefreshResult:
    """Fail-closed refresh for a materialized learned-query result.

    Unchanged read cells reuse the committed result. A stale snapshot first
    attempts a deterministic VM replay and binds a new snapshot; only an
    unexecutable plan is handed back to the learned compiler.
    """
    if not verify_result_binding(snapshot, program, cached_answer):
        return RefreshResult(
            "recompile",
            "",
            None,
            ("cached program, answer, or VM version does not match snapshot",),
        )
    check = verify_lineage(example, snapshot)
    if check.current:
        return RefreshResult("reuse", cached_answer, snapshot)
    if not snapshot.lineage_certified:
        return RefreshResult(
            "recompile",
            "",
            None,
            ("stale result is not certified for deterministic VM replay",),
        )
    replay = execute(example, program)
    if not replay.valid:
        return RefreshResult("recompile", "", None, replay.errors)
    rebound = bind_lineage(example, program)
    return RefreshResult("replay", replay.answer, rebound)
