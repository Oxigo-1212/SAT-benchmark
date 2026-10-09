from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

Clause: TypeAlias = list[int]
Clauses: TypeAlias = list[list[int]]
Variable: TypeAlias = int
Model: TypeAlias = list[int]


@runtime_checkable
class Solver(Protocol):
    def add_clause(self, clause: Clause) -> None: ...

    def solve(self) -> Model | None: ...

    def get_model(self) -> Model | None: ...


def decode_board(solution: Model, n: int) -> list[list[int]]:
    """Decode row-major board variables, ignoring auxiliary literals."""
    positive = {literal for literal in solution if 0 < literal <= n * n}
    return [[int(row * n + col + 1 in positive) for col in range(n)]
            for row in range(n)]


