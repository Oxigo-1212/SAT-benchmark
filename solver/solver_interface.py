from __future__ import annotations

import importlib
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


def cnf_variable_count(clauses: Clauses, minimum: int = 0) -> int:
    """Validate signed, one-based CNF literals and include auxiliary variables."""
    count = minimum
    for clause in clauses:
        for literal in clause:
            if isinstance(literal, bool) or not isinstance(literal, int) or literal == 0:
                raise ValueError("CNF literals must be nonzero integers")
            count = max(count, abs(literal))
    return count


def clause_to_linear(clause: Clause) -> tuple[list[int], list[int], int]:
    """Map OR to sum(positive x) - sum(negative x) >= 1 - negative_count."""
    coefficients: dict[int, int] = {}
    rhs = 1
    for literal in clause:
        index = abs(literal) - 1
        coefficients[index] = coefficients.get(index, 0) + (1 if literal > 0 else -1)
        rhs -= int(literal < 0)
    return list(coefficients), list(coefficients.values()), rhs


def decode_board(solution: Model, n: int) -> list[list[int]]:
    """Decode row-major board variables, ignoring auxiliary literals."""
    positive = {literal for literal in solution if 0 < literal <= n * n}
    return [[int(row * n + col + 1 in positive) for col in range(n)]
            for row in range(n)]


def solve_cnf_board(n: int, clauses: Clauses, backend: str) -> list[list[int]] | None:
    """Keep the encoders' board API while dispatching CNF to any configured solver."""
    from helper import SAT_SOLVERS, solve_n_queens

    if backend in SAT_SOLVERS:
        cnf_variable_count(clauses, n * n)
        adapter_name, backend_name = SAT_SOLVERS[backend]
        adapter = getattr(importlib.import_module(f"solver.pySAT.{adapter_name}"), adapter_name)
        solver_class = getattr(importlib.import_module("pysat.solvers"), backend_name)
        with solver_class(bootstrap_with=clauses) as native:
            model = adapter(native).solve()
            return None if model is None else decode_board(model, n)

    result = solve_n_queens(n, backend, clauses=clauses)
    if result["status"] == "unsat":
        return None
    if result["status"] == "timeout":
        raise TimeoutError(f"{backend}: {result['backend_status']}")
    if result["status"] != "sat":
        raise RuntimeError(f"{backend}: {result['backend_status']}")
    return decode_board([row * n + col + 1 for row, col in enumerate(result["columns"])], n)


