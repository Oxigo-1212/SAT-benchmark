"""Single-run Kissat adapter for the shared solver interface."""

from pysat.solvers import Kissat404

from ..solver_interface import Clause, Model, Solver


class Kissat(Solver):
    def __init__(self, kissat: Kissat404) -> None:
        self.kissat = kissat
        self._solved = kissat.get_status() is not None

    def add_clause(self, clause: Clause) -> None:
        if self._solved:
            raise RuntimeError("Kissat is non-incremental; use a fresh backend after solving")
        self.kissat.add_clause(clause)

    def solve(self) -> Model | None:
        if self._solved:
            raise RuntimeError("Kissat is non-incremental; use a fresh backend after solving")
        self._solved = True
        if not self.kissat.solve():
            return None
        model = self.get_model()
        if model is None:
            raise RuntimeError("SAT solver returned no model after a SAT result")
        return model

    def get_model(self) -> Model | None:
        return self.kissat.get_model()

