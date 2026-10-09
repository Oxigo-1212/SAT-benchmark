"""CaDiCaL adapter for the shared solver interface."""

from pysat.solvers import Cadical300

from ..solver_interface import Clause, Model, Solver


class CaDiCaL(Solver):
    def __init__(self, cadical: Cadical300) -> None:
        self.cadical = cadical

    def add_clause(self, clause: Clause) -> None:
        self.cadical.add_clause(clause)

    def solve(self) -> Model | None:
        if not self.cadical.solve():
            return None
        model = self.get_model()
        if model is None:
            raise RuntimeError("SAT solver returned no model after a SAT result")
        return model

    def get_model(self) -> Model | None:
        return self.cadical.get_model()

