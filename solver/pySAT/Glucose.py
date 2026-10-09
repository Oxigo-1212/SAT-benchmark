"""PySAT solver adapter backed by Glucose3."""

from pysat.solvers import Glucose3

from ..solver_interface import Clause, Model, Solver


class Glucose(Solver):
    def __init__(self, glucose: Glucose3) -> None:
        self.glucose = glucose

    def add_clause(self, clause: Clause) -> None:
        self.glucose.add_clause(clause)

    def solve(self) -> Model | None:
        if not self.glucose.solve():
            return None
        model = self.get_model()
        if model is None:
            raise RuntimeError("SAT solver returned no model after a SAT result")
        return model

    def get_model(self) -> Model | None:
        return self.glucose.get_model()

