"""CryptoMiniSat adapter for the shared solver interface."""

from pysat.solvers import CryptoMinisat

from ..solver_interface import Clause, Model, Solver


class CryptoMiniSat(Solver):
    def __init__(self, cryptominisat: CryptoMinisat) -> None:
        self.cryptominisat = cryptominisat

    def add_clause(self, clause: Clause) -> None:
        self.cryptominisat.add_clause(clause)

    def solve(self) -> Model | None:
        if not self.cryptominisat.solve():
            return None
        model = self.get_model()
        if model is None:
            raise RuntimeError("SAT solver returned no model after a SAT result")
        return model

    def get_model(self) -> Model | None:
        return self.cryptominisat.get_model()

