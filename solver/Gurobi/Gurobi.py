"""Gurobi N-Queens backend with direct models and CNF clause mapping."""

import time

from helper import NativeResult, _binary_columns, _result, board_lines
from solver.solver_interface import Clauses, clause_to_linear, cnf_variable_count


def _solve_n_queens(n: int, time_limit: float, threads: int,
                    clauses: Clauses | None = None, on_loaded=None) -> NativeResult:
    import gurobipy as gp

    # Record license startup in whole-process wall time, outside model construction.
    with gp.Env(params={"OutputFlag": 0}) as environment:
        started = time.perf_counter()
        with gp.Model("n_queens", env=environment) as model:
            model.Params.Threads = threads
            model.Params.TimeLimit = time_limit
            variables = n * n if clauses is None else cnf_variable_count(clauses, n * n)
            queens = [model.addVar(vtype=gp.GRB.BINARY) for _ in range(variables)]
            count = 0
            if clauses is None:
                for indices, sense in board_lines(n):
                    expression = gp.quicksum(queens[index] for index in indices)
                    model.addConstr(expression == 1 if sense == "E" else expression <= 1)
                    count += 1
            else:
                for clause in clauses:
                    indices, coefficients, rhs = clause_to_linear(clause)
                    model.addConstr(gp.LinExpr(coefficients, [queens[i] for i in indices]) >= rhs)
                count = len(clauses)
            model.update()
            loaded = time.perf_counter()
            if on_loaded is not None:
                on_loaded(loaded - started)
            model.optimize()
            columns: list[int] | None = (
                _binary_columns([q.X for q in queens[:n * n]], n) if model.SolCount else None)
            status = "sat" if columns is not None else "unsat" if model.Status == gp.GRB.INFEASIBLE else (
                "timeout" if model.Status == gp.GRB.TIME_LIMIT else "unknown")
            return _result(started, loaded, columns, status, str(model.Status),
                           ".".join(map(str, gp.gurobi.version())), variables, count)
