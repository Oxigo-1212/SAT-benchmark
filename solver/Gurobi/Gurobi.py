"""Native N-Queens implementation using Gurobi."""

import time

from benchmark import NativeResult, _binary_columns, _result, board_lines


def _solve_n_queens(n: int, time_limit: float, threads: int) -> NativeResult:
    import gurobipy as gp

    # Record license startup in whole-process wall time, outside model construction.
    with gp.Env(params={"OutputFlag": 0}) as environment:
        started = time.perf_counter()
        with gp.Model("n_queens", env=environment) as model:
            model.Params.Threads = threads
            model.Params.TimeLimit = time_limit
            queens: list[gp.Var] = [model.addVar(vtype=gp.GRB.BINARY) for _ in range(n * n)]
            count = 0
            for indices, sense in board_lines(n):
                expression = gp.quicksum(queens[index] for index in indices)
                model.addConstr(expression == 1 if sense == "E" else expression <= 1)
                count += 1
            model.update()
            loaded = time.perf_counter()
            model.optimize()
            columns: list[int] | None = (
                _binary_columns([q.X for q in queens], n) if model.SolCount else None)
            status = "sat" if columns is not None else "unsat" if model.Status == gp.GRB.INFEASIBLE else (
                "timeout" if model.Status == gp.GRB.TIME_LIMIT else "unknown")
            return _result(started, loaded, columns, status, str(model.Status),
                           ".".join(map(str, gp.gurobi.version())), n * n, count)
