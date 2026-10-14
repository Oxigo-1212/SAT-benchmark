"""Native N-Queens implementation using CPSAT."""

import time

from benchmark import NativeResult, _result


def _solve_n_queens(n: int, time_limit: float, threads: int) -> NativeResult:
    import ortools
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    model = cp_model.CpModel()
    queens: list[cp_model.IntVar] = [model.new_int_var(0, n - 1, f"q_{row}") for row in range(n)]
    model.add_all_different(queens)
    model.add_all_different([queens[row] + row for row in range(n)])
    model.add_all_different([queens[row] - row for row in range(n)])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = threads
    loaded = time.perf_counter()
    code = solver.solve(model)
    columns: list[int] | None = (
        [solver.value(q) for q in queens] if code in (cp_model.FEASIBLE, cp_model.OPTIMAL) else None)
    status = "sat" if columns is not None else "unsat" if code == cp_model.INFEASIBLE else (
        "timeout" if code == cp_model.UNKNOWN else "error")
    return _result(started, loaded, columns, status, solver.status_name(code),
                   ortools.__version__, n, 3)
