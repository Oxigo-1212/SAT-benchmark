"""CP-SAT N-Queens backend with direct models and CNF clause mapping."""

import time

from helper import NativeResult, SOLVER_SEEDS, _binary_columns, _result
from solver.solver_interface import Clauses, cnf_variable_count


def _solve_n_queens(n: int, time_limit: float, threads: int,
                    clauses: Clauses | None = None, on_loaded=None) -> NativeResult:
    import ortools
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    model = cp_model.CpModel()
    if clauses is None:
        queens = [model.new_int_var(0, n - 1, f"q_{row}") for row in range(n)]
        model.add_all_different(queens)
        model.add_all_different([queens[row] + row for row in range(n)])
        model.add_all_different([queens[row] - row for row in range(n)])
        constraints = 3
    else:
        queens = [model.new_bool_var(f"x_{i + 1}")
                  for i in range(cnf_variable_count(clauses, n * n))]
        for clause in clauses:
            model.add_bool_or([queens[abs(lit) - 1] if lit > 0 else queens[-lit - 1].negated()
                               for lit in clause])
        constraints = len(clauses)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = threads
    solver.parameters.random_seed = SOLVER_SEEDS["cp_sat"]
    loaded = time.perf_counter()
    if on_loaded is not None:
        on_loaded(loaded - started)
    code = solver.solve(model)
    columns = None
    if code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        values = [solver.value(q) for q in queens[:n * n]]
        columns = values if clauses is None else _binary_columns(values, n)
    status = "sat" if columns is not None else "unsat" if code == cp_model.INFEASIBLE else (
        "timeout" if code == cp_model.UNKNOWN else "error")
    return _result(started, loaded, columns, status, solver.status_name(code),
                   ortools.__version__, len(queens), constraints)
