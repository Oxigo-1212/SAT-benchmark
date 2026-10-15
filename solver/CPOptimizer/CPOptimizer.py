"""CP Optimizer N-Queens backend with direct models and CNF clause mapping."""

import os
from pathlib import Path
import shutil
import sys
import time

from helper import NativeResult, _binary_columns, _result
from solver.solver_interface import Clauses, cnf_variable_count


def _solve_n_queens(n: int, time_limit: float, threads: int,
                    clauses: Clauses | None = None, on_loaded=None) -> NativeResult:
    from docplex.cp.model import CpoModel

    executable = os.environ.get("CPO_EXECUTABLE") or shutil.which("cpoptimizer")
    if not executable:
        candidate = Path(sys.executable).parent / "cpoptimizer"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        raise FileNotFoundError("Set CPO_EXECUTABLE to cpoptimizer or put it on PATH")

    started = time.perf_counter()
    model = CpoModel(name="n_queens")
    if clauses is None:
        queens = model.integer_var_list(n, 0, n - 1, name="q")
        model.add(model.all_diff(queens))
        model.add(model.all_diff([queens[row] + row for row in range(n)]))
        model.add(model.all_diff([queens[row] - row for row in range(n)]))
        constraints = 3
    else:
        queens = model.binary_var_list(cnf_variable_count(clauses, n * n), name="x")
        for clause in clauses:
            model.add(model.logical_or([queens[abs(lit) - 1] == int(lit > 0) for lit in clause]))
        constraints = len(clauses)
    loaded = time.perf_counter()
    if on_loaded is not None:
        on_loaded(loaded - started)
    solution = model.solve(execfile=executable, TimeLimit=time_limit,
                           Workers=threads, LogVerbosity="Quiet")
    detail = solution.get_solve_status()
    columns = None
    if solution.is_solution():
        values = [solution[q] for q in queens[:n * n]]
        columns = values if clauses is None else _binary_columns(values, n)
    status = "sat" if columns is not None else "unsat" if detail == "Infeasible" else (
        "timeout" if solution.get_stop_cause() == "SearchStoppedByLimit" else "unknown")
    version = str(solution.get_process_infos().get("SolverVersion", "unknown"))
    return _result(started, loaded, columns, status, detail, version, len(queens), constraints)
