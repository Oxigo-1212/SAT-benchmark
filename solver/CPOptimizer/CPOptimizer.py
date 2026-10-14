"""Native N-Queens implementation using CPOptimizer."""

import os
from pathlib import Path
import shutil
import sys
import time

from benchmark import NativeResult, _result


def _solve_n_queens(n: int, time_limit: float, threads: int) -> NativeResult:
    from docplex.cp.model import CpoModel
    from docplex.cp.expression import CpoIntVar

    executable = os.environ.get("CPO_EXECUTABLE") or shutil.which("cpoptimizer")
    if not executable:
        candidate = Path(sys.executable).parent / "cpoptimizer"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        raise FileNotFoundError("Set CPO_EXECUTABLE to cpoptimizer or put it on PATH")

    started = time.perf_counter()
    model = CpoModel(name="n_queens")
    queens: list[CpoIntVar] = model.integer_var_list(n, 0, n - 1, name="q")
    model.add(model.all_diff(queens))
    model.add(model.all_diff([queens[row] + row for row in range(n)]))
    model.add(model.all_diff([queens[row] - row for row in range(n)]))
    loaded = time.perf_counter()
    solution = model.solve(execfile=executable, TimeLimit=time_limit,
                           Workers=threads, LogVerbosity="Quiet")
    detail = solution.get_solve_status()
    columns: list[int] | None = [solution[q] for q in queens] if solution.is_solution() else None
    status = "sat" if columns is not None else "unsat" if detail == "Infeasible" else (
        "timeout" if solution.get_stop_cause() == "SearchStoppedByLimit" else "unknown")
    version = str(solution.get_process_infos().get("SolverVersion", "unknown"))
    return _result(started, loaded, columns, status, detail, version, n, 3)
