"""Native N-Queens implementation using CPLEX."""

import time

from benchmark import NativeResult, _binary_columns, _result, board_lines


def _solve_n_queens(n: int, time_limit: float, threads: int) -> NativeResult:
    import cplex

    started = time.perf_counter()
    with cplex.Cplex() as model:
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.set_warning_stream(None)
        model.parameters.threads.set(threads)
        model.parameters.timelimit.set(time_limit)
        model.variables.add(types="B" * (n * n))
        expressions: list[cplex.SparsePair] = []
        senses: list[str] = []
        for indices, sense in board_lines(n):
            expressions.append(cplex.SparsePair(ind=indices, val=[1] * len(indices)))
            senses.append(sense)
        model.linear_constraints.add(lin_expr=expressions, senses="".join(senses),
                                     rhs=[1] * len(expressions))
        loaded = time.perf_counter()
        model.solve()
        code = model.solution.get_status()
        detail = model.solution.get_status_string()
        columns: list[int] | None = None
        if model.solution.is_primal_feasible():
            columns = _binary_columns(model.solution.get_values(), n)
        statuses = model.solution.status
        status = "sat" if columns is not None else "unsat" if code == statuses.MIP_infeasible else (
            "timeout" if code == statuses.MIP_time_limit_infeasible else "unknown")
        return _result(started, loaded, columns, status, detail, model.get_version(),
                       n * n, len(expressions))
