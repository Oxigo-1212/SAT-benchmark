"""CPLEX N-Queens backend with direct models and CNF clause mapping."""

import time

from helper import NativeResult, SOLVER_SEEDS, _binary_columns, _result, board_lines
from solver.solver_interface import Clauses, clause_to_linear, cnf_variable_count


def _solve_n_queens(n: int, time_limit: float, threads: int,
                    clauses: Clauses | None = None, on_loaded=None) -> NativeResult:
    import cplex

    started = time.perf_counter()
    with cplex.Cplex() as model:
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.set_warning_stream(None)
        model.parameters.threads.set(threads)
        model.parameters.timelimit.set(time_limit)
        model.parameters.randomseed.set(SOLVER_SEEDS["cplex_mip"])
        variables = n * n if clauses is None else cnf_variable_count(clauses, n * n)
        model.variables.add(types="B" * variables)
        if clauses is None:
            expressions = []
            senses = []
            for indices, sense in board_lines(n):
                expressions.append(cplex.SparsePair(ind=indices, val=[1] * len(indices)))
                senses.append(sense)
            model.linear_constraints.add(lin_expr=expressions, senses="".join(senses),
                                         rhs=[1] * len(expressions))
            constraints = len(expressions)
        else:
            for clause in clauses:
                indices, coefficients, rhs = clause_to_linear(clause)
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=indices, val=coefficients)], senses="G", rhs=[rhs])
            constraints = len(clauses)
        loaded = time.perf_counter()
        if on_loaded is not None:
            on_loaded(loaded - started)
        model.solve()
        code = model.solution.get_status()
        detail = model.solution.get_status_string()
        columns: list[int] | None = None
        if model.solution.is_primal_feasible():
            columns = _binary_columns(model.solution.get_values(0, n * n - 1), n)
        statuses = model.solution.status
        status = "sat" if columns is not None else "unsat" if code == statuses.MIP_infeasible else (
            "timeout" if code == statuses.MIP_time_limit_infeasible else "unknown")
        return _result(started, loaded, columns, status, detail, model.get_version(),
                       variables, constraints)
