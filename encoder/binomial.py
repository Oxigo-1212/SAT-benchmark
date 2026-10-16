"""N-Queens CNF using PySAT's pairwise cardinality encoding."""

from pysat.card import EncType

from encoder.pysat_cardinality import generate_clauses as _generate_clauses, generate_variables
from solver.solver_interface import solve_cnf_board


def generate_clauses(n: int, variables: list[list[int]]) -> list[list[int]]:
    return _generate_clauses(n, variables, EncType.pairwise)


def solve_n_queens(n: int, backend: str = "glucose") -> list[list[int]] | None:
    variables = generate_variables(n)
    clauses = generate_clauses(n, variables)

    return solve_cnf_board(n, clauses, backend)


def print_solution(solution: list[list[int]] | None) -> None:
    if solution is None:
        print("No solution found.")
    else:
        print(solution)
        for row in solution:
            print(" ".join("Q" if cell else "." for cell in row))


if __name__ == "__main__":
    n = 8
    # print(generate_clauses(n, generate_variables(n)))
    solution = solve_n_queens(n)
    print_solution(solution)
