from solver.solver_interface import solve_cnf_board


def generate_variables(n: int) -> list[list[int]]:
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]


def at_most_one(clauses: list[list[int]], variables: list[int]) -> list[list[int]]:
    for i in range(0, len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses


def exactly_one(clauses: list[list[int]], variables: list[int]) -> None:
    clauses.append(variables)
    at_most_one(clauses, variables)


def generate_clauses(n: int, variables: list[list[int]]) -> list[list[int]]:
    clauses = []

    # Exactly one queen in each row
    for i in range(n):
        exactly_one(clauses, variables[i])

    # Exactly one queen in each column
    for j in range(n):
        exactly_one(clauses, [variables[i][j] for i in range(n)])


    # At most one queen in each diagonal
    for i in range(n):
        for j in range(n):
            for k in range(1, n):
                if i + k < n and j + k < n:
                    at_most_one(clauses, [variables[i][j], variables[i + k][j + k]])
                if i + k < n and j - k >= 0:
                    at_most_one(clauses, [variables[i][j], variables[i + k][j - k]])

    return clauses


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
