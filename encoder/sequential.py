board_size: int

from pysat.solvers import Glucose3

from solver.pySAT.Glucose import Glucose
from solver.solver_interface import decode_board

new_variables_count: int = 0
def generate_variables(n: int) -> list[list[int]]:
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]

def generate_new_variables(new_variables_count: int, length: int) -> list[int]:
    start = board_size ** 2 + new_variables_count
    return [i for i in range(start + 1,start + length)]


def at_most_one(clauses: list[list[int]], variables: list[int]) -> None:
    if len(variables) < 2:
        return
    global new_variables_count
    new_variables = generate_new_variables(new_variables_count, len(variables))
    new_variables_count += (len(variables) - 1)
    clauses.append([-variables[0], new_variables[0]])
    for i in range(1, len(variables) - 1):
        clauses.append([-variables[i], new_variables[i]])
        clauses.append([-new_variables[i - 1], new_variables[i]])
        clauses.append([-new_variables[i - 1], -variables[i]])
    clauses.append([-new_variables[len(variables) - 2], -variables[len(variables) - 1]])

def exactly_one(clauses: list[list[int]], variables: list[int]) -> None:
    clauses.append(variables)
    at_most_one(clauses, variables)


def generate_clauses(n: int, variables: list[list[int]]) -> list[list[int]]:
    global board_size, new_variables_count
    board_size = n
    new_variables_count = 0
    clauses = []

    # Exactly one queen in each row
    for row in range(n):
        exactly_one(clauses, variables[row])

    # Exactly one queen in each column
    for col in range(n):
        exactly_one(clauses, [variables[row][col] for row in range(n)])


    # At most one queen in each diagonal
    for i in range(1, n):
        diagonal = []
        row = i
        col = 0
        while row >= 0 and col < n:
            diagonal.append(variables[row][col])
            row -= 1
            col += 1
        at_most_one(clauses, diagonal)

    for j in range(1, n - 1):
        diagonal = []
        row = n - 1
        col = j
        while row >= 0 and col < n:
            diagonal.append(variables[row][col])
            row -= 1
            col += 1
        at_most_one(clauses, diagonal)

    for i in range(n - 1):
        diagonal = []
        row = i
        col = 0
        while row < n and col < n:
            diagonal.append(variables[row][col])
            row += 1
            col += 1
        at_most_one(clauses, diagonal)

    for j in range(1, n - 1):
        diagonal = []
        row = 0
        col = j
        while row < n and col < n:
            diagonal.append(variables[row][col])
            row += 1
            col += 1
        at_most_one(clauses, diagonal)

    return clauses


def solve_n_queens(n: int) -> list[list[int]] | None:
    variables = generate_variables(n)
    clauses = generate_clauses(n, variables)

    with Glucose3(bootstrap_with=clauses) as glucose:
        model = Glucose(glucose).solve()
        return None if model is None else decode_board(model, n)


def print_solution(solution: list[list[int]] | None) -> None:
    if solution is None:
        print("No solution found.")
    else:
        for row in solution:
            print(" ".join("Q" if cell else "." for cell in row))


if __name__ == "__main__":
    n = 64
    solution = solve_n_queens(n)
    print_solution(solution)

