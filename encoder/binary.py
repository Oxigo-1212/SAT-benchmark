board_size: int


from solver.solver_interface import solve_cnf_board
import math


def generate_variables(n: int) -> list[list[int]]:
    return [[i * n + j + 1 for j in range(n)] for i in range(n)]


def binary_encoding(clauses: list[list[int]], target: int, new_variables: list[int]) -> None:
    for i in range(len(new_variables)):
        clauses.append([-target, new_variables[i]])


def generate_binary_combinations(n: int) -> list[str]:
    binary_combinations = []
    for i in range(1 << n):
        binary_combinations.append(format(i, '0' + str(n) + 'b'))
    return binary_combinations

def generate_new_variables(end: int, length: int) -> list[int]:
    return [i for i in range(end + 1, end + math.ceil(math.log(length, 2)) + 1)]


def at_most_one(clauses: list[list[int]], new_variables: list[int], variables: list[int]) -> None:
    if len(variables) < 2:
        return
    temp_new_variables = generate_new_variables(board_size ** 2 + len(new_variables), len(variables))
    new_variables += temp_new_variables
    binary_combinations = generate_binary_combinations(len(temp_new_variables))

    for i in range(len(variables)):
        combination = binary_combinations[i]
        temp_clause = []
        for j in range(len(combination) - 1, -1, -1):
            index = len(combination) - j - 1
            temp_clause.append({True: temp_new_variables[index], False: -temp_new_variables[index]} [int(combination[j]) == 1])

        binary_encoding(clauses, variables[i], temp_clause)


def exactly_one(clauses: list[list[int]], new_variables: list[int], variables: list[int]) -> None:
    clauses.append(variables)
    at_most_one(clauses, new_variables, variables)


def generate_clauses(n: int, variables: list[list[int]]) -> list[list[int]]:
    global board_size
    board_size = n
    clauses = []
    new_variables = []

    # Exactly one queen in each row
    for i in range(n):
        exactly_one(clauses, new_variables, variables[i])

    # Exactly one queen in each column
    for j in range(n):
        exactly_one(clauses, new_variables, [variables[i][j] for i in range(n)])


    # At most one queen in each diagonal
    for i in range(1, n):
        diagonal = []
        row = i
        col = 0
        while row >= 0 and col < n:
            diagonal.append(variables[row][col])
            row -= 1
            col += 1
        at_most_one(clauses, new_variables, diagonal)

    for j in range(1, n - 1):
        diagonal = []
        row = n - 1
        col = j
        while row >= 0 and col < n:
            diagonal.append(variables[row][col])
            row -= 1
            col += 1
        at_most_one(clauses, new_variables, diagonal)

    for i in range(n - 1):
        diagonal = []
        row = i
        col = 0
        while row < n and col < n:
            diagonal.append(variables[row][col])
            row += 1
            col += 1
        at_most_one(clauses, new_variables, diagonal)

    for j in range(1, n - 1):
        diagonal = []
        row = 0
        col = j
        while row < n and col < n:
            diagonal.append(variables[row][col])
            row += 1
            col += 1
        at_most_one(clauses, new_variables, diagonal)

    return clauses


def solve_n_queens(n: int, backend: str = "glucose") -> list[list[int]] | None:
    variables = generate_variables(n)
    clauses = generate_clauses(n, variables)

    return solve_cnf_board(n, clauses, backend)


def print_solution(solution: list[list[int]] | None) -> None:
    if solution is None:
        print("No solution found.")
    else:
        for row in solution:
            print(" ".join("Q" if cell else "." for cell in row))


if __name__ == "__main__":
    n = 512
    solution = solve_n_queens(n)
    print_solution(solution)

