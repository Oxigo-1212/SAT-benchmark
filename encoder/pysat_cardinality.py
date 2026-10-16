"""Build N-Queens CNF with PySAT's cardinality encoders."""

from pysat.card import CardEnc
from pysat.formula import IDPool

from helper import board_lines


def generate_variables(n: int) -> list[list[int]]:
    return [[row * n + col + 1 for col in range(n)] for row in range(n)]


def generate_clauses(n: int, variables: list[list[int]], encoding: int) -> list[list[int]]:
    board = [variable for row in variables for variable in row]
    # Reserve the whole board before any constraint allocates auxiliary IDs.
    pool = IDPool(start_from=max(board, default=0) + 1)
    clauses = []
    for indices, sense in board_lines(n):
        encode = CardEnc.equals if sense == "E" else CardEnc.atmost
        cnf = encode(lits=[board[index] for index in indices], bound=1,
                     vpool=pool, encoding=encoding)
        clauses.extend(cnf.clauses)
    return clauses
