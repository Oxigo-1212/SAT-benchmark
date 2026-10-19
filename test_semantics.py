"""Exhaustive bounded checks of AMO semantics and projected board solutions."""

from collections import Counter
import importlib
from itertools import chain, product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose3

from helper import ENCODINGS, run_case, summarize_results


def backtracking_solutions(n):
    """Independent row-by-row oracle; no CNF, shared board lines, or validator."""
    solutions = set()

    def visit(columns):
        row = len(columns)
        if row == n:
            solutions.add(tuple(columns))
            return
        for col in range(n):
            if all(col != previous and abs(col - previous) != row - r
                   for r, previous in enumerate(columns)):
                visit(columns + [col])

    visit([])
    return solutions


class SemanticChecks(unittest.TestCase):
    def test_amo_truth_tables(self):
        library_encodings = {"binomial": EncType.pairwise, "binary": EncType.bitwise,
                             "sequential": EncType.seqcounter}
        for encoding in ENCODINGS:
            module = importlib.import_module(f"encoder.{encoding}")
            # Custom encoders depend on board size, including their local branch thresholds.
            board_sizes = (1, 4, 11, 16, 31, 32, 33, 128) if encoding not in library_encodings else (11,)
            for n in board_sizes:
                for size in range(min(n, 11) + 1):
                    for signs in ("positive", "negative", "mixed"):
                        literals = [i if signs == "positive" or (signs == "mixed" and i % 2)
                                    else -i for i in range(1, size + 1)]
                        if encoding in library_encodings:
                            clauses = CardEnc.atmost(literals, bound=1, top_id=n * n,
                                                     encoding=library_encodings[encoding]).clauses
                        else:
                            clauses = []
                            with patch.object(module, "board_size", n, create=True), \
                                    patch.object(module, "new_variables_count", 0):
                                module.at_most_one(clauses, literals)
                        with self.subTest(encoding=encoding, n=n, size=size, signs=signs), \
                                Glucose3(bootstrap_with=clauses) as solver:
                            for values in product((False, True), repeat=size):
                                assumptions = [lit if value else -lit
                                               for lit, value in zip(literals, values)]
                                self.assertEqual(solver.solve(assumptions=assumptions),
                                                 sum(values) <= 1, (literals, values))

    def test_all_projected_solutions(self):
        expected_counts = (1, 0, 0, 2, 10, 4, 40, 92)
        for n, count in enumerate(expected_counts, start=1):
            expected = backtracking_solutions(n)
            self.assertEqual(len(expected), count)
            for encoding in ENCODINGS:
                module = importlib.import_module(f"encoder.{encoding}")
                for formulation in ("minimal", "extended"):
                    clauses = module.generate_clauses(n, module.generate_variables(n), formulation)
                    actual = set()
                    with self.subTest(n=n, encoding=encoding, formulation=formulation), \
                            Glucose3(bootstrap_with=clauses) as solver:
                        while solver.solve():
                            selected = sorted(lit for lit in solver.get_model() if 0 < lit <= n * n)
                            self.assertEqual([(lit - 1) // n for lit in selected], list(range(n)))
                            columns = tuple((lit - 1) % n for lit in selected)
                            self.assertNotIn(columns, actual)
                            actual.add(columns)
                            self.assertLessEqual(len(actual), count)
                            # Block only the placement, never its auxiliary assignments.
                            solver.add_clause([-lit for lit in selected])
                        self.assertEqual(actual, expected)

    def test_formulations_differ_only_by_column_alo(self):
        for encoding in ENCODINGS:
            module = importlib.import_module(f"encoder.{encoding}")
            for n in (1, 2, 3, 8, 11, 32):
                board = module.generate_variables(n)
                extended = module.generate_clauses(n, board)
                minimal = module.generate_clauses(n, board, "minimal")
                column_alo = [tuple(board[r][c] for r in range(n)) for c in range(n)]
                with self.subTest(encoding=encoding, n=n):
                    self.assertEqual(Counter(map(tuple, extended)),
                                     Counter(map(tuple, minimal)) + Counter(column_alo))
                    self.assertEqual(max(map(abs, chain.from_iterable(extended))),
                                     max(map(abs, chain.from_iterable(minimal))))
            with self.assertRaises(ValueError):
                module.generate_clauses(4, module.generate_variables(4), "typo")

    def test_formulation_reaches_worker_and_summary(self):
        with TemporaryDirectory() as directory:
            rows = [run_case("product", 8, 15, 1024, "glucose", log_dir=Path(directory),
                             formulation=formulation) for formulation in ("minimal", "extended")]
        self.assertEqual([row["status"] for row in rows], ["sat", "sat"])
        self.assertEqual([row["formulation"] for row in rows], ["minimal", "extended"])
        self.assertEqual(rows[1]["clauses"] - rows[0]["clauses"], 8)
        self.assertEqual(len(summarize_results(rows)), 2)


if __name__ == "__main__":
    unittest.main()
