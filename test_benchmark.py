"""Small checks for repeated board sizes, UNSAT boards, and model validation."""

import importlib
import unittest

from benchmark import ENCODINGS, validate_columns


class BenchmarkChecks(unittest.TestCase):
    def test_repeated_sizes_and_unsat(self) -> None:
        for encoding in ENCODINGS:
            module = importlib.import_module(encoding)
            for n in (4, 8, 5, 2, 3, 4):
                with self.subTest(encoding=encoding, n=n):
                    solution = module.solve_n_queens(n)
                    if n in (2, 3):
                        self.assertIsNone(solution)
                    else:
                        self.assertEqual(len(solution), n)
                        self.assertTrue(all(sum(row) == 1 for row in solution))
                        columns = [row.index(1) for row in solution]
                        self.assertTrue(validate_columns(columns, n))

    def test_validation_rejects_attacks(self) -> None:
        self.assertTrue(validate_columns([1, 3, 0, 2], 4))
        for columns in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 0, 2],
                        [1, 3, 0], [1, 4, 0, 2]):
            self.assertFalse(validate_columns(columns, 4))


if __name__ == "__main__":
    unittest.main()
