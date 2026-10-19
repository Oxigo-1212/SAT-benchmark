"""Scoring must retain failures, validate witnesses, and reject duplicated observations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_comparison import analyze, score_run


class ComparisonChecks(unittest.TestCase):
    def test_scores_worker_time_and_penalizes_all_failures(self):
        row = dict(status="sat", n=4, wall_s=1.5, total_s=0.1,
                   queen_columns_1based=[2, 4, 1, 3])
        self.assertEqual(score_run(row, 120), 1.5)
        for status in ("timeout", "memory_limit", "error", "invalid", "unknown"):
            self.assertEqual(score_run({"status": status}, 120), 240)
        self.assertEqual(score_run(dict(status="unsat", n=3, wall_s=0.2), 120), 0.2)
        with self.assertRaises(ValueError):
            score_run({**row, "queen_columns_1based": [1, 2, 3, 4]}, 120)
        with self.assertRaises(ValueError):
            score_run({**row, "status": "unsat"}, 120)

    def test_analysis_rejects_duplicates_and_unfinished_experiments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results.json"
            data = dict(metadata=dict(state="completed", timeout_s=120, memory_mib=3072),
                        results=[dict(solver="glucose", encoding="product", formulation="minimal",
                                      n=8, repetition=1, status="memory_limit")])
            path.write_text(json.dumps(data))
            analyze([path], root / "analysis")
            self.assertIn("240.0000", (root / "analysis/comparison.md").read_text())
            with self.assertRaises(ValueError):
                analyze([path, path], root / "analysis")
            data["metadata"]["state"] = "running"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                analyze([path], root / "analysis")


if __name__ == "__main__":
    unittest.main()
