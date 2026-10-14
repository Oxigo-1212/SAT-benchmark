"""Small checks for repeated board sizes, UNSAT boards, and model validation."""

import importlib
import json
import signal
import subprocess
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock
import unittest
from unittest.mock import Mock, patch

from pysat.solvers import Glucose3

from benchmark import BACKENDS, ENCODINGS, SAT_SOLVERS, SOLVERS, main, run_case, validate_columns
from solver.pySAT.Glucose import Glucose
from solver.solver_interface import Solver, decode_board


class BenchmarkChecks(unittest.TestCase):
    def test_cancel_kills_and_reaps_worker(self) -> None:
        process = Mock()
        process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
        process.poll.return_value = None
        with patch("benchmark.subprocess.Popen", return_value=process) as popen, \
                patch("benchmark.os.killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                run_case("binomial", 4, 10, 3072)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_cancel_saves_completed_results(self) -> None:
        completed = {"solver": "glucose", "encoding": "binomial", "n": 4, "status": "sat",
                     "total_s": 0.1, "wall_s": 0.2}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = ["benchmark.py", "--sizes", "4", "8", "16",
                         "--encodings", "binomial", "--solvers", "glucose",
                         "--jobs", "1", "--output", directory]
            with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                    patch("benchmark.run_case", side_effect=[completed, KeyboardInterrupt]) as run:
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 130)
            self.assertEqual(run.call_count, 2)
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(saved["results"], [completed])
            self.assertEqual(saved["metadata"]["stop_reason"], "Cancelled with Ctrl+C")
            self.assertEqual(saved["metadata"]["interrupted_cases"],
                             [{"solver": "glucose", "encoding": "binomial", "n": 8}])
            self.assertEqual(len((output / "results.csv").read_text().splitlines()), 2)
            report = (output / "REPORT.md").read_text()
            self.assertIn("Cancelled with Ctrl+C", report)
            self.assertIn("| 8 | stopped |", report)
            self.assertIn("| 16 | not run |", report)

    def test_cancel_during_worker_launch(self) -> None:
        process = Mock()
        process.poll.return_value = None
        with patch("benchmark.subprocess.Popen", return_value=process), \
                patch("benchmark.os.killpg") as kill, \
                patch("benchmark.signal.pthread_sigmask",
                      side_effect=[set(), KeyboardInterrupt, set()]):
            with self.assertRaises(KeyboardInterrupt):
                run_case("binomial", 4, 10, 3072)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        process.communicate.assert_called_once_with()

    def test_parallel_matrix_and_size_barrier(self) -> None:
        lock = Lock()
        barriers = {n: Barrier(2) for n in (4, 8)}
        started = {4: 0, 8: 0}
        completed = {4: 0, 8: 0}
        active = peak = 0
        cases_per_n = len(SAT_SOLVERS) * len(ENCODINGS) + len(BACKENDS)

        def run(encoding, n, timeout, memory, solver, cancel):
            nonlocal active, peak
            with lock:
                if n == 8:
                    self.assertEqual(completed[4], cases_per_n)
                started[n] += 1
                first_pair = started[n] <= 2
                active += 1
                peak = max(peak, active)
            if first_pair:
                barriers[n].wait(timeout=5)
            with lock:
                active -= 1
                completed[n] += 1
            return {"solver": solver, "encoding": encoding, "n": n,
                    "status": "error" if solver == "gurobi" and n == 4 else "sat",
                    "total_s": 0.1, "wall_s": 0.2}

        with TemporaryDirectory() as directory:
            arguments = ["benchmark.py", "--sizes", "4", "8", "--jobs", "2",
                         "--output", directory]
            with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                    patch("benchmark.run_case", side_effect=run):
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(peak, 2)
            saved = json.loads((Path(directory) / "results.json").read_text())
            expected = {(n, solver, encoding) for n in (4, 8) for solver in SOLVERS
                        for encoding in (ENCODINGS if solver in SAT_SOLVERS else ("native",))}
            self.assertEqual(len(saved["results"]), len(expected))
            self.assertEqual({(row["n"], row["solver"], row["encoding"])
                              for row in saved["results"]}, expected)
            self.assertEqual(saved["metadata"]["jobs"], 2)
            report = (Path(directory) / "REPORT.md").read_text()
            for solver in SOLVERS:
                self.assertIn(f"### {solver}", report)
            self.assertIn("| 4 | error |", report)
            self.assertIn("--jobs 2", report)

    def test_cancel_event_kills_running_worker(self) -> None:
        cancel = Event()
        process = Mock()
        process.poll.return_value = None

        def communicate(**kwargs):
            if kwargs:
                cancel.set()
                raise subprocess.TimeoutExpired("worker", 0.1)
            return "", ""

        process.communicate.side_effect = communicate
        with patch("benchmark.subprocess.Popen", return_value=process), \
                patch("benchmark.os.killpg") as kill:
            with self.assertRaises(CancelledError):
                run_case("binomial", 4, 10, 3072, cancel=cancel)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)

    def test_cancel_event_skips_worker_launch(self) -> None:
        cancel = Event()
        cancel.set()
        with patch("benchmark.subprocess.Popen") as popen:
            with self.assertRaises(CancelledError):
                run_case("binomial", 4, 10, 3072, cancel=cancel)
        popen.assert_not_called()

    def test_timeout_preserves_partial_result(self) -> None:
        process = Mock()
        process.poll.return_value = -9
        process.communicate.return_value = ('{"phase": "loading", "generation_s": 0.5}\n', "")
        with patch("benchmark.subprocess.Popen", return_value=process), \
                patch("benchmark.os.killpg") as kill, \
                patch("benchmark.time.perf_counter", side_effect=[0, 20, 20]), \
                patch("benchmark.Path.read_text", side_effect=FileNotFoundError):
            result = run_case("binomial", 4, 10, 3072, "cadical")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["solver"], "cadical")
        self.assertEqual(result["generation_s"], 0.5)
        self.assertEqual(result["phase"], "loading")
        kill.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_invalid_cli_limits(self) -> None:
        for arguments in (["--jobs", "0"], ["--sizes", "1"],
                          ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--memory-mib", "0"]):
            with self.subTest(arguments=arguments), \
                    patch("benchmark.sys.argv", ["benchmark.py", *arguments]), \
                    patch("benchmark.sys.stderr"):
                with self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 2)

    def test_solver_interface(self) -> None:
        with Glucose3() as glucose:
            solver = Glucose(glucose)
            self.assertIsInstance(solver, Solver)
            self.assertIs(solver.glucose, glucose)
            solver.add_clause([1])
            solver.add_clause([-2])
            self.assertEqual(solver.solve(), [1, -2])
            self.assertEqual(decode_board(solver.get_model(), 2),
                             [[1, 0], [0, 0]])
            solver.add_clause([-1])
            self.assertIsNone(solver.solve())
            self.assertIsNone(solver.get_model())

    def test_injected_backend(self) -> None:
        glucose = Mock(spec=Glucose3)
        glucose.solve.return_value = True
        glucose.get_model.return_value = [1, -2]
        solver = Glucose(glucose)
        solver.add_clause([1])
        self.assertIs(solver.solve(), glucose.get_model.return_value)
        glucose.add_clause.assert_called_once_with([1])
        glucose.solve.assert_called_once_with()
        glucose.get_model.assert_called_once_with()

    def test_missing_model(self) -> None:
        glucose = Mock(spec=Glucose3)
        glucose.solve.return_value = True
        glucose.get_model.return_value = None
        with self.assertRaisesRegex(RuntimeError, "returned no model"):
            Glucose(glucose).solve()

    def test_decode_board(self) -> None:
        self.assertEqual(decode_board([5, -4, 2, -1, -3], 2),
                         [[0, 1], [0, 0]])
        self.assertEqual(decode_board([-1, -2, -3, -4, 5], 2),
                         [[0, 0], [0, 0]])
        self.assertEqual(decode_board([], 0), [])

    def test_shared_solver(self) -> None:
        with Glucose3() as glucose:
            solver = Glucose(glucose)
            solver.add_clause([1])
            self.assertEqual(decode_board(solver.solve(), 1), [[1]])
            solver.add_clause([-1])
            self.assertIsNone(solver.solve())

    def test_repeated_sizes_and_unsat(self) -> None:
        for encoding in ENCODINGS:
            module = importlib.import_module(f"encoder.{encoding}")
            for n in (4, 8, 5, 2, 3, 1, 4):
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
