"""Small checks for repeated board sizes, UNSAT boards, and model validation."""

import csv
import importlib
import itertools
import json
import signal
import subprocess
import sys
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
import unittest
from unittest.mock import Mock, patch

from pysat.solvers import Cadical300, Glucose3

from benchmark import (
    BACKENDS, ENCODINGS, SAT_SOLVERS, SOLVERS, main, run_case, validate_columns,
)
from helper import ENCODING_IMPLEMENTATIONS, SOLVER_SEEDS, sat_worker, save_results, summarize_results
from solver.pySAT.Glucose import Glucose
from solver.solver_interface import Solver, decode_board


class BenchmarkChecks(unittest.TestCase):
    def test_cadical_seed_is_applied_before_loading_clauses(self) -> None:
        result = {}
        with patch.object(Cadical300, "configure", autospec=True,
                          side_effect=Cadical300.configure) as configure, patch("helper.emit"):
            sat_worker("product", 8, "cadical", result)
        self.assertEqual(configure.call_args.args[1], {"seed": SOLVER_SEEDS["cadical"]})
        self.assertEqual(result["status"], "sat")

    def test_cardinality_encodings_match_board_constraints(self) -> None:
        for encoding in ("binomial", "binary", "sequential", "commander"):
            module = importlib.import_module(f"encoder.{encoding}")
            for n in (1, 2, 3, 4, 5):
                variables = module.generate_variables(n)
                clauses = module.generate_clauses(n, variables)
                with self.subTest(encoding=encoding, n=n), Glucose3(bootstrap_with=clauses) as solver:
                    for columns in itertools.product(range(n), repeat=n):
                        assumptions = [variable if columns[row] == col else -variable
                                       for row, cells in enumerate(variables)
                                       for col, variable in enumerate(cells)]
                        self.assertEqual(solver.solve(assumptions=assumptions),
                                         validate_columns(columns, n), columns)
                    self.assertFalse(solver.solve(assumptions=[-v for row in variables for v in row]))
                    if n > 1:
                        self.assertFalse(solver.solve(assumptions=[v for row in variables for v in row]))
                self.assertEqual(clauses, module.generate_clauses(n, variables))

    def test_commander_grouping_preserves_inputs_and_boundaries(self) -> None:
        from encoder.commander import grouping_variables

        for size in (0, 1, 23, 24, 25, 512):
            variables = list(range(1, size + 1))
            with self.subTest(size=size):
                groups = grouping_variables(variables, 24)
                self.assertEqual([v for group in groups for v in group], variables)
                self.assertTrue(all(0 < len(group) <= 24 for group in groups))
                self.assertTrue(all(len(group) == 24 for group in groups[:-1]))

    def test_commander_non_square_board_uses_multiple_groups(self) -> None:
        from encoder import commander

        clauses = []
        with patch.object(commander, "board_size", 512, create=True), \
                patch.object(commander, "new_variables_count", 0):
            commander.at_most_one(clauses, list(range(1, 513)))
            # 22 groups of at most 24 inputs, rather than one group of 512.
            self.assertEqual(commander.new_variables_count, 130)
            self.assertEqual(len(clauses), 3309)
        with Glucose3(bootstrap_with=clauses) as solver:
            self.assertTrue(solver.solve(assumptions=list(range(-512, 0))))
            for variable in range(1, 513):
                self.assertTrue(solver.solve(assumptions=[variable]), variable)
            for pair in ((1, 2), (1, 24), (24, 25), (25, 48), (48, 49), (504, 505), (1, 512)):
                self.assertFalse(solver.solve(assumptions=list(pair)), pair)

    def test_library_auxiliaries_reserve_sparse_board_ids(self) -> None:
        variables = [[100 + 3 * (row * 4 + col) for col in range(4)] for row in range(4)]
        for encoding in ("binary", "sequential"):
            module = importlib.import_module(f"encoder.{encoding}")
            clauses = module.generate_clauses(4, variables)
            with self.subTest(encoding=encoding), Glucose3(bootstrap_with=clauses) as solver:
                for columns in itertools.product(range(4), repeat=4):
                    assumptions = [variable if columns[row] == col else -variable
                                   for row, cells in enumerate(variables)
                                   for col, variable in enumerate(cells)]
                    self.assertEqual(solver.solve(assumptions=assumptions),
                                     validate_columns(columns, 4), columns)

    def test_summary_separates_outcomes_and_successful_timings(self) -> None:
        rows = [dict(solver="cadical", encoding="product", n=8, status=status,
                     solving_s=seconds, total_s=seconds + 1, peak_rss_mib=100 + seconds)
                for status, seconds in (("sat", 1), ("sat", 3), ("sat", 5), ("unsat", 7),
                                        ("timeout", 120), ("memory_limit", 50),
                                        ("error", 60), ("invalid", 70), ("unknown", 80))]
        summary, = summarize_results(rows, repeats=10)
        self.assertEqual(summary["attempts"], 9)
        self.assertEqual(summary["requested_attempts"], 10)
        self.assertEqual(summary["solved"], 4)
        self.assertAlmostEqual(summary["success_rate"], 4 / 9)
        self.assertEqual(summary["sat"], 3)
        for status in ("unsat", "timeout", "memory_limit", "error", "invalid", "unknown"):
            self.assertEqual(summary[status], 1)
        self.assertEqual(summary["solving_s_samples"], 4)
        self.assertEqual(summary["solving_s_median"], 4)
        self.assertEqual(summary["solving_s_iqr"], 3)
        self.assertEqual(summary["solving_s_min"], 1)
        self.assertEqual(summary["solving_s_max"], 7)
        self.assertEqual(summary["peak_rss_mib_max"], 220)
        self.assertEqual(summary["peak_rss_samples"], 9)

    def test_exports_leave_missing_metrics_blank(self) -> None:
        rows = [dict(solver="glucose", encoding="product", n=4, status="sat",
                     repetition=1, solving_s=0, total_s=0, wall_s=0.1),
                dict(solver="cadical", encoding="product", n=4, status="memory_limit")]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_results(output, {"repeats": 3}, rows)
            with (output / "summary.csv").open() as stream:
                solved, oom = list(csv.DictReader(stream))
            self.assertEqual(float(solved["solving_s_median"]), 0)
            self.assertEqual(solved["solving_s_iqr"], "")
            self.assertEqual(solved["generation_s_median"], "")
            self.assertEqual(solved["peak_rss_mib_max"], "")
            self.assertEqual(oom["solving_s_median"], "")
            self.assertEqual(oom["success_rate"], "0.0")
            with (output / "runs.csv").open() as stream:
                first, second = list(csv.DictReader(stream))
            self.assertEqual(first["clauses"], "")
            self.assertEqual(second["solving_s"], "")
            self.assertEqual(json.loads((output / "experiment.json").read_text())["completed_attempts"], 2)

    def test_repetitions_and_experiment_directories(self) -> None:
        calls = []

        def run(encoding, n, timeout, memory, solver, *, log_dir, repetition, formulation):
            self.assertTrue(log_dir.is_dir())
            calls.append((repetition, n, solver, encoding))
            return dict(solver=solver, encoding=encoding, n=n, repetition=repetition,
                        status="sat", solving_s=repetition, total_s=repetition, wall_s=repetition)

        with TemporaryDirectory() as directory:
            arguments = ["benchmark.py", "--sizes", "4", "8", "--solvers", "glucose", "cp_sat",
                         "--encodings", "product", "--repeats", "3", "--output", directory]
            for _ in range(2):
                with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                        patch("benchmark.run_case", side_effect=run):
                    main()
            experiments = list(Path(directory).iterdir())
            self.assertEqual(len(experiments), 2)
            expected = [(r, n, solver, "product" if solver == "glucose" else "native")
                        for r in (1, 2, 3) for n in (4, 8) for solver in ("glucose", "cp_sat")]
            self.assertEqual(calls, expected * 2)
            for output in experiments:
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertEqual(metadata["state"], "completed")
                self.assertEqual(metadata["planned_attempts"], 12)
                self.assertEqual(metadata["completed_attempts"], 12)
                self.assertEqual(metadata["solver_seeds"], {"glucose": None, "cp_sat": 0})
                with (output / "summary.csv").open() as stream:
                    summaries = list(csv.DictReader(stream))
                self.assertEqual(len(summaries), 4)
                self.assertTrue(all(row["attempts"] == "3" for row in summaries))
                self.assertTrue(all(row["solving_s_median"] == "2" for row in summaries))
                self.assertIn("3 / 3", (output / "report.md").read_text())

    def test_native_allocation_failures_are_not_generic_errors(self) -> None:
        for stderr, code, status in (("terminate: std::bad_alloc", -6, "memory_limit"),
                                     ("kissat: out-of-memory reallocating", -6, "memory_limit"),
                                     ("MemoryError", 1, "memory_limit"),
                                     ("cannot allocate memory for thread-local data: ABORT", 127, "memory_limit"),
                                     ("error while loading shared libraries: missing.so", 127, "error"),
                                     ("", -9, "error"), ("segmentation fault", -11, "error")):
            with self.subTest(stderr=stderr, code=code):
                process = Mock(returncode=code)
                process.communicate.return_value = ('{"phase":"loading"}\n', stderr)
                process.poll.return_value = code
                with patch("helper.subprocess.Popen", return_value=process):
                    row = run_case("product", 4, 10, 512)
                self.assertEqual(row["status"], status)
                self.assertEqual(row["phase"], "loading")
                self.assertEqual(row["returncode"], code)

    def test_loader_allocation_failure_preserves_phase_and_logs(self) -> None:
        popen = subprocess.Popen
        message = "cannot allocate memory for thread-local data: ABORT"
        script = (
            "import os\n"
            "print('{\"phase\":\"loading\",\"generation_s\":28.649,\"clauses\":12560392}', flush=True)\n"
            f"os.write(2, {message.encode()!r} + b'\\n')\n"
            "os._exit(127)\n"
        )
        with TemporaryDirectory() as directory:
            def launch(command, **kwargs):
                return popen([sys.executable, "-c", script], **kwargs)

            with patch("helper.subprocess.Popen", side_effect=launch):
                row = run_case("sequential", 1024, 10, 3072, "cadical", log_dir=Path(directory))
            self.assertEqual(row["status"], "memory_limit")
            self.assertEqual(row["phase"], "loading")
            self.assertEqual(row["returncode"], 127)
            self.assertEqual(row["clauses"], 12560392)
            self.assertEqual(row["generation_s"], 28.649)
            self.assertNotIn("solving_s", row)
            self.assertNotIn("signal", row)
            self.assertIn(message, row["error"])
            self.assertEqual((Path(row["log_dir"]) / "stderr.log").read_text(), message + "\n")
            self.assertIn(message, Path(row["log_file"]).read_text())
            summary, = summarize_results([row])
            self.assertEqual(summary["memory_limit"], 1)
            self.assertEqual(summary["error"], 0)
            self.assertEqual(summary["solving_s_samples"], 0)

    def test_native_oom_log_preserves_original_failure(self) -> None:
        popen = subprocess.Popen
        script = (
            "import faulthandler, os, resource\n"
            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
            "faulthandler.enable()\n"
            "print('{\"phase\":\"solving\"}', flush=True)\n"
            "os.write(2, b\"terminate called: std::bad_alloc\\n\")\n"
            "os.abort()\n"
        )
        with TemporaryDirectory() as directory:
            def launch(command, **kwargs):
                return popen([sys.executable, "-c", script], **kwargs)

            with patch("helper.subprocess.Popen", side_effect=launch):
                row = run_case("product", 4, 10, 512, log_dir=Path(directory), repetition=2)
            self.assertEqual(row["status"], "memory_limit")
            self.assertEqual(row["signal"], "SIGABRT")
            self.assertEqual(row["phase"], "solving")
            self.assertEqual(row["repetition"], 2)
            self.assertNotIn("\n", row["error"])
            log = Path(row["log_file"]).read_text()
            self.assertIn("std::bad_alloc", log)
            self.assertIn("SOLVER STDERR (raw)", log)
            self.assertIn("Fatal Python error: Aborted", log)

    def test_explicit_log_parent_can_equal_output_parent(self) -> None:
        with TemporaryDirectory() as directory:
            row = dict(solver="glucose", encoding="product", n=4,
                       status="sat", total_s=0.1, wall_s=0.2)
            arguments = ["benchmark.py", "--sizes", "4", "--solvers", "glucose",
                         "--encodings", "product", "--output", directory, "--log-dir", directory]
            with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                    patch("benchmark.run_case", return_value=row) as run:
                main()
            output, = Path(directory).iterdir()
            self.assertEqual(run.call_args.kwargs["log_dir"], output / "logs")
            self.assertTrue((output / "logs").is_dir())

    def test_cancel_kills_and_reaps_worker(self) -> None:
        process = Mock()
        process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
        process.poll.return_value = None
        with patch("helper.subprocess.Popen", return_value=process) as popen, \
                patch("helper.os.killpg") as kill:
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
                         "--output", directory]
            with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                    patch("benchmark.run_case", side_effect=[completed, KeyboardInterrupt]) as run:
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 130)
            self.assertEqual(run.call_count, 2)
            output = next(output.iterdir())
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(saved["results"], [completed])
            self.assertEqual(saved["metadata"]["stop_reason"], "Cancelled with Ctrl+C")
            self.assertEqual(saved["metadata"]["interrupted_cases"],
                             [{"solver": "glucose", "encoding": "binomial", "n": 8, "repetition": 1}])
            self.assertEqual(len((output / "runs.csv").read_text().splitlines()), 2)
            report = (output / "report.md").read_text()
            self.assertIn("Cancelled with Ctrl+C", report)
            self.assertIn("| 8 | stopped |", report)
            self.assertIn("| 16 | not run |", report)

    def test_cancel_during_worker_launch(self) -> None:
        process = Mock()
        process.poll.return_value = None
        with patch("helper.subprocess.Popen", return_value=process), \
                patch("helper.os.killpg") as kill, \
                patch("helper.signal.pthread_sigmask",
                      side_effect=[set(), KeyboardInterrupt, set()]):
            with self.assertRaises(KeyboardInterrupt):
                run_case("binomial", 4, 10, 3072)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        process.communicate.assert_called_once_with()

    def test_sequential_matrix_and_case_order(self) -> None:
        thread = get_ident()
        calls = []

        def run(encoding, n, timeout, memory, solver, **kwargs):
            self.assertEqual(get_ident(), thread)
            calls.append((n, solver, encoding))
            return {"solver": solver, "encoding": encoding, "n": n,
                    "status": "error" if solver == "gurobi" and n == 4 else "sat",
                    "total_s": 0.1, "wall_s": 0.2}

        with TemporaryDirectory() as directory:
            arguments = ["benchmark.py", "--sizes", "4", "8",
                         "--output", directory]
            with patch("benchmark.sys.argv", arguments), patch("builtins.print"), \
                    patch("benchmark.run_case", side_effect=run):
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 1)
            output = next(Path(directory).iterdir())
            saved = json.loads((output / "results.json").read_text())
            expected = [(n, solver, encoding) for n in (4, 8) for solver in SOLVERS
                        for encoding in (ENCODINGS if solver in SAT_SOLVERS else ["native"])]
            self.assertEqual(calls, expected)
            self.assertEqual([(row["n"], row["solver"], row["encoding"])
                              for row in saved["results"]], expected)
            self.assertNotIn("jobs", saved["metadata"])
            self.assertEqual(saved["metadata"]["solver_threads"], 1)
            self.assertEqual(saved["metadata"]["encodings"], list(ENCODINGS))
            self.assertEqual(saved["metadata"]["encoding_implementations"], ENCODING_IMPLEMENTATIONS)
            self.assertNotIn("--jobs", saved["metadata"]["command"])
            report = (output / "report.md").read_text()
            for solver in SOLVERS:
                self.assertIn(f"### {solver}", report)
            self.assertEqual(report.count("| N | native |"),
                             len(BACKENDS))
            self.assertIn("| 4 | error |", report)
            self.assertIn("run sequentially in fresh processes", report)
            self.assertNotIn("parallel", report)
            self.assertNotIn("--jobs", report)
            self.assertIn("timing variability cannot be estimated", report)

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
        with patch("helper.subprocess.Popen", return_value=process), \
                patch("helper.os.killpg") as kill:
            with self.assertRaises(CancelledError):
                run_case("binomial", 4, 10, 3072, cancel=cancel)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)

    def test_cancel_event_skips_worker_launch(self) -> None:
        cancel = Event()
        cancel.set()
        with patch("helper.subprocess.Popen") as popen:
            with self.assertRaises(CancelledError):
                run_case("binomial", 4, 10, 3072, cancel=cancel)
        popen.assert_not_called()

    def test_timeout_preserves_partial_result(self) -> None:
        process = Mock()
        process.poll.return_value = -9
        process.returncode = -9
        process.communicate.return_value = ('{"phase": "loading", "generation_s": 0.5, "peak_rss_mib": 10}\n', "")
        with patch("helper.subprocess.Popen", return_value=process), \
                patch("helper.os.killpg") as kill, \
                patch("helper.time.perf_counter", side_effect=[0, 20, 20]), \
                patch("helper.Path.read_text", return_value="VmHWM:\t2048000 kB"):
            result = run_case("binomial", 4, 10, 3072, "cadical")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["solver"], "cadical")
        self.assertEqual(result["generation_s"], 0.5)
        self.assertEqual(result["phase"], "loading")
        self.assertEqual(result["peak_rss_mib"], 2000)
        kill.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_diagnostic_logs_and_unique_case_directories(self) -> None:
        with TemporaryDirectory() as directory:
            rows = [run_case("product", 4, 10, 512, log_dir=Path(directory)) for _ in range(2)]
            self.assertNotEqual(rows[0]["log_dir"], rows[1]["log_dir"])
            for row in rows:
                self.assertEqual(row["status"], "sat")
                self.assertEqual(row["returncode"], 0)
                logs = Path(row["log_dir"])
                output = [json.loads(line) for line in (logs / "stdout.jsonl").read_text().splitlines()]
                self.assertEqual([entry["phase"] for entry in output],
                                 ["generation", "loading", "solving", "complete", "complete"])
                self.assertTrue(all("timestamp_utc" in entry and "peak_rss_mib" in entry
                                    for entry in output))
                events = [json.loads(line) for line in (logs / "state.jsonl").read_text().splitlines()]
                self.assertEqual(events[0]["event"], "start")
                self.assertEqual(events[-1]["event"], "finished")
                self.assertEqual(events[-1]["phase"], "complete")
                self.assertEqual(events[-1]["status"], "sat")
                self.assertTrue(any(event["event"] == "memory" for event in events))
                self.assertTrue((logs / "stderr.log").exists())
                readable = Path(row["log_file"]).read_text()
                self.assertIn("PROGRESS", readable)
                self.assertIn("GENERATION", readable)
                self.assertIn("SOLVING", readable)
                self.assertIn("Validation        : passed", readable)
                self.assertEqual(row["repetition"], 1)
                self.assertEqual(row["encoding_implementation"], "custom/product")
                self.assertIn("Encoder source    : custom/product", readable)

    def test_diagnostics_preserve_native_crash_and_timeout(self) -> None:
        popen = subprocess.Popen
        for crash in (True, False):
            with self.subTest(crash=crash), TemporaryDirectory() as directory:
                script = (
                    "import faulthandler, os, resource, signal, time\n"
                    "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
                    "faulthandler.enable()\n"
                    "faulthandler.register(signal.SIGUSR1, all_threads=True)\n"
                    "print('{\"phase\": \"solving\", \"clauses\": 123}', flush=True)\n"
                    + ("os.kill(os.getpid(), signal.SIGSEGV)\n" if crash else "time.sleep(60)\n")
                )
                def launch(command, **kwargs):
                    return popen([sys.executable, "-c", script], **kwargs)

                with patch("helper.subprocess.Popen", side_effect=launch):
                    row = run_case("product", 256, 2, 512, log_dir=Path(directory))
                logs = Path(row["log_dir"])
                self.assertEqual(row["phase"], "solving")
                self.assertEqual(row["clauses"], 123)
                self.assertEqual(row["status"], "error" if crash else "timeout")
                self.assertEqual(row["signal"], "SIGSEGV" if crash else "SIGKILL")
                self.assertIn("<string>", (logs / "stderr.log").read_text())
                self.assertIn("<string>", Path(row["log_file"]).read_text())
                self.assertIn(row["status"], Path(row["log_file"]).read_text())
                self.assertIn('"phase": "solving"', (logs / "stdout.jsonl").read_text())
                events = [json.loads(line) for line in (logs / "state.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["signal"], row["signal"])

    def test_diagnostics_preserve_python_traceback(self) -> None:
        with TemporaryDirectory() as directory:
            row = run_case("missing_encoder", 256, 10, 512, log_dir=Path(directory))
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["phase"], "generation")
            self.assertIn("ModuleNotFoundError", row["error"])
            self.assertIn("Traceback", (Path(row["log_dir"]) / "stderr.log").read_text())

    def test_invalid_cli_limits(self) -> None:
        for arguments in (["--jobs", "1"], ["--jobs", "2"], ["--sizes", "1"],
                          ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--memory-mib", "0"], ["--repeats", "0"], ["--repeats", "-1"]):
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
