"""Shared N-Queens solver and benchmark helpers."""

import csv
import faulthandler
import importlib
import json
import math
import os
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import CancelledError
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from threading import Event
from typing import Any, Iterator, Sequence, TypedDict


class NativeResult(TypedDict):
    """Type annotations for the existing native result dictionary."""

    status: str
    columns: list[int] | None
    backend_status: str
    version: str
    variables: int
    constraints: int
    building_s: float
    solving_s: float
    total_s: float


ENCODINGS: tuple[str, ...] = ("binomial", "binary", "commander", "product", "sequential")
SAT_SOLVERS = {
    "glucose": ("Glucose", "Glucose3"),
    "cadical": ("CaDiCaL", "Cadical300"),
    "kissat": ("Kissat", "Kissat404"),
    "cryptominisat": ("CryptoMiniSat", "CryptoMinisat"),
}
NATIVE_MODULES = {"cplex_cp": "CPOptimizer", "cplex_mip": "CPLEX",
                  "gurobi": "Gurobi", "cp_sat": "CPSAT"}
BACKENDS = tuple(NATIVE_MODULES)
SOLVERS = (*SAT_SOLVERS, *BACKENDS)
FIELDS: tuple[str, ...] = (
    "solver", "encoding", "n", "status", "phase", "generation_s", "loading_s",
    "solving_s", "total_s", "wall_s", "clauses", "variables", "peak_rss_mib",
    "constraints", "version", "backend_status", "conflicts", "decisions", "propagations", "error",
    "returncode", "signal", "log_dir",
)


def validate_columns(columns: Sequence[int], n: int) -> bool:
    """Check queen count, columns, and both diagonal directions independently."""
    return (
        len(columns) == n
        and all(0 <= col < n for col in columns)
        and len(set(columns)) == n
        and len({row - col for row, col in enumerate(columns)}) == n
        and len({row + col for row, col in enumerate(columns)}) == n
    )


def board_lines(n: int) -> Iterator[tuple[list[int], str]]:
    """Yield flattened row/column equalities and nontrivial diagonal inequalities."""
    for row in range(n):
        yield [row * n + col for col in range(n)], "E"
    for col in range(n):
        yield [row * n + col for row in range(n)], "E"
    for direction in (-1, 1):
        diagonals: dict[int, list[int]] = {}
        for row in range(n):
            for col in range(n):
                diagonals.setdefault(row + direction * col, []).append(row * n + col)
        for indices in diagonals.values():
            if len(indices) > 1:
                yield indices, "L"


def _result(started: float, loaded: float, columns: list[int] | None, status: str,
            detail: str, version: str, variables: int, constraints: int) -> NativeResult:
    finished = time.perf_counter()
    return dict(status=status, columns=columns, backend_status=detail,
                version=version, variables=variables, constraints=constraints,
                building_s=loaded - started, solving_s=finished - loaded,
                total_s=finished - started)


def _binary_columns(values: Sequence[float], n: int) -> list[int]:
    columns: list[int] = []
    for row in range(n):
        selected = [col for col in range(n) if values[row * n + col] > 0.5]
        if len(selected) != 1:
            raise RuntimeError("MIP solution does not have exactly one queen per row")
        columns.append(selected[0])
    return columns


def solve_n_queens(n: int, backend: str, time_limit: float = 30, threads: int = 1) -> NativeResult:
    """Return status, zero-based queen columns, model size, and wall-clock timings.

    A time limit without a solution is distinct from proven infeasibility.
    Missing installations and license errors propagate to the caller.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if backend not in NATIVE_MODULES:
        raise ValueError(f"Unknown backend: {backend}")
    name = NATIVE_MODULES[backend]
    module = importlib.import_module(f"solver.{name}.{name}")
    result: NativeResult = module._solve_n_queens(n, time_limit, threads)
    if result["columns"] is not None and not validate_columns(result["columns"], n):
        raise RuntimeError(f"{backend} returned an invalid N-Queens solution")
    return result


def emit(data: dict[str, Any]) -> None:
    print(json.dumps({**data, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "pid": os.getpid(), "peak_rss_mib":
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)


def worker(encoding: str, n: int, memory_mib: int, solver_name: str = "glucose",
           timeout_s: float = 120) -> None:
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    limit = memory_mib * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result: dict[str, Any] = {"solver": solver_name, "encoding": encoding,
                              "n": n, "phase": "generation"}
    emit(result)
    try:
        if solver_name in BACKENDS:
            native_result = solve_n_queens(n, solver_name, time_limit=timeout_s, threads=1)
            result.update(native_result, generation_s=native_result["building_s"],
                          loading_s=0, phase="complete")
            if native_result["columns"] is not None:
                result["queen_columns_1based"] = [col + 1 for col in native_result["columns"]]
            if result["status"] == "unsat" and n not in (2, 3):
                result["status"] = "invalid"
        else:
            sat_worker(encoding, n, solver_name, result)
    except MemoryError:
        traceback.print_exc()
        result.update(status="memory_limit", error="Address-space memory cap reached")
    except Exception as exc:
        traceback.print_exc()
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    # shortcut: CP Optimizer child RSS is excluded, include it when comparing native memory.
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    emit(result)


def sat_worker(encoding: str, n: int, solver_name: str, result: dict[str, Any]) -> None:
    module = importlib.import_module(f"encoder.{encoding}")
    adapter_name, backend_name = SAT_SOLVERS[solver_name]
    adapter = getattr(importlib.import_module(f"solver.pySAT.{adapter_name}"), adapter_name)
    backend = getattr(importlib.import_module("pysat.solvers"), backend_name)
    started = time.perf_counter()
    variables = module.generate_variables(n)
    clauses = module.generate_clauses(n, variables)
    result.update(generation_s=time.perf_counter() - started,
                  clauses=len(clauses), phase="loading")
    emit(result)
    loading = time.perf_counter()
    with backend() as native:
        solver = adapter(native)
        for clause in clauses:
            solver.add_clause(clause)
        result.update(loading_s=time.perf_counter() - loading,
                      variables=native.nof_vars(), phase="solving")
        emit(result)
        solving = time.perf_counter()
        model = solver.solve()
        result.update(solving_s=time.perf_counter() - solving,
                      total_s=time.perf_counter() - started)
        try:
            result.update(native.accum_stats() or {})
        except NotImplementedError:
            pass  # Kissat and CryptoMiniSat do not expose statistics through PySAT.
        if model is not None:
            queens = [lit - 1 for lit in model if 0 < lit <= n * n]
            rows = [position // n for position in queens]
            columns = [position % n for position in sorted(queens)]
            valid = len(set(rows)) == n and validate_columns(columns, n)
            result.update(status="sat" if valid else "invalid",
                          queen_columns_1based=[col + 1 for col in columns])
        else:
            result["status"] = "unsat" if n in (2, 3) else "invalid"
        result["phase"] = "complete"
        emit(result)  # Preserve the validated result if native solver teardown crashes.


def run_case(encoding: str, n: int, timeout_s: float, memory_mib: int,
             solver_name: str = "glucose", cancel: Event | None = None,
             log_dir: Path | None = None) -> dict[str, Any]:
    if cancel is not None and cancel.is_set():
        raise CancelledError
    command = [sys.executable, str(Path(__file__).resolve().with_name("benchmark.py")), "--worker",
               encoding, str(n), str(memory_mib), solver_name, str(timeout_s)]
    started = time.perf_counter()
    result: dict[str, Any] = {"solver": solver_name, "encoding": encoding, "n": n}
    timed_out = False
    process = None
    logs = ExitStack()
    state_log = None
    case_dir = None

    def log_event(event: str, **data: Any) -> None:
        if state_log is not None:
            print(json.dumps({**data, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                              "event": event}), file=state_log, flush=True)

    # Defer Ctrl+C until the worker exists and can be cleaned up.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        stdout = stderr = subprocess.PIPE
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            case_dir = Path(mkdtemp(prefix=f"{solver_name}-{encoding}-n{n}-", dir=log_dir))
            stdout = logs.enter_context((case_dir / "stdout.jsonl").open("w"))
            stderr = logs.enter_context((case_dir / "stderr.log").open("w"))
            state_log = logs.enter_context((case_dir / "state.jsonl").open("w"))
            result["log_dir"] = str(case_dir.resolve())
            log_event("start", **result, command=command, timeout_s=timeout_s,
                      memory_mib=memory_mib, python=sys.version, platform=platform.platform())
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   text=True, start_new_session=True,
                                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
                                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        log_event("worker_started", pid=process.pid)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        next_sample = started
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise CancelledError
                remaining = timeout_s - (time.perf_counter() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                if state_log is not None and time.perf_counter() >= next_sample:
                    try:
                        snapshot = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                    for line in Path(f"/proc/{process.pid}/status").read_text().splitlines()
                                    if line.startswith(("State:", "VmRSS:", "VmHWM:", "VmSize:", "Threads:"))}
                        snapshot.update({"host_" + line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                         for line in Path("/proc/meminfo").read_text().splitlines()
                                         if line.startswith(("MemAvailable:", "SwapFree:"))})
                        if "VmHWM" in snapshot:
                            result["peak_rss_mib"] = max(result.get("peak_rss_mib", 0),
                                                         int(snapshot["VmHWM"].split()[0]) / 1024)
                        log_event("memory", pid=process.pid, wall_s=time.perf_counter() - started,
                                  **snapshot)
                    except FileNotFoundError:
                        pass
                    next_sample = time.perf_counter() + 1
                try:
                    output, errors = process.communicate(timeout=min(remaining, 0.1))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            timed_out = True
            log_event("timeout", pid=process.pid)
            try:
                for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        result["peak_rss_mib"] = int(line.split()[1]) / 1024
            except FileNotFoundError:
                pass
            if case_dir is not None:
                try:
                    os.kill(process.pid, signal.SIGUSR1)
                    process.communicate(timeout=0.2)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, errors = process.communicate()
    except BaseException:
        log_event("interrupted", traceback=traceback.format_exc())
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        logs.close()
    if case_dir is not None:
        output = (case_dir / "stdout.jsonl").read_text(errors="replace")
        errors = (case_dir / "stderr.log").read_text(errors="replace")
    for line in output.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # Native backends can print license or startup messages.
        if isinstance(data, dict):
            if "peak_rss_mib" in result and "peak_rss_mib" in data:
                data["peak_rss_mib"] = max(result["peak_rss_mib"], data["peak_rss_mib"])
            result.update(data)
    result["wall_s"] = time.perf_counter() - started
    result["returncode"] = process.returncode
    if process.returncode < 0:
        result["signal"] = signal.Signals(-process.returncode).name
    if timed_out:
        result.update(status="timeout", error=f"Exceeded {timeout_s:g} s per-case wall limit")
    elif process.returncode:
        result.update(status="error", error=f"Exit {process.returncode}: {errors.strip()}")
    elif "status" not in result:
        result.update(status="error", error="Worker exited without a result")
    if case_dir is not None:
        with (case_dir / "state.jsonl").open("a") as state_log:
            log_event("finished", **result)
    return result


def save_results(directory: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def write_report(directory: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        "# N-Queens solver benchmark", "",
        f"Run: {metadata['started_utc']}. Python {metadata['python']}; "
        f"python-sat {metadata['python_sat']}; {metadata['cpu']}.", "",
        f"For each N, all selected solver cases run in fresh processes with up to "
        f"{metadata['jobs']} parallel jobs. The next N starts after all cases finish. "
        "Each solver uses one thread. Concurrent jobs compete for CPU and memory, "
        "so these timings measure parallel throughput rather than isolated solver speed. "
        "Timings are wall-clock seconds; there are no warmups or statistical repeats. "
        "Generation includes board variables and the full Python clause list. "
        "Loading uses the original clause-by-clause insertion. SAT total covers generation, "
        "loading and solve (including model extraction), excluding imports, validation, and teardown. "
        "Native backends run once per N using their direct models; generation is model "
        "construction and loading is zero. Native total follows the existing backend "
        "timings; Gurobi excludes license startup. Wall includes the entire worker process. "
        "Peak RSS measures the Python worker on Linux and excludes CP Optimizer child memory.", "",
        f"Limits: {metadata['timeout_s']:g} seconds wall time and "
        f"{metadata['memory_mib']} MiB virtual address space per case. "
        "A timeout or memory-limit result is an incomplete attempt, not an UNSAT result. "
        "Partial timings cover only completed phases. No constructive solution, "
        "symmetry constraints, solver tuning, or replacement encoding was used.", "",
        "All SAT results are independently checked for exactly one queen per row and "
        "column and no shared diagonals. Their 1-based queen columns are saved in results.json.", "",
        "The original encodings are retained, including their existing grouping and "
        "product matrix choices. The modules were made safe to import, board size and "
        "auxiliary-variable state reset for each generation, and solvers closed explicitly.", "",
    ]
    encodings = metadata["encodings"]
    if metadata.get("stop_reason"):
        lines += [f"Stopped: {metadata['stop_reason']}. Remaining cases were not run.", ""]
    lines += ["## Comparison", "", "Cells show total seconds for validated SAT "
              "results, or the stopping condition.", ""]
    indexed = {(row["n"], row["solver"], row["encoding"]): row for row in results}
    for solver in metadata["solvers"]:
        columns = encodings if solver in SAT_SOLVERS else ["native"]
        lines += [f"### {solver}", "", "| N | " + " | ".join(columns) + " |",
                  "|---:|" + "---:|" * len(columns)]
        for n in metadata["sizes"]:
            cells = []
            for encoding in columns:
                row = indexed.get((n, solver, encoding))
                missing = "not run" if metadata.get("stop_reason") else "pending"
                if {"n": n, "solver": solver, "encoding": encoding} in metadata.get("interrupted_cases", []):
                    missing = "stopped"
                cells.append(missing if row is None else
                             f"{row['total_s']:.4f}" if row["status"] == "sat" else
                             row["status"].replace("_", " "))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["", "## Detailed measurements", "",
              "| Solver | Encoding | N | Status | Generation (s) | Loading (s) | Solve (s) | Total (s) | Peak RSS (MiB) | Clauses | Constraints |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        def timing(key: str, measurement: dict[str, Any] = row) -> str:
            return f"{measurement[key]:.4f}" if key in measurement else "—"
        lines.append(f"| {row['solver']} | {row['encoding']} | {row['n']} | {row['status']} | "
                     f"{timing('generation_s')} | {timing('loading_s')} | "
                     f"{timing('solving_s')} | {timing('total_s')} | "
                     f"{timing('peak_rss_mib')} | {row.get('clauses', '—')} | "
                     f"{row.get('constraints', '—')} |")
    lines += ["", "Reproduce:", "", "```sh",
              shlex.join(metadata["command"]), "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
