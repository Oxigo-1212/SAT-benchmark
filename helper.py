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
import statistics
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


REPOSITORY_URL = "https://github.com/Oxigo-1212/SAT-benchmark"
ENCODINGS: tuple[str, ...] = ("binomial", "binary", "commander", "product", "sequential")
ENCODING_IMPLEMENTATIONS = {
    "binomial": "pysat.card.CardEnc/pairwise",
    "binary": "pysat.card.CardEnc/bitwise",
    "commander": "custom/commander",
    "product": "custom/product",
    "sequential": "pysat.card.CardEnc/seqcounter",
}
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
SOLVER_SEEDS = {name: 0 if name in ("cadical", *BACKENDS) else None for name in SOLVERS}
FIELDS: tuple[str, ...] = (
    "solver", "encoding", "formulation", "encoding_implementation", "n", "repetition", "seed", "status", "phase",
    "generation_s", "loading_s",
    "solving_s", "total_s", "wall_s", "clauses", "variables", "peak_rss_mib",
    "constraints", "version", "backend_status", "conflicts", "decisions", "propagations", "error",
    "returncode", "signal", "log_dir", "log_file",
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


def board_lines(n: int, formulation: str = "extended") -> Iterator[tuple[list[int], str]]:
    """Yield flattened row/column equalities and nontrivial diagonal inequalities."""
    if formulation not in ("minimal", "extended"):
        raise ValueError(f"Unknown formulation: {formulation}")
    for row in range(n):
        yield [row * n + col for col in range(n)], "E"
    for col in range(n):
        yield [row * n + col for row in range(n)], "E" if formulation == "extended" else "L"
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


def solve_n_queens(n: int, backend: str, time_limit: float = 30, threads: int = 1,
                   *, on_loaded=None) -> NativeResult:
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
    result: NativeResult = module._solve_n_queens(
        n, time_limit, threads, **({"on_loaded": on_loaded} if on_loaded is not None else {}))
    if result["columns"] is not None and not validate_columns(result["columns"], n):
        raise RuntimeError(f"{backend} returned an invalid N-Queens solution")
    return result


def emit(data: dict[str, Any]) -> None:
    print(json.dumps({**data, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "pid": os.getpid(), "peak_rss_mib":
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)


def worker(encoding: str, n: int, memory_mib: int, solver_name: str = "glucose",
           timeout_s: float = 120, formulation: str = "extended") -> None:
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    limit = memory_mib * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result: dict[str, Any] = {"solver": solver_name, "encoding": encoding,
                              "formulation": formulation if solver_name in SAT_SOLVERS else "native",
                              "n": n, "seed": SOLVER_SEEDS.get(solver_name),
                              "phase": "generation", "encoding_implementation":
                              ENCODING_IMPLEMENTATIONS.get(encoding, "unknown")
                              if solver_name in SAT_SOLVERS else "native"}
    emit(result)
    try:
        if solver_name in BACKENDS:
            def loaded(building_s: float) -> None:
                result.update(generation_s=building_s, loading_s=0, phase="solving")
                emit(result)

            native_result = solve_n_queens(n, solver_name, time_limit=timeout_s,
                                           threads=1, on_loaded=loaded)
            result.update(native_result, generation_s=native_result["building_s"],
                          loading_s=0, phase="complete")
            if native_result["columns"] is not None:
                result["queen_columns_1based"] = [col + 1 for col in native_result["columns"]]
            if result["status"] == "unsat" and n not in (2, 3):
                result["status"] = "invalid"
        else:
            sat_worker(encoding, n, solver_name, result)
    except MemoryError:
        os.write(2, b"MemoryError: address-space allocation failed\n")
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
    clauses = module.generate_clauses(n, variables, formulation=result.get("formulation", "extended"))
    result.update(generation_s=time.perf_counter() - started,
                  clauses=len(clauses), phase="loading")
    emit(result)
    loading = time.perf_counter()
    with backend() as native:
        if solver_name == "cadical":
            native.configure({"seed": SOLVER_SEEDS[solver_name]})
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
             log_dir: Path | None = None, repetition: int = 1,
             formulation: str = "extended") -> dict[str, Any]:
    if formulation not in ("minimal", "extended"):
        raise ValueError(f"Unknown formulation: {formulation}")
    if cancel is not None and cancel.is_set():
        raise CancelledError
    command = [sys.executable, str(Path(__file__).resolve().with_name("benchmark.py")), "--worker",
               encoding, str(n), str(memory_mib), solver_name, str(timeout_s), formulation]
    started = time.perf_counter()
    result: dict[str, Any] = {"solver": solver_name, "encoding": encoding, "n": n,
                              "formulation": formulation if solver_name in SAT_SOLVERS else "native",
                              "repetition": repetition, "seed": SOLVER_SEEDS.get(solver_name),
                              "phase": "startup",
                              "encoding_implementation": ENCODING_IMPLEMENTATIONS.get(encoding, "unknown")
                              if solver_name in SAT_SOLVERS else "native"}
    timed_out = False
    process = None
    logs = ExitStack()
    state_log = None
    case_dir = None
    readable_log = checkpoint_stream = None
    previous_phase = None

    def checkpoint(data: dict[str, Any]) -> None:
        nonlocal previous_phase
        if "peak_rss_mib" in result and "peak_rss_mib" in data:
            data["peak_rss_mib"] = max(result["peak_rss_mib"], data["peak_rss_mib"])
        result.update(data)
        phase = data.get("phase")
        if readable_log is not None and phase is not None and phase != previous_phase:
            details = " | ".join(f"{key}={data[key]}" for key in
                                 ("status", "clauses", "variables") if key in data)
            print(f"  {data.get('timestamp_utc', '')}  {phase.upper()}"
                  f"{(' | ' + details) if details else ''}", file=readable_log, flush=True)
            previous_phase = phase

    def read_checkpoints() -> None:
        if checkpoint_stream is None:
            return
        while True:
            position = checkpoint_stream.tell()
            line = checkpoint_stream.readline()
            if not line.endswith("\n"):
                checkpoint_stream.seek(position)
                return
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                checkpoint(data)

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
            case_dir = Path(mkdtemp(prefix=f"{solver_name}-{encoding}-n{n}-r{repetition:02d}-", dir=log_dir))
            stdout = logs.enter_context((case_dir / "stdout.jsonl").open("w"))
            stderr = logs.enter_context((case_dir / "stderr.log").open("w"))
            checkpoint_stream = logs.enter_context((case_dir / "stdout.jsonl").open(errors="replace"))
            state_log = logs.enter_context((case_dir / "state.jsonl").open("w"))
            readable_log = logs.enter_context((case_dir / "run.log").open("w"))
            result["log_dir"] = str(case_dir.resolve())
            result["log_file"] = str((case_dir / "run.log").resolve())
            print(f"RUN: {solver_name}_{encoding}_n{n}_r{repetition:02d}\n"
                  + "=" * 72 + "\nINPUT\n"
                  f"  Solver / encoding : {solver_name} / {encoding}\n"
                  f"  Formulation       : {result['formulation']}\n"
                  f"  Encoder source    : {result['encoding_implementation']}\n"
                  f"  Board / repetition: {n} / {repetition}\n"
                  f"  Started UTC       : {datetime.now(timezone.utc).isoformat()}\n"
                  f"  Worker wall limit : {timeout_s:g} s (all phases)\n"
                  f"  Address-space cap : {memory_mib} MiB (RLIMIT_AS)\n"
                  "  Solver threads    : 1\n"
                  f"  Seed              : {result['seed'] if result['seed'] is not None else 'backend default; interface has no seed control'}\n"
                  f"  Command           : {shlex.join(command)}\n\nPROGRESS",
                  file=readable_log, flush=True)
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
                read_checkpoints()
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
        if readable_log is not None:
            print("\nRESULT\n  Status: interrupted; worker cleanup requested",
                  file=readable_log, flush=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        read_checkpoints()
        logs.close()
    if case_dir is not None:
        output = (case_dir / "stdout.jsonl").read_text(errors="replace")
        errors = (case_dir / "stderr.log").read_text(errors="replace")
    for line in (output.splitlines() if case_dir is None else []):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # Native backends can print license or startup messages.
        if isinstance(data, dict):
            checkpoint(data)
    result["wall_s"] = time.perf_counter() - started
    result["returncode"] = process.returncode
    if process.returncode < 0:
        result["signal"] = signal.Signals(-process.returncode).name
    if timed_out:
        result.update(status="timeout", error=f"Exceeded {timeout_s:g} s per-case wall limit")
    elif process.returncode:
        oom = any(marker in errors.lower() for marker in
                  ("std::bad_alloc", "memoryerror", "out-of-memory", "out of memory",
                   "cannot allocate memory"))
        detail = "Allocation failure" if oom else "Worker failure"
        first_line = next((line.strip() for line in errors.splitlines() if line.strip()), "")
        result.update(status="memory_limit" if oom else "error",
                      error=f"{detail}; exit {process.returncode}: {first_line[:500]}")
    elif "status" not in result:
        result.update(status="error", error="Worker exited without a result")
    if case_dir is not None:
        with (case_dir / "state.jsonl").open("a") as state_log:
            log_event("finished", **result)
        with (case_dir / "run.log").open("a") as stream:
            stream.write("\nRESULT\n")
            for key in ("status", "phase", "version", "generation_s", "loading_s", "solving_s",
                        "total_s", "wall_s", "peak_rss_mib", "clauses", "variables", "constraints",
                        "conflicts", "decisions", "propagations", "returncode", "signal", "error"):
                stream.write(f"  {key:18s}: {result.get(key, 'unavailable')}\n")
            stream.write("  RSS scope         : Python worker only; excludes native child processes\n")
            stream.write("  Validation        : " + ("passed" if result.get("status") == "sat"
                         else "failed" if result.get("status") == "invalid" else "no SAT result") + "\n")
            stream.write("\nSOLVER STDOUT (raw)\n" + "-" * 72 + "\n" + output)
            stream.write("\nSOLVER STDERR (raw)\n" + "-" * 72 + "\n" + errors)
    return result


SUCCESS_STATUSES = ("sat", "unsat")
OUTCOMES = ("sat", "unsat", "timeout", "memory_limit", "error", "invalid", "unknown")
TIMINGS = ("generation_s", "loading_s", "solving_s", "total_s", "wall_s")
SUMMARY_FIELDS = (
    "solver", "encoding", "formulation", "n", "requested_attempts", "attempts", "solved", *OUTCOMES,
    "success_rate", "peak_rss_samples", "peak_rss_mib_max",
    *(f"{key}_{stat}" for key in TIMINGS
      for stat in ("samples", "median", "q1", "q3", "iqr", "min", "max")),
)


def summarize_results(results: list[dict[str, Any]], repeats: int = 1) -> list[dict[str, Any]]:
    """Summarize completed attempts; successful timings never include timeouts or errors."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in results:
        formulation = row.get("formulation", "native" if row["encoding"] == "native" else "extended")
        grouped.setdefault((row["solver"], row["encoding"], formulation, row["n"]), []).append(row)
    summaries = []
    for (solver, encoding, formulation, n), rows in grouped.items():
        solved = [row for row in rows if row["status"] in SUCCESS_STATUSES]
        summary = dict(solver=solver, encoding=encoding, formulation=formulation, n=n,
                       requested_attempts=repeats, attempts=len(rows), solved=len(solved),
                       success_rate=len(solved) / len(rows))
        summary.update({status: sum(row["status"] == status for row in rows) for status in OUTCOMES})
        summary["unknown"] += sum(row["status"] not in OUTCOMES for row in rows)
        peaks = [row["peak_rss_mib"] for row in rows if row.get("peak_rss_mib") is not None]
        summary["peak_rss_samples"] = len(peaks)
        if peaks:
            summary["peak_rss_mib_max"] = max(peaks)
        for key in TIMINGS:
            values = sorted(row[key] for row in solved if row.get(key) is not None)
            summary[f"{key}_samples"] = len(values)
            if values:
                summary.update({f"{key}_median": statistics.median(values),
                                f"{key}_min": min(values), f"{key}_max": max(values)})
            if len(values) > 1:
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                summary.update({f"{key}_q1": q1, f"{key}_q3": q3, f"{key}_iqr": q3 - q1})
        summaries.append(summary)
    return summaries


def save_results(directory: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata["completed_attempts"] = len(results)
    (directory / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Retain full solutions and worker checkpoints outside the tabular exports.
    (directory / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")
    for filename, fields, rows in (
        ("runs.csv", FIELDS, results),
        ("summary.csv", SUMMARY_FIELDS, summarize_results(results, metadata.get("repeats", 1))),
    ):
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def write_report(directory: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    def measurement(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return f"{value:.4f}" if value is not None else "—"

    summaries = summarize_results(results, metadata.get("repeats", 1))
    lines = [
        "# N-Queens solver benchmark", "",
        f"Source: [SAT-benchmark on GitHub]({REPOSITORY_URL}).", "",
        f"Experiment: {metadata['started_utc']} | State: {metadata.get('state', 'running')}.",
        f"Python {metadata['python']}; python-sat {metadata['python_sat']}; {metadata['cpu']}.",
        f"Code revision: {metadata.get('git_revision') or 'unavailable'}; "
        f"tracked changes: {metadata.get('tracked_changes', 'unavailable')}.", "",
        "All selected solver cases run sequentially in fresh processes. Each case finishes "
        "before the next starts, and each solver uses one thread.",
        f"Requested repetitions: {metadata.get('repeats', 1)} per case; "
        f"completed attempts: {len(results)}/{metadata.get('planned_attempts', 'unavailable')}. "
        "There are no warmups. Execution order is repetition, size, solver, encoding.",
        f"SAT formulation: {metadata.get('formulation', 'extended')}; native models are unchanged.",
        f"Seeds: {metadata.get('seed_policy', 'backend defaults; not explicitly configured')}.", "",
        "Encoding implementations: " + ("; ".join(
            f"{name}: {implementation}" for name, implementation
            in metadata.get("encoding_implementations", {}).items())
            or "not applicable or unrecorded") + ".", "",
        f"Limits: {metadata['timeout_s']:g} seconds for the entire worker, including imports, "
        f"generation, loading, solving, validation and teardown; {metadata['memory_mib']} MiB "
        "virtual address space (RLIMIT_AS). Allocation failures are OOM outcomes; their "
        "messages alone do not distinguish the address-space cap from system exhaustion. "
        "Unexplained signals remain errors.", "",
        "Timings are wall-clock seconds. SAT generation builds the full Python clause list; "
        "loading inserts clauses; solving includes model extraction. SAT total excludes imports, "
        "validation and teardown. Native generation is model construction, loading is zero, "
        "and total follows the backend timings (Gurobi excludes license startup). "
        "Wall time covers the entire worker. A native phase checkpoint marks the start of solving.",
        "Peak RSS measures the Python worker on Linux, excludes native child processes "
        "such as CP Optimizer, and can miss peaks on abrupt crashes between samples.", "",
        "SAT results are independently validated for one queen per row and column and no "
        "shared diagonals. Solutions are retained in results.json. Missing measurements are "
        "blank in CSV and shown as — here. Solver search counters are available only where "
        "the backend exposes them. SAT encodings and native CP/MIP models are different formulations.", "",
    ]
    if "source_snapshot" in metadata:
        lines += [
            "Reproduction files: [source snapshot](source/), "
            "[installed packages](installed-packages.txt), and "
            "[procedure](source/REPRODUCIBILITY.md). See experiment.json for per-solver seeds "
            "and the source-file checksum manifest.", "",
        ]
    if metadata.get("repeats", 1) < 2:
        lines += ["Only one attempt was requested per case; timing variability cannot be estimated.", ""]
    lines += [
        "## Result summary", "",
        "SAT/UNSAT/TO/OOM/ERR/INVALID/UNKNOWN counts include every finished attempt. "
        "Success is validated SAT or expected UNSAT divided by finished attempts; pending or "
        "interrupted attempts are excluded. Timing statistics use successful attempts only "
        "and are conditional on success. Timeouts are never treated as solved times. "
        "IQR = Q3 − Q1 using inclusive quartiles; unavailable with fewer than two timing samples. "
        "Repeated fixed settings measure timing variability and do not establish independent "
        "randomized search trials.", "",
        "| Solver | Encoding | N | Attempts / requested | SAT / UNSAT / TO / OOM / ERR / INVALID / UNKNOWN | Success | Solve samples | Solve median [IQR] (s) | Solve min–max (s) | Max worker RSS (MiB) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        outcomes = " / ".join(str(row[status]) for status in OUTCOMES)
        lines.append(f"| {row['solver']} | {row['encoding']} | {row['n']} | "
                     f"{row['attempts']} / {row['requested_attempts']} | {outcomes} | "
                     f"{row['success_rate']:.1%} | {row['solving_s_samples']} | "
                     f"{measurement(row, 'solving_s_median')} [{measurement(row, 'solving_s_iqr')}] | "
                     f"{measurement(row, 'solving_s_min')}–{measurement(row, 'solving_s_max')} | "
                     f"{measurement(row, 'peak_rss_mib_max')} |")
    if metadata.get("stop_reason"):
        lines += ["", f"Stopped: {metadata['stop_reason']}. Remaining attempts were not run."]
    lines += ["", "## Comparison", "", "Cells show median total seconds for successful attempts "
              "and solved/attempted counts when repeated or mixed; otherwise the stopping condition.", ""]
    indexed = {(row["n"], row["solver"], row["encoding"]): row for row in summaries}
    for solver in metadata["solvers"]:
        columns = metadata["encodings"] if solver in SAT_SOLVERS else ["native"]
        lines += [f"### {solver}", "", "| N | " + " | ".join(columns) + " |",
                  "|---:|" + "---:|" * len(columns)]
        for n in metadata["sizes"]:
            cells = []
            for encoding in columns:
                row = indexed.get((n, solver, encoding))
                missing = "not run" if metadata.get("stop_reason") else "pending"
                interrupted = any(case["n"] == n and case["solver"] == solver
                                  and case["encoding"] == encoding
                                  for case in metadata.get("interrupted_cases", []))
                if row is None:
                    cell = "stopped" if interrupted else missing
                elif row["solved"]:
                    cell = measurement(row, "total_s_median")
                    if row["attempts"] > 1 or row["solved"] != row["attempts"]:
                        cell += f" ({row['solved']}/{row['attempts']} solved)"
                else:
                    cell = ", ".join(status.replace("_", " ") for status in OUTCOMES if row[status])
                if row is not None and row["attempts"] < metadata.get("repeats", 1):
                    cell += "; incomplete"
                cells.append(cell)
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Individual attempts", "",
              "| Solver | Encoding | N | Repeat | Status | Phase | Generation (s) | Loading (s) | Solve (s) | Total (s) | Wall (s) | Worker RSS (MiB) | Variables | Clauses | Constraints | Log |",
              "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in results:
        log_file = row.get("log_file")
        log_link = f"[log](<{os.path.relpath(log_file, directory)}>)" if log_file else "—"
        cells = [str(row[key]) for key in ("solver", "encoding", "n")]
        cells += [str(row.get("repetition", 1)), row["status"], row.get("phase", "startup")]
        cells += [measurement(row, key) for key in (*TIMINGS, "peak_rss_mib")]
        cells += [str(row.get(key, "—")) for key in ("variables", "clauses", "constraints")]
        lines.append("| " + " | ".join([*cells, log_link]) + " |")
    lines += ["", "Reproduce:", "", "```sh", shlex.join(metadata["command"]), "```", ""]
    (directory / "report.md").write_text("\n".join(lines))
