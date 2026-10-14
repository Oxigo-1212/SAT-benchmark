"""Benchmark all N-Queens solvers in parallel, bounded processes for each size."""

import argparse
import csv
import hashlib
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
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Iterator, Sequence, TypedDict

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
)


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
    print(json.dumps(data), flush=True)


def worker(encoding: str, n: int, memory_mib: int, solver_name: str = "glucose",
           timeout_s: float = 120) -> None:
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
        result.update(status="memory_limit", error="Address-space memory cap reached")
    except Exception as exc:
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


def run_case(encoding: str, n: int, timeout_s: float, memory_mib: int,
             solver_name: str = "glucose", cancel: Event | None = None) -> dict[str, Any]:
    if cancel is not None and cancel.is_set():
        raise CancelledError
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               encoding, str(n), str(memory_mib), solver_name, str(timeout_s)]
    started = time.perf_counter()
    result: dict[str, Any] = {"solver": solver_name, "encoding": encoding, "n": n}
    timed_out = False
    process = None
    # Defer Ctrl+C until the worker exists and can be cleaned up.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True,
                                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
                                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise CancelledError
                remaining = timeout_s - (time.perf_counter() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                try:
                    output, errors = process.communicate(timeout=min(remaining, 0.1))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        result["peak_rss_mib"] = int(line.split()[1]) / 1024
            except FileNotFoundError:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, errors = process.communicate()
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    for line in output.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # Native backends can print license or startup messages.
        if isinstance(data, dict):
            result.update(data)
    result["wall_s"] = time.perf_counter() - started
    if timed_out:
        result.update(status="timeout", error=f"Exceeded {timeout_s:g} s per-case wall limit")
    elif process.returncode:
        result.update(status="error", error=f"Exit {process.returncode}: {errors.strip()}")
    elif "status" not in result:
        result.update(status="error", error="Worker exited without a result")
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
               sys.argv[5] if len(sys.argv) > 5 else "glucose",
               float(sys.argv[6]) if len(sys.argv) > 6 else 120)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=SOLVERS)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="Maximum parallel cases per N (default: up to 4)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if (min(args.sizes) < 2 or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.memory_mib <= 0 or args.jobs < 1):
        parser.error("Sizes must be >= 2; jobs and limits must be positive; timeout must be finite")
    args.sizes = list(dict.fromkeys(args.sizes))
    args.encodings = list(dict.fromkeys(args.encodings))
    args.solvers = list(dict.fromkeys(args.solvers))
    import pysat
    cpu = next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "python_sat": pysat.__version__,
        "platform": platform.platform(), "cpu": cpu, "solvers": args.solvers,
        "jobs": args.jobs, "solver_threads": 1,
        "timeout_s": args.timeout, "memory_mib": args.memory_mib,
        "sizes": args.sizes, "encodings": list(args.encodings), "repeats": 1,
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parent /
            (name + ".py")).read_bytes()).hexdigest() for name in [
                *(f"encoder/{name}" for name in args.encodings),
                *(f"solver/pySAT/{SAT_SOLVERS[name][0]}" for name in args.solvers if name in SAT_SOLVERS),
                *(f"solver/{NATIVE_MODULES[name]}/{NATIVE_MODULES[name]}" for name in args.solvers if name in BACKENDS),
                "solver/solver_interface", "benchmark"]},
        "command": [sys.executable, "benchmark.py", "--sizes", *map(str, args.sizes),
                    "--encodings", *args.encodings, "--solvers", *args.solvers,
                    "--jobs", str(args.jobs), "--timeout", str(args.timeout),
                    "--memory-mib", str(args.memory_mib), "--output", str(args.output)],
    }
    results: list[dict[str, Any]] = []
    cancel = Event()
    futures = {}
    save_results(args.output, metadata, results)
    write_report(args.output, metadata, results)
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            try:
                for n in args.sizes:
                    futures = {}
                    for solver in args.solvers:
                        for encoding in args.encodings if solver in SAT_SOLVERS else ["native"]:
                            future = pool.submit(run_case, encoding, n, args.timeout,
                                                 args.memory_mib, solver, cancel)
                            futures[future] = {"solver": solver, "encoding": encoding, "n": n}
                    print(f"Running {len(futures)} cases for N={n}, up to {args.jobs} in parallel ...", flush=True)
                    for future in as_completed(futures):
                        row = future.result()
                        results.append(row)
                        del futures[future]
                        save_results(args.output, metadata, results)
                        write_report(args.output, metadata, results)
                        print(f"  {row['solver']:12s} {row['encoding']:10s} N={n}: "
                              f"{row['status']}, wall={row['wall_s']:.3f}s, "
                              f"peak RSS={row.get('peak_rss_mib', 0):.1f} MiB, "
                              f"phase={row.get('phase', 'startup')}", flush=True)
            finally:
                cancel.set()
                for future in futures:
                    future.cancel()
    except KeyboardInterrupt:
        metadata["stop_reason"] = "Cancelled with Ctrl+C"
        interrupted_cases = []
        for future, case in futures.items():
            try:
                results.append(future.result())
            except (CancelledError, KeyboardInterrupt):
                interrupted_cases.append(case)
        metadata["interrupted_cases"] = interrupted_cases
        save_results(args.output, metadata, results)
        write_report(args.output, metadata, results)
        print(f"\nCancelled. Saved {len(results)} completed cases to {args.output.resolve()}",
              flush=True)
        sys.exit(130)
    print(f"Saved {len(results)} cases to {args.output.resolve()}", flush=True)
    if any(row["status"] in ("invalid", "error") for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
