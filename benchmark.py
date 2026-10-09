"""Benchmark the repository's SAT encodings in isolated, bounded processes."""

import argparse
import csv
import hashlib
import importlib
import json
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pysat.solvers import Glucose3

from solver.pySAT.Glucose import Glucose

ENCODINGS: tuple[str, ...] = ("binomial", "binary", "commander", "product", "sequential")
FIELDS: tuple[str, ...] = (
    "encoding", "n", "status", "phase", "generation_s", "loading_s",
    "solving_s", "total_s", "wall_s", "clauses", "variables", "peak_rss_mib",
    "conflicts", "decisions", "propagations", "error",
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


def emit(data: dict[str, Any]) -> None:
    print(json.dumps(data), flush=True)


def worker(encoding: str, n: int, memory_mib: int) -> None:
    limit = memory_mib * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    module = importlib.import_module(f"encoder.{encoding}")
    result: dict[str, Any] = {"encoding": encoding, "n": n, "phase": "generation"}
    started = time.perf_counter()
    emit(result)
    try:
        variables = module.generate_variables(n)
        clauses = module.generate_clauses(n, variables)
        result.update(generation_s=time.perf_counter() - started,
                      clauses=len(clauses), phase="loading")
        emit(result)
        loading = time.perf_counter()
        with Glucose3() as glucose:
            solver = Glucose(glucose)
            for clause in clauses:
                solver.add_clause(clause)
            result.update(loading_s=time.perf_counter() - loading,
                          variables=solver.glucose.nof_vars(), phase="solving")
            emit(result)
            solving = time.perf_counter()
            model = solver.solve()
            result.update(solving_s=time.perf_counter() - solving,
                          total_s=time.perf_counter() - started)
            result.update(solver.glucose.accum_stats() or {})
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
    except MemoryError:
        result.update(status="memory_limit", error="Address-space memory cap reached")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    emit(result)


def run_case(encoding: str, n: int, timeout_s: float, memory_mib: int) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               encoding, str(n), str(memory_mib)]
    started = time.perf_counter()
    result: dict[str, Any] = {"encoding": encoding, "n": n}
    timed_out = False
    process = None
    # Defer Ctrl+C until the worker exists and can be cleaned up.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        try:
            output, errors = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        result["peak_rss_mib"] = int(line.split()[1]) / 1024
            except FileNotFoundError:
                pass
            process.kill()
            output, errors = process.communicate()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    for line in output.splitlines():
        result.update(json.loads(line))
    result["wall_s"] = time.perf_counter() - started
    if timed_out:
        result.update(status="timeout", error=f"Exceeded {timeout_s:g} s per-case wall limit")
    elif process.returncode:
        result.update(status="error", error=f"Exit {process.returncode}: {errors.strip()}")
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
        "# N-Queens SAT benchmark", "",
        f"Run: {metadata['started_utc']}. Python {metadata['python']}; "
        f"python-sat {metadata['python_sat']}; Glucose3; {metadata['cpu']}.", "",
        "Each encoding/size runs once in a fresh process, sequentially. "
        "Timings are wall-clock seconds; there are no warmups or statistical repeats. "
        "Generation includes board variables and the full Python clause list. "
        "Loading uses the original clause-by-clause insertion. Total covers generation, "
        "loading and solve (including model extraction), excluding imports, validation, and teardown. "
        "Wall includes the entire worker process. Peak RSS is measured on Linux.", "",
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
    lines += ["## Comparison", "", "Cells show generation + loading + solving "
              "seconds for validated SAT results, or the stopping condition.", "",
              "| N | " + " | ".join(encodings) + " |",
              "|---:|" + "---:|" * len(encodings)]
    indexed = {(row["n"], row["encoding"]): row for row in results}
    for n in metadata["sizes"]:
        cells = []
        for encoding in encodings:
            row = indexed.get((n, encoding))
            missing = "not run" if metadata.get("stop_reason") else "pending"
            if metadata.get("interrupted_case") == {"n": n, "encoding": encoding}:
                missing = "stopped"
            cells.append(missing if row is None else
                         f"{row['total_s']:.4f}" if row["status"] == "sat" else
                         row["status"].replace("_", " "))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Detailed measurements", "",
              "| Encoding | N | Status | Generation (s) | Loading (s) | Solve (s) | Total (s) | Peak RSS (MiB) | Clauses |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        def timing(key: str, measurement: dict[str, Any] = row) -> str:
            return f"{measurement[key]:.4f}" if key in measurement else "—"
        lines.append(f"| {row['encoding']} | {row['n']} | {row['status']} | "
                     f"{timing('generation_s')} | {timing('loading_s')} | "
                     f"{timing('solving_s')} | {timing('total_s')} | "
                     f"{timing('peak_rss_mib')} | {row.get('clauses', '—')} |")
    lines += ["", "Reproduce:", "", "```sh",
              ".venv/bin/python benchmark.py --output benchmark_results", "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.timeout <= 0 or args.memory_mib <= 0:
        parser.error("Sizes must be >= 2 and resource limits must be positive")
    import pysat
    cpu = next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "python_sat": pysat.__version__,
        "platform": platform.platform(), "cpu": cpu, "solver": "Glucose3",
        "timeout_s": args.timeout, "memory_mib": args.memory_mib,
        "sizes": args.sizes, "encodings": list(args.encodings), "repeats": 1,
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parent /
            (name + ".py")).read_bytes()).hexdigest() for name in [
                *(f"encoder/{name}" for name in args.encodings),
                "solver/pySAT/Glucose", "solver/solver_interface", "benchmark"]},
    }
    results: list[dict[str, Any]] = []
    active_case: dict[str, Any] | None = None
    try:
        for n in args.sizes:
            for encoding in args.encodings:
                print(f"Running {encoding:10s} N={n} ...", flush=True)
                active_case = {"encoding": encoding, "n": n}
                row = run_case(encoding, n, args.timeout, args.memory_mib)
                results.append(row)
                active_case = None
                save_results(args.output, metadata, results)
                write_report(args.output, metadata, results)
                print(f"  {row['status']}: wall={row['wall_s']:.3f}s, "
                      f"peak RSS={row.get('peak_rss_mib', 0):.1f} MiB, "
                      f"phase={row.get('phase', 'startup')}", flush=True)
    except KeyboardInterrupt:
        metadata["stop_reason"] = "Cancelled with Ctrl+C"
        if active_case is not None:
            metadata["interrupted_case"] = active_case
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
