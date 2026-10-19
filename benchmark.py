"""Benchmark N-Queens solvers one case at a time in bounded processes."""

import argparse
import hashlib
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from helper import (
    BACKENDS, ENCODINGS, ENCODING_IMPLEMENTATIONS, FIELDS, NATIVE_MODULES,
    REPOSITORY_URL, SAT_SOLVERS, SOLVERS, SOLVER_SEEDS,
    NativeResult, _binary_columns, _result, board_lines, emit, run_case,
    sat_worker, save_results, solve_n_queens, validate_columns, worker, write_report,
)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
               sys.argv[5] if len(sys.argv) > 5 else "glucose",
               float(sys.argv[6]) if len(sys.argv) > 6 else 120,
               sys.argv[7] if len(sys.argv) > 7 else "extended")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=SOLVERS)
    parser.add_argument("--formulation", choices=("minimal", "extended"), default="extended",
                        help="SAT column constraints: AMO only or exactly one (default: extended)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Attempts per solver/encoding/size (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"),
                        help="Parent directory for a new timestamped experiment")
    parser.add_argument("--log-dir", type=Path,
                        help="Optional parent directory for logs (default: inside the experiment)")
    args = parser.parse_args()
    if (min(args.sizes) < 2 or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.memory_mib <= 0 or args.repeats <= 0):
        parser.error("Sizes must be >= 2; limits and repeats must be positive; timeout must be finite")
    args.sizes = list(dict.fromkeys(args.sizes))
    args.encodings = list(dict.fromkeys(args.encodings))
    args.solvers = list(dict.fromkeys(args.solvers))
    started = datetime.now(timezone.utc)
    directory = args.output / started.strftime("%Y-%m-%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    log_dir = args.log_dir / directory.name / "logs" if args.log_dir is not None else directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repository,
                               capture_output=True).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    cpu = next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata: dict[str, Any] = {
        "repository_url": REPOSITORY_URL,
        "started_utc": started.isoformat(), "state": "running",
        "git_revision": revision, "tracked_changes": dirty,
        "python": platform.python_version(), "python_sat": version("python-sat"),
        "platform": platform.platform(), "cpu": cpu, "solvers": args.solvers,
        "logical_cpus": os.cpu_count(),
        "host_memory_mib": next(int(line.split()[1]) / 1024
                                for line in Path("/proc/meminfo").read_text().splitlines()
                                if line.startswith("MemTotal:")),
        "sat_backend_classes": {name: SAT_SOLVERS[name][1]
                                for name in args.solvers if name in SAT_SOLVERS},
        "solver_threads": 1,
        "timeout_s": args.timeout, "memory_mib": args.memory_mib,
        "sizes": args.sizes, "encodings": list(args.encodings), "repeats": args.repeats,
        "formulation": args.formulation,
        "encoding_implementations": {name: ENCODING_IMPLEMENTATIONS[name] for name in args.encodings}
        if any(solver in SAT_SOLVERS for solver in args.solvers) else {},
        "timeout_scope": "Entire worker: imports, generation, loading, solving, validation, teardown",
        "memory_limit_type": "RLIMIT_AS: virtual address space per worker",
        "rss_scope": "Python worker only; excludes native child processes; crash peaks may be sampled",
        "seed_policy": "Fixed seed 0 for CaDiCaL and native CP/MIP; backend defaults for "
                       "Glucose, Kissat, CryptoMiniSat (no seed control in these PySAT interfaces); "
                       "seeds are not varied between repetitions",
        "solver_seeds": {name: SOLVER_SEEDS[name] for name in args.solvers},
        "log_dir": str(log_dir.resolve()),
        "planned_attempts": args.repeats * len(args.sizes) * sum(
            len(args.encodings) if solver in SAT_SOLVERS else 1 for solver in args.solvers),
        "command": [sys.executable, "benchmark.py", "--sizes", *map(str, args.sizes),
                    "--encodings", *args.encodings, "--solvers", *args.solvers,
                    "--formulation", args.formulation,
                    "--timeout", str(args.timeout),
                    "--memory-mib", str(args.memory_mib), "--repeats", str(args.repeats),
                    "--output", str(args.output)],
    }
    if "cplex_cp" in args.solvers:
        from solver.CPOptimizer.CPOptimizer import find_executable

        try:
            executable = find_executable()
            with executable.open("rb") as stream:
                metadata["cp_optimizer_executable"] = {
                    "path": str(executable), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
        except OSError as error:
            metadata["cp_optimizer_executable"] = {"error": str(error)}
    results: list[dict[str, Any]] = []
    if args.log_dir is not None:
        metadata["command"] += ["--log-dir", str(args.log_dir)]
    case = None
    save_results(directory, metadata, results)
    write_report(directory, metadata, results)
    print(f"Experiment: {directory.resolve()}", flush=True)
    try:
        for repetition in range(1, args.repeats + 1):
            for n in args.sizes:
                print(f"Repetition {repetition}/{args.repeats}, N={n}, one case at a time ...", flush=True)
                for solver in args.solvers:
                    for encoding in args.encodings if solver in SAT_SOLVERS else ["native"]:
                        case = {"solver": solver, "encoding": encoding, "n": n,
                                "repetition": repetition}
                        row = run_case(encoding, n, args.timeout, args.memory_mib, solver,
                                       log_dir=log_dir, repetition=repetition,
                                       formulation=args.formulation)
                        results.append(row)
                        case = None
                        save_results(directory, metadata, results)
                        write_report(directory, metadata, results)
                        rss = row.get("peak_rss_mib")
                        print(f"  {row['solver']:12s} {row['encoding']:10s} N={n}: "
                              f"{row['status']}, wall={row['wall_s']:.3f}s, "
                              f"peak RSS={f'{rss:.1f} MiB' if rss is not None else 'unavailable'}, "
                              f"phase={row.get('phase', 'startup')}", flush=True)
    except KeyboardInterrupt:
        metadata["stop_reason"] = "Cancelled with Ctrl+C"
        metadata.update(state="cancelled", finished_utc=datetime.now(timezone.utc).isoformat())
        metadata["interrupted_cases"] = [case] if case is not None else []
        save_results(directory, metadata, results)
        write_report(directory, metadata, results)
        print(f"\nCancelled. Saved {len(results)} completed cases to {directory.resolve()}",
              flush=True)
        sys.exit(130)
    metadata.update(state="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    save_results(directory, metadata, results)
    write_report(directory, metadata, results)
    print(f"Saved {len(results)} cases to {directory.resolve()}", flush=True)
    if any(row["status"] in ("invalid", "error") for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
