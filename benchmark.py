"""Benchmark all N-Queens solvers in parallel, bounded processes for each size."""

import argparse
import hashlib
import math
import os
import platform
import sys
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from helper import (
    BACKENDS, ENCODINGS, FIELDS, NATIVE_MODULES, SAT_SOLVERS, SOLVERS,
    NativeResult, _binary_columns, _result, board_lines, emit, run_case,
    sat_worker, save_results, solve_n_queens, validate_columns, worker, write_report,
)


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
    parser.add_argument("--log-dir", type=Path,
                        help="Write live per-case output, memory snapshots, and crash tracebacks")
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
                "solver/solver_interface", "helper", "benchmark"]},
        "command": [sys.executable, "benchmark.py", "--sizes", *map(str, args.sizes),
                    "--encodings", *args.encodings, "--solvers", *args.solvers,
                    "--jobs", str(args.jobs), "--timeout", str(args.timeout),
                    "--memory-mib", str(args.memory_mib), "--output", str(args.output)],
    }
    results: list[dict[str, Any]] = []
    if args.log_dir is not None:
        metadata["log_dir"] = str(args.log_dir.resolve())
        metadata["command"] += ["--log-dir", str(args.log_dir)]
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
                                                 args.memory_mib, solver, cancel,
                                                 **({"log_dir": args.log_dir} if args.log_dir is not None else {}))
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
