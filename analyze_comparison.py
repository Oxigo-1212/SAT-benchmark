"""Export per-size timing variability and failure-penalized scores from saved runs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from helper import OUTCOMES, summarize_results, validate_columns


def score_run(row, timeout):
    """PAR-2-style resource score: completed worker wall time, otherwise 2T."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    if row["status"] not in ("sat", "unsat"):
        return 2 * timeout
    if row["status"] == "unsat" and row["n"] not in (2, 3):
        raise ValueError("Unexpected UNSAT on a feasible unrestricted board")
    if row["status"] == "sat":
        columns = [value - 1 for value in row["queen_columns_1based"]]
        if not validate_columns(columns, row["n"]):
            raise ValueError("Invalid saved placement")
    wall = row["wall_s"]
    if not math.isfinite(wall) or wall < 0:
        raise ValueError("Successful run requires finite nonnegative worker wall time")
    return wall


def analyze(paths, output):
    groups, sources, seen = {}, [], set()
    for path in paths:
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        if metadata and metadata.get("state") != "completed":
            raise ValueError(f"Experiment is incomplete: {path}")
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "metadata": metadata or data.get("sources")})
        for original in data["results"]:
            row = dict(original)
            timeout = row.get("timeout_s", metadata.get("timeout_s"))
            memory = row.get("memory_mib", metadata.get("memory_mib"))
            formulation = row.get("formulation", "native" if row["encoding"] == "native" else "extended")
            row["formulation"] = formulation
            key = (timeout, memory, row["solver"], row["encoding"], formulation)
            identity = (*key, row["n"], row.get("repetition", 1))
            if identity in seen:
                raise ValueError(f"Duplicate configuration/repetition: {identity}")
            seen.add(identity)
            row["resource_par2_s"] = score_run(row, timeout)
            groups.setdefault(key, []).append(row)
    aggregates, per_size = [], []
    for (timeout, memory, solver, encoding, formulation), rows in sorted(groups.items()):
        base = dict(timeout_s=timeout, memory_mib=memory, solver=solver,
                    encoding=encoding, formulation=formulation)
        aggregates.append({**base, "sizes": " ".join(map(str, sorted({r["n"] for r in rows}))),
                           "attempts": len(rows),
                           **{status: sum(r["status"] == status for r in rows) for status in OUTCOMES},
                           "resource_par2_s": statistics.mean(r["resource_par2_s"] for r in rows)})
        repeats = max(r.get("repetition", 1) for r in rows)
        for summary in summarize_results(rows, repeats):
            size_rows = [r for r in rows if r["n"] == summary["n"]]
            per_size.append({**base, **summary,
                             "resource_par2_s": statistics.mean(r["resource_par2_s"] for r in size_rows)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("aggregate.csv", aggregates), ("per-size.csv", per_size)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    lines = ["# Fixed-configuration comparison", "",
             "Each solver/encoding/formulation remains fixed across the listed sizes. "
             "Do not compare aggregate scores across different budgets or size sets.", "",
             "Resource PAR-2 uses validated successful worker wall time; timeout, memory-limit, "
             "error, invalid and unknown outcomes each contribute twice the worker deadline. "
             "This is a scoring convention, not an imputed solve time. Errors remain explicit. "
             "Per-size medians and inclusive IQRs in per-size.csv use successful runs only; "
             "always read their sample counts. These repetitions measure timing variation, "
             "not independent instances or randomized search seeds.", "",
             "| Budget (s/MiB) | Solver | Encoding | Formulation | SAT/attempts | TO | ML | Errors/invalid/unknown | Resource PAR-2 (s) |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        errors = row["error"] + row["invalid"] + row["unknown"]
        lines.append(f"| {row['timeout_s']:g}/{row['memory_mib']} | {row['solver']} | {row['encoding']} | "
                     f"{row['formulation']} | {row['sat']}/{row['attempts']} | {row['timeout']} | "
                     f"{row['memory_limit']} | {errors} | {row['resource_par2_s']:.4f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.inputs, args.output)
