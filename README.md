# N-Queens Solver Benchmark

Benchmark SAT, CP, and MIP solvers on the N-Queens feasibility problem. Each case
runs in a fresh worker process with a wall-time and memory limit. Results are not
stored in this repository; regenerate them with the steps below.

## Installation

Linux only. Dependency pins were captured with CPython 3.14.0.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The commercial backends also need vendor engines and licenses:

- **CPLEX** (`cplex_mip`): connect a full CPLEX Optimization Studio install, since
  the bundled Community Edition is size-limited:
  `.venv/bin/docplex config --upgrade /path/to/CPLEX_Studio`
- **CP Optimizer** (`cplex_cp`): `export CPO_EXECUTABLE=/absolute/path/to/cpoptimizer`
- **Gurobi** (`gurobi`): a working license (WLS needs network access).

Missing backends are recorded as `error` results; other cases continue.

## Usage

```sh
.venv/bin/python benchmark.py \
  --sizes 4 8 \
  --solvers glucose cadical \
  --encodings product sequential \
  --repeats 3
```

Each run creates a new timestamped folder under `benchmark_results/`; open its
`report.md`. Ctrl+C stops the run and keeps finished attempts.

| Option | Default | Values |
|---|---|---|
| `--sizes` | `4 8 16 32 64 128 256 512 1024` | Board sizes ≥ 2 |
| `--solvers` | all | `glucose cadical kissat cryptominisat cp_sat cplex_cp cplex_mip gurobi` |
| `--encodings` | all | `binomial binary commander product sequential` (SAT solvers only) |
| `--formulation` | `extended` | `extended` (columns exactly one) or `minimal` (at most one) |
| `--repeats` | `1` | Attempts per case |
| `--timeout` | `120` | Seconds per case |
| `--memory-mib` | `3072` | Memory cap per case |
| `--output` | `benchmark_results` | Output parent folder |
| `--log-dir` | inside the experiment | Separate log folder |

## Reproducing the paper

1. **Check the setup:**

   ```sh
   .venv/bin/python -m unittest discover
   ```

2. **Run both memory budgets** (600 attempts each; keep the machine idle and
   don't edit source files while they run):

   ```sh
   for mib in 2048 6144; do
     .venv/bin/python benchmark.py \
       --sizes 8 16 32 64 128 \
       --encodings sequential product binomial binary commander \
       --solvers glucose cadical kissat cryptominisat cplex_cp cplex_mip gurobi cp_sat \
       --formulation extended --timeout 120 --memory-mib "$mib" --repeats 5 \
       --output "benchmark_results/proposed_${mib}MiB"
   done
   ```

   Each run writes `report.md` with per-size outcomes and median/IQR timings.

3. **Compute the PAR-2 table** (failed attempts count as 2 × 120 s):

   ```sh
   .venv/bin/python analyze_comparison.py \
     benchmark_results/proposed_*MiB/*/results.json \
     --output benchmark_results/analysis
   ```

   `comparison.md` is the paper's fixed-configuration table; `per-size.csv` has
   per-size medians.

Outcomes and model sizes should match the paper exactly; timings depend on
hardware, so compare them as distributions. Each run's `experiment.json` records
the `git_revision` it used.
