# Magellan Communication-Dependency Overhead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce four PP=4 Dense/MoE comparisons that measure steady-state iteration time for CrossPipe 1F1B versus a configuration-specific Magellan order with mandatory Gloo communication dependencies.

**Architecture:** A small pure-Python utility validates schedules, derives solver calibration from CrossPipe `total.json`, parses Megatron iteration logs, and writes reproducible statistics. A single shell launcher owns all model/runtime arguments, while calibration, local Magellan solving, repeated server runs, and result collection remain separate phases with explicit artifacts.

**Tech Stack:** Python 3.12, pytest, Bash, PyTorch/Megatron-Core, torchrun, Gloo/NCCL, OR-Tools CP-SAT, JSON/CSV.

## Global Constraints

- Server repository: `/home/songxb26/mnist/crosspipi-magellan`, branch `crosspipe`.
- Dataset: `/home/songxb26/mnist/crosspipe-old/data`.
- Hardware: one node, four V100 32 GB GPUs; PP=4, TP=1, DP=1.
- Runtime communication uses original NCCL only: `num_dc=1` and delay factors exactly `0,0`.
- Magellan always loads both a configuration-specific order JSON and a non-empty communication-dependency JSON.
- No activation recomputation, synthetic payload, nsys capture, JSONL trace, per-event flush, or custom delay-injection PyTorch build.
- Each measured arm uses three independent launches, 20 iterations per launch, discards iterations 1-5, and retains iterations 6-20.
- Solver topology is the logical star `0-1, 1-2, 1-3`; CP-SAT uses link-exclusive communication with `r_save=1`, `mem=8`, and no rate allocation.
- Performance results are descriptive; no preselected overhead threshold determines success.

---

### Task 1: Add calibration, validation, and timing utilities

**Files:**
- Create: `tools/magellan_overhead.py`
- Create: `tests/custom_schedule_tools/test_magellan_overhead.py`

**Interfaces:**
- Consumes: CrossPipe profile `total.json`, Magellan order/dependency JSON, and Megatron `train.log`.
- Produces:
  - `derive_calibration(profile: dict) -> dict`
  - `validate_schedule(order: dict, dependencies: object, microbatches: int, stages: int) -> dict`
  - `parse_iteration_times(text: str, first_iteration: int = 6, last_iteration: int = 20) -> list[dict]`
  - `summarize(values: list[float]) -> dict`
  - CLI subcommands `calibrate`, `validate`, and `summarize`.

- [ ] **Step 1: Write failing unit tests**

Cover these exact cases:

```python
def test_calibration_uses_middle_stage_forward_and_adjacent_p2p_medians():
    profile = {
        "T_F": [[0.003, 0.005, 0.007, 0.004]],
        "T_alpha": [
            [0, 0.00010, 0, 0],
            [0.00010, 0, 0.00020, 0],
            [0, 0.00020, 0, 0.00030],
            [0, 0, 0.00030, 0],
        ],
        "T_bw": [
            [0, 0.00001, 0, 0],
            [0.00001, 0, 0.00002, 0],
            [0, 0.00002, 0, 0.00003],
            [0, 0, 0.00003, 0],
        ],
    }
    result = derive_calibration(profile)
    assert result["t_f_ref_seconds"] == pytest.approx(0.006)
    assert result["t_comm_ref_seconds"] == pytest.approx(0.00022)
    assert result["comm_units"] == pytest.approx(0.00022 / 0.006)


def test_iteration_parser_keeps_exactly_iterations_6_through_20():
    text = "\n".join(
        f"iteration {i}/ 20 | elapsed time per iteration (ms): {100+i}.0 |"
        for i in range(1, 21)
    )
    rows = parse_iteration_times(text)
    assert [row["iteration"] for row in rows] == list(range(6, 21))
    assert len(rows) == 15


def test_schedule_validator_rejects_nonempty_dependency_cycle():
    with pytest.raises(ValueError, match="cycle"):
        validate_schedule(cyclic_order, cyclic_dependencies, 1, 2)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  tests/custom_schedule_tools/test_magellan_overhead.py -q
```

Expected: import or missing-function failure from `tools.magellan_overhead`.

- [ ] **Step 3: Implement the pure functions and CLI**

Implementation requirements:

```python
ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+)"
)
```

`derive_calibration` must use `median(T_F[0][1:3])` and the median of
`T_alpha[i][i+1] + T_bw[i][i+1]` for `i=0,1,2`. It must reject missing,
non-finite, or non-positive values.

`validate_schedule` must normalize the repository's supported dependency JSON
shapes, require at least one dependency, verify endpoint existence, verify all
`F_m_s` and `B_m_s` operations for `m=0..N-1` and `s=0..S-1`, and run a
topological sort over order edges plus dependency edges.

`summarize` must return count, mean, median, sample standard deviation,
minimum, and maximum.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 pytest command and require exit code 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/magellan_overhead.py \
  tests/custom_schedule_tools/test_magellan_overhead.py
git commit -m "test: add Magellan overhead experiment utilities"
```

---

### Task 2: Add one controlled Dense/MoE launcher

**Files:**
- Create: `test_crossdc/magellan_overhead/run_training.sh`
- Create: `test_crossdc/magellan_overhead/configs.json`

**Interfaces:**
- Consumes: `<config-id> <CALIBRATE|1F1B|MAGELLAN> <run-dir> [order-json dependency-json]`.
- Produces: `run.info`, `train.log`, profile `total.json`, and normal Megatron logs in the requested run directory.

- [ ] **Step 1: Add the four exact model configurations**

`configs.json` must encode:

```json
{
  "D1": {"hidden": 1024, "ffn": 4096, "heads": 16, "seq": 256, "mbs": 1, "gbs": 8, "experts": null},
  "D2": {"hidden": 512, "ffn": 2048, "heads": 8, "seq": 512, "mbs": 2, "gbs": 16, "experts": null},
  "M1": {"hidden": 768, "ffn": 3072, "heads": 12, "seq": 256, "mbs": 1, "gbs": 8, "experts": 4, "topk": 2},
  "M2": {"hidden": 384, "ffn": 1536, "heads": 6, "seq": 512, "mbs": 2, "gbs": 16, "experts": 4, "topk": 2}
}
```

- [ ] **Step 2: Implement argument validation and shared runtime arguments**

The launcher must use the existing virtual environment and dataset paths,
`CUDA_VISIBLE_DEVICES=0,1,2,3`, eight layers, `train-iters=20`, fixed
`--loss-scale 1`, FP16, no recomputation flag, `cdc_profile_iter=2`,
`cdc_exp_test_start_iter=3`, and `cdc_exp_per_cfg_test_iters=17`.

MoE configurations add:

```text
--num-experts 4
--moe-router-topk 2
--expert-model-parallel-size 1
--moe-token-dispatcher-type alltoall
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.01
```

- [ ] **Step 3: Implement the three modes**

- `CALIBRATE` and `1F1B`: `--enable_cdcpp_scheduler --static_schedule 1F1B`.
- `MAGELLAN`: `--custom-pipeline-schedule ORDER --custom-comm-dependency DEP`.
- All modes: `--num_dc 1 --cdc_latency_bandwidth_delay_as_F_stage 0,0`.
- `MAGELLAN` must reject missing or empty order/dependency files before
  launching torchrun.

- [ ] **Step 4: Shell-parse and dry-run validation**

Run:

```bash
bash -n test_crossdc/magellan_overhead/run_training.sh
test_crossdc/magellan_overhead/run_training.sh INVALID 1F1B /tmp/invalid
```

Expected: `bash -n` succeeds; invalid config exits nonzero before torchrun.

- [ ] **Step 5: Commit Task 2**

```bash
git add test_crossdc/magellan_overhead
git commit -m "exp: add Dense and MoE overhead launcher"
```

---

### Task 3: Prove Dense and MoE runtime compatibility

**Files:**
- Modify only if diagnostics require it: `test_crossdc/magellan_overhead/run_training.sh`
- Record: `runs/magellan_comm_dependency_overhead_pp4/smoke/`

**Interfaces:**
- Consumes: Task 2 launcher.
- Produces: one successful short Dense 1F1B log and one successful short MoE 1F1B log.

- [ ] **Step 1: Run D1 1F1B smoke**

Temporarily override the experiment-test count so the process exits after the
first post-profile iteration. Require no OOM, NaN, or distributed timeout.

- [ ] **Step 2: Run M1 1F1B smoke**

Use the same short-run override. If standard MoE arguments are incompatible
with the CDC scheduler, diagnose the first failing stack trace before changing
arguments.

- [ ] **Step 3: Verify profile shape**

For each smoke run, require `total.json` to contain one `T_F` row with four
stage values and 4×4 `T_alpha`/`T_bw` matrices.

- [ ] **Step 4: Re-run launcher validation after any fix**

Run `bash -n`, D1 smoke, and M1 smoke again; require all three to pass.

- [ ] **Step 5: Commit only required compatibility fixes**

Do not commit smoke caches or TensorBoard events.

---

### Task 4: Calibrate all four configurations

**Files:**
- Create results under: `results/magellan_comm_dependency_overhead_pp4/<ID>/calibration/`

**Interfaces:**
- Consumes: four successful `CALIBRATE` runs and Task 1 `calibrate` CLI.
- Produces: four `total.json`, `calibration.json`, `run.info`, and compact calibration logs.

- [ ] **Step 1: Run one uncounted calibration per configuration**

Run D1, D2, M1, and M2 sequentially to avoid GPU contention.

- [ ] **Step 2: Derive calibration JSON**

For each configuration:

```bash
python tools/magellan_overhead.py calibrate \
  --profile-total <run>/total.json \
  --output results/.../<ID>/calibration/calibration.json
```

- [ ] **Step 3: Validate calibration values**

Require finite positive `t_f_ref_seconds`, `t_comm_ref_seconds`, and
`comm_units`. Preserve at least six decimal digits in solver input.

- [ ] **Step 4: Copy compact calibration artifacts**

Copy `total.json`, `run.info`, and the relevant profile lines from `train.log`;
do not copy cache directories.

---

### Task 5: Solve and validate four Magellan schedules

**Files:**
- Create results under: `results/magellan_comm_dependency_overhead_pp4/<ID>/solver/`

**Interfaces:**
- Consumes: local Magellan checkout at `C:\Users\86159\Desktop\MAGELLAN`, local Python with OR-Tools 9.15.6755, and four calibration JSON files copied from the server.
- Produces: four order files, four non-empty notification dependency files, solver summaries, and exact command records copied back to the server.

- [ ] **Step 1: Copy calibration JSON files from server to the local workspace**

Use explicit configuration paths and retain configuration IDs.

- [ ] **Step 2: Run one CP-SAT solve per configuration**

Use:

```text
--topo-folder Motivation/Topology/star_4/bw_300Gbps_lat_0.01s
--microbatches 8 --stages 4
--comm-units <configuration comm_units>
--delay-units 0 0 0 0 0 0
--t-fwd-s <configuration t_f_ref_seconds>
--r-save 1 --mem 8 --time-limit 600 --seed 0
--experiments opt_sim --sat-link-exclusive
--sat-comm-split-k 1 --comm-round-decimals 4
```

- [ ] **Step 3: Require feasible/optimal output and locate canonical files**

For each solver output, require solver status `FEASIBLE` or `OPTIMAL`,
`exp_opt_sim/replay.order.json`, and the generated
`*.notification_deps.json`.

- [ ] **Step 4: Copy solver artifacts to the server**

Copy canonical files and compact solver metadata into the configuration's
`solver/` directory.

- [ ] **Step 5: Run Task 1 schedule validation**

Validate all four schedule/dependency pairs with `N=8`, `S=4`. Require a
nonzero dependency count and an acyclic combined graph.

---

### Task 6: Run the 24 measured training processes

**Files:**
- Create run artifacts under each configuration's `1f1b/` and `magellan_dependency/` directories.

**Interfaces:**
- Consumes: Task 2 launcher and Task 5 validated schedules.
- Produces: three 1F1B and three Magellan logs for each of D1, D2, M1, M2.

- [ ] **Step 1: Run D1 repetitions**

Run 1F1B repetitions 1-3, then Magellan repetitions 1-3, sequentially.

- [ ] **Step 2: Run D2 repetitions**

Use the same ordering and isolation.

- [ ] **Step 3: Run M1 repetitions**

Use the same ordering and isolation.

- [ ] **Step 4: Run M2 repetitions**

Use the same ordering and isolation.

- [ ] **Step 5: Audit process outcomes**

Require 24 completed launch markers, no deadlock/timeout/OOM/NaN text, and
logs proving delay factors `0,0`. Magellan `run.info` must name both canonical
JSON inputs.

---

### Task 7: Summarize and verify results

**Files:**
- Create: `results/magellan_comm_dependency_overhead_pp4/samples.csv`
- Create: `results/magellan_comm_dependency_overhead_pp4/summary.csv`
- Create: `results/magellan_comm_dependency_overhead_pp4/summary.json`
- Create: `results/magellan_comm_dependency_overhead_pp4/README.md`

**Interfaces:**
- Consumes: 24 raw `train.log` files.
- Produces: 360 iteration rows, eight arm summaries, four paired relative-change values, and reproducibility documentation.

- [ ] **Step 1: Parse all logs**

Run the Task 1 `summarize` CLI with the result root. Require exactly iterations
6-20 from every log.

- [ ] **Step 2: Verify sample cardinality**

Require:

```text
24 logs × 15 samples = 360 rows
4 configurations × 2 arms = 8 summary rows
45 samples per configuration/arm
```

- [ ] **Step 3: Compute paired comparison fields**

For each configuration, calculate
`(median_magellan_ms - median_1f1b_ms) / median_1f1b_ms * 100`.

- [ ] **Step 4: Write README**

Document hardware/software, exact configurations, calibration values, solver
statuses, dependency counts, commands, statistics, and the limitation that the
star is a logical scheduler model rather than the single-node physical path.

- [ ] **Step 5: Regenerate summaries and compare hashes**

Delete only generated summary files, regenerate them from raw logs, and require
the regenerated CSV/JSON content to match.

- [ ] **Step 6: Run final repository checks**

Run focused unit tests, `bash -n`, schedule validation for all four inputs,
sample-count checks, and `git diff --check`.

- [ ] **Step 7: Commit and push**

Commit scripts and compact results without caches or large profiler files, then
push branch `crosspipe` to the configured remote.
