# Remaining Magellan Communication-Dependency Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain the completed D1 result and produce D2, M1, and M2 PP=4 comparisons of CrossPipe 1F1B versus a configuration-specific Magellan schedule with mandatory Gloo communication dependencies.

**Architecture:** The existing launcher owns all shared Megatron arguments and reads one exact configuration from `configs.json`. Each new configuration passes a four-iteration 1F1B memory/calibration gate, is solved locally with its measured compute/communication ratio, is validated on the server, and then runs one 20-iteration 1F1B arm and one 20-iteration Magellan arm. Existing pure-Python utilities validate schedules and generate statistics from committed raw logs.

**Tech Stack:** Python 3.12, pytest, Bash, PyTorch/Megatron-Core, torchrun, NCCL, Gloo, OR-Tools CP-SAT, JSON/CSV.

## Global Constraints

- Server repository: `/home/songxb26/mnist/crosspipi-magellan`, branch `crosspipe`.
- Server Python: `/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python`.
- Dataset: `/home/songxb26/mnist/crosspipe-old/data`.
- Local solver checkout: `C:\Users\86159\Desktop\MAGELLAN`.
- Local solver Python: `C:\Users\86159\Desktop\MAGELLAN\.venv_run\Scripts\python.exe`.
- Hardware: one node, four V100 32 GB GPUs; PP=4, TP=1, DP=1.
- D1 remains unchanged: hidden 1024, FFN 4096, heads 16, sequence 256, MBS 1, N/GBS 8.
- D2: hidden 1536, FFN 6144, heads 24, sequence 512, MBS 2, N 16, GBS 32.
- M1: hidden 1024, FFN 4096, heads 16, sequence 256, MBS 1, N 16, GBS 16, eight experts, top-2.
- M2: hidden 768, FFN 3072, heads 12, sequence 512, MBS 2, N 16, GBS 32, eight experts, top-2.
- All models use eight transformer layers, FP16, one virtual chunk, and no activation recomputation.
- Runtime uses original NCCL only: `num_dc=1`, delay factors `0,0`, and no synthetic payload.
- Solver topology is `0-1, 1-2, 1-3`; CP-SAT uses link exclusivity, `r_save=1`, `mem=8`, no rate allocation, and N=16 for all new solves.
- Every Magellan arm must load both its own order JSON and a dependency JSON with at least one extra edge.
- Each arm runs once for 20 iterations; statistics use iterations 6 through 19.
- Preserve unrelated server worktree changes. Stage and commit only files belonging to the task being committed.

---

### Task 1: Update the approved matrix and timing window with TDD

**Files:**
- Modify: `test_crossdc/magellan_overhead/configs.json`
- Modify: `test_crossdc/magellan_overhead/run_training.sh`
- Modify: `tools/magellan_overhead.py`
- Modify: `tests/custom_schedule_tools/test_magellan_overhead.py`

**Interfaces:**
- Consumes: configuration ID `D1|D2|M1|M2`.
- Produces: exact approved model arguments and default timing rows for iterations 6–19.

- [ ] **Step 1: Change tests first**

Update the expected matrix to:

```python
{
    "D1": {"hidden": 1024, "ffn": 4096, "heads": 16, "seq": 256,
           "mbs": 1, "gbs": 8, "experts": None},
    "D2": {"hidden": 1536, "ffn": 6144, "heads": 24, "seq": 512,
           "mbs": 2, "gbs": 32, "experts": None},
    "M1": {"hidden": 1024, "ffn": 4096, "heads": 16, "seq": 256,
           "mbs": 1, "gbs": 16, "experts": 8, "topk": 2},
    "M2": {"hidden": 768, "ffn": 3072, "heads": 12, "seq": 512,
           "mbs": 2, "gbs": 32, "experts": 8, "topk": 2},
}
```

Rename the parser test to `test_iteration_parser_keeps_exactly_iterations_6_through_19` and require `list(range(6, 20))` with 14 rows. Remove the obsolete top-1 pre-softmax launcher test. Add an assertion that both MoE configurations use `topk == 2` and `experts == 8`.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  tests/custom_schedule_tools/test_magellan_overhead.py -q
```

Expected: matrix and iteration-window assertions fail against the old implementation.

- [ ] **Step 3: Implement the approved configuration**

Update `configs.json`, set `parse_iteration_times(..., last_iteration=19)`, set the summarize CLI default expected sample count to 14, and write `measured_iterations=6-19` in `run.info`. Remove the unreachable `MOE_TOPK == 1` pre-softmax branch.

- [ ] **Step 4: Run focused validation and verify GREEN**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  tests/custom_schedule_tools/test_magellan_overhead.py -q
bash -n test_crossdc/magellan_overhead/run_training.sh
git diff --check
```

Expected: all tests pass and both shell/diff checks exit 0.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add tools/magellan_overhead.py \
  tests/custom_schedule_tools/test_magellan_overhead.py \
  test_crossdc/magellan_overhead/configs.json \
  test_crossdc/magellan_overhead/run_training.sh
git commit -m "exp: finalize remaining overhead configurations"
```

---

### Task 2: Preflight and calibrate D2, M1, and M2

**Files:**
- Create runtime artifacts: `runs/magellan_comm_dependency_overhead_pp4/preflight_20260730/<ID>/`
- Create compact results: `results/magellan_comm_dependency_overhead_pp4/<config-dir>/calibration/`

**Interfaces:**
- Consumes: Task 1 launcher in `CALIBRATE` mode.
- Produces: one valid `total.json`, `run.info`, `calibration.json`, and memory decision per configuration.

- [ ] **Step 1: Run D2 preflight on port 29650**

Set `CDC_OVERHEAD_TRAIN_ITERS=4`, `CDC_OVERHEAD_EXP_TEST_ITERS=1`, and run D2 `CALIBRATE`. Require four iterations, zero skip/NaN, and no OOM or timeout.

- [ ] **Step 2: Enforce the D2 28 GiB gate**

Read the maximum `max allocated` value from all ranks in `train.log`. If it is at least 28672 MiB or the run OOMs, change only D2 MBS/GBS to 1/16, rerun the Task 1 RED/GREEN cycle for that approved fallback, and repeat the preflight.

- [ ] **Step 3: Run and gate M1 on port 29651**

Use the same four-iteration mode. If memory reaches 28672 MiB or OOMs, change M1 hidden/FFN/heads to 768/3072/12 while retaining MBS 1, GBS/N 16, eight experts, and top-2; rerun tests before repeating.

- [ ] **Step 4: Run and gate M2 on port 29652**

Use the same four-iteration mode. If memory reaches 28672 MiB or OOMs, change only M2 MBS/GBS to 1/16, rerun tests, and repeat.

- [ ] **Step 5: Derive and validate calibration JSON**

For each successful run:

```bash
python tools/magellan_overhead.py calibrate \
  --profile-total RUN_DIR/total.json \
  --output RESULT_DIR/calibration/calibration.json
```

Require finite positive `t_f_ref_seconds`, `t_comm_ref_seconds`, and `comm_units`. Copy `total.json` and `run.info` into the same result calibration directory.

---

### Task 3: Solve and validate three N=16 schedules

**Files:**
- Create local outputs: `.codex-temp/remaining_overhead_solver_20260730/<ID>/`
- Create server outputs: `results/magellan_comm_dependency_overhead_pp4/<config-dir>/solver/`

**Interfaces:**
- Consumes: each new `calibration.json`.
- Produces: `replay.order.json`, notification dependency JSON, solver summary, command metadata, and `validation.json`.

- [ ] **Step 1: Copy the three calibration JSON files to the local solver workspace**

Use SCP with explicit D2, M1, and M2 paths. Do not use the stale calibration files already present from the previous matrix.

- [ ] **Step 2: Solve each configuration sequentially**

For each ID, read `comm_units` and `t_f_ref_seconds` from its copied JSON and invoke `replay_cp_sat_on_topology.py` with:

```text
--topo-folder C:\Users\86159\Desktop\MAGELLAN\Motivation\Topology\star_4\bw_300Gbps_lat_0.01s
--microbatches 16 --stages 4
--comm-units CALIBRATED_COMM_UNITS
--delay-units 0 0 0 0 0 0
--t-fwd-s CALIBRATED_T_F
--r-save 1 --mem 8 --time-limit 600 --seed 0
--experiments opt_sim --sat-link-exclusive
--sat-comm-split-k 1 --comm-round-decimals 4
```

Use the existing `Magellan` package bootstrap required by the Windows checkout. Poll long-running solves without restarting them.

- [ ] **Step 3: Upload canonical artifacts**

For each successful solve, upload `replay.order.json`,
`replay.notification.no_competition.notification_deps.json`,
`experiment.summary.json`, and `debug.sat_solution.json`.

- [ ] **Step 4: Validate with both validators**

Run:

```bash
python tools/magellan_overhead.py validate \
  --order RESULT_DIR/solver/replay.order.json \
  --dependencies RESULT_DIR/solver/replay.notification.no_competition.notification_deps.json \
  --microbatches 16 --stages 4 \
  --output RESULT_DIR/solver/validation.json
```

Then load the same pair through `load_custom_schedule(..., pp_size=4,
num_microbatches=16)`. Require an acyclic graph, complete 224-operation
coverage, at least one extra dependency, and at least one remote Gloo
dependency.

---

### Task 4: Run six measured training processes

**Files:**
- Create: `runs/magellan_comm_dependency_overhead_pp4/formal_20260730/<ID>_<arm>/`
- Create compact copies under each configuration result directory.

**Interfaces:**
- Consumes: exact preflight-approved configuration and validated schedule.
- Produces: one 1F1B and one Magellan 20-iteration log for each of D2, M1, and M2.

- [ ] **Step 1: Run D2**

Run 1F1B on port 29660, then Magellan on port 29661. Set
`CDC_OVERHEAD_TRAIN_ITERS=20` and `CDC_OVERHEAD_EXP_TEST_ITERS=17`.

- [ ] **Step 2: Run M1**

Run 1F1B on port 29662, then Magellan on port 29663 with the same iteration settings.

- [ ] **Step 3: Run M2**

Run 1F1B on port 29664, then Magellan on port 29665 with the same iteration settings.

- [ ] **Step 4: Audit every process immediately**

Require `completed.txt`, exactly 20 iteration records, zero skipped and NaN
iterations, no OOM/deadlock/timeout, and identical per-iteration loss between
the two arms of the same configuration. Confirm Magellan `run.info` names both
JSON inputs and both arms record `num_dc=1`, `delay_pairs=0,0`.

- [ ] **Step 5: Copy compact artifacts**

Copy `train.log`, `run.info`, `total.json`, and `completed.txt` into
`1f1b/` or `magellan_dependency/`. Exclude Triton caches and TensorBoard files.

---

### Task 5: Generate the four-configuration summary

**Files:**
- Create: `results/magellan_comm_dependency_overhead_pp4/samples.csv`
- Create: `results/magellan_comm_dependency_overhead_pp4/summary.csv`
- Create: `results/magellan_comm_dependency_overhead_pp4/summary.json`
- Create or update: `results/magellan_comm_dependency_overhead_pp4/README.md`

**Interfaces:**
- Consumes: the retained D1 logs and six new logs.
- Produces: 112 measured rows, eight arm summaries, and four relative-change values.

- [ ] **Step 1: Parse iterations 6–19**

Require 14 rows from each of eight arm logs. D1 uses the retained
`pilot_20260730/raw` logs; D2/M1/M2 use Task 4 compact copies.

- [ ] **Step 2: Write deterministic summaries**

For each arm, write count, mean, median, sample standard deviation, minimum,
and maximum. For each configuration, compute:

```python
(median_magellan_ms - median_1f1b_ms) / median_1f1b_ms * 100.0
```

- [ ] **Step 3: Document interpretation**

Record exact final configurations, any fallback taken, peak memory,
calibration ratios, CP-SAT status, raw/extra/remote dependency counts, timing
statistics, and the limitation that each arm is a single run.

- [ ] **Step 4: Regenerate and compare**

Regenerate CSV/JSON from raw logs and require byte-identical content.

---

### Task 6: Final verification, commit, and push

**Files:**
- Commit only approved scripts, plan, and compact result artifacts.

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: reproducible results on remote branch `crosspipe`.

- [ ] **Step 1: Run fresh verification**

Run the full focused pytest suite, `bash -n`, all four schedule validations,
JSON parsing over the result tree, the eight-log sample cardinality audit,
loss/skip/NaN checks, and `git diff --check`.

- [ ] **Step 2: Inspect scope**

Use `git status --short` and `git diff --stat`. Preserve unrelated or stale
untracked results outside the approved final directories.

- [ ] **Step 3: Commit results**

Stage only the plan, approved launcher/config/tool changes, compact D2/M1/M2
artifacts, and root summaries. Commit with an experiment-specific message.

- [ ] **Step 4: Push and verify remote branch**

Push `crosspipe` to `origin` and verify the remote branch contains the new
commit.
