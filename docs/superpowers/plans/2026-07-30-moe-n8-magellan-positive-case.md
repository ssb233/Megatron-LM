# MoE N=8 Magellan Positive-Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and verify one PP=4, N=8 MoE configuration whose Magellan schedule retains mandatory communication dependencies and is no slower than CrossPipe 1F1B.

**Architecture:** Extend the existing overhead launcher with one primary M3 configuration, collect a configuration-specific CrossPipe profile, and solve a fresh N=8 star-topology schedule. Run a single screening pair first; only a positive screen is expanded into balanced repeated runs. If M3 fails, add and test exactly one sequence-length fallback M4.

**Tech Stack:** Python 3.12, pytest, Bash, PyTorch/Megatron-Core, torchrun, NCCL, Gloo, OR-Tools CP-SAT, JSON/CSV.

## Global Constraints

- Server repository: `/home/songxb26/mnist/crosspipi-magellan`, branch `crosspipe`.
- Server Python: `/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python`.
- Dataset: `/home/songxb26/mnist/crosspipe-old/data`.
- Local solver checkout: `C:\Users\86159\Desktop\MAGELLAN`.
- Local solver Python: `C:\Users\86159\Desktop\MAGELLAN\.venv_run\Scripts\python.exe`.
- Hardware: one node, four V100 32 GB GPUs; PP=4, TP=1, DP=1.
- D1 and its committed results remain unchanged and are not rerun.
- M3: eight layers, hidden 1024, FFN 4096, heads 16, sequence 256, MBS 1, N/GBS 8, eight experts, top-2.
- M4 is permitted only if M3 screening is negative: same as M3 except sequence length and maximum positions are 512.
- FP16, one virtual chunk, expert parallelism 1, no activation recomputation.
- Runtime uses original NCCL: `num_dc=1`, delay factors `0,0`, no synthetic payload.
- Solver topology is `0-1, 1-2, 1-3`; link exclusivity, `r_save=1`, `mem=8`, no rate allocation, zero propagation-delay units.
- Every accepted Magellan schedule must contain at least one extra dependency and at least one remote Gloo dependency.
- Every arm runs 20 iterations; statistics use iterations 6 through 19.
- Preserve unrelated server worktree changes and stage only files from this experiment.

---

### Task 1: Add the M3 launcher configuration with TDD

**Files:**
- Modify: `test_crossdc/magellan_overhead/configs.json`
- Modify: `tests/custom_schedule_tools/test_magellan_overhead.py`

**Interfaces:**
- Consumes: configuration ID `M3`.
- Produces: launcher values `{hidden: 1024, ffn: 4096, heads: 16, seq: 256, mbs: 1, gbs: 8, experts: 8, topk: 2}`.

- [ ] **Step 1: Extend the expected configuration test first**

Add this exact expected entry:

```python
"M3": {
    "hidden": 1024,
    "ffn": 4096,
    "heads": 16,
    "seq": 256,
    "mbs": 1,
    "gbs": 8,
    "experts": 8,
    "topk": 2,
},
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  tests/custom_schedule_tools/test_magellan_overhead.py -q
```

Expected: the matrix assertion fails because `configs.json` has no M3.

- [ ] **Step 3: Add M3 to `configs.json`**

Add the same numeric fields without changing D1–M2.

- [ ] **Step 4: Verify GREEN**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  tests/custom_schedule_tools/test_magellan_overhead.py -q
bash -n test_crossdc/magellan_overhead/run_training.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add test_crossdc/magellan_overhead/configs.json \
  tests/custom_schedule_tools/test_magellan_overhead.py
git commit -m "exp: add N8 MoE positive-case configuration"
```

---

### Task 2: Preflight and calibrate M3

**Files:**
- Create runtime artifacts: `runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_preflight/`
- Create compact results: `results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/calibration/`

**Interfaces:**
- Consumes: M3 launcher configuration.
- Produces: `total.json`, `run.info`, `calibration.json`, and peak-memory evidence.

- [ ] **Step 1: Run four-iteration preflight**

Use port 29670:

```bash
CDC_OVERHEAD_TRAIN_ITERS=4 \
CDC_OVERHEAD_EXP_TEST_ITERS=1 \
CDC_OVERHEAD_MASTER_PORT=29670 \
bash test_crossdc/magellan_overhead/run_training.sh \
  M3 CALIBRATE \
  runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_preflight
```

- [ ] **Step 2: Audit the preflight**

Require iterations 1–4, zero skipped/NaN, no OOM/timeout, and maximum
`max allocated` below 28672 MiB.

- [ ] **Step 3: Derive M3 calibration**

```bash
python tools/magellan_overhead.py calibrate \
  --profile-total runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_preflight/total.json \
  --output results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/calibration/calibration.json
```

Copy `total.json` and `run.info` to the same calibration directory. Require
finite positive `t_f_ref_seconds`, `t_comm_ref_seconds`, and `comm_units`.

---

### Task 3: Solve and validate a fresh N=8 schedule

**Files:**
- Create local output: `.codex-temp/moe_n8_positive_case_20260730/M3/solver/`
- Create server output: `results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/solver/`

**Interfaces:**
- Consumes: M3 `calibration.json`.
- Produces: order JSON, notification dependency JSON, solver metadata, and `validation.json`.

- [ ] **Step 1: Copy M3 calibration to the local solver workspace**

Use SCP with `ClearAllForwardings=yes`; do not reuse M1 calibration.

- [ ] **Step 2: Solve M3**

Invoke `replay_cp_sat_on_topology.py` with:

```text
--topo-folder C:\Users\86159\Desktop\MAGELLAN\Motivation\Topology\star_4\bw_300Gbps_lat_0.01s
--microbatches 8 --stages 4
--comm-units M3_CALIBRATED_COMM_UNITS
--delay-units 0 0 0 0 0 0
--t-fwd-s M3_CALIBRATED_T_F
--r-save 1 --mem 8 --time-limit 600 --seed 0
--experiments opt_sim --sat-link-exclusive
--sat-comm-split-k 1 --comm-round-decimals 4
```

- [ ] **Step 3: Upload and independently validate**

Upload `replay.order.json`,
`replay.notification.no_competition.notification_deps.json`,
`experiment.summary.json`, and `debug.sat_solution.json`.

Run:

```bash
python tools/magellan_overhead.py validate \
  --order results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/solver/replay.order.json \
  --dependencies results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/solver/replay.notification.no_competition.notification_deps.json \
  --microbatches 8 --stages 4 \
  --output results/magellan_comm_dependency_overhead_pp4/M3_moe_n8/solver/validation.json
```

- [ ] **Step 4: Validate through CrossPipe's loader**

Call `load_custom_schedule(..., pp_size=4, num_microbatches=8)`. Require 112
operations, an acyclic graph, at least one extra dependency, and at least one
dependency whose `is_remote` field is true.

---

### Task 4: Run the M3 screening pair

**Files:**
- Create: `runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_screen_1f1b/`
- Create: `runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_screen_magellan/`

**Interfaces:**
- Consumes: validated M3 schedule and dependency file.
- Produces: one 1F1B and one Magellan 20-iteration screening log.

- [ ] **Step 1: Run 1F1B on port 29671**

Set `CDC_OVERHEAD_TRAIN_ITERS=20` and
`CDC_OVERHEAD_EXP_TEST_ITERS=17`.

- [ ] **Step 2: Run Magellan on port 29672**

Pass both M3 solver JSON paths and use the same training settings.

- [ ] **Step 3: Audit and summarize**

Require completed markers, iterations 1–20, zero skipped/NaN, finite loss,
`num_dc=1`, `delay_pairs=0,0`, and 14 samples per arm from iterations 6–19.
Calculate mean, median, standard deviation, minimum, maximum, and:

```python
(magellan_median_ms / one_f_one_b_median_ms - 1.0) * 100.0
```

- [ ] **Step 4: Apply the conditional gate**

If Magellan median is no greater than 1F1B median, continue to Task 5. If it
is greater, skip Task 5 and execute Task 6.

---

### Task 5: Confirm a positive M3 with balanced run order

**Files:**
- Create six additional compact run directories below:
  `runs/magellan_comm_dependency_overhead_pp4/moe_n8_search_20260730/M3_confirm/`

**Interfaces:**
- Consumes: positive M3 screening result.
- Produces: four runs per arm total, balanced across starting order.

- [ ] **Step 1: Run sequence A**

Run `Magellan` then `1F1B` on ports 29673 and 29674.

- [ ] **Step 2: Run sequence B**

Run `1F1B` then `Magellan` on ports 29675 and 29676.

- [ ] **Step 3: Run sequence C**

Run `Magellan` then `1F1B` on ports 29677 and 29678. Combined with the
screening order `1F1B→Magellan`, each arm starts first twice.

- [ ] **Step 4: Aggregate and accept or reject**

Require 56 samples per arm. Accept M3 only when all eight runs pass their
audits and aggregate Magellan median is no greater than aggregate 1F1B
median. Also report all four paired median differences to expose variance.

---

### Task 6: Run the single permitted M4 fallback only after a negative M3 screen

**Files:**
- Modify with TDD only if entered: `test_crossdc/magellan_overhead/configs.json`
- Modify with TDD only if entered: `tests/custom_schedule_tools/test_magellan_overhead.py`
- Create: `results/magellan_comm_dependency_overhead_pp4/M4_moe_n8_seq512/`

**Interfaces:**
- Consumes: negative M3 screening result.
- Produces: one independently profiled, solved, validated, and screened M4 pair.

- [ ] **Step 1: Add M4 through the same RED/GREEN cycle**

M4 equals M3 except `"seq": 512`. Use ports 29679–29681 for preflight, 1F1B,
and Magellan.

- [ ] **Step 2: Repeat Tasks 2–4 for M4**

Use M4's own calibration and an N=8 solve. Require the same memory, schedule,
dependency, completion, and sample-count gates.

- [ ] **Step 3: Stop after screening**

If M4 screens positive, use ports 29682–29687 to perform the same balanced
confirmation as Task 5. If M4 is negative, stop the search and report both
negative candidates without trying additional BSH combinations.

---

### Task 7: Archive, verify, commit, and push

**Files:**
- Create or update: `results/magellan_comm_dependency_overhead_pp4/README.md`
- Create: candidate-specific logs, summary CSV/JSON, calibration, solver, and validation files.

**Interfaces:**
- Consumes: M3 and, only if entered, M4 evidence.
- Produces: reproducible screening/confirmation result on `origin/crosspipe`.

- [ ] **Step 1: Copy compact artifacts**

Store `train.log`, `run.info`, `total.json`, and `completed.txt` for each
executed arm. Exclude TensorBoard event files and caches.

- [ ] **Step 2: Document the result**

Record exact BSH parameters, measured calibration, peak memory, solver status,
raw/extra/remote dependency counts, every run median, aggregate statistics,
execution order, acceptance decision, and the single-node topology limitation.

- [ ] **Step 3: Run fresh verification**

Run focused pytest, `bash -n`, independent and authoritative schedule
validation, JSON parsing, exact iteration/sample audits, skip/NaN checks,
`git diff --check`, and confirm GPUs are idle.

- [ ] **Step 4: Commit and push**

Stage only approved configuration and compact result files. Commit with
experiment-specific messages, push `crosspipe`, and verify local and remote
branch SHAs match.
