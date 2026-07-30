# Custom Schedule Delay Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable CrossPipe's existing transfer-delay injection for an externally supplied custom schedule, run experiment C with `0.5×F` and `1.0×F` artificial transfer delays on four V100 GPUs, and produce a communication-visible publication trace.

**Architecture:** Keep the Magellan order and dependency graph authoritative while allowing the experiment manager to update only the active delay fields. Import the custom PyTorch build from `/home/songxb26/mnist/pytorch-corsspipe`, inject delay on every logical PP boundary, record the active delay in each trace iteration, and analyze/plot the zero-, half-, and one-stage-delay traces without mixing them into the original A/B/C performance comparison.

**Tech Stack:** Python 3.12, PyTorch `2.5.0a0+git00abf21` with CUDA 12.6, NCCL, Gloo, pytest, Bash, Matplotlib.

## Global Constraints

- Work in `/home/songxb26/mnist/crosspipi-magellan` on branch `crosspipe`.
- Use `/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python`.
- Prepend `/home/songxb26/mnist/pytorch-corsspipe` to `PYTHONPATH` for every delay-injection process.
- Verify `torch.__file__` resolves below `/home/songxb26/mnist/pytorch-corsspipe` before launching four ranks.
- Use PP=4, TP=1, DP=1, eight microbatches, one virtual chunk, FP16, and no activation recomputation.
- Use only transfer-delay pairs `0,0.5` and `0,1.0`; do not add dummy tensors.
- Keep the original zero-delay validation under `results/custom_schedule_v100/validation_20260730/` unchanged.
- Label all new traces and figures `visualization_only`; do not include them in A/B/C throughput claims.
- Preserve the external custom TaskNodes, event plan, receive placement, and communication dependencies.

---

## File Structure

### Create

- `megatron/core/pipeline_parallel/cdc_scheduler/delay_config.py`
  Lightweight validation for custom-schedule delay configurations.
- `tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py`
  Unit coverage for validation, delay updates, and trace metadata.
- `tools/analyze_custom_schedule_delay_trace.py`
  Validate delay intervals and dependency ordering; emit source-data JSON/CSV.
- `tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py`
  Synthetic trace tests for the analyzer.
- `tools/plot_custom_schedule_delay_trace.py`
  Python-only publication plot for zero-, half-, and one-stage traces.
- `results/custom_schedule_v100/visualization_delay_20260730/README.md`
  Experiment scope, configuration, measurements, and figure interpretation.

### Modify

- `megatron/training/arguments.py`
  Replace the unconditional custom zero-delay assertions with shared validation.
- `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`
  Apply delay state before returning from the custom path and record trace metadata.
- `megatron/core/pipeline_parallel/cdc_scheduler/custom_schedule_trace.py`
  Add a typed delay-configuration event.
- `test_crossdc/custom_schedule_v100/run_custom_schedule.sh`
  Add `C_VIS`/smoke support and import the custom PyTorch source tree.
- `docs/custom_pipeline_schedule.md`
  Document visualization-only delay injection and the custom PyTorch requirement.

---

### Task 1: Validate Custom Transfer-Delay Configurations

**Files:**
- Create: `megatron/core/pipeline_parallel/cdc_scheduler/delay_config.py`
- Modify: `megatron/training/arguments.py:165-180`
- Modify: `megatron/training/arguments.py:335-355`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py:1010-1045`
- Test: `tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py`

**Interfaces:**
- Produces: `validate_custom_delay_configuration(*, pp_size: int, num_dc: int, pp_stages_per_dc: Sequence[int], delay_pairs: Sequence[Tuple[float, float]]) -> None`
- Consumes: argparse values already present on `args`.

- [ ] **Step 1: Write validation tests**

```python
import pytest

from megatron.core.pipeline_parallel.cdc_scheduler.delay_config import (
    validate_custom_delay_configuration,
)


def test_zero_delay_custom_schedule_accepts_single_dc():
    validate_custom_delay_configuration(
        pp_size=4,
        num_dc=1,
        pp_stages_per_dc=[],
        delay_pairs=[(0.0, 0.0)],
    )


def test_nonzero_delay_rejects_single_dc():
    with pytest.raises(ValueError, match="num_dc=4"):
        validate_custom_delay_configuration(
            pp_size=4,
            num_dc=1,
            pp_stages_per_dc=[],
            delay_pairs=[(0.0, 0.5)],
        )


def test_transfer_delay_accepts_one_stage_per_dc():
    validate_custom_delay_configuration(
        pp_size=4,
        num_dc=4,
        pp_stages_per_dc=[1, 1, 1, 1],
        delay_pairs=[(0.0, 0.5), (0.0, 1.0)],
    )


@pytest.mark.parametrize(
    "pairs,match",
    [
        ([(-0.1, 0.5)], "non-negative"),
        ([(0.5, 0.0)], "transfer-delay"),
    ],
)
def test_visualization_delay_rejects_unsupported_pairs(pairs, match):
    with pytest.raises(ValueError, match=match):
        validate_custom_delay_configuration(
            pp_size=4,
            num_dc=4,
            pp_stages_per_dc=[1, 1, 1, 1],
            delay_pairs=pairs,
        )
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py -q
```

Expected: collection fails because `delay_config.py` does not exist.

- [ ] **Step 3: Implement the lightweight validator**

```python
from typing import Sequence, Tuple


def validate_custom_delay_configuration(
    *,
    pp_size: int,
    num_dc: int,
    pp_stages_per_dc: Sequence[int],
    delay_pairs: Sequence[Tuple[float, float]],
) -> None:
    pairs = tuple((float(lat), float(bw)) for lat, bw in delay_pairs)
    if any(lat < 0 or bw < 0 for lat, bw in pairs):
        raise ValueError("custom schedule delay factors must be non-negative")

    has_delay = any(lat != 0 or bw != 0 for lat, bw in pairs)
    if not has_delay:
        if num_dc != 1:
            raise ValueError("zero-delay custom validation requires num_dc=1")
        return

    if any(lat != 0 for lat, _ in pairs):
        raise ValueError(
            "custom visualization supports transfer-delay injection only"
        )
    if pp_size != 4 or num_dc != 4:
        raise ValueError(
            "custom visualization delay requires PP=4 and num_dc=4"
        )
    if list(pp_stages_per_dc) != [1, 1, 1, 1]:
        raise ValueError(
            "custom visualization delay requires "
            "pp_stages_per_dc=[1,1,1,1]"
        )
```

- [ ] **Step 4: Call the validator from both runtime validation sites**

Replace the `num_dc == 1` and all-zero assertions in `arguments.py` and
`pp_scheduler.py` with:

```python
validate_custom_delay_configuration(
    pp_size=args.pipeline_model_parallel_size,
    num_dc=args.num_dc,
    pp_stages_per_dc=args.pp_stages_per_dc,
    delay_pairs=args.cdc_latency_bandwidth_delay_as_F_stage,
)
```

Raise `ValueError` rather than relying on optimized-away `assert` statements for
user configuration errors.

- [ ] **Step 5: Run focused and existing custom-schedule tests**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  megatron/core/pipeline_parallel/cdc_scheduler/delay_config.py \
  megatron/training/arguments.py \
  megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py
git commit -m "feat: validate custom schedule transfer delay"
```

---

### Task 2: Apply Delay Without Rebuilding the Custom Schedule

**Files:**
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py:814-840`
- Test: `tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py`

**Interfaces:**
- Consumes: `ExperimentManager.get_injected_latency_bandwidth_delay_seconds()`
- Produces: updated `scheduler.injected_latency_delay` and `scheduler.injected_bandwidth_delay`; custom schedule object identity remains unchanged.

- [ ] **Step 1: Add a failing scheduler test**

Create a scheduler with `__new__` and a fake experiment manager:

```python
class _FakeExperimentManager:
    profile_result = {"T_F": [0.01]}
    T_F_stage = 0.01

    def need_schedule_update_in_current_iter(self):
        return True

    def get_injected_latency_bandwidth_delay_seconds(self):
        return 0.0, 0.005

    def get_injected_latency_bandwidth_delay_as_F_stage(self):
        return 0.0, 0.5


def test_custom_schedule_updates_delay_without_replacing_plan():
    scheduler = CDCPPScheduler.__new__(CDCPPScheduler)
    marker = object()
    scheduler.custom_schedule_spec = marker
    scheduler.pp_execution_plan = marker
    scheduler.exp_manager = _FakeExperimentManager()
    scheduler.injected_latency_delay = (0.0, 0.0)
    scheduler.injected_bandwidth_delay = (0.0, 0.0)
    scheduler.cdc_print = lambda *args, **kwargs: None

    scheduler.update_schedule_with_latency_bandwidth()

    assert scheduler.injected_latency_delay == (0.0, 0.0)
    assert scheduler.injected_bandwidth_delay == (0.5, 0.005)
    assert scheduler.pp_execution_plan is marker
```

- [ ] **Step 2: Run the test and observe the current early-return failure**

Expected: bandwidth delay remains `(0.0, 0.0)`.

- [ ] **Step 3: Move only the custom return**

Restructure the method in this order:

```python
if self.exp_manager.profile_result is None:
    return
if not self.exp_manager.need_schedule_update_in_current_iter():
    return

latency_sec, bandwidth_sec = (
    self.exp_manager.get_injected_latency_bandwidth_delay_seconds()
)
latency_factor, bandwidth_factor = (
    self.exp_manager.get_injected_latency_bandwidth_delay_as_F_stage()
)
self.injected_latency_delay = (latency_factor, latency_sec)
self.injected_bandwidth_delay = (bandwidth_factor, bandwidth_sec)
self.cdc_print(...)

if self.custom_schedule_spec is not None:
    return
```

Leave all default static/dynamic schedule reconstruction code after this return
unchanged.

- [ ] **Step 4: Run focused tests**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add \
  megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py
git commit -m "feat: apply delay to external static schedules"
```

---

### Task 3: Record the Active Delay in Every Trace Interval

**Files:**
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/custom_schedule_trace.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py:1855-1870`
- Modify: `tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py`
- Test: `tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py`

**Interfaces:**
- Produces: `CustomScheduleTrace.record_delay_configuration(...) -> None`
- Event schema: `event="delay_config"` with five numeric fields.

- [ ] **Step 1: Add the failing trace schema test**

```python
trace.record_delay_configuration(
    latency_factor=0.0,
    bandwidth_factor=0.5,
    latency_seconds=0.0,
    bandwidth_seconds=0.005,
    forward_stage_seconds=0.01,
)
```

Assert the JSONL record equals:

```python
{
    "event": "delay_config",
    "latency_factor": 0.0,
    "bandwidth_factor": 0.5,
    "latency_seconds": 0.0,
    "bandwidth_seconds": 0.005,
    "forward_stage_seconds": 0.01,
}
```

excluding common rank/timestamp/iteration fields.

- [ ] **Step 2: Run the trace test and verify `AttributeError`**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py -q
```

- [ ] **Step 3: Implement the trace method**

```python
def record_delay_configuration(
    self,
    *,
    latency_factor: float,
    bandwidth_factor: float,
    latency_seconds: float,
    bandwidth_seconds: float,
    forward_stage_seconds: float,
) -> None:
    self.record(
        "delay_config",
        latency_factor=float(latency_factor),
        bandwidth_factor=float(bandwidth_factor),
        latency_seconds=float(latency_seconds),
        bandwidth_seconds=float(bandwidth_seconds),
        forward_stage_seconds=float(forward_stage_seconds),
    )
```

- [ ] **Step 4: Record configuration immediately after scheduler update**

After `self.update_schedule_with_latency_bandwidth()`:

```python
if self.custom_schedule_trace is not None:
    self.custom_schedule_trace.record_delay_configuration(
        latency_factor=self.injected_latency_delay[0],
        bandwidth_factor=self.injected_bandwidth_delay[0],
        latency_seconds=self.injected_latency_delay[1],
        bandwidth_seconds=self.injected_bandwidth_delay[1],
        forward_stage_seconds=self.exp_manager.T_F_stage,
    )
```

- [ ] **Step 5: Run trace and custom-delay tests**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py -q
```

- [ ] **Step 6: Commit**

```bash
git add \
  megatron/core/pipeline_parallel/cdc_scheduler/custom_schedule_trace.py \
  megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py
git commit -m "feat: trace custom schedule delay configuration"
```

---

### Task 4: Add a Custom-PyTorch Visualization Launch Mode

**Files:**
- Modify: `test_crossdc/custom_schedule_v100/run_custom_schedule.sh`
- Modify: `docs/custom_pipeline_schedule.md`

**Interfaces:**
- Produces: launch modes `C_VIS` and environment override `CDC_EXP_PER_CFG_TEST_ITERS`.
- Consumes: custom PyTorch root `/home/songxb26/mnist/pytorch-corsspipe`.

- [ ] **Step 1: Add mode-specific variables before `COMMON_ARGS`**

```bash
NUM_DC=1
PP_STAGES_PER_DC=()
DELAY_PAIRS=(0,0)
EXP_TEST_ITERS="${CDC_EXP_PER_CFG_TEST_ITERS:-8}"
USE_CROSSPIPE_TORCH=0

if [[ "${MODE}" == "C_VIS" ]]; then
  NUM_DC=4
  PP_STAGES_PER_DC=(1 1 1 1)
  DELAY_PAIRS=(0,0.5 0,1.0)
  USE_CROSSPIPE_TORCH=1
fi
```

- [ ] **Step 2: Import and verify the custom PyTorch build**

```bash
if [[ "${USE_CROSSPIPE_TORCH}" == "1" ]]; then
  PYTORCH_CROSSPIPE_ROOT="/home/songxb26/mnist/pytorch-corsspipe"
  export PYTHONPATH="${PYTORCH_CROSSPIPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON}" -c '
from pathlib import Path
import torch
root = Path("/home/songxb26/mnist/pytorch-corsspipe").resolve()
loaded = Path(torch.__file__).resolve()
assert loaded.is_relative_to(root), (loaded, root)
assert torch.__version__ == "2.5.0a0+git00abf21", torch.__version__
print(f"custom torch: {loaded} ({torch.__version__})")
'
fi
```

- [ ] **Step 3: Build delay arguments from arrays**

Replace hard-coded values with:

```bash
COMMON_ARGS+=(
  --num_dc "${NUM_DC}"
  --cdc_exp_per_cfg_test_iters "${EXP_TEST_ITERS}"
  --cdc_latency_bandwidth_delay_as_F_stage "${DELAY_PAIRS[@]}"
)
if [[ "${#PP_STAGES_PER_DC[@]}" -gt 0 ]]; then
  COMMON_ARGS+=(--pp_stages_per_dc "${PP_STAGES_PER_DC[@]}")
fi
```

- [ ] **Step 4: Add `C_VIS` scheduling arguments**

```bash
C_VIS)
  SCHEDULE_ARGS+=(
    --custom-pipeline-schedule "${ORDER_FILE}"
    --custom-comm-dependency "${DEPENDENCY_FILE}"
    --custom-schedule-trace-dir "${RUN_DIR}/trace"
  )
  ;;
```

Use `RUN_NAME="C_visualization_only_delay"` and a distinct master port.

- [ ] **Step 5: Run shell and import checks**

```bash
bash -n test_crossdc/custom_schedule_v100/run_custom_schedule.sh
env PYTHONPATH=/home/songxb26/mnist/pytorch-corsspipe \
  /home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -c \
  'import torch; print(torch.__file__, torch.__version__)'
```

Expected path/version:

```text
/home/songxb26/mnist/pytorch-corsspipe/torch/__init__.py
2.5.0a0+git00abf21
```

- [ ] **Step 6: Document visualization-only semantics**

Add the exact `C_VIS` command, iteration mapping, and warning that the delay is
implemented by a GPU sleep on the NCCL stream in the custom PyTorch build.

- [ ] **Step 7: Commit**

```bash
git add \
  test_crossdc/custom_schedule_v100/run_custom_schedule.sh \
  docs/custom_pipeline_schedule.md
git commit -m "exp: add custom schedule delay visualization mode"
```

---

### Task 5: Analyze and Plot Delayed Dependency Traces

**Files:**
- Create: `tools/analyze_custom_schedule_delay_trace.py`
- Create: `tools/plot_custom_schedule_delay_trace.py`
- Create: `tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py`

**Interfaces:**
- Analyzer CLI consumes `--baseline-trace`, `--delayed-trace`, `--output-json`, and `--output-csv`.
- Plotter CLI consumes analyzer JSON/CSV and emits SVG/PDF/PNG plus figure source-data CSV.

- [ ] **Step 1: Write synthetic analyzer tests**

Create minimal JSONL events for dependency 6:

```text
delay_config
target_submit Comm_F_4_1_2
comm_complete Comm_F_4_1_2
signal_send_start dependency 6
signal_recv dependency 6
target_submit Comm_B_0_3_2
```

Test:

1. factors `0.0`, `0.5`, and `1.0` are separated;
2. dependency ordering violations raise `ValueError`;
3. the explicitly supplied legacy baseline trace may omit `delay_config` and is
   classified as factor `0.0`;
4. a delayed trace missing `delay_config` raises `ValueError`;
5. communication duration is
   `comm_complete - target_submit`;
6. sender and receiver ranks match the dependency event.

- [ ] **Step 2: Run tests and verify the analyzer import fails**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py -q
```

- [ ] **Step 3: Implement analyzer output**

The summary JSON must contain:

```json
{
  "configurations": {
    "0.0": {"iterations": [], "communication_duration_us": []},
    "0.5": {"iterations": [], "communication_duration_us": []},
    "1.0": {"iterations": [], "communication_duration_us": []}
  },
  "dependency_checks": {
    "checked": 0,
    "violations": []
  },
  "representative_iterations": {
    "0.5": 0,
    "1.0": 0
  }
}
```

Select representative iterations by minimum absolute distance from each
configuration's median selected-communication duration.

- [ ] **Step 4: Implement the publication plot**

Use Python/Matplotlib with:

```python
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial", "DejaVu Sans", "Liberation Sans"
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
```

Build:

- panel a: `1.0×F` four-stage compute/P2P timeline;
- panel b: measured dependency-6/dependency-1 chain;
- panel c: zero/half/one transfer-duration comparison.

Add the figure annotation:

```text
Artificial transfer delay for visualization only
```

- [ ] **Step 5: Run analyzer and plotting tests**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py -q
```

- [ ] **Step 6: Commit**

```bash
git add \
  tools/analyze_custom_schedule_delay_trace.py \
  tools/plot_custom_schedule_delay_trace.py \
  tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py
git commit -m "feat: analyze delayed custom schedule traces"
```

---

### Task 6: Run Four-GPU Smoke and Full Visualization Experiments

**Files:**
- Runtime output: `runs/custom_schedule_v100/visualization_delay_20260730/`

**Interfaces:**
- Consumes: `C_VIS` launcher and custom PyTorch.
- Produces: per-rank JSONL traces for both delay factors.

- [ ] **Step 1: Verify the four GPUs are idle**

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

Expected: no unrelated process occupying GPUs 0–3.

- [ ] **Step 2: Run a one-iteration-per-factor smoke test**

```bash
cd /home/songxb26/mnist/crosspipi-magellan
export CDC_EXP_PER_CFG_TEST_ITERS=1
test_crossdc/custom_schedule_v100/run_custom_schedule.sh \
  C_VIS \
  "$PWD/runs/custom_schedule_v100/visualization_delay_20260730_smoke"
```

Expected:

- custom torch path/version is printed;
- delay config updates to `0.5` and `1.0`;
- the converted bandwidth delay printed for each nonzero factor is at least
  `1 ms`, so the integer NCCL tag cannot truncate it to zero;
- no missing `wait_with_lat_delay_in_ms`;
- no NCCL/Gloo timeout;
- four rank JSONL files contain both `delay_config` values.

- [ ] **Step 3: Inspect the smoke trace**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  tools/analyze_custom_schedule_delay_trace.py \
  --baseline-trace \
    results/custom_schedule_v100/validation_20260730/trace \
  --delayed-trace \
    runs/custom_schedule_v100/visualization_delay_20260730_smoke/C_visualization_only_delay/trace \
  --output-json /tmp/custom_delay_smoke.json \
  --output-csv /tmp/custom_delay_smoke.csv
```

Expected: no dependency violations and positive durations for both factors.

- [ ] **Step 4: Run the full visualization experiment**

```bash
unset CDC_EXP_PER_CFG_TEST_ITERS
test_crossdc/custom_schedule_v100/run_custom_schedule.sh \
  C_VIS \
  "$PWD/runs/custom_schedule_v100/visualization_delay_20260730"
```

Expected: iterations 3–10 use `0.5×F`, iterations 11–18 use `1.0×F`,
and the logger exits at iteration 19.

- [ ] **Step 5: Analyze the full trace**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  tools/analyze_custom_schedule_delay_trace.py \
  --baseline-trace \
    results/custom_schedule_v100/validation_20260730/trace \
  --delayed-trace \
    runs/custom_schedule_v100/visualization_delay_20260730/C_visualization_only_delay/trace \
  --output-json \
    runs/custom_schedule_v100/visualization_delay_20260730/delay_summary.json \
  --output-csv \
    runs/custom_schedule_v100/visualization_delay_20260730/delay_source_data.csv
```

Acceptance:

```text
dependency violations = 0
remote dependencies per measured iteration = 7
median duration(0×) < median duration(0.5×) < median duration(1.0×)
```

- [ ] **Step 6: Generate and visually inspect the figure**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  tools/plot_custom_schedule_delay_trace.py \
  --summary \
    runs/custom_schedule_v100/visualization_delay_20260730/delay_summary.json \
  --source-data \
    runs/custom_schedule_v100/visualization_delay_20260730/delay_source_data.csv \
  --output-dir \
    runs/custom_schedule_v100/visualization_delay_20260730/figure
```

Inspect PNG at final width. Verify labels, arrows, and artificial-delay warning
do not overlap.

---

### Task 7: Package, Verify, Commit, and Push Results

**Files:**
- Create: `results/custom_schedule_v100/visualization_delay_20260730/README.md`
- Create: `results/custom_schedule_v100/visualization_delay_20260730/MANIFEST.sha256`
- Add: selected traces, metrics, figure, source data, run info, and logs.
- Modify: `docs/custom_pipeline_schedule.md`

**Interfaces:**
- Produces: self-contained result archive on branch `crosspipe`.

- [ ] **Step 1: Build the result package**

Include:

```text
README.md
MANIFEST.sha256
inputs/
metrics/delay_summary.json
metrics/delay_source_data.csv
trace/rank_0.jsonl ... rank_3.jsonl
figure/custom_schedule_delay_trace.svg
figure/custom_schedule_delay_trace.pdf
figure/custom_schedule_delay_trace.png
figure/figure_caption.txt
logs/run.info
logs/train.log
```

- [ ] **Step 2: Write exact README measurements**

Document:

- loaded custom torch path and version;
- profiled `T_F_stage`;
- converted `0.5×F` and `1.0×F` delays in milliseconds;
- iteration ranges;
- dependency check count and zero violations;
- communication-duration medians;
- visualization-only warning.

- [ ] **Step 3: Generate and verify the manifest**

```bash
cd results/custom_schedule_v100/visualization_delay_20260730
find . -type f ! -name MANIFEST.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

Expected: every file reports `OK`.

- [ ] **Step 4: Run the full relevant regression suite**

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python -m pytest \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_custom_schedule.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_delay.py \
  tests/unit_tests/pipeline_parallel/test_analyze_custom_schedule_delay_trace.py \
  tests/unit_tests/pipeline_parallel/test_compare_custom_schedule_runs.py -q
bash -n test_crossdc/custom_schedule_v100/run_custom_schedule.sh
git diff --check
```

- [ ] **Step 5: Commit source and results**

```bash
git add \
  megatron \
  tests/unit_tests/pipeline_parallel \
  tools \
  test_crossdc/custom_schedule_v100 \
  docs \
  results/custom_schedule_v100/visualization_delay_20260730
git add -f \
  results/custom_schedule_v100/visualization_delay_20260730/logs
git commit -m "results: add delayed custom schedule visualization"
```

- [ ] **Step 6: Push and verify the remote hash**

```bash
git push git@github.com:ssb233/Megatron-LM.git crosspipe
git ls-remote \
  git@github.com:ssb233/Megatron-LM.git \
  refs/heads/crosspipe
git status --short
```

Expected: the remote hash equals local `HEAD`, and the worktree is clean.
