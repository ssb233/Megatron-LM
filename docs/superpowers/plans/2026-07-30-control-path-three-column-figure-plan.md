# Three-Column Control-Path Latency Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a publication-ready three-column control-path latency figure separating sender dispatch, Gloo transfer, and receiver dispatch overhead.

**Architecture:** Extend the archived figure script with one pure data-extraction function that joins sender `signal_send_start` events to trigger `comm_complete` events and merges them with the existing validated Gloo/receiver metrics. Keep plotting and CSV export as consumers of the same sample records, then regenerate both the complete multi-panel figure and a standalone latency panel.

**Tech Stack:** Python 3.12, matplotlib, NumPy, pytest, archived CrossPipe JSONL traces.

## Global Constraints

- Use formal iterations 3-10 from the archived `C_TRACE` run.
- Require exactly 56 raw dependency samples.
- Plot sender dispatch samples at or below 600 us only; exactly one 1.19 ms sample is excluded.
- Preserve the excluded sample in CSV with `included_in_plot=false`.
- Preserve the existing Gloo `ready -> recv` and receiver `recv -> submit` definitions.
- Export editable SVG, vector PDF, and 300-DPI-or-higher PNG.
- Do not rerun training or modify runtime scheduler code.

---

### Task 1: Extract and validate three-stage latency samples

**Files:**
- Modify: `results/custom_schedule_v100/validation_20260730/figure/plot_custom_schedule_trace.py`
- Create: `tests/custom_schedule_tools/test_control_path_latency_figure.py`

**Interfaces:**
- Consumes: raw trace rows from `load_rows()` and the existing `signal_ready_latency.json` payload.
- Produces: `build_control_latency_samples(rows, payload, sender_cutoff_us=600.0) -> list[dict]`.

- [ ] **Step 1: Write the failing extraction test**

Load the real archived plotting module with `importlib.util`, call the wished-for function, and assert literal expected results:

```python
samples = module.build_control_latency_samples(
    module.load_rows(),
    json.loads(module.SIGNAL_DATA_PATH.read_text()),
)
assert len(samples) == 56
assert sum(row["sender_included_in_plot"] for row in samples) == 55
assert np.median([
    row["sender_complete_to_send_us"]
    for row in samples
    if row["sender_included_in_plot"]
]) == pytest.approx(183.266)
assert np.median(
    [row["gloo_ready_to_recv_us"] for row in samples]
) == pytest.approx(225.405)
assert np.median(
    [row["receiver_recv_to_submit_us"] for row in samples]
) == pytest.approx(345.858)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  -m pytest tests/custom_schedule_tools/test_control_path_latency_figure.py -q
```

Expected: fail because `build_control_latency_samples` does not exist.

- [ ] **Step 3: Implement minimal extraction and validation**

Add a function that:

```python
def build_control_latency_samples(
    rows: list[dict],
    payload: dict,
    sender_cutoff_us: float = 600.0,
) -> list[dict]:
    # Index formal payload rows by (iteration, dependency_id).
    # Match each signal_send_start to exactly one trigger comm_complete.
    # Compute sender_complete_to_send_us.
    # Copy the existing Gloo and receiver metrics.
    # Set sender_included_in_plot using the cutoff.
    # Validate 56 keys and timestamp ordering.
```

Every output row contains:

```python
{
    "iteration": int,
    "dependency_id": int,
    "sender_complete_to_send_us": float,
    "gloo_ready_to_recv_us": float,
    "receiver_recv_to_submit_us": float,
    "sender_included_in_plot": bool,
}
```

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  -m pytest tests/custom_schedule_tools/test_control_path_latency_figure.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit extraction**

```bash
git add \
  results/custom_schedule_v100/validation_20260730/figure/plot_custom_schedule_trace.py \
  tests/custom_schedule_tools/test_control_path_latency_figure.py
git commit -m "test: validate three-stage control latency samples"
```

### Task 2: Render three-column panel and standalone artifacts

**Files:**
- Modify: `results/custom_schedule_v100/validation_20260730/figure/plot_custom_schedule_trace.py`
- Modify: `tests/custom_schedule_tools/test_control_path_latency_figure.py`
- Regenerate: `results/custom_schedule_v100/validation_20260730/figure/custom_schedule_trace.{svg,pdf,png}`
- Create: `results/custom_schedule_v100/validation_20260730/figure/control_path_latency_3stage.{svg,pdf,png}`
- Replace: `results/custom_schedule_v100/validation_20260730/figure/source_data_signal_latency.csv`

**Interfaces:**
- Consumes: sample records from `build_control_latency_samples`.
- Produces: `draw_latency(ax, samples)`, `write_control_latency_source(samples, path)`, and `create_standalone_latency_figure(samples)`.

- [ ] **Step 1: Add failing artifact test**

Call the script in a subprocess, then assert:

```python
for suffix in (".svg", ".pdf", ".png"):
    assert (figure_dir / f"control_path_latency_3stage{suffix}").stat().st_size > 0
rows = list(csv.DictReader(
    (figure_dir / "source_data_signal_latency.csv").open()
))
assert len(rows) == 56
assert sum(row["sender_included_in_plot"] == "true" for row in rows) == 55
```

- [ ] **Step 2: Run the test and verify RED**

Expected: fail because the standalone files and new CSV schema do not exist.

- [ ] **Step 3: Implement publication plotting**

Update `draw_latency` to use three columns:

```text
sender
complete->send

Gloo
ready->recv

receiver
recv->submit
```

Use blue/teal, purple, and orange fills; boxplots with jittered raw points;
direct median labels; 0-600 us y-axis; and an annotation:
`sender n = 55; Gloo/receiver n = 56; iterations 3-10`.

Add a standalone `(4.4, 4.0)`-inch figure and save SVG, PDF, and 600-DPI PNG.
Write all 56 rows to CSV, including the excluded sender row with
`sender_included_in_plot=false`.

- [ ] **Step 4: Regenerate and run tests**

Run:

```bash
cd results/custom_schedule_v100/validation_20260730/figure
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  plot_custom_schedule_trace.py
cd /home/songxb26/mnist/crosspipi-magellan
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  -m pytest tests/custom_schedule_tools/test_control_path_latency_figure.py -q
```

Expected: generated artifacts exist and tests pass.

- [ ] **Step 5: Commit rendering**

```bash
git add results/custom_schedule_v100/validation_20260730/figure \
  tests/custom_schedule_tools/test_control_path_latency_figure.py
git commit -m "fig: add three-stage control latency distribution"
```

### Task 3: Document, visually inspect, and verify outputs

**Files:**
- Modify: `results/custom_schedule_v100/validation_20260730/README.md`
- Modify: `results/custom_schedule_v100/validation_20260730/figure/figure_caption.txt`
- Modify: `results/custom_schedule_v100/validation_20260730/MANIFEST.sha256`

**Interfaces:**
- Consumes: generated figures and source CSV.
- Produces: documented timing definitions and checksummed archive.

- [ ] **Step 1: Update documentation**

Document the three exact equations, sample counts, filtered sender tail policy,
and CPU host-side timing limitation. Update the caption so panel c describes
all three distributions.

- [ ] **Step 2: Rebuild the manifest**

Generate SHA-256 entries for every archived file except `MANIFEST.sha256`,
using stable sorted relative paths.

- [ ] **Step 3: Run complete verification**

Run:

```bash
/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python \
  -m pytest tests/custom_schedule_tools/test_control_path_latency_figure.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py -q
git diff --check
```

Parse every JSON/CSV, confirm medians and sample counts, confirm SVG/PDF/PNG
are non-empty, and verify all manifest hashes.

- [ ] **Step 4: Visually inspect PNG**

Copy the standalone PNG locally and inspect it at original resolution. Check
that all labels, median annotations, points, and boxplots are legible without
overlap and that no 1.19 ms point or annotation appears.

- [ ] **Step 5: Commit and push**

```bash
git add results/custom_schedule_v100/validation_20260730 \
  tests/custom_schedule_tools/test_control_path_latency_figure.py
git commit -m "docs: archive three-stage control latency figure"
git push origin crosspipe
```
