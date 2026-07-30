# Custom Schedule Delay Visualization Design

Date: 2026-07-30
Status: Approved direction, pending final document review

## Goal

Produce a clearer publication trace for CrossPipe experiment C by using the
existing CrossPipe communication-delay injection to lengthen NCCL transfer
completion, while preserving the Magellan custom order and communication
dependencies.

The experiment must demonstrate:

1. the selected communications have visible durations in the timeline;
2. the dependency chain remains
   `F4 S1→S2 → B0 S3→S2 → F5 S1→S2`;
3. each target NCCL send is submitted only after its Gloo dependency signal;
4. artificial delay is clearly separated from the original A/B/C performance
   results.

## Non-goals

- Do not model real cross-DC networking.
- Do not add dummy tensors or change activation shapes.
- Do not use these runs for A/B/C throughput conclusions.
- Do not regenerate or optimize the Magellan static schedule.
- Do not add communication rate allocation or bandwidth sharing.

## Existing Mechanism

CrossPipe accepts:

```text
--cdc_latency_bandwidth_delay_as_F_stage latency_factor,bandwidth_factor
```

Both factors are multiplied by the profiled maximum forward-stage time:

```text
delay_seconds = factor × T_F_stage
```

The first tuple component is injected when a receiver waits for communication
and represents latency-like waiting. The second component is passed through the
custom NCCL P2P path as a bandwidth/transfer delay. This experiment therefore
uses the second component:

```text
0,0.5
0,1.0
```

The custom schedule path currently rejects this configuration in two validation
locations and returns from `update_schedule_with_latency_bandwidth()` before
updating the active delay. Those restrictions must be adjusted without allowing
the profiling logic to replace the external static schedule.

## Runtime Design

### Validation

The zero-delay behavior remains the default:

```text
num_dc=1
delay=0,0
```

A custom schedule may use non-zero injected delay only when:

```text
PP=4
num_dc=4
pp_stages_per_dc=[1,1,1,1]
```

This makes every adjacent pipeline-stage boundary an injected logical DC
boundary while all four ranks remain on the same physical host.

Unsupported combinations fail before distributed execution. In particular,
non-zero delay with `num_dc=1` must not silently run without injection.

### Schedule Update

`update_schedule_with_latency_bandwidth()` will:

1. read the active delay tuple for the current experiment interval;
2. convert both factors to seconds using the profiled `T_F_stage`;
3. update `injected_latency_delay` and `injected_bandwidth_delay`;
4. emit the delay configuration to logs and trace;
5. return early for a custom schedule before any schedule reconstruction.

The external `replay.order.json` remains authoritative. No TaskNode,
communication event, receive placement, or dependency is regenerated.

### Dependency Semantics

The dependency controller continues to wait for the trigger NCCL `Work` to
complete. The bandwidth delay therefore extends the trigger completion time
observed by the controller:

```text
target_submit(trigger)
  → injected transfer delay
  → comm_complete(trigger)
  → Gloo signal
  → signal_recv(target sender)
  → target_submit(target)
```

The signal mechanism itself is unchanged.

## Experiment Design

Use the existing single-node 4×V100 C configuration:

```text
PP=4, TP=1, DP=1
8 microbatches
one virtual chunk
recomputation disabled
custom order + communication dependency
```

Run one visualization-only trace job containing two delay intervals:

```bash
--num_dc 4
--pp_stages_per_dc 1 1 1 1
--cdc_latency_bandwidth_delay_as_F_stage 0,0.5 0,1.0
--cdc_exp_test_start_iter 3
--cdc_exp_per_cfg_test_iters 8
--custom-schedule-trace-dir \
  results/custom_schedule_v100/visualization_delay_20260730/trace
```

Expected interval mapping:

```text
profile iteration: 2
0.5× transfer-delay samples: iterations 3–10
1.0× transfer-delay samples: iterations 11–18
automatic exit: iteration 19
```

The existing zero-delay C_TRACE remains the baseline and is not rerun unless
required by a regression.

## Trace Design

Each iteration must identify its active configuration:

```text
latency_factor
bandwidth_factor
latency_seconds
bandwidth_seconds
profiled_T_F_stage
```

Required causal events remain:

```text
target_submit
comm_complete
signal_send_start
signal_recv
target_submit of dependent communication
```

The trace analysis will verify, for every remote dependency:

```text
trigger comm_complete
  < signal_send_start
  < signal_recv
  < target target_submit
```

For the publication chain it will additionally verify:

```text
Comm_F_4_1_2
  → dependency 6
  → Comm_B_0_3_2
  → dependency 1
  → Comm_F_5_1_2
```

## Figure Design

Backend: Python/Matplotlib only.

Output:

```text
editable SVG
PDF
600 dpi PNG
source-data CSV
caption
```

The figure is an asymmetric three-panel trace:

- **Panel a — communication-visible timeline:** representative `1.0×F`
  visualization-only iteration, with separate compute and P2P lanes for S0–S3.
- **Panel b — dependency zoom:** measured `F4 → dep6 → B0 → dep1 → F5`
  chain, showing NCCL completion, Gloo signal, and target submission.
- **Panel c — delay validation:** selected communication durations for the
  original `0×`, `0.5×`, and `1.0×` traces, demonstrating that visibility
  increases while dependency order remains unchanged.

The panel and caption must include:

```text
Artificial transfer delay for visualization only.
Not used for end-to-end performance comparison.
Host-side instrumentation; not GPU kernel duration.
```

The original zero-delay figure and data remain in the existing result archive.

## Tests

### Unit tests

Add tests covering:

1. zero-delay custom schedules retain `num_dc=1`;
2. non-zero custom delay with `num_dc=1` is rejected;
3. custom visualization delay accepts `num_dc=4` and
   `pp_stages_per_dc=[1,1,1,1]`;
4. custom delay updates delay fields without replacing the static schedule;
5. trace configuration metadata matches the active experiment interval.

### Four-GPU verification

The visualization run must satisfy:

1. both `0.5×` and `1.0×` intervals finish without timeout or deadlock;
2. all expected compute and communication operations are present;
3. all 14 dependencies remain ordered;
4. all 7 remote dependencies per iteration contain the expected Gloo events;
5. selected communication duration increases from `0×` to `0.5×` to `1.0×`;
6. the generated figure is readable at publication width.

## Risks and Controls

### Custom PyTorch dependency

The bandwidth-delay path uses the custom P2P interpretation already present in
CrossPipe. Before the full run, execute a short four-rank smoke test and confirm
that the server build honors the non-zero delay.

### Misinterpretation as real networking

All paths, output directories, captions, and README text use
`visualization_only` naming and state that the delay is artificial.

### Schedule mutation

The custom path returns immediately after updating delay state, before any
schedule/performance-model reconstruction. Tests compare the static schedule
identity before and after a configuration update.

### Trace ambiguity

Every traced iteration records the active factor and converted delay in seconds.
The plotting script refuses to combine iterations whose configuration metadata
is absent or inconsistent.

## Deliverables

```text
source changes and unit tests
visualization-only launch mode
0.5× and 1.0× raw traces
delay comparison metrics
updated SVG/PDF/PNG figure
source-data CSV and caption
result README and SHA-256 manifest
```

The new results will be stored separately from the original validation:

```text
results/custom_schedule_v100/visualization_delay_20260730/
```
