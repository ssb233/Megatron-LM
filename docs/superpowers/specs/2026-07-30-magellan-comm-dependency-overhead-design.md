# Magellan Communication-Dependency Overhead Experiment Design

Date: 2026-07-30

Status: Approved and authorized for execution

## 1. Objective

Measure the end-to-end iteration-time overhead of executing a Magellan custom
pipeline schedule with explicit communication dependencies, relative to a
1F1B schedule, on one node with four V100 32 GB GPUs.

The experiment is a framework and control-path evaluation. It does not claim
that a logical cross-DC topology changes the physical single-node NCCL path,
and it does not inject artificial communication latency, bandwidth limits, or
dummy payloads. The Magellan arm must load both the generated operation order
and its non-empty communication-dependency file so that Gloo control signals
are included in the measured iteration time.

## 2. Experiment Matrix

All configurations use PP=4, TP=1, DP=1, eight transformer layers, FP16, no
activation recomputation, and the same GPT-2 dataset and tokenizer already
configured on the server. D1 is the completed pilot and remains unchanged.
D2, M1, and M2 use 16 microbatches to reduce the PP=4 pipeline bubble and to
exercise a longer custom schedule with more communication dependencies.

| ID | Model | Hidden | FFN hidden | Heads | Sequence | Microbatch size | Microbatches | Global batch | MoE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D1 | Dense, existing result | 1024 | 4096 | 16 | 256 | 1 | 8 | 8 | disabled |
| D2 | Dense, larger model and payload | 1536 | 6144 | 24 | 512 | 2 | 16 | 32 | disabled |
| M1 | MoE, compute-heavy | 1024 | 4096 | 16 | 256 | 1 | 16 | 16 | 8 experts, top-2 |
| M2 | MoE, communication-richer | 768 | 3072 | 12 | 512 | 2 | 16 | 32 | 8 experts, top-2 |

D2 raises hidden size by 1.5x, sequence length by 2x, and microbatch size by 2x
relative to D1. Its inter-stage activation has approximately six times as many
elements as D1 while transformer compute also increases substantially.

M1 keeps a relatively large hidden size and shorter sequence to emphasize
expert compute. M2 uses a smaller hidden size but doubles sequence length and
microbatch size, increasing pipeline activation bytes relative to transformer
compute. Cross-configuration throughput is not compared because global batch
and model dimensions differ. Every reported comparison is between 1F1B and
Magellan under the exact same configuration.

Both MoE configurations use eight experts and top-2 routing. MoE uses
expert-model-parallel size 1 and an all-to-all token dispatcher. Experts are
local to each pipeline rank, so the experiment does not add inter-GPU
expert-parallel traffic. Increasing the expert count beyond eight would mainly
increase parameters and optimizer state without serving the pipeline
communication-dependency objective.

## 3. Magellan Solver Calibration

The completed D1 calibration, solve, and measured runs are retained. Each of
D2, M1, and M2 receives its own calibration and solve. Reusing a schedule
across configurations is forbidden because their measured
compute-to-communication ratios differ.

Run one uncounted 1F1B calibration job for each new configuration using the
same model arguments as the subsequent comparison. Read the CrossPipe profile
output and derive:

- `T_F_ref`: the median forward time of pipeline stages 1 and 2, which are the
  transformer-dominant middle stages;
- `T_comm_ref`: the median measured original-NCCL P2P transfer time over both
  directions of the three adjacent pipeline-stage pairs;
- `comm_units = T_comm_ref / T_F_ref`.

The solver uses the `0-1, 1-2, 1-3` logical star in
`Motivation/Topology/star_4/bw_300Gbps_lat_0.01s`. The topology determines
routes and shared directed-link exclusivity only. Physical propagation delay
is overridden to zero and communication duration is supplied by
`comm_units`.

Required solver arguments:

```text
--microbatches 16
--stages 4
--comm-units <measured T_comm_ref / T_F_ref>
--delay-units 0 0 0 0 0 0
--t-fwd-s <measured T_F_ref>
--r-save 1
--mem 8
--time-limit 600
--seed 0
--experiments opt_sim
--sat-link-exclusive
--sat-comm-split-k 1
--comm-round-decimals 4
```

Do not enable layer-count optimization, recomputation optimization, simulated
annealing, measured-communication refinement, or segmented rate allocation.

For each solve, retain:

- `replay.order.json`;
- the generated notification communication-dependency JSON;
- solver summary and status;
- the exact calibration values and solver command.

Before training, validate that the order contains every forward and backward
operation for the configuration's microbatch count, the dependency file is
non-empty, at least one dependency remains after removing each channel's
intrinsic serialization edges, every dependency endpoint exists in the order,
and the combined operation graph is acyclic.

## 4. Preflight Memory Gate

Before each D2, M1, or M2 solve, run the exact model configuration for four
1F1B iterations. The preflight must complete without OOM, NaN, skipped
iterations, or distributed timeout. Continue with the configured model when
peak allocated GPU memory is below 28 GiB, leaving at least 4 GiB for runtime
variation.

Use these deterministic fallbacks only when the preflight exceeds the gate or
OOMs:

1. D2: reduce microbatch size from 2 to 1 and global batch from 32 to 16,
   preserving 16 microbatches.
2. M1: reduce hidden/FFN/heads from 1024/4096/16 to 768/3072/12, preserving
   eight experts and top-2 routing.
3. M2: reduce microbatch size from 2 to 1 and global batch from 32 to 16,
   preserving 16 microbatches.

Activation recomputation must remain disabled and is not a fallback.

## 5. Compared Runtime Arms

Each configuration has two runtime arms:

1. **1F1B:** CrossPipe's static 1F1B adapter, without a custom order or
   communication-dependency file.
2. **Magellan+dependency:** the configuration-specific `replay.order.json`
   plus its configuration-specific communication-dependency JSON.

The Magellan arm must initialize the Gloo CPU process group and execute the
dependency controller. A target communication may be submitted only after all
of its declared trigger communications have completed and their control
messages have been received.

Both arms use the same Python environment, PyTorch build, GPU/rank mapping,
dataset, optimizer, seed, logging level, and model arguments. Runtime delay
injection is disabled:

```text
num_dc=1
cdc_latency_bandwidth_delay_as_F_stage=0,0
```

Nsight Systems, JSONL custom-schedule tracing, per-event flushing, and synthetic
communication payloads are disabled during performance runs.

## 6. Repetition and Timing

For every configuration and runtime arm:

- launch one training process;
- run 20 iterations;
- treat iterations 1-5 as warm-up;
- measure iterations 6-19, giving 14 samples per configuration/arm;
- exclude iteration 20 because CrossPipe performs experiment-finalization work
  in that iteration, which produced a visible outlier in the completed D1
  baseline;
- use Megatron's end-to-end `elapsed time per iteration (ms)` value;
- preserve the raw log and exact command.

The primary statistic is the median over the 14 steady-state samples. Also
report the arithmetic mean, sample standard deviation, minimum, and maximum.
These are single-run functional and overhead results, not statistically
replicated benchmark claims.

For each configuration, compute:

```text
relative_change_percent =
    (median_magellan_ms - median_1f1b_ms) / median_1f1b_ms * 100
```

A positive value is Magellan communication-dependency overhead; a negative
value is an observed speedup. Results are reported without imposing a
preselected pass/fail overhead threshold.

## 7. Result Layout

Store all committed artifacts under:

```text
results/magellan_comm_dependency_overhead_pp4/
```

Use one directory per configuration:

```text
D1_dense_compute_heavy/
D2_dense_communication_richer/
M1_moe_compute_heavy/
M2_moe_communication_richer/
```

Each directory contains `calibration/`, `solver/`, `1f1b/`, and
`magellan_dependency/`. The result root contains:

- `summary.csv`: one row per configuration and runtime arm;
- `samples.csv`: one row per measured iteration;
- `summary.json`: machine-readable configuration, calibration, and statistics;
- `README.md`: commands, environment, methodology, dependency validation, and
  interpretation.

Large checkpoints, data caches, TensorBoard event files, and profiler reports
are not committed. Raw text training logs, solver JSON files, compact metadata,
and summary files are committed.

## 8. Acceptance Checks

The experiment is complete only when:

1. the retained D1 artifacts remain valid, and each of the three new
   calibration jobs produces finite `T_F_ref`, `T_comm_ref`, and positive
   `comm_units`;
2. the three new CP-SAT solves return a feasible or optimal solution;
3. all four communication-dependency files contain at least one extra edge and
   pass endpoint and cycle validation;
4. all six new measured training processes complete without deadlock, OOM,
   NaN, skipped iteration, or distributed timeout;
5. every measured arm contributes exactly 14 iterations from 6 through 19;
6. logs confirm zero runtime communication-delay injection;
7. logs confirm that the Magellan arm loaded both its custom order and
   communication dependencies;
8. the summary statistics can be regenerated from committed raw logs by one
   documented command.
