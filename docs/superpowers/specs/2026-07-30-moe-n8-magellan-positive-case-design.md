# MoE N=8 Magellan Positive-Case Experiment Design

## Objective

Keep the completed D1 result unchanged and find one PP=4 MoE configuration
where a Magellan custom schedule with mandatory communication dependencies is
no slower than CrossPipe 1F1B on a single node with four V100-32GB GPUs.

This is a targeted follow-up experiment, not an unrestricted parameter search.
The result must retain at least one extra communication dependency and at least
one cross-rank Gloo control signal.

## Evidence and Working Hypothesis

The previous D2, M1, and M2 experiments did use independent runtime profiles:

- D2 `comm_units = 0.054135`
- M1 `comm_units = 0.011597`
- M2 `comm_units = 0.012225`

Despite these different ratios, all three N=16 CP-SAT solves produced the same
canonical schedule. Each schedule contained 46 extra dependencies and 21
remote Gloo dependencies, compared with 6 extra and 3 remote dependencies for
the N=8 D1 schedule.

The working hypothesis is that N=16 increases CPU-side NCCL completion waits,
Gloo sends/receives, signal threads, and dependency-induced serialization
enough to dominate the custom schedule benefit on a single-node fabric that
does not physically reproduce the solver's shared star-link bottleneck. N=8
both reduces the control-plane dependency count and exposes a larger pipeline
bubble that a custom compute order can potentially reduce.

## Primary Candidate

The first candidate, named `M3`, uses:

| Parameter | Value |
|---|---:|
| Transformer layers | 8 |
| Pipeline parallelism | 4 |
| Tensor/data parallelism | 1 / 1 |
| Hidden size | 1024 |
| FFN hidden size | 4096 |
| Attention heads | 16 |
| Sequence length | 256 |
| Micro-batch size | 1 |
| Global batch size / microbatches | 8 |
| Experts | 8 |
| Router top-k | 2 |
| Expert parallelism | 1 |
| Precision | FP16 |
| Activation recomputation | disabled |
| Communication injection | disabled |

The logical solver topology remains the star `0-1, 1-2, 1-3`. Runtime remains
single-node original NCCL with `num_dc=1` and delay factors `0,0`.

## Experimental Flow

1. Run a four-iteration 1F1B preflight and collect a fresh CrossPipe profile.
   Reject OOM, skipped iterations, NaNs, or peak allocation at or above 28 GiB.
2. Derive M3-specific forward-compute and adjacent-P2P communication times.
3. Solve a new N=8 star-topology schedule with link exclusivity, no rate
   allocation, `r_save=1`, and zero propagation-delay units.
4. Validate complete 112-operation coverage, uniqueness, DAG acyclicity, at
   least one extra dependency, and at least one remote Gloo dependency through
   both the independent validator and CrossPipe's authoritative loader.
5. Screening run: one 20-iteration 1F1B arm and one 20-iteration Magellan arm.
   Use iterations 6 through 19 as the 14 measured samples.
6. If the screening median favors Magellan, run three additional arms so that
   the final comparison contains two runs of each execution order:
   `1F1B→Magellan` and `Magellan→1F1B`.
7. Accept M3 only if all runs complete without OOM/deadlock/skip/NaN and the
   aggregate Magellan median is no greater than the aggregate 1F1B median.

## BSH Fallback

If M3 is slower in screening, perform only one fallback candidate, `M4`, with
the same N=8 MoE configuration but sequence length 512. Increasing sequence
length raises attention computation faster than pipeline activation bytes and
avoids increasing expert parameter memory. M4 receives its own preflight,
profile, solve, validation, and screening; M3's calibration or schedule must
not be reused.

If neither M3 nor M4 meets the criterion, stop and report that the current
single-node dependency implementation has not produced a verified positive
MoE case. Do not continue parameter fishing.

## Result and Reproducibility Requirements

Store compact calibration, solver, validation, raw training logs, per-run
statistics, aggregate CSV/JSON, and a README under
`results/magellan_comm_dependency_overhead_pp4/`. Document that B, S, and H
change both model efficiency and the compute/communication ratio, and that
single-node physical links do not reproduce the logical star bottleneck.

The final result must distinguish screening evidence from confirmed evidence.
Only the balanced repeated comparison may be described as a stable improvement.
