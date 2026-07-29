# Custom Pipeline Schedule and Communication Dependency Design

## 1. Scope

This change adds externally supplied one-chunk pipeline schedules and explicit
cross-channel communication dependencies to CrossPipe.

The initial supported deployment is:

- one physical node;
- four V100 GPUs;
- pipeline parallel size 4;
- tensor parallel size 1;
- data parallel size 1;
- one model chunk;
- training mode with activation recomputation disabled;
- no injected communication latency or bandwidth delay.

The implementation consumes Magellan output without converting it offline:

- `replay.order.json` supplies per-stage compute order and per-channel
  communication order;
- `replay.notification.no_competition.notification_deps.json` (or any file
  with the same schema) supplies extra cross-channel communication
  dependencies.

The feature is enabled when `--custom-pipeline-schedule` is provided. The
dependency file is optional, but cannot be supplied without a schedule.

## 2. Command-Line Interface

Add these arguments in `megatron/training/arguments.py`:

```text
--custom-pipeline-schedule <path>
--custom-comm-dependency <path>
--custom-schedule-trace-dir <path>
```

`--custom-pipeline-schedule` and the existing `--dynamic_schedule` are
mutually exclusive. A custom schedule selects CrossPipe static scheduling and
does not require `--static_schedule`. Supplying both a custom schedule and
`--static_schedule` is rejected to avoid ambiguous precedence.

`--custom-comm-dependency` requires `--custom-pipeline-schedule`.
`--custom-schedule-trace-dir` enables per-rank structured trace output for
custom scheduling.

## 3. Input Formats

### 3.1 Schedule

The schedule loader accepts Magellan `replay.order.json` version 1:

```json
{
  "version": 1,
  "compute": {
    "stage_0": ["F_0_0", "F_1_0", "B_0_0", "B_1_0"],
    "stage_1": ["F_0_1", "F_1_1", "B_0_1", "B_1_1"]
  },
  "comm": {
    "F_0_1": ["Comm_F_0_0_1", "Comm_F_1_0_1"],
    "B_1_0": ["Comm_B_0_1_0", "Comm_B_1_1_0"]
  }
}
```

Operation identifiers are parsed into structured values:

```text
Compute: F_<microbatch>_<stage> | B_<microbatch>_<stage>
Comm:    Comm_F_<microbatch>_<src_stage>_<dst_stage>
         Comm_B_<microbatch>_<src_stage>_<dst_stage>
```

Only chunk 0 is supported. Every stage must contain each expected forward and
backward compute operation exactly once. Every adjacent directed stage pair
must contain each expected communication operation exactly once.

### 3.2 Extra communication dependencies

The dependency loader accepts Magellan's notification dependency report:

```json
{
  "edge_count": 2,
  "edges": [
    {
      "reason": "insert_notify_delay_op",
      "from_op": "Comm_F_0_1_2",
      "to_op": "Notify_dcdir_1_3_...",
      "trigger_comm": "Comm_F_0_1_2",
      "target_comm": "Comm_F_1_2_3"
    },
    {
      "reason": "insert_notify_delay_op",
      "from_op": "Notify_dcdir_1_3_...",
      "to_op": "Comm_F_1_2_3",
      "trigger_comm": "Comm_F_0_1_2",
      "target_comm": "Comm_F_1_2_3"
    }
  ]
}
```

Edges that share `trigger_comm` and `target_comm` are collapsed into one
runtime dependency:

```text
trigger_comm completion -> CPU signal -> target_comm submission
```

Direct communication-to-communication edges emitted with
`serialize_same_src_dc_on_directed_link` are accepted through `from_op` and
`to_op`.

The normalized dependency has:

```text
dependency_id
trigger CommOpId
target CommOpId
reason
directed_link (optional)
```

Duplicate normalized dependencies are rejected.

## 4. Static Schedule Integration

Introduce `CustomOneChunkPipeline`, based on the existing
`OneChunkPipelineTemplate`.

For every `compute.stage_N` array, the loader creates `TaskNode` objects in
the exact supplied order. Consecutive tasks on a stage are linked through
`prev_device_task`. CrossPipe's existing `_resolve_batch_dependency()` then
adds the one-chunk microbatch path:

```text
F_m_0 -> F_m_1 -> F_m_2 -> F_m_3
F_m_3 -> B_m_3
B_m_3 -> B_m_2 -> B_m_1 -> B_m_0
```

The resulting pipeline is passed to the existing `ExecutionPlanner`, which
continues to create compute, local-copy, POST send/recv, and WAIT recv events.
Custom scheduling therefore remains inside CrossPipe's static scheduling
path.

The communication arrays are not metadata. They determine:

- sender submission order for every forward/backward channel;
- receiver pre-post order for the matching channel;
- local completion dependencies between adjacent operations in that channel.

The planner uses canonical `CommOpId` values rather than parsing strings at
runtime.

## 5. Dependency Validation

Before distributed training events begin, construct a directed graph
containing:

- per-stage compute-order edges;
- intrinsic one-chunk microbatch edges;
- compute-to-communication edges;
- communication-to-downstream-compute edges;
- per-channel communication-order edges;
- normalized extra communication dependencies.

Validation rejects:

- unsupported JSON version;
- missing or extra stages;
- PP size or microbatch count mismatch;
- malformed operation names;
- operations assigned to the wrong stage or channel;
- duplicate or missing operations;
- non-adjacent pipeline communication;
- chunk IDs other than zero;
- dependency references to unknown communication operations;
- self dependencies;
- duplicate extra dependencies;
- cycles in the combined graph.

All four ranks normalize the input and compare a canonical SHA256 digest
before executing the schedule. A mismatch raises an error before NCCL or
Gloo control traffic is issued.

## 6. CPU Signal Runtime

### 6.1 Control group

Create a dedicated Gloo process group for the four pipeline ranks. The group
is initialized collectively with the other model-parallel groups and exposed
through `parallel_state`. It is destroyed with the other process groups.

The initial implementation requires world size 4, PP 4, TP 1, and DP 1 when
extra communication dependencies are enabled.

### 6.2 Sender-side gate

Only the target communication sender waits for a signal. The target receiver
posts its NCCL `irecv` in the schedule's communication order without waiting
for the signal.

For a dependency `A -> B`:

1. A's sender submits the NCCL `isend` and stores its `Work`.
2. A background control worker waits for A's `Work` to complete.
3. The worker records `comm_complete` and sends a CPU token through Gloo.
4. B's sender reaches its POST send event and blocks on Gloo receive.
5. After receiving the token, B's sender records `signal_receive` and submits
   the NCCL `isend`.

The main scheduling thread on A is not blocked by A's communication
completion. GPU work already submitted by A can continue while the control
worker waits.

If one trigger has multiple targets, one worker waits once and sends one
token per target dependency. If one target has multiple predecessors, its
sender waits for all predecessor tokens before submission.

The runtime joins all control workers at iteration cleanup. Missing signals,
unknown work handles, duplicate sends, and duplicate receives are hard
errors.

### 6.3 Deadlock constraints

The following invariants are mandatory:

- receivers never wait for Gloo signals before posting NCCL receives;
- the trigger signal is sent only after trigger NCCL completion;
- target NCCL send is not posted before every required signal arrives;
- every extra dependency is known on all ranks before execution;
- no process creates Gloo groups lazily or in a rank-dependent order;
- combined dependency graph acyclicity is checked before training.

## 7. Trace Design

When `--custom-schedule-trace-dir` is supplied, each rank writes one JSONL
file using `time.perf_counter_ns()`. Since the supported deployment is one
node, timestamps are comparable across the four processes.

Each record includes:

```text
timestamp_ns
global_rank
pipeline_rank
iteration
event
operation
kind
direction
microbatch
chunk
src_stage
dst_stage
dependency_id (when applicable)
```

Events include:

```text
compute_start
compute_end
comm_recv_post
comm_send_post
comm_wait_start
comm_complete
signal_wait_start
signal_send_start
signal_send_end
signal_receive
target_comm_submit
```

The trace writer flushes at iteration boundaries and closes during scheduler
cleanup. Tracing is disabled by default.

An offline merger converts the four JSONL files into Chrome Trace JSON:

- one compute lane per PP rank;
- one communication lane per PP rank;
- one control-signal lane per PP rank;
- duration blocks for compute, communication, and signal waits;
- flow arrows from trigger communication through signal to target
  communication;
- labels for F/B, microbatch, source/destination stage, and dependency ID.

The merged trace opens in Perfetto or TensorBoard's trace viewer.

For every dependency:

```text
signal_latency = signal_receive - signal_send_start
gate_wait       = signal_receive - signal_wait_start
submit_delay    = target_comm_submit - signal_receive
```

## 8. End-to-End Comparison

Run three configurations with no injected communication latency or bandwidth
delay:

```text
A: existing static 1F1B
B: custom replay.order.json without an extra dependency file
C: the same custom replay.order.json with the dependency file
```

Interpretation:

```text
C - A: complete custom mechanism versus default 1F1B
B - A: schedule-order effect
C - B: extra serialization plus Gloo signal overhead
```

The comparison report includes median, minimum, maximum, and p95 iteration
time after excluding warmup/profile iterations. It also reports dependency
count, signal latency distribution, gate-wait distribution, and target
submission delay.

## 9. Error Handling

Configuration and schema errors use `ValueError` with the file path and
offending operation. Distributed digest mismatch and control protocol
violations use `RuntimeError` with rank and dependency ID.

Control waits support a configurable internal timeout derived from Megatron's
distributed timeout. A timeout reports:

- waiting rank;
- dependency ID;
- trigger and target operation;
- expected signal source rank;
- whether the trigger Work was registered and completed.

No failure silently falls back to 1F1B.

## 10. Verification Strategy

The Windows editing environment will not execute training or tests.
Implementation will include tests intended for the Linux four-GPU host:

- pure parser tests for valid and malformed Magellan files;
- operation completeness and stage/channel validation tests;
- combined dependency cycle tests;
- dependency normalization tests for Notify and direct edges;
- planner tests proving custom send/recv ordering;
- a fake-control-runtime test for one-to-one, one-to-many, and
  many-to-one dependencies;
- trace merger tests;
- four-rank integration commands for A/B/C experiments.

Static review in Windows will check imports, signatures, call sites, process
group lifecycle, and the correspondence between operation IDs, event keys,
and trace labels. No runtime-success claim is made until the Linux
four-V100 experiment completes.
