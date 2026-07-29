# Custom Pipeline Communication Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load Magellan one-chunk schedules and communication dependencies into CrossPipe, enforce cross-channel ordering with a four-rank Gloo control group, and emit mergeable execution traces.

**Architecture:** A pure-Python loader normalizes and validates Magellan JSON into typed operation IDs and a combined dependency DAG. A custom one-chunk `Pipeline` feeds the existing `ExecutionPlanner`; planner metadata drives ordered receive posting and sender-side communication gates. A dedicated Gloo runtime waits for completed NCCL works in background workers and releases target sends, while a JSONL tracer records compute, NCCL, and signal events.

**Tech Stack:** Python 3, PyTorch distributed (NCCL and Gloo), CrossPipe static scheduler, JSON/JSONL, Chrome Trace JSON, pytest.

## Global Constraints

- Initial runtime support is exactly one node, four GPUs, PP=4, TP=1, DP=1, and one model chunk.
- Consume Magellan `replay.order.json` version 1 without offline conversion.
- Consume Magellan `replay.notification.no_competition.notification_deps.json` schema.
- Activation recomputation must remain disabled for the target experiment.
- Do not inject communication latency or bandwidth delay in the comparison experiment.
- Do not run CrossPipe tests or training in the Windows editing environment; author tests and Linux commands only.
- Preserve existing 1F1B, GPipe, Interleaved1F1B, ZBH1, ZBV, and dynamic scheduler behavior when custom arguments are absent.

---

### Task 1: Typed Magellan Input Loader and DAG Validation

**Files:**
- Create: `megatron/core/pipeline_parallel/cdc_scheduler/custom_schedule.py`
- Create: `tests/unit_tests/pipeline_parallel/test_custom_schedule.py`
- Create: `tests/unit_tests/pipeline_parallel/fixtures/replay_order_pp4_n4.json`
- Create: `tests/unit_tests/pipeline_parallel/fixtures/notification_deps_pp4_n4.json`

**Interfaces:**
- Produces: `ComputeOpId`, `CommOpId`, `CommDependency`, `CustomScheduleSpec`
- Produces: `load_custom_schedule(schedule_path, dependency_path, pp_size, num_microbatches) -> CustomScheduleSpec`
- Produces: `CustomScheduleSpec.canonical_sha256: str`
- Produces: `CustomScheduleSpec.local_predecessors: Dict[CommOpId, Tuple[CommOpId, ...]]`
- Produces: `CustomScheduleSpec.remote_predecessors: Dict[CommOpId, Tuple[CommDependency, ...]]`

- [ ] **Step 1: Add parser fixtures copied from the validated Magellan star topology output**

The schedule fixture contains all PP=4, N=4 compute and communication operations. The dependency fixture contains:

```json
{
  "edge_count": 3,
  "edges": [
    {
      "reason": "insert_notify_delay_op",
      "from_op": "Comm_F_3_1_2",
      "to_op": "Notify_dcdir_1_2_1_to_3_fixture",
      "trigger_comm": "Comm_F_3_1_2",
      "target_comm": "Comm_B_0_3_2"
    },
    {
      "reason": "insert_notify_delay_op",
      "from_op": "Notify_dcdir_1_2_1_to_3_fixture",
      "to_op": "Comm_B_0_3_2",
      "trigger_comm": "Comm_F_3_1_2",
      "target_comm": "Comm_B_0_3_2"
    },
    {
      "reason": "serialize_same_src_dc_on_directed_link",
      "from_op": "Comm_F_3_2_3",
      "to_op": "Comm_B_0_2_1",
      "src_dc": 2
    }
  ]
}
```

- [ ] **Step 2: Write parser and normalization tests**

Cover:

```python
def test_loads_magellan_schedule_and_collapses_notify_edges(tmp_path):
    spec = load_custom_schedule(order_path, deps_path, pp_size=4, num_microbatches=4)
    target = CommOpId("B", 0, 3, 2)
    assert spec.remote_predecessors[target][0].trigger == CommOpId("F", 3, 1, 2)

def test_direct_same_source_edge_becomes_local_dependency(tmp_path):
    spec = load_custom_schedule(order_path, deps_path, pp_size=4, num_microbatches=4)
    target = CommOpId("B", 0, 2, 1)
    assert CommOpId("F", 3, 2, 3) in spec.local_predecessors[target]

def test_rejects_cycle_across_compute_and_communication_edges(tmp_path):
    with pytest.raises(ValueError, match="cycle"):
        load_custom_schedule(cyclic_order, cyclic_deps, pp_size=4, num_microbatches=4)
```

Also test malformed names, duplicates, missing stages, missing operations, wrong stage/channel, unsupported version, dependency-only configuration, and PP/N mismatch.

- [ ] **Step 3: Implement immutable operation IDs**

Use frozen dataclasses with strict regular expressions:

```python
@dataclass(frozen=True, order=True)
class CommOpId:
    direction: str
    microbatch: int
    src_stage: int
    dst_stage: int

    @property
    def name(self) -> str:
        return f"Comm_{self.direction}_{self.microbatch}_{self.src_stage}_{self.dst_stage}"
```

- [ ] **Step 4: Implement schedule and dependency normalization**

Collapse two Notify edges through `trigger_comm` and `target_comm`. Classify a normalized dependency as local when both communication sources are the same stage; otherwise classify it as remote and assign a stable integer `dependency_id` from canonical sort order.

Per-channel predecessor edges from `replay.order.json` are included in the graph. Same-channel dependency-file edges that duplicate those edges are accepted and deduplicated.

- [ ] **Step 5: Implement complete graph validation and canonical digest**

Build adjacency over compute and communication operation names, run Kahn topological sort, and include the normalized JSON serialization in SHA256:

```python
canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 6: Leave Linux test commands in the test module docstring**

```bash
pytest tests/unit_tests/pipeline_parallel/test_custom_schedule.py -v
```

Expected on Linux: all parser and DAG tests pass.

- [ ] **Step 7: Commit**

```bash
git add megatron/core/pipeline_parallel/cdc_scheduler/custom_schedule.py tests/unit_tests/pipeline_parallel
git commit -m "feat: parse custom pipeline schedules"
```

---

### Task 2: Custom One-Chunk Static Pipeline

**Files:**
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_generator/pipeline.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_generator/__init__.py`
- Modify: `tests/unit_tests/pipeline_parallel/test_custom_schedule.py`

**Interfaces:**
- Consumes: `CustomScheduleSpec.compute_order`
- Produces: `CustomOneChunkPipeline(SystemConfig, CustomScheduleSpec)`
- Produces: `get_custom_static_schedule(spec, num_devices, num_microbatches) -> CustomOneChunkPipeline`
- Exposes: `Pipeline.custom_schedule_spec: Optional[CustomScheduleSpec]`

- [ ] **Step 1: Write static-pipeline construction tests**

Assert that every device task list exactly follows `compute.stage_N`, every consecutive pair is linked by `prev_device_task`, and `_resolve_batch_dependency()` produces the standard one-chunk F-forward/B-reverse microbatch chain.

- [ ] **Step 2: Add optional custom metadata to `Pipeline`**

Initialize:

```python
self.custom_schedule_spec = None
```

Existing pipelines retain `None`.

- [ ] **Step 3: Implement `CustomOneChunkPipeline`**

Subclass `OneChunkPipelineTemplate`. Its `schedule()` creates `TaskNode` objects from typed compute IDs, chains device order, calls `_resolve_batch_dependency()`, and sets `pipeline_name()` to `"Custom"`.

- [ ] **Step 4: Implement the custom factory**

Use `T_F=20`, `T_B=40`, `T_W=0`, and zero communication timing, matching existing one-chunk static defaults. Call `schedule()` and `solve_dependencies()` before returning.

- [ ] **Step 5: Export the factory**

Add `CustomOneChunkPipeline` and `get_custom_static_schedule` to `pp_generator/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add megatron/core/pipeline_parallel/cdc_scheduler/pp_generator tests/unit_tests/pipeline_parallel/test_custom_schedule.py
git commit -m "feat: build custom one-chunk pipelines"
```

---

### Task 3: Arguments and Static Scheduler Selection

**Files:**
- Modify: `megatron/training/arguments.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`
- Modify: `tests/unit_tests/pipeline_parallel/test_custom_schedule.py`

**Interfaces:**
- Consumes: `load_custom_schedule(...)`
- Consumes: `get_custom_static_schedule(...)`
- Produces CLI attributes: `custom_pipeline_schedule`, `custom_comm_dependency`, `custom_schedule_trace_dir`
- Produces scheduler state: `self.custom_schedule_spec`

- [ ] **Step 1: Add argument validation tests**

Cover dependency without schedule, custom plus dynamic, custom plus static, PP/TP/DP mismatch, virtual pipeline size other than 1, and recomputation enabled.

- [ ] **Step 2: Add the three arguments**

```python
group.add_argument('--custom-pipeline-schedule', type=str, default=None)
group.add_argument('--custom-comm-dependency', type=str, default=None)
group.add_argument('--custom-schedule-trace-dir', type=str, default=None)
```

- [ ] **Step 3: Select custom static scheduling in `CDCPPScheduler.__init__`**

Load and validate both files before constructing `ExperimentManager`. Set:

```python
self.use_static_schedule = True
self.use_custom_schedule = True
self.custom_schedule_spec = spec
self.pp_schedule = get_custom_static_schedule(spec, pp_size, num_microbatch)
```

When no custom path is supplied, retain existing static/dynamic branches unchanged.

- [ ] **Step 4: Extend `validate_args()`**

Require PP=4, TP=1, DP=1, `num_subparts == 1`, virtual pipeline size 1, no recomputation, no dynamic/static schedule selector, and no latency/bandwidth injection when custom communication dependencies are enabled.

- [ ] **Step 5: Prevent profile-driven schedule replacement**

`update_schedule_with_latency_bandwidth()` must not regenerate a custom pipeline. It returns without changing the schedule when `self.use_custom_schedule` is true.

- [ ] **Step 6: Commit**

```bash
git add megatron/training/arguments.py megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py tests/unit_tests/pipeline_parallel/test_custom_schedule.py
git commit -m "feat: select custom static schedules"
```

---

### Task 4: Planner-Supplied Communication Ordering

**Files:**
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/execution_planner.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`
- Modify: `tests/unit_tests/pipeline_parallel/test_custom_schedule.py`

**Interfaces:**
- Consumes: `Pipeline.custom_schedule_spec`
- Produces: `CommEvent.op_id: Optional[CommOpId]`
- Produces: `ComputeTask` pre/post events with receive posting ordered by the matching `comm` array
- Produces: sender gate lookup through `CustomScheduleSpec`

- [ ] **Step 1: Write planner event-order tests**

For channel `F_0_1 = [mb0, mb2, mb1, mb3]`, assert sender POST order and receiver POST order both expose `[0, 2, 1, 3]`. Assert every WAIT recv remains attached to its target compute task.

- [ ] **Step 2: Add canonical operation IDs to communication events**

Add:

```python
op_id: Optional[CommOpId] = None
```

Populate it for POST and WAIT events from task type, microbatch, source stage, and destination stage.

- [ ] **Step 3: Replace custom-mode receive sorting**

In custom mode, sort each receive list by the index of its `CommOpId` in the relevant communication array. Choose insertion positions monotonically in that order and never later than the corresponding WAIT target task.

Existing schedules keep completion-time sorting.

- [ ] **Step 4: Build sender gating metadata**

When a POST send is scheduled, use `event.op_id` to retrieve local and remote predecessors. Local predecessors wait on stored NCCL `Work`; remote predecessors wait through the control runtime.

- [ ] **Step 5: Commit**

```bash
git add megatron/core/pipeline_parallel/cdc_scheduler/execution_planner.py megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py tests/unit_tests/pipeline_parallel/test_custom_schedule.py
git commit -m "feat: honor custom communication order"
```

---

### Task 5: Four-Rank Gloo Signal Runtime

**Files:**
- Modify: `megatron/core/parallel_state.py`
- Create: `megatron/core/pipeline_parallel/cdc_scheduler/comm_dependency_runtime.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`
- Create: `tests/unit_tests/pipeline_parallel/test_comm_dependency_runtime.py`

**Interfaces:**
- Produces: `parallel_state.get_pipeline_control_group_gloo()`
- Produces: `parallel_state.get_pipeline_control_global_ranks()`
- Produces: `CommDependencyRuntime.register_work(op_id, work)`
- Produces: `CommDependencyRuntime.wait_before_submit(op_id)`
- Produces: `CommDependencyRuntime.finish_iteration()`

- [ ] **Step 1: Write fake-runtime protocol tests**

Use fake Work objects and a fake transport to verify:

- one trigger releases one target;
- one trigger releases multiple targets;
- one target waits for multiple triggers;
- local dependency waits without a Gloo token;
- duplicate registration and missing Work raise errors;
- timeout messages include rank, dependency ID, trigger, and target.

- [ ] **Step 2: Add the Gloo pipeline control group**

During PP group creation, collectively create:

```python
control_group = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
```

Store the group and global ranks for the current process. Add getters and destroy the group in `destroy_model_parallel()`.

- [ ] **Step 3: Implement transport and runtime**

Use CPU `torch.int64` tensors and stable dependency IDs as tags. A background worker waits once per unique trigger Work and sends one token per remote target. Target sender uses blocking Gloo receive before NCCL submission.

- [ ] **Step 4: Integrate runtime into the scheduler**

Create the runtime only when normalized extra dependencies are non-empty. Register every relevant POST send Work. Call `wait_before_submit()` before target `isend`. Finish workers before clearing request dictionaries.

- [ ] **Step 5: Compare normalized digests**

Use `torch.distributed.all_gather_object` on the Gloo control group and reject differing SHA256 values before the first scheduled task.

- [ ] **Step 6: Commit**

```bash
git add megatron/core/parallel_state.py megatron/core/pipeline_parallel/cdc_scheduler/comm_dependency_runtime.py megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py tests/unit_tests/pipeline_parallel/test_comm_dependency_runtime.py
git commit -m "feat: gate communications with gloo signals"
```

---

### Task 6: Runtime JSONL and Chrome Trace

**Files:**
- Create: `megatron/core/pipeline_parallel/cdc_scheduler/schedule_trace.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`
- Modify: `megatron/core/pipeline_parallel/cdc_scheduler/comm_dependency_runtime.py`
- Create: `tools/merge_custom_schedule_trace.py`
- Create: `tests/unit_tests/pipeline_parallel/test_schedule_trace.py`

**Interfaces:**
- Produces: `ScheduleTraceWriter.emit(event, op_id, iteration, dependency_id=None, **fields)`
- Produces: one `rank_<global_rank>.jsonl` per rank
- Produces: `merge_trace_files(input_dir, output_path) -> Dict`

- [ ] **Step 1: Write trace serialization and merge tests**

Assert required fields, monotonic per-rank ordering, duration pairing, signal latency calculation, and Chrome flow events connecting trigger, signal, and target.

- [ ] **Step 2: Implement the per-rank writer**

Use `time.perf_counter_ns()`, one buffered JSONL stream per rank, a lock for background control workers, iteration-boundary flush, and explicit close.

- [ ] **Step 3: Instrument compute and communication**

Emit compute start/end around F/B/W NVTX ranges. Emit receive post, send post, wait start, and completion around P2P work handling.

- [ ] **Step 4: Instrument signals**

Emit signal wait/send/receive and target submit events with dependency IDs. Keep the trace API optional and no-op when no trace directory is supplied.

- [ ] **Step 5: Implement Chrome Trace merging**

Create compute/communication/signal lanes, duration events, and `ph: "s"/"f"` flow arrows. Include microbatch, F/B, src/dst, operation name, and measured signal/gate timing in event args.

- [ ] **Step 6: Commit**

```bash
git add megatron/core/pipeline_parallel/cdc_scheduler/schedule_trace.py megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py megatron/core/pipeline_parallel/cdc_scheduler/comm_dependency_runtime.py tools/merge_custom_schedule_trace.py tests/unit_tests/pipeline_parallel/test_schedule_trace.py
git commit -m "feat: trace custom pipeline execution"
```

---

### Task 7: Four-GPU Experiment and Comparison Documentation

**Files:**
- Create: `tools/compare_custom_schedule_runs.py`
- Create: `docs/custom_pipeline_schedule.md`
- Modify: `tests/unit_tests/pipeline_parallel/test_schedule_trace.py`

**Interfaces:**
- Consumes: A/B/C run logs and merged signal trace
- Produces: summary JSON with iteration statistics and signal statistics

- [ ] **Step 1: Write comparison aggregation tests**

Verify median, min, max, p95, `C-A`, `B-A`, and `C-B` from deterministic fixture data.

- [ ] **Step 2: Implement comparison tool**

Read CrossPipe experiment JSON and merged trace metrics. Emit machine-readable JSON and a concise table.

- [ ] **Step 3: Document exact A/B/C commands**

Document:

```text
A: --static_schedule 1F1B
B: --custom-pipeline-schedule replay.order.json
C: --custom-pipeline-schedule replay.order.json
   --custom-comm-dependency replay.notification.no_competition.notification_deps.json
```

All commands specify PP=4, TP=1, no recomputation flags, no communication delay/bandwidth injection, and a distinct trace/output directory.

- [ ] **Step 4: Document expected trace assertions**

For the validated star-topology fixture:

```text
Comm_F_3_1_2 completes
Gloo signal rank 1 -> rank 3
Comm_B_0_3_2 submits after signal_receive
```

Also verify the local dependency:

```text
Comm_F_3_2_3 completes before Comm_B_0_2_1 submits
```

- [ ] **Step 5: Perform Windows static review without execution**

Inspect imports, type signatures, all call sites, group lifecycle, operation mapping, JSON schema handling, and trace event pairing. Record that runtime verification remains pending on Linux with four V100 GPUs.

- [ ] **Step 6: Commit**

```bash
git add tools/compare_custom_schedule_runs.py docs/custom_pipeline_schedule.md tests/unit_tests/pipeline_parallel/test_schedule_trace.py
git commit -m "docs: add custom schedule experiment workflow"
```

