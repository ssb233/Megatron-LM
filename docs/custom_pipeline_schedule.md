# Custom Pipeline Schedule Validation (PP=4)

This feature loads a Magellan one-chunk schedule directly into CrossPipe and
optionally enforces Magellan's extra communication dependencies. The first
supported setup is one Linux node with four V100 GPUs:

```text
PP=4, TP=1, DP=1, one model chunk, four microbatches
activation recomputation off
num_dc=1 and no injected communication delay
```

The Windows checkout is for editing and static review only. Run all commands
below in the Linux environment that owns the four GPUs.

## Inputs

Use the two files produced by the validated Magellan star topology:

```text
replay.order.json
replay.notification.no_competition.notification_deps.json
```

The relevant extra dependencies in that result are:

```text
local:  Comm_F_3_2_3 completion -> Comm_B_0_2_1 submission
remote: Comm_F_3_1_2 completion -> Gloo rank 1 to rank 3
        -> Comm_B_0_3_2 submission
```

The dependency filename is not fixed; its JSON schema and contents determine
the behavior.

## Required CrossPipe arguments

Keep the model/data arguments from the existing four-GPU CrossPipe launch
script. The common scheduler portion for all three runs is:

```bash
COMMON_CDC_ARGS="\
  --pipeline-model-parallel-size 4 \
  --tensor-model-parallel-size 1 \
  --num-layers-per-virtual-pipeline-stage <layers-per-physical-stage> \
  --head_tail_as_one_layer \
  --num_subparts 1 \
  --no-align-grad-reduce \
  --no-align-param-gather \
  --num_dc 1 \
  --cdc_exp_logging \
  --cdc_exp_test_start_iter 3 \
  --cdc_exp_per_cfg_test_iters 10"
```

Do not pass any recomputation option:

```text
--recompute-activations
--recompute-granularity
--recompute-method
--recompute-num-layers
```

Do not pass a nonzero `--cdc_latency_bandwidth_delay_as_F_stage`. The custom
schedule path automatically enables the CrossPipe scheduler. Run A still
needs `--enable_cdcpp_scheduler` because it has no custom path.

`<layers-per-physical-stage>` must make Megatron create exactly one virtual
chunk. With CrossPipe's `head_tail_as_one_layer` layout, use the same value as
the existing one-chunk PP=4 configuration for the selected model.

## A/B/C experiment

Use identical model, batch, optimizer, data, logging, warmup, and iteration
arguments. Use separate tensorboard and experiment directories.

### A: default CrossPipe 1F1B

```bash
torchrun --standalone --nproc_per_node=4 pretrain_gpt.py \
  <identical-model-and-data-args> \
  ${COMMON_CDC_ARGS} \
  --enable_cdcpp_scheduler \
  --static_schedule 1F1B \
  --tensorboard-dir runs/custom_schedule/A_default_1f1b
```

### B: custom compute and communication order, no extra dependency file

```bash
torchrun --standalone --nproc_per_node=4 pretrain_gpt.py \
  <identical-model-and-data-args> \
  ${COMMON_CDC_ARGS} \
  --custom-pipeline-schedule /abs/path/replay.order.json \
  --custom-schedule-trace-dir runs/custom_schedule/B_trace \
  --tensorboard-dir runs/custom_schedule/B_custom_order
```

### C: custom order plus communication dependencies

```bash
torchrun --standalone --nproc_per_node=4 pretrain_gpt.py \
  <identical-model-and-data-args> \
  ${COMMON_CDC_ARGS} \
  --custom-pipeline-schedule /abs/path/replay.order.json \
  --custom-comm-dependency \
    /abs/path/replay.notification.no_competition.notification_deps.json \
  --custom-schedule-trace-dir runs/custom_schedule/C_trace \
  --tensorboard-dir runs/custom_schedule/C_custom_dependency
```

The supplied `replay.order.json` must contain exactly four microbatches,
matching Megatron's runtime microbatch count in both training and evaluation.

## Linux verification

Run the authored unit tests before the four-GPU experiment:

```bash
pytest \
  tests/unit_tests/pipeline_parallel/test_custom_schedule.py \
  tests/unit_tests/pipeline_parallel/test_custom_schedule_trace.py \
  tests/unit_tests/pipeline_parallel/test_compare_custom_schedule_runs.py \
  -v
```

Convert C's per-rank JSONL files:

```bash
python tools/custom_schedule_trace_to_chrome.py \
  runs/custom_schedule/C_trace
```

This produces:

```text
custom_schedule.chrome.json
custom_schedule.summary.json
```

Open the Chrome JSON in Perfetto or `chrome://tracing`. Check both flow
relationships:

```text
Comm_F_3_1_2 comm_complete
  -> signal_send_start on pipeline rank 1
  -> signal_recv on pipeline rank 3
  -> Comm_B_0_3_2 target_submit

Comm_F_3_2_3 comm_complete
  -> local dependency wait end on pipeline rank 2
  -> Comm_B_0_2_1 target_submit
```

`custom_schedule.summary.json` reports:

```text
completion_to_target_submit_us
signal_send_to_recv_us
```

These timestamps use `time.perf_counter_ns()`. They are comparable across
processes for this supported single-node experiment. Compute ranges are host
submission ranges; use the existing NVTX ranges with Nsight Systems when GPU
kernel timing is required.

Compare end-to-end results:

```bash
python tools/compare_custom_schedule_runs.py \
  --run-a /abs/path/A/exp_final.json \
  --run-b /abs/path/B/exp_final.json \
  --run-c /abs/path/C/exp_final.json \
  --signal-summary runs/custom_schedule/C_trace/custom_schedule.summary.json \
  --warmup 1 \
  --output runs/custom_schedule/comparison.json
```

Interpret the median deltas as:

```text
B - A: custom ordering change relative to default 1F1B
C - B: extra serialization plus Gloo control-signal overhead
C - A: total custom-schedule change relative to default 1F1B
```

Trace collection itself adds host-side file-writing overhead. For final
end-to-end timing, repeat A/B/C without `--custom-schedule-trace-dir`; use a
separate traced C run to validate ordering and signal latency.

## Failure behavior

CrossPipe fails before scheduling if JSON operations are missing, duplicated,
assigned to the wrong stage/channel, cyclic, or inconsistent across ranks. It
also rejects custom mode unless PP=4, TP=1, DP=1, one chunk, recomputation off,
and communication-delay injection off.

At runtime, a target sender errors if a required local NCCL Work was never
submitted. Signal workers are joined at iteration cleanup and report a
timeout or background Gloo exception instead of silently continuing.

Runtime verification is intentionally pending until these changes are moved
to the Linux four-V100 machine.
