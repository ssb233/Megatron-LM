# CrossPipe 自定义流水线调度：AI 实验上下文（PP=4）

本文是本分支自定义调度功能的入口文档。远程服务器上的 AI 在运行、修改或优化实验前，应先阅读本文，再阅读 `megatron/core/pipeline_parallel/cdc_scheduler/` 下的实现。

本功能将 Magellan 的 one-chunk 静态调度直接加载到 CrossPipe，并可选地执行 Magellan 生成的额外通信依赖。目标是让调度表同时控制计算 event、通信 event、通信顺序和跨 rank 的控制 signal，而不是重新实现一个固定的 1F1B 策略。

当前支持目标是单机 4 张 V100：

```text
PP=4, TP=1, DP=1, one model chunk, eight microbatches
recomputation disabled
num_dc=1, no injected communication delay or bandwidth constraint
```

Windows 工作树仅用于代码编辑和静态检查；真实训练必须在拥有 4 张 V100 的 Linux 服务器上执行。

## 1. 已完成的修改

### 参数与调度模式

`megatron/training/arguments.py` 新增：

```text
--custom-pipeline-schedule <replay.order.json>
--custom-comm-dependency <comm_dependency.json>
--custom-schedule-trace-dir <trace-directory>
```

传入 order 文件后会自动启用 CrossPipe scheduler；custom mode 不能与
`--static_schedule` 或 `--dynamic_schedule` 同时使用。dependency 文件必须
和 order 文件一起使用。未显式配置延迟时，custom mode 使用零延迟默认值。

### JSON 解析和静态 pipeline

`cdc_scheduler/custom_schedule.py` 负责解析和校验 compute/communication
操作、PP/microbatch 数量、通道方向、操作覆盖、重复操作和循环依赖，并生成
canonical SHA256 供各 rank 做输入一致性检查。`pp_generator` 将解析结果
转换为 one-chunk static pipeline；`pp_scheduler.py` 和 `execution_planner.py`
使用这个静态表生成计算与通信 event，不回退到默认 1F1B 通信排序。

### 通信依赖、Gloo 和 trace

`comm_dependency.py` 在 NCCL 发送提交前执行依赖门控：local dependency 等待
同 rank 的前置 NCCL Work 完成，remote dependency 由前置通信完成后的后台
线程通过 CPU/Gloo process group 发出 `int64` signal。目标发送端接收 signal
后才提交 NCCL send。Gloo `Work` 必须调用带超时的 `wait()` 驱动完成；仅轮询
`is_completed()` 在当前 PyTorch/Gloo 版本上不会取得进展。

`parallel_state.py` 创建和管理 pipeline control Gloo group；
`custom_schedule_trace.py` 记录 compute、NCCL、local wait、signal send/recv
和 target submit；两个 `tools/` 脚本分别用于生成 Chrome/Perfetto trace 和
比较默认 1F1B、custom order、custom order + dependency 的端到端结果。

运行时遇到 timeout、Gloo 异常、missing predecessor 或 operation mismatch
必须显式失败，不能用静默等待掩盖死锁。当前 Windows 没有真实 CUDA/NCCL/Gloo
训练验证，Linux 4×V100 运行是下一步验证。

## Inputs

Use the two files committed in this branch, produced by the validated Magellan
star topology `0-1, 1-2, 1-3`:

```text
tests/unit_tests/pipeline_parallel/fixtures/replay_order_pp4_n8_star.json
tests/unit_tests/pipeline_parallel/fixtures/notification_deps_pp4_n8_star.json
```

The result uses Magellan's 300 Gbps / 10 ms setup and directed-link-exclusive
model, with eight microbatches. It contains 64 compute operations, 48
communication operations, and 57 dependency edges. The dependency file includes
both `serialize_same_src_dc_on_directed_link` and `insert_notify_delay_op`.
The latter is the additional cross-rank control dependency, not ordinary
channel ordering or rate allocation.

The representative dependencies in this result are:

```text
local:  Comm_F_4_2_3 completion -> Comm_B_0_2_1 submission
remote: Comm_F_4_1_2 completion -> Gloo rank 1 to rank 3
        -> Comm_B_0_3_2 submission
```

The dependency filename is not fixed; its JSON schema and contents determine
the behavior.

## Required CrossPipe arguments

On the Linux server, set the paths to the committed fixtures:

```bash
export CROSSPIPE_ROOT=/path/to/crosspipe
export ORDER_JSON="$CROSSPIPE_ROOT/tests/unit_tests/pipeline_parallel/fixtures/replay_order_pp4_n8_star.json"
export DEP_JSON="$CROSSPIPE_ROOT/tests/unit_tests/pipeline_parallel/fixtures/notification_deps_pp4_n8_star.json"
```

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
  --cdc_latency_bandwidth_delay_as_F_stage 0,0 \
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

The explicit `0,0` pair is required by CrossPipe's experiment logger and does
not inject delay. Do not pass a nonzero
`--cdc_latency_bandwidth_delay_as_F_stage`. The custom schedule path
automatically enables the CrossPipe scheduler. Run A still needs
`--enable_cdcpp_scheduler` because it has no custom path.

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
  --custom-pipeline-schedule "$ORDER_JSON" \
  --custom-schedule-trace-dir runs/custom_schedule/B_trace \
  --tensorboard-dir runs/custom_schedule/B_custom_order
```

### C: custom order plus communication dependencies

```bash
torchrun --standalone --nproc_per_node=4 pretrain_gpt.py \
  <identical-model-and-data-args> \
  ${COMMON_CDC_ARGS} \
  --custom-pipeline-schedule "$ORDER_JSON" \
  --custom-comm-dependency \
    "$DEP_JSON" \
  --custom-schedule-trace-dir runs/custom_schedule/C_trace \
  --tensorboard-dir runs/custom_schedule/C_custom_dependency
```

The supplied `replay_order_pp4_n8_star.json` contains exactly eight microbatches,
so Megatron's runtime microbatch count must also be eight in both training and
evaluation. The count is derived from the batch/DP configuration; do not only
replace the JSON while leaving the runtime at four microbatches.

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
Comm_F_4_1_2 comm_complete
  -> signal_send_start on pipeline rank 1
  -> signal_recv on pipeline rank 3
  -> Comm_B_0_3_2 target_submit

Comm_F_4_2_3 comm_complete
  -> local dependency wait end on pipeline rank 2
  -> Comm_B_0_2_1 target_submit
```

`custom_schedule.summary.json` reports:

```text
completion_to_target_submit_us
signal_send_to_recv_us
gate_wait_us
signal_recv_to_target_submit_us
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

## 4×V100 实机验证（2026-07-30）

验证服务器使用单节点 4 张 Tesla V100-SXM2-32GB。可复现实验入口为：

```bash
cd /home/songxb26/mnist/crosspipi-magellan
RUN_ROOT="$PWD/runs/custom_schedule_v100/validation_20260730"

test_crossdc/custom_schedule_v100/run_custom_schedule.sh A "$RUN_ROOT"
test_crossdc/custom_schedule_v100/run_custom_schedule.sh B "$RUN_ROOT"
test_crossdc/custom_schedule_v100/run_custom_schedule.sh C "$RUN_ROOT"
test_crossdc/custom_schedule_v100/run_custom_schedule.sh C_TRACE "$RUN_ROOT"
```

脚本使用 8 个 microbatch、8 层小型 GPT、PP=4、TP=DP=1、FP16，不启用重计算，
也不注入通信带宽或时延。A 是默认 CrossPipe 1F1B，B 是自定义 order，C 是
自定义 order 加通信依赖；`C_TRACE` 只用于核对事件关系。

本次运行发现并修复了三个真实问题：

1. local GPT LayerNorm 路径因函数内覆盖 `LNImpl` 触发
   `UnboundLocalError`；
2. custom execution planner 把所有 receive 提前到首个 compute 前，导致
   NCCL channel 顺序死锁；现在 receive 按 sender 完成时间、consumer 使用时间
   和 channel 单调顺序放置，并拒绝无法满足的顺序；
3. CPU/Gloo `irecv` 仅轮询 `Work.is_completed()` 无法完成；现在使用带超时的
   `Work.wait()`。

A、B、C 和 `C_TRACE` 都以 `--train-iters 20` 启动；CrossPipe 实验 logger
在收集 iteration 3–10 后于 iteration 11 主动退出。去掉 1 个计时 warmup 后，
每组 7 个正式样本的结果为：

| 模式 | median iteration |
|---|---:|
| A 默认 1F1B | 227.526 ms |
| B 自定义 order | 257.033 ms |
| C 自定义 order + dependency | 268.739 ms |

对应差值：

```text
B - A = 29.507 ms (+12.969%)
C - B = 11.706 ms (+4.554%)
C - A = 41.213 ms (+18.114%)
```

`C-B` 是额外串行约束与 Gloo 控制信号的合计开销，不能解释成纯 signal
开销。trace 中共有 7 个 remote dependency；取 iteration 3..10 的 56 个
样本，使用
`signal_recv - max(signal_send_start, signal_wait_start)` 作为“两端均已提交
之后”的 signal 完成延迟，median 为 225.405 μs，p95 为 290.480 μs。
`signal_recv -> target_submit` 的 median 为 345.858 μs。

实验原始结果留在服务器的（不提交到 Git）：

```text
runs/custom_schedule_v100/validation_20260730/
  comparison.json
  signal_ready_latency.json
  C_trace/trace/custom_schedule.chrome.json
  C_trace/trace/custom_schedule.summary.json
  C_trace/trace/rank_0.jsonl ... rank_3.jsonl
```

Chrome trace 可用 Perfetto 打开，以 `microbatch`、F/B、compute/communication、
dependency id 和 signal flow 检查执行顺序。此次实机验证证明当前实现可在
单机 4×V100 上执行非默认通信顺序及额外跨 rank 通信依赖；它不是跨 DC
性能结论，因为实验刻意使用 `num_dc=1` 和零注入时延。
# Visualization-only transfer-delay experiment

Experiment `C_VIS` keeps the external PP=4 order and communication dependency
graph unchanged, but makes P2P communication visible in the trace by injecting
transfer delay on every logical pipeline boundary:

```bash
test_crossdc/custom_schedule_v100/run_custom_schedule.sh \
  C_VIS \
  "$PWD/runs/custom_schedule_v100/visualization_delay_20260730"
```

The launcher prepends `/home/songxb26/mnist/pytorch-corsspipe` to
`PYTHONPATH` and rejects any other PyTorch path or version. This custom build
implements transfer delay as a GPU sleep on the NCCL stream. It sends no dummy
payload and does not change the Magellan order or `comm_dependency` edges.

With the default settings, iteration 2 profiles `T_F_stage`, iterations 3–10
use a `0.5×T_F_stage` transfer delay, and iterations 11–18 use
`1.0×T_F_stage`. The run exits at iteration 19. Set
`CDC_EXP_PER_CFG_TEST_ITERS=1` for a smoke run.

> **Warning:** `C_VIS` is a visualization-only experiment. Do not mix its
> iteration times with the zero-delay A/B/C throughput comparison.
