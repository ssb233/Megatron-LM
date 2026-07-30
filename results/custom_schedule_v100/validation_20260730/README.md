# CrossPipe PP=4 自定义通信调度实验结果

本目录归档 CrossPipe 自定义调度 C 实验及 A/B/C 对比结果。目标是验证：

1. CrossPipe 能按 Magellan 给出的静态 order 执行非默认通信顺序；
2. 两个原本可并发、但共享逻辑链路的通信能够通过额外依赖强制串行；
3. 跨 rank 依赖由 CPU/Gloo signal 解锁，不会阻塞已经提交到 GPU 的计算；
4. trace 能同时还原 stage、microbatch、前向/反向计算、P2P 通信和 signal。

## 实验配置

| 项目 | 配置 |
|---|---|
| 服务器 | 单节点 4×Tesla V100-SXM2-32GB |
| 并行配置 | PP=4, TP=1, DP=1 |
| Microbatch | 8，MBS=1，GBS=8 |
| 模型 | 8 layers, hidden 512, FFN 2048, 8 heads, sequence length 256 |
| 数值格式 | FP16 |
| 重计算 | 关闭 |
| 运行长度 | `train-iters=20`；实验 logger 完成 iteration 3–10 后在 iteration 11 主动退出 |
| 物理 DC | `num_dc=1`，不注入带宽或时延 |
| 逻辑求解拓扑 | `0–1, 1–2, 1–3`，300 Gbps / 10 ms |
| Stage 到逻辑 DC | `stage2dc: [0, 1, 2, 3]` |

逻辑 stage/DC 与真机 rank 的对应关系为：

| Stage | 逻辑 DC | 真机 GPU/rank |
|---|---:|---:|
| S0 | DC0 | GPU0/rank0 |
| S1 | DC1 | GPU1/rank1 |
| S2 | DC2 | GPU2/rank2 |
| S3 | DC3 | GPU3/rank3 |

真机上的四张 GPU 均位于同一节点。逻辑 DC 映射仅用于解释 Magellan 生成的
通信依赖，当前实验不用于评估真实跨 DC 带宽或时延。

## A/B/C 定义

- **A**：CrossPipe 默认静态 1F1B。
- **B**：加载 `replay_order_pp4_n8_star.json`，执行自定义计算和通信顺序。
- **C**：在 B 的基础上加载 `notification_deps_pp4_n8_star.json`，执行额外通信依赖。
- **C_TRACE**：与 C 相同，但开启逐 rank JSONL trace。

## 端到端结果

去掉一个计时 warmup 后，每组包含 7 个正式样本：

| 模式 | Median | Mean | P95 |
|---|---:|---:|---:|
| A：默认 1F1B | 227.526 ms | 234.528 ms | 265.435 ms |
| B：自定义 order | 257.033 ms | 251.100 ms | 269.084 ms |
| C：order + dependency | 268.739 ms | 269.510 ms | 272.960 ms |

差值：

```text
B - A = 29.507 ms (+12.969%)
C - B = 11.706 ms (+4.554%)
C - A = 41.213 ms (+18.114%)
```

`C-B` 包含额外通信串行化和 Gloo 控制路径的合计影响，不能视为纯 signal
软件开销。

## 图中展示的通信依赖

论文图选择了两个连续的 remote dependency：

```text
Comm_F_4_1_2 完成
  └─ dependency 6: Gloo rank1 → rank3
       └─ 解锁 Comm_B_0_3_2

Comm_B_0_3_2 完成
  └─ dependency 1: Gloo rank3 → rank1
       └─ 解锁 Comm_F_5_1_2
```

即：

```text
F4: S1→S2  →  B0: S3→S2  →  F5: S1→S2
```

在逻辑星型拓扑中：

- `S1→S2` 使用 `DC1→DC2`，路径为 `1→2`；
- `S3→S2` 使用 `DC3→DC2`，路径为 `3→1→2`。

二者共享有向链路 `1→2`，因此无竞争调度要求它们依次执行。C 的 trace 中，
目标 NCCL send 的 `target_submit` 均发生在相应 `signal_recv` 之后。

## Signal 测量

正式 iteration 3–10 共包含 56 个 remote dependency 样本：

| 指标 | Median | P95 |
|---|---:|---:|
| 两端均已提交后的 Gloo 完成延迟 | 225.405 μs | 290.480 μs |
| `signal_recv` 到目标 NCCL `target_submit` | 345.858 μs | 485.881 μs |

第一个指标定义为：

```text
signal_recv - max(signal_send_start, signal_wait_start)
```

该定义排除了发送端或接收端尚未进入 Gloo 操作的调度等待时间。

## 目录结构

```text
inputs/
  replay_order_pp4_n8_star.json
  notification_deps_pp4_n8_star.json

metrics/
  A_default_1f1b.exp_final.json
  B_custom_order.exp_final.json
  C_custom_order_dependency.exp_final.json
  C_trace.exp_final.json
  comparison.json
  signal_ready_latency.json

trace/
  rank_0.jsonl ... rank_3.jsonl
  custom_schedule.chrome.json
  custom_schedule.summary.json

plans/
  A_default_1f1b/
  B_custom_order/
  C_custom_order_dependency/
  C_trace/

logs/
  A_default_1f1b/
  B_custom_order/
  C_custom_order_dependency/
  C_trace/

figure/
  custom_schedule_trace.svg
  custom_schedule_trace.pdf
  custom_schedule_trace.png
  figure_caption.txt
  plot_custom_schedule_trace.py
  source_data_iteration5.csv
  source_data_signal_latency.csv

MANIFEST.sha256
```

## 文件使用

- `custom_schedule_trace.svg`：论文排版首选，文字保持可编辑。
- `custom_schedule_trace.pdf`：LaTeX/论文系统直接引用。
- `custom_schedule_trace.png`：600 dpi 预览。
- `custom_schedule.chrome.json`：使用 Perfetto 或 `chrome://tracing` 查看完整 trace。
- `rank_*.jsonl`：原始逐 rank host-side trace。
- `source_data_*.csv`：论文图对应的整理后数据。
- `comparison.json`：A/B/C 完整统计和逐样本数据。
- `MANIFEST.sha256`：归档文件完整性校验。

## Trace 口径

图中的 compute 区间和通信事件来自 host-side instrumentation。它可以严格验证
事件下发顺序、依赖关系和 CPU/Gloo 控制路径，但不等价于 CUDA kernel 或 NCCL
kernel 的 GPU 执行时长。如需 GPU 时间线，应另行使用 Nsight Systems，并将
NVTX range 与本目录的 operation/microbatch 标识对应。
