# D1 Dense：1F1B 与 Magellan 通信依赖调度 Pilot

## 实验目标

在单机 4×V100、PP=4、无重计算、无通信时延/带宽注入的条件下，对比：

- CrossPipe 静态 `1F1B`
- Magellan 自定义计算/通信顺序，并启用额外通信依赖的 Gloo 控制消息

本实验主要验证带通信依赖的自定义调度能够正确完成训练，且控制依赖不会引入明显的端到端性能退化。

## Dense 配置

| 参数 | 值 |
|---|---:|
| Transformer layers | 8 |
| hidden size | 1024 |
| FFN hidden size | 4096 |
| attention heads | 16 |
| sequence length | 256 |
| micro batch size | 1 |
| global batch size / microbatches | 8 |
| pipeline parallel size | 4 |
| virtual pipeline chunks | 1 |
| recompute | disabled |
| training iterations | 20 |
| measured steady window | iterations 6–19 |

两组使用相同 seed、数据、优化器和原始 NCCL P2P 通信。`num_dc=1`，延迟配置为 `0,0`，没有使用修改版 PyTorch 的通信注入能力。

## Magellan 求解

- 求解拓扑：4 DC 星形，链路为 `0-1、1-2、1-3`
- 链路参数：300 Gbps、10 ms；本次 CP-SAT 通信时长使用实机 profile 标定值，传播时延不进入求解
- `T_F = 5.425743 ms`
- 相邻 P2P 通信中位数：`0.147789 ms`
- `comm_units = T_comm / T_F = 0.02723856`
- `microbatches=8`，`stages=4`
- `--sat-link-exclusive`
- `--sat-comm-split-k 1`
- `--r-save 1`
- `--mem 8`
- CP-SAT makespan：`33.162 × T_F`（约 `179.928 ms`）

CrossPipe 校验结果：

- 操作总数：112
- 原始 dependency 文件中的有效通信边：46
- 去除每个通信 channel 自带的串行边后，额外通信依赖：6
- 需要跨 rank Gloo signal 的依赖：3
- 完整依赖图：无环
- 调度 SHA-256：`bc41868d5223cc82bc6e44bdd09c990306ce92d9eae9b728f701888c0b8ac136`

运行时真正需要的输入是 `solver/replay.order.json` 和
`solver/replay.notification.no_competition.notification_deps.json`。

## Pilot 结果

推荐使用 iterations 6–19，排除初始化、profile、配置切换和 iteration 20 的退出扰动：

| 调度 | 样本数 | 平均值 (ms) | 中位数 (ms) | 标准差 (ms) | 最小/最大 (ms) |
|---|---:|---:|---:|---:|---:|
| 1F1B | 14 | 299.443 | 314.2 | 26.664 | 238.1 / 323.6 |
| Magellan + dependency | 14 | 277.686 | 281.4 | 25.988 | 211.7 / 310.0 |

相对 1F1B：

- Magellan 中位 iteration 时间减少 `32.8 ms`（`10.44%`）
- Magellan 平均 iteration 时间减少 `21.76 ms`（`7.27%`）

若按预先定义的 iterations 6–20 统计，1F1B/Magellan 中位数分别为
`315.1 ms` 和 `282.0 ms`。1F1B 的 iteration 20 为 `776.0 ms`，明显包含
实验结束处理，因此不适合用于均值比较；中位数不受该点显著影响。

两组均完成 20 iterations，`skipped iterations=0`、`nan iterations=0`。
相同 iteration 的 loss 完全一致。Magellan 的 3 个跨-rank 控制依赖没有导致
死锁或可见的性能退化。

## 结论与限制

这个单次 pilot 已经达到功能验证目的：PP=4 的 Dense 模型可以使用 Magellan
自定义顺序和额外通信依赖完成端到端训练。当前结果甚至快于 1F1B，但只有一次
运行，不能把 `10.44%` 作为最终性能结论。正式结果应交替运行多次
1F1B/Magellan，并报告跨重复实验的中位数和置信区间。

## 文件说明

- `solver/replay.order.json`：计算和通信静态顺序
- `solver/replay.notification.no_competition.notification_deps.json`：通信依赖
- `solver/validation.json`：覆盖性与无环验证
- `solver/experiment.summary.json`：CP-SAT/replay 摘要
- `calibration/`：D1 实机 profile 和求解标定值
- `pilot_20260730/raw/1f1b/`：1F1B 原始日志、参数和 profile
- `pilot_20260730/raw/magellan/`：Magellan 原始日志、参数和 profile
- `pilot_20260730/*_iter6_20.json`：预定义窗口的统计
- `pilot_20260730/pilot_summary.json`：推荐窗口 iterations 6–19 的比较
