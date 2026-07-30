# H=4096、S=2048 单轮对比

本实验在单机 4×V100-32GB 上比较 CrossPipe 1F1B 与带通信依赖的
Magellan 自定义调度。每个 arm 运行 20 iterations，只统计 iterations
6–19，共 14 个样本；不注入通信时延或额外 payload。

## 公共配置

- PP=4、TP=1、DP=1、8 层、1 个 virtual chunk
- hidden size 4096、FFN hidden size 4096、16 heads、sequence length 2048
- micro-batch size 1、global batch size / microbatch 数量 8
- FP16，关闭 activation recomputation
- 逻辑求解拓扑：星形 `0-1、1-2、1-3`
- 真实运行：单机原始 NCCL，`num_dc=1`，delay factors `0,0`

MoE 额外使用 8 experts、top-k=2、EP=1、all-to-all dispatcher 和
aux-loss coefficient 0.01。

## 单轮结果

| 模型 | 1F1B median | Magellan median | median 变化 | 1F1B mean | Magellan mean | mean 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 583.00 ms | 581.15 ms | -0.32% | 583.47 ms | 581.39 ms | -0.36% |
| MoE | 721.50 ms | 729.95 ms | +1.17% | 721.55 ms | 729.39 ms | +1.09% |

负值表示 Magellan 更快。Dense 在本次单轮试验中近似持平并略快；MoE
略慢。两者差异都很小，但这里只运行了一轮，不能据此声称稳定优势。

## 调度和显存

Dense 调度包含 112 个操作、6 条额外通信依赖，其中 3 条需要跨 rank
Gloo signal。MoE 调度包含 112 个操作、4 条额外通信依赖，其中 3 条
需要跨 rank Gloo signal。两份调度均通过 DAG 无环和 CrossPipe loader
校验。

峰值 allocated memory：

- Dense：1F1B 6156.9 MiB，Magellan 9857.4 MiB
- MoE：1F1B 12978.5 MiB，Magellan 15024.9 MiB

## 求解口径限制

按本次快速试跑要求，没有针对 H=4096 重新执行 4-iteration profile。
求解复用了同模型 H=1024、S=2048 的实测 `comm_units`：

- Dense：`0.04107446185203048`
- MoE：`0.017679909587792298`

仅将 `t_fwd_s` 按 hidden size 的 4 倍缩放，用于求解输出中的秒单位展示。
因此这里验证的是“该近似求解调度在 H=4096 下能否执行以及单轮开销”，
不是 H=4096 下经过独立通信/计算比例校准后的最优 Magellan 调度。
