# PP=4 Magellan 通信依赖开销实验

本目录比较单机 4×V100-32GB 上默认 1F1B 与 Magellan 自定义静态调度的
iteration 时间。Magellan 版本严格执行求解器产生的通信依赖；实验不注入额外
通信时延或带宽限制，也不启用重计算，因此差值主要反映自定义事件调度与 Gloo
控制信号的端到端软件开销。

## 实验设置

- PP=4、TP=1、DP=1，8 层，单机 4 卡。
- 逻辑拓扑为星形 `0-1, 1-2, 1-3`；stage 与 rank/DC 一一对应。
- 每个 arm 运行 20 次 iteration，只统计第 6–19 次，共 14 个样本。
- D1 使用 8 个 microbatch；D2、M1、M2 使用 16 个 microbatch。
- MoE 使用 8 个专家、top-k=2。
- 每个配置只进行一次正式运行，结果适合功能与开销初步验证，不代表多次运行的
置信区间。

## 结果

| ID | 模型 | hidden | seq | MBS | microbatches | experts/top-k | 1F1B median (ms) | Magellan median (ms) | median overhead | extra/remote deps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | Dense | 1024 | 256 | 1 | 8 | - | 314.20 | 281.40 | -10.44% | 6/3 |
| D2 | Dense | 1536 | 512 | 2 | 16 | - | 545.05 | 584.60 | +7.26% | 46/21 |
| M1 | MoE | 1024 | 256 | 1 | 16 | 8/top-2 | 852.70 | 919.20 | +7.80% | 46/21 |
| M2 | MoE | 768 | 512 | 2 | 16 | 8/top-2 | 859.05 | 890.00 | +3.60% | 46/21 |
| M3* | MoE | 1024 | 256 | 1 | 8 | 8/top-2 | 435.10 | 426.20 | -2.05% | 6/3 |

`extra/remote deps` 中，前者是排除单通信通道天然顺序后的额外依赖数，后者是
起点和终点 rank 均不同、需要 Gloo control message 的依赖数。所有调度均通过
operation coverage、去重、DAG 无环校验；所有训练日志均无 skipped/NaN
iteration。

## 标定与求解校验

| ID | preflight peak memory (MiB) | comm_units | CP-SAT | raw/extra/remote deps |
|---|---:|---:|---:|---:|
| D1 | 1246.0 | 0.02723856 | OPTIMAL | 46/6/3 |
| D2 | 2070.4 | 0.05413524 | OPTIMAL | 114/46/21 |
| M1 | 2739.7 | 0.01159703 | OPTIMAL | 114/46/21 |
| M2 | 1573.4 | 0.01222493 | OPTIMAL | 114/46/21 |
| M3 | 2739.7 | 0.01068607 | OPTIMAL | 46/6/3 |

三组新增配置的峰值显存均远低于 28 GiB 门限，未触发备用配置。D1 峰值取其正式
基线日志，其余三组取各自四次迭代的 preflight 日志。`comm_units` 是实测相邻
P2P 通信时间与中间 stage 前向计算时间之比；求解中传播时延设为 0。

## 文件说明

- `summary.csv`：每个配置的一行聚合统计。
- `samples.csv`：8 个 arm 的 112 个原始 iteration 样本。
- `summary.json`：参数、统计、校准和依赖数量的机器可读汇总。
- 根目录 `summary.csv`、`summary.json` 和 `samples.csv` 保留原 D1–M2
  单次运行矩阵；M3 的重复实验使用 `M3_moe_n8/` 下的独立汇总，避免混合统计口径。
- 每个配置的 `solver/`：`replay.order.json`、通信依赖 JSON、求解摘要和验证结果。
- 每个配置的 `calibration/`：实测计算/通信基准以及归一化 `comm_units`。
- `1f1b/` 与 `magellan_dependency/`（D1 为 `pilot_20260730/raw/`）：
  原始 `train.log`、命令参数和完成标记。

## 解读限制

两种调度改变了 microbatch 计算事件的提交顺序，因此 dropout/MoE 路由等随机
算子的 RNG 消耗顺序也可能改变；loss 应检查为有限且趋势正常，不要求两个 arm
逐 iteration 完全相同。当前实验关注训练可完成性与 iteration 开销。

## M3 重复 MoE 实验

`M3*` 不是单次运行结果，而是每个 arm 四次运行、每个 arm 56 个样本，并平衡了
`1F1B→Magellan` 和 `Magellan→1F1B` 两种启动顺序。聚合中位数显示 Magellan
改善 2.05%，但聚合均值慢 3.47%；四对中只有两对的中位数更快，因为 1F1B
呈现明显的双峰运行状态。完整且非选择性的结果见 `M3_moe_n8/README.md` 和
`M3_moe_n8/confirmation_summary.json`。
