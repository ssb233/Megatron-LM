# M3：N=8 MoE Magellan 通信依赖调度

## 目标与配置

M3 用于寻找一个保留真实额外通信依赖、且 Magellan iteration 中位数不差于
1F1B 的 MoE case。配置为 PP=4、8 层、hidden 1024、FFN 4096、sequence
256、MBS 1、GBS/N 8、8 experts、top-2。实验运行在单机 4×V100-32GB，
关闭重计算，不注入通信时延或额外数据。

M3 进行了独立 profile：`T_F=13.6851 ms`，
相邻 P2P 中位数 `0.1462 ms`，
`comm_units=0.01068607`。峰值显存为 `2739.7 MiB`。

## 调度

求解拓扑为星形 `0-1、1-2、1-3`，CP-SAT 状态为 `OPTIMAL`。调度包含
112 个操作、46 条原始通信边；去除 channel 天然顺序后有 6 条额外依赖，
其中 3 条需要跨-rank Gloo signal。完整图通过 CrossPipe 实际加载和 DAG
无环校验。

## 四对平衡顺序结果

每次运行 20 iterations，统计 iterations 6–19。两种启动顺序各使用两次：

| Pair | 执行顺序 | 1F1B median (ms) | Magellan median (ms) | Magellan 相对变化 |
|---|---|---:|---:|---:|
| screen_1f1b_first | 1F1B->Magellan | 444.40 | 438.10 | -1.42% |
| A_magellan_first | Magellan->1F1B | 349.70 | 423.05 | +20.98% |
| B_1f1b_first | 1F1B->Magellan | 444.85 | 422.65 | -4.99% |
| C_magellan_first | Magellan->1F1B | 401.00 | 423.15 | +5.52% |

聚合 56 个样本/arm：

- 1F1B：mean `412.789 ms`，median `435.100 ms`，stdev `49.904 ms`
- Magellan：mean `427.104 ms`，median `426.200 ms`，stdev `19.639 ms`
- 中位数变化：`-2.05%`
- 均值变化：`+3.47%`

M3 通过了预注册的“聚合中位数不慢于 1F1B”判据，而且 Magellan 的标准差
明显更低。但是它不是逐对稳定胜出：四对中只有两对中位数更快，且聚合均值
反而更慢。1F1B 在不同进程中呈现约 350 ms 与 440 ms 两种运行状态，因此该
结果应表述为“中位数改善、抖动降低的 case”，不能表述为全面稳定优于 1F1B。

## 文件

- `calibration/`：M3 实机 profile 与归一化通信比例。
- `solver/`：order、通信依赖、CP-SAT 摘要与验证结果。
- `runs/`：四对、八次训练的原始日志和运行参数。
- `samples.csv`：112 个 measured iteration 样本。
- `run_summary.csv`：每次运行的统计。
- `confirmation_summary.json`：参数、调度、逐对与聚合结果。
