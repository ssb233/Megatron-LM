# D1H4096S2048：Dense 单轮实验

配置：PP=4、TP=1、DP=1、8 层、H=4096、FFN=4096、heads=16、
S=2048、MBS=1、GBS/N=8、FP16、无重计算、无通信注入。

每个 arm 运行 20 iterations，统计 iterations 6–19：

| arm | mean (ms) | median (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|
| 1F1B | 583.471 | 583.000 | 2.174 | 579.7 | 586.8 |
| Magellan | 581.393 | 581.150 | 2.704 | 575.8 | 586.8 |

Magellan 的 median 变化为 `-0.317%`，mean 变化为 `-0.356%`。
调度有 112 个操作、6 条额外依赖、3 条跨 rank Gloo 依赖。

本 case 未在 H=4096 下重新 profile；求解复用 H=1024、S=2048 的
`comm_units=0.04107446185203048`。结果仅代表一次快速单轮比较。
