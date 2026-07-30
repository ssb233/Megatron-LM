# M3H4096S2048：MoE 单轮实验

配置：PP=4、TP=1、DP=1、8 层、H=4096、FFN=4096、heads=16、
S=2048、MBS=1、GBS/N=8、8 experts、top-k=2、EP=1、FP16、
无重计算、无通信注入。

每个 arm 运行 20 iterations，统计 iterations 6–19：

| arm | mean (ms) | median (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|
| 1F1B | 721.550 | 721.500 | 3.780 | 716.6 | 728.4 |
| Magellan | 729.386 | 729.950 | 2.773 | 724.5 | 733.3 |

Magellan 的 median 变化为 `+1.171%`，mean 变化为 `+1.086%`。
调度有 112 个操作、4 条额外依赖、3 条跨 rank Gloo 依赖。

本 case 未在 H=4096 下重新 profile；求解复用 H=1024、S=2048 的
`comm_units=0.017679909587792298`。结果仅代表一次快速单轮比较。
