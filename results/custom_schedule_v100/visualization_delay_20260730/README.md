# CrossPipe PP=4 通信依赖可视化实验

本目录归档自定义调度 C 的“方式 1”可视化实验：使用 CrossPipe 自带的
NCCL-stream 时延注入，不增加无效通信张量，分别注入 `0.5 × F` 和 `1.0 × F`
的传输时延，使通信和额外通信依赖在论文 trace 中清晰可见。

该实验用于验证执行顺序和生成说明图，不用于吞吐量或真实跨 DC 性能比较。

## 实验配置

| 项目 | 配置 |
|---|---|
| 真机 | 单节点 4 × Tesla V100-SXM2-32GB |
| 并行配置 | PP=4, TP=1, DP=1 |
| Microbatch | 8，MBS=1，GBS=8 |
| 重计算 | 关闭 |
| 模型 | 8 layers, hidden 512, FFN 2048, 8 heads, sequence length 256 |
| 自定义 PyTorch | `/home/songxb26/mnist/pytorch-corsspipe`，`2.5.0a0+git00abf21` |
| 逻辑拓扑 | `0–1, 1–2, 1–3` |
| Stage/DC 映射 | S0/DC0/rank0，S1/DC1/rank1，S2/DC2/rank2，S3/DC3/rank3 |
| 调度输入 | `replay_order_pp4_n8_star.json` |
| 依赖输入 | `notification_deps_pp4_n8_star.json` |
| 时延组 | `0.5 × F` 和 `1.0 × F`，每组 8 个实测迭代 |

profile 得到的前向 stage 时间为约 `6.900 ms`，因此两组配置的注入传输时延分别为：

- `0.5 × F = 3.450 ms`
- `1.0 × F = 6.900 ms`

## 结果

正式运行完成 iteration 3–18：

- iteration 3–10：`0.5 × F`
- iteration 11–18：`1.0 × F`
- 两档均正常完成，无 NCCL/Gloo timeout；
- 延迟实验产生 `16 × 7 = 112` 个 remote dependency 样本；
- 连同零时延基线的 56 个样本，共检查 168 条依赖，违例为 0。

严格校验的顺序为：

```text
trigger NCCL complete
  <= Gloo signal_send_start
  <= Gloo signal_recv
  <= target NCCL target_submit
```

## 图中说明的通信依赖

panel b 聚焦 dependency 6：

```text
Comm_F_4_1_2 完成
  └─ rank1/stage1 发送 Gloo control signal
       └─ rank3/stage3 收到 signal
            └─ 提交 Comm_B_0_3_2
```

即强制：

```text
F4: Stage1 → Stage2
  before
B0: Stage3 → Stage2
```

逻辑星型拓扑中，`DC1→DC2` 使用链路 `1→2`；`DC3→DC2` 使用路径
`3→1→2`。两条通信共享有向链路 `1→2`，因此无竞争调度要求它们串行。

执行器同时在共享接收端 Stage2 延迟提交受约束的目标 recv：先完成 trigger recv，
再下发 target recv。这避免了未匹配的提前 NCCL recv 占住通信执行路径，同时不改变
求解器规定的发送完成顺序。

## 图和计时口径

- `figure/custom_schedule_delay_trace.svg`：论文排版首选，文字可编辑。
- `figure/custom_schedule_delay_trace.pdf`：LaTeX/论文系统直接引用。
- `figure/custom_schedule_delay_trace.png`：300 dpi 预览。
- panel a 的计算区间和事件位置来自 host-side trace；通信条在 host trace 的基础上
  包含配置的人工传输窗口，以便可视化。
- panel b 的 NCCL complete、control send/recv、target submit 均为实际 CPU 打点。
- panel c 展示配置值，不声称是实际网络测量值。
- 如需精确 GPU kernel 时间，应使用 Nsight Systems/NVTX；本图用于证明调度下发与
  control dependency 的执行顺序。

## 目录结构

```text
inputs/       调度和通信依赖 JSON
analysis/     依赖检查汇总和逐样本 CSV
trace/        四个 rank 的原始 JSONL trace
figure/       SVG、PDF、PNG、caption 和图源数据
metrics/      CrossPipe exp_final.json
plans/        各 rank 静态执行计划和 schedule_init.svg
logs/         train.log、run.info、exec_plan_init.log
scripts/      trace 分析和绘图脚本快照
MANIFEST.sha256
```

## 复现

```bash
cd /home/songxb26/mnist/crosspipi-magellan
unset CDC_EXP_PER_CFG_TEST_ITERS CUSTOM_SCHEDULE_TRACE_FLUSH_EACH_EVENT
test_crossdc/custom_schedule_v100/run_custom_schedule.sh \
  C_VIS \
  /home/songxb26/mnist/crosspipi-magellan/runs/custom_schedule_v100/visualization_delay_20260730
```

运行器会校验导入的 PyTorch 路径和版本，并使用 `num_dc=4`、
`pp_stages_per_dc=1 1 1 1`。人工时延仅在 `C_VIS` 模式启用。
