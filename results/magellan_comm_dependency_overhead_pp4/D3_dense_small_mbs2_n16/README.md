# D3: dense small model, 16-microbatch comparison

This directory contains one single-run comparison between the default 1F1B
schedule and a freshly calibrated Magellan custom schedule with communication
dependencies.

## Configuration

- Hardware: one node, 4 x V100 32 GB
- Parallelism: TP=1, PP=4, DP=1
- Model: dense GPT, 8 transformer layers
- Hidden size: 1024
- FFN hidden size: 4096
- Attention heads: 16
- Sequence length: 256
- Micro-batch size: 2
- Global batch size: 32
- Number of microbatches per iteration: 16
- Precision: FP16
- Recomputation: disabled
- Artificial communication delay/bandwidth injection: disabled
- Logical solver topology: star edges 0-1, 1-2, and 1-3
- Solver bandwidth/latency: 300 Gbps and 10 ms (latency delay units are set to
  zero for this no-injection experiment)

Both formal arms ran 20 training iterations. Statistics use iterations 6-19,
giving 14 samples per arm. This is one run, not a repeated-run confidence
interval.

## Calibration and schedule

The D3 profile was measured independently before solving:

- Reference forward compute time: 5.753095727 ms
- Reference adjacent P2P time: 0.304605378 ms
- Normalized communication duration: 0.052946343 compute units

The N=16 static schedule contains 224 operations (128 compute and 96
communication operations). Validation reports an acyclic graph and 14
framework-relevant extra cross-rank communication dependencies.

## Result

| Schedule | Mean (ms) | Median (ms) | Std. dev. (ms) | Samples |
|---|---:|---:|---:|---:|
| 1F1B | 562.393 | 563.800 | 10.824 | 14 |
| Magellan | 476.364 | 491.800 | 27.715 | 14 |

Relative to 1F1B, Magellan reduces mean iteration time by 15.30%, corresponding
to a 1.181x speedup (approximately 18.06% higher sample throughput).

The comparison demonstrates that, for this 16-microbatch case, the custom
schedule remains faster even while enforcing additional communication
dependencies through the CPU/Gloo control path. Because all four GPUs are on
one node and no physical WAN impairment is injected, this result measures the
implemented schedule and dependency-control behavior; it does not reproduce a
real shared cross-DC bottleneck.

