# M4: small MoE model, 16-microbatch comparison

This directory contains one single-run comparison between the default 1F1B
schedule and a freshly calibrated Magellan custom schedule with communication
dependencies.

## Configuration

- Hardware: one node, 4 x V100 32 GB
- Parallelism: TP=1, PP=4, DP=1, expert parallel size=1
- Model: MoE GPT, 8 transformer layers
- Hidden size: 1024
- FFN hidden size: 4096
- Attention heads: 16
- Sequence length: 256
- Experts: 8
- Router TopK: 2
- Token dispatcher: all-to-all
- Micro-batch size: 2
- Global batch size: 32
- Number of microbatches per iteration: 16
- Precision: FP16
- Recomputation: disabled
- Artificial communication delay/bandwidth injection: disabled
- Logical solver topology: star edges 0-1, 1-2, and 1-3

Both formal arms ran 20 training iterations. Statistics use iterations 6-19,
giving 14 samples per arm. This is one run, not a repeated-run confidence
interval.

## Calibration and schedule

The M4 profile was measured independently before solving:

- Reference forward compute time: 13.595753815 ms
- Reference adjacent P2P time: 0.178408981 ms
- Normalized communication duration: 0.013122404 compute units

The N=16 static schedule contains 224 operations (128 compute and 96
communication operations). Validation reports an acyclic graph and 14
framework-relevant extra cross-rank communication dependencies.

## Result

| Schedule | Mean (ms) | Median (ms) | Std. dev. (ms) | Samples |
|---|---:|---:|---:|---:|
| 1F1B | 843.379 | 849.250 | 27.017 | 14 |
| Magellan | 706.314 | 707.900 | 23.267 | 14 |

Relative to 1F1B, Magellan reduces mean iteration time by 16.25%, corresponding
to a 1.194x speedup (approximately 19.41% higher sample throughput).

The comparison demonstrates that, for this 16-microbatch MoE case, the custom
schedule remains faster while enforcing additional communication dependencies
through the CPU/Gloo control path. Because all four GPUs are on one node and no
physical WAN impairment is injected, this result measures the implemented
schedule and dependency-control behavior; it does not reproduce a real shared
cross-DC bottleneck.

