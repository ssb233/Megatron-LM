#!/usr/bin/env python3
"""Compare default, custom-order, and custom-dependency CrossPipe runs."""

import argparse
import json
import math
import statistics


def _flatten_numbers(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _flatten_numbers(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _flatten_numbers(nested)


def load_iteration_seconds(path, warmup=0):
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if "iter_time" not in document:
        raise ValueError(f"{path} does not contain iter_time")
    values = list(_flatten_numbers(document["iter_time"]))
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    values = values[warmup:]
    if not values:
        raise ValueError(f"{path} has no iteration samples after warmup")
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{path} contains invalid iteration times")
    return values


def summarize_samples(samples):
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(ordered),
        "min_ms": ordered[0] * 1000,
        "median_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[p95_index] * 1000,
        "max_ms": ordered[-1] * 1000,
        "mean_ms": statistics.mean(ordered) * 1000,
    }


def summarize_runs(a_samples, b_samples, c_samples, signal_summary=None):
    runs = {
        "A_default_1f1b": summarize_samples(a_samples),
        "B_custom_order": summarize_samples(b_samples),
        "C_custom_order_and_dependency": summarize_samples(c_samples),
    }
    a_median = runs["A_default_1f1b"]["median_ms"]
    b_median = runs["B_custom_order"]["median_ms"]
    c_median = runs["C_custom_order_and_dependency"]["median_ms"]

    def delta(lhs, rhs):
        absolute = lhs - rhs
        return {
            "milliseconds": absolute,
            "percent": absolute / rhs * 100,
        }

    result = {
        "runs": runs,
        "deltas": {
            "B_minus_A": delta(b_median, a_median),
            "C_minus_A": delta(c_median, a_median),
            "C_minus_B": delta(c_median, b_median),
        },
        "interpretation": {
            "B_minus_A": "custom order overhead/change versus default 1F1B",
            "C_minus_B": (
                "extra communication serialization plus Gloo signal overhead"
            ),
        },
    }
    if signal_summary is not None:
        result["signal"] = signal_summary
    return result


def _format_table(summary):
    lines = [
        "run                              count   median_ms   p95_ms",
        "----------------------------------------------------------------",
    ]
    for name, stats in summary["runs"].items():
        lines.append(
            f"{name:32} {stats['count']:5d} "
            f"{stats['median_ms']:11.3f} {stats['p95_ms']:9.3f}"
        )
    lines.append("")
    for name, delta in summary["deltas"].items():
        lines.append(
            f"{name}: {delta['milliseconds']:+.3f} ms "
            f"({delta['percent']:+.3f}%)"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="A exp_final.json")
    parser.add_argument("--run-b", required=True, help="B exp_final.json")
    parser.add_argument("--run-c", required=True, help="C exp_final.json")
    parser.add_argument("--signal-summary", default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signal_summary = None
    if args.signal_summary is not None:
        with open(args.signal_summary, "r", encoding="utf-8") as stream:
            signal_summary = json.load(stream)
    summary = summarize_runs(
        load_iteration_seconds(args.run_a, args.warmup),
        load_iteration_seconds(args.run_b, args.warmup),
        load_iteration_seconds(args.run_c, args.warmup),
        signal_summary,
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(_format_table(summary))
    print(args.output)


if __name__ == "__main__":
    main()
