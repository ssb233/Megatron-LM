#!/usr/bin/env python3
"""Convert CrossPipe custom-schedule JSONL files to Chrome Trace JSON."""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


LANES = {
    "compute": 1,
    "nccl": 2,
    "gloo": 3,
    "scheduler": 4,
}


def _lane(event):
    name = event["event"]
    if name.startswith("compute_"):
        return "compute"
    if name.startswith("signal_"):
        return "gloo"
    if name.startswith("comm_") or name == "target_submit":
        return "nccl"
    return "scheduler"


def _args(event):
    return {
        key: value
        for key, value in event.items()
        if key
        not in {
            "timestamp_ns",
            "event",
            "rank",
            "pipeline_rank",
        }
    }


def load_events(trace_dir):
    events = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "rank_*.jsonl"))):
        with open(path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
    if not events:
        raise ValueError(f"no rank_*.jsonl files found in {trace_dir}")
    return sorted(events, key=lambda event: event["timestamp_ns"])


def convert(events):
    origin_ns = min(event["timestamp_ns"] for event in events)
    chrome = []
    starts = defaultdict(list)

    for event in events:
        rank = event["rank"]
        lane = _lane(event)
        timestamp_us = (event["timestamp_ns"] - origin_ns) / 1000
        name = event["event"]
        args = _args(event)

        if name == "compute_start":
            key = (
                rank,
                event["iteration"],
                event["operation"],
            )
            starts[key].append(event)
            continue
        if name == "compute_end":
            key = (
                rank,
                event["iteration"],
                event["operation"],
            )
            if not starts[key]:
                raise ValueError(f"compute_end without compute_start: {key}")
            start = starts[key].pop(0)
            chrome.append(
                {
                    "name": event["operation"],
                    "cat": "compute_host",
                    "ph": "X",
                    "pid": rank,
                    "tid": LANES["compute"],
                    "ts": (start["timestamp_ns"] - origin_ns) / 1000,
                    "dur": (
                        event["timestamp_ns"] - start["timestamp_ns"]
                    )
                    / 1000,
                    "args": args,
                }
            )
            continue

        chrome.append(
            {
                "name": name,
                "cat": lane,
                "ph": "i",
                "s": "t",
                "pid": rank,
                "tid": LANES[lane],
                "ts": timestamp_us,
                "args": args,
            }
        )

        if name == "comm_complete" and event.get("dependency_ids"):
            for dependency_id in event["dependency_ids"]:
                flow_id = f"{event['iteration']}:{dependency_id}"
                chrome.append(
                    {
                        "name": f"dependency_{dependency_id}",
                        "cat": "dependency",
                        "ph": "s",
                        "id": flow_id,
                        "pid": rank,
                        "tid": LANES["nccl"],
                        "ts": timestamp_us,
                        "args": args,
                    }
                )
        elif name == "target_submit" and "dependency_ids" in event:
            for dependency_id in event["dependency_ids"]:
                flow_id = f"{event['iteration']}:{dependency_id}"
                chrome.append(
                    {
                        "name": f"dependency_{dependency_id}",
                        "cat": "dependency",
                        "ph": "f",
                        "bp": "e",
                        "id": flow_id,
                        "pid": rank,
                        "tid": LANES["nccl"],
                        "ts": timestamp_us,
                        "args": args,
                    }
                )

    dangling = [key for key, values in starts.items() if values]
    if dangling:
        raise ValueError(f"unmatched compute_start events: {dangling[:8]}")
    return {"traceEvents": chrome, "displayTimeUnit": "ms"}


def _aggregate(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(ordered),
        "min_us": ordered[0],
        "median_us": statistics.median(ordered),
        "p95_us": ordered[p95_index],
        "max_us": ordered[-1],
        "mean_us": statistics.mean(ordered),
    }


def summarize_dependencies(events):
    completion = {}
    target_submit = {}
    signal_send = {}
    signal_recv = {}
    for event in events:
        iteration = event["iteration"]
        if event["event"] == "comm_complete":
            for dependency_id in event.get("dependency_ids", []):
                completion[(iteration, dependency_id)] = event["timestamp_ns"]
        elif event["event"] == "target_submit":
            for dependency_id in event.get("dependency_ids", []):
                target_submit[(iteration, dependency_id)] = event["timestamp_ns"]
        elif event["event"] == "signal_send_start":
            signal_send[(iteration, event["dependency_id"])] = event[
                "timestamp_ns"
            ]
        elif event["event"] == "signal_recv":
            signal_recv[(iteration, event["dependency_id"])] = event[
                "timestamp_ns"
            ]

    control_latencies = []
    gloo_latencies = []
    samples = []
    for key in sorted(set(completion) & set(target_submit)):
        control_us = (target_submit[key] - completion[key]) / 1000
        sample = {
            "iteration": key[0],
            "dependency_id": key[1],
            "completion_to_target_submit_us": control_us,
        }
        control_latencies.append(control_us)
        if key in signal_send and key in signal_recv:
            gloo_us = (signal_recv[key] - signal_send[key]) / 1000
            sample["signal_send_to_recv_us"] = gloo_us
            gloo_latencies.append(gloo_us)
        samples.append(sample)

    return {
        "samples": samples,
        "completion_to_target_submit": _aggregate(control_latencies),
        "signal_send_to_recv": _aggregate(gloo_latencies),
        "clock": "time.perf_counter_ns; comparable across processes on one node",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <trace_dir>/custom_schedule.chrome.json",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Default: <trace_dir>/custom_schedule.summary.json",
    )
    args = parser.parse_args()

    trace_dir = os.path.abspath(args.trace_dir)
    output = args.output or os.path.join(
        trace_dir,
        "custom_schedule.chrome.json",
    )
    events = load_events(trace_dir)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(convert(events), stream)
    summary_output = args.summary_output or os.path.join(
        trace_dir,
        "custom_schedule.summary.json",
    )
    with open(summary_output, "w", encoding="utf-8") as stream:
        json.dump(summarize_dependencies(events), stream, indent=2)
    print(output)
    print(summary_output)


if __name__ == "__main__":
    main()
