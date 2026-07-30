#!/usr/bin/env python3
"""Create a publication trace for delayed custom communication dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tools.analyze_custom_schedule_delay_trace import _load_events


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.8

COLORS = {
    "forward_compute": "#3B82C4",
    "backward_compute": "#E68632",
    "forward_comm": "#2A9D8F",
    "backward_comm": "#9467BD",
    "signal": "#D62728",
    "grid": "#D9DEE7",
}


def _iteration_events(trace_dir: Path, iteration: int) -> list[dict[str, Any]]:
    return [
        event
        for event in _load_events(trace_dir)
        if int(event.get("iteration", -1)) == iteration
    ]


def _paired_intervals(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    starts: dict[tuple[int, str], dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: int(item["timestamp_ns"])):
        event_type = event.get("event")
        operation = event.get("operation")
        if not operation:
            continue
        key = (int(event.get("rank", -1)), str(operation))
        if event_type == "compute_start":
            starts[key] = event
        elif event_type == "compute_end" and key in starts:
            start = starts.pop(key)
            intervals.append(
                {
                    "kind": "compute",
                    "rank": key[0],
                    "operation": key[1],
                    "start_ns": int(start["timestamp_ns"]),
                    "end_ns": int(event["timestamp_ns"]),
                    "direction": start.get("direction", ""),
                    "microbatch": int(start.get("microbatch", -1)),
                }
            )

    submissions = [
        event for event in events if event.get("event") == "target_submit"
    ]
    completions: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") == "comm_complete":
            completions[
                (int(event.get("rank", -1)), str(event.get("operation", "")))
            ].append(event)
    for submission in submissions:
        key = (
            int(submission.get("rank", -1)),
            str(submission.get("operation", "")),
        )
        start_ns = int(submission["timestamp_ns"])
        candidates = [
            event
            for event in completions.get(key, [])
            if int(event["timestamp_ns"]) >= start_ns
        ]
        if not candidates:
            continue
        completion = min(
            candidates, key=lambda event: int(event["timestamp_ns"])
        )
        intervals.append(
            {
                "kind": "comm",
                "rank": key[0],
                "operation": key[1],
                "start_ns": start_ns,
                "end_ns": int(completion["timestamp_ns"]),
                "direction": submission.get("direction", ""),
                "microbatch": int(submission.get("microbatch", -1)),
                "src_stage": int(submission.get("src_stage", -1)),
                "dst_stage": int(submission.get("dst_stage", -1)),
            }
        )
    return [item for item in intervals if item["end_ns"] > item["start_ns"]]


def _panel_timeline(
    axis,
    events: list[dict[str, Any]],
    representative_iteration: int,
) -> None:
    intervals = _paired_intervals(events)
    if not intervals:
        axis.text(
            0.5,
            0.5,
            "Timeline events unavailable in synthetic trace",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    origin = min(item["start_ns"] for item in intervals)
    for item in intervals:
        start_ms = (item["start_ns"] - origin) / 1e6
        width_ms = (item["end_ns"] - item["start_ns"]) / 1e6
        direction = str(item["direction"]).lower()
        if item["kind"] == "compute":
            color = COLORS[
                "forward_compute" if direction == "f" else "backward_compute"
            ]
            height = 0.52
            y = item["rank"] - height / 2
        else:
            color = COLORS[
                "forward_comm" if direction == "f" else "backward_comm"
            ]
            height = 0.22
            y = item["rank"] + 0.22
        axis.broken_barh(
            [(start_ms, width_ms)],
            (y, height),
            facecolors=color,
            edgecolors="white",
            linewidth=0.25,
        )
    axis.set_yticks(range(4), [f"Stage {rank}" for rank in range(4)])
    axis.set_ylim(-0.55, 3.55)
    axis.invert_yaxis()
    axis.set_xlabel("Time from first event (ms)")
    axis.set_title(
        f"a  Measured PP=4 timeline, 1.0×F (iteration {representative_iteration})",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Patch(color=COLORS["forward_compute"], label="Forward compute"),
            Patch(color=COLORS["backward_compute"], label="Backward compute"),
            Patch(color=COLORS["forward_comm"], label="Forward P2P"),
            Patch(color=COLORS["backward_comm"], label="Backward P2P"),
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
    )


def _panel_dependency(axis, summary: dict[str, Any], iteration: int) -> None:
    records = [
        record
        for record in summary["dependency_records"]
        if float(record["factor"]) == 1.0
        and int(record["iteration"]) == iteration
        and int(record["dependency_id"]) in {6, 1}
    ]
    if not records:
        axis.text(
            0.5,
            0.5,
            "Dependency-chain events unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    origin = min(int(record["trigger_complete_ns"]) for record in records)
    lane = {1: 1.0, 3: 0.0}
    for record in sorted(records, key=lambda item: int(item["trigger_complete_ns"])):
        sender = int(record["sender_rank"])
        receiver = int(record["receiver_rank"])
        points = {
            "NCCL complete": int(record["trigger_complete_ns"]),
            "control send": int(record["signal_send_start_ns"]),
            "control recv": int(record["signal_recv_ns"]),
            "target submit": int(record["target_submit_ns"]),
        }
        x_values = {
            label: (timestamp - origin) / 1_000.0
            for label, timestamp in points.items()
        }
        axis.plot(
            [x_values["NCCL complete"], x_values["control send"]],
            [lane[sender], lane[sender]],
            color=COLORS["signal"],
            linewidth=2.2,
        )
        axis.annotate(
            "",
            xy=(x_values["control recv"], lane[receiver]),
            xytext=(x_values["control send"], lane[sender]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["signal"],
                "lw": 1.6,
                "connectionstyle": "arc3,rad=0.08",
            },
        )
        axis.plot(
            [x_values["control recv"], x_values["target submit"]],
            [lane[receiver], lane[receiver]],
            color=COLORS["signal"],
            linewidth=2.2,
        )
        for label, x_value in x_values.items():
            event_lane = (
                lane[sender]
                if label in {"NCCL complete", "control send"}
                else lane[receiver]
            )
            axis.scatter(
                x_value,
                event_lane,
                s=32,
                color=COLORS["signal"],
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
        midpoint = (
            x_values["control send"] + x_values["control recv"]
        ) / 2
        axis.text(
            midpoint,
            0.5,
            f"dep {record['dependency_id']}",
            color=COLORS["signal"],
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    axis.set_yticks([1.0, 0.0], ["Stage 1", "Stage 3"])
    axis.set_ylim(-0.45, 1.45)
    axis.set_xlabel("Time from first trigger completion (µs)")
    axis.set_title(
        "b  Measured Gloo control chain enforces dependency 6 → 1",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    axis.set_axisbelow(True)


def _panel_duration(axis, rows: list[dict[str, Any]]) -> None:
    factors = [0.0, 0.5, 1.0]
    labels = ["0×F", "0.5×F", "1.0×F"]
    values = [
        [
            float(row["communication_duration_us"])
            for row in rows
            if float(row["factor"]) == factor
        ]
        for factor in factors
    ]
    positions = range(1, 4)
    box = axis.boxplot(
        values,
        positions=list(positions),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": "#667085"},
        capprops={"color": "#667085"},
    )
    for patch, color in zip(
        box["boxes"], ["#A7C7E7", "#66C2A5", "#2A9D8F"]
    ):
        patch.set_facecolor(color)
        patch.set_edgecolor("#475467")
    for position, samples in zip(positions, values):
        axis.scatter(
            [position] * len(samples),
            samples,
            s=15,
            alpha=0.65,
            color="#344054",
            zorder=3,
        )
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Comm_F_4_1_2 duration (µs)")
    axis.set_title(
        "c  Injected transfer delay makes communication visible",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    axis.set_axisbelow(True)


def _write_figure_source_data(
    output_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    fields = [
        "panel",
        "factor",
        "iteration",
        "dependency_id",
        "operation",
        "metric",
        "value",
        "unit",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "panel": "c",
                    "factor": row["factor"],
                    "iteration": row["iteration"],
                    "dependency_id": 6,
                    "operation": row["operation"],
                    "metric": "communication_duration",
                    "value": row["communication_duration_us"],
                    "unit": "us",
                }
            )
        for record in summary["dependency_records"]:
            for metric in (
                "signal_dispatch_us",
                "signal_transit_us",
                "receiver_release_us",
            ):
                writer.writerow(
                    {
                        "panel": "b",
                        "factor": record["factor"],
                        "iteration": record["iteration"],
                        "dependency_id": record["dependency_id"],
                        "operation": f"{record['trigger']} -> {record['target']}",
                        "metric": metric.removesuffix("_us"),
                        "value": record[metric],
                        "unit": "us",
                    }
                )


def create_figure(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    trace_dir: Path | str,
    output_dir: Path | str,
) -> list[Path]:
    """Create SVG, PDF, PNG, caption, and auditable source data."""

    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    representative = int(summary["representative_iterations"]["1.0"])
    events = _iteration_events(trace_dir, representative)

    figure = plt.figure(figsize=(12.2, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=[1.65, 1.0, 1.0])
    axis_timeline = figure.add_subplot(grid[0])
    axis_dependency = figure.add_subplot(grid[1])
    axis_duration = figure.add_subplot(grid[2])
    _panel_timeline(axis_timeline, events, representative)
    _panel_dependency(axis_dependency, summary, representative)
    _panel_duration(axis_duration, rows)
    figure.suptitle(
        "Custom pipeline schedule with explicit communication dependencies",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.995,
        0.003,
        "Artificial transfer delay for visualization only",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#B42318",
        fontweight="bold",
    )

    stem = output_dir / "custom_schedule_delay_trace"
    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
    ]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], dpi=300, bbox_inches="tight")
    plt.close(figure)

    caption = output_dir / "figure_caption.txt"
    caption.write_text(
        "Custom PP=4 schedule trace with explicit cross-rank communication "
        "dependencies. Panel a shows measured compute and P2P events with a "
        "1.0×F artificial NCCL-stream transfer delay. Panel b shows the Gloo "
        "control path from trigger NCCL completion to target communication "
        "submission for dependencies 6 and 1. Panel c reports the selected "
        "communication duration at 0×F, 0.5×F, and 1.0×F. Artificial transfer "
        "delay is for visualization only and is excluded from throughput "
        "comparisons.\n",
        encoding="utf-8",
    )
    source_data = output_dir / "figure_source_data.csv"
    _write_figure_source_data(source_data, rows, summary)
    return outputs + [caption, source_data]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    trace_dir = args.trace_dir
    if trace_dir is None:
        trace_dir = Path(summary["trace_directories"]["delayed"])
    outputs = create_figure(
        summary,
        _read_rows(args.source_data),
        trace_dir,
        args.output_dir,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
