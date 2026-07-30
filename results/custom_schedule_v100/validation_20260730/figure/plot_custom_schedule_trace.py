#!/usr/bin/env python3
"""Create a publication-ready figure from the CrossPipe C_TRACE JSONL files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


HERE = Path(__file__).resolve().parent
TRACE_DIR = (
    HERE / "source_trace"
    if (HERE / "source_trace").exists()
    else HERE.parent / "trace"
)
SIGNAL_DATA_PATH = (
    TRACE_DIR / "signal_ready_latency.json"
    if (TRACE_DIR / "signal_ready_latency.json").exists()
    else HERE.parent / "metrics" / "signal_ready_latency.json"
)
ITERATION = 5

COLORS = {
    "f_compute": "#4C78A8",
    "b_compute": "#D9828F",
    "f_comm": "#2A9D8F",
    "b_comm": "#E39C37",
    "signal": "#6D3FC0",
    "dependency": "#3F216B",
    "neutral": "#737373",
    "light": "#E7E7E7",
    "shade": "#F1ECF8",
}


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(TRACE_DIR.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def pair_compute(rows: list[dict], iteration: int) -> list[dict]:
    starts = {}
    intervals = []
    for row in sorted(rows, key=lambda item: item.get("timestamp_ns", 0)):
        if row.get("iteration") != iteration:
            continue
        key = (row.get("pipeline_rank"), row.get("operation"))
        if row.get("event") == "compute_start":
            starts[key] = row
        elif row.get("event") == "compute_end" and key in starts:
            start = starts.pop(key)
            intervals.append(
                {
                    "stage": row["pipeline_rank"],
                    "operation": row["operation"],
                    "direction": row["direction"],
                    "microbatch": row["microbatch"],
                    "start_ns": start["timestamp_ns"],
                    "end_ns": row["timestamp_ns"],
                }
            )
    return intervals


def build_communications(rows: list[dict], iteration: int) -> list[dict]:
    selected = [row for row in rows if row.get("iteration") == iteration]
    receives = defaultdict(list)
    completes = defaultdict(list)
    for row in selected:
        if row.get("event") == "comm_wait_end" and row.get("comm_kind") == "recv":
            receives[(row["pipeline_rank"], row["operation"])].append(row["timestamp_ns"])
        if row.get("event") == "comm_complete":
            completes[(row["pipeline_rank"], row["operation"])].append(row)

    communications = []
    for submit in selected:
        if submit.get("event") != "target_submit":
            continue
        source = submit["src_stage"]
        destination = submit["dst_stage"]
        operation = submit["operation"]
        submit_ns = submit["timestamp_ns"]

        recv_candidates = [
            timestamp
            for timestamp in receives[(destination, operation)]
            if timestamp >= submit_ns
        ]
        completion_candidates = [
            row
            for row in completes[(source, operation)]
            if row["timestamp_ns"] >= submit_ns
        ]
        dependency_completions = [
            row
            for row in completion_candidates
            if row.get("dependency_ids")
        ]
        trigger_complete_ns = (
            min(
                row["timestamp_ns"]
                for row in (dependency_completions or completion_candidates)
            )
            if completion_candidates
            else submit_ns
        )
        receive_ns = min(recv_candidates) if recv_candidates else trigger_complete_ns
        communications.append(
            {
                "operation": operation,
                "direction": submit["direction"],
                "microbatch": submit["microbatch"],
                "source": source,
                "destination": destination,
                "submit_ns": submit_ns,
                "trigger_complete_ns": trigger_complete_ns,
                "receive_ns": receive_ns,
                "dependency_ids": submit.get("dependency_ids", []),
            }
        )
    return communications


def first_event(
    rows: list[dict],
    *,
    iteration: int,
    event: str,
    dependency_id: int | None = None,
    operation: str | None = None,
    dependency_completion: bool = False,
) -> dict:
    candidates = []
    for row in rows:
        if row.get("iteration") != iteration or row.get("event") != event:
            continue
        if dependency_id is not None and row.get("dependency_id") != dependency_id:
            continue
        if operation is not None and row.get("operation") != operation:
            continue
        if dependency_completion and not row.get("dependency_ids"):
            continue
        candidates.append(row)
    if not candidates:
        raise ValueError(
            f"missing event={event}, dependency={dependency_id}, operation={operation}"
        )
    return min(candidates, key=lambda row: row["timestamp_ns"])


def stage_compute_y(stage: int) -> float:
    return 7.25 - 2.0 * stage


def stage_comm_y(stage: int) -> float:
    return 6.55 - 2.0 * stage


def ms(timestamp_ns: int, base_ns: int) -> float:
    return (timestamp_ns - base_ns) / 1e6


def draw_overview(
    ax: plt.Axes,
    rows: list[dict],
    compute: list[dict],
    communications: list[dict],
    base_ns: int,
) -> tuple[float, float]:
    for stage in range(4):
        ax.axhspan(
            stage_comm_y(stage) - 0.31,
            stage_compute_y(stage) + 0.31,
            color="#FAFAFA" if stage % 2 == 0 else "#F5F5F5",
            zorder=0,
        )

    for interval in compute:
        x = ms(interval["start_ns"], base_ns)
        width = ms(interval["end_ns"], interval["start_ns"])
        y = stage_compute_y(interval["stage"])
        color = (
            COLORS["f_compute"]
            if interval["direction"] == "F"
            else COLORS["b_compute"]
        )
        ax.add_patch(
            Rectangle(
                (x, y - 0.25),
                width,
                0.5,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
        )
        if width > 5.0:
            ax.text(
                x + width / 2,
                y,
                f"{interval['direction']}{interval['microbatch']}",
                ha="center",
                va="center",
                color="white",
                fontsize=5.2,
                fontweight="bold",
                clip_on=True,
                zorder=4,
            )

    highlighted = {
        "Comm_F_4_1_2": ("F4  S1→S2", (2, 8)),
        "Comm_B_0_3_2": ("B0  S3→S2", (2, 8)),
        "Comm_F_5_1_2": ("F5  S1→S2", (2, -11)),
    }
    selected_times = []
    for communication in communications:
        x0 = ms(communication["submit_ns"], base_ns)
        x1 = ms(communication["receive_ns"], base_ns)
        source_y = stage_comm_y(communication["source"])
        destination_y = stage_comm_y(communication["destination"])
        color = (
            COLORS["f_comm"]
            if communication["direction"] == "F"
            else COLORS["b_comm"]
        )
        is_highlighted = communication["operation"] in highlighted
        arrow = FancyArrowPatch(
            (x0, source_y),
            (x1, destination_y),
            arrowstyle="-|>",
            mutation_scale=5.5 if not is_highlighted else 7.5,
            linewidth=0.55 if not is_highlighted else 1.6,
            color=COLORS["dependency"] if is_highlighted else color,
            alpha=0.24 if not is_highlighted else 1.0,
            connectionstyle="arc3,rad=0.0",
            zorder=2 if not is_highlighted else 6,
        )
        ax.add_patch(arrow)
        if is_highlighted:
            selected_times.extend((x0, x1))
            label, offset = highlighted[communication["operation"]]
            ax.scatter(
                [x0],
                [source_y],
                s=13,
                marker="D",
                color=COLORS["dependency"],
                edgecolor="white",
                linewidth=0.35,
                zorder=7,
            )
            ax.annotate(
                label,
                (x0, source_y),
                xytext=offset,
                textcoords="offset points",
                fontsize=5.6,
                color=COLORS["dependency"],
                fontweight="bold",
                zorder=8,
            )

    zoom_start = min(selected_times) - 1.0
    zoom_end = max(selected_times) + 1.0
    ax.axvspan(
        zoom_start,
        zoom_end,
        facecolor=COLORS["shade"],
        edgecolor=COLORS["signal"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        alpha=0.72,
        zorder=1,
    )
    ax.text(
        (zoom_start + zoom_end) / 2,
        7.73,
        "causal chain enlarged in b",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["signal"],
    )

    labels, positions = [], []
    for stage in range(4):
        labels.extend((f"S{stage}  compute", f"S{stage}  P2P"))
        positions.extend((stage_compute_y(stage), stage_comm_y(stage)))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.05, 7.95)
    end_ns = max(
        row["timestamp_ns"]
        for row in rows
        if row.get("iteration") == ITERATION and row.get("event") == "iteration_end"
    )
    ax.set_xlim(0, ms(end_ns, base_ns))
    ax.set_xlabel("Time from iteration start (ms)")
    ax.set_title(
        "a   C schedule: four-stage execution trace (iteration 5)",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=6,
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2.5)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.4, alpha=0.7)
    return zoom_start, zoom_end


def add_vertical_dependency(
    ax: plt.Axes,
    x: float,
    y_from: float,
    y_to: float,
    label: str,
) -> None:
    ax.annotate(
        "",
        xy=(x, y_to),
        xytext=(x, y_from),
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["dependency"],
            "linewidth": 1.1,
            "mutation_scale": 8,
        },
        zorder=7,
    )
    ax.text(
        x + 0.08,
        (y_from + y_to) / 2,
        label,
        va="center",
        fontsize=5.6,
        color=COLORS["dependency"],
        fontweight="bold",
    )


def draw_causal_zoom(ax: plt.Axes, rows: list[dict], base_ns: int) -> None:
    trigger_f4 = first_event(
        rows,
        iteration=ITERATION,
        event="comm_complete",
        operation="Comm_F_4_1_2",
        dependency_completion=True,
    )
    submit_f4 = first_event(
        rows,
        iteration=ITERATION,
        event="target_submit",
        operation="Comm_F_4_1_2",
    )
    submit_b0 = first_event(
        rows,
        iteration=ITERATION,
        event="target_submit",
        operation="Comm_B_0_3_2",
    )
    complete_b0 = first_event(
        rows,
        iteration=ITERATION,
        event="comm_complete",
        operation="Comm_B_0_3_2",
        dependency_completion=True,
    )
    submit_f5 = first_event(
        rows,
        iteration=ITERATION,
        event="target_submit",
        operation="Comm_F_5_1_2",
    )
    complete_f5 = first_event(
        rows,
        iteration=ITERATION,
        event="comm_complete",
        operation="Comm_F_5_1_2",
        dependency_completion=True,
    )

    t0 = trigger_f4["timestamp_ns"]
    lane_y = {
        "f4": 4,
        "dep6": 3,
        "b0": 2,
        "dep1": 1,
        "f5": 0,
    }

    def relative(timestamp_ns: int) -> float:
        return (timestamp_ns - t0) / 1e6

    nccl_rows = [
        (
            lane_y["f4"],
            submit_f4["timestamp_ns"],
            trigger_f4["timestamp_ns"],
            COLORS["f_comm"],
        ),
        (
            lane_y["b0"],
            submit_b0["timestamp_ns"],
            complete_b0["timestamp_ns"],
            COLORS["b_comm"],
        ),
        (
            lane_y["f5"],
            submit_f5["timestamp_ns"],
            complete_f5["timestamp_ns"],
            COLORS["f_comm"],
        ),
    ]
    for y, start_ns, end_ns, color in nccl_rows:
        start, end = relative(start_ns), relative(end_ns)
        ax.plot([start, end], [y, y], color=color, linewidth=7, solid_capstyle="butt")
        ax.scatter(
            [start, end],
            [y, y],
            marker="|",
            s=42,
            color="#333333",
            linewidth=0.9,
            zorder=5,
        )

    signal_events = {}
    for dependency_id, key, y in ((6, "dep6", lane_y["dep6"]), (1, "dep1", lane_y["dep1"])):
        send = first_event(
            rows,
            iteration=ITERATION,
            event="signal_send_start",
            dependency_id=dependency_id,
        )
        wait = first_event(
            rows,
            iteration=ITERATION,
            event="signal_wait_start",
            dependency_id=dependency_id,
        )
        receive = first_event(
            rows,
            iteration=ITERATION,
            event="signal_recv",
            dependency_id=dependency_id,
        )
        send_x, wait_x, receive_x = (
            relative(send["timestamp_ns"]),
            relative(wait["timestamp_ns"]),
            relative(receive["timestamp_ns"]),
        )
        ready_x = max(send_x, wait_x)
        ax.plot(
            [send_x, ready_x],
            [y, y],
            color=COLORS["signal"],
            linewidth=1.0,
            linestyle=(0, (2, 2)),
            alpha=0.65,
        )
        ax.plot(
            [ready_x, receive_x],
            [y, y],
            color=COLORS["signal"],
            linewidth=7,
            solid_capstyle="butt",
        )
        ax.scatter(
            [send_x],
            [y],
            marker=">",
            s=18,
            color=COLORS["signal"],
            zorder=6,
        )
        ax.scatter(
            [wait_x],
            [y],
            marker="^",
            s=18,
            color=COLORS["signal"],
            zorder=6,
        )
        ax.scatter(
            [receive_x],
            [y],
            marker="o",
            s=18,
            color=COLORS["signal"],
            edgecolor="white",
            linewidth=0.35,
            zorder=6,
        )
        active_us = (receive_x - ready_x) * 1000
        ax.text(
            (ready_x + receive_x) / 2,
            y + 0.22,
            f"{active_us:.0f} μs",
            ha="center",
            color=COLORS["signal"],
            fontsize=5.5,
            fontweight="bold",
        )
        signal_events[key] = (send_x, wait_x, receive_x)

    add_vertical_dependency(
        ax,
        relative(trigger_f4["timestamp_ns"]),
        lane_y["f4"] - 0.12,
        lane_y["dep6"] + 0.12,
        "complete",
    )
    add_vertical_dependency(
        ax,
        signal_events["dep6"][2],
        lane_y["dep6"] - 0.12,
        lane_y["b0"] + 0.12,
        "unlock",
    )
    add_vertical_dependency(
        ax,
        relative(complete_b0["timestamp_ns"]),
        lane_y["b0"] - 0.12,
        lane_y["dep1"] + 0.12,
        "complete",
    )
    add_vertical_dependency(
        ax,
        signal_events["dep1"][2],
        lane_y["dep1"] - 0.12,
        lane_y["f5"] + 0.12,
        "unlock",
    )

    ax.set_yticks(list(lane_y.values()))
    ax.set_yticklabels(
        [
            "F4  NCCL  S1→S2",
            "dep 6  Gloo  S1→S3",
            "B0  NCCL  S3→S2",
            "dep 1  Gloo  S3→S1",
            "F5  NCCL  S1→S2",
        ]
    )
    ax.set_xlim(relative(submit_f4["timestamp_ns"]) - 0.35, relative(complete_f5["timestamp_ns"]) + 0.3)
    ax.set_ylim(-0.55, 4.55)
    ax.set_xlabel("Time from F4 communication completion (ms)")
    ax.set_title(
        "b   Measured cross-rank dependency chain",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=6,
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2.5)
    ax.grid(axis="x", color="#DEDEDE", linewidth=0.4, alpha=0.7)
    ax.text(
        0.99,
        0.98,
        "dashed: sender posted before receiver wait\nsolid purple: both Gloo endpoints ready",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["neutral"],
    )


def draw_latency(ax: plt.Axes) -> None:
    payload = json.loads(SIGNAL_DATA_PATH.read_text())
    samples = [
        row
        for row in payload["samples"]
        if 3 <= row["iteration"] <= 10
    ]
    gloo = np.array(
        [row["both_endpoints_ready_to_signal_recv_us"] for row in samples]
    )
    dispatch = np.array(
        [row["signal_recv_to_target_submit_us"] for row in samples]
    )
    data = [gloo, dispatch]
    colors = [COLORS["signal"], COLORS["b_comm"]]

    boxes = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"color": COLORS["neutral"], "linewidth": 0.8},
        capprops={"color": COLORS["neutral"], "linewidth": 0.8},
        boxprops={"linewidth": 0.7, "edgecolor": "#444444"},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    rng = np.random.default_rng(7)
    for position, values, color in zip([1, 2], data, colors):
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=6,
            color=color,
            edgecolor="white",
            linewidth=0.2,
            alpha=0.58,
            zorder=3,
        )
        median = np.median(values)
        ax.text(
            position,
            max(values) + 18,
            f"median {median:.0f} μs",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Gloo\nready→recv", "dispatch\nrecv→submit"])
    ax.set_ylabel("Latency (μs)")
    ax.set_ylim(0, max(dispatch) + 82)
    ax.set_title(
        "c   Control-path latency",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=6,
    )
    ax.text(
        0.98,
        0.05,
        "n = 56 dependencies\niterations 3–10",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=COLORS["neutral"],
    )
    ax.tick_params(length=2.5)
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.4, alpha=0.7)

    with (HERE / "source_data_signal_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "iteration",
                "dependency_id",
                "gloo_both_endpoints_ready_to_recv_us",
                "signal_recv_to_target_submit_us",
            ]
        )
        for row in samples:
            writer.writerow(
                [
                    row["iteration"],
                    row["dependency_id"],
                    row["both_endpoints_ready_to_signal_recv_us"],
                    row["signal_recv_to_target_submit_us"],
                ]
            )


def write_iteration_source(
    compute: list[dict],
    communications: list[dict],
    base_ns: int,
) -> None:
    with (HERE / "source_data_iteration5.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "kind",
            "operation",
            "direction",
            "microbatch",
            "stage_or_source",
            "destination",
            "start_ms",
            "end_ms",
            "dependency_ids",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for interval in compute:
            writer.writerow(
                {
                    "kind": "compute",
                    "operation": interval["operation"],
                    "direction": interval["direction"],
                    "microbatch": interval["microbatch"],
                    "stage_or_source": interval["stage"],
                    "destination": "",
                    "start_ms": f"{ms(interval['start_ns'], base_ns):.6f}",
                    "end_ms": f"{ms(interval['end_ns'], base_ns):.6f}",
                    "dependency_ids": "",
                }
            )
        for communication in communications:
            writer.writerow(
                {
                    "kind": "communication",
                    "operation": communication["operation"],
                    "direction": communication["direction"],
                    "microbatch": communication["microbatch"],
                    "stage_or_source": communication["source"],
                    "destination": communication["destination"],
                    "start_ms": f"{ms(communication['submit_ns'], base_ns):.6f}",
                    "end_ms": f"{ms(communication['receive_ns'], base_ns):.6f}",
                    "dependency_ids": ";".join(
                        str(value) for value in communication["dependency_ids"]
                    ),
                }
            )


def main() -> None:
    rows = load_rows()
    selected = [row for row in rows if row.get("iteration") == ITERATION]
    base_ns = min(
        row["timestamp_ns"]
        for row in selected
        if row.get("event") == "iteration_start"
    )
    compute = pair_compute(rows, ITERATION)
    communications = build_communications(rows, ITERATION)
    if len(compute) != 64 or len(communications) != 48:
        raise ValueError(
            f"unexpected trace coverage: {len(compute)} compute, "
            f"{len(communications)} communication"
        )

    fig = plt.figure(figsize=(7.2, 5.75))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.45, 1.0],
        width_ratios=[2.55, 1.0],
        hspace=0.42,
        wspace=0.36,
    )
    overview_ax = fig.add_subplot(grid[0, :])
    zoom_ax = fig.add_subplot(grid[1, 0])
    latency_ax = fig.add_subplot(grid[1, 1])

    draw_overview(overview_ax, rows, compute, communications, base_ns)
    draw_causal_zoom(zoom_ax, rows, base_ns)
    draw_latency(latency_ax)
    write_iteration_source(compute, communications, base_ns)

    legend_handles = [
        Patch(facecolor=COLORS["f_compute"], label="Forward compute"),
        Patch(facecolor=COLORS["b_compute"], label="Backward compute"),
        Patch(facecolor=COLORS["f_comm"], label="Forward communication"),
        Patch(facecolor=COLORS["b_comm"], label="Backward communication"),
        Patch(facecolor=COLORS["signal"], label="Gloo control signal"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        fontsize=6,
        columnspacing=1.1,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.955, bottom=0.115)

    stem = HERE / "custom_schedule_trace"
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    caption = (
        "CrossPipe custom schedule C enforces additional communication dependencies. "
        "(a) Host-side trace for a representative steady-state iteration on four "
        "pipeline stages; arrows connect sender submission to receiver completion. "
        "(b) Enlargement of the measured alternating dependency chain on the shared "
        "star-topology link: forward communication F4 (S1→S2) completes before a "
        "Gloo signal unlocks backward communication B0 (S3→S2), whose completion "
        "then signals and unlocks F5 (S1→S2). Solid purple segments measure Gloo "
        "completion after both endpoints are posted; dashed segments indicate that "
        "the sender was posted before the receiver began waiting. (c) Control-path "
        "latencies over 56 remote dependencies from steady-state iterations 3–10. "
        "Compute intervals and communication events are host-side instrumentation, "
        "not GPU kernel-duration measurements."
    )
    (HERE / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "iteration": ITERATION,
                "compute_intervals": len(compute),
                "communications": len(communications),
                "outputs": [
                    str(stem.with_suffix(extension))
                    for extension in (".svg", ".pdf", ".png")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
