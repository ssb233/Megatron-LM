#!/usr/bin/env python3
"""Analyze custom-schedule traces with visualization-only transfer delay."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SELECTED_DEPENDENCY_ID = 6
SELECTED_COMMUNICATION = "Comm_F_4_1_2"


def _load_events(trace_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(trace_dir).glob("rank_*.jsonl"))
    if not paths:
        raise ValueError(f"no rank_*.jsonl files found in {trace_dir}")
    events: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {error}"
                ) from error
    return events


def _events_by_iteration(
    events: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        iteration = int(event.get("iteration", -1))
        if iteration >= 0:
            grouped[iteration].append(event)
    return dict(grouped)


def _delay_configs(
    grouped: dict[int, list[dict[str, Any]]],
    *,
    allow_missing: bool,
) -> dict[int, dict[str, float]]:
    configs: dict[int, dict[str, float]] = {}
    for iteration, events in grouped.items():
        records = [event for event in events if event.get("event") == "delay_config"]
        if not records:
            if not allow_missing and any(
                event.get("event") == "signal_send_start" for event in events
            ):
                raise ValueError(
                    f"delayed trace iteration {iteration} is missing delay_config"
                )
            continue
        normalized = {
            (
                float(record["latency_factor"]),
                float(record["bandwidth_factor"]),
                float(record["latency_seconds"]),
                float(record["bandwidth_seconds"]),
                float(record["forward_stage_seconds"]),
            )
            for record in records
        }
        if len(normalized) != 1:
            raise ValueError(
                f"inconsistent delay_config records in iteration {iteration}"
            )
        latency_factor, bandwidth_factor, latency_seconds, bandwidth_seconds, forward = (
            normalized.pop()
        )
        configs[iteration] = {
            "latency_factor": latency_factor,
            "bandwidth_factor": bandwidth_factor,
            "latency_seconds": latency_seconds,
            "bandwidth_seconds": bandwidth_seconds,
            "forward_stage_seconds": forward,
        }
    return configs


def _single_event(
    events: list[dict[str, Any]],
    *,
    description: str,
    predicate,
) -> dict[str, Any]:
    matches = [event for event in events if predicate(event)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {description}, found {len(matches)}"
        )
    return matches[0]


def _analyze_iteration(
    iteration: int,
    events: list[dict[str, Any]],
    factor: float,
    config: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    dependencies: list[dict[str, Any]] = []
    selected_row = None
    send_events = [
        event for event in events if event.get("event") == "signal_send_start"
    ]
    for send_event in send_events:
        dependency_id = int(send_event["dependency_id"])
        sender_rank = int(send_event["rank"])
        receiver_rank = int(send_event["peer_rank"])
        trigger = str(send_event["trigger"])
        target = str(send_event["target"])
        trigger_complete = _single_event(
            events,
            description=f"trigger completion for dependency {dependency_id}",
            predicate=lambda event: (
                event.get("event") == "comm_complete"
                and event.get("operation") == trigger
                and int(event.get("rank", -1)) == sender_rank
                and dependency_id in event.get("dependency_ids", [])
            ),
        )
        trigger_submit = _single_event(
            events,
            description=f"trigger submit for dependency {dependency_id}",
            predicate=lambda event: (
                event.get("event") == "target_submit"
                and event.get("operation") == trigger
                and int(event.get("rank", -1)) == sender_rank
            ),
        )
        signal_recv = _single_event(
            events,
            description=f"signal receive for dependency {dependency_id}",
            predicate=lambda event: (
                event.get("event") == "signal_recv"
                and int(event.get("dependency_id", -1)) == dependency_id
                and int(event.get("rank", -1)) == receiver_rank
                and int(event.get("peer_rank", -1)) == sender_rank
            ),
        )
        target_submit = _single_event(
            events,
            description=f"target submit for dependency {dependency_id}",
            predicate=lambda event: (
                event.get("event") == "target_submit"
                and event.get("operation") == target
                and int(event.get("rank", -1)) == receiver_rank
                and dependency_id in event.get("dependency_ids", [])
            ),
        )
        timestamps = [
            int(trigger_complete["timestamp_ns"]),
            int(send_event["timestamp_ns"]),
            int(signal_recv["timestamp_ns"]),
            int(target_submit["timestamp_ns"]),
        ]
        if timestamps != sorted(timestamps):
            raise ValueError(
                "dependency ordering violation "
                f"in iteration {iteration}, dependency {dependency_id}: "
                f"{timestamps}"
            )
        dependency_record = {
            "factor": factor,
            "iteration": iteration,
            "dependency_id": dependency_id,
            "trigger": trigger,
            "target": target,
            "sender_rank": sender_rank,
            "receiver_rank": receiver_rank,
            "trigger_complete_ns": timestamps[0],
            "signal_send_start_ns": timestamps[1],
            "signal_recv_ns": timestamps[2],
            "target_submit_ns": timestamps[3],
            "signal_dispatch_us": (timestamps[1] - timestamps[0]) / 1_000.0,
            "signal_transit_us": (timestamps[2] - timestamps[1]) / 1_000.0,
            "receiver_release_us": (timestamps[3] - timestamps[2]) / 1_000.0,
        }
        dependencies.append(dependency_record)
        if (
            dependency_id == SELECTED_DEPENDENCY_ID
            and trigger == SELECTED_COMMUNICATION
        ):
            submit_ns = int(trigger_submit["timestamp_ns"])
            complete_ns = int(trigger_complete["timestamp_ns"])
            selected_row = {
                "factor": factor,
                "iteration": iteration,
                "operation": trigger,
                "sender_rank": sender_rank,
                "receiver_rank": receiver_rank,
                "submit_ns": submit_ns,
                "complete_ns": complete_ns,
                "communication_duration_us": (complete_ns - submit_ns) / 1_000.0,
                **config,
            }
    return dependencies, selected_row


def analyze_traces(
    baseline_trace: Path | str,
    delayed_trace: Path | str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze zero-delay and delayed trace directories."""

    baseline_grouped = _events_by_iteration(_load_events(Path(baseline_trace)))
    delayed_grouped = _events_by_iteration(_load_events(Path(delayed_trace)))
    delayed_configs = _delay_configs(delayed_grouped, allow_missing=False)
    if not delayed_configs:
        raise ValueError("delayed trace contains no delay_config events")

    first_delayed_iteration = min(delayed_configs)
    baseline_configs = _delay_configs(baseline_grouped, allow_missing=True)
    all_rows: list[dict[str, Any]] = []
    all_dependencies: list[dict[str, Any]] = []

    for iteration, events in sorted(baseline_grouped.items()):
        if iteration < first_delayed_iteration:
            continue
        config = baseline_configs.get(
            iteration,
            {
                "latency_factor": 0.0,
                "bandwidth_factor": 0.0,
                "latency_seconds": 0.0,
                "bandwidth_seconds": 0.0,
                "forward_stage_seconds": 0.0,
            },
        )
        dependencies, row = _analyze_iteration(
            iteration, events, 0.0, config
        )
        all_dependencies.extend(dependencies)
        if row is not None:
            all_rows.append(row)

    for iteration, events in sorted(delayed_grouped.items()):
        if iteration not in delayed_configs:
            continue
        config = delayed_configs[iteration]
        factor = float(config["bandwidth_factor"])
        dependencies, row = _analyze_iteration(
            iteration, events, factor, config
        )
        all_dependencies.extend(dependencies)
        if row is not None:
            all_rows.append(row)

    configurations: dict[str, dict[str, Any]] = {}
    representative_iterations: dict[str, int] = {}
    for factor in (0.0, 0.5, 1.0):
        factor_rows = [
            row for row in all_rows if float(row["factor"]) == factor
        ]
        durations = [
            float(row["communication_duration_us"]) for row in factor_rows
        ]
        key = f"{factor:.1f}"
        configurations[key] = {
            "iterations": [int(row["iteration"]) for row in factor_rows],
            "communication_duration_us": durations,
            "median_communication_duration_us": (
                statistics.median(durations) if durations else None
            ),
        }
        if durations:
            median = statistics.median(durations)
            representative = min(
                factor_rows,
                key=lambda row: abs(
                    float(row["communication_duration_us"]) - median
                ),
            )
            representative_iterations[key] = int(representative["iteration"])

    summary = {
        "selected_dependency_id": SELECTED_DEPENDENCY_ID,
        "selected_communication": SELECTED_COMMUNICATION,
        "trace_directories": {
            "baseline": str(Path(baseline_trace).resolve()),
            "delayed": str(Path(delayed_trace).resolve()),
        },
        "configurations": configurations,
        "dependency_checks": {
            "checked": len(all_dependencies),
            "violations": [],
        },
        "dependency_records": all_dependencies,
        "representative_iterations": representative_iterations,
    }
    return summary, all_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "factor",
        "iteration",
        "operation",
        "sender_rank",
        "receiver_rank",
        "submit_ns",
        "complete_ns",
        "communication_duration_us",
        "latency_factor",
        "bandwidth_factor",
        "latency_seconds",
        "bandwidth_seconds",
        "forward_stage_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--delayed-trace", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, rows = analyze_traces(args.baseline_trace, args.delayed_trace)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "dependency_checks": summary["dependency_checks"],
                "representative_iterations": summary[
                    "representative_iterations"
                ],
                "medians_us": {
                    factor: values["median_communication_duration_us"]
                    for factor, values in summary["configurations"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
