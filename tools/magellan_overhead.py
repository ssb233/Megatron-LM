#!/usr/bin/env python3
"""Utilities for the PP=4 Magellan communication-dependency experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import deque
from pathlib import Path
from typing import Iterable


ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+)"
)


def _finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return number


def derive_calibration(profile: dict) -> dict:
    """Derive the normalized communication duration used by Magellan."""

    try:
        forward = profile["T_F"][0]
        alpha = profile["T_alpha"]
        bandwidth = profile["T_bw"]
        middle_forward = [
            _finite_positive(forward[index], f"T_F[0][{index}]")
            for index in (1, 2)
        ]
        adjacent_comm = [
            _finite_positive(
                float(alpha[index][index + 1])
                + float(bandwidth[index][index + 1]),
                f"T_comm[{index}][{index + 1}]",
            )
            for index in range(3)
        ]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid PP=4 CrossPipe profile structure") from exc

    t_f_ref = statistics.median(middle_forward)
    t_comm_ref = statistics.median(adjacent_comm)
    return {
        "t_f_ref_seconds": t_f_ref,
        "t_comm_ref_seconds": t_comm_ref,
        "comm_units": t_comm_ref / t_f_ref,
        "middle_stage_forward_seconds": middle_forward,
        "adjacent_p2p_seconds": adjacent_comm,
    }


def parse_iteration_times(
    text: str,
    first_iteration: int = 6,
    last_iteration: int = 20,
) -> list[dict]:
    """Parse a closed interval of Megatron iteration-time log records."""

    rows = []
    for match in ITERATION_RE.finditer(text):
        iteration = int(match.group("iteration"))
        if first_iteration <= iteration <= last_iteration:
            rows.append(
                {
                    "iteration": iteration,
                    "milliseconds": float(match.group("ms")),
                }
            )
    return rows


def summarize(values: list[float]) -> dict:
    """Return descriptive timing statistics in milliseconds."""

    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    numbers = [
        _finite_positive(value, f"timing[{index}]")
        for index, value in enumerate(values)
    ]
    return {
        "count": len(numbers),
        "mean_ms": statistics.mean(numbers),
        "median_ms": statistics.median(numbers),
        "stdev_ms": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min_ms": min(numbers),
        "max_ms": max(numbers),
    }


def _ordered_lists(order: dict) -> tuple[list[list[str]], list[list[str]]]:
    compute = order.get("compute")
    comm = order.get("comm")
    if not isinstance(compute, dict) or not isinstance(comm, dict):
        raise ValueError("order must contain compute and comm objects")
    compute_lists = []
    comm_lists = []
    for label, mapping, target in (
        ("compute", compute, compute_lists),
        ("comm", comm, comm_lists),
    ):
        for key, operations in mapping.items():
            if not isinstance(operations, list) or not all(
                isinstance(operation, str) for operation in operations
            ):
                raise ValueError(f"{label}.{key} must be an operation array")
            target.append(operations)
    return compute_lists, comm_lists


def _dependency_pairs(
    dependencies: object,
    known_comm: set[str],
) -> set[tuple[str, str]]:
    if isinstance(dependencies, dict):
        edges = dependencies.get("edges")
    else:
        edges = dependencies
    if not isinstance(edges, list):
        raise ValueError("dependency JSON must provide an edges array")

    pairs = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"dependency edge {index} must be an object")
        trigger = edge.get("trigger_comm")
        target = edge.get("target_comm")
        if trigger is None and target is None:
            trigger = edge.get("from_op")
            target = edge.get("to_op")
        elif trigger is None or target is None:
            raise ValueError(
                f"dependency edge {index} must provide trigger_comm and target_comm"
            )
        if trigger not in known_comm or target not in known_comm:
            # Notification halves are represented twice. Their trigger/target
            # fields above collapse both halves to a known communication pair.
            raise ValueError(
                f"dependency edge {index} references unknown communication: "
                f"{trigger!r} -> {target!r}"
            )
        if trigger == target:
            raise ValueError(f"self communication dependency: {trigger}")
        pairs.add((trigger, target))
    return pairs


def _add_edge(
    adjacency: dict[str, set[str]],
    indegree: dict[str, int],
    source: str,
    target: str,
) -> None:
    if target not in adjacency[source]:
        adjacency[source].add(target)
        indegree[target] += 1


def validate_schedule(
    order: dict,
    dependencies: object,
    microbatches: int,
    stages: int,
) -> dict:
    """Validate operation coverage, extra dependencies, and acyclicity."""

    if microbatches <= 0 or stages <= 1:
        raise ValueError("schedule validation requires N > 0 and S > 1")
    if order.get("version") != 1:
        raise ValueError("order version must be 1")

    compute_lists, comm_lists = _ordered_lists(order)
    compute_ops = [operation for values in compute_lists for operation in values]
    comm_ops = [operation for values in comm_lists for operation in values]
    if len(set(compute_ops)) != len(compute_ops):
        raise ValueError("duplicate compute operation")
    if len(set(comm_ops)) != len(comm_ops):
        raise ValueError("duplicate communication operation")

    expected_compute = {
        f"{direction}_{microbatch}_{stage}"
        for direction in ("F", "B")
        for microbatch in range(microbatches)
        for stage in range(stages)
    }
    expected_comm = {
        f"Comm_F_{microbatch}_{stage}_{stage + 1}"
        for microbatch in range(microbatches)
        for stage in range(stages - 1)
    } | {
        f"Comm_B_{microbatch}_{stage + 1}_{stage}"
        for microbatch in range(microbatches)
        for stage in range(stages - 1)
    }
    if set(compute_ops) != expected_compute:
        raise ValueError("compute operation coverage mismatch")
    if set(comm_ops) != expected_comm:
        raise ValueError("communication operation coverage mismatch")

    channel_edges = {
        pair
        for values in comm_lists
        for pair in zip(values[:-1], values[1:])
    }
    raw_pairs = _dependency_pairs(dependencies, expected_comm)
    extra_pairs = raw_pairs - channel_edges
    if not extra_pairs:
        raise ValueError("communication dependency file has no extra dependencies")

    nodes = expected_compute | expected_comm
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for values in compute_lists + comm_lists:
        for source, target in zip(values[:-1], values[1:]):
            _add_edge(adjacency, indegree, source, target)

    for microbatch in range(microbatches):
        for stage in range(stages - 1):
            _add_edge(
                adjacency,
                indegree,
                f"F_{microbatch}_{stage}",
                f"Comm_F_{microbatch}_{stage}_{stage + 1}",
            )
            _add_edge(
                adjacency,
                indegree,
                f"Comm_F_{microbatch}_{stage}_{stage + 1}",
                f"F_{microbatch}_{stage + 1}",
            )
            _add_edge(
                adjacency,
                indegree,
                f"B_{microbatch}_{stage + 1}",
                f"Comm_B_{microbatch}_{stage + 1}_{stage}",
            )
            _add_edge(
                adjacency,
                indegree,
                f"Comm_B_{microbatch}_{stage + 1}_{stage}",
                f"B_{microbatch}_{stage}",
            )
        _add_edge(
            adjacency,
            indegree,
            f"F_{microbatch}_{stages - 1}",
            f"B_{microbatch}_{stages - 1}",
        )
    for source, target in extra_pairs:
        _add_edge(adjacency, indegree, source, target)

    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError("custom schedule and communication dependencies contain a cycle")

    return {
        "operation_count": len(nodes),
        "raw_dependency_count": len(raw_pairs),
        "dependency_count": len(extra_pairs),
        "acyclic": True,
    }


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _calibrate_command(args: argparse.Namespace) -> None:
    result = derive_calibration(_read_json(args.profile_total))
    _write_json(args.output, result)


def _validate_command(args: argparse.Namespace) -> None:
    result = validate_schedule(
        _read_json(args.order),
        _read_json(args.dependencies),
        microbatches=args.microbatches,
        stages=args.stages,
    )
    if args.output:
        _write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


def _summarize_command(args: argparse.Namespace) -> None:
    rows = []
    for log_path in args.logs:
        parsed = parse_iteration_times(Path(log_path).read_text(encoding="utf-8"))
        if len(parsed) != args.expected_samples:
            raise ValueError(
                f"{log_path} has {len(parsed)} measured iterations; "
                f"expected {args.expected_samples}"
            )
        rows.extend(parsed)
    result = summarize([row["milliseconds"] for row in rows])
    _write_json(args.output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--profile-total", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(handler=_calibrate_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--order", required=True)
    validate.add_argument("--dependencies", required=True)
    validate.add_argument("--microbatches", type=int, default=8)
    validate.add_argument("--stages", type=int, default=4)
    validate.add_argument("--output")
    validate.set_defaults(handler=_validate_command)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--logs", nargs="+", required=True)
    summary.add_argument("--expected-samples", type=int, default=15)
    summary.add_argument("--output", required=True)
    summary.set_defaults(handler=_summarize_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
