"""Load and validate Magellan custom pipeline schedules.

This module deliberately has no torch dependency.  JSON files are normalized
and checked before the distributed runtime starts issuing communication.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_COMPUTE_RE = re.compile(r"^(F|B)_(\d+)_(\d+)$")
_COMM_RE = re.compile(r"^Comm_(F|B)_(\d+)_(\d+)_(\d+)$")
_FORWARD_CHANNEL_RE = re.compile(r"^F_(\d+)_(\d+)$")
_BACKWARD_CHANNEL_RE = re.compile(r"^B_(\d+)_(\d+)$")


@dataclass(frozen=True, order=True)
class ComputeOpId:
    """A one-chunk forward or backward compute operation."""

    direction: str
    microbatch: int
    stage: int

    def __post_init__(self) -> None:
        if self.direction not in ("F", "B"):
            raise ValueError(f"unsupported compute direction {self.direction!r}")
        if self.microbatch < 0 or self.stage < 0:
            raise ValueError(f"negative compute operation index: {self}")

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.microbatch}_{self.stage}"


@dataclass(frozen=True, order=True)
class CommOpId:
    """An adjacent-stage one-chunk pipeline communication operation."""

    direction: str
    microbatch: int
    src_stage: int
    dst_stage: int

    def __post_init__(self) -> None:
        if self.direction not in ("F", "B"):
            raise ValueError(f"unsupported communication direction {self.direction!r}")
        if min(self.microbatch, self.src_stage, self.dst_stage) < 0:
            raise ValueError(f"negative communication operation index: {self}")

    @property
    def name(self) -> str:
        return (
            f"Comm_{self.direction}_{self.microbatch}_"
            f"{self.src_stage}_{self.dst_stage}"
        )

    @property
    def channel(self) -> str:
        return f"{self.direction}_{self.src_stage}_{self.dst_stage}"


@dataclass(frozen=True)
class CommDependency:
    """A normalized communication completion-to-submission dependency."""

    dependency_id: int
    trigger: CommOpId
    target: CommOpId
    reason: str
    directed_link: Optional[str]
    is_remote: bool


@dataclass(frozen=True)
class CustomScheduleSpec:
    """Normalized, immutable custom schedule consumed by CrossPipe."""

    version: int
    pp_size: int
    num_microbatches: int
    compute_order: Tuple[Tuple[ComputeOpId, ...], ...]
    comm_order: Mapping[str, Tuple[CommOpId, ...]]
    comm_positions: Mapping[CommOpId, int]
    local_predecessors: Mapping[CommOpId, Tuple[CommOpId, ...]]
    remote_predecessors: Mapping[CommOpId, Tuple[CommDependency, ...]]
    dependencies: Tuple[CommDependency, ...]
    canonical_sha256: str
    schedule_path: str
    dependency_path: Optional[str]

    def predecessors_for(self, op: CommOpId) -> Tuple[CommOpId, ...]:
        """Return local completion predecessors for ``op``."""

        return self.local_predecessors.get(op, ())

    def remote_dependencies_for(
        self, op: CommOpId
    ) -> Tuple[CommDependency, ...]:
        """Return remote Gloo dependencies that gate ``op``."""

        return self.remote_predecessors.get(op, ())

    def comm_position(self, op: CommOpId) -> int:
        try:
            return self.comm_positions[op]
        except KeyError as exc:
            raise ValueError(f"communication operation is not in schedule: {op.name}") from exc


def parse_compute_op(name: str) -> ComputeOpId:
    match = _COMPUTE_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"invalid compute operation name: {name!r}")
    direction, microbatch, stage = match.groups()
    return ComputeOpId(direction, int(microbatch), int(stage))


def parse_comm_op(name: str) -> CommOpId:
    match = _COMM_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"invalid communication operation name: {name!r}")
    direction, microbatch, src_stage, dst_stage = match.groups()
    return CommOpId(
        direction,
        int(microbatch),
        int(src_stage),
        int(dst_stage),
    )


def _read_json(path: str, label: str):
    normalized_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(normalized_path):
        raise ValueError(f"{label} file does not exist: {normalized_path}")
    try:
        with open(normalized_path, "r", encoding="utf-8") as stream:
            return json.load(stream), normalized_path
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {label} file {normalized_path}: {exc}"
        ) from exc


def _require_list(value, description: str) -> List:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array")
    return value


def _expected_comm_channels(pp_size: int) -> Tuple[str, ...]:
    channels: List[str] = []
    for stage in range(pp_size - 1):
        channels.append(f"F_{stage}_{stage + 1}")
        channels.append(f"B_{stage + 1}_{stage}")
    return tuple(channels)


def _parse_channel_key(key: str) -> Tuple[str, int, int]:
    forward = _FORWARD_CHANNEL_RE.fullmatch(key)
    backward = _BACKWARD_CHANNEL_RE.fullmatch(key)
    match = forward or backward
    if match is None:
        raise ValueError(f"invalid communication channel key: {key!r}")
    direction = "F" if forward is not None else "B"
    src_stage, dst_stage = (int(value) for value in match.groups())
    return direction, src_stage, dst_stage


def _normalize_compute_order(
    raw_compute,
    *,
    pp_size: int,
    num_microbatches: int,
) -> Tuple[Tuple[ComputeOpId, ...], ...]:
    if not isinstance(raw_compute, dict):
        raise ValueError("compute must be a JSON object")

    expected_keys = {f"stage_{stage}" for stage in range(pp_size)}
    actual_keys = set(raw_compute)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"compute stage mismatch: missing={missing}, extra={extra}")

    normalized: List[Tuple[ComputeOpId, ...]] = []
    for stage in range(pp_size):
        key = f"stage_{stage}"
        names = _require_list(raw_compute[key], f"compute.{key}")
        parsed = tuple(parse_compute_op(name) for name in names)
        if len(parsed) != 2 * num_microbatches:
            raise ValueError(
                f"{key} has {len(parsed)} operations; expected "
                f"{2 * num_microbatches}"
            )
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"{key} contains duplicate compute operations")
        wrong_stage = [op.name for op in parsed if op.stage != stage]
        if wrong_stage:
            raise ValueError(f"{key} contains operations for another stage: {wrong_stage}")

        expected = {
            ComputeOpId(direction, microbatch, stage)
            for direction in ("F", "B")
            for microbatch in range(num_microbatches)
        }
        actual = set(parsed)
        if actual != expected:
            missing = sorted(op.name for op in expected - actual)
            extra = sorted(op.name for op in actual - expected)
            raise ValueError(
                f"{key} operation mismatch: missing={missing}, extra={extra}"
            )
        normalized.append(parsed)

    return tuple(normalized)


def _normalize_comm_order(
    raw_comm,
    *,
    pp_size: int,
    num_microbatches: int,
) -> Tuple[Dict[str, Tuple[CommOpId, ...]], Dict[CommOpId, int]]:
    if not isinstance(raw_comm, dict):
        raise ValueError("comm must be a JSON object")

    expected_keys = set(_expected_comm_channels(pp_size))
    actual_keys = set(raw_comm)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"communication channel mismatch: missing={missing}, extra={extra}"
        )

    normalized: Dict[str, Tuple[CommOpId, ...]] = {}
    positions: Dict[CommOpId, int] = {}
    all_ops = set()
    for key in _expected_comm_channels(pp_size):
        direction, src_stage, dst_stage = _parse_channel_key(key)
        if abs(src_stage - dst_stage) != 1:
            raise ValueError(f"non-adjacent communication channel: {key}")
        if direction == "F" and dst_stage != src_stage + 1:
            raise ValueError(f"invalid forward communication channel: {key}")
        if direction == "B" and dst_stage != src_stage - 1:
            raise ValueError(f"invalid backward communication channel: {key}")

        names = _require_list(raw_comm[key], f"comm.{key}")
        parsed = tuple(parse_comm_op(name) for name in names)
        if len(parsed) != num_microbatches:
            raise ValueError(
                f"comm.{key} has {len(parsed)} operations; expected "
                f"{num_microbatches}"
            )
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"comm.{key} contains duplicate operations")
        for index, op in enumerate(parsed):
            if (
                op.direction != direction
                or op.src_stage != src_stage
                or op.dst_stage != dst_stage
            ):
                raise ValueError(
                    f"comm.{key} contains operation for another channel: {op.name}"
                )
            if op.microbatch >= num_microbatches:
                raise ValueError(
                    f"comm.{key} contains out-of-range microbatch: {op.name}"
                )
            positions[op] = index

        expected = {
            CommOpId(direction, microbatch, src_stage, dst_stage)
            for microbatch in range(num_microbatches)
        }
        actual = set(parsed)
        if actual != expected:
            missing = sorted(op.name for op in expected - actual)
            extra = sorted(op.name for op in actual - expected)
            raise ValueError(
                f"comm.{key} operation mismatch: missing={missing}, extra={extra}"
            )
        overlap = all_ops.intersection(actual)
        if overlap:
            raise ValueError(
                "communication operation appears in multiple channels: "
                f"{sorted(op.name for op in overlap)}"
            )
        all_ops.update(actual)
        normalized[key] = parsed

    return normalized, positions


def _normalize_dependency_edges(
    raw_dependency,
    *,
    comm_ops: Iterable[CommOpId],
    dependency_path: str,
) -> Tuple[Tuple[CommOpId, CommOpId, str, Optional[str]], ...]:
    if not isinstance(raw_dependency, dict):
        raise ValueError(f"dependency file must contain an object: {dependency_path}")
    edges = _require_list(raw_dependency.get("edges"), "dependency edges")
    declared_count = raw_dependency.get("edge_count")
    if declared_count is not None and declared_count != len(edges):
        raise ValueError(
            f"dependency edge_count={declared_count} but file contains {len(edges)} edges"
        )

    known = {op.name: op for op in comm_ops}
    normalized: Dict[
        Tuple[CommOpId, CommOpId],
        Tuple[CommOpId, CommOpId, str, Optional[str]],
    ] = {}
    notify_pairs = set()

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"dependency edge {index} must be an object")
        reason = str(edge.get("reason", ""))
        directed_link = edge.get("directed_link")
        if directed_link is not None:
            directed_link = str(directed_link)

        trigger_name = edge.get("trigger_comm")
        target_name = edge.get("target_comm")
        if trigger_name is not None or target_name is not None:
            if trigger_name is None or target_name is None:
                raise ValueError(
                    f"dependency edge {index} must provide both trigger_comm and "
                    "target_comm"
                )
            if trigger_name not in known or target_name not in known:
                raise ValueError(
                    f"dependency edge {index} references unknown communication: "
                    f"{trigger_name!r} -> {target_name!r}"
                )
            pair = (known[trigger_name], known[target_name])
            notify_pairs.add(pair)
            normalized[pair] = (
                pair[0],
                pair[1],
                reason or "insert_notify_delay_op",
                directed_link,
            )
            continue

        from_name = edge.get("from_op")
        to_name = edge.get("to_op")
        if from_name not in known or to_name not in known:
            raise ValueError(
                f"dependency edge {index} must be communication-to-communication "
                f"or provide trigger_comm/target_comm: {from_name!r} -> {to_name!r}"
            )
        pair = (known[from_name], known[to_name])
        if pair in normalized and pair not in notify_pairs:
            raise ValueError(
                f"duplicate communication dependency: {pair[0].name} -> "
                f"{pair[1].name}"
            )
        normalized[pair] = (
            pair[0],
            pair[1],
            reason or "explicit_comm_dependency",
            directed_link,
        )

    result = []
    for trigger, target, reason, directed_link in normalized.values():
        if trigger == target:
            raise ValueError(f"self communication dependency: {trigger.name}")
        result.append((trigger, target, reason, directed_link))
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item[0].name,
                item[1].name,
                item[2],
                item[3] or "",
            ),
        )
    )


def _add_edge(
    adjacency: Dict[str, set],
    indegree: Dict[str, int],
    source: str,
    target: str,
) -> None:
    if target not in adjacency[source]:
        adjacency[source].add(target)
        indegree[target] += 1


def _validate_acyclic(
    compute_order: Sequence[Sequence[ComputeOpId]],
    comm_order: Mapping[str, Sequence[CommOpId]],
    dependencies: Sequence[Tuple[CommOpId, CommOpId, str, Optional[str]]],
    *,
    pp_size: int,
    num_microbatches: int,
) -> None:
    compute_ops = [op for stage in compute_order for op in stage]
    comm_ops = [op for channel in comm_order.values() for op in channel]
    nodes = {op.name for op in compute_ops}
    nodes.update(op.name for op in comm_ops)
    adjacency: Dict[str, set] = {name: set() for name in nodes}
    indegree = {name: 0 for name in nodes}

    for stage_order in compute_order:
        for previous, current in zip(stage_order[:-1], stage_order[1:]):
            _add_edge(adjacency, indegree, previous.name, current.name)

    for channel_order in comm_order.values():
        for previous, current in zip(channel_order[:-1], channel_order[1:]):
            _add_edge(adjacency, indegree, previous.name, current.name)

    for microbatch in range(num_microbatches):
        for stage in range(pp_size - 1):
            forward_compute = ComputeOpId("F", microbatch, stage)
            forward_comm = CommOpId("F", microbatch, stage, stage + 1)
            next_forward = ComputeOpId("F", microbatch, stage + 1)
            _add_edge(adjacency, indegree, forward_compute.name, forward_comm.name)
            _add_edge(adjacency, indegree, forward_comm.name, next_forward.name)

            backward_compute = ComputeOpId("B", microbatch, stage + 1)
            backward_comm = CommOpId("B", microbatch, stage + 1, stage)
            next_backward = ComputeOpId("B", microbatch, stage)
            _add_edge(adjacency, indegree, backward_compute.name, backward_comm.name)
            _add_edge(adjacency, indegree, backward_comm.name, next_backward.name)

        _add_edge(
            adjacency,
            indegree,
            ComputeOpId("F", microbatch, pp_size - 1).name,
            ComputeOpId("B", microbatch, pp_size - 1).name,
        )

    for trigger, target, _reason, _directed_link in dependencies:
        _add_edge(adjacency, indegree, trigger.name, target.name)

    ready = deque(sorted(name for name, degree in indegree.items() if degree == 0))
    visited = []
    while ready:
        current = ready.popleft()
        visited.append(current)
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if len(visited) != len(nodes):
        cyclic = sorted(name for name, degree in indegree.items() if degree > 0)
        raise ValueError(
            "custom schedule dependency graph contains a cycle involving: "
            f"{cyclic[:16]}"
        )


def load_custom_schedule(
    schedule_path: str,
    dependency_path: Optional[str],
    *,
    pp_size: int,
    num_microbatches: int,
) -> CustomScheduleSpec:
    """Load, normalize, and validate Magellan schedule files."""

    if pp_size <= 1:
        raise ValueError(f"custom pipeline schedule requires PP > 1, got {pp_size}")
    if num_microbatches <= 0:
        raise ValueError(
            f"custom pipeline schedule requires microbatches > 0, got "
            f"{num_microbatches}"
        )

    raw_schedule, normalized_schedule_path = _read_json(
        schedule_path, "custom pipeline schedule"
    )
    if not isinstance(raw_schedule, dict):
        raise ValueError("custom pipeline schedule must contain a JSON object")
    version = raw_schedule.get("version")
    if version != 1:
        raise ValueError(f"unsupported replay.order.json version: {version!r}")

    compute_order = _normalize_compute_order(
        raw_schedule.get("compute"),
        pp_size=pp_size,
        num_microbatches=num_microbatches,
    )
    comm_order, comm_positions = _normalize_comm_order(
        raw_schedule.get("comm"),
        pp_size=pp_size,
        num_microbatches=num_microbatches,
    )
    all_comm_ops = tuple(op for order in comm_order.values() for op in order)

    normalized_dependency_path = None
    raw_dependencies: Tuple[
        Tuple[CommOpId, CommOpId, str, Optional[str]], ...
    ] = ()
    if dependency_path is not None:
        raw_dependency, normalized_dependency_path = _read_json(
            dependency_path, "custom communication dependency"
        )
        raw_dependencies = _normalize_dependency_edges(
            raw_dependency,
            comm_ops=all_comm_ops,
            dependency_path=normalized_dependency_path,
        )

    channel_edges = {
        (previous, current)
        for order in comm_order.values()
        for previous, current in zip(order[:-1], order[1:])
    }
    extra_dependencies = tuple(
        dependency
        for dependency in raw_dependencies
        if (dependency[0], dependency[1]) not in channel_edges
    )

    _validate_acyclic(
        compute_order,
        comm_order,
        extra_dependencies,
        pp_size=pp_size,
        num_microbatches=num_microbatches,
    )

    normalized_dependencies: List[CommDependency] = []
    local_predecessors = defaultdict(list)
    remote_predecessors = defaultdict(list)

    for previous, current in sorted(
        channel_edges, key=lambda pair: (pair[1].name, pair[0].name)
    ):
        local_predecessors[current].append(previous)

    for dependency_id, (
        trigger,
        target,
        reason,
        directed_link,
    ) in enumerate(extra_dependencies):
        is_remote = trigger.src_stage != target.src_stage
        dependency = CommDependency(
            dependency_id=dependency_id,
            trigger=trigger,
            target=target,
            reason=reason,
            directed_link=directed_link,
            is_remote=is_remote,
        )
        normalized_dependencies.append(dependency)
        if is_remote:
            remote_predecessors[target].append(dependency)
        else:
            local_predecessors[target].append(trigger)

    canonical_object = {
        "version": version,
        "pp_size": pp_size,
        "num_microbatches": num_microbatches,
        "compute": {
            f"stage_{stage}": [op.name for op in order]
            for stage, order in enumerate(compute_order)
        },
        "comm": {
            channel: [op.name for op in order]
            for channel, order in sorted(comm_order.items())
        },
        "dependencies": [
            {
                "dependency_id": dependency.dependency_id,
                "trigger": dependency.trigger.name,
                "target": dependency.target.name,
                "reason": dependency.reason,
                "directed_link": dependency.directed_link,
                "is_remote": dependency.is_remote,
            }
            for dependency in normalized_dependencies
        ],
    }
    canonical = json.dumps(
        canonical_object,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return CustomScheduleSpec(
        version=version,
        pp_size=pp_size,
        num_microbatches=num_microbatches,
        compute_order=compute_order,
        comm_order=comm_order,
        comm_positions=comm_positions,
        local_predecessors={
            target: tuple(sorted(predecessors))
            for target, predecessors in local_predecessors.items()
        },
        remote_predecessors={
            target: tuple(
                sorted(
                    dependencies,
                    key=lambda dependency: dependency.dependency_id,
                )
            )
            for target, dependencies in remote_predecessors.items()
        },
        dependencies=tuple(normalized_dependencies),
        canonical_sha256=digest,
        schedule_path=normalized_schedule_path,
        dependency_path=normalized_dependency_path,
    )
