"""CPU control-plane enforcement for custom communication dependencies."""

from __future__ import annotations

from collections import defaultdict
import threading
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist

from .custom_schedule import CommDependency, CommOpId, CustomScheduleSpec


_SIGNAL_TAG_BASE = 31000


class CommDependencyController:
    """Gate NCCL submissions using local Work waits and remote Gloo signals.

    Receivers never wait on this control plane.  Only the rank that will submit
    a target NCCL send may block, after the target tensor has been produced.
    """

    def __init__(
        self,
        custom_schedule_spec: CustomScheduleSpec,
        *,
        pipeline_stage: int,
        control_group,
        pipeline_global_ranks: Sequence[int],
        trace=None,
    ) -> None:
        self.spec = custom_schedule_spec
        self.pipeline_stage = pipeline_stage
        self.control_group = control_group
        self.pipeline_global_ranks = tuple(pipeline_global_ranks)
        self.trace = trace

        if len(self.pipeline_global_ranks) != self.spec.pp_size:
            raise ValueError(
                "pipeline control rank count does not match custom schedule: "
                f"{len(self.pipeline_global_ranks)} != {self.spec.pp_size}"
            )

        self._work_by_op: Dict[CommOpId, dist.Work] = {}
        self._remote_outgoing = defaultdict(list)
        self._dependency_by_pair = {}
        for dependency in self.spec.dependencies:
            self._dependency_by_pair[
                (dependency.trigger, dependency.target)
            ] = dependency
            if dependency.is_remote:
                self._remote_outgoing[dependency.trigger].append(dependency)

        self._threads: List[threading.Thread] = []
        self._thread_errors: List[BaseException] = []
        self._lock = threading.Lock()

    def _record(self, event: str, **fields) -> None:
        if self.trace is not None:
            self.trace.record(event, **fields)

    @staticmethod
    def _dependency_is_active(
        dependency: CommDependency,
        forward_only: bool,
    ) -> bool:
        return not forward_only or dependency.target.direction == "F"

    def before_send(self, operation: CommOpId, *, forward_only: bool) -> None:
        """Wait for completion prerequisites before submitting ``operation``."""

        if operation.src_stage != self.pipeline_stage:
            raise RuntimeError(
                f"stage {self.pipeline_stage} cannot submit {operation.name}"
            )

        for predecessor in self.spec.predecessors_for(operation):
            dependency = self._dependency_by_pair.get(
                (predecessor, operation)
            )
            dependency_id = (
                dependency.dependency_id if dependency is not None else None
            )
            try:
                work = self._work_by_op[predecessor]
            except KeyError as exc:
                raise RuntimeError(
                    f"local communication predecessor {predecessor.name} was "
                    f"not submitted before {operation.name}"
                ) from exc
            self._record(
                "comm_dependency_wait_start",
                dependency_kind="local",
                dependency_id=dependency_id,
                trigger=predecessor.name,
                target=operation.name,
            )
            work.wait()
            self._record(
                "comm_complete",
                operation=predecessor.name,
                dependency_ids=(
                    [dependency_id] if dependency_id is not None else []
                ),
            )
            self._record(
                "comm_dependency_wait_end",
                dependency_kind="local",
                dependency_id=dependency_id,
                trigger=predecessor.name,
                target=operation.name,
            )

        for dependency in self.spec.remote_dependencies_for(operation):
            if not self._dependency_is_active(dependency, forward_only):
                continue
            source_rank = self.pipeline_global_ranks[
                dependency.trigger.src_stage
            ]
            payload = torch.empty(1, dtype=torch.int64, device="cpu")
            self._record(
                "signal_wait_start",
                dependency_id=dependency.dependency_id,
                trigger=dependency.trigger.name,
                target=dependency.target.name,
                peer_rank=source_rank,
            )
            dist.recv(
                payload,
                src=source_rank,
                group=self.control_group,
                tag=self._signal_tag(dependency),
            )
            received_id = int(payload.item())
            if received_id != dependency.dependency_id:
                raise RuntimeError(
                    f"received dependency signal {received_id}, expected "
                    f"{dependency.dependency_id}"
                )
            self._record(
                "signal_recv",
                dependency_id=dependency.dependency_id,
                trigger=dependency.trigger.name,
                target=dependency.target.name,
                peer_rank=source_rank,
            )

    def dependency_ids_for_target(
        self,
        operation: CommOpId,
        *,
        forward_only: bool,
    ) -> List[int]:
        return [
            dependency.dependency_id
            for dependency in self.spec.dependencies
            if dependency.target == operation
            and self._dependency_is_active(dependency, forward_only)
        ]

    def register_send(
        self,
        operation: CommOpId,
        work: dist.Work,
        *,
        forward_only: bool,
    ) -> None:
        """Register an issued NCCL send and start any remote notifications."""

        if operation.src_stage != self.pipeline_stage:
            raise RuntimeError(
                f"stage {self.pipeline_stage} cannot register {operation.name}"
            )
        if operation in self._work_by_op:
            raise RuntimeError(
                f"communication operation submitted more than once: "
                f"{operation.name}"
            )
        self._work_by_op[operation] = work

        outgoing = [
            dependency
            for dependency in self._remote_outgoing.get(operation, ())
            if self._dependency_is_active(dependency, forward_only)
        ]
        if not outgoing:
            return

        thread = threading.Thread(
            target=self._wait_and_signal,
            args=(operation, work, tuple(outgoing)),
            name=f"cdc-signal-{operation.name}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def _wait_and_signal(
        self,
        operation: CommOpId,
        work: dist.Work,
        dependencies: Iterable[CommDependency],
    ) -> None:
        try:
            self._record("comm_complete_wait_start", operation=operation.name)
            work.wait()
            self._record(
                "comm_complete",
                operation=operation.name,
                dependency_ids=[
                    dependency.dependency_id
                    for dependency in dependencies
                ],
            )
            for dependency in dependencies:
                target_rank = self.pipeline_global_ranks[
                    dependency.target.src_stage
                ]
                payload = torch.tensor(
                    [dependency.dependency_id],
                    dtype=torch.int64,
                    device="cpu",
                )
                self._record(
                    "signal_send_start",
                    dependency_id=dependency.dependency_id,
                    trigger=dependency.trigger.name,
                    target=dependency.target.name,
                    peer_rank=target_rank,
                )
                dist.send(
                    payload,
                    dst=target_rank,
                    group=self.control_group,
                    tag=self._signal_tag(dependency),
                )
                self._record(
                    "signal_send_end",
                    dependency_id=dependency.dependency_id,
                    trigger=dependency.trigger.name,
                    target=dependency.target.name,
                    peer_rank=target_rank,
                )
        except BaseException as exc:
            with self._lock:
                self._thread_errors.append(exc)

    @staticmethod
    def _signal_tag(dependency: CommDependency) -> int:
        return _SIGNAL_TAG_BASE + dependency.dependency_id

    def finish_iteration(self, timeout_seconds: float = 60.0) -> None:
        """Join signal workers and reset operation state for the next iteration."""

        for thread in self._threads:
            thread.join(timeout=timeout_seconds)
            if thread.is_alive():
                raise RuntimeError(
                    f"timed out waiting for communication signal worker "
                    f"{thread.name}"
                )
        if self._thread_errors:
            errors = self._thread_errors
            self._thread_errors = []
            raise RuntimeError(
                "communication signal worker failed: "
                + "; ".join(repr(error) for error in errors)
            )
        self._threads.clear()
        self._work_by_op.clear()
