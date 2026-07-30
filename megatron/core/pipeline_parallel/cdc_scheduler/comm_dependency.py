"""CPU control-plane enforcement for custom communication dependencies."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist

from .custom_schedule import CommDependency, CommOpId, CustomScheduleSpec


_SIGNAL_TAG_BASE = 31000


class _DeferredRecvWork:
    """Materialize an NCCL receive only when its result is first awaited."""

    def __init__(self, launch: Callable[[], dist.Work]) -> None:
        self._launch = launch
        self._work: Optional[dist.Work] = None
        self._lock = threading.Lock()

    def materialize(self) -> dist.Work:
        if self._work is None:
            with self._lock:
                if self._work is None:
                    self._work = self._launch()
        return self._work

    def wait(self, *args, **kwargs):
        return self.materialize().wait(*args, **kwargs)

    def wait_with_lat_delay_in_ms(self, *args, **kwargs):
        work = self.materialize()
        return work.wait_with_lat_delay_in_ms(*args, **kwargs)

    def is_completed(self) -> bool:
        if self._work is None:
            return False
        return self._work.is_completed()


class CommDependencyController:
    """Gate NCCL submissions using local Work waits and remote Gloo signals.

    Target senders block on the CPU after their tensor has been produced.  When
    a remote dependency's trigger and target share a receiving stage, that
    stage also defers the target recv until the trigger recv has completed. This
    prevents an unmatched early NCCL recv from blocking the trigger link.
    """

    def __init__(
        self,
        custom_schedule_spec: CustomScheduleSpec,
        *,
        pipeline_stage: int,
        control_group,
        pipeline_global_ranks: Sequence[int],
        timeout_seconds: float = 600.0,
        trace=None,
    ) -> None:
        self.spec = custom_schedule_spec
        self.pipeline_stage = pipeline_stage
        self.control_group = control_group
        self.pipeline_global_ranks = tuple(pipeline_global_ranks)
        self.timeout_seconds = timeout_seconds
        self.trace = trace

        if len(self.pipeline_global_ranks) != self.spec.pp_size:
            raise ValueError(
                "pipeline control rank count does not match custom schedule: "
                f"{len(self.pipeline_global_ranks)} != {self.spec.pp_size}"
            )

        self._work_by_op: Dict[CommOpId, dist.Work] = {}
        self._recv_work_by_op: Dict[CommOpId, dist.Work] = {}
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
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        """Enable dependency enforcement for the current iteration."""

        self.enabled = bool(enabled)

    @staticmethod
    def _wait_for_work_completion(
        work: dist.Work,
        operation: CommOpId,
        timeout_seconds: float = 600.0,
    ) -> None:
        """Wait for actual NCCL completion, not only CUDA stream ordering."""

        if hasattr(work, "is_completed"):
            deadline = time.monotonic() + timeout_seconds
            while not work.is_completed():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"timed out waiting for NCCL completion of "
                        f"{operation.name}"
                    )
                time.sleep(0.0001)
            return
        # Preserve support for test/fallback Work implementations that do not
        # expose is_completed(). Avoid Work.wait() in the Gloo worker thread
        # for NCCL Work because CUDA's current device is thread-local.
        work.wait()

    def _record(self, event: str, **fields) -> None:
        if self.trace is not None:
            self.trace.record(event, **fields)

    @staticmethod
    def _dependency_is_active(
        dependency: CommDependency,
        forward_only: bool,
    ) -> bool:
        return not forward_only or (
            dependency.trigger.direction == "F"
            and dependency.target.direction == "F"
        )

    def _wait_for_signal(
        self,
        work: dist.Work,
        dependency: CommDependency,
    ) -> None:
        try:
            completed = work.wait(
                timeout=timedelta(seconds=self.timeout_seconds)
            )
        except TypeError:
            # Test and compatibility Work implementations may expose wait()
            # without the optional timeout argument.
            completed = work.wait()
        except RuntimeError as exc:
            raise RuntimeError(
                f"rank/stage {self.pipeline_stage} timed out waiting for "
                f"dependency {dependency.dependency_id} "
                f"{dependency.trigger.name} -> {dependency.target.name}"
            ) from exc

        if completed is False:
            raise RuntimeError(
                f"rank/stage {self.pipeline_stage} timed out waiting for "
                f"dependency {dependency.dependency_id} "
                f"{dependency.trigger.name} -> {dependency.target.name}"
            )

    def before_send(self, operation: CommOpId, *, forward_only: bool) -> None:
        """Wait for completion prerequisites before submitting ``operation``."""

        if not self.enabled:
            return
        if operation.src_stage != self.pipeline_stage:
            raise RuntimeError(
                f"stage {self.pipeline_stage} cannot submit {operation.name}"
            )

        for predecessor in self.spec.predecessors_for(operation):
            if forward_only and predecessor.direction != "F":
                continue
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
            self._wait_for_work_completion(
                work,
                predecessor,
                self.timeout_seconds,
            )
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
            signal_work = dist.irecv(
                payload,
                src=source_rank,
                group=self.control_group,
                tag=self._signal_tag(dependency),
            )
            self._wait_for_signal(signal_work, dependency)
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
        if not self.enabled:
            return []
        return [
            dependency.dependency_id
            for dependency in self.spec.dependencies
            if dependency.target == operation
            and self._dependency_is_active(dependency, forward_only)
        ]

    def post_recv(
        self,
        operation: CommOpId,
        launch_recv: Callable[[], dist.Work],
        *,
        forward_only: bool,
    ) -> dist.Work:
        """Post or safely defer a target recv on a shared receiving stage."""

        shared_receiver_dependencies = [
            dependency
            for dependency in self.spec.dependencies
            if self.enabled
            and dependency.is_remote
            and dependency.target == operation
            and dependency.trigger.dst_stage == self.pipeline_stage
            and dependency.target.dst_stage == self.pipeline_stage
            and self._dependency_is_active(dependency, forward_only)
        ]

        if shared_receiver_dependencies:
            work = _DeferredRecvWork(
                lambda: self._launch_recv_after_shared_predecessors(
                    operation,
                    launch_recv,
                    shared_receiver_dependencies,
                )
            )
        else:
            work = launch_recv()
        self._recv_work_by_op[operation] = work
        return work

    def _launch_recv_after_shared_predecessors(
        self,
        operation: CommOpId,
        launch_recv: Callable[[], dist.Work],
        dependencies: Sequence[CommDependency],
    ) -> dist.Work:
        for dependency in dependencies:
            try:
                predecessor_work = self._recv_work_by_op[
                    dependency.trigger
                ]
            except KeyError as exc:
                raise RuntimeError(
                    f"shared receiver stage {self.pipeline_stage} reached "
                    f"{operation.name} before trigger recv "
                    f"{dependency.trigger.name} was posted"
                ) from exc
            if isinstance(predecessor_work, _DeferredRecvWork):
                predecessor_work = predecessor_work.materialize()
            self._record(
                "recv_dependency_wait_start",
                dependency_id=dependency.dependency_id,
                trigger=dependency.trigger.name,
                target=operation.name,
            )
            self._wait_for_work_completion(
                predecessor_work,
                dependency.trigger,
                self.timeout_seconds,
            )
            self._record(
                "recv_dependency_wait_end",
                dependency_id=dependency.dependency_id,
                trigger=dependency.trigger.name,
                target=operation.name,
            )
        self._record(
            "recv_deferred_submit",
            operation=operation.name,
            dependency_ids=[
                dependency.dependency_id for dependency in dependencies
            ],
        )
        return launch_recv()

    def register_send(
        self,
        operation: CommOpId,
        work: dist.Work,
        *,
        forward_only: bool,
    ) -> None:
        """Register an issued NCCL send and start any remote notifications."""

        if not self.enabled:
            return
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
            self._wait_for_work_completion(
                work,
                operation,
                self.timeout_seconds,
            )
            self._record(
                "comm_complete",
                operation=operation.name,
                dependency_ids=[
                    dependency.dependency_id
                    for dependency in dependencies
                ],
            )
            pending_signals = []
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
                signal_work = dist.isend(
                    payload,
                    dst=target_rank,
                    group=self.control_group,
                    tag=self._signal_tag(dependency),
                )
                # Retain the CPU tensor until Gloo has consumed it. Posting all
                # target signals before waiting avoids head-of-line deadlocks
                # when one trigger releases multiple target ranks.
                pending_signals.append(
                    (dependency, target_rank, payload, signal_work)
                )

            for (
                dependency,
                target_rank,
                _payload,
                signal_work,
            ) in pending_signals:
                signal_work.wait()
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

    def finish_iteration(self, timeout_seconds: Optional[float] = None) -> None:
        """Join signal workers and reset operation state for the next iteration."""

        timeout_seconds = (
            self.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
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
        self._recv_work_by_op.clear()
