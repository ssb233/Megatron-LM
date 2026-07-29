"""Low-overhead host timeline for custom pipeline schedule validation."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Optional


class CustomScheduleTrace:
    """Write one thread-safe JSONL event stream per distributed rank."""

    def __init__(
        self,
        trace_dir: str,
        *,
        global_rank: int,
        pipeline_rank: int,
        schedule_digest: str,
    ) -> None:
        self.trace_dir = os.path.abspath(os.path.expanduser(trace_dir))
        os.makedirs(self.trace_dir, exist_ok=True)
        self.global_rank = global_rank
        self.pipeline_rank = pipeline_rank
        self.schedule_digest = schedule_digest
        self.iteration = -1
        self._lock = threading.Lock()
        self._stream = open(
            os.path.join(self.trace_dir, f"rank_{global_rank}.jsonl"),
            "w",
            encoding="utf-8",
        )
        atexit.register(self.close)
        self.record(
            "trace_metadata",
            schedule_digest=schedule_digest,
        )

    def begin_iteration(self, *, forward_only: bool) -> None:
        self.iteration += 1
        self.record("iteration_start", forward_only=forward_only)

    def end_iteration(self) -> None:
        self.record("iteration_end")
        with self._lock:
            self._stream.flush()

    def record(self, event: str, **fields) -> None:
        item = {
            "timestamp_ns": time.perf_counter_ns(),
            "event": event,
            "rank": self.global_rank,
            "pipeline_rank": self.pipeline_rank,
            "iteration": self.iteration,
        }
        item.update(fields)
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._stream.write(encoded + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                self._stream.close()

    def __del__(self):
        stream = getattr(self, "_stream", None)
        if stream is not None and not stream.closed:
            stream.close()
