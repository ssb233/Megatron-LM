import json

from megatron.core.pipeline_parallel.cdc_scheduler.custom_schedule_trace import (
    CustomScheduleTrace,
)


def test_trace_writes_rank_local_jsonl(tmp_path):
    trace = CustomScheduleTrace(
        str(tmp_path),
        global_rank=2,
        pipeline_rank=2,
        schedule_digest="abc",
    )
    trace.begin_iteration(forward_only=False)
    trace.record(
        "compute_start",
        operation="F_0_2",
        direction="F",
        microbatch=0,
    )
    trace.record(
        "compute_end",
        operation="F_0_2",
        direction="F",
        microbatch=0,
    )
    trace.end_iteration()
    trace.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank_2.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["event"] for record in records] == [
        "trace_metadata",
        "iteration_start",
        "compute_start",
        "compute_end",
        "iteration_end",
    ]
    assert records[2]["operation"] == "F_0_2"
    assert all(record["rank"] == 2 for record in records)
