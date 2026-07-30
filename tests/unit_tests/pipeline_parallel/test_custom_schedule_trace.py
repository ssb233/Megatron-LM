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


def test_trace_records_active_delay_configuration(tmp_path):
    trace = CustomScheduleTrace(
        str(tmp_path),
        global_rank=1,
        pipeline_rank=1,
        schedule_digest="abc",
    )
    trace.begin_iteration(forward_only=False)
    trace.record_delay_configuration(
        latency_factor=0.0,
        bandwidth_factor=0.5,
        latency_seconds=0.0,
        bandwidth_seconds=0.005,
        forward_stage_seconds=0.01,
    )
    trace.end_iteration()
    trace.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank_1.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    delay_record = next(
        record for record in records if record["event"] == "delay_config"
    )
    common_fields = {
        "timestamp_ns",
        "event",
        "rank",
        "pipeline_rank",
        "iteration",
    }
    assert {
        key: value
        for key, value in delay_record.items()
        if key not in common_fields
    } == {
        "latency_factor": 0.0,
        "bandwidth_factor": 0.5,
        "latency_seconds": 0.0,
        "bandwidth_seconds": 0.005,
        "forward_stage_seconds": 0.01,
    }


def test_trace_can_flush_each_event_for_deadlock_diagnostics(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUSTOM_SCHEDULE_TRACE_FLUSH_EACH_EVENT", "1")
    trace = CustomScheduleTrace(
        str(tmp_path),
        global_rank=3,
        pipeline_rank=3,
        schedule_digest="abc",
    )

    text_before_close = (tmp_path / "rank_3.jsonl").read_text(
        encoding="utf-8"
    )
    trace.close()

    assert '"event":"trace_metadata"' in text_before_close
