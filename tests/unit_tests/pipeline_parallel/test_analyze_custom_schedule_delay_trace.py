import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.analyze_custom_schedule_delay_trace import analyze_traces
from tools.plot_custom_schedule_delay_trace import create_figure


def _write_rank(trace_dir, rank, records):
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"rank_{rank}.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _iteration_records(iteration, factor, start_ns, *, include_config=True):
    rank_1 = []
    rank_3 = []
    if include_config:
        config = {
            "event": "delay_config",
            "iteration": iteration,
            "bandwidth_factor": factor,
            "bandwidth_seconds": factor * 0.01,
            "latency_factor": 0.0,
            "latency_seconds": 0.0,
            "forward_stage_seconds": 0.01,
            "timestamp_ns": start_ns,
        }
        rank_1.append({**config, "rank": 1, "pipeline_rank": 1})
        rank_3.append({**config, "rank": 3, "pipeline_rank": 3})
    rank_1.extend(
        [
            {
                "event": "target_submit",
                "iteration": iteration,
                "rank": 1,
                "pipeline_rank": 1,
                "operation": "Comm_F_4_1_2",
                "dependency_ids": [],
                "timestamp_ns": start_ns + 1_000,
            },
            {
                "event": "comm_complete",
                "iteration": iteration,
                "rank": 1,
                "pipeline_rank": 1,
                "operation": "Comm_F_4_1_2",
                "dependency_ids": [6],
                "timestamp_ns": start_ns + 2_000 + int(factor * 10_000),
            },
            {
                "event": "signal_send_start",
                "iteration": iteration,
                "rank": 1,
                "pipeline_rank": 1,
                "dependency_id": 6,
                "peer_rank": 3,
                "trigger": "Comm_F_4_1_2",
                "target": "Comm_B_0_3_2",
                "timestamp_ns": start_ns + 13_000,
            },
        ]
    )
    rank_3.extend(
        [
            {
                "event": "signal_recv",
                "iteration": iteration,
                "rank": 3,
                "pipeline_rank": 3,
                "dependency_id": 6,
                "peer_rank": 1,
                "trigger": "Comm_F_4_1_2",
                "target": "Comm_B_0_3_2",
                "timestamp_ns": start_ns + 14_000,
            },
            {
                "event": "target_submit",
                "iteration": iteration,
                "rank": 3,
                "pipeline_rank": 3,
                "operation": "Comm_B_0_3_2",
                "dependency_ids": [6],
                "timestamp_ns": start_ns + 15_000,
            },
        ]
    )
    return rank_1, rank_3


def _build_trace(trace_dir, configs, *, include_config=True):
    rank_records = {1: [], 3: []}
    for offset, (iteration, factor) in enumerate(configs):
        rank_1, rank_3 = _iteration_records(
            iteration,
            factor,
            1_000_000 * (offset + 1),
            include_config=include_config,
        )
        rank_records[1].extend(rank_1)
        rank_records[3].extend(rank_3)
    for rank, records in rank_records.items():
        _write_rank(trace_dir, rank, records)


def test_analyzer_separates_factors_and_accepts_legacy_baseline(tmp_path):
    baseline = tmp_path / "baseline"
    delayed = tmp_path / "delayed"
    _build_trace(baseline, [(3, 0.0)], include_config=False)
    _build_trace(delayed, [(3, 0.5), (4, 1.0)])

    summary, rows = analyze_traces(baseline, delayed)

    assert sorted(summary["configurations"]) == ["0.0", "0.5", "1.0"]
    assert summary["configurations"]["0.0"]["iterations"] == [3]
    assert summary["configurations"]["0.5"]["iterations"] == [3]
    assert summary["configurations"]["1.0"]["iterations"] == [4]
    assert summary["dependency_checks"]["checked"] == 3
    assert summary["dependency_checks"]["violations"] == []
    assert len(rows) == 3
    record = summary["dependency_records"][0]
    assert record["sender_rank"] == 1
    assert record["receiver_rank"] == 3


def test_analyzer_rejects_dependency_order_violation(tmp_path):
    baseline = tmp_path / "baseline"
    delayed = tmp_path / "delayed"
    _build_trace(baseline, [(3, 0.0)], include_config=False)
    _build_trace(delayed, [(3, 0.5)])
    rank_3_path = delayed / "rank_3.jsonl"
    records = [
        json.loads(line)
        for line in rank_3_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["event"] == "target_submit":
            record["timestamp_ns"] -= 2_000
    rank_3_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dependency ordering"):
        analyze_traces(baseline, delayed)


def test_analyzer_requires_delay_config_for_delayed_trace(tmp_path):
    baseline = tmp_path / "baseline"
    delayed = tmp_path / "delayed"
    _build_trace(baseline, [(3, 0.0)], include_config=False)
    _build_trace(delayed, [(3, 0.5)], include_config=False)

    with pytest.raises(ValueError, match="delay_config"):
        analyze_traces(baseline, delayed)


def test_communication_duration_uses_dependency_trigger_completion(tmp_path):
    baseline = tmp_path / "baseline"
    delayed = tmp_path / "delayed"
    _build_trace(baseline, [(3, 0.0)], include_config=False)
    _build_trace(delayed, [(3, 0.5)])

    summary, _ = analyze_traces(baseline, delayed)

    assert summary["configurations"]["0.0"]["communication_duration_us"] == [
        1.0
    ]
    assert summary["configurations"]["0.5"]["communication_duration_us"] == [
        6.0
    ]


def test_plotter_emits_publication_formats_and_caption(tmp_path):
    baseline = tmp_path / "baseline"
    delayed = tmp_path / "delayed"
    output = tmp_path / "figure"
    _build_trace(baseline, [(3, 0.0)], include_config=False)
    _build_trace(delayed, [(3, 0.5), (4, 1.0)])
    summary, rows = analyze_traces(baseline, delayed)

    created = create_figure(summary, rows, delayed, output)

    assert {path.suffix for path in created} >= {".svg", ".pdf", ".png"}
    caption = (output / "figure_caption.txt").read_text(encoding="utf-8")
    assert "visualization only" in caption.lower()
    svg = (output / "custom_schedule_delay_trace.svg").read_text(
        encoding="utf-8"
    )
    assert "Artificial transfer delay for visualization only" in svg
    assert "1.0 x F" in svg
    assert "脳" not in svg
    assert "碌" not in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    source_csv = (output / "figure_source_data.csv").read_bytes()
    assert b"\r\n" not in source_csv


def test_plotter_cli_runs_directly_from_outside_repository(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    plotter = repository / "tools" / "plot_custom_schedule_delay_trace.py"

    completed = subprocess.run(
        [sys.executable, str(plotter), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--summary" in completed.stdout
