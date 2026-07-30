import json
from pathlib import Path
import subprocess

import pytest

from tools.magellan_overhead import (
    derive_calibration,
    parse_iteration_times,
    summarize,
    validate_schedule,
)


def test_calibration_uses_middle_stage_forward_and_adjacent_p2p_medians():
    profile = {
        "T_F": [[0.003, 0.005, 0.007, 0.004]],
        "T_alpha": [
            [0, 0.00010, 0, 0],
            [0.00010, 0, 0.00020, 0],
            [0, 0.00020, 0, 0.00030],
            [0, 0, 0.00030, 0],
        ],
        "T_bw": [
            [0, 0.00001, 0, 0],
            [0.00001, 0, 0.00002, 0],
            [0, 0.00002, 0, 0.00003],
            [0, 0, 0.00003, 0],
        ],
    }

    result = derive_calibration(profile)

    assert result["t_f_ref_seconds"] == pytest.approx(0.006)
    assert result["t_comm_ref_seconds"] == pytest.approx(0.00022)
    assert result["comm_units"] == pytest.approx(0.00022 / 0.006)


def test_iteration_parser_keeps_exactly_iterations_6_through_19():
    text = "\n".join(
        f"iteration {iteration}/ 20 | "
        f"elapsed time per iteration (ms): {100 + iteration}.0 |"
        for iteration in range(1, 21)
    )

    rows = parse_iteration_times(text)

    assert [row["iteration"] for row in rows] == list(range(6, 20))
    assert rows[0]["milliseconds"] == 106.0
    assert len(rows) == 14


def _one_microbatch_two_stage_order():
    return {
        "version": 1,
        "compute": {
            "stage_0": ["F_0_0", "B_0_0"],
            "stage_1": ["F_0_1", "B_0_1"],
        },
        "comm": {
            "F_0_1": ["Comm_F_0_0_1"],
            "B_1_0": ["Comm_B_0_1_0"],
        },
    }


def test_schedule_validator_accepts_notification_trigger_target_edges():
    dependencies = {
        "edges": [
            {
                "from_op": "Comm_F_0_0_1",
                "to_op": "Notify_x",
                "trigger_comm": "Comm_F_0_0_1",
                "target_comm": "Comm_B_0_1_0",
            },
            {
                "from_op": "Notify_x",
                "to_op": "Comm_B_0_1_0",
                "trigger_comm": "Comm_F_0_0_1",
                "target_comm": "Comm_B_0_1_0",
            },
        ]
    }

    result = validate_schedule(
        _one_microbatch_two_stage_order(),
        dependencies,
        microbatches=1,
        stages=2,
    )

    assert result["dependency_count"] == 1
    assert result["operation_count"] == 6
    assert result["acyclic"] is True


def test_schedule_validator_rejects_dependency_cycle():
    dependencies = {
        "edges": [
            {
                "from_op": "Comm_F_0_0_1",
                "to_op": "Comm_B_0_1_0",
            },
            {
                "from_op": "Comm_B_0_1_0",
                "to_op": "Comm_F_0_0_1",
            },
        ]
    }

    with pytest.raises(ValueError, match="cycle"):
        validate_schedule(
            _one_microbatch_two_stage_order(),
            dependencies,
            microbatches=1,
            stages=2,
        )


def test_summary_reports_sample_standard_deviation():
    result = summarize([100.0, 110.0, 120.0])

    assert result == {
        "count": 3,
        "mean_ms": 110.0,
        "median_ms": 110.0,
        "stdev_ms": 10.0,
        "min_ms": 100.0,
        "max_ms": 120.0,
    }


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "test_crossdc/magellan_overhead/run_training.sh"
CONFIGS = REPO_ROOT / "test_crossdc/magellan_overhead/configs.json"


def test_experiment_matrix_includes_moe_n8_positive_case():
    configs = json.loads(CONFIGS.read_text(encoding="utf-8"))

    assert configs == {
        "D1": {
            "hidden": 1024,
            "ffn": 4096,
            "heads": 16,
            "seq": 256,
            "mbs": 1,
            "gbs": 8,
            "experts": None,
        },
        "D2": {
            "hidden": 1536,
            "ffn": 6144,
            "heads": 24,
            "seq": 512,
            "mbs": 2,
            "gbs": 32,
            "experts": None,
        },
        "M1": {
            "hidden": 1024,
            "ffn": 4096,
            "heads": 16,
            "seq": 256,
            "mbs": 1,
            "gbs": 16,
            "experts": 8,
            "topk": 2,
        },
        "M2": {
            "hidden": 768,
            "ffn": 3072,
            "heads": 12,
            "seq": 512,
            "mbs": 2,
            "gbs": 32,
            "experts": 8,
            "topk": 2,
        },
        "M3": {
            "hidden": 1024,
            "ffn": 4096,
            "heads": 16,
            "seq": 256,
            "mbs": 1,
            "gbs": 8,
            "experts": 8,
            "topk": 2,
        },
    }

    assert all(
        configs[config_id]["experts"] == 8
        and configs[config_id]["topk"] == 2
        for config_id in ("M1", "M2", "M3")
    )


def test_launcher_rejects_unknown_config_before_torchrun(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), "INVALID", "1F1B", str(tmp_path)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "unknown configuration" in result.stderr


def test_magellan_launcher_requires_both_schedule_files(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), "D1", "MAGELLAN", str(tmp_path)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "requires order and dependency JSON" in result.stderr


def test_moe_launcher_disables_linear_bias_for_alltoall_dispatcher():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    moe_block = launcher.split(
        'if [[ -n "${NUM_EXPERTS}" ]]; then',
        maxsplit=1,
    )[1].split("fi", maxsplit=1)[0]

    assert "--disable-bias-linear" in moe_block
