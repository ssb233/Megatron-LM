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


def test_iteration_parser_keeps_exactly_iterations_6_through_20():
    text = "\n".join(
        f"iteration {iteration}/ 20 | "
        f"elapsed time per iteration (ms): {100 + iteration}.0 |"
        for iteration in range(1, 21)
    )

    rows = parse_iteration_times(text)

    assert [row["iteration"] for row in rows] == list(range(6, 21))
    assert rows[0]["milliseconds"] == 106.0
    assert len(rows) == 15


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
