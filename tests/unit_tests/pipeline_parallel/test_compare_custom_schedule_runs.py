import pytest

from tools.compare_custom_schedule_runs import summarize_runs


def test_comparison_reports_expected_medians_and_deltas():
    summary = summarize_runs(
        [1.0, 1.1, 0.9],
        [1.2, 1.1, 1.0],
        [1.3, 1.2, 1.1],
    )

    assert summary["runs"]["A_default_1f1b"]["median_ms"] == 1000
    assert summary["runs"]["B_custom_order"]["median_ms"] == 1100
    assert (
        summary["runs"]["C_custom_order_and_dependency"]["median_ms"]
        == 1200
    )
    assert summary["deltas"]["B_minus_A"]["milliseconds"] == pytest.approx(
        100
    )
    assert summary["deltas"]["C_minus_B"]["milliseconds"] == pytest.approx(
        100
    )
    assert summary["deltas"]["C_minus_A"]["percent"] == pytest.approx(20)
