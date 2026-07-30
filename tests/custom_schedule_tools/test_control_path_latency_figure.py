import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = (
    REPO_ROOT
    / "results"
    / "custom_schedule_v100"
    / "validation_20260730"
    / "figure"
)
PLOT_SCRIPT = FIGURE_DIR / "plot_custom_schedule_trace.py"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_custom_schedule_trace",
        PLOT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_control_latency_samples_matches_archived_trace():
    module = _load_plot_module()
    payload = json.loads(
        module.SIGNAL_DATA_PATH.read_text(encoding="utf-8")
    )

    samples = module.build_control_latency_samples(
        module.load_rows(),
        payload,
    )

    assert len(samples) == 56
    assert sum(
        row["sender_included_in_plot"] for row in samples
    ) == 55
    sender = [
        row["sender_complete_to_send_us"]
        for row in samples
        if row["sender_included_in_plot"]
    ]
    gloo = [row["gloo_ready_to_recv_us"] for row in samples]
    receiver = [
        row["receiver_recv_to_submit_us"] for row in samples
    ]
    assert np.median(sender) == pytest.approx(183.266)
    assert np.median(gloo) == pytest.approx(225.405)
    assert np.median(receiver) == pytest.approx(345.858)

