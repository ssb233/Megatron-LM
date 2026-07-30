import pytest

from megatron.core.pipeline_parallel.cdc_scheduler.delay_config import (
    validate_custom_delay_configuration,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_scheduler import (
    CDCPPScheduler,
    process_pp_stages_per_dc,
)


def test_zero_delay_custom_schedule_accepts_single_dc():
    validate_custom_delay_configuration(
        pp_size=4,
        num_dc=1,
        pp_stages_per_dc=[],
        delay_pairs=[(0.0, 0.0)],
    )


def test_nonzero_delay_rejects_single_dc():
    with pytest.raises(ValueError, match="num_dc=4"):
        validate_custom_delay_configuration(
            pp_size=4,
            num_dc=1,
            pp_stages_per_dc=[],
            delay_pairs=[(0.0, 0.5)],
        )


def test_transfer_delay_accepts_one_stage_per_dc():
    validate_custom_delay_configuration(
        pp_size=4,
        num_dc=4,
        pp_stages_per_dc=[1, 1, 1, 1],
        delay_pairs=[(0.0, 0.5), (0.0, 1.0)],
    )


@pytest.mark.parametrize(
    "pairs,match",
    [
        ([(-0.1, 0.5)], "non-negative"),
        ([(0.5, 0.0)], "transfer-delay"),
    ],
)
def test_visualization_delay_rejects_unsupported_pairs(pairs, match):
    with pytest.raises(ValueError, match=match):
        validate_custom_delay_configuration(
            pp_size=4,
            num_dc=4,
            pp_stages_per_dc=[1, 1, 1, 1],
            delay_pairs=pairs,
        )


class _FakeExperimentManager:
    profile_result = {"T_F": [0.01]}
    T_F_stage = 0.01

    def need_schedule_update_in_current_iter(self):
        return True

    def get_injected_latency_bandwidth_delay_seconds(self):
        return 0.0, 0.005

    def get_injected_latency_bandwidth_delay_as_F_stage(self):
        return 0.0, 0.5


def test_custom_schedule_updates_delay_without_replacing_plan():
    scheduler = CDCPPScheduler.__new__(CDCPPScheduler)
    marker = object()
    scheduler.custom_schedule_spec = marker
    scheduler.pp_execution_plan = marker
    scheduler.exp_manager = _FakeExperimentManager()
    scheduler.injected_latency_delay = (0.0, 0.0)
    scheduler.injected_bandwidth_delay = (0.0, 0.0)
    scheduler.cdc_print = lambda *args, **kwargs: None

    scheduler.update_schedule_with_latency_bandwidth()

    assert scheduler.injected_latency_delay == (0.0, 0.0)
    assert scheduler.injected_bandwidth_delay == (0.5, 0.005)
    assert scheduler.pp_execution_plan is marker


def test_process_pp_stages_per_dc_accepts_explicit_stage_counts():
    assert process_pp_stages_per_dc([1, 1, 1, 1], 4, 4) == [1, 1, 1, 1]
