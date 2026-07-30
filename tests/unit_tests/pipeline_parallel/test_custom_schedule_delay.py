import pytest

from megatron.core.pipeline_parallel.cdc_scheduler.delay_config import (
    validate_custom_delay_configuration,
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
