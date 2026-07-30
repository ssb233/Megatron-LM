"""Validation for custom-schedule communication-delay experiments."""

from typing import Sequence, Tuple


def custom_dependencies_enabled(
    *,
    delay_pairs: Sequence[Tuple[float, float]],
    profile_ready: bool,
) -> bool:
    """Keep visualization-only dependencies off during profiling warmup."""

    has_visualization_delay = any(
        float(latency) != 0 or float(bandwidth) != 0
        for latency, bandwidth in delay_pairs
    )
    return not has_visualization_delay or profile_ready


def validate_custom_delay_configuration(
    *,
    pp_size: int,
    num_dc: int,
    pp_stages_per_dc: Sequence[int],
    delay_pairs: Sequence[Tuple[float, float]],
) -> None:
    """Validate the restricted delay modes supported by external schedules."""

    pairs = tuple((float(latency), float(bandwidth)) for latency, bandwidth in delay_pairs)
    if any(latency < 0 or bandwidth < 0 for latency, bandwidth in pairs):
        raise ValueError("custom schedule delay factors must be non-negative")

    has_delay = any(latency != 0 or bandwidth != 0 for latency, bandwidth in pairs)
    if not has_delay:
        if num_dc != 1:
            raise ValueError("zero-delay custom validation requires num_dc=1")
        return

    if any(latency != 0 for latency, _ in pairs):
        raise ValueError("custom visualization supports transfer-delay injection only")
    if pp_size != 4 or num_dc != 4:
        raise ValueError("custom visualization delay requires PP=4 and num_dc=4")
    if list(pp_stages_per_dc) != [1, 1, 1, 1]:
        raise ValueError(
            "custom visualization delay requires pp_stages_per_dc=[1,1,1,1]"
        )
