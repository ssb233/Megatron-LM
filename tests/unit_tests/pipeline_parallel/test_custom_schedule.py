import copy
import json
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.cdc_scheduler.custom_schedule import (
    CommOpId,
    load_custom_schedule,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    CustomOneChunkPipeline,
    get_custom_static_schedule,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import (
    SystemConfig,
)


FIXTURES = Path(__file__).parent / "fixtures"
SCHEDULE = FIXTURES / "replay_order_pp4_n4.json"
DEPENDENCIES = FIXTURES / "notification_deps_pp4_n4.json"


def _write_json(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _load_schedule_json():
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def _load_dependency_json():
    return json.loads(DEPENDENCIES.read_text(encoding="utf-8"))


def test_loads_pp4_schedule_and_normalizes_dependencies():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )

    assert [op.name for op in spec.compute_order[3]] == [
        "F_0_3",
        "B_0_3",
        "F_1_3",
        "B_1_3",
        "F_2_3",
        "B_2_3",
        "F_3_3",
        "B_3_3",
    ]
    assert len(spec.comm_order) == 6
    assert len(spec.dependencies) == 2
    assert len(spec.canonical_sha256) == 64

    local_target = CommOpId("B", 0, 2, 1)
    assert CommOpId("F", 3, 2, 3) in spec.predecessors_for(local_target)

    remote_target = CommOpId("B", 0, 3, 2)
    remote = spec.remote_dependencies_for(remote_target)
    assert len(remote) == 1
    assert remote[0].trigger == CommOpId("F", 3, 1, 2)
    assert remote[0].target == remote_target
    assert remote[0].is_remote


def test_notify_pair_collapses_to_one_remote_dependency():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )

    matching = [
        dependency
        for dependency in spec.dependencies
        if dependency.trigger.name == "Comm_F_3_1_2"
        and dependency.target.name == "Comm_B_0_3_2"
    ]
    assert len(matching) == 1


def test_same_channel_dependency_is_not_duplicated():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )

    target = CommOpId("F", 1, 0, 1)
    assert spec.predecessors_for(target) == (CommOpId("F", 0, 0, 1),)
    assert all(dependency.target != target for dependency in spec.dependencies)


def test_rejects_missing_compute_operation(tmp_path):
    schedule = _load_schedule_json()
    schedule["compute"]["stage_0"].pop()
    path = _write_json(tmp_path, "bad-order.json", schedule)

    with pytest.raises(ValueError, match="stage_0 has 7 operations"):
        load_custom_schedule(
            str(path),
            None,
            pp_size=4,
            num_microbatches=4,
        )


def test_rejects_unknown_dependency_endpoint(tmp_path):
    dependencies = _load_dependency_json()
    dependencies["edges"][0]["from_op"] = "Comm_F_99_0_1"
    path = _write_json(tmp_path, "bad-dependencies.json", dependencies)

    with pytest.raises(ValueError, match="communication-to-communication"):
        load_custom_schedule(
            str(SCHEDULE),
            str(path),
            pp_size=4,
            num_microbatches=4,
        )


def test_rejects_cycle_in_combined_compute_comm_graph(tmp_path):
    dependencies = {
        "edge_count": 1,
        "edges": [
            {
                "from_op": "Comm_B_0_3_2",
                "to_op": "Comm_F_0_0_1",
                "reason": "test_cycle"
            }
        ],
    }
    path = _write_json(tmp_path, "cycle.json", dependencies)

    with pytest.raises(ValueError, match="contains a cycle"):
        load_custom_schedule(
            str(SCHEDULE),
            str(path),
            pp_size=4,
            num_microbatches=4,
        )


def test_digest_is_independent_of_json_object_order(tmp_path):
    schedule = _load_schedule_json()
    reordered = copy.deepcopy(schedule)
    reordered["compute"] = dict(reversed(list(reordered["compute"].items())))
    reordered["comm"] = dict(reversed(list(reordered["comm"].items())))
    path = _write_json(tmp_path, "reordered.json", reordered)

    first = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    second = load_custom_schedule(
        str(path),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    assert first.canonical_sha256 == second.canonical_sha256


def test_custom_pipeline_preserves_each_stage_compute_order():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    pipeline = get_custom_static_schedule(spec)

    for stage, expected in enumerate(spec.compute_order):
        actual = pipeline.device_scheduled_tasks[stage]
        assert [
            (task.task_type, task.microbatch_id, task.chunk_id) for task in actual
        ] == [(operation.direction, operation.microbatch, 0) for operation in expected]
        assert all(
            task.prev_device_task is previous
            for task, previous in zip(actual[1:], actual[:-1])
        )

    assert pipeline.device_scheduled_tasks[3][1].task_type == "B"
    assert pipeline.device_scheduled_tasks[3][1].microbatch_id == 0


def test_custom_pipeline_rejects_mismatched_system_config():
    spec = load_custom_schedule(
        str(SCHEDULE),
        None,
        pp_size=4,
        num_microbatches=4,
    )
    config = SystemConfig(num_devices=3, num_microbatches=4, num_chunks=1)

    with pytest.raises(ValueError, match="PP size"):
        CustomOneChunkPipeline(config, spec)
