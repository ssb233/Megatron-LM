import copy
import argparse
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
from megatron.core.pipeline_parallel.cdc_scheduler.execution_planner import (
    CommEvent,
    CommEventType,
    ExecutionPlanner,
)
from megatron.core.pipeline_parallel.cdc_scheduler.comm_dependency import (
    CommDependencyController,
)
from megatron.core.pipeline_parallel.cdc_scheduler import comm_dependency
from megatron.training.arguments import _add_distributed_args


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


def test_custom_schedule_command_line_arguments_are_independent():
    parser = _add_distributed_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--custom-pipeline-schedule",
            "replay.order.json",
            "--custom-comm-dependency",
            "notification_deps.json",
            "--custom-schedule-trace-dir",
            "trace",
        ]
    )

    assert args.custom_pipeline_schedule == "replay.order.json"
    assert args.custom_comm_dependency == "notification_deps.json"
    assert args.custom_schedule_trace_dir == "trace"


def _events_for_stage(planner, stage, event_types):
    return [
        event
        for compute_task in planner.execution_plan[stage]
        for event in compute_task.pre_events + compute_task.post_events
        if isinstance(event, CommEvent) and event.type in event_types
    ]


def test_execution_planner_rejects_receive_order_after_consumer(tmp_path):
    schedule = _load_schedule_json()
    schedule["comm"]["F_0_1"].reverse()
    path = _write_json(tmp_path, "reversed-forward-channel.json", schedule)
    spec = load_custom_schedule(
        str(path),
        None,
        pp_size=4,
        num_microbatches=4,
    )

    planner = ExecutionPlanner(get_custom_static_schedule(spec))

    with pytest.raises(ValueError, match="after its consumer"):
        planner.generate_execution_plan()


def test_execution_planner_does_not_hoist_future_receives():
    spec = load_custom_schedule(
        str(SCHEDULE),
        None,
        pp_size=4,
        num_microbatches=4,
    )
    planner = ExecutionPlanner(get_custom_static_schedule(spec))
    planner.generate_execution_plan()

    first_task_events = planner.execution_plan[0][0].pre_events
    assert not any(
        isinstance(event, CommEvent)
        and event.type == CommEventType.POST_RECV_NEXT
        for event in first_task_events
    )

    backward_zero = next(
        task
        for task in planner.execution_plan[0]
        if task.task_desc.type == "B" and task.task_desc.mb_id == 0
    )
    recv_types = [
        event.type
        for event in backward_zero.pre_events
        if isinstance(event, CommEvent)
    ]
    assert recv_types[:2] == [
        CommEventType.POST_RECV_NEXT,
        CommEventType.WAIT_RECV_NEXT,
    ]


def test_execution_planner_places_local_cross_channel_dependency_in_order():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    planner = ExecutionPlanner(get_custom_static_schedule(spec))
    planner.generate_execution_plan()

    sends = _events_for_stage(
        planner,
        2,
        {CommEventType.POST_SEND_NEXT, CommEventType.POST_SEND_PREV},
    )
    names = [
        CommOpId(
            event.task_type,
            event.mb_id,
            event.src_dev_id,
            event.dst_dev_id,
        ).name
        for event in sends
    ]
    assert names.index("Comm_F_3_2_3") < names.index("Comm_B_0_2_1")


class _FakeWork:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class _FakeGlooWork:
    def __init__(self):
        self.wait_count = 0

    def is_completed(self):
        # Matches the server's Gloo Work behavior: polling alone does not
        # advance completion, while wait() consumes the completed receive.
        return False

    def wait(self, timeout=None):
        self.wait_count += 1
        return True


def test_local_dependency_waits_for_predecessor_work():
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    controller = CommDependencyController(
        spec,
        pipeline_stage=2,
        control_group=object(),
        pipeline_global_ranks=[0, 1, 2, 3],
    )
    trigger = CommOpId("F", 3, 2, 3)
    target = CommOpId("B", 0, 2, 1)
    work = _FakeWork()

    controller.register_send(trigger, work, forward_only=False)
    controller.before_send(target, forward_only=False)

    assert work.wait_count == 1


def test_remote_dependency_sends_and_receives_cpu_signal(monkeypatch):
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    sent = []

    def fake_isend(payload, dst, group, tag):
        sent.append((int(payload.item()), dst, group, tag))
        return _FakeWork()

    monkeypatch.setattr(comm_dependency.dist, "isend", fake_isend)
    source = CommDependencyController(
        spec,
        pipeline_stage=1,
        control_group="gloo",
        pipeline_global_ranks=[0, 1, 2, 3],
    )
    trigger = CommOpId("F", 3, 1, 2)
    work = _FakeWork()
    source.register_send(trigger, work, forward_only=False)
    source.finish_iteration()

    assert work.wait_count == 1
    assert len(sent) == 1
    dependency_id, destination, group, tag = sent[0]
    assert destination == 3
    assert group == "gloo"

    def fake_irecv(payload, src, group, tag):
        assert src == 1
        assert group == "gloo"
        payload.fill_(dependency_id)
        return _FakeWork()

    monkeypatch.setattr(comm_dependency.dist, "irecv", fake_irecv)
    target = CommDependencyController(
        spec,
        pipeline_stage=3,
        control_group="gloo",
        pipeline_global_ranks=[0, 1, 2, 3],
    )
    target.before_send(
        CommOpId("B", 0, 3, 2),
        forward_only=False,
    )


def test_remote_dependency_waits_on_gloo_work_instead_of_polling(monkeypatch):
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    remote_dependency = next(
        dependency
        for dependency in spec.dependencies
        if dependency.is_remote
    )
    signal_work = _FakeGlooWork()

    def fake_irecv(payload, src, group, tag):
        payload.fill_(remote_dependency.dependency_id)
        return signal_work

    monkeypatch.setattr(comm_dependency.dist, "irecv", fake_irecv)
    target = CommDependencyController(
        spec,
        pipeline_stage=remote_dependency.target.src_stage,
        control_group="gloo",
        pipeline_global_ranks=[0, 1, 2, 3],
        timeout_seconds=0,
    )

    target.before_send(
        remote_dependency.target,
        forward_only=False,
    )

    assert signal_work.wait_count == 1


def test_forward_only_skips_signal_whose_target_is_backward(monkeypatch):
    spec = load_custom_schedule(
        str(SCHEDULE),
        str(DEPENDENCIES),
        pp_size=4,
        num_microbatches=4,
    )
    monkeypatch.setattr(
        comm_dependency.dist,
        "isend",
        lambda *args, **kwargs: pytest.fail("unexpected signal"),
    )
    source = CommDependencyController(
        spec,
        pipeline_stage=1,
        control_group="gloo",
        pipeline_global_ranks=[0, 1, 2, 3],
    )
    source.register_send(
        CommOpId("F", 3, 1, 2),
        _FakeWork(),
        forward_only=True,
    )
    source.finish_iteration()
