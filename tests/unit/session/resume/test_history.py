"""Replay consumers receive one fixed history boundary with retryable delivery."""

import asyncio
from contextlib import closing
from types import SimpleNamespace

import pytest

from kohakuterrarium.builtins.outputs.none import NoneOutput
from kohakuterrarium.builtins.outputs.stdout import StdoutOutput
from kohakuterrarium.modules.output.base import BaseOutputModule, wants_resume_history
from kohakuterrarium.modules.output.router.router import OutputRouter
from kohakuterrarium.session.resume.history import (
    ResumeHistory,
    replay_output_history,
    stage_output_history,
    take_output_history,
)
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.history.read import read_restore_events


class Recorder(NoneOutput):
    def __init__(self):
        super().__init__()
        self.batches = []

    async def on_resume(self, events):
        self.batches.append(events)


def staged(store, output):
    store.append_event("a", "user_input", {"content": "old"})
    store.flush()
    agent = SimpleNamespace(output_router=OutputRouter(output))
    stage_output_history(agent, store, "a", store.get_events("a"))
    return agent


async def test_headless_defers_normalization_and_late_attachment_stays_frozen(tmp_path):
    with closing(SessionStore(tmp_path / "history.kohakutr")) as store:
        agent = staged(store, NoneOutput())
        await replay_output_history(agent)
        store.append_event("a", "user_input", {"content": "new"})
        consumer = Recorder()
        agent.output_router.default_output = consumer
        await replay_output_history(agent)
        await replay_output_history(agent)
        assert [[e["content"] for e in batch] for batch in consumer.batches] == [
            ["old"]
        ]
        assert agent._resume_history is None


async def test_outputs_rebuild_history_from_a_lightweight_reference(tmp_path):
    with closing(SessionStore(tmp_path / "owned.kohakutr")) as store:
        consumer = Recorder()
        agent = staged(store, consumer)
        agent.output_router.default_output = NoneOutput()
        await replay_output_history(agent)
        agent.output_router.default_output = consumer
        await replay_output_history(agent)
        assert consumer.batches[0][0]["content"] == "old"


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_delivery_failure_preserves_history_for_retry(tmp_path, failure):
    class Failing(Recorder):
        async def on_resume(self, events):
            raise failure()

    with closing(SessionStore(tmp_path / "retry.kohakutr")) as store:
        agent = staged(store, Failing())
        with pytest.raises(failure):
            await replay_output_history(agent)
        consumer = Recorder()
        agent.output_router.default_output = consumer
        await replay_output_history(agent)
        assert consumer.batches[0][0]["content"] == "old"


async def test_output_replaced_during_preparation_does_not_receive_stale_data(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Waiting(Recorder):
        async def prepare_resume(self, history):
            entered.set()
            await release.wait()
            return await history.events()

    with closing(SessionStore(tmp_path / "replace.kohakutr")) as store:
        original = Waiting()
        replacement = Recorder()
        agent = staged(store, original)
        task = asyncio.create_task(replay_output_history(agent))
        await entered.wait()
        await replay_output_history(agent)
        agent.output_router.default_output = replacement
        release.set()
        await task
        assert original.batches == replacement.batches == []
        await replay_output_history(agent)
        assert replacement.batches[0][0]["content"] == "old"


async def test_new_restore_staged_during_old_delivery_is_not_discarded(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Waiting(Recorder):
        async def prepare_resume(self, history):
            entered.set()
            await release.wait()
            return await history.events()

    with closing(SessionStore(tmp_path / "restage.kohakutr")) as store:
        output = Waiting()
        agent = staged(store, output)
        task = asyncio.create_task(replay_output_history(agent))
        await entered.wait()
        store.append_event("a", "user_input", {"content": "new"})
        stage_output_history(agent, store, "a", store.get_events("a"))
        replacement = agent._resume_history
        release.set()
        await task
        assert agent._resume_history is replacement
        await replay_output_history(agent)
        assert [[e["content"] for e in batch] for batch in output.batches] == [
            ["old"],
            ["old", "new"],
        ]


async def test_manual_replay_is_consumed_once(tmp_path):
    with closing(SessionStore(tmp_path / "manual.kohakutr")) as store:
        agent = staged(store, NoneOutput())
        assert (await take_output_history(agent))[0]["content"] == "old"
        assert await take_output_history(agent) is None


def test_consumers_include_legacy_callbacks_and_explicit_opt_out():
    assert not wants_resume_history(NoneOutput())
    assert wants_resume_history(Recorder())
    output = Recorder()
    output.needs_resume_history = False
    assert not wants_resume_history(output)
    assert BaseOutputModule.on_resume is not Recorder.on_resume


@pytest.mark.parametrize("sequence", [0, 999999])
@pytest.mark.parametrize("mode", ["headless", "pending", "dropped"])
async def test_frozen_replay_excludes_later_mirrored_ids(tmp_path, sequence, mode):
    with closing(SessionStore(tmp_path / "mirror.kohakutr")) as store:
        store._event_seq["a"] = sequence
        store.append_event("a", "user_input", {"event_id": 10, "content": "old"})
        consumer = Recorder()
        agent = SimpleNamespace(
            output_router=OutputRouter(NoneOutput() if mode == "headless" else consumer)
        )
        stage_output_history(agent, store, "a", store.get_events("a"))
        if mode == "dropped":
            agent.output_router.default_output = NoneOutput()
            await replay_output_history(agent)
        store.append_event("a", "user_input", {"event_id": 5, "content": "new"})
        agent.output_router.default_output = consumer
        await replay_output_history(agent)
        assert [
            [event["content"] for event in batch] for batch in consumer.batches
        ] == [["old"]]


def test_captured_boundary_is_usable_after_later_mirror_arrivals(tmp_path):
    with closing(SessionStore(tmp_path / "captured.kohakutr")) as store:
        store.append_event("a", "user_input", {"event_id": 10, "content": "old"})
        captured = store.capture_event_read_boundary("a")
        store.append_event("a", "user_input", {"event_id": 5, "content": "new"})
        frozen = ResumeHistory(
            store, "a", 10, before_sequence=captured[0], legacy_keys=captured[1]
        )
        assert [event["content"] for event in frozen.read()] == ["old"]


def test_empty_namespace_freezes_without_native_receipt_metadata(tmp_path):
    with closing(SessionStore(tmp_path / "empty.kohakutr")) as store:
        store.append_event("b", "user_input", {"event_id": 10, "content": "other"})
        frozen = ResumeHistory(store, "a", 10)
        store.append_event("a", "user_input", {"event_id": 5, "content": "new"})
        assert frozen.read() == []


def test_opaque_legacy_keys_freeze_without_dropping_existing_records(tmp_path):
    with closing(SessionStore(tmp_path / "legacy-keys.kohakutr")) as store:
        store.append_event("a", "user_input", {"event_id": 10, "content": "own"})
        store.events["a:ez"] = {
            "type": "user_input",
            "event_id": 9,
            "content": "legacy",
        }
        store.events["a:e①"] = {
            "type": "user_input",
            "event_id": 8,
            "content": "unicode",
        }
        frozen = ResumeHistory(store, "a", 10)
        store.events["a:ea"] = {
            "type": "user_input",
            "event_id": 5,
            "content": "future",
        }
        store.append_event("a", "user_input", {"event_id": 4, "content": "later"})
        assert [event["content"] for event in frozen.read()] == [
            "own",
            "legacy",
            "unicode",
        ]


def test_direct_legacy_numeric_key_uses_fallback_sequence_boundary(tmp_path):
    with closing(SessionStore(tmp_path / "direct-key.kohakutr")) as store:
        store.append_event("a", "user_input", {"event_id": 10, "content": "own"})
        store.events["a:e000009"] = {
            "type": "user_input",
            "event_id": 9,
            "content": "legacy",
        }
        captured = store.capture_event_read_boundary("a")
        assert captured[0] == 10
        frozen = ResumeHistory(
            store, "a", 10, before_sequence=captured[0], legacy_keys=captured[1]
        )
        assert frozen.before_sequence == 10
        store.append_event("a", "user_input", {"event_id": 5, "content": "future"})
        assert [event["content"] for event in frozen.read()] == ["own", "legacy"]


@pytest.mark.parametrize("sequence", [999998, 999999])
def test_direct_numeric_key_across_width_boundary_is_part_of_frozen_history(
    tmp_path, sequence
):
    with closing(SessionStore(tmp_path / "direct-wide-key.kohakutr")) as store:
        store._event_seq["a"] = sequence
        store.append_event("a", "user_input", {"event_id": 10, "content": "own"})
        store.events["a:e1000000"] = {
            "type": "user_input",
            "event_id": 9,
            "content": "legacy",
        }
        frozen = ResumeHistory(
            store,
            "a",
            10,
        )
        store.append_event("a", "user_input", {"event_id": 5, "content": "future"})
        assert {event["content"] for event in frozen.read()} == {"own", "legacy"}


def test_overlapping_namespaces_cannot_expand_a_frozen_replay(tmp_path):
    with closing(SessionStore(tmp_path / "prefix-collision.kohakutr")) as store:
        store.append_event(
            "a:e-other", "user_input", {"event_id": 9, "content": "other"}
        )
        store.append_event("a", "user_input", {"event_id": 10, "content": "own"})
        frozen = ResumeHistory(store, "a", 10)
        store.append_event(
            "a:e-other", "user_input", {"event_id": 5, "content": "future"}
        )
        assert [event["content"] for event in frozen.read()] == ["own"]


async def test_visible_append_after_frozen_read_cannot_poison_a_future_projection(
    tmp_path,
):
    from_output = lambda events, **kwargs: ",".join(
        e.get("content", "") for e in events
    )
    with closing(SessionStore(tmp_path / "projection-boundary.kohakutr")) as store:
        store.append_event("a", "user_input", {"content": "old"})
        captured = store.capture_event_read_boundary("a")
        frozen = ResumeHistory(
            store, "a", 1, before_sequence=captured[0], legacy_keys=captured[1]
        )
        store.append_event("a", "user_input", {"content": "new"})
        assert await frozen.projection(from_output) == "old"
        current = ResumeHistory(
            store,
            "a",
            2,
        )
        assert await current.projection(from_output) == "old,new"


async def test_stdout_subclass_legacy_callback_receives_complete_events(tmp_path):
    class Custom(StdoutOutput):
        received = None

        async def on_resume(self, events):
            self.received = events

    with closing(SessionStore(tmp_path / "subclass.kohakutr")) as store:
        custom = Custom()
        agent = staged(store, custom)
        store.append_event(
            "a", "tool_call", {"call_id": "pending", "name": "read", "args": {"x": 1}}
        )
        stage_output_history(agent, store, "a", read_restore_events(store, "a"))
        await replay_output_history(agent)
        assert custom.received == store.get_resumable_events("a")
        assert custom.received[-1]["interrupted"] is True
        assert custom.received[-1]["args"] == {"x": 1}
        assert any(event.get("_synthetic_announce") for event in custom.received)


async def test_projection_normalizes_unfinished_work_by_default(tmp_path):
    with closing(SessionStore(tmp_path / "normalized.kohakutr")) as store:
        store.append_event(
            "a", "tool_call", {"call_id": "pending", "name": "read", "args": {"x": 1}}
        )
        source = ResumeHistory(store, "a", 1)

        def render(events, **kwargs):
            assert events[-1]["interrupted"] is True
            assert events[-1]["args"] == {"x": 1}
            return ",".join(event["type"] for event in events)

        assert await source.projection(render) == (
            "tool_call,assistant_tool_calls,tool_result"
        )


@pytest.mark.parametrize("failed_preparation", [False, True])
async def test_raw_rewrite_between_resume_and_replay_is_observed(
    tmp_path, failed_preparation
):
    with closing(SessionStore(tmp_path / "rewritten.kohakutr")) as store:
        key, _ = store.append_event("a", "user_input", {"content": "old"})
        agent = SimpleNamespace(output_router=OutputRouter(StdoutOutput()))
        stage_output_history(agent, store, "a", read_restore_events(store, "a"))
        source = agent._resume_history
        if failed_preparation:

            def fail(events, **kwargs):
                raise RuntimeError("renderer failed")

            with pytest.raises(RuntimeError):
                await source.projection(fail)
        event = store.events[key]
        event["content"] = "rewritten"
        store.events[key] = event
        assert source.read()[0]["content"] == "rewritten"
