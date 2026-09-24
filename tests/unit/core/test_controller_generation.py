"""Generation admission preserves identity across its asynchronous confirmation."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.controller_generation import start_generation
from kohakuterrarium.errors import ConflictError, SessionError
from kohakuterrarium.modules.output.router import OutputRouter
from kohakuterrarium.modules.output.event import OutputEvent
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.testing.output import OutputRecorder


def _setup(store):
    agent = SimpleNamespace(_turn_index=1, _branch_id=1, _parent_branch_path=[])
    sink = SessionOutput("alice", store, agent)
    router = OutputRouter(OutputRecorder())
    router.add_secondary(sink)
    controller = SimpleNamespace(
        llm=ScriptedLLM(["unused"]),
        config=SimpleNamespace(tool_format="native"),
        session_store=store,
        output_router=router,
    )
    return agent, sink, controller


@pytest.mark.parametrize("change", ["provider", "format", "cancel"])
async def test_confirmation_does_not_admit_changed_provider_or_cancelled_waiter(
    tmp_path, change, monkeypatch
):
    store = SessionStore(tmp_path / "identity.kohakutr")
    agent, sink, controller = _setup(store)
    entered, release = threading.Event(), threading.Event()
    original = store.append_event
    task = None

    def append(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("unreleased origin")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "append_event", append)
    try:
        task = asyncio.create_task(start_generation(controller))
        assert await asyncio.to_thread(entered.wait, 5)
        if change == "provider":
            controller.llm = ScriptedLLM(["replacement"])
        elif change == "format":
            controller.config.tool_format = "bracket"
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        agent._turn_index = 2
        agent._branch_id = 7
        release.set()
        if change != "cancel":
            with pytest.raises(ConflictError):
                await task
        await sink.drain()
        events = await store.run(store.get_events, "alice")
        assert len(events) == 1
        assert events[0]["turn_index"] == events[0]["branch_id"] == 1
        assert sink._generation_id is None
        assert controller._active_generation is None
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await sink.stop()
        await asyncio.to_thread(store.close)


async def test_observer_payload_mutations_are_isolated(tmp_path):
    store = SessionStore(tmp_path / "observers.kohakutr")
    _, sink, controller = _setup(store)
    seen = []

    class Mutator(OutputRecorder):
        def on_activity_with_metadata(self, kind, detail, metadata=None):
            metadata["source"]["model"] = "observer mutation"
            raise RuntimeError("observer failed")

    class Observer(OutputRecorder):
        def on_activity_with_metadata(self, kind, detail, metadata=None):
            seen.append(metadata)

    controller.output_router.default_output = Mutator()
    controller.output_router.add_secondary(Observer())
    try:
        payload = await start_generation(controller)
        assert len(seen) == 1
        assert seen[0] == payload
        assert payload["source"]["model"] != "observer mutation"
        events = await store.run(store.get_events, "alice")
        assert len(events) == 1 and events[0]["source"] == payload["source"]
    finally:
        await sink.stop()
        await asyncio.to_thread(store.close)


async def test_user_input_boundary_releases_both_generation_stamp_and_coordinates(
    tmp_path,
):
    store = SessionStore(tmp_path / "input.kohakutr")
    agent, sink, controller = _setup(store)
    try:
        await start_generation(controller)
        agent._turn_index, agent._branch_id = 2, 3
        await sink.emit(OutputEvent(type="user_input", content="next"))
        await sink.write_stream("unattributed")
        await sink.flush()
        answer = (await store.run(store.get_events, "alice"))[-1]
        assert "generation_id" not in answer
        assert (answer["turn_index"], answer["branch_id"]) == (2, 3)
    finally:
        await sink.stop()
        await asyncio.to_thread(store.close)


async def test_duplicate_canonical_sinks_fail_before_origin_write(tmp_path):
    store = SessionStore(tmp_path / "duplicate.kohakutr")
    agent, sink, controller = _setup(store)
    other = SessionOutput("alice", store, agent)
    controller.output_router.add_secondary(other)
    try:
        with pytest.raises(SessionError, match="canonical"):
            await start_generation(controller)
        assert await store.run(store.get_events, "alice") == []
    finally:
        await sink.stop()
        await other.stop()
        await asyncio.to_thread(store.close)


async def test_attached_store_without_canonical_output_fails_closed(tmp_path):
    store = SessionStore(tmp_path / "missing.kohakutr")
    _, sink, controller = _setup(store)
    controller.output_router._secondary_outputs.clear()
    try:
        with pytest.raises(SessionError, match="canonical"):
            await start_generation(controller)
        assert controller._active_generation is None
        controller.session_store = None
        assert (await start_generation(controller))["schema"] == 1
    finally:
        await sink.stop()
        await asyncio.to_thread(store.close)
