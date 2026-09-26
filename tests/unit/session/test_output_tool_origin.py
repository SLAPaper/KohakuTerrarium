"""Tool activity persistence preserves explicit identity and dispatch coordinates."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore


async def test_legacy_terminal_does_not_invent_execution_class(tmp_path):
    store = SessionStore(str(tmp_path / "legacy.kohakutr"))
    sink = SessionOutput(
        "alice",
        store,
        SimpleNamespace(_turn_index=1, _branch_id=1, _parent_branch_path=[]),
    )
    try:
        sink.on_activity_with_metadata(
            "tool_done", "[read] done", {"job_id": "job", "result": "done"}
        )
        await sink.drain()
        event = (await store.run(store.get_events, "alice"))[0]
        assert event["call_id"] == event["job_id"] == "job"
        assert "execution_class" not in event
        assert "background" not in event
        assert "origin" not in event
    finally:
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("damage", ["coordinates", "path", "class"])
async def test_declared_invalid_origin_is_not_persisted(tmp_path, damage):
    store = SessionStore(str(tmp_path / "invalid.kohakutr"))
    sink = SessionOutput(
        "alice",
        store,
        SimpleNamespace(_turn_index=1, _branch_id=1, _parent_branch_path=[]),
    )
    origin = {
        "turn_index": 2,
        "branch_id": 1,
        "parent_branch_path": [[1, 1]],
        "execution_class": "background",
    }
    metadata = {"job_id": "job", "origin": origin, "background": True}
    if damage == "coordinates":
        origin["turn_index"] = True
    elif damage == "path":
        origin["parent_branch_path"] = [[3, 1]]
    else:
        metadata["execution_class"] = "direct"
    try:
        with pytest.raises(ValueError):
            sink.on_activity_with_metadata("tool_done", "[read] done", metadata)
        await sink.drain()
        assert await store.run(store.get_events, "alice") == []
    finally:
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("terminal", ["tool_done", "tool_error"])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("tuple_path", [False, True])
async def test_activity_retains_job_provider_and_origin_after_live_branch_moves(
    tmp_path, terminal, background, tuple_path
):
    store = SessionStore(str(tmp_path / "activity.kohakutr"))
    agent = SimpleNamespace(_turn_index=2, _branch_id=1, _parent_branch_path=[(1, 1)])
    sink = SessionOutput("alice", store, agent, event_key_prefix="host:alice")
    origin = {
        "runtime_scope_id": "scope",
        "turn_index": 2,
        "branch_id": 1,
        "parent_branch_path": [[1, 1]],
        "origin_path": [[1, 1], [2, 1]],
        "delivery_id": "scope:job",
        "execution_class": "background" if background else "direct",
    }
    if tuple_path:
        origin["origin_path"] = [(1, 1), (2, 1)]
    metadata = {
        "job_id": "job",
        "tool_call_id": "provider-id",
        "tool_name": "read",
        "args": {},
        "background": background,
        "execution_class": origin["execution_class"],
        "origin": origin,
    }
    expected = deepcopy(origin)
    expected["origin_path"] = [[1, 1], [2, 1]]
    try:
        sink.on_activity_with_metadata("tool_start", "[read] start", metadata)
        await sink.drain()
        agent._turn_index = 3
        agent._branch_id = 2
        agent._parent_branch_path = [(1, 1), (2, 2)]
        sink.on_activity_with_metadata(
            terminal,
            "[read] terminal",
            {
                **metadata,
                "result": "done",
                "error": "failed" if terminal == "tool_error" else "",
            },
        )
        origin["parent_branch_path"][0][1] = 999
        await sink.drain()
        events = await store.run(store.get_events, "host:alice")
        assert [e["type"] for e in events] == ["tool_call", "tool_result"]
        for event in events:
            assert event["job_id"] == "job"
            assert event["tool_call_id"] == "provider-id"
            assert event["execution_class"] == expected["execution_class"]
            assert event["background"] is background
            assert event["origin"] == expected
            assert (
                event["turn_index"],
                event["branch_id"],
                event["parent_branch_path"],
            ) == (2, 1, [[1, 1]])
        assert events[-1]["output"] == "done"
    finally:
        await sink.drain()
        await asyncio.to_thread(store.close)
