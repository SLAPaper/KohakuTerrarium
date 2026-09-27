"""Child allocation reservations survive reopen without a completed metadata row."""

import asyncio

import pytest

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.subagent_lineage import lineage_key, read_lineage
from kohakuterrarium.terrarium.session_subagent_state import (
    capture_children,
    depart_children,
    validate_children,
    import_children,
)


async def test_reserved_child_indices_survive_reopen(tmp_path):
    path = tmp_path / "reserved.kohakutr"
    store = SessionStore(path)
    try:
        assert await store.run(store.next_subagent_run, "parent", "child") == 0
        assert await store.run(store.next_subagent_run, "parent", "child") == 1
    finally:
        await asyncio.to_thread(store.close)
    reopened = SessionStore(path)
    try:
        assert await reopened.run(reopened.next_subagent_run, "parent", "child") == 2
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.parametrize("departed", [False, True])
async def test_fork_does_not_duplicate_child_ownership(tmp_path, departed):
    source = SessionStore(tmp_path / "source.kohakutr")
    current = SessionStore(tmp_path / "current.kohakutr")
    forked = None
    try:
        _, event_id = await source.run(
            source.append_event, "parent", "user_message", {"content": "fork point"}
        )
        await source.run(source.next_subagent_run, "parent", "child")
        await source.run(
            source.save_subagent, "parent", "child", 0, {"job_id": "j"}, "child history"
        )
        if departed:
            snapshot = await source.run(capture_children, source, "parent")
            await source.run(depart_children, source, snapshot, "successor")
            await current.run(import_children, current, snapshot, "successor")
        forked = source.fork(str(tmp_path / "fork.kohakutr"), at_event_id=event_id)
        original = await source.run(read_lineage, source, "parent")
        copied = await forked.run(read_lineage, forked, "parent")
        assert copied is None or (copied["origin"], copied["token"]) != (
            original["origin"],
            original["token"],
        )
        assert (
            await forked.run(forked.load_subagent_conversation, "parent", "child", 0)
            == "child history"
        )
        if departed:
            snapshot = await current.run(capture_children, current, "parent")
            with pytest.raises(ConflictError):
                await forked.run(validate_children, forked, snapshot)
        assert await forked.run(forked.next_subagent_run, "parent", "child") >= 1
    finally:
        for store in (source, current, forked):
            if store is not None:
                await asyncio.to_thread(store.close)


@pytest.mark.parametrize("state", ["corrupt", "installing"])
async def test_unavailable_child_namespace_does_not_hide_session_history(
    tmp_path, state
):
    path = tmp_path / "readable.kohakutr"
    store = SessionStore(path)
    await store.run(
        store.append_event, "parent", "user_message", {"content": "retained history"}
    )
    await store.run(store.next_subagent_run, "parent", "child")

    def damage():
        key = lineage_key("parent")
        record = store.state[key]
        if state == "corrupt":
            record.pop("origin")
        else:
            record["phase"] = "installing"
        store.state[key] = record

    await store.run(damage)
    await asyncio.to_thread(store.close)
    reopened = SessionStore(path)
    try:
        assert (await reopened.run(reopened.get_events, "parent"))[0][
            "content"
        ] == "retained history"
        with pytest.raises(ConflictError):
            await reopened.run(reopened.next_subagent_run, "parent", "child")
        assert await reopened.run(reopened.next_subagent_run, "unrelated", "child") == 0
    finally:
        await asyncio.to_thread(reopened.close)


async def test_child_name_cannot_alias_a_different_parent_namespace(tmp_path):
    store = SessionStore(tmp_path / "names.kohakutr")
    try:
        assert await store.run(store.next_subagent_run, "a:b", "c") == 0
        with pytest.raises(ConflictError):
            await store.run(store.next_subagent_run, "a", "b:c")
        with pytest.raises(ConflictError):
            await store.run(store.save_subagent, "a", "b:c", 0, {"job_id": "alias"})
        assert await store.run(store.load_subagent_meta, "a:b", "c", 0) is None
        assert await store.run(store.next_subagent_run, "a:b", "c") == 1
    finally:
        await asyncio.to_thread(store.close)
