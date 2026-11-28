"""Source-bound context: identity, branch scope, and bounded traversal."""

import re
from types import SimpleNamespace

import numpy as np
import pytest

import kohakuterrarium.builtins.tools.search_memory as memory_tool
import kohakuterrarium.session.memory.navigation as navigation
from kohakuterrarium.core.conversation.continuation import persist_branch_view
from kohakuterrarium.core.session import Session
from kohakuterrarium.modules.tool.base import ToolContext
from kohakuterrarium.session.memory.embedding import BaseEmbedder
from kohakuterrarium.session.memory.index import SessionMemory
from kohakuterrarium.session.memory.source_events import sourced_events
from kohakuterrarium.session.store import SessionStore


class Embedder(BaseEmbedder):
    dimensions = 4

    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def context(store):
    return ToolContext(
        agent_name="alice",
        session=Session(key="memory"),
        agent=SimpleNamespace(session_store=store, config=None),
        working_dir=None,
    )


async def invoke(ctx, **args):
    result = await memory_tool.SearchMemoryTool().execute(args, context=ctx)
    assert result.error is None, result.error
    return result.output


def anchors(output):
    return re.findall(r"^anchor: (\S+)$", output, re.M)


async def find(ctx, query, **args):
    output = await invoke(ctx, query=query, mode=args.pop("mode", "fts"), **args)
    (anchor,) = anchors(output)
    return anchor


@pytest.fixture
def session(tmp_path):
    store = SessionStore(tmp_path / "memory.kohakutr")
    store.init_meta("s", "agent", "", str(tmp_path), ["alice", "bob"])
    try:
        yield store, context(store)
    finally:
        store.close(update_status=False)


async def test_lifecycle_independent_search_and_cross_turn_navigation(session):
    store, ctx = session

    def append(kind, **data):
        return store.append_event("alice", kind, data)

    append("user_input", content="ORIGINAL_REQUEST")
    append("tool_call", name="bash", args={"command": "ORIGINAL_CALL"})
    append("processing_end")
    anchor = await find(ctx, "ORIGINAL_CALL")
    append("tool_result", name="bash", output="LATE_BACKGROUND")
    late = await invoke(ctx, anchor=anchor, before=1, after=1)
    assert "ORIGINAL_REQUEST" in late and "LATE_BACKGROUND" in late
    assert "has_before=false" in late and "has_after=false" in late
    append("processing_start")
    append("tool_call", name="bash", args={"command": "SYNC_CALL"})
    append("tool_result", name="bash", output="EACCES")
    append("tool_result", name="bash", output="PARTIAL_STDOUT", error="TIMEOUT_ERROR")
    append("text_chunk", content="FOLLOWUP_REPLY")
    append("processing_end")
    append("user_input", content="NEXT_REQUEST")
    for marker in (
        "LATE_BACKGROUND",
        "SYNC_CALL",
        "EACCES",
        "PARTIAL_STDOUT",
        "TIMEOUT_ERROR",
        "FOLLOWUP_REPLY",
    ):
        hit = await find(ctx, marker)
        assert marker in await invoke(ctx, anchor=hit, before=0, after=0)
    hit = await find(ctx, "TIMEOUT_ERROR")
    window = await invoke(ctx, anchor=hit, before=1, after=2, scope="raw")
    assert all(
        x in window
        for x in ("EACCES", "PARTIAL_STDOUT", "TIMEOUT_ERROR", "NEXT_REQUEST")
    )
    assert (
        window.index("EACCES")
        < window.index("PARTIAL_STDOUT")
        < window.index("NEXT_REQUEST")
    )


async def test_selected_view_switch_and_raw_navigation(session):
    store, ctx = session
    for branch, marker in ((1, "EARLIER"), (2, "LATEST")):
        store.append_event(
            "alice",
            "user_input",
            {"content": marker},
            turn_index=1,
            branch_id=branch,
            parent_branch_path=[],
        )
        store.append_event(
            "alice",
            "text",
            {"content": f"{marker}_ANSWER"},
            turn_index=1,
            branch_id=branch,
            parent_branch_path=[],
        )
    old = await find(ctx, "LATEST_ANSWER")
    assert "No results found." in await invoke(ctx, query="EARLIER_ANSWER", mode="fts")
    selected = await invoke(ctx, anchor=old, before=3, after=3)
    assert "LATEST_ANSWER" in selected and "EARLIER" not in selected
    raw = await invoke(ctx, anchor=old, before=3, after=3, scope="raw")
    assert raw.index("EARLIER_ANSWER") < raw.index("LATEST_ANSWER")
    assert "turn=1 branch=1" in raw and "turn=1 branch=2" in raw
    sibling = anchors(raw)[0]
    assert "EARLIER" not in await invoke(ctx, anchor=sibling, before=0, after=0)
    assert "EARLIER" in await invoke(
        ctx, anchor=sibling, scope="raw", before=0, after=0
    )

    persist_branch_view(store, "alice", {1: 1})
    # Switching alone, with no append, changes search selection.
    first = await find(ctx, "EARLIER_ANSWER")
    assert "No results found." in await invoke(ctx, query="LATEST_ANSWER", mode="fts")
    store.append_event(
        "alice",
        "user_input",
        {"content": "RESUMED_REQUEST"},
        turn_index=2,
        branch_id=1,
        parent_branch_path=[(1, 1)],
    )
    fresh = await find(ctx, "RESUMED_REQUEST")
    assert "RESUMED_REQUEST" in await invoke(ctx, anchor=fresh, before=1, after=0)
    assert "LATEST" not in await invoke(ctx, anchor=first, before=3, after=3)
    original = await invoke(ctx, anchor=old, before=3, after=3)
    assert (
        "LATEST_ANSWER" in original
        and "EARLIER" not in original
        and "RESUMED_REQUEST" not in original
    )
    persist_branch_view(store, "alice", None)
    assert await find(ctx, "LATEST_ANSWER")
    assert "No results found." in await invoke(ctx, query="RESUMED_REQUEST", mode="fts")


async def test_legacy_lineage_uses_event_order_and_observes_late_append(session):
    store, ctx = session
    store.append_event(
        "alice", "user_input", {"content": "FIRST_TURN"}, turn_index=1, branch_id=1
    )
    store.append_event(
        "alice", "user_input", {"content": "SECOND_TURN"}, turn_index=2, branch_id=1
    )
    store.append_event(
        "alice",
        "tool_call",
        {"name": "bash", "args": {"command": "LEGACY_CALL"}},
        turn_index=2,
        branch_id=1,
    )
    store.append_event("alice", "processing_end", {}, turn_index=2, branch_id=1)
    anchor = await find(ctx, "LEGACY_CALL")
    store.append_event(
        "alice",
        "tool_result",
        {"name": "bash", "output": "LATE_COMPATIBLE"},
        turn_index=2,
        branch_id=1,
    )
    window = await invoke(ctx, anchor=anchor, before=2, after=1)
    assert "FIRST_TURN" in window and "LATE_COMPATIBLE" in window
    store.append_event(
        "alice",
        "user_input",
        {"content": "SIBLING_ROOT"},
        turn_index=1,
        branch_id=2,
        parent_branch_path=[],
    )
    store.append_event(
        "alice",
        "tool_result",
        {"name": "bash", "output": "LATE_INCOMPATIBLE"},
        turn_index=2,
        branch_id=1,
    )
    window = await invoke(ctx, anchor=anchor, before=0, after=3)
    assert (
        "LATE_COMPATIBLE" in window
        and "SIBLING_ROOT" not in window
        and "LATE_INCOMPATIBLE" not in window
    )
    assert "LATE_INCOMPATIBLE" in await invoke(
        ctx, anchor=anchor, before=0, after=3, scope="raw"
    )


async def test_ambiguous_reused_coordinates_do_not_authorize_other_lineage(session):
    store, ctx = session
    for turn, branch, path, marker in (
        (1, 1, [], "ROOT_ONE"),
        (2, 1, [(1, 1)], "CHILD_ONE"),
        (1, 2, [], "ROOT_TWO"),
        (2, 1, [(1, 2)], "CHILD_TWO"),
    ):
        store.append_event(
            "alice",
            "user_input",
            {"content": marker},
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=path,
        )
    token = await find(ctx, "ROOT_TWO")
    window = await invoke(ctx, anchor=token, before=3, after=3)
    assert "ROOT_TWO" in window and not any(
        x in window for x in ("ROOT_ONE", "CHILD_ONE", "CHILD_TWO")
    )


async def test_deep_legacy_unknown_neighbor_does_not_block_known_context(
    session, monkeypatch
):
    store, ctx = session
    store.append_event(
        "alice",
        "user_input",
        {"content": "ROOT"},
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    )
    for _ in range(navigation.MAX_LEGACY_PREFIX):
        store.append_event("alice", "processing_start", {}, turn_index=0, branch_id=0)
    for content, path in (
        ("KNOWN_FOCUS", [(1, 1)]),
        ("LEGACY_UNKNOWN", None),
        ("KNOWN_LATER", [(1, 1)]),
    ):
        store.append_event(
            "alice",
            "text",
            {"content": content},
            turn_index=2,
            branch_id=1,
            parent_branch_path=path,
        )
    focus = await find(ctx, "KNOWN_FOCUS")
    legacy = await find(ctx, "LEGACY_UNKNOWN")

    def forbid(*args, **kwargs):
        pytest.fail("Anchor expansion scanned the entire event history")

    monkeypatch.setattr(store, "_event_rows", forbid)
    selected = await invoke(ctx, anchor=focus, before=0, after=2)
    assert "KNOWN_LATER" in selected and "LEGACY_UNKNOWN" not in selected
    assert "incomplete" in selected and "has_after=unknown" in selected
    assert "LEGACY_UNKNOWN" not in await invoke(ctx, anchor=legacy, before=0, after=0)
    assert "LEGACY_UNKNOWN" in await invoke(
        ctx, anchor=legacy, before=0, after=0, scope="raw"
    )


async def test_bounded_navigation_handles_sparse_nonmonotonic_history(
    session, monkeypatch
):
    store, ctx = session
    store.append_event("alice", "user_input", {"content": "FOCUS", "event_id": 50})
    gap, _ = store.append_event("alice", "processing_end", {})
    store.append_event("alice", "text", {"content": "NEAR", "event_id": 10})
    store.flush()
    del store.events[gap]
    anchor = await find(ctx, "FOCUS")
    window = await invoke(ctx, anchor=anchor, before=0, after=1)
    assert "NEAR" in window and "has_after=false" in window
    for _ in range(navigation.MAX_SCANNED_PER_SIDE):
        store.append_event("alice", "processing_start", {})
    store.append_event("alice", "text", {"content": "DISTANT"})

    def forbid(*args, **kwargs):
        pytest.fail("Anchor expansion scanned the entire event history")

    monkeypatch.setattr(store, "_event_rows", forbid)
    window = await invoke(ctx, anchor=anchor, before=0, after=2)
    assert "FOCUS" in window and "NEAR" in window and "DISTANT" not in window
    assert "has_after=unknown" in window
    exact = await invoke(ctx, anchor=anchor, before=0, after=0)
    assert "FOCUS" in exact and "has_after=true" in exact
    near = anchors(window)[-1]
    exact = await invoke(ctx, anchor=near, before=0, after=0)
    assert "NEAR" in exact and "has_after=unknown" in exact


async def test_whole_event_pagination_from_call_and_paragraph_hit(session):
    store, ctx = session
    store.append_event(
        "alice", "tool_call", {"name": "bash", "args": {"command": "ORIGINAL_COMMAND"}}
    )
    payload = "FIRST_OUTPUT " + "x" * 1800 + " OUTPUT_TAIL"
    result_key, _ = store.append_event(
        "alice", "tool_result", {"name": "bash", "output": payload}
    )
    text = "FIRST_PARAGRAPH " + "İSTANBUL İZMİR " * 80 + "\n\n" + "UniCode_Tail_Marker"
    store.append_event("alice", "text", {"content": text})
    found = await invoke(ctx, query="ORIGINAL_COMMAND", mode="fts")
    (call,) = anchors(found)
    assert f'search_memory(anchor="{call}", before=0, after=3)' in found
    live = ctx.session._memory
    for scope in ("selected", "raw"):
        window = await invoke(ctx, anchor=call, before=0, after=1, scope=scope)
        result = anchors(window)[-1]
        offset = int(re.search(r"next_offset=(\d+)", window)[1])
        continuation = (
            f'search_memory(anchor="{result}", before=0, after=0, offset={offset}'
        )
        assert continuation + (', scope="raw")' if scope == "raw" else ")") in window
        page = (
            await live.read_context(result, before=0, after=0, offset=0, scope=scope)
        ).events[0]
        chunks = [page.content]
        while page.end < page.total:
            offset = page.end
            page = (
                await live.read_context(
                    result, before=0, after=0, offset=offset, scope=scope
                )
            ).events[0]
            assert page.start == offset < page.end
            chunks.append(page.content)
        assert "".join(chunks) == payload

    token = await find(ctx, "UNICODE_TAIL_MARKER")
    focused = (await live.read_context(token, before=0, after=0)).events[0]
    assert "UniCode_Tail_Marker" in focused.content
    assert focused.content == text[focused.start : focused.end]
    assert focused.total == len(text)
    first = (await live.read_context(token, before=0, after=0, offset=0)).events[0]
    assert first.content == text[: first.end] and first.end < len(text)
    page = (await live.read_context(token, before=0, after=0, offset=first.end)).events[
        0
    ]
    assert page.content == text[first.end : page.end]
    assert first.content + page.content == text

    # Changing the exact source invalidates the token in either scope.
    store.events[result_key] = {**store.events[result_key], "output": "changed"}
    for scope in ("selected", "raw"):
        error = await memory_tool.SearchMemoryTool().execute(
            {"anchor": result, "scope": scope}, context=ctx
        )
        assert "ANCHOR_EXPIRED" in error.error


async def test_invalid_navigation_arguments(session):
    store, ctx = session
    store.append_event("alice", "user_input", {"content": "QUERY_MARKER"})
    anchor = await find(ctx, "QUERY_MARKER")
    for args, message in (
        ({"anchor": "unknown"}, "INVALID_ANCHOR"),
        ({"anchor": anchor, "offset": -1}, "INVALID_OFFSET"),
        ({"anchor": anchor, "offset": 1000}, "INVALID_OFFSET"),
        ({"anchor": anchor, "before": 11}, "INVALID_WINDOW"),
        ({"anchor": anchor, "scope": "other"}, "INVALID_SCOPE"),
        ({"query": "QUERY_MARKER", "scope": "raw"}, "INVALID_SCOPE"),
        ({"query": "QUERY_MARKER", "anchor": anchor}, "exactly one"),
    ):
        result = await memory_tool.SearchMemoryTool().execute(args, context=ctx)
        assert message in result.error


async def test_saved_index_peer_hits_and_partial_source_failure(tmp_path, monkeypatch):
    path = tmp_path / "peers.kohakutr"
    store = SessionStore(path)
    store.init_meta("s", "agent", "", str(tmp_path), ["alice", "bob"])
    keys = {}
    try:
        index = SessionMemory(store.path, Embedder(), store=store)
        try:
            for agent in ("alice", "bob"):
                key, _ = store.append_event(
                    agent, "user_input", {"content": f"COMMON {agent.upper()}_EVIDENCE"}
                )
                keys[agent] = key
                index.index_events(agent, sourced_events(store, agent))
        finally:
            index.close()
    finally:
        store.close(update_status=False)

    store = SessionStore(path)
    ctx = context(store)
    monkeypatch.setattr(memory_tool, "create_embedder", lambda _: Embedder())
    original = store._event_rows

    def read(agent, *args, **kwargs):
        if agent == "bob":
            pytest.fail("Peer search must not scan or refresh unrelated history")
        return original(agent, *args, **kwargs)

    monkeypatch.setattr(store, "_event_rows", read)
    try:
        for mode in ("fts", "semantic", "hybrid", "auto"):
            token = await find(ctx, "BOB_EVIDENCE", mode=mode, agent="bob")
            assert "BOB_EVIDENCE" in await invoke(ctx, anchor=token, before=0, after=0)
        store.append_event("bob", "text", {"content": "PEER_NEW_NEIGHBOR"})
        assert "PEER_NEW_NEIGHBOR" in await invoke(ctx, anchor=token, before=0, after=1)
        store.events[keys["bob"]] = {**store.events[keys["bob"]], "content": "changed"}
        found = await invoke(ctx, query="COMMON", mode="fts")
        assert "BOB_EVIDENCE" in found and "ALICE_EVIDENCE" in found
        assert "Anchors unavailable for some hits" in found and "STALE_INDEX" in found
        (valid,) = anchors(found)
        assert "ALICE_EVIDENCE" in await invoke(ctx, anchor=valid, before=0, after=0)
    finally:
        store.close(update_status=False)


async def test_legacy_and_misbound_index_rows_keep_search_without_false_anchor(session):
    store, ctx = session
    store.append_event("alice", "user_input", {"content": "LEGACY_SOURCE"})
    store.append_event("alice", "text", {"content": "OTHER_SOURCE"})
    # Build an index without source metadata, as older releases did.
    memory = SessionMemory(store.path)
    try:
        memory.index_events("alice", store.get_events("alice"))
    finally:
        memory.close()
    for _ in range(2):
        output = await invoke(ctx, query="LEGACY_SOURCE", mode="fts")
        assert "LEGACY_SOURCE" in output and "ANCHOR_REBUILD_REQUIRED" in output
        assert not anchors(output)
    store.append_event("alice", "text", {"content": "NEW_SOURCE"})
    store.append_event("alice", "text", {"content": "WRONG_SOURCE"})
    await find(ctx, "NEW_SOURCE")
    await find(ctx, "WRONG_SOURCE")
    memory = ctx.session._memory._memory
    row, _, meta = memory._fts.search("NEW_SOURCE", k=1)[0]
    _, _, wrong = memory._fts.search("WRONG_SOURCE", k=1)[0]
    # The wrong event's fingerprint is real, but it does not produce this hit.
    memory._fts.update(
        row,
        value={
            **meta,
            "source_ref": wrong["source_ref"],
            "source_fingerprint": wrong["source_fingerprint"],
        },
    )
    output = await invoke(ctx, query="NEW_SOURCE", mode="fts")
    assert "NEW_SOURCE" in output and "STALE_INDEX" in output
    assert not anchors(output)


async def test_legacy_error_projection_and_unavailable_selected_view(session):
    store, ctx = session
    store.append_event(
        "alice",
        "tool_result",
        {"name": "bash", "error": "LEGACY_TIMEOUT_ERROR", "output": "LEGACY_STDOUT"},
        turn_index=1,
        branch_id=1,
    )
    await find(ctx, "LEGACY_TIMEOUT_ERROR")
    memory = ctx.session._memory._memory
    row, _, meta = memory._fts.search("LEGACY_TIMEOUT_ERROR", k=1)[0]
    memory._fts.delete(row)
    memory._fts.insert("[result:bash] LEGACY_TIMEOUT_ERROR", meta)
    # Persist a historical all-branch checkpoint with no selected-view proof.
    key = "alice:cursor:keywords"
    memory._state[key] = {**memory._state[key], "all_branches": True}
    token = await find(ctx, "LEGACY_TIMEOUT_ERROR")
    assert "LEGACY_TIMEOUT_ERROR" not in await invoke(
        ctx, anchor=token, before=0, after=0
    )
    raw = await invoke(ctx, anchor=token, before=0, after=0, scope="raw")
    assert "LEGACY_TIMEOUT_ERROR" in raw and "LEGACY_STDOUT" in raw
    row, _, meta = memory._fts.search("LEGACY_TIMEOUT_ERROR", k=1)[0]
    memory._fts.delete(row)
    memory._fts.insert("[result:bash] FORGED_ERROR", meta)
    output = await invoke(ctx, query="FORGED_ERROR", mode="fts")
    assert "STALE_INDEX" in output and not anchors(output)


async def test_anchor_survives_issuing_neighbor_at_capacity(session, monkeypatch):
    store, ctx = session
    # Exercise the public cursor contract with a deliberately small budget.
    monkeypatch.setattr(navigation, "MAX_ANCHORS", 3)
    store.append_event("alice", "user_input", {"content": "FOCUS"})
    store.append_event("alice", "text", {"content": "NEIGHBOR"})
    active, expired, newest = [await find(ctx, "FOCUS") for _ in range(3)]
    window = await invoke(ctx, anchor=active, before=0, after=1)
    assert "NEIGHBOR" in window
    assert "FOCUS" in await invoke(ctx, anchor=active, before=0, after=0)
    assert "FOCUS" in await invoke(ctx, anchor=newest, before=0, after=0)
    result = await memory_tool.SearchMemoryTool().execute(
        {"anchor": expired}, context=ctx
    )
    assert "INVALID_ANCHOR" in result.error
