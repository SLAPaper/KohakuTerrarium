"""Selected snapshot branch_view survives close/reopen and fail-closes.

The cold-resume path reads ``snapshot_branch.branch_view`` as the selected
raw map. A non-latest selection must restore those coordinates, not the
latest live subtree. A view that disagrees with raw ancestry raises
SessionError instead of silently loading latest.
"""

import asyncio

import pytest

from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.compact_branch import persist_compacted
from kohakuterrarium.errors import ConflictError, SessionError
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.resume import inject_saved_state
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


def _seed_two_branches(store: SessionStore) -> None:
    """Turn 1 has branch 1 ("first") and later branch 2 ("second")."""
    store.append_event(
        "alice",
        "user_message",
        {"content": "question"},
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    )
    store.append_event(
        "alice",
        "text_chunk",
        {"content": "first"},
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    )
    store.append_event(
        "alice",
        "user_message",
        {"content": "question"},
        turn_index=1,
        branch_id=2,
        parent_branch_path=[],
    )
    store.append_event(
        "alice",
        "text_chunk",
        {"content": "second"},
        turn_index=1,
        branch_id=2,
        parent_branch_path=[],
    )


def _save_selected(store: SessionStore, *, view: dict, turn: int, branch: int) -> None:
    store.save_conversation(
        "alice",
        [
            {
                "role": "user",
                "content": "question",
                "metadata": {"turn_index": turn, "branch_id": branch},
            },
            {
                "role": "assistant",
                "content": "first" if branch == 1 else "second",
                "metadata": {"turn_index": turn, "branch_id": branch},
            },
        ],
    )
    store.state["alice:snapshot_event_id"] = store.max_event_id("alice")
    store.state["alice:snapshot_branch"] = {
        "turn_index": turn,
        "branch_id": branch,
        "parent_branch_path": [],
        "branch_view": view,
    }


async def _fresh_agent():
    return await Agent.build(
        AgentConfig(name="alice", system_prompt="keep"),
        llm=ScriptedLLM(["continued"]),
        io="headless",
    )


def _assistant(agent) -> list[str]:
    return [
        message.content
        for message in agent.controller.conversation.get_messages()
        if message.role == "assistant"
    ]


async def test_nonlatest_view_restores_selected_coordinates(tmp_path):
    path = tmp_path / "selected.kohakutr"
    store = SessionStore(str(path))
    try:
        _seed_two_branches(store)
        _save_selected(store, view={1: 1}, turn=1, branch=1)
        store.flush()
    finally:
        store.close(update_status=False)

    reopened = SessionStore(str(path))
    agent = await _fresh_agent()
    try:
        await inject_saved_state_async(agent, reopened, "alice")
        assert agent._branch_id == 1
        assert agent._turn_index == 1
        assert agent._branch_view == {1: 1}
        assert (1, 2) not in agent._parent_branch_path
        assert _assistant(agent) == ["first"]
        await reopened.run(agent.attach_session_store, reopened)
        await agent.start()
        assert (await agent.run("next", timeout=5)).text == "continued"
        assert (1, 1) in agent._parent_branch_path
        assert (1, 2) not in agent._parent_branch_path
        assert agent._branch_id == 1
    finally:
        await agent.stop()
        reopened.close(update_status=False)


async def test_sparse_view_expands_to_raw_ancestry(tmp_path):
    """A sparse override of a later turn keeps the ancestor the raw path names."""
    store = SessionStore(str(tmp_path / "sparse.kohakutr"))
    try:
        store.append_event(
            "alice",
            "user_message",
            {"content": "one"},
            turn_index=1,
            branch_id=1,
            parent_branch_path=[],
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "a1"},
            turn_index=1,
            branch_id=1,
            parent_branch_path=[],
        )
        store.append_event(
            "alice",
            "user_message",
            {"content": "two-a"},
            turn_index=2,
            branch_id=1,
            parent_branch_path=[(1, 1)],
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "b1"},
            turn_index=2,
            branch_id=1,
            parent_branch_path=[(1, 1)],
        )
        store.append_event(
            "alice",
            "user_message",
            {"content": "two-b"},
            turn_index=2,
            branch_id=2,
            parent_branch_path=[(1, 1)],
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "b2"},
            turn_index=2,
            branch_id=2,
            parent_branch_path=[(1, 1)],
        )
        store.save_conversation(
            "alice",
            [
                {
                    "role": "user",
                    "content": "one",
                    "metadata": {"turn_index": 1, "branch_id": 1},
                },
                {
                    "role": "assistant",
                    "content": "a1",
                    "metadata": {"turn_index": 1, "branch_id": 1},
                },
                {
                    "role": "user",
                    "content": "two-a",
                    "metadata": {"turn_index": 2, "branch_id": 1},
                },
                {
                    "role": "assistant",
                    "content": "b1",
                    "metadata": {"turn_index": 2, "branch_id": 1},
                },
            ],
        )
        store.state["alice:snapshot_event_id"] = store.max_event_id("alice")
        # Sparse: only the non-latest turn is named. Ancestry fills turn 1.
        store.state["alice:snapshot_branch"] = {
            "turn_index": 2,
            "branch_id": 1,
            "parent_branch_path": [[1, 1]],
            "branch_view": {2: 1},
        }
        agent = await _fresh_agent()
        inject_saved_state(agent, store, "alice")
        assert agent._branch_view[1] == 1
        assert agent._branch_view[2] == 1
        assert agent._branch_id == 1
        assert "b2" not in _assistant(agent)
        assert "b1" in _assistant(agent)
    finally:
        store.close(update_status=False)


def test_inconsistent_view_fails_closed(tmp_path):
    store = SessionStore(str(tmp_path / "bad.kohakutr"))
    try:
        _seed_two_branches(store)
        _save_selected(store, view={1: 9}, turn=1, branch=1)
        agent_holder = {}

        async def _go():
            agent_holder["agent"] = await _fresh_agent()

        asyncio.run(_go())
        with pytest.raises(SessionError):
            inject_saved_state(agent_holder["agent"], store, "alice")
        # Failed injection must not have installed the latest sibling.
        assert getattr(agent_holder["agent"], "_branch_id", 0) != 2
    finally:
        store.close(update_status=False)


def test_view_disagreeing_with_terminal_fails_closed(tmp_path):
    store = SessionStore(str(tmp_path / "disagree.kohakutr"))
    try:
        _seed_two_branches(store)
        store.save_conversation(
            "alice",
            [
                {
                    "role": "user",
                    "content": "question",
                    "metadata": {"turn_index": 1, "branch_id": 1},
                }
            ],
        )
        store.state["alice:snapshot_event_id"] = store.max_event_id("alice")
        # View names branch 1, terminal claims branch 2.
        store.state["alice:snapshot_branch"] = {
            "turn_index": 1,
            "branch_id": 2,
            "parent_branch_path": [],
            "branch_view": {1: 1},
        }

        agent = asyncio.run(_fresh_agent())
        with pytest.raises(SessionError):
            inject_saved_state(agent, store, "alice")
    finally:
        store.close(update_status=False)


def test_legacy_snapshot_without_view_still_loads(tmp_path):
    store = SessionStore(str(tmp_path / "legacy.kohakutr"))
    try:
        store.append_event(
            "alice",
            "user_message",
            {"content": "q"},
            turn_index=1,
            branch_id=1,
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "a"},
            turn_index=1,
            branch_id=1,
        )
        store.save_conversation(
            "alice",
            [
                {
                    "role": "user",
                    "content": "q",
                    "metadata": {"turn_index": 1, "branch_id": 1},
                },
                {"role": "assistant", "content": "a"},
            ],
        )
        store.state["alice:snapshot_event_id"] = store.max_event_id("alice")
        store.state["alice:snapshot_branch"] = {
            "turn_index": 1,
            "branch_id": 1,
            "parent_branch_path": [],
        }

        agent = asyncio.run(_fresh_agent())
        inject_saved_state(agent, store, "alice")
        assert agent._branch_id == 1
        assert _assistant(agent) == ["a"]
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("has_raw_ancestor", [False, True])
def test_selected_snapshot_requires_real_ancestor(tmp_path, has_raw_ancestor):
    """Selected snapshots retain compacted context without inventing raw ancestry."""
    store = SessionStore(str(tmp_path / "compact.kohakutr"))
    try:
        if has_raw_ancestor:
            store.append_event(
                "alice",
                "user_message",
                {"content": "old"},
                turn_index=1,
                branch_id=1,
                parent_branch_path=[],
            )
        store.append_event(
            "alice",
            "user_message",
            {"content": "kept"},
            turn_index=2,
            branch_id=1,
            parent_branch_path=[(1, 1)],
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "live"},
            turn_index=2,
            branch_id=1,
            parent_branch_path=[(1, 1)],
        )
        # Sibling that latest-selection would prefer if the snapshot were dropped.
        store.append_event(
            "alice",
            "user_message",
            {"content": "other"},
            turn_index=2,
            branch_id=2,
            parent_branch_path=[(1, 1)],
        )
        store.append_event(
            "alice",
            "text_chunk",
            {"content": "sibling"},
            turn_index=2,
            branch_id=2,
            parent_branch_path=[(1, 1)],
        )
        store.save_conversation(
            "alice",
            [
                {"role": "system", "content": "COMPACTED PREFIX"},
                {
                    "role": "user",
                    "content": "kept",
                    "metadata": {"turn_index": 2, "branch_id": 1},
                },
                {
                    "role": "assistant",
                    "content": "live",
                    "metadata": {"turn_index": 2, "branch_id": 1},
                },
            ],
        )
        store.state["alice:snapshot_event_id"] = store.max_event_id("alice")
        store.state["alice:snapshot_branch"] = {
            "turn_index": 2,
            "branch_id": 1,
            "parent_branch_path": [[1, 1]],
            "branch_view": {2: 1},
        }

        agent = asyncio.run(_fresh_agent())
        if not has_raw_ancestor:
            before = list(agent.controller.conversation.get_messages())
            with pytest.raises(SessionError, match="ancestry"):
                inject_saved_state(agent, store, "alice")
            assert agent.controller.conversation.get_messages() == before
            return
        inject_saved_state(agent, store, "alice")
        contents = [
            message.content for message in agent.controller.conversation.get_messages()
        ]
        assert "COMPACTED PREFIX" in contents
        assert "sibling" not in contents
        assert agent._branch_id == 1
    finally:
        store.close(update_status=False)


def test_running_owner_rejects_injection_before_mutate(tmp_path):
    store = SessionStore(str(tmp_path / "running.kohakutr"))
    try:
        _seed_two_branches(store)
        _save_selected(store, view={1: 1}, turn=1, branch=1)

        agent = asyncio.run(_fresh_agent())

        class _Started:
            has_started = True

        agent._session_state_owner = _Started()
        before = list(agent.controller.conversation.get_messages())
        with pytest.raises(ConflictError):
            inject_saved_state(agent, store, "alice")
        assert list(agent.controller.conversation.get_messages()) == before
        assert getattr(agent, "_branch_id", 0) != 1 or agent._branch_view != {1: 1}
    finally:
        store.close(update_status=False)


async def test_compact_persistence_retains_selected_branch_on_reopen(tmp_path):
    path = tmp_path / "compact-selected.kohakutr"
    store = SessionStore(str(path))
    agent = await _fresh_agent()
    try:
        _seed_two_branches(store)
        _save_selected(store, view={1: 1}, turn=1, branch=1)
        inject_saved_state(agent, store, "alice")
        agent.controller.conversation.append("system", "COMPACTED SELECTED PREFIX")
        await store.run(
            persist_compacted,
            store,
            "alice",
            agent,
            agent.controller.conversation,
            1,
            123.0,
        )
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)
    reopened = SessionStore(str(path))
    resumed = await _fresh_agent()
    try:
        await inject_saved_state_async(resumed, reopened, "alice")
        assert resumed._branch_id == 1
        assert resumed._branch_view == {1: 1}
        assert "COMPACTED SELECTED PREFIX" in [
            m.content for m in resumed.controller.conversation.get_messages()
        ]
        assert "second" not in _assistant(resumed)
    finally:
        await resumed.stop()
        await asyncio.to_thread(reopened.close)


def test_snapshot_tag_uses_current_view_not_stale(tmp_path):
    store = SessionStore(str(tmp_path / "tag.kohakutr"))
    try:
        agent_ns = type("A", (), {})()
        agent_ns._turn_index = 2
        agent_ns._branch_id = 1
        agent_ns._parent_branch_path = [(1, 1)]
        # Stale view from before the continuation turn.
        agent_ns._branch_view = {1: 1}
        agent_ns.session = None
        sink = SessionOutput("alice", store, agent_ns, capture_activity=False)
        assert sink._current_branch_view() == {1: 1, 2: 1}
    finally:
        store.close(update_status=False)
