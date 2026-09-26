"""Closed-store cold recovery keeps selected child evidence without execution."""

import asyncio

import pytest

from kohakuterrarium.session.resume import resume_agent
from kohakuterrarium.session.resume_replay import replay_resume_history
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


def _seed(
    tmp_path,
    native,
    *,
    damage=None,
    compact=False,
    sibling=False,
    background=False,
    continuation=False,
):
    config = tmp_path / "creature"
    config.mkdir()
    (config / "config.yaml").write_text(
        "name: alice\ninput: {type: none}\noutput: {type: stdout}\ntool_format: "
        + ("native" if native else "bracket")
        + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "cold.kohakutr"
    store = SessionStore(str(path))
    try:
        store.init_meta("cold", "agent", str(config), str(tmp_path), ["alice"])

        def append(kind, payload, branch=1):
            store.append_event(
                "alice",
                kind,
                payload,
                turn_index=1,
                branch_id=branch,
                parent_branch_path=[],
            )

        append("system_prompt_set", {"content": "saved system"})
        append("user_message", {"content": "question"})
        call = {
            "job_id": "agent_child_one",
            "name": "child",
            "task": "work",
            "background": background,
        }
        if background:
            call.update(
                execution_class="background",
                origin={
                    "runtime_scope_id": "old-runtime",
                    "delivery_id": "old-runtime:agent_child_one",
                    "turn_index": 1,
                    "branch_id": 1,
                    "parent_branch_path": [],
                    "origin_path": [[1, 1]],
                    "execution_class": "background",
                },
            )
        if native:
            call.update(
                tool_call_id="provider-child",
                tool_call_arguments='{ "task": "work", "run_in_background": false }',
            )
        if native and background:
            call["tool_call_arguments"] = '{"task":"work","run_in_background":true}'
        result = {**call, "output": "child done", "success": True}
        if damage == "missing_args":
            call.pop("tool_call_arguments")
        if damage == "foreign":
            result["tool_call_id"] = "foreign"
        if damage in ("synthetic", "synthetic_int"):
            result["_synthetic_resume"] = True if damage == "synthetic" else 1
        append("subagent_call", call)
        if damage != "pending":
            append("subagent_result", result)
        if damage == "duplicate":
            append("subagent_result", result)
        if damage == "duplicate_provider":
            append("subagent_call", {**call, "job_id": "agent_child_two"})
            append("subagent_result", {**result, "job_id": "agent_child_two"})
        if damage != "pending":
            append("text_chunk", {"content": "summary"})
        if continuation:
            store.append_event(
                "alice",
                "user_message",
                {"content": "later question"},
                turn_index=2,
                branch_id=1,
                parent_branch_path=[(1, 1)],
            )
            store.append_event(
                "alice",
                "text_chunk",
                {"content": "later answer"},
                turn_index=2,
                branch_id=1,
                parent_branch_path=[(1, 1)],
            )
        if compact:
            store.append_event(
                "alice",
                "compact_replace",
                {
                    "replaced_from_event_id": 2,
                    "replaced_to_event_id": store.max_event_id("alice"),
                    "summary": "saved compact summary",
                },
                turn_index=1,
                branch_id=1,
                parent_branch_path=[],
            )
        if sibling:
            append("user_message", {"content": "sibling question"}, branch=2)
            append(
                "subagent_call",
                {
                    "job_id": "foreign",
                    "name": "foreign",
                    "tool_call_id": "bad",
                    "task": "missing native args",
                },
                branch=2,
            )
            store.state["alice:snapshot_branch"] = {
                "turn_index": 1,
                "branch_id": 1,
                "parent_branch_path": [],
                "branch_view": {1: 1},
            }
        store.flush()
    finally:
        store.close()
    return path


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("sibling", [False, True])
async def test_no_snapshot_cold_child_history_survives_continuation(
    tmp_path, native, sibling
):
    path = _seed(tmp_path, native, sibling=sibling)
    llm = ScriptedLLM(["continued"])
    agent, store = resume_agent(path, llm=llm)
    try:
        messages = agent.controller.conversation.get_messages()
        assert llm.call_count == 0
        assert "sibling question" not in agent.controller.conversation.to_json()
        assert not agent.subagent_manager._tasks
        assert any(m.content == "saved system" for m in messages)
        if native:
            calls = [call for m in messages for call in (m.tool_calls or [])]
            assert len(calls) == 1
            assert calls[0]["id"] == "provider-child"
            assert (
                calls[0]["function"]["arguments"]
                == '{ "task": "work", "run_in_background": false }'
            )
            assert any(
                m.role == "tool"
                and m.tool_call_id == "provider-child"
                and m.content == "child done"
                for m in messages
            )
        else:
            assert any(
                m.metadata.get("kind") == "tool_results" and "child done" in m.content
                for m in messages
            )
            assert not any(m.tool_calls for m in messages)
        await agent.start()
        assert (await agent.run("next", timeout=5)).text == "continued"
        assert llm.call_count == 1
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("native", [False, True])
def test_compacted_completed_child_does_not_resurrect_hidden_pair(tmp_path, native):
    path = _seed(tmp_path, native, compact=True)
    agent, store = resume_agent(path, llm=ScriptedLLM(["unused"]))
    try:
        messages = agent.controller.conversation.get_messages()
        assert any(m.content == "saved compact summary" for m in messages)
        assert any(m.content == "saved system" for m in messages)
        assert not any(m.tool_calls or m.role == "tool" for m in messages)
        assert "child done" not in agent.controller.conversation.to_json()
    finally:
        store.close()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_args",
        "foreign",
        "duplicate",
        "duplicate_provider",
        "synthetic",
        "synthetic_int",
    ],
)
@pytest.mark.parametrize("compact", [False, True])
def test_bad_native_child_cold_replay_rejects_and_releases_store(
    tmp_path, damage, compact
):
    path = _seed(tmp_path, True, damage=damage, compact=compact)
    llm = ScriptedLLM(["must not run"])
    with pytest.raises(ValueError):
        resume_agent(path, llm=llm)
    assert llm.call_count == 0
    store = SessionStore(str(path))
    try:
        assert store.get_events("alice")
    finally:
        store.close()


@pytest.mark.parametrize(
    "damage",
    [
        "origin",
        "class",
        "flag",
        "delivery",
        "coordinates",
        "scope",
        "boolean",
        "row_boolean",
    ],
)
def test_unproved_pending_child_never_gets_interruption(tmp_path, damage):
    path = _seed(tmp_path, True, damage="pending", background=True)
    store = SessionStore(str(path))
    try:
        events = store.get_events("alice")
    finally:
        store.close()
    call = next(e for e in events if e["type"] == "subagent_call")
    if damage == "origin":
        call.pop("origin")
    elif damage == "class":
        call["execution_class"] = "direct"
    elif damage == "flag":
        call.pop("background")
    elif damage == "delivery":
        call["origin"]["delivery_id"] = "old-runtime:other"
    elif damage == "coordinates":
        call["origin"]["branch_id"] = 2
    elif damage == "scope":
        call["origin"]["runtime_scope_id"] = "foreign"
    elif damage == "row_boolean":
        call["turn_index"] = True
    else:
        call["origin"]["turn_index"] = True
    messages = replay_resume_history(events)
    assert not any(m["role"] == "tool" for m in messages)
    assert any(m.get("tool_calls") for m in messages)


@pytest.mark.parametrize("native", [False, True])
def test_proven_background_child_interruption_is_ephemeral_and_idempotent(
    tmp_path, native
):
    path = _seed(tmp_path, native, damage="pending", background=True, continuation=True)
    for _ in range(2):
        llm = ScriptedLLM(["unused"])
        agent, store = resume_agent(path, llm=llm)
        try:
            messages = agent.controller.conversation.get_messages()
            feedback = [
                m for m in messages if "Interrupted by session resume" in str(m.content)
            ]
            assert len(feedback) == 1
            if native:
                assert (
                    feedback[0].role == "tool"
                    and feedback[0].tool_call_id == "provider-child"
                )
            else:
                assert feedback[0].metadata.get("kind") == "tool_results"
            assert any(m.content == "later question" for m in messages)
            assert llm.call_count == 0 and not agent.subagent_manager._tasks
            assert not any(
                e["type"] == "subagent_result" for e in store.get_events("alice")
            )
        finally:
            store.close()


def test_pending_native_child_keeps_actual_announcement_without_synthetic_terminal(
    tmp_path,
):
    path = _seed(tmp_path, True, damage="pending")
    agent, store = resume_agent(path, llm=ScriptedLLM(["unused"]))
    try:
        messages = agent.controller.conversation.get_messages()
        calls = [call for m in messages for call in (m.tool_calls or [])]
        assert [call["id"] for call in calls] == ["provider-child"]
        assert not any(m.role == "tool" for m in messages)
        assert not any(
            e["type"] == "subagent_result" for e in store.get_events("alice")
        )
    finally:
        store.close()
