"""Cold snapshot recovery uses selected durable calls and terminals across watermarks."""

import asyncio

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.resume import resume_agent
from kohakuterrarium.session.resume_reconciliation import reconcile_snapshot
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


def _seed(tmp_path, kind, position, *, corrupt=False):
    config = tmp_path / "creature"
    config.mkdir()
    (config / "config.yaml").write_text(
        "name: alice\ninput: {type: none}\noutput: {type: stdout}\ntool_format: native\n",
        encoding="utf-8",
    )
    path = tmp_path / "snapshot.kohakutr"
    store = SessionStore(str(path))
    try:
        store.init_meta("snapshot", "agent", str(config), str(tmp_path), ["alice"])

        def append(event_type, payload):
            return store.append_event(
                "alice",
                event_type,
                payload,
                turn_index=1,
                branch_id=1,
                parent_branch_path=[],
            )

        append("user_message", {"content": "question"})
        identity = {
            "job_id": "job",
            "tool_call_id": "provider",
            "name": "work",
            "execution_class": "background",
            "background": True,
            "origin": {
                "runtime_scope_id": "old",
                "delivery_id": "old:job",
                "turn_index": 1,
                "branch_id": 1,
                "parent_branch_path": [],
                "origin_path": [[1, 1]],
                "execution_class": "background",
            },
        }
        if kind == "tool":
            identity["call_id"] = "job"
        append(
            f"{kind}_call",
            {**identity, "args": {}, "tool_call_arguments": "{}", "task": "work"},
        )
        before = store.max_event_id("alice")
        if position != "pending":
            append(f"{kind}_result", {**identity, "output": "durable done"})
        snapshot = [
            {"role": "system", "content": "saved system"},
            {"role": "assistant", "content": "compact summary"},
            {
                "role": "user",
                "content": "question",
                "metadata": {"turn_index": 1, "branch_id": 1},
            },
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "provider",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            },
        ]
        if corrupt:
            snapshot.append(
                {"role": "tool", "tool_call_id": "foreign", "content": "corrupt"}
            )
        store.save_conversation("alice", snapshot)
        if position != "legacy":
            store.state["alice:snapshot_event_id"] = (
                before if position == "after" else store.max_event_id("alice")
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


@pytest.mark.parametrize("kind", ["tool", "subagent"])
@pytest.mark.parametrize("position", ["before", "after", "legacy", "pending"])
async def test_pending_snapshot_closes_only_from_selected_evidence(
    tmp_path, kind, position
):
    path = _seed(tmp_path, kind, position)
    llm = ScriptedLLM(["continued"])
    agent, store = resume_agent(path, llm=llm)
    try:
        messages = agent.controller.conversation.get_messages()
        results = [m for m in messages if m.role == "tool"]
        assert len(results) == 1
        assert results[0].tool_call_id == "provider"
        assert results[0].content == (
            "Interrupted by session resume" if position == "pending" else "durable done"
        )
        assert any(m.content == "compact summary" for m in messages)
        assert (
            llm.call_count == 0
            and not agent.executor._tasks
            and not agent.subagent_manager._tasks
        )
        raw = store.get_events("alice")
        assert not any(e.get("_synthetic_resume") for e in raw)
        if position == "pending":
            assert not any(e["type"].endswith("_result") for e in raw)
        await agent.start()
        assert (await agent.run("continue", timeout=5)).text == "continued"
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("native", [False, True])
def test_wholly_post_watermark_child_tail_is_not_lost(tmp_path, native):
    path = _seed(tmp_path, "tool", "before")
    store = SessionStore(str(path))
    try:
        identity = {
            "job_id": "child-job",
            "name": "child",
            "task": "task",
            "background": False,
        }
        if native:
            identity.update(
                tool_call_id="child-provider", tool_call_arguments='{"task":"task"}'
            )
        for kind, extra in (
            ("subagent_call", {}),
            ("subagent_result", {"output": "later child"}),
        ):
            store.append_event(
                "alice",
                kind,
                {**identity, **extra},
                turn_index=1,
                branch_id=1,
                parent_branch_path=[],
            )
    finally:
        store.close()
    agent, store = resume_agent(path, llm=ScriptedLLM(["unused"]))
    try:
        messages = agent.controller.conversation.get_messages()
        assert sum("later child" in str(m.content) for m in messages) == 1
        assert any(m.content == "compact summary" for m in messages)
    finally:
        store.close()


def test_unknown_base_dispatch_after_watermark_is_not_reannounced(tmp_path):
    path = _seed(tmp_path, "tool", "pending")
    store = SessionStore(str(path))
    try:
        raw = store.get_events("alice")
        call = next(e for e in raw if e["type"] == "tool_call")
        for key in ("origin", "execution_class", "background"):
            call.pop(key)
        store.state["alice:snapshot_event_id"] = 1
        snapshot = store.load_conversation("alice")
    finally:
        store.close()
    from_messages, consumed = reconcile_snapshot(snapshot, raw, 1)
    assert from_messages == snapshot
    assert call["event_id"] in consumed


@pytest.mark.parametrize("kind", ["tool", "subagent"])
def test_corrupt_base_is_rejected_before_terminal_can_repair_it(tmp_path, kind):
    path = _seed(tmp_path, kind, "after", corrupt=True)
    with pytest.raises(HistoryCompatibilityError):
        resume_agent(path, llm=ScriptedLLM(["unused"]))
