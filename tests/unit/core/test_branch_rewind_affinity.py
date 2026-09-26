"""Rewind validates retained history and persists snapshots off the owner loop."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.branch_rewind import _prepare
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import ConversationConfig
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("kind", ["user", "native"])
@pytest.mark.parametrize("evidence_case", ["aligned", "missing_anchor", "later_switch"])
def test_rewind_dropped_duplicate_does_not_ambiguate_retained_origin(
    tmp_path, kind, evidence_case
):
    store = SessionStore(str(tmp_path / "duplicate.kohakutr"))
    source = {"backend_key": "new", "backend_type": "openai", "tool_format": "native"}
    calls = [
        {
            "id": "same-id",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
    ]
    try:
        store.append_event(
            "alice",
            "model_switch",
            {
                "schema": 1,
                "classification": "portable",
                "from_source": source,
                "to_source": source,
            },
        )
        messages = []
        for turn in (1, 2):
            store.append_event(
                "alice",
                "user_message",
                {"content": "repeated"},
                turn_index=turn,
                branch_id=1,
                parent_branch_path=[(1, 1)] if turn == 2 else [],
            )
            messages.append(
                Message(
                    role="user",
                    content="repeated",
                    name="legacy" if kind == "user" else None,
                )
            )
            if kind == "native":
                store.append_event(
                    "alice",
                    "assistant_tool_calls",
                    {"tool_calls": calls},
                    turn_index=turn,
                    branch_id=1,
                )
                store.append_event(
                    "alice",
                    "tool_result",
                    {"tool_call_id": "same-id", "output": "done"},
                    turn_index=turn,
                    branch_id=1,
                )
                messages.extend(
                    [
                        Message(role="assistant", content="", tool_calls=calls),
                        Message(role="tool", content="done", tool_call_id="same-id"),
                    ]
                )
        if evidence_case == "missing_anchor":
            store.append_event(
                "alice",
                "user_message",
                {"content": "unmapped"},
                turn_index=2,
                branch_id=1,
            )
        elif evidence_case == "later_switch":
            store.append_event(
                "alice",
                "model_switch",
                {
                    "schema": 1,
                    "classification": "portable",
                    "from_source": {
                        "backend_key": "other",
                        "backend_type": "different",
                    },
                    "to_source": source,
                },
            )
        cutoff = len(messages) // 2
        if evidence_case != "aligned":
            with pytest.raises(HistoryCompatibilityError):
                _prepare(
                    store,
                    "alice",
                    tuple(messages),
                    ConversationConfig(),
                    (2, 1, ((1, 1),)),
                    source,
                    cutoff,
                )
            return
        count, snapshot = _prepare(
            store,
            "alice",
            tuple(messages),
            ConversationConfig(),
            (2, 1, ((1, 1),)),
            source,
            cutoff,
        )
        assert count == cutoff
        assert [m["role"] for m in snapshot] == [m.role for m in messages[:cutoff]]
    finally:
        store.close()


async def test_rewind_cannot_leave_unpaired_native_announcement(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question", timeout=5)
        conv = agent.controller.conversation
        conv.append(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "native",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        )
        result_index = len(conv.get_messages())
        conv.append("tool", "result", tool_call_id="native", name="read")
        before = conv.get_messages()[:]
        snapshot = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        with pytest.raises(HistoryCompatibilityError):
            await agent.rewind_to(result_index)
        assert conv.get_messages() == before
        assert (
            await agent.session_store.run(
                agent.session_store.load_conversation, "alice"
            )
            == snapshot
        )


async def test_cancelled_rewind_joins_worker_save_without_blocking_owner(
    tmp_path, monkeypatch
):
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question", timeout=5)
        store = agent.session_store
        save = store.save_conversation

        def held(*args):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return save(*args)

        monkeypatch.setattr(store, "save_conversation", held)
        cutoff = len(agent.controller.conversation.get_messages()) - 1
        task = asyncio.create_task(agent.rewind_to(cutoff))
        stopping = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert (
                await asyncio.wait_for(
                    agent._session_state_owner.submit_internal(
                        lambda: "owner responsive"
                    ),
                    1,
                )
                == "owner responsive"
            )
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            stopping = asyncio.create_task(agent.stop())
            await asyncio.sleep(0)
            assert not stopping.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            await asyncio.wait_for(stopping, 5)
            snapshot = await store.run(store.load_conversation, "alice")
            assert not any(m["role"] == "assistant" for m in snapshot)
            assert not agent._processing_lock.locked()
        finally:
            release.set()
            await asyncio.gather(
                *[t for t in (task, stopping) if t is not None], return_exceptions=True
            )


@pytest.mark.parametrize("save_fails", [False, True])
async def test_rewind_owner_commit_and_worker_save_preserve_warning_policy(
    tmp_path, monkeypatch, save_fails
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question", timeout=5)
        conv = agent.controller.conversation
        kept = conv.get_messages()[:-1]
        threads, tasks = [], []
        save = agent.session_store.save_conversation
        truncate = conv.truncate_from

        def observed_save(*args):
            threads.append(threading.get_ident())
            if save_fails:
                raise OSError("snapshot not saved")
            return save(*args)

        def observed_truncate(index):
            tasks.append(asyncio.current_task())
            return truncate(index)

        monkeypatch.setattr(agent.session_store, "save_conversation", observed_save)
        monkeypatch.setattr(conv, "truncate_from", observed_truncate)
        await agent.rewind_to(len(kept))
        assert agent.controller.conversation is conv
        assert conv.get_messages() == kept
        assert all(a is b for a, b in zip(conv.get_messages(), kept))
        assert threads == [agent.session_store._affinity_worker_ident]
        assert tasks == [agent._session_state_owner.task]
        if not save_fails:
            snapshot = await agent.session_store.run(
                agent.session_store.load_conversation, "alice"
            )
            assert [m["role"] for m in snapshot] == [m.role for m in kept]
