"""Background terminal durability is independent from controller consumption."""

import asyncio
import threading
from concurrent.futures import Future
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("error", [None, "failed", "cancelled"])
async def test_duplicate_callback_records_one_origin_terminal(tmp_path, child, error):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        job_id = "agent_child_one" if child else "read_one"
        context = capture_tool_context(agent, job_id, "background", "provider")
        if error is not None:
            context.update(error=error, cancelled=error == "cancelled")
        event = create_tool_complete_event(job_id=job_id, content="result", **context)
        agent._bg_controller_notify[job_id] = False
        agent._on_bg_complete(event)
        agent._on_bg_complete(deepcopy(event))
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        terminals = [
            e
            for e in raw
            if e["type"] == ("subagent_result" if child else "tool_result")
            and e.get("job_id") == job_id
        ]
        assert len(terminals) == 1
        assert terminals[0]["origin"] == context["origin"]
        assert terminals[0]["tool_call_id"] == "provider"
        assert llm.call_count == 1
        assert not agent._event_inbox


@pytest.mark.parametrize("capture", [False, True])
async def test_provider_id_reuse_does_not_merge_jobs_and_capture_mode_is_explicit(
    tmp_path, capture
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        agent._session_output._capture_activity = capture
        for job_id in ("read_one", "read_two"):
            event = create_tool_complete_event(
                job_id=job_id,
                content=job_id,
                **capture_tool_context(agent, job_id, "background", "reused-provider"),
            )
            agent._bg_controller_notify[job_id] = False
            agent._on_bg_complete(event)
            await agent.output_router.flush()
            event.content = "duplicate changed result"
            agent._on_bg_complete(event)
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["output"] for e in raw if e["type"] == "tool_result"] == (
            ["read_one", "read_two"] if capture else []
        )
        assert len(agent._background_terminals) == 2
        assert all(
            entry["state"] == "published" and entry["persisted"] is capture
            for entry in agent._background_terminals.values()
        )
        assert not agent._event_inbox


@pytest.mark.parametrize("after_append", [False, True])
@pytest.mark.parametrize("child", [False, True])
async def test_failed_receipt_retry_reconciles_first_outcome(
    tmp_path, monkeypatch, after_append, child
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        job_id = "agent_child_retry" if child else "read_one"
        event_type = "subagent_result" if child else "tool_result"
        event = create_tool_complete_event(
            job_id=job_id,
            content="first outcome",
            **capture_tool_context(agent, job_id, "background", "provider"),
        )
        if child:
            agent.output_router.notify_activity(
                "subagent_start",
                "[child] start",
                {
                    "job_id": job_id,
                    "subagent": "child",
                    "model": "child-model",
                    "llm_name": "child-profile",
                    "task": "work",
                    **event.context,
                },
            )
            await agent.output_router.flush()
        agent._bg_controller_notify[event.job_id] = False
        store = agent.session_store
        append = type(store).append_event
        failed = False

        def fail_once(self, namespace, kind, data, **kwargs):
            nonlocal failed
            if kind == event_type and not failed:
                failed = True
                if after_append:
                    append(self, namespace, kind, data, **kwargs)
                raise OSError("terminal receipt failed")
            return append(self, namespace, kind, data, **kwargs)

        monkeypatch.setattr(type(store), "append_event", fail_once)
        agent._on_bg_complete(event)
        await agent.output_router.flush()
        entry = agent._background_terminals[event.context["origin"]["delivery_id"]]
        assert entry["state"] == "uncertain" and not entry["persisted"]
        assert not agent._event_inbox
        retry = deepcopy(event)
        retry.content = "changed retry must not replace first outcome"
        agent._on_bg_complete(retry)
        await agent.output_router.flush()
        assert entry["state"] == "published" and entry["persisted"]
        raw = await store.run(store.get_events, "alice")
        terminals = [e for e in raw if e["type"] == event_type]
        assert [e["output"] for e in terminals] == ["first outcome"]


async def test_cancelled_write_receipt_is_uncertain_not_auto_republished(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        event = create_tool_complete_event(
            job_id="read_one",
            content="once",
            **capture_tool_context(agent, "read_one", "background", "provider"),
        )
        agent._bg_controller_notify[event.job_id] = False
        submit = agent._session_output._submit_store_write
        cancelled_once = False

        def cancel_receipt(fn, *args, **kwargs):
            nonlocal cancelled_once
            receipt = submit(fn, *args, **kwargs)
            if (
                getattr(fn, "__name__", "") == "_persist_terminal"
                and not cancelled_once
            ):
                cancelled_once = True
                lost = Future()
                lost.cancel()
                return lost
            return receipt

        monkeypatch.setattr(
            agent._session_output, "_submit_store_write", cancel_receipt
        )
        agent._on_bg_complete(event)
        await asyncio.wait_for(agent.output_router.flush(), 3)
        entry = agent._background_terminals[event.context["origin"]["delivery_id"]]
        assert entry["state"] == "uncertain" and not entry["persisted"]
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert len([e for e in raw if e["type"] == "tool_result"]) == 1
        agent._on_bg_complete(event)
        await agent.output_router.flush()
        assert entry["state"] == "published" and entry["persisted"]
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert len([e for e in raw if e["type"] == "tool_result"]) == 1


@pytest.mark.parametrize("cancel_before_start", [False, True])
async def test_cancel_and_stop_join_accepted_terminal_write(
    tmp_path, monkeypatch, cancel_before_start
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        store = agent.session_store
        append = type(store).append_event
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def held(self, namespace, event_type, data, **kwargs):
            if event_type == "tool_result":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return append(self, namespace, event_type, data, **kwargs)

        monkeypatch.setattr(type(store), "append_event", held)
        event = create_tool_complete_event(
            job_id="read_one",
            content="detached",
            **capture_tool_context(agent, "read_one", "background", "provider"),
        )
        agent._bg_controller_notify[event.job_id] = False
        task = agent._on_bg_complete(event)
        event.content = "mutated"
        try:
            if not cancel_before_start:
                await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            stopped = asyncio.create_task(agent.stop())
            await asyncio.wait_for(entered.wait(), 3)
            assert not stopped.done()
            release.set()
            await asyncio.wait_for(stopped, 3)
            raw = await store.run(store.get_events, "alice")
            assert [e["output"] for e in raw if e["type"] == "tool_result"] == [
                "detached"
            ]
        finally:
            release.set()
