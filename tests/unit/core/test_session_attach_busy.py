"""Explicit session replacement rejects busy agents before modifying bindings."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.session_attachment_prepare import PreparedAttachment
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import session_attach


@pytest.mark.parametrize("busy", ["controller", "background_tool"])
async def test_explicit_attach_rejects_before_changing_any_binding(tmp_path, busy):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldLLM(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if busy == "controller":
                entered.set()
                await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    class HeldTool(BaseTool):
        tool_name = "held"
        description = "Return after release"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            return ToolResult(output="finished")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=HeldLLM(
                [
                    "answer" if busy == "controller" else "launched\n[/held]\n[held/]",
                    "handled",
                ]
            ),
            tools=[HeldTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        original, sink = agent.session_store, agent._session_output
        replacement = SessionStore(tmp_path / "replacement.kohakutr")
        running = asyncio.create_task(creature.run("start"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if busy == "background_tool":
                await asyncio.wait_for(running, 3)
                assert not agent._processing_lock.locked()
                assert agent.executor.get_running_jobs()
            with pytest.raises(ConflictError):
                agent.attach_session_store(replacement)
            assert agent.session_store is original
            assert agent.controller.session_store is original
            assert agent.subagent_manager._session_store is original
            assert agent.trigger_manager._session_store is original
            assert agent._session_output is sink
            assert sink in agent.output_router._secondary_outputs
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
            if agent.session_store is not original:
                agent.attach_session_store(original)
            await asyncio.to_thread(replacement.close)


async def test_engine_busy_attach_does_not_retire_original_store(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldLLM(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=HeldLLM(["answer"]),
            io="headless",
            start=True,
        )
        original = creature.agent.session_store
        replacement = tmp_path / "must-not-mint.kohakutr"
        running = asyncio.create_task(creature.run("start", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(ConflictError):
                await engine.attach_session(creature.graph_id, replacement)
            assert not original._closed
            assert not replacement.exists()
            assert engine._session_stores[creature.graph_id] is original
            assert creature.agent.session_store is original
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize("phase", ["child", "posthook"])
async def test_explicit_attach_rejects_live_child_and_posthook(tmp_path, phase):
    entered, release = asyncio.Event(), asyncio.Event()

    class Parent(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "handled"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if not self.call_count:
                self.call_count += 1
                self.last_tool_calls = [
                    NativeToolCall(
                        "child-call", "work", '{"task":"run", "run_in_background":true}'
                    )
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    class Child(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if phase == "child":
                entered.set()
                await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    class Posthook(BasePlugin):
        name = "held_posthook"

        async def post_subagent_run(self, result, **kwargs):
            if phase == "posthook":
                entered.set()
                await release.wait()
            return result

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Parent(),
            plugins=[Posthook()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        cfg = SubAgentConfig(name="work", description="Child", tools=[], max_turns=1)
        agent.subagent_manager.register(cfg)
        agent.registry.register_subagent("work", cfg)
        agent.subagent_manager.llm = Child(["child result"])
        agent.subagent_manager._parent_plugins = agent.plugins
        original = agent.session_store
        replacement = SessionStore(tmp_path / "replacement.kohakutr")
        try:
            assert (await creature.run("start", timeout=5)).ok
            await asyncio.wait_for(entered.wait(), 3)
            assert not agent._processing_lock.locked()
            with pytest.raises(ConflictError):
                agent.attach_session_store(replacement)
            assert agent.session_store is original
            assert agent.subagent_manager._session_store is original
            assert not original._closed
        finally:
            release.set()
            await agent.subagent_manager.wait_all(timeout=3)
            await asyncio.to_thread(replacement.close)


async def test_engine_attach_reserves_turn_admission_across_drive_await(
    tmp_path, monkeypatch, caplog
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["after attach"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        original = agent.session_store
        detach = engine._drive_runtime.detach_graph

        async def held(graph_id):
            entered.set()
            await release.wait()
            await detach(graph_id)

        monkeypatch.setattr(engine._drive_runtime, "detach_graph", held)
        attaching = asyncio.create_task(
            engine.attach_session(creature.graph_id, tmp_path / "replacement.kohakutr")
        )
        running = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert not original._closed
            running = asyncio.create_task(creature.run("queued input"))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(running), 0.05)
            assert llm.call_count == 0
            with pytest.raises(ConflictError):
                agent.update_system_prompt("must reject")
            release.set()
            await asyncio.wait_for(attaching, 5)
            assert (await asyncio.wait_for(running, 5)).text == "after attach"
            assert agent.session_store is not original
            assert not agent._session_attach_pending
            assert not any(
                "option" in record.message and "skipped" in record.message
                for record in caplog.records
            )
            records = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert (
                sum(
                    e.get("type") == "user_message"
                    and e.get("content") == "queued input"
                    for e in records
                )
                == 1
            )
        finally:
            release.set()
            await asyncio.gather(
                *[t for t in (attaching, running) if t is not None],
                return_exceptions=True,
            )


async def test_cancel_accepted_attach_joins_before_releasing_members(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        original = agent.session_store
        bind = engine._drive_runtime.bind_graph_store

        async def held(graph_id, store):
            entered.set()
            await release.wait()
            await bind(graph_id, store)

        monkeypatch.setattr(engine._drive_runtime, "bind_graph_store", held)
        attaching = asyncio.create_task(
            engine.attach_session(creature.graph_id, tmp_path / "replacement.kohakutr")
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            attaching.cancel()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(attaching), 0.05)
            assert agent._processing_lock.locked()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await attaching
            assert agent.session_store is engine._session_stores[creature.graph_id]
            assert (
                agent.session_store is not original and not agent.session_store._closed
            )
            assert not agent._session_attach_pending
            assert (
                await creature.run("after cancellation", timeout=5)
            ).text == "answer"
        finally:
            release.set()
            await asyncio.gather(attaching, return_exceptions=True)


async def test_add_creature_cannot_join_graph_during_attachment(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        original = alice.agent.session_store
        detach = engine._drive_runtime.detach_graph

        async def held(gid):
            entered.set()
            await release.wait()
            await detach(gid)

        monkeypatch.setattr(engine._drive_runtime, "detach_graph", held)
        attaching = asyncio.create_task(
            engine.attach_session(alice.graph_id, tmp_path / "replacement.kohakutr")
        )
        before = set(engine._creatures)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(ConflictError):
                await engine.add_creature(
                    AgentConfig(name="late"),
                    graph=alice.graph_id,
                    llm=ScriptedLLM(["late"]),
                    io="headless",
                    start=True,
                )
            assert set(engine._creatures) == before
            assert not original._closed
        finally:
            release.set()
            await asyncio.gather(attaching, return_exceptions=True)


@pytest.mark.parametrize("failure", ["drive", "checkpoint", "member"])
async def test_failed_drive_binding_keeps_original_session_usable(
    tmp_path, monkeypatch, failure
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["still usable"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        original = agent.session_store
        bind = engine._drive_runtime.bind_graph_store

        async def fail_new(gid, store):
            if store is not original:
                raise RuntimeError("new Drive binding failed")
            await bind(gid, store)

        if failure == "drive":
            monkeypatch.setattr(engine._drive_runtime, "bind_graph_store", fail_new)
        elif failure == "checkpoint":

            async def fail_checkpoint(*args):
                raise RuntimeError("new Drive binding failed")

            monkeypatch.setattr(
                session_attach.checkpoint, "checkpoint", fail_checkpoint
            )
        else:
            original_install = PreparedAttachment.install

            def fail_member(prepared):
                original_install(prepared)
                if prepared.agent is agent and prepared.store is not original:
                    raise RuntimeError("new Drive binding failed")

            monkeypatch.setattr(PreparedAttachment, "install", fail_member)
        with pytest.raises(RuntimeError, match="new Drive binding failed"):
            await engine.attach_session(
                creature.graph_id, tmp_path / "replacement.kohakutr"
            )
        assert not original._closed
        assert engine._session_stores[creature.graph_id] is original
        assert agent.session_store is original
        assert agent._session_output._store is original
        assert (
            sum(
                getattr(s, "_store", None) is original
                for s in agent.output_router._secondary_outputs
            )
            == 1
        )
        assert not agent._session_attach_pending
        assert (await creature.run("continue", timeout=5)).text == "still usable"


async def test_failed_first_attach_restores_unpersisted_agent(tmp_path, monkeypatch):
    minted = []
    mint = session_attach.autosession.mint_store

    def capture(*args, **kwargs):
        store = mint(*args, **kwargs)
        minted.append(store)
        return store

    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["still unpersisted"]),
            session=False,
            io="headless",
            start=True,
        )
        agent = creature.agent
        monkeypatch.setattr(session_attach.autosession, "mint_store", capture)

        async def fail(*args):
            raise RuntimeError("checkpoint failed")

        with monkeypatch.context() as patch:
            patch.setattr(session_attach.checkpoint, "checkpoint", fail)
            try:
                with pytest.raises(RuntimeError, match="checkpoint failed"):
                    await engine.attach_session(
                        creature.graph_id, tmp_path / "first.kohakutr"
                    )
                assert agent.session_store is None
                assert agent.controller.session_store is None
                assert agent._session_output is None
                assert not getattr(agent, "_session_attach_uncertain", False)
                assert minted[0]._closed
            finally:
                if minted and not minted[0]._closed:
                    minted[0].close()
        assert (await creature.run("continue", timeout=5)).text == "still unpersisted"


@pytest.mark.parametrize("kind", ["creature", "recipe"])
async def test_autosession_minted_store_is_closed_on_attach_rejection(
    tmp_path, monkeypatch, kind
):
    minted = []
    mint = session_attach.autosession.mint_store

    def capture(*args, **kwargs):
        store = mint(*args, **kwargs)
        minted.append(store)
        return store

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            session=False,
            io="headless",
            start=True,
        )
        gid = creature.graph_id

        async def refuse(*args, **kwargs):
            raise ConflictError("attachment refused")

        monkeypatch.setattr(session_attach.autosession, "mint_store", capture)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "attach_session", refuse)
            try:
                with pytest.raises(ConflictError, match="attachment refused"):
                    if kind == "creature":
                        await session_attach.autosession.attach_for_new_creature(
                            engine,
                            creature,
                            config=AgentConfig(name="alice"),
                            session=True,
                        )
                    else:
                        await session_attach.autosession.attach_for_recipe(
                            engine, gid, recipe="recipe.yaml", session=True
                        )
                assert minted[0]._closed
                assert gid not in engine._owned_sessions
                assert gid not in engine._session_stores
            finally:
                if minted and not minted[0]._closed:
                    minted[0].close()
