"""Unit tests for :mod:`kohakuterrarium.terrarium.runtime_prompt`."""

import asyncio
from types import SimpleNamespace


import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import runtime_prompt as rp
from kohakuterrarium.terrarium.events import EngineEvent, EventKind
from kohakuterrarium.testing.terrarium import TestTerrariumBuilder

# ── build_runtime_graph_section ───────────────────────────────


def _msg(content):
    return SimpleNamespace(content=content)


class _MockConversation:
    def __init__(self, content):
        self._msg = _msg(content)

    def get_system_message(self):
        return self._msg


class _MockController:
    def __init__(self, content):
        self.conversation = _MockConversation(content)


@pytest.mark.parametrize("entry", ["explicit", "debounce"])
async def test_live_refresh_splices_only_on_owner(tmp_path, monkeypatch, entry):
    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            is_privileged=True,
            start=True,
        )
        prompt = rp.RuntimeGraphPrompt(engine)
        tasks = []
        apply = rp.apply_managed_section

        def observed(agent, content):
            tasks.append(asyncio.current_task())
            return apply(agent, content)

        monkeypatch.setattr(rp, "apply_managed_section", observed)
        if entry == "explicit":
            await prompt.refresh_creature(creature)
        else:
            prompt._do_refresh(creature.creature_id)
            await creature.agent._submit_session_state(lambda: None)
        assert tasks == [creature.agent._session_state_owner.task]
        assert rp._BEGIN in creature.agent.get_system_prompt()


async def test_off_owner_direct_apply_rejects_live_mutation():
    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        before = creature.agent.get_system_prompt()
        with pytest.raises(ConflictError):
            rp.apply_managed_section(creature.agent, "not admitted")
        assert creature.agent.get_system_prompt() == before


@pytest.mark.parametrize("action", ["cancel", "close", "rename", "remove"])
async def test_accepted_refresh_joins_and_uses_current_graph(monkeypatch, action):
    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            is_privileged=True,
            start=True,
        )
        owner = creature.agent._session_state_owner
        await owner.submit(lambda: None)
        prompt = rp.RuntimeGraphPrompt(engine)
        before = creature.agent.get_system_prompt()
        admitted = asyncio.Event()
        submit, wake = owner.submit, owner._wake
        receipts = []

        def record(callback):
            receipt = submit(callback)
            receipts.append(receipt)
            admitted.set()
            return receipt

        monkeypatch.setattr(owner, "_wake", lambda: None)
        monkeypatch.setattr(owner, "submit", record)
        waiting = asyncio.create_task(prompt.refresh_creature(creature))
        try:
            await asyncio.wait_for(admitted.wait(), 3)
            if action == "cancel":
                waiting.cancel()
            elif action == "close":
                owner.close_admission()
            elif action == "rename":
                creature.name = "renamed"
            else:
                engine._creatures.pop(creature.creature_id)
            assert creature.agent.get_system_prompt() == before
            wake()
            if action == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await waiting
            else:
                await waiting
            assert receipts[0].done() and not receipts[0].cancelled()
            current = creature.agent.get_system_prompt()
            if action == "remove":
                assert current == before
            else:
                assert rp._BEGIN in current
                if action == "rename":
                    assert "renamed" in current
        finally:
            engine._creatures[creature.creature_id] = creature
            monkeypatch.setattr(owner, "_wake", wake)
            wake()
            await asyncio.gather(waiting, return_exceptions=True)


async def test_refresh_rejects_closed_owner_then_works_after_restart():
    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            is_privileged=True,
            start=True,
        )
        prompt = rp.RuntimeGraphPrompt(engine)
        await engine.stop(creature)
        before = creature.agent.get_system_prompt()
        with pytest.raises(InvalidRequestError):
            await prompt.refresh_creature(creature)
        assert creature.agent.get_system_prompt() == before
        await engine.start(creature)
        await prompt.refresh_creature(creature)
        assert rp._BEGIN in creature.agent.get_system_prompt()


@pytest.mark.parametrize("entry", ["explicit", "debounce"])
async def test_refresh_error_is_observed_without_partial_prompt(
    monkeypatch, caplog, entry
):
    async with Terrarium() as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            is_privileged=True,
            start=True,
        )
        prompt = rp.RuntimeGraphPrompt(engine)
        before = creature.agent.get_system_prompt()

        def broken_builder(*args):
            raise ValueError("prompt build failed")

        monkeypatch.setattr(rp, "build_runtime_graph_section", broken_builder)
        if entry == "explicit":
            with pytest.raises(ValueError, match="prompt build failed"):
                await prompt.refresh_creature(creature)
        else:
            observed = asyncio.Event()
            observe = prompt._observe_refresh

            def record(receipt):
                observe(receipt)
                observed.set()

            monkeypatch.setattr(prompt, "_observe_refresh", record)
            prompt._do_refresh(creature.creature_id)
            await asyncio.wait_for(observed.wait(), 3)
            assert any(
                r.getMessage() == "runtime-graph refresh failed" for r in caplog.records
            )
        assert creature.agent.get_system_prompt() == before


class TestStripExistingBlock:
    def test_no_block(self):
        assert rp._strip_existing_block("hi") == "hi"

    def test_strips_block(self):
        text = "You are X.\n\n" f"{rp._BEGIN}\nGRAPH\n{rp._END}\n\nMore content."
        out = rp._strip_existing_block(text)
        assert rp._BEGIN not in out
        assert rp._END not in out
        assert "You are X." in out
        assert "More content." in out

    def test_unclosed_block_returns_as_is(self):
        text = f"You are X.{rp._BEGIN} hanging"
        out = rp._strip_existing_block(text)
        assert out == text


class TestApplyManagedSection:
    def test_no_controller_silent(self):
        agent = SimpleNamespace(controller=None)
        rp.apply_managed_section(agent, "block")

    def test_no_conversation_silent(self):
        agent = SimpleNamespace(controller=SimpleNamespace(conversation=None))
        rp.apply_managed_section(agent, "block")

    def test_no_get_system_silent(self):
        agent = SimpleNamespace(
            controller=SimpleNamespace(conversation=SimpleNamespace())
        )
        rp.apply_managed_section(agent, "block")

    def test_sys_msg_is_none_silent(self):
        agent = SimpleNamespace(
            controller=SimpleNamespace(
                conversation=SimpleNamespace(get_system_message=lambda: None)
            )
        )
        rp.apply_managed_section(agent, "block")

    def test_sys_msg_non_string_silent(self):
        agent = SimpleNamespace(
            controller=SimpleNamespace(
                conversation=SimpleNamespace(
                    get_system_message=lambda: SimpleNamespace(content=[1, 2])
                )
            )
        )
        rp.apply_managed_section(agent, "block")

    def test_inserts_block(self):
        agent = SimpleNamespace(controller=_MockController("You are X."))
        rp.apply_managed_section(agent, "GRAPH")
        msg = agent.controller.conversation.get_system_message()
        assert rp._BEGIN in msg.content
        assert "GRAPH" in msg.content

    def test_replaces_existing_block(self):
        original = "You are X.\n\n" f"{rp._BEGIN}\nOLD\n{rp._END}"
        agent = SimpleNamespace(controller=_MockController(original))
        rp.apply_managed_section(agent, "NEW")
        msg = agent.controller.conversation.get_system_message()
        assert "OLD" not in msg.content
        assert "NEW" in msg.content

    def test_empty_block_strips_existing(self):
        original = "You are X.\n\n" f"{rp._BEGIN}\nOLD\n{rp._END}"
        agent = SimpleNamespace(controller=_MockController(original))
        rp.apply_managed_section(agent, "")
        msg = agent.controller.conversation.get_system_message()
        assert rp._BEGIN not in msg.content
        assert "OLD" not in msg.content


# ── build_runtime_graph_section ───────────────────────────────


class TestBuildRuntimeGraphSection:
    async def test_unknown_graph_returns_empty(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            c = t.get_creature("alice")
            c.graph_id = "ghost"
            assert rp.build_runtime_graph_section(t, c) == ""
        finally:
            await t.shutdown()

    async def test_empty_creature_returns_empty(self):
        # Non-privileged creature with no channels/wiring → empty section.
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            c = t.get_creature("alice")
            out = rp.build_runtime_graph_section(t, c)
            # Solo creature with no listens/sends and not privileged → empty.
            assert out == ""
        finally:
            await t.shutdown()

    async def test_with_listen_and_send(self):
        t = await (
            TestTerrariumBuilder()
            .with_creature("alice")
            .with_creature("bob")
            .with_channel("chat")
            .with_connection("alice", "bob", channel="chat")
            .build()
        )
        try:
            alice = t.get_creature("alice")
            out = rp.build_runtime_graph_section(t, alice)
            assert "## Working with the group" in out
            assert "chat" in out
            assert "send_channel" in out
        finally:
            await t.shutdown()

    async def test_privileged_includes_suffix(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            c = t.get_creature("alice")
            c.is_privileged = True
            out = rp.build_runtime_graph_section(t, c)
            assert "privileged" in out
        finally:
            await t.shutdown()

    async def test_foreign_graph_wiring_is_not_rendered(self):
        t = await (
            TestTerrariumBuilder()
            .with_creature("alice")
            .with_creature("local-worker")
            .build()
        )
        try:
            alice = t.get_creature("alice")
            foreign = t.get_creature("local-worker")
            foreign.creature_id = "foreign-id"
            foreign.name = "worker"
            foreign.graph_id = "foreign-graph"
            t._creatures.pop("local-worker")
            t._creatures[foreign.creature_id] = foreign
            t._topology.graphs[alice.graph_id].creature_ids.discard("local-worker")
            from kohakuterrarium.terrarium.topology import GraphTopology

            t._topology.graphs[foreign.graph_id] = GraphTopology(
                graph_id=foreign.graph_id,
                creature_ids={foreign.creature_id},
            )
            t._topology.creature_to_graph.pop("local-worker", None)
            t._topology.creature_to_graph[foreign.creature_id] = foreign.graph_id
            foreign.agent.config.output_wiring = [{"to": alice.name}]
            alice.agent.config.output_wiring = [{"to": foreign.creature_id}]

            out = rp.build_runtime_graph_section(t, alice)

            assert "foreign-id" not in out
            assert "worker" not in out
        finally:
            await t.shutdown()

    async def test_privileged_with_spawned_child(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            alice = t.get_creature("alice")
            alice.is_privileged = True
            # Manually add a fake spawned creature in a different graph.
            from kohakuterrarium.terrarium.creature_host import Creature
            from kohakuterrarium.testing.terrarium import _FakeAgent

            child = Creature(
                creature_id="child",
                name="kid",
                agent=_FakeAgent(name="kid"),
            )
            child.parent_creature_id = "alice"
            child.graph_id = "g-other"
            child.config = SimpleNamespace(name="recipe-name")
            t._creatures["child"] = child
            out = rp.build_runtime_graph_section(t, alice)
            assert "Spawned" in out
            assert "kid" in out
        finally:
            await t.shutdown()


# ── RuntimeGraphPrompt ────────────────────────────────────────


class TestRuntimeGraphPrompt:
    async def test_attach_idempotent(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            prompt.attach()
            prompt.attach()  # idempotent
            assert prompt._task is not None
            prompt.detach()
        finally:
            await t.shutdown()

    async def test_attach_no_running_loop_silent(self):
        # No loop → attach is a no-op.
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            # Simulate no loop by patching get_running_loop.
            import kohakuterrarium.terrarium.runtime_prompt as rp_mod

            real = asyncio.get_running_loop

            def _boom():
                raise RuntimeError("no loop")

            rp_mod.asyncio.get_running_loop = _boom  # type: ignore
            try:
                prompt.attach()
                assert prompt._task is None
                assert prompt._attached is False
            finally:
                rp_mod.asyncio.get_running_loop = real
        finally:
            await t.shutdown()

    async def test_detach_cancels_pending(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            prompt.attach()
            prompt._schedule_refresh("alice")
            assert prompt._pending
            prompt.detach()
            assert prompt._pending == {}
        finally:
            await t.shutdown()

    async def test_schedule_refresh_replaces_pending(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            prompt._schedule_refresh("alice")
            first = prompt._pending["alice"]
            prompt._schedule_refresh("alice")
            second = prompt._pending["alice"]
            assert first is not second
        finally:
            await t.shutdown()

    async def test_do_refresh_unknown_creature_silent(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            prompt._do_refresh("ghost")  # no error
        finally:
            await t.shutdown()

    async def test_refresh_creature_immediate(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            await prompt.refresh_creature(t.get_creature("alice"))
        finally:
            await t.shutdown()


class TestOutputWiringInSection:
    async def test_inbound_output_wires(self):

        t = await (
            TestTerrariumBuilder().with_creature("alice").with_creature("bob").build()
        )
        try:
            bob = t.get_creature("bob")
            # Stamp bob's agent.config with an output_wiring that targets alice.
            bob.agent.config = SimpleNamespace(
                name="bob",
                output_wiring=[SimpleNamespace(to="alice")],
            )
            alice = t.get_creature("alice")
            out = rp.build_runtime_graph_section(t, alice)
            assert "inbound from" in out
            assert "bob" in out
        finally:
            await t.shutdown()

    async def test_outbound_output_wires(self):
        t = await (
            TestTerrariumBuilder().with_creature("alice").with_creature("bob").build()
        )
        try:
            alice = t.get_creature("alice")
            bob = t.get_creature("bob")
            alice.agent.config = SimpleNamespace(
                name="alice",
                output_wiring=[SimpleNamespace(to=bob.creature_id)],
            )
            out = rp.build_runtime_graph_section(t, alice)
            assert "outbound to" in out
            assert bob.creature_id in out
        finally:
            await t.shutdown()


class TestRunLoopBehavior:
    async def test_run_loop_skips_irrelevant_events(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            # An event whose kind is not a refresh-trigger must not
            # schedule any creature for a prompt refresh.
            prompt._schedule_refresh_for_event(
                EngineEvent(kind=EventKind.CHANNEL_MESSAGE)
            )
            assert prompt._pending == {}
        finally:
            await t.shutdown()


class TestScheduleRefreshNoLoop:
    def test_no_running_loop_skips(self):
        engine = SimpleNamespace(_creatures={})
        prompt = rp.RuntimeGraphPrompt(engine)
        # No loop in this sync test → _schedule_refresh is a no-op.
        prompt._schedule_refresh("alice")
        assert prompt._pending == {}


class TestScheduleRefreshForEvent:
    async def test_creature_id_targets(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            ev = EngineEvent(
                kind=EventKind.CREATURE_STARTED,
                creature_id="alice",
            )
            prompt._schedule_refresh_for_event(ev)
            assert "alice" in prompt._pending
            prompt.detach()
        finally:
            await t.shutdown()

    async def test_graph_id_pulls_members(self):
        t = await (
            TestTerrariumBuilder().with_creature("alice").with_creature("bob").build()
        )
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            graph = t.list_graphs()[0]
            ev = EngineEvent(
                kind=EventKind.CREATURE_STARTED,
                graph_id=graph.graph_id,
            )
            prompt._schedule_refresh_for_event(ev)
            # Both creatures should be pending.
            assert "alice" in prompt._pending
            assert "bob" in prompt._pending
            prompt.detach()
        finally:
            await t.shutdown()

    async def test_topology_changed_pulls_old_new_affected(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            graph = t.list_graphs()[0]
            ev = EngineEvent(
                kind=EventKind.TOPOLOGY_CHANGED,
                payload={
                    "old_graph_ids": [graph.graph_id],
                    "new_graph_ids": [],
                    "affected": ["alice"],
                },
            )
            prompt._schedule_refresh_for_event(ev)
            assert "alice" in prompt._pending
            prompt.detach()
        finally:
            await t.shutdown()

    async def test_parent_link_changed_pulls_parent(self):
        t = await TestTerrariumBuilder().with_creature("alice").build()
        try:
            prompt = rp.RuntimeGraphPrompt(t)
            ev = EngineEvent(
                kind=EventKind.PARENT_LINK_CHANGED,
                payload={"parent": "alice"},
            )
            prompt._schedule_refresh_for_event(ev)
            assert "alice" in prompt._pending
            prompt.detach()
        finally:
            await t.shutdown()
