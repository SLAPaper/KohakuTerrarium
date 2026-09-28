"""Real graph connects reserve public admissions before awaiting source drains."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import session_coord, tools_group_channel
from kohakuterrarium.terrarium.group_tool_context import GroupContext


@pytest.mark.parametrize(
    "operation", ["disconnect", "unwire", "remove", "remove_active"]
)
async def test_split_copy_does_not_publish_projected_graphs_while_worker_is_held(
    tmp_path, monkeypatch, operation
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    generating = asyncio.Event()

    class ActiveProvider(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            yield "unfinished response"
            generating.set()
            await asyncio.Event().wait()

    turn = None
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        if operation.startswith("remove"):
            bridge = await engine.add_creature(
                AgentConfig(name="bridge"),
                llm=(
                    ActiveProvider(["bridge"])
                    if operation == "remove_active"
                    else ScriptedLLM(["bridge"])
                ),
                io="headless",
                start=True,
            )
            await engine.connect(alice, bridge, channel="left")
            await engine.connect(bridge, bob, channel="right")
            if operation == "remove_active":
                turn = asyncio.create_task(bridge.run("busy", raise_on_error=False))
                await asyncio.wait_for(generating.wait(), 3)
        else:
            await engine.connect(alice, bob, channel="bridge")
        original_graph = alice.graph_id
        capture = session_coord.capture_runtime_transfer

        def held(store, names):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return capture(store, names)

        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held)
        watchdog = threading.Timer(3, release.set)
        watchdog.start()
        if operation == "unwire":
            context = GroupContext(
                engine, alice, engine._topology.graphs[original_graph]
            )
            moving = tools_group_channel._unwire(context, bob, "bridge", "listen")
        elif operation.startswith("remove"):
            moving = engine.remove_creature(bridge)
        else:
            moving = engine.disconnect(alice, bob, channel="bridge")
        pending = asyncio.create_task(moving)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not release.is_set(), "split copy blocked the event loop"
            assert alice.graph_id == bob.graph_id == original_graph
            assert len(engine.list_graphs()) == 1
            if operation.startswith("remove"):
                assert engine.get_creature(bridge.creature_id) is bridge
                assert not bridge.is_running
            release.set()
            await asyncio.wait_for(pending, 8)
            assert alice.graph_id != bob.graph_id
            if turn is not None:
                result = await asyncio.wait_for(turn, 3)
                assert result.status == "interrupted"
                assert not result.interrupted_by_user
        finally:
            release.set()
            watchdog.cancel()
            await asyncio.gather(
                pending, *([turn] if turn else []), return_exceptions=True
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "channel",
        "attach",
        "root",
        "start",
        "stop",
        "stop_graph",
        "sink",
        "unsink",
        "shutdown",
        "wire_existing",
        "add_creature",
        "recipe",
        "adopt",
    ],
)
async def test_engine_mutations_reject_while_removal_is_quiescing(
    tmp_path, monkeypatch, mutation
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        await engine.connect(alice, bob, channel="bridge")
        gid, source = alice.graph_id, alice.agent.session_store
        original_stop = type(bob).stop

        async def held_stop(creature, **kwargs):
            if creature is bob and not entered.is_set():
                entered.set()
                await release.wait()
            return await original_stop(creature, **kwargs)

        monkeypatch.setattr(type(bob), "stop", held_stop)
        removal = asyncio.create_task(engine.remove_creature(bob))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert getattr(alice.agent, "_topology_handoff_task", None) is None
            actions = {
                "channel": lambda: engine.add_channel(gid, "forbidden"),
                "attach": lambda: engine.attach_session(gid, source),
                "root": lambda: engine.assign_root(alice),
                "start": lambda: engine.start(alice),
                "stop": lambda: engine.stop(alice),
                "stop_graph": lambda: engine.stop_graph(gid),
                "sink": lambda: engine.wire_output_sink(
                    alice, alice.agent._session_output
                ),
                "unsink": lambda: engine.unwire_output_sink(alice, "unknown"),
                "shutdown": engine.shutdown,
                "wire_existing": lambda: tools_group_channel._wire(
                    GroupContext(engine, alice, engine.get_graph(gid)),
                    alice,
                    "bridge",
                    "listen",
                ),
                "add_creature": lambda: engine.add_creature(
                    AgentConfig(name="third"), llm=ScriptedLLM(["third"]), io="headless"
                ),
                "recipe": lambda: engine.apply_recipe(
                    {"terrarium": {"name": "extra", "creatures": []}}
                ),
                "adopt": lambda: engine.adopt_session(source),
            }
            with pytest.raises(ConflictError, match="handoff"):
                await asyncio.wait_for(actions[mutation](), 2)
            assert alice.is_running and bob.is_running
            assert "forbidden" not in engine.get_graph(gid).channels
            assert not alice.is_privileged
            release.set()
            await asyncio.wait_for(removal, 5)
        finally:
            release.set()
            await asyncio.gather(removal, return_exceptions=True)


async def test_child_completing_during_handoff_persists_on_new_parent_store(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), threading.Event()
    child_entered, child_release, child_returned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    loop = asyncio.get_running_loop()

    class Child(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            child_entered.set()
            await child_release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk
            child_returned.set()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        await alice.run("first")
        agent, source = alice.agent, alice.agent.session_store
        manager = agent.subagent_manager
        manager.register(SubAgentConfig(name="child", max_turns=1))
        manager.llm = Child(["child result"])
        context = capture_tool_context(agent, "child_handoff", "background")
        job = await manager.spawn(
            "child", "task", job_id="child_handoff", completion_context=context
        )
        agent._bg_controller_notify[job] = False
        await asyncio.wait_for(child_entered.wait(), 3)
        flush, save = type(source).flush, type(source).save_subagent
        destinations, once = [], False

        def saving(store, *args, **kwargs):
            destinations.append(store)
            return save(store, *args, **kwargs)

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return flush(store)

        monkeypatch.setattr(type(source), "save_subagent", saving)
        monkeypatch.setattr(type(source), "flush", held)
        moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            child_release.set()
            await asyncio.wait_for(child_returned.wait(), 3)
            await agent._session_state_owner.submit_internal(lambda: None)
            release.set()
            await asyncio.wait_for(moving, 8)
            result = await manager.wait_for(job, timeout=5)
            assert result.output == "child result"
            assert destinations and all(
                store is agent.session_store for store in destinations
            )
            row = await agent.session_store.run(
                agent.session_store.find_subagent_run, job, parent="alice"
            )
            assert row["meta"]["success"] is True
        finally:
            child_release.set()
            release.set()
            await asyncio.gather(moving, return_exceptions=True)
            await manager.wait_all(timeout=5)


async def test_new_background_terminal_chooses_rebound_sink_after_reserved_handoff(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        await alice.run("first")
        agent, source = alice.agent, alice.agent.session_store
        context = capture_tool_context(
            agent, "read_during_move", "background", "native-call"
        )
        event = create_tool_complete_event(
            job_id="read_during_move", content="late completion", **context
        )
        agent._bg_controller_notify[event.job_id] = False
        publish, destinations = agent._publish_bg_complete, []

        def publishing(*args, **kwargs):
            destinations.append(agent.session_store)
            return publish(*args, **kwargs)

        monkeypatch.setattr(agent, "_publish_bg_complete", publishing)
        flush = type(source).flush
        once = False

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return flush(store)

        monkeypatch.setattr(type(source), "flush", held)
        moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            agent._on_bg_complete(event)
            publications = tuple(agent.output_router._terminal_publications)
            await agent._session_state_owner.submit_internal(lambda: None)
            release.set()
            await asyncio.wait_for(moving, 8)
            await asyncio.wait_for(asyncio.gather(*publications), 5)
            assert agent.session_store is not source
            assert destinations == [agent.session_store]
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert (
                sum(
                    row["type"] == "tool_result" and row.get("job_id") == event.job_id
                    for row in rows
                )
                == 1
            )
        finally:
            release.set()
            await asyncio.gather(moving, return_exceptions=True)


@pytest.mark.parametrize("operation", ["connect", "disconnect", "remove_channel"])
async def test_handoff_cancellation_joins_reserved_drain_and_rejects_inherited_task(
    tmp_path, monkeypatch, operation
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["alice"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["bob"]), io="headless", start=True
        )
        if operation != "connect":
            await engine.connect(alice, bob, channel="bridge")
        source = bob.agent.session_store
        before = len(engine.list_graphs())
        flush = type(source).flush
        once, inherited = False, []

        async def descendant():
            with pytest.raises(ConflictError, match="handoff"):
                await bob.agent._submit_session_state(lambda: inherited.append("wrong"))
            inherited.append("rejected")

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return flush(store)

        drain = type(bob.agent._session_output).drain
        descendants = []

        async def draining(sink):
            if (
                sink._store is source
                and getattr(bob.agent, "_topology_handoff_task", None)
                is asyncio.current_task()
            ):
                descendants.append(asyncio.create_task(descendant()))
            return await drain(sink)

        monkeypatch.setattr(type(source), "flush", held)
        monkeypatch.setattr(type(bob.agent._session_output), "drain", draining)
        completions = []
        drain_drives = engine._drain_drive_topology

        async def finish():
            await drain_drives()
            completions.append(True)

        monkeypatch.setattr(engine, "_drain_drive_topology", finish)
        pending = asyncio.create_task(
            engine.remove_channel(alice.graph_id, "bridge")
            if operation == "remove_channel"
            else getattr(engine, operation)(alice, bob, channel="bridge")
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert len(engine.list_graphs()) == before
            with pytest.raises(ConflictError, match="handoff"):
                await engine.connect(alice, bob, channel="competing")
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            assert not pending.done()
            assert bob.agent._topology_handoff_task is not None
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 8)
            await asyncio.gather(*descendants)
            assert completions == [True]
            assert inherited and set(inherited) == {"rejected"}
            assert len(engine.list_graphs()) == (1 if operation == "connect" else 2)
            assert all(
                getattr(agent, "_topology_handoff_task", None) is None
                for agent in (alice.agent, bob.agent)
            )
        finally:
            release.set()
            await asyncio.gather(pending, *descendants, return_exceptions=True)


async def test_replaced_output_during_drain_is_rejected_before_topology_change(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        store, original = bob.agent.session_store, bob.agent._session_output
        old_graphs = alice.graph_id, bob.graph_id
        flush = type(store).flush

        def held(source):
            if source is store and not release.is_set():
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return flush(source)

        monkeypatch.setattr(type(store), "flush", held)
        pending = asyncio.create_task(engine.connect(alice, bob, channel="bridge"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            bob.agent._session_output = alice.agent._session_output
            release.set()
            with pytest.raises(ConflictError, match="binding"):
                await pending
            assert (alice.graph_id, bob.graph_id) == old_graphs
        finally:
            release.set()
            bob.agent._session_output = original
            await asyncio.gather(pending, return_exceptions=True)


async def test_failed_source_drain_leaves_topology_and_releases_all_leases(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["alice"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["bob"]), io="headless", start=True
        )
        old = (alice.graph_id, bob.graph_id)
        flush = type(alice.agent.session_store).flush

        def failed(store):
            if store is alice.agent.session_store:
                raise OSError("source drain failed")
            return flush(store)

        with monkeypatch.context() as patch:
            patch.setattr(type(alice.agent.session_store), "flush", failed)
            with pytest.raises(OSError, match="source drain failed"):
                await engine.connect(alice, bob, channel="bridge")
        assert (alice.graph_id, bob.graph_id) == old
        assert all(
            getattr(agent, "_topology_handoff_task", None) is None
            for agent in (alice.agent, bob.agent)
        )
        assert getattr(engine, "_topology_handoff_task", None) is None
        await bob.agent._submit_session_state(lambda: None)


@pytest.mark.parametrize("phase", ["drain", "copy"])
async def test_connect_from_running_tool_drains_without_publishing_or_taking_turn_lock(
    tmp_path, monkeypatch, phase
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:

        class Connect(BaseTool):
            tool_name = "connect_now"
            description = "Connect reserved graphs"
            execution_mode = ExecutionMode.DIRECT

            async def _execute(self, args, **kwargs):
                assert alice.agent._processing_lock.locked()
                await engine.connect(alice, bob, channel="bridge")
                return ToolResult(output="connected")

        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(["[/connect_now][connect_now/]", "done"]),
            tools=[Connect()],
            is_privileged=True,
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["bob"]), io="headless", start=True
        )
        original_graphs = (alice.graph_id, bob.graph_id)
        source = bob.agent.session_store
        flush = type(source).flush
        once = False

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return flush(store)

        if phase == "drain":
            monkeypatch.setattr(type(source), "flush", held)
        else:
            capture = session_coord.capture_runtime_transfer

            def held_copy(store, names):
                if store is source:
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(5)
                return capture(store, names)

            monkeypatch.setattr(session_coord, "capture_runtime_transfer", held_copy)
        watchdog = threading.Timer(3, release.set)
        watchdog.start()
        pending = asyncio.create_task(alice.run("join", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not release.is_set(), "source drain blocked the owner event loop"
            assert (alice.graph_id, bob.graph_id) == original_graphs
            assert len(engine.list_graphs()) == 2
            with pytest.raises(ConflictError, match="handoff"):
                await bob.agent._submit_session_state(lambda: None)
            context = PluginContext(
                agent_name="bob", _host_agent=bob.agent, _plugin_name="probe"
            )
            with pytest.raises(ConflictError, match="handoff"):
                await context.set_state_async("new", "rejected")
            batch_entered = asyncio.Event()
            turn = bob.agent._run_turn_for_batch

            async def attempted(batch):
                batch_entered.set()
                return await turn(batch)

            monkeypatch.setattr(bob.agent, "_run_turn_for_batch", attempted)
            queued = asyncio.create_task(
                bob.run("queued while moving", raise_on_error=False)
            )
            await asyncio.wait_for(batch_entered.wait(), 2)
            await asyncio.sleep(0)
            release.set()
            result = await asyncio.wait_for(pending, 8)
            queued_result = await asyncio.wait_for(queued, 5)
            assert queued_result.ok and queued_result.text == "bob"
            assert result.ok and result.text == "done"
            assert alice.graph_id == bob.graph_id
            assert getattr(bob.agent, "_topology_handoff_task", None) is None
        finally:
            release.set()
            watchdog.cancel()
            await asyncio.gather(pending, return_exceptions=True)
