"""Acknowledged plugin writes racing a store handoff follow the live binding."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_scope import state_prefix
from kohakuterrarium.terrarium import session_plugin_state
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("operation", ["merge", "split"])
async def test_threaded_write_after_capture_is_not_lost(
    tmp_path, monkeypatch, operation
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["ok"]), io="headless", start=True
        )
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        contexts = {
            state_prefix(creature.name): PluginContext(
                _host_agent=creature.agent, _plugin_name="threaded"
            )
            for creature in (alice, bob)
        }
        for context in contexts.values():
            context.set_state("value", "before")
        selected = []
        start, attempting, acknowledged = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        errors = []

        def writer():
            try:
                assert start.wait(5)
                attempting.set()
                selected[0].set_state("value", "after")
                acknowledged.set()
            except BaseException as exc:
                errors.append(exc)

        replace = session_plugin_state._replace

        def replace_while_writer_races(store, prefix, values):
            if not selected:
                selected.append(contexts[prefix])
                start.set()
                assert attempting.wait(5)
                # An unfenced writer can acknowledge before the stale copy lands.
                acknowledged.wait(0.2)
            return replace(store, prefix, values)

        monkeypatch.setattr(
            session_plugin_state, "_replace", replace_while_writer_races
        )
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            if operation == "merge":
                await engine.connect(alice, bob, channel="bridge")
            else:
                await engine.disconnect(alice, bob, channel="bridge")
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
            assert errors == []
            assert acknowledged.is_set()
            assert selected[0].get_state("value") == "after"
        finally:
            start.set()
            await asyncio.to_thread(thread.join, 5)
