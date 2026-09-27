"""Explicit attachment failures preserve runtime state as well as store bindings."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.session_attachment_state import AttachmentJournal
from kohakuterrarium.core.session_attachment_prepare import PreparedAttachment
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.modules.tool.base import BaseTool, ToolResult
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import session_attach


class OptionPlugin(BasePlugin):
    name = "attach_options"

    def __init__(self):
        super().__init__()
        self.options = {"limit": 2}
        self.limit = 2

    @classmethod
    def option_schema(cls):
        return {"limit": {"type": "int", "default": 2}}

    def refresh_options(self):
        self.limit = self.options.get("limit", 2)


class OptionTool(BaseTool):
    tool_name = "attach_tool"
    description = "Expose attachment option state."

    def __init__(self):
        super().__init__()
        self.limit = 2

    @classmethod
    def runtime_option_schema(cls):
        return {"limit": {"type": "int", "default": 2}}

    def refresh_runtime_options(self, options):
        self.limit = options.get("limit", 2)

    def get_parameters_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, args, context=None):
        return ToolResult(output=str(self.limit))


class NativeOptionTool(OptionTool):
    tool_name = "attach_native"
    is_provider_native = True

    @classmethod
    def provider_native_option_schema(cls):
        return {"limit": {"type": "int", "default": 2}}

    def refresh_native_options(self):
        self.limit = self.config.extra.get("limit", 2)


@pytest.mark.parametrize("field", ["compact", "plugin", "native", "tool"])
async def test_replacement_empty_target_does_not_inherit_source_options(
    tmp_path, field
):
    plugin, native, ordinary = OptionPlugin(), NativeOptionTool(), OptionTool()
    target = SessionStore(tmp_path / "empty.kohakutr")
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["ok"]),
                plugins=[plugin],
                tools=[native, ordinary],
                io="headless",
                start=True,
            )
            agent = creature.agent
            agent.plugin_options.set("attach_options", {"limit": 8})
            agent.native_tool_options.set("attach_native", {"limit": 8})
            agent.tool_options.set("attach_tool", {"limit": 8})
            await agent._submit_session_state(
                lambda: setattr(agent.compact_manager, "_compact_count", 8)
            )
            await agent._session_output.flush()
            await engine.attach_session(creature.graph_id, target)
            state = {
                "compact": agent.compact_manager._compact_count,
                "plugin": plugin.limit,
                "native": native.limit,
                "tool": ordinary.limit,
            }
            assert state[field] == (0 if field == "compact" else 2)
            assert (await creature.run("continue", timeout=5)).text == "ok"
    finally:
        target.close(update_status=False)


async def test_explicit_target_option_reads_use_destination_worker(
    tmp_path, monkeypatch
):
    target = SessionStore(tmp_path / "target.kohakutr")
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["ok"]),
                io="headless",
                start=True,
            )
            await creature.agent._session_output.flush()
            worker = await target.run(threading.get_ident)
            get, put = type(target.state).get, type(target.state).__setitem__
            reads, writes, installs = [], [], []
            install = PreparedAttachment.install

            def observed_install(prepared):
                installs.append(asyncio.current_task())
                return install(prepared)

            monkeypatch.setattr(PreparedAttachment, "install", observed_install)
            keys = {
                "alice:compact_count",
                "alice:last_compact_time",
                "alice:native_tool_options",
                "alice:plugin_options",
                "alice:tool_options",
            }

            def checked(table, key, *args, **kwargs):
                if table is target.state and key in keys:
                    reads.append(threading.get_ident())
                return get(table, key, *args, **kwargs)

            def checked_put(table, key, value):
                if table is target.state and key in keys:
                    writes.append(threading.get_ident())
                return put(table, key, value)

            monkeypatch.setattr(type(target.state), "get", checked)
            monkeypatch.setattr(type(target.state), "__setitem__", checked_put)
            await engine.attach_session(creature.graph_id, target)
            assert reads and set(reads) == {worker}
            assert writes and set(writes) == {worker}
            assert installs == [creature.agent._session_state_owner.task]
    finally:
        target.close(update_status=False)


@pytest.mark.parametrize("failure", ["prepare", "refresh", "persist"])
async def test_prepared_attach_failure_restores_source_without_accepting_target(
    tmp_path, monkeypatch, failure
):
    plugin = OptionPlugin()
    target = SessionStore(tmp_path / "target.kohakutr")
    try:
        await target.run(
            lambda: target.state.__setitem__(
                "alice:plugin_options",
                {"attach_options": {"limit": "invalid" if failure == "prepare" else 9}},
            )
        )
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["ok"]),
                plugins=[plugin],
                io="headless",
                start=True,
            )
            agent = creature.agent
            original, sink = agent.session_store, agent._session_output
            await sink.flush()
            if failure == "refresh":
                refresh = plugin.refresh_options

                def fail_refresh():
                    if plugin.options.get("limit") == 9:
                        raise RuntimeError("forward refresh failed")
                    refresh()

                monkeypatch.setattr(plugin, "refresh_options", fail_refresh)
            elif failure == "persist":
                put = type(target.state).__setitem__

                def fail_put(table, key, value):
                    if table is target.state and key == "alice:plugin_options":
                        raise RuntimeError("target write failed")
                    return put(table, key, value)

                monkeypatch.setattr(type(target.state), "__setitem__", fail_put)
            with pytest.raises((ValueError, RuntimeError)):
                await engine.attach_session(creature.graph_id, target)
            assert agent.session_store is original and agent._session_output is sink
            assert plugin.options == {"limit": 2} and plugin.limit == 2
            assert not getattr(agent, "_session_attach_uncertain", False)
            assert (await creature.run("source still usable", timeout=5)).text == "ok"
    finally:
        target.close(update_status=False)


@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize("field", ["compact", "plugin", "native", "tool"])
async def test_failed_attachment_restores_preexisting_runtime_state(
    tmp_path, monkeypatch, persisted, field
):
    plugin, native, ordinary = OptionPlugin(), NativeOptionTool(), OptionTool()
    target = SessionStore(tmp_path / "target.kohakutr")
    keys = {
        "compact": ("alice:compact_count", 9),
        "plugin": ("alice:plugin_options", {"attach_options": {"limit": 9}}),
        "native": ("alice:native_tool_options", {"attach_native": {"limit": 9}}),
        "tool": ("alice:tool_options", {"attach_tool": {"limit": 9}}),
    }
    key, value = keys[field]
    await target.run(lambda: target.state.__setitem__(key, value))
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["still usable"]),
                plugins=[plugin],
                tools=[native, ordinary],
                session=persisted,
                io="headless",
                start=True,
            )
            agent, original = creature.agent, creature.agent.session_store
            if agent._session_output is not None:
                await agent._session_output.flush()
            sink = agent._session_output
            extra = {
                key: deepcopy(value) if key.endswith("_options") else value
                for key, value in agent.session.extra.items()
            }
            names = [entry[0] for entry in keys.values()]

            def source_values():
                return {name: deepcopy(original.state.get(name)) for name in names}

            stored = await original.run(source_values) if original else None
            owners = []
            restore_values, restore_bindings = (
                AttachmentJournal.restore_values,
                AttachmentJournal.restore_bindings,
            )

            def observed_values(journal):
                owners.append(asyncio.current_task())
                return restore_values(journal)

            def observed_bindings(journal):
                owners.append(asyncio.current_task())
                return restore_bindings(journal)

            monkeypatch.setattr(AttachmentJournal, "restore_values", observed_values)
            monkeypatch.setattr(
                AttachmentJournal, "restore_bindings", observed_bindings
            )
            before = (
                agent.compact_manager._compact_count,
                plugin.limit,
                native.limit,
                ordinary.limit,
            )

            async def failed_checkpoint(*args):
                observed = {
                    "compact": agent.compact_manager._compact_count,
                    "plugin": plugin.limit,
                    "native": native.limit,
                    "tool": ordinary.limit,
                }
                assert observed[field] == 9
                raise RuntimeError("checkpoint rejected target")

            with monkeypatch.context() as patch:
                patch.setattr(
                    session_attach.checkpoint, "checkpoint", failed_checkpoint
                )
                with pytest.raises(RuntimeError, match="checkpoint rejected target"):
                    await engine.attach_session(creature.graph_id, target)
            assert agent.session_store is original
            assert agent._session_output is sink
            assert agent.session.extra == extra
            assert owners == [agent._session_state_owner.task] * 2
            if original:
                assert await original.run(source_values) == stored
            assert (
                agent.compact_manager._compact_count,
                plugin.limit,
                native.limit,
                ordinary.limit,
            ) == before
            assert not getattr(agent, "_session_attach_uncertain", False)
            assert (await creature.run("continue", timeout=5)).text == "still usable"
    finally:
        target.close(update_status=False)


@pytest.mark.parametrize("persisted", [False, True])
async def test_failed_attach_preserves_legacy_key_and_sink_identity(
    tmp_path, monkeypatch, persisted
):
    native = NativeOptionTool()
    target = SessionStore(tmp_path / "target.kohakutr")
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["ok"]),
                tools=[native],
                session=persisted,
                io="headless",
                start=True,
            )
            agent = creature.agent
            if agent._session_output:
                await agent._session_output.flush()
                agent._session_output._capture_activity = False
            sink = agent._session_output
            pad = agent.session.scratchpad
            observer = pad._on_write
            legacy = '{"attach_native":{"limit":5}}'
            pad._data["__native_tool_options__"] = legacy
            agent.session.extra["native_tool_options"] = {}
            original_outputs = list(agent.output_router._secondary_outputs)

            async def reject(*args):
                assert pad.get("__native_tool_options__") == (
                    legacy if persisted else None
                )
                assert native.limit == (2 if persisted else 5)
                raise RuntimeError("reject after legacy consumption")

            with monkeypatch.context() as patch:
                patch.setattr(session_attach.checkpoint, "checkpoint", reject)
                with pytest.raises(RuntimeError, match="legacy consumption"):
                    await engine.attach_session(creature.graph_id, target)
            assert native.limit == 2
            assert pad.get("__native_tool_options__") == legacy
            assert pad._on_write is observer
            assert agent.session.extra["native_tool_options"] == {}
            assert agent._session_output is sink
            assert agent.output_router._secondary_outputs == original_outputs
            if sink:
                assert not sink._capture_activity
    finally:
        target.close(update_status=False)


@pytest.mark.parametrize("fail_refresh", [False, True])
async def test_compensation_is_external_and_failure_keeps_uncertain_fence(
    tmp_path, monkeypatch, fail_refresh
):
    plugin = OptionPlugin()
    target = SessionStore(tmp_path / "target.kohakutr")
    await target.run(
        lambda: target.state.__setitem__(
            "alice:plugin_options", {"attach_options": {"limit": 9}}
        )
    )
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["usable"]),
                plugins=[plugin],
                io="headless",
                start=True,
            )
            agent = creature.agent
            original, sink = agent.session_store, agent._session_output
            await sink.flush()
            effects = []
            refresh = plugin.refresh_options

            def tracked_refresh():
                effects.append(
                    (
                        plugin.options["limit"],
                        asyncio.current_task(),
                        agent.session_store,
                    )
                )
                if fail_refresh and plugin.options["limit"] == 2:
                    raise RuntimeError("compensation failed")
                refresh()

            monkeypatch.setattr(plugin, "refresh_options", tracked_refresh)

            async def reject(*args):
                raise RuntimeError("primary checkpoint failure")

            with monkeypatch.context() as patch:
                patch.setattr(session_attach.checkpoint, "checkpoint", reject)
                with pytest.raises(RuntimeError, match="primary checkpoint failure"):
                    await engine.attach_session(creature.graph_id, target)
            assert [entry[0] for entry in effects] == [9, 2]
            assert all(
                task is not agent._session_state_owner.task and store is target
                for _, task, store in effects
            )
            assert agent.session_store is original and agent._session_output is sink
            assert plugin.options == {"limit": 2}
            assert (
                bool(getattr(agent, "_session_attach_uncertain", False)) == fail_refresh
            )
            result = await creature.run("continue", timeout=5, raise_on_error=False)
            assert result.status == ("error" if fail_refresh else "ok")
            assert agent.llm.call_count == (0 if fail_refresh else 1)
            assert not target._closed
        assert not target._closed
    finally:
        target.close(update_status=False)
