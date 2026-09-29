"""Manager provenance survives initialization failures and suspended loading."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.job import JobState
from kohakuterrarium.core.registry import Registry
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.subagent.manager import (
    SubAgentManager,
    subagent_background_requested,
)
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize(
    "value, expected",
    [(False, False), (True, True), ("false", False), ("TRUE", True), ("off", False)],
)
def test_wait_intent_normalization(value, expected):
    assert subagent_background_requested({"run_in_background": value}) is expected


def test_invalid_wait_intent_rejected():
    with pytest.raises(ValueError, match="boolean"):
        subagent_background_requested({"run_in_background": "maybe"})


async def test_cancel_before_runner_entry_finishes_status():
    manager = SubAgentManager(Registry(), ScriptedLLM(["must not run"]))
    manager.register(SubAgentConfig(name="child", default_plugins=False))
    context = {"origin": {"turn_index": 1}, "tool_call_id": "provider"}
    job_id = await manager.spawn("child", "work", completion_context=context)
    task = manager._tasks[job_id]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert manager.job_store.get_status(job_id).state == JobState.CANCELLED
    assert manager.job_store.get_status(job_id).context == context
    assert manager.get_result(job_id).cancelled is True


@pytest.mark.parametrize("cancel", [False, True])
async def test_plugin_initialization_is_a_registered_job(tmp_path, cancel):
    entered, release = asyncio.Event(), asyncio.Event()

    class StartupPlugin(BasePlugin):
        name = "startup_origin_probe"

        def __init__(self):
            super().__init__()
            self.unloaded = 0

        async def on_load(self, context):
            entered.set()
            await release.wait()
            raise RuntimeError("plugin load failed")

        async def on_unload(self):
            self.unloaded += 1

    plugin = StartupPlugin()
    provider = ScriptedLLM(["must not run"])
    agent = await Agent.build(
        AgentConfig(name="parent", agent_path=tmp_path), llm=provider, io="headless"
    )
    await agent.start()
    manager = agent.subagent_manager
    manager._strict = True
    manager.register(SubAgentConfig(name="child", default_plugins=False))
    context = {
        "origin": {"turn_index": 1, "parent_branch_path": []},
        "tool_call_id": "provider",
    }
    expected = deepcopy(context)
    try:
        job_id = await manager.spawn(
            "child", "work", job_id="agent_child_one", completion_context=context
        )
        manager._jobs[job_id].subagent.plugins.register(plugin)
        try:
            await asyncio.wait_for(entered.wait(), 3)
        except TimeoutError:
            pytest.fail(
                f"Plugin startup never entered: {manager.get_result('agent_child_one')}"
            )
        status = manager.job_store.get_status("agent_child_one")
        assert status is not None
        assert status.context == expected
        context["origin"]["turn_index"] = 99
        assert provider.call_count == 0
        if cancel:
            manager._tasks[job_id].cancel()
        else:
            release.set()
        result = await manager.wait_for(job_id)
        assert not result.success
        assert manager.job_store.get_status(job_id).state == (
            JobState.CANCELLED if cancel else JobState.ERROR
        )
        assert manager.job_store.get_status(job_id).context == expected
        assert plugin.unloaded == 1
        assert provider.call_count == 0
    finally:
        release.set()
        await agent.stop()
