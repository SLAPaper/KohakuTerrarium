"""Manager provenance survives initialization failures and suspended loading."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium.core.job import JobState
from kohakuterrarium.core.registry import Registry
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
async def test_plugin_initialization_is_a_registered_job(monkeypatch, cancel):
    entered, release = asyncio.Event(), asyncio.Event()

    class Plugins:
        def __init__(self):
            self.unloaded = 0

        async def load_all(self, context):
            entered.set()
            await release.wait()
            raise RuntimeError("plugin load failed")

        async def unload_all(self):
            self.unloaded += 1

    plugins = Plugins()
    monkeypatch.setattr(
        "kohakuterrarium.modules.subagent.manager.build_plugin_manager",
        lambda *args: plugins,
    )
    manager = SubAgentManager(Registry(), ScriptedLLM(["must not run"]))
    manager.register(SubAgentConfig(name="child", default_plugins=False))
    context = {
        "origin": {"turn_index": 1, "parent_branch_path": []},
        "tool_call_id": "provider",
    }
    expected = deepcopy(context)
    spawning = asyncio.create_task(
        manager.spawn(
            "child", "work", job_id="agent_child_one", completion_context=context
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        status = manager.job_store.get_status("agent_child_one")
        assert status is not None
        assert status.context == expected
        context["origin"]["turn_index"] = 99
        assert spawning.done()
        job_id = await spawning
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
        assert plugins.unloaded == 1
    finally:
        release.set()
        await asyncio.gather(spawning, return_exceptions=True)
        await manager.cancel_all()
