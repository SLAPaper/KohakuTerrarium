"""Concurrent example hooks retain their persisted cumulative state."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("budget_enforcer", "BudgetEnforcerPlugin"),
        ("tool_timer", "ToolTimerPlugin"),
        ("subagent_tracker", "SubagentTrackerPlugin"),
    ],
)
async def test_example_counters_survive_concurrent_hooks(
    tmp_path, module_name, class_name
):
    path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "plugins"
        / f"{module_name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"example_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = getattr(module, class_name)()
    store = SessionStore(tmp_path / "examples.kohakutr")
    host = SimpleNamespace(config=SimpleNamespace(name="alice"), session_store=store)
    context = PluginContext(_host_agent=host, _plugin_name=plugin.name)
    try:
        await plugin.on_load(context)
        match module_name:
            case "budget_enforcer":
                await asyncio.gather(
                    *(
                        plugin.post_llm_call(
                            [], "answer", {"prompt_tokens": 1, "completion_tokens": 2}
                        )
                        for _ in range(20)
                    )
                )
                expected = {"total_input": 20, "total_output": 40}
            case "tool_timer":
                for index in range(20):
                    await plugin.pre_tool_execute({}, job_id=f"j{index}")
                await asyncio.gather(
                    *(
                        plugin.post_tool_execute(
                            "ok", job_id=f"j{index}", tool_name="read"
                        )
                        for index in range(20)
                    )
                )
                expected = {"call_count": 20}
            case "subagent_tracker":
                await asyncio.gather(
                    *(
                        plugin.post_subagent_run(
                            SimpleNamespace(total_tokens=3), job_id=f"j{index}"
                        )
                        for index in range(20)
                    )
                )
                expected = {"completed": 20, "total_tokens": 60}
        for key, value in expected.items():
            assert await context.get_state_async(key) == value
        path = store.path
    finally:
        await asyncio.to_thread(store.close)
    reopened = SessionStore(path)
    host.session_store = reopened
    try:
        for key, value in expected.items():
            assert await context.get_state_async(key) == value
    finally:
        await asyncio.to_thread(reopened.close)
