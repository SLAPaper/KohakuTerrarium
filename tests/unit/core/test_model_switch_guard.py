"""Runtime mutations reject a pending model replacement before changing state."""

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.modules.tool.base import BaseTool, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


class _Tool(BaseTool):
    tool_name = "guarded_tool"
    description = "A runtime tool"

    async def _execute(self, args, **kwargs):
        return ToolResult(output="ok")


class _Plugin(BasePlugin):
    name = "guarded_plugin"


@pytest.mark.parametrize(
    "action",
    [
        "add_tool",
        "add_plugin",
        "refresh",
        "native_options",
        "local_options",
        "plugin_options",
        "disable_plugin",
        "unregister_plugin",
        "replace_prompt",
        "attach",
        "compact",
    ],
)
async def test_model_pending_guard_precedes_live_mutation(action):
    agent = await Agent.build(
        AgentConfig(name="guard", system_prompt="original"),
        llm=ScriptedLLM(["ok"]),
        plugins=[_Plugin()],
        io="headless",
    )
    await agent.start()
    before = agent.controller.conversation.get_system_message().content
    agent._model_switch_pending = True
    try:
        with pytest.raises(ConflictError, match="Model switch"):
            match action:
                case "add_tool":
                    agent.add_tool(_Tool())
                case "add_plugin":
                    await agent.add_plugin(_Plugin())
                case "refresh":
                    agent.refresh_system_prompt()
                case "native_options":
                    agent.native_tool_options.set("absent", {})
                case "local_options":
                    agent.tool_options.set("absent", {})
                case "plugin_options":
                    agent.plugin_options.set("guarded_plugin", {})
                case "disable_plugin":
                    agent.plugins.disable("guarded_plugin")
                case "unregister_plugin":
                    agent.plugins.unregister("guarded_plugin")
                case "replace_prompt":
                    agent.update_system_prompt("replacement", replace=True)
                case "attach":
                    agent.attach_session_store(None)
                case "compact":
                    agent.compact_manager.trigger_compact()
        assert agent.registry.get_tool("guarded_tool") is None
        assert agent.plugins.is_enabled("guarded_plugin")
        assert agent.controller.conversation.get_system_message().content == before
    finally:
        agent._model_switch_pending = False
        await agent.stop()
