"""Detached model-switch preparation against a real unstarted Agent."""

import pytest

from kohakuterrarium.builtins.tools.image_gen import ImageGenTool
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.model_switch_prepare import get_builtin_tool, prepare
from kohakuterrarium.modules.tool.base import BaseTool, ToolConfig, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


class _Probe(ScriptedLLM):
    def __init__(self, provider_name, offered, *, model="probe", max_context=1000):
        super().__init__(["OK"])
        self.provider_name = provider_name
        self.provider_native_tools = frozenset(offered)
        self.model = model
        self._profile_max_context = max_context
        self.prompt_cache_key = None
        self.drop_callbacks = []

    def on_emergency_drop(self, callback):
        self.drop_callbacks.append(callback)


class _LocalOffered(BaseTool):
    def __init__(self, config=None):
        super().__init__(config=config or ToolConfig())
        self.refreshed = None

    @property
    def tool_name(self):
        return "offered_local"

    @property
    def description(self):
        return "local tool offered by a provider"

    def runtime_option_schema(self):
        return {"mode": {"type": "string", "default": "safe"}}

    def refresh_runtime_options(self, options):
        self.refreshed = dict(options)

    async def _execute(self, args, **kwargs):
        return ToolResult(output="ok")


def _catalog(name, config=None):
    if name == "offered_local":
        return _LocalOffered(config)
    return get_builtin_tool(name, config)


@pytest.fixture
async def agent(monkeypatch):
    monkeypatch.setattr(
        "kohakuterrarium.core.model_switch_prepare.get_builtin_tool", _catalog
    )
    built = await Agent.build(
        AgentConfig(
            name="switch-agent",
            system_prompt="base",
            tools=[],
            tool_format="xml",
        ),
        llm=_Probe("openai", ()),
        io="headless",
    )
    built.llm.prompt_cache_key = "cache-1"
    return built


@pytest.mark.asyncio
async def test_prepare_does_not_mutate_live_agent(agent):
    old_llm = agent.llm
    identities = {
        name: agent.registry.get_tool(name) for name in agent.registry.list_tools()
    }
    prompt_before = agent.controller.conversation.get_system_message().content

    bundle = prepare(
        agent,
        "probe/new",
        _Probe("other", ("offered_local",), model="next"),
        "probe/next",
        {},
    )

    assert agent.llm is old_llm
    assert agent._llm_identifier == ""
    assert agent.controller.llm is old_llm
    assert "offered_local" not in agent.registry.list_tools()
    assert agent.controller.conversation.get_system_message().content == prompt_before
    for name, tool in identities.items():
        assert agent.registry.get_tool(name) is tool
    assert bundle.system_prompt and "offered_local" in bundle.system_prompt
    assert bundle.llm.prompt_cache_key == "cache-1"
    assert bundle.registry_tools["skill"] is identities["skill"]


@pytest.mark.asyncio
async def test_publish_swaps_new_tool_and_preserves_existing_identity(agent):
    existing = ImageGenTool()
    agent.registry.register_tool(existing)
    agent.executor.register_tool(existing)
    skill = agent.registry.get_tool("skill")
    new_llm = _Probe("codex", ("offered_local", "image_gen"), model="codex-next")

    bundle = prepare(agent, "codex/next", new_llm, "codex/next", {})
    bundle.publish(agent)

    assert agent.llm is new_llm
    assert agent.controller.llm is new_llm
    assert agent.subagent_manager.llm is new_llm
    assert agent._llm_selector == "codex/next"
    assert agent.registry.get_tool("image_gen") is existing
    assert agent.executor.get_tool("image_gen") is existing
    assert agent.registry.get_tool("skill") is skill
    added = agent.registry.get_tool("offered_local")
    assert added is agent.executor.get_tool("offered_local")
    assert "offered_local" in agent._auto_provider_tools
    assert agent.controller.conversation.get_system_message().content == (
        bundle.system_prompt
    )


@pytest.mark.asyncio
async def test_saved_native_options_apply_to_copy_only(agent, monkeypatch):
    created: list[ImageGenTool] = []

    def catalog(name, config=None):
        if name == "image_gen":
            tool = ImageGenTool(config=config)
            created.append(tool)
            return tool
        return _catalog(name, config)

    monkeypatch.setattr(
        "kohakuterrarium.core.model_switch_prepare.get_builtin_tool", catalog
    )
    skill = agent.registry.get_tool("skill")
    new_llm = _Probe("codex", ("image_gen",), model="codex-next")
    bundle = prepare(
        agent,
        "codex/next",
        new_llm,
        "codex/next",
        {"native": {"image_gen": {"quality": "high"}}},
    )

    staged = bundle.registry_tools["image_gen"]
    assert created and staged is not created[0]
    assert staged.quality == "high"
    assert created[0].quality is None
    assert created[0].config.extra == {}
    assert bundle.registry_tools["skill"] is skill
    bundle.publish(agent)
    assert agent.registry.get_tool("image_gen") is staged
    assert agent.registry.get_tool("skill") is skill
    assert agent.native_tool_options.get("image_gen") == {"quality": "high"}


@pytest.mark.asyncio
async def test_failure_before_bundle_leaves_original_resources(agent, monkeypatch):
    def broken(name, config=None):
        raise RuntimeError("catalog failed")

    monkeypatch.setattr(
        "kohakuterrarium.core.model_switch_prepare.get_builtin_tool", broken
    )
    old_llm = agent.llm
    before = list(agent.registry.list_tools())
    with pytest.raises(RuntimeError, match="catalog failed"):
        prepare(agent, "x", _Probe("other", ("missing",)), "x", {})
    assert agent.llm is old_llm
    assert agent.registry.list_tools() == before


async def test_prompt_skill_registration_cannot_mutate_live_executor(agent):
    agent.registry.unregister_tool("skill")
    agent.executor.unregister_tool("skill")
    bundle = prepare(
        agent, "probe/new", _Probe("other", ("offered_local",)), "probe/new", {}
    )
    assert agent.registry.get_tool("skill") is None
    assert agent.executor.get_tool("skill") is None
    assert bundle.registry_tools["skill"] is bundle.executor_tools["skill"]


@pytest.mark.asyncio
async def test_prompt_rebuild_uses_staged_registry(agent):
    plugins = agent.plugins
    bundle = prepare(
        agent, "probe/new", _Probe("other", ("offered_local",)), "probe/new", {}
    )
    assert "offered_local" in bundle.registry_tools
    assert bundle.system_prompt and "offered_local" in bundle.system_prompt
    assert agent.plugins is plugins
    assert "offered_local" not in (
        agent.controller.conversation.get_system_message().content
    )
