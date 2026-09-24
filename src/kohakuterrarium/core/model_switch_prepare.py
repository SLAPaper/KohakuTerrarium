"""Prepare detached model bindings, provider tools, option state, and prompt text."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from kohakuterrarium.builtins.tool_catalog import get_builtin_tool
from kohakuterrarium.core.agent_native_tools import (
    NATIVE_TOOL_OPTIONS_STATE_SUFFIX,
    NativeToolOptions,
)
from kohakuterrarium.core.agent_tool_options import (
    TOOL_OPTIONS_STATE_SUFFIX,
    ToolOptions,
)
from kohakuterrarium.core.registry import Registry
from kohakuterrarium.modules.tool.base import ToolConfig, ToolInfo
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedModelSwitch:
    """Staged switch state. No provider is owned by the bundle."""

    llm: Any
    selector: str
    identifier: str
    registry_tools: dict[str, Any]
    registry_infos: dict[str, ToolInfo]
    executor_tools: dict[str, Any]
    auto_provider_tools: set[str]
    system_prompt: str | None
    option_state: dict[str, dict[str, Any]]

    def publish(self, agent: Any) -> None:
        """Assign prepared state. Does not build, validate, or emit events."""
        agent._llm_selector = self.selector
        agent._llm_identifier = self.identifier
        agent.llm = self.llm
        agent.controller.llm = self.llm
        agent.registry._tools = self.registry_tools
        agent.registry._tool_infos = self.registry_infos
        agent.executor._tools = self.executor_tools
        agent._auto_provider_tools = set(self.auto_provider_tools)
        for name, values in self.option_state.items():
            options = getattr(agent, name)
            for key, value in values.items():
                setattr(options, key, value)
        if self.system_prompt is not None:
            agent._controller_config.system_prompt = self.system_prompt
            agent.controller.config.system_prompt = self.system_prompt
            system = agent.controller.conversation.get_system_message()
            if system is not None:
                system.content = self.system_prompt
        subagent_manager = getattr(agent, "subagent_manager", None)
        if subagent_manager is not None:
            subagent_manager.llm = self.llm


def prepare(
    agent: Any,
    selector: str,
    new_llm: Any,
    identifier: str,
    persisted_options: dict[str, Any],
) -> PreparedModelSwitch:
    """Stage provider tools and prompt text from ``persisted_options``."""
    cache_key = getattr(agent.llm, "prompt_cache_key", None)
    if cache_key and hasattr(new_llm, "prompt_cache_key"):
        new_llm.prompt_cache_key = cache_key

    tools, infos, executor_tools, automatic, option_state = _prepare_tools(
        agent, new_llm, persisted_options or {}
    )
    return PreparedModelSwitch(
        llm=new_llm,
        selector=selector,
        identifier=identifier,
        registry_tools=tools,
        registry_infos=infos,
        executor_tools=executor_tools,
        auto_provider_tools=automatic,
        system_prompt=_prepare_prompt(agent, new_llm, tools, infos, executor_tools),
        option_state=option_state,
    )


def _prepare_tools(
    agent: Any, new_llm: Any, persisted_options: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, ToolInfo], dict[str, Any], set[str], dict]:
    provider = getattr(new_llm, "provider_name", "") or ""
    offered = set(getattr(new_llm, "provider_native_tools", ()) or ())
    offered.difference_update(getattr(agent.config, "disable_provider_tools", ()) or ())
    automatic = set(getattr(agent, "_auto_provider_tools", set()))
    tools = dict(agent.registry._tools)
    executor_tools = dict(agent.executor._tools)
    stale = automatic - offered
    for name, tool in tools.items():
        support = getattr(tool, "provider_support", frozenset())
        if getattr(tool, "is_provider_native", False) and provider not in support:
            stale.add(name)
    for name in stale:
        tools.pop(name, None)
        executor_tools.pop(name, None)
        automatic.discard(name)

    added_native = False
    added_local = False
    for name in sorted(offered):
        if name in tools:
            continue
        tool = get_builtin_tool(name)
        if tool is None:
            logger.warning(
                "provider_native_tool_not_in_catalog",
                tool_name=name,
                active_provider=provider,
            )
            continue
        tools[name] = tool
        executor_tools[name] = tool
        automatic.add(name)
        native = bool(getattr(tool, "is_provider_native", False))
        added_native = added_native or native
        added_local = added_local or not native
    option_state = {}
    if added_native or added_local:
        tools, executor_tools, option_state = _apply_options(
            agent,
            tools,
            executor_tools,
            persisted_options,
            native=added_native,
            local=added_local,
        )
    infos = {name: ToolInfo.from_tool(tool) for name, tool in tools.items()}
    return tools, infos, executor_tools, automatic, option_state


def _apply_options(
    agent: Any,
    tools: dict[str, Any],
    executor_tools: dict[str, Any],
    persisted_options: dict[str, Any],
    *,
    native: bool,
    local: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict]:
    """Refresh option copies and retain their detached publication state."""
    name = agent.config.name
    state = {
        f"{name}:{NATIVE_TOOL_OPTIONS_STATE_SUFFIX}": copy.deepcopy(
            persisted_options.get("native") or {}
        ),
        f"{name}:{TOOL_OPTIONS_STATE_SUFFIX}": copy.deepcopy(
            persisted_options.get("local") or {}
        ),
    }
    staging = SimpleNamespace(
        config=agent.config,
        registry=Registry(),
        session=SimpleNamespace(extra={}, scratchpad=None),
        _explicit_session=None,
        session_store=SimpleNamespace(state=state),
    )
    isolated = {tool_name: _isolate_tool(tool) for tool_name, tool in tools.items()}
    for tool in isolated.values():
        staging.registry.register_tool(tool)
    option_state = {}
    for enabled, attr, cls in (
        (native, "native_tool_options", NativeToolOptions),
        (local, "tool_options", ToolOptions),
    ):
        if not enabled:
            continue
        original = getattr(agent, attr, None)
        options = copy.copy(original) if original is not None else cls(staging)
        options._agent = staging
        keys = (
            ("_values",)
            if attr == "native_tool_options"
            else ("_values", "_pending_values", "_baselines")
        )
        for key in keys:
            setattr(options, key, copy.deepcopy(getattr(options, key)))
        options.apply()
        option_state[attr] = {key: getattr(options, key) for key in keys}
    for tool_name, original in tools.items():
        if _config_extra(isolated[tool_name]) == _config_extra(original):
            isolated[tool_name] = original
    mapped = {
        tool_name: isolated.get(tool_name, tool)
        for tool_name, tool in executor_tools.items()
    }
    return isolated, mapped, option_state


def _isolate_tool(tool: Any) -> Any:
    clone = copy.copy(tool)
    config = getattr(tool, "config", None)
    if isinstance(config, ToolConfig):
        clone.config = replace(
            config, env=dict(config.env), extra=copy.deepcopy(config.extra)
        )
    return clone


def _config_extra(tool: Any) -> Any:
    return getattr(getattr(tool, "config", None), "extra", None)


def _prepare_prompt(
    agent: Any,
    new_llm: Any,
    tools: dict[str, Any],
    infos: dict[str, ToolInfo],
    executor_tools: dict[str, Any],
) -> str | None:
    registry = Registry()
    registry._tools = tools
    registry._tool_infos = infos
    registry._subagents = dict(agent.registry._subagents)
    host = copy.copy(agent)
    host.llm = new_llm
    host.registry = registry
    host.executor = copy.copy(agent.executor)
    host.executor._tools = executor_tools
    if getattr(agent, "plugins", None) is not None:
        host.plugins = copy.copy(agent.plugins)
        if agent.plugins._load_context is not None:
            host.plugins._load_context = copy.copy(agent.plugins._load_context)
            host.plugins._load_context.model = getattr(new_llm, "model", "")
            host.plugins._load_context._host_agent = host
    return host._build_aggregated_prompt()
