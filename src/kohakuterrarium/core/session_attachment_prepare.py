"""Detached destination state and nonpersisting explicit attachment installation."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kohakuterrarium.core.agent_observability import (
    wire_plugin_hook_timing,
    wire_scratchpad_observer,
)
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import strict_option_refresh
from kohakuterrarium.modules.plugin.option_validation import validate_plugin_options
from kohakuterrarium.modules.subagent.persistence import settle
from kohakuterrarium.modules.tool.runtime_options import validate_tool_options
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.text_buffer import last_persisted_turn_branch

_KEYS = (
    "native_tool_options",
    "tool_options",
    "plugin_options",
    "compact_count",
    "last_compact_time",
)


def _read_destination(store, name):
    values = {key: deepcopy(store.state.get(f"{name}:{key}")) for key in _KEYS}
    text = store.state.get(f"{name}:open_text")
    text = text if isinstance(text, str) else ""
    coordinates = (
        last_persisted_turn_branch(store, name) if text else (None, None, None)
    )
    return values, (text, *coordinates)


def _mapping(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Session options must be a mapping")
    return deepcopy(value)


def _persist(store, name, values):
    for key, value in values.items():
        store.state[f"{name}:{key}"] = value
    store.flush()


@dataclass
class PreparedAttachment:
    agent: Any
    source: Any
    store: Any
    sink: Any
    values: dict
    native: dict
    tools: dict
    pending: dict
    baselines: dict
    tool_configs: list
    plugin_configs: list
    children: list
    compact_count: int
    compact_time: float
    recovery: tuple

    def stage(self):
        if self.sink is None or self.sink._store is not self.store:
            self.sink = SessionOutput(
                self.agent.config.name,
                self.store,
                self.agent,
                prepared_recovery=self.recovery,
            )
        for index, (child, _) in enumerate(self.children):
            if child._run_index is None and child._run_receipt is None:
                receipt = self.store.submit(
                    self.store.next_subagent_run,
                    self.agent.config.name,
                    child.config.name,
                )
                self.children[index] = (child, receipt)

    async def drain(self):
        error = None
        receipts = [receipt for _, receipt in self.children if receipt is not None]
        if self.sink is not None and self.sink._store is self.store:
            receipts.extend(self.sink._pending_writes)
        for receipt in receipts:
            try:
                await settle(receipt)
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def install(self):
        agent = self.agent
        if agent.session_store is not self.source or not agent._session_attach_pending:
            raise ConflictError("Prepared attachment source or reservation changed")
        agent.session_store = self.store
        agent.controller.session_store = self.store
        if agent._session_output is not self.sink:
            outputs = agent.output_router._secondary_outputs
            if agent._session_output in outputs:
                outputs.remove(agent._session_output)
            outputs.append(self.sink)
            agent._session_output = self.sink
        manager = agent.subagent_manager
        manager._session_store, manager._parent_name = self.store, agent.config.name
        for child, receipt in self.children:
            child._session_store, child._parent_name = self.store, agent.config.name
            if receipt is not None:
                child._run_receipt = receipt
        agent.trigger_manager._session_store = self.store
        agent.trigger_manager._agent_name = agent.config.name
        if agent.compact_manager is not None:
            compact = agent.compact_manager
            compact._session_store = self.store
            compact._compact_count, compact._last_compact_time = (
                self.compact_count,
                self.compact_time,
            )
        agent.native_tool_options._values = deepcopy(self.native)
        agent.tool_options._values = deepcopy(self.tools)
        agent.tool_options._pending_values = deepcopy(self.pending)
        agent.tool_options._baselines = deepcopy(self.baselines)
        agent.plugin_options._values = deepcopy(self.values["plugin_options"])
        for tool, config, _ in self.tool_configs:
            tool.config.extra = deepcopy(config)
        for plugin, options in self.plugin_configs:
            plugin.options = deepcopy(options)
        extra = getattr(agent.session, "extra", None)
        if isinstance(extra, dict):
            for key in _KEYS[:3]:
                if self.values[key]:
                    extra[key] = deepcopy(self.values[key])
                else:
                    extra.pop(key, None)
        pad = getattr(agent.session, "scratchpad", None)
        if pad is not None and self.source is None:
            pad._data.pop("__native_tool_options__", None)

    def refresh(self):
        for tool, _, effective in self.tool_configs:
            if getattr(tool, "is_provider_native", False):
                callback = getattr(tool, "refresh_native_options", None)
                if callable(callback):
                    callback()
            else:
                callback = getattr(tool, "refresh_runtime_options", None)
                if callable(callback):
                    callback(deepcopy(effective))
        with strict_option_refresh():
            for plugin, options in self.plugin_configs:
                plugin.set_options(deepcopy(options))
        wire_scratchpad_observer(self.agent)
        wire_plugin_hook_timing(self.agent)

    async def persist(self):
        await settle(
            self.store.submit(
                _persist, self.store, self.agent.config.name, deepcopy(self.values)
            )
        )
        await self.drain()
        await self.sink.drain()


async def prepare_attachment(agent, store):
    source, name = agent.session_store, agent.config.name
    values, recovery = await store.run(_read_destination, store, name)
    extra = getattr(agent.session, "extra", {})
    if source is None:
        for key in _KEYS[:3]:
            if values[key] is None:
                values[key] = deepcopy(extra.get(key, {}))
    native = _mapping(values["native_tool_options"])
    if source is None:
        pad = getattr(agent.session, "scratchpad", None)
        legacy = pad.get("__native_tool_options__") if pad is not None else None
        if legacy:
            native.update(_mapping(legacy))
    ordinary = _mapping(values["tool_options"])
    plugin_values = _mapping(values["plugin_options"])
    native = {
        key: agent.native_tool_options._validate(key, value)
        for key, value in native.items()
    }
    baselines = deepcopy(agent.tool_options._baselines)
    tool_configs, tools, pending = [], {}, {}
    for tool_name in agent.registry.list_tools():
        tool = agent.registry.get_tool(tool_name)
        if getattr(tool, "is_provider_native", False):
            overrides = native.get(tool_name, {})
            tool_configs.append((tool, overrides, overrides))
            continue
        schema = agent.tool_options._schema(tool)
        if not schema:
            continue
        baseline = baselines.setdefault(tool_name, deepcopy(tool.config.extra))
        effective = {
            key: spec["default"]
            for key, spec in schema.items()
            if isinstance(spec, dict) and "default" in spec
        }
        effective.update(
            {key: value for key, value in baseline.items() if key in schema}
        )
        overrides = ordinary.get(tool_name, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Invalid tool options for {tool_name}")
        try:
            candidate = validate_tool_options(
                tool_name, {**effective, **overrides}, schema
            )
        except ValueError:
            available = {
                key: {k: v for k, v in spec.items() if k != "disabled_values"}
                for key, spec in schema.items()
            }
            validate_tool_options(tool_name, {**effective, **overrides}, available)
            pending[tool_name] = deepcopy(overrides)
            candidate, overrides = effective, {}
        if overrides:
            tools[tool_name] = deepcopy(overrides)
        tool_configs.append((tool, {**baseline, **overrides}, candidate))
    plugins = []
    normalized_plugins = {}
    for plugin in agent.plugins._plugins:
        baseline = deepcopy(getattr(plugin, "_session_option_baseline", {}))
        incoming = plugin_values.get(plugin.name, {})
        cleaned = validate_plugin_options(
            plugin.name, incoming, type(plugin).option_schema()
        )
        options = {**baseline, **cleaned}
        plugins.append((plugin, options))
        if cleaned:
            normalized_plugins[plugin.name] = options
    count = int(values["compact_count"] or 0)
    cooldown = float(values["last_compact_time"] or 0)
    if source is None:
        if values["compact_count"] is None and agent.compact_manager:
            count = agent.compact_manager._compact_count
        if values["last_compact_time"] is None and agent.compact_manager:
            cooldown = agent.compact_manager._last_compact_time
    values = dict(
        native_tool_options=native,
        tool_options={**pending, **tools},
        plugin_options=normalized_plugins,
        compact_count=count,
        last_compact_time=cooldown,
    )
    children = [(job.subagent, None) for job in agent.subagent_manager._jobs.values()]
    return PreparedAttachment(
        agent,
        source,
        store,
        agent._session_output,
        values,
        native,
        tools,
        pending,
        baselines,
        tool_configs,
        plugins,
        children,
        count,
        cooldown,
        recovery,
    )


async def install_attachment(prepared):
    prepared.stage()
    await submit_internal_session_state(prepared.agent, prepared.install)
    prepared.refresh()
    await prepared.persist()
