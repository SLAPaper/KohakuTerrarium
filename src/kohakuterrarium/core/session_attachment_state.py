"""In-memory rollback journal for explicit session attachment."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.modules.subagent.persistence import settle

_MISSING = object()
_OPTION_KEYS = ("native_tool_options", "tool_options", "plugin_options")
_LEGACY_KEY = "__native_tool_options__"


@dataclass
class AttachmentJournal:
    agent: Any
    references: list = field(default_factory=list)
    values: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    plugins: list = field(default_factory=list)
    children: list = field(default_factory=list)
    outputs: list = field(default_factory=list)

    def reference(self, obj, *names):
        if obj is not None:
            self.references.extend(
                (obj, name, getattr(obj, name, _MISSING)) for name in names
            )

    def value(self, obj, *names):
        if obj is not None:
            for name in names:
                value = getattr(obj, name, _MISSING)
                self.values.append(
                    (obj, name, deepcopy(value) if value is not _MISSING else value)
                )

    def entry(self, mapping, key):
        if isinstance(mapping, dict):
            value = mapping.get(key, _MISSING)
            self.entries.append(
                (mapping, key, deepcopy(value) if value is not _MISSING else value)
            )

    @classmethod
    def capture(cls, agent):
        journal = cls(agent)
        journal.reference(agent, "session_store", "_session_output")
        journal.reference(getattr(agent, "controller", None), "session_store")
        sink = getattr(agent, "_session_output", None)
        journal.reference(sink, "_capture_activity")
        router = getattr(agent, "output_router", None)
        journal.outputs = list(getattr(router, "_secondary_outputs", ()))
        for name in ("trigger_manager", "compact_manager", "subagent_manager"):
            manager = getattr(agent, name, None)
            journal.reference(manager, "_session_store", "_agent_name", "_parent_name")
        journal.value(
            getattr(agent, "compact_manager", None),
            "_compact_count",
            "_last_compact_time",
        )
        manager = getattr(agent, "subagent_manager", None)
        for job in getattr(manager, "_jobs", {}).values():
            child = job.subagent
            journal.children.append((child, child._run_receipt))
            journal.reference(
                child, "_session_store", "_parent_name", "_run_index", "_run_receipt"
            )
        for name in _OPTION_KEYS:
            journal.value(
                getattr(agent, name, None), "_values", "_pending_values", "_baselines"
            )
        registry = getattr(agent, "registry", None)
        if registry is not None:
            for name in registry.list_tools():
                tool = registry.get_tool(name)
                config = getattr(tool, "config", None)
                if config is not None:
                    journal.tools.append((tool, deepcopy(config.extra)))
        plugins = getattr(agent, "plugins", None)
        journal.reference(plugins, "_on_hook_timing")
        for plugin in getattr(plugins, "_plugins", ()):
            options = getattr(plugin, "options", _MISSING)
            journal.plugins.append(
                (plugin, deepcopy(options) if options is not _MISSING else options)
            )
        seen = set()
        for session in (
            getattr(agent, "session", None),
            getattr(agent, "_explicit_session", None),
        ):
            if session is None or id(session) in seen:
                continue
            seen.add(id(session))
            for key in _OPTION_KEYS:
                journal.entry(getattr(session, "extra", None), key)
            pad = getattr(session, "scratchpad", None)
            journal.reference(pad, "_on_write")
            journal.entry(getattr(pad, "_data", None), _LEGACY_KEY)
        return journal

    @staticmethod
    def _assign(obj, name, value):
        if value is _MISSING:
            if hasattr(obj, name):
                delattr(obj, name)
        else:
            setattr(obj, name, value)

    def restore_values(self):
        changed_tools, changed_plugins = [], []
        for obj, name, value in self.values:
            self._assign(obj, name, deepcopy(value) if value is not _MISSING else value)
        for tool, extra in self.tools:
            if tool.config.extra != extra:
                changed_tools.append(tool)
            tool.config.extra = deepcopy(extra)
        for plugin, options in self.plugins:
            if getattr(plugin, "options", _MISSING) != options:
                changed_plugins.append(plugin)
            self._assign(
                plugin,
                "options",
                deepcopy(options) if options is not _MISSING else options,
            )
        for mapping, key, value in self.entries:
            if value is _MISSING:
                mapping.pop(key, None)
            else:
                mapping[key] = deepcopy(value)
        return changed_tools, changed_plugins

    def restore_bindings(self):
        failed_sink = getattr(self.agent, "_session_output", None)
        for obj, name, value in self.references:
            self._assign(obj, name, value)
        router = getattr(self.agent, "output_router", None)
        if router is not None:
            additions = [
                output
                for output in router._secondary_outputs
                if output is not failed_sink
                and all(output is not prior for prior in self.outputs)
            ]
            router._secondary_outputs[:] = self.outputs + additions

    async def restore(self):
        try:
            for child, original_receipt in self.children:
                receipt = child._run_receipt
                if receipt is not None and receipt is not original_receipt:
                    await settle(receipt)
            tools, plugins = await submit_internal_session_state(
                self.agent, self.restore_values
            )
            for tool in tools:
                if getattr(tool, "is_provider_native", False):
                    refresh = getattr(tool, "refresh_native_options", None)
                    if callable(refresh):
                        refresh()
                else:
                    refresh = getattr(tool, "refresh_runtime_options", None)
                    if callable(refresh):
                        schema = tool.runtime_option_schema()
                        options = {
                            key: spec["default"]
                            for key, spec in schema.items()
                            if isinstance(spec, dict) and "default" in spec
                        }
                        options.update(
                            {
                                key: value
                                for key, value in tool.config.extra.items()
                                if key in schema
                            }
                        )
                        refresh(options)
            for plugin in plugins:
                plugin.refresh_options()
        finally:
            await submit_internal_session_state(self.agent, self.restore_bindings)


async def capture_attachment_state(agent):
    return await submit_internal_session_state(
        agent, lambda: AttachmentJournal.capture(agent)
    )
