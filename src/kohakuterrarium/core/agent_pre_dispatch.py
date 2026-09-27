"""Pre-tool-dispatch plugin chain.

Plugins can rewrite a ``ToolCallEvent`` between parser emission and
executor dispatch, or veto the call via ``PluginBlockError``.
"""

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from kohakuterrarium.core.events import TriggerEvent, create_tool_complete_event
from kohakuterrarium.core.controller_history import commit_controller_state
from kohakuterrarium.core.interrupted_history import (
    append_pending_result,
    pending_result_slot,
)
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.modules.plugin.base import (
    BasePlugin,
    PluginBlockError,
    PluginContext,
)
from kohakuterrarium.parsing import SubAgentCallEvent, ToolCallEvent
from kohakuterrarium.utils.logging import get_logger

if TYPE_CHECKING:
    from kohakuterrarium.core.agent import Agent
    from kohakuterrarium.core.controller import Controller

logger = get_logger(__name__)


async def run_pre_tool_dispatch(
    agent: "Agent", parse_event: ToolCallEvent, controller: "Controller"
) -> ToolCallEvent | None:
    """Run the ``pre_tool_dispatch`` plugin chain.

    Returns the (possibly rewritten) ToolCallEvent to dispatch, or
    ``None`` if the call was vetoed — in which case the block error
    is synthesised into the conversation as the tool result so the
    model sees a sensible next turn.
    """
    plugins = getattr(agent, "plugins", None)
    if plugins is None or not plugins._plugins:
        return parse_event

    applicable = plugins._applicable_plugins()
    base_method = getattr(BasePlugin, "pre_tool_dispatch", None)
    hook_plugins = [
        p
        for p in applicable
        if getattr(type(p), "pre_tool_dispatch", None) not in (None, base_method)
    ]
    if not hook_plugins:
        return parse_event

    original = deepcopy(parse_event)
    target = await _capture_native_target(controller, original)
    current = parse_event
    for plugin in hook_plugins:
        plugin_name = getattr(plugin, "name", "?")
        wd = (
            Path(getattr(agent.executor, "_working_dir", "."))
            if agent.executor
            else Path(".")
        )
        ctx = PluginContext(
            agent_name=agent.config.name,
            working_dir=wd,
            model=getattr(agent.llm, "model", ""),
            _host_agent=agent,
            _plugin_name=plugin_name,
        )
        try:
            rewritten = await plugin.pre_tool_dispatch(current, ctx)
        except PluginBlockError as block:
            logger.info(
                "Tool call vetoed by plugin",
                plugin_name=plugin_name,
                tool_name=current.name,
            )
            await _synthesize_blocked_tool_result(
                original, controller, str(block), plugin_name, target=target
            )
            return None
        except Exception as e:
            logger.warning(
                "pre_tool_dispatch raised",
                plugin_name=plugin_name,
                error=str(e),
                exc_info=True,
            )
            continue
        if rewritten is not None:
            if not isinstance(rewritten, ToolCallEvent):
                logger.warning(
                    "pre_tool_dispatch returned non-ToolCallEvent; ignoring",
                    plugin_name=plugin_name,
                    returned_type=type(rewritten).__name__,
                )
                continue
            current = rewritten

    # Verify the (possibly renamed) tool still resolves against the
    # registry — otherwise treat it as a veto with a descriptive
    # error so the model doesn't hit a generic "unknown tool".
    if current.name != parse_event.name:
        known = agent.registry.list_tools() if agent.registry else []
        if current.name not in known:
            logger.warning(
                "pre_tool_dispatch rewrote to unknown tool",
                original=parse_event.name,
                rewritten=current.name,
            )
            await _synthesize_blocked_tool_result(
                original,
                controller,
                f"unknown tool after rewrite: {current.name}",
                "pre_tool_dispatch",
                target=target,
            )
            return None
    return current


async def run_pre_subagent_dispatch(
    agent: "Agent", parse_event: SubAgentCallEvent, controller: "Controller"
) -> SubAgentCallEvent | None:
    """Run parent ``pre_subagent_run`` hooks before a child is spawned."""
    plugins = getattr(agent, "plugins", None)
    if plugins is None or not plugins._plugins:
        return parse_event

    original = deepcopy(parse_event)
    target = await _capture_native_target(controller, original)
    task = parse_event.args.get("task", parse_event.args.get("content", ""))
    try:
        updated_task = await plugins.run_pre_hooks(
            "pre_subagent_run",
            task,
            name=parse_event.name,
            is_background=parse_event.args.get("run_in_background", True),
        )
    except PluginBlockError as block:
        logger.info(
            "Sub-agent dispatch vetoed by plugin",
            subagent_name=parse_event.name,
            error=str(block),
        )
        await _synthesize_blocked_subagent_result(
            original, controller, str(block), target=target
        )
        return None
    if updated_task is not None and updated_task != task:
        args = dict(parse_event.args)
        if "task" in args or "content" not in args:
            args["task"] = updated_task
        else:
            args["content"] = updated_task
        return SubAgentCallEvent(name=parse_event.name, args=args, raw=parse_event.raw)
    return parse_event


async def _capture_native_target(controller, event):
    if (
        getattr(controller.config, "tool_format", None) != "native"
        or getattr(controller, "_state_host", None) is None
    ):
        return None

    def capture():
        conversation = controller.conversation
        job_id = f"blocked_{event.name}_{uuid4().hex}"
        context = capture_tool_context(
            controller._state_host, job_id, "direct", event.args.get("_tool_call_id")
        )
        announcement = conversation.get_last_assistant_message()
        calls = announcement.tool_calls or [] if announcement else []
        declaration = {"content": "", "tool_calls": deepcopy(calls)}
        if announcement and announcement.metadata.get("generation"):
            declaration["generation_id"] = announcement.metadata["generation"][
                "generation_id"
            ]
        declaration["veto_group_id"] = (
            f"{context['origin']['runtime_scope_id']}:{declaration.get('generation_id') or id(announcement)}"
        )
        return conversation, announcement, job_id, context, declaration

    pending = asyncio.create_task(commit_controller_state(controller, capture))
    cancelled = False
    while True:
        try:
            target = await asyncio.shield(pending)
            break
        except asyncio.CancelledError:
            if pending.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return target


def _veto_target_current(controller, target):
    return (
        controller.conversation is target[0]
        and target[0].get_last_assistant_message() is target[1]
    )


async def _finish_native_veto(controller, entry):
    target, metadata = entry["target"], entry["metadata"]
    host = controller._state_host
    try:
        if not entry["confirmed"]:
            receipt = host.output_router.notify_terminal_activity(
                "tool_error",
                f"[{metadata['tool_name']}] {entry['content']}",
                metadata,
            )
            if receipt is not None:
                future = asyncio.wrap_future(receipt)
                while True:
                    try:
                        await asyncio.shield(future)
                        break
                    except asyncio.CancelledError:
                        if future.cancelled():
                            raise RuntimeError(
                                "Native veto persistence receipt cancelled"
                            ) from None
            await commit_controller_state(
                controller, lambda: entry.update(confirmed=True)
            )

        def publish():
            if _veto_target_current(controller, target) and append_pending_result(
                host,
                target[2],
                entry["pair"],
                entry["content"],
            ):
                controller.push_event_sync(
                    TriggerEvent(type="tool_complete", content="")
                )
            entry["state"] = "finished"

        await commit_controller_state(controller, publish)
    except Exception as exc:
        await commit_controller_state(
            controller, lambda: entry.update(state="uncertain")
        )
        logger.error("Native veto publication failed", job_id=target[2], error=str(exc))
        return exc
    return None


async def _join_veto_operation(task):
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    if isinstance(result, Exception):
        raise result
    return result


async def _publish_native_veto(controller, event, content, target):
    if target is None:
        controller.conversation.append(
            "tool", content, tool_call_id=event.args["_tool_call_id"], name=event.name
        )
        controller.push_event_sync(TriggerEvent(type="tool_complete", content=""))
        return

    def reserve():
        if not _veto_target_current(controller, target):
            return None
        ledger = getattr(controller, "_native_veto_publications", None)
        if ledger is None:
            ledger = controller._native_veto_publications = {}
        key = (id(target[0]), id(target[1]), event.args["_tool_call_id"])
        entry = ledger.get(key)
        if entry is not None and entry["state"] != "uncertain":
            return entry
        if entry is None:
            pair = {
                "tool_call_id": event.args["_tool_call_id"],
                "name": event.name,
                "completion_context": target[3],
            }
            if not pending_result_slot(controller._state_host, target[2], pair):
                return None
            entry = {
                "target": target,
                "pair": pair,
                "content": content,
                "confirmed": False,
                "metadata": {
                    **target[3],
                    "job_id": target[2],
                    "tool_name": event.name,
                    "result": content,
                    "error": content,
                    "exit_code": 1,
                    "_native_veto_declaration": target[4],
                },
            }
            ledger[key] = entry
        else:
            entry["metadata"]["_native_veto_retry"] = True
        entry["state"] = "reserved"
        task = asyncio.create_task(_finish_native_veto(controller, entry))
        entry["task"] = task
        pending = controller._state_host.output_router._terminal_publications
        pending.add(task)
        task.add_done_callback(pending.discard)
        return entry

    async def accepted():
        entry = await commit_controller_state(controller, reserve)
        if entry is not None:
            return await asyncio.shield(entry["task"])

    await _join_veto_operation(asyncio.create_task(accepted()))


async def _synthesize_blocked_subagent_result(
    original_event: SubAgentCallEvent,
    controller: "Controller",
    message: str,
    *,
    target=None,
) -> None:
    """Inject synthetic feedback when a sub-agent dispatch is vetoed."""
    error_text = f"[{original_event.name}] {message}"
    if getattr(
        controller.config, "tool_format", None
    ) == "native" and original_event.args.get("_tool_call_id"):
        await _publish_native_veto(controller, original_event, error_text, target)
        return
    controller.push_event_sync(
        create_tool_complete_event(
            job_id=f"blocked_agent_{original_event.name}",
            content=error_text,
            exit_code=1,
            error=error_text,
        )
    )


async def _synthesize_blocked_tool_result(
    original_event: ToolCallEvent,
    controller: "Controller",
    message: str,
    plugin_name: str,
    *,
    target=None,
) -> None:
    """Inject a synthetic tool result when pre_tool_dispatch vetoes."""
    tool_call_id = original_event.args.get("_tool_call_id") or ""
    native_mode = getattr(controller.config, "tool_format", None) == "native"
    error_text = f"[{plugin_name}] {message}"
    if native_mode and tool_call_id:
        await _publish_native_veto(controller, original_event, error_text, target)
    else:
        controller.push_event_sync(
            create_tool_complete_event(
                job_id=f"blocked_{original_event.name}",
                content=error_text,
                exit_code=1,
                error=error_text,
            )
        )


__all__: tuple[str, ...] = ("run_pre_subagent_dispatch", "run_pre_tool_dispatch")
