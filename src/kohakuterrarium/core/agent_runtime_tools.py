import asyncio
from functools import partial

from kohakuterrarium.core.backgroundify import BackgroundifyHandle
from kohakuterrarium.core.background_publication import schedule_background_terminal
from kohakuterrarium.core.controller import Controller
from kohakuterrarium.core.events import TriggerEvent
from kohakuterrarium.core.job import generate_job_id
from kohakuterrarium.core.job_label import make_job_label
from kohakuterrarium.core.tool_output import render_content_text
from kohakuterrarium.core.tool_origin import (
    capture_tool_context,
    tool_completion_context,
)
from kohakuterrarium.errors import InvalidRequestError
from kohakuterrarium.llm.message import content_parts_to_dicts
from kohakuterrarium.modules.subagent.manager import subagent_background_requested
from kohakuterrarium.parsing import CommandResultEvent, SubAgentCallEvent, ToolCallEvent
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


_make_job_label = make_job_label


class AgentRuntimeToolsMixin:
    def _should_notify_controller_on_background_complete(self, job_id: str) -> bool:
        meta = self._direct_job_meta.get(job_id)
        if meta and "notify_controller_on_background_complete" in meta:
            return bool(meta["notify_controller_on_background_complete"])
        return bool(self._bg_controller_notify.get(job_id, True))

    def _notify_command_result(self, parse_event: CommandResultEvent) -> None:
        """Route command results to activity log (not user-facing output)."""
        activity = "command_error" if parse_event.error else "command_done"
        detail = (
            f"[{parse_event.command}] {parse_event.error}"
            if parse_event.error
            else f"[{parse_event.command}] OK"
        )
        self.output_router.notify_activity(activity, detail)

    def _notify_tool_start(
        self,
        parse_event: ToolCallEvent,
        job_id: str,
        is_direct: bool,
        *,
        tool_call_id: str | None = None,
    ) -> None:
        """Notify output of a tool start with a human-readable preview."""
        _, label = _make_job_label(job_id)
        full_args, arg_parts = {}, []
        for k, v in (parse_event.args or {}).items():
            if k.startswith("_"):
                continue
            full_args[k] = v
            arg_parts.append(f"{k}={str(v)[:40]}")
        bg_tag = " (bg)" if not is_direct else ""
        metadata = {
            "job_id": job_id,
            "tool_name": parse_event.name,
            "args": full_args,
            "background": not is_direct,
        }
        metadata.update(tool_completion_context(self, job_id))
        if tool_call_id:
            metadata.update(
                tool_call_id=tool_call_id, tool_call_arguments=parse_event.raw
            )
        self.output_router.notify_activity(
            "tool_start",
            f"[{label}]{bg_tag} {' '.join(arg_parts)[:80]}",
            metadata=metadata,
        )

    def _emit_token_usage(self, controller: Controller) -> None:
        """Emit and accumulate usage from the last LLM turn.

        The accumulated usage supports one turn-level event, while provider
        cache counts are emitted separately when available.
        """
        usage = getattr(controller, "_last_usage", {})
        if not usage:
            return
        self.output_router.notify_activity(
            "token_usage",
            f"tokens: {usage.get('prompt_tokens', 0)} in, "
            f"{usage.get('completion_tokens', 0)} out",
            metadata=usage,
        )
        accum = getattr(self, "_turn_usage_accum", None)
        if isinstance(accum, dict):
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "total_tokens",
            ):
                accum[k] = accum.get(k, 0) + int(usage.get(k, 0) or 0)

        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(
            usage.get("cached_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or 0
        )
        if cache_write or cache_read:
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            ratio = (cache_read / prompt) if prompt else 0.0
            self.output_router.notify_activity(
                "cache_stats",
                f"cache: r={cache_read} w={cache_write}",
                metadata={
                    "agent": getattr(self.config, "name", ""),
                    "cache_write": cache_write,
                    "cache_read": cache_read,
                    "cache_hit_ratio": ratio,
                },
            )

    def _cancel_handles(self, handles: dict[str, BackgroundifyHandle]) -> None:
        """Cancel all non-promoted handles (on interrupt)."""
        for job_id, handle in handles.items():
            if handle.promoted:
                logger.debug("Skipping promoted handle", job_id=job_id)
                continue
            if not handle.done:
                handle.task.cancel()
                logger.debug("Cancelled direct handle", job_id=job_id)

    def _reset_output_state(self) -> None:
        """Reset output router and default output for a new iteration."""
        self.output_router.reset()
        if hasattr(self.output_router.default_output, "reset"):
            self.output_router.default_output.reset()

    async def _flush_output(self) -> None:
        """Flush buffered output and reset default output."""
        await self.output_router.flush()
        if hasattr(self.output_router.default_output, "reset"):
            self.output_router.default_output.reset()

    async def _start_subagent_async(
        self, event: SubAgentCallEvent, *, tool_call_id: str | None = None
    ) -> tuple[str, bool]:
        """Start a sub-agent execution."""
        logger.info(
            "Starting sub-agent",
            subagent_type=event.name,
            task=event.args.get("task", "")[:50],
        )
        job_id = generate_job_id(f"agent_{event.name}")
        context = capture_tool_context(self, job_id, "direct", tool_call_id)
        if tool_call_id:
            context["tool_call_arguments"] = event.raw
        is_background = False
        try:
            is_background = subagent_background_requested(event.args)
            if is_background:
                context["execution_class"] = context["origin"]["execution_class"] = (
                    "background"
                )
            return await self.subagent_manager.spawn_from_event(
                event, job_id=job_id, completion_context=context
            )
        except Exception as e:
            logger.error(
                "Sub-agent dispatch rejected", subagent_name=event.name, error=str(e)
            )
            self.subagent_manager.record_rejection(event.name, job_id, str(e), context)
            return job_id, is_background

    def _on_bg_complete(self, event: TriggerEvent) -> None:
        """Handle background tool/sub-agent completion."""
        context = event.context or {}
        classified = any(key in context for key in ("origin", "execution_class"))
        if not self._running:
            owner = getattr(self, "_session_state_owner", None)
            if (
                classified
                and getattr(self, "_model_switch_closing", False)
                and owner is not None
                and owner.task is not None
                and not owner.task.done()
            ):
                return self._admit_bg_terminal(event, notify=False)
            return
        if classified:
            return self._admit_bg_terminal(
                event,
                notify=False if getattr(self, "_model_switch_closing", False) else None,
            )
        return self._publish_bg_complete(event)

    def _admit_bg_terminal(self, event, *, notify=None):
        try:
            return schedule_background_terminal(self, event, notify=notify)
        except InvalidRequestError as exc:
            logger.error(
                "Background terminal rejected", job_id=event.job_id, error=str(exc)
            )
            return None

    def _publish_bg_complete(self, event: TriggerEvent, *, consume=True, retry=False):
        """Emit a detached outcome and return its canonical persistence receipt."""
        emit = (
            self.output_router.notify_activity
            if consume
            else partial(self.output_router.notify_terminal_activity, retry=retry)
        )
        receipt = None
        job_id = getattr(event, "job_id", "")
        is_subagent = job_id.startswith("agent_")
        context = event.context or {}
        origin_metadata = {
            key: context[key]
            for key in ("origin", "execution_class", "tool_call_id")
            if key in context
        }
        error = context.get("error")
        exit_code = context.get("exit_code")
        try:
            failed_exit = exit_code is not None and int(exit_code) != 0
        except (TypeError, ValueError):
            failed_exit = False
        if failed_exit and not error:
            error = f"Tool exited with code {exit_code}"
        content = render_content_text(event.content)
        _, label = _make_job_label(job_id)
        activity_done, activity_error = (
            ("subagent_done", "subagent_error")
            if is_subagent
            else ("tool_done", "tool_error")
        )
        sa_meta = event.context.get("subagent_metadata", {}) if event.context else {}
        tools_used = sa_meta.get("tools_used", [])

        if error:
            interrupted = (
                bool(event.context.get("interrupted")) if event.context else False
            )
            cancelled = bool(event.context.get("cancelled")) if event.context else False
            final_state = (
                "interrupted" if interrupted else "cancelled" if cancelled else "error"
            )
            state_label = (
                "INTERRUPTED" if interrupted else "CANCELLED" if cancelled else "ERROR"
            )
            receipt = emit(
                activity_error,
                f"[{label}] {state_label}: {error}",
                metadata={
                    "job_id": job_id,
                    **origin_metadata,
                    "subagent": sa_meta.get("subagent", ""),
                    "llm_name": sa_meta.get("llm_name", ""),
                    "model": sa_meta.get("model", ""),
                    "error": error,
                    "exit_code": exit_code,
                    "interrupted": interrupted,
                    "cancelled": cancelled,
                    "final_state": final_state,
                    "result": content,
                    "tools_used": tools_used,
                    "turns": sa_meta.get("turns", 0),
                    "duration": sa_meta.get("duration", 0),
                    "total_tokens": sa_meta.get("total_tokens", 0),
                    "prompt_tokens": sa_meta.get("prompt_tokens", 0),
                    "completion_tokens": sa_meta.get("completion_tokens", 0),
                    "cached_tokens": sa_meta.get("cached_tokens", 0),
                },
            )
        elif is_subagent:
            tools_summary = ", ".join(tools_used[:10]) if tools_used else "none"
            receipt = emit(
                activity_done,
                f"[{label}] tools: {tools_summary}",
                metadata={
                    "job_id": job_id,
                    **origin_metadata,
                    "subagent": sa_meta.get("subagent", ""),
                    "llm_name": sa_meta.get("llm_name", ""),
                    "model": sa_meta.get("model", ""),
                    "tools_used": tools_used,
                    "result": content,
                    "turns": sa_meta.get("turns", 0),
                    "duration": sa_meta.get("duration", 0),
                    "total_tokens": sa_meta.get("total_tokens", 0),
                    "prompt_tokens": sa_meta.get("prompt_tokens", 0),
                    "completion_tokens": sa_meta.get("completion_tokens", 0),
                    "cached_tokens": sa_meta.get("cached_tokens", 0),
                },
            )
        else:
            result_metadata = context.get("result_metadata", {})
            structured_result = (
                content_parts_to_dicts(event.content)
                if isinstance(event.content, list)
                else content
            )
            metadata = {
                "job_id": job_id,
                "result": structured_result,
                "output": event.content,
                "output_preview": content[:5000],
                "exit_code": exit_code,
            }
            if isinstance(result_metadata, dict):
                canvas_preview = result_metadata.get("canvas_preview")
                if canvas_preview:
                    metadata["canvas_preview"] = canvas_preview
                session_metadata = result_metadata.get("session_metadata")
                if isinstance(session_metadata, dict):
                    metadata["tool_metadata"] = dict(session_metadata)
            receipt = emit(
                activity_done,
                f"[{label}] DONE",
                metadata={**metadata, **origin_metadata},
            )

        logger.info("Background job completed", job_id=job_id)
        if not consume:
            return receipt
        if self._should_notify_controller_on_background_complete(job_id):
            asyncio.create_task(self._process_event(event))
        else:
            self._bg_controller_notify.pop(job_id, None)
