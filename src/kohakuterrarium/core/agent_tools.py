"""Agent tool execution mixin — handles tool dispatch, result collection, and background jobs."""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from kohakuterrarium.core.agent_runtime_tools import (
    AgentRuntimeToolsMixin,
    _make_job_label,
)
from kohakuterrarium.core.agent_tools_metrics import (
    emit_completion_metrics as _emit_completion_metrics,
)
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.backgroundify import BackgroundifyHandle, PromotionResult
from kohakuterrarium.core.interrupted_history import (
    append_pending_result,
    claim_job_terminal,
)
from kohakuterrarium.core.controller import Controller
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.core.job import JobResult, generate_job_id
from kohakuterrarium.core.native_ack import append_ack
from kohakuterrarium.core.feedback_dispatch import append_dispatch_ack
from kohakuterrarium.core.tool_origin import (
    capture_tool_context,
    promote_tool_context,
    tool_completion_context,
)
from kohakuterrarium.core.tool_output import render_content_text
from kohakuterrarium.llm.message import content_parts_to_dicts
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode
from kohakuterrarium.parsing import ToolCallEvent
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def _is_user_input(event: Any) -> bool:
    return getattr(event, "type", "") == "user_input"


class AgentToolsMixin(AgentRuntimeToolsMixin):
    """Provide tool execution, result handling, and background job management."""

    async def _start_tool_async(
        self,
        tool_call: ToolCallEvent,
        *,
        tool_call_id: str | None = None,
        run_bg: bool = False,
    ) -> tuple[str, asyncio.Task, bool]:
        """Start a tool asynchronously and return its ID, task, and wait mode."""
        try:
            logger.info("Running tool: %s", tool_call.name)
            tool = self.executor.get_tool(tool_call.name)
            is_direct = True
            if tool and isinstance(tool, BaseTool):
                is_direct = tool.execution_mode == ExecutionMode.DIRECT

            job_id = generate_job_id(tool_call.name)
            completion_context = capture_tool_context(
                self,
                job_id,
                "direct" if is_direct and not run_bg else "background",
                tool_call_id,
            )
            job_id = await self.executor.submit_from_event(
                tool_call,
                is_direct=is_direct,
                job_id=job_id,
                completion_context=completion_context,
            )
            task = self.executor.get_task(job_id)
            if task is None:

                async def _get_result():
                    return self.executor.get_result(job_id)

                task = asyncio.create_task(_get_result())

            return job_id, task, is_direct
        except Exception as e:
            logger.error("Failed to start tool", tool_name=tool_call.name, error=str(e))
            error_msg = str(e)
            error_job_id = f"error_{tool_call.name}"

            async def _error_result():
                return JobResult(job_id=error_job_id, error=error_msg)

            task = asyncio.create_task(_error_result())
            return error_job_id, task, True

    async def _wait_handles(
        self,
        handles: dict[str, BackgroundifyHandle],
        handle_order: list[str],
        controller: Controller,
        tool_call_ids: dict[str, str],
        native_mode: bool,
    ) -> tuple[dict[str, Any], bool]:
        """Collect direct results while processing background promotions.

        Queued user input must not wait out a long direct job: the only
        mid-turn drain runs AFTER this wait, so its arrival promotes the
        remaining handles — the round boundary (and the drain behind it)
        runs now, and each promoted job's real result returns through the
        background-completion fold in a later round. Background
        completions alone keep the natural boundary.
        """
        if not handles:
            return {}, False

        results: dict[str, Any] = {}
        had_promotions = False
        pending = dict(handles)
        waiters = {
            jid: asyncio.create_task(handle.wait()) for jid, handle in pending.items()
        }

        while waiters:
            if self._event_inbox.has_event(_is_user_input):
                for handle in pending.values():
                    if not handle.promoted:
                        handle.promote()
            arrival = asyncio.create_task(self._event_inbox.wait_put())
            try:
                done, _ = await asyncio.wait(
                    set(waiters.values()) | {arrival},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not arrival.done():
                    arrival.cancel()
                    try:
                        await arrival
                    except asyncio.CancelledError:
                        pass

            finished_job_ids = [jid for jid, task in waiters.items() if task in done]
            for jid in finished_job_ids:
                future = waiters.pop(jid)
                pending.pop(jid, None)
                try:
                    result = future.result()
                except (asyncio.CancelledError, Exception) as exc:
                    result = exc

                if isinstance(result, PromotionResult):
                    admitted = await self._handle_promotion(
                        jid, controller, tool_call_ids, native_mode
                    )
                    await submit_internal_session_state(
                        self, lambda: self._clear_direct_job_tracking(jid)
                    )
                    had_promotions = had_promotions or admitted
                else:
                    captured = deepcopy(result)

                    def commit():
                        if not claim_job_terminal(self, jid):
                            self._clear_direct_job_tracking(jid)
                            prior = self._job_terminal_claims[jid]
                            if prior is not None and not prior["feedback_taken"]:
                                feedback = deepcopy(prior["result"])
                                prior["feedback_taken"] = True
                                return None, prior["admitted"], feedback
                            return None
                        try:
                            meta = deepcopy(self._direct_job_meta.get(jid, {}))
                            meta["completion_context"] = tool_completion_context(
                                self, jid
                            ) or meta.get("completion_context", {})
                            accepted = True
                            if native_mode and tool_call_ids:
                                accepted = (
                                    jid
                                    in self._add_native_results_to_conversation(
                                        controller,
                                        [jid],
                                        {jid: captured},
                                        tool_call_ids,
                                        require_pending=True,
                                    )
                                )
                        except BaseException:
                            self._job_terminal_claims.pop(jid, None)
                            raise
                        self._job_terminal_claims[jid] = {
                            "result": captured,
                            "kind": meta.get("kind", "tool"),
                            "admitted": accepted,
                            "feedback_taken": True,
                        }
                        self._clear_direct_job_tracking(jid)
                        return meta, accepted, captured

                    meta = await submit_internal_session_state(self, commit)
                    if meta is not None:
                        meta, accepted, outcome = meta
                        if accepted:
                            results[jid] = outcome
                        else:
                            logger.warning(
                                "Direct result rejected by current native occurrence",
                                job_id=jid,
                            )
                        if meta is not None:
                            self._emit_direct_completion_activity(
                                jid, outcome, meta=meta
                            )

        return results, had_promotions

    async def _append_background_ack(self, job_id, tool_call_id, name, content):
        """Pair a background acknowledgement on its captured native occurrence."""
        admitted = await append_ack(self, job_id, tool_call_id, name, content)
        if not admitted:
            logger.warning(
                "Background acknowledgement rejected by current native occurrence",
                job_id=job_id,
            )
        return admitted

    async def _handle_promotion(
        self,
        job_id: str,
        controller: Controller,
        tool_call_ids: dict[str, str],
        native_mode: bool,
    ) -> bool:
        """Commit promotion state and acknowledgement before external notifications."""
        tool_name, label = _make_job_label(job_id)
        logger.info("Task promoted to background", job_id=job_id)
        tool_call_id = tool_call_ids.get(job_id)

        def commit():
            promote_tool_context(self, job_id)
            meta = self._direct_job_meta.get(job_id)
            if meta is not None:
                meta["background"] = True
                meta["interruptible"] = False
            if self._bg_controller_notify.get(job_id, True):
                self._turn_dispatched_bg.add(job_id)
            if not native_mode:
                append_dispatch_ack(
                    self,
                    job_id,
                    f"[{tool_name} → {job_id}] Promoted to background. Result arrives in a later turn.",
                )

        await submit_internal_session_state(self, commit)
        admitted = True
        if native_mode and tool_call_id:
            admitted = await append_ack(
                self,
                job_id,
                tool_call_id,
                tool_name,
                f"[{tool_name}] Promoted to background. Result arrives in a later turn.",
            )
        if not admitted:
            logger.warning(
                "Promotion acknowledgement rejected by current native occurrence",
                job_id=job_id,
            )

        if hasattr(self, "plugins") and self.plugins:
            asyncio.create_task(
                self.plugins.notify(
                    "on_task_promoted", job_id=job_id, tool_name=tool_name
                )
            )

        self.output_router.notify_activity(
            "task_promoted",
            f"[{label}] Moved to background",
            metadata={"job_id": job_id},
        )
        return admitted

    def _interrupt_direct_job(self, job_id: str) -> bool:
        """Cancel and finalize a direct job tracked by the current run."""
        meta = self._direct_job_meta.get(job_id)
        handle = self._active_handles.get(job_id)
        if not meta or not handle or handle.promoted or handle.done:
            return False
        if meta.get("kind") == "subagent":
            job = self.subagent_manager._jobs.get(job_id)
            if job and hasattr(job, "subagent"):
                job.subagent.cancel()
        handle.task.cancel()
        finalizers = getattr(self, "_interruption_tasks", None)
        if finalizers is None:
            finalizers = self._interruption_tasks = set()
        task = asyncio.create_task(self._finalize_interrupted_direct_job(job_id))
        finalizers.add(task)

        def completed(task):
            finalizers.discard(task)
            if task.cancelled():
                logger.error("Interrupted job finalizer cancelled", job_id=job_id)
            elif task.exception() is not None:
                logger.error(
                    "Interrupted job finalizer failed",
                    job_id=job_id,
                    error=str(task.exception()),
                )

        task.add_done_callback(completed)
        return True

    def _register_direct_job(
        self,
        job_id: str,
        *,
        kind: str,
        name: str,
        tool_call_id: str | None = None,
        notify_controller_on_background_complete: bool = True,
    ) -> None:
        """Track a direct job so interrupt/cancel can finalize it reliably."""
        self._direct_job_meta[job_id] = {
            "kind": kind,
            "name": name,
            "completion_context": tool_completion_context(self, job_id),
            "tool_call_id": tool_call_id or job_id,
            "background": False,
            "interruptible": True,
            "notify_controller_on_background_complete": notify_controller_on_background_complete,
            # Monotonic time avoids wall-clock adjustments corrupting durations.
            "started_at": time.monotonic(),
        }

    def _clear_direct_job_tracking(self, job_id: str) -> None:
        self._active_handles.pop(job_id, None)
        self._direct_job_meta.pop(job_id, None)
        self._bg_controller_notify.pop(job_id, None)

    def _emit_interrupted_activity(
        self, job_id: str, result: Any, *, meta=None
    ) -> None:
        """Emit terminal activity for an interrupted direct job."""
        if meta is None:
            meta = self._direct_job_meta.get(job_id, {})
        kind = meta.get("kind", "tool")
        _, label = _make_job_label(job_id)
        error = getattr(result, "error", None) or "User manually interrupted this job."
        activity = "subagent_error" if kind == "subagent" else "tool_error"
        # User interruption is a terminal outcome and must contribute to metrics.
        started_at = meta.get("started_at")
        duration_ms = (
            (time.monotonic() - started_at) * 1000.0
            if isinstance(started_at, (int, float))
            else 0.0
        )
        metric_name = meta.get("name") or job_id.rsplit("_", 1)[0]
        _emit_completion_metrics(
            kind == "subagent", metric_name, "interrupted", duration_ms
        )
        activity_meta: dict[str, Any] = {
            "job_id": job_id,
            "interrupted": True,
            "cancelled": False,
            "final_state": "interrupted",
            "error": error,
        }
        if kind == "subagent":
            activity_meta["result"] = getattr(result, "output", "") or error
            activity_meta["turns"] = getattr(result, "turns", 0)
            activity_meta["duration"] = getattr(result, "duration", 0)
            activity_meta["total_tokens"] = getattr(result, "total_tokens", 0)
            activity_meta["prompt_tokens"] = getattr(result, "prompt_tokens", 0)
            activity_meta["completion_tokens"] = getattr(result, "completion_tokens", 0)
            activity_meta["cached_tokens"] = getattr(result, "cached_tokens", 0)
            activity_meta["tools_used"] = getattr(result, "metadata", {}).get(
                "tools_used", []
            )
        activity_meta.update(
            tool_completion_context(self, job_id) or meta.get("completion_context", {})
        )
        self.output_router.notify_activity(
            activity,
            f"[{label}] INTERRUPTED: {error}",
            metadata=activity_meta,
        )

    async def _finalize_interrupted_direct_job(self, job_id: str) -> None:
        """Wait for cancellation to settle, then emit a terminal interrupted event."""
        handle = self._active_handles.get(job_id)
        meta = self._direct_job_meta.get(job_id)
        if not handle or not meta:
            return

        try:
            result = await asyncio.shield(handle.task)
        except asyncio.CancelledError:
            result = None

        if result is None:
            kind = meta.get("kind", "tool")
            if kind == "subagent":
                result = self.subagent_manager.get_result(job_id)
            else:
                result = self.executor.get_result(job_id)

        if result is None:
            result = JobResult(
                job_id=job_id,
                error="User manually interrupted this job.",
                metadata={"interrupted": True, "final_state": "interrupted"},
            )

        feedback = JobResult(
            job_id=job_id,
            output=deepcopy(getattr(result, "output", "")),
            error=getattr(result, "error", None)
            or "User manually interrupted this job.",
            metadata={"interrupted": True, "final_state": "interrupted"},
        )

        def commit():
            if not claim_job_terminal(self, job_id):
                self._clear_direct_job_tracking(job_id)
                return False
            try:
                paired = self._append_interrupted_tool_message(job_id, meta, result)
            except BaseException:
                self._job_terminal_claims.pop(job_id, None)
                raise
            native = (
                getattr(getattr(self.controller, "config", None), "tool_format", None)
                == "native"
            )
            self._job_terminal_claims[job_id] = {
                "result": feedback,
                "admitted": paired or not native,
                "feedback_taken": False,
            }
            self._clear_direct_job_tracking(job_id)
            return True

        if await submit_internal_session_state(self, commit):
            self._emit_interrupted_activity(job_id, result, meta=meta)

    def _append_interrupted_tool_message(
        self, job_id: str, meta: dict[str, Any], result: Any
    ) -> bool:
        """Add the tool response required for an interrupted direct job.

        Interrupting ``_processing_task`` can prevent the normal result path from
        pairing the assistant's tool call with a tool response. Providers reject
        that incomplete conversation on the next request.

        Existing responses are preserved because cancellation can race with the
        normal result path.
        """
        error_text = getattr(result, "error", None) or "User cancelled this job."
        return append_pending_result(self, job_id, meta, f"Interrupted: {error_text}")

    async def _on_backgroundify_complete(
        self, job_id: str, result: Any, *, completion_context: dict | None = None
    ) -> None:
        """Callback when a promoted (backgroundified) task completes.

        Builds a TriggerEvent and reuses the existing ``_on_bg_complete``
        path for activity notification and event processing.
        """
        promote_tool_context(self, job_id)
        # ``CancelledError`` is a ``BaseException`` and must precede the generic
        # arm so cancellation is reported as an interrupt rather than an error.
        if isinstance(result, asyncio.CancelledError):
            extra_context: dict[str, Any] = {
                "interrupted": True,
                "final_state": "interrupted",
            }
            event = create_tool_complete_event(
                job_id=job_id,
                content="",
                error="User manually interrupted this job.",
                **extra_context,
            )
        elif isinstance(result, BaseException):
            extra_context = {}
            event = create_tool_complete_event(
                job_id=job_id, content="", error=str(result), **extra_context
            )
        elif hasattr(result, "output"):
            extra_context = {
                "interrupted": bool(getattr(result, "interrupted", False)),
                "cancelled": bool(getattr(result, "cancelled", False)),
            }
            if extra_context["interrupted"]:
                extra_context["final_state"] = "interrupted"
            elif extra_context["cancelled"]:
                extra_context["final_state"] = "cancelled"
            event = create_tool_complete_event(
                job_id=job_id,
                content=result.output or "",
                exit_code=getattr(result, "exit_code", 0),
                error=result.error if hasattr(result, "error") else None,
                result_metadata=dict(getattr(result, "metadata", {}) or {}),
                **extra_context,
            )
            if hasattr(result, "turns"):
                if event.context is None:
                    event.context = {}
                result_metadata = getattr(result, "metadata", {})
                event.context["subagent_metadata"] = {
                    "tools_used": result_metadata.get("tools_used", []),
                    "subagent": result_metadata.get("subagent", ""),
                    "llm_name": result_metadata.get("llm_name", ""),
                    "model": result_metadata.get("model", ""),
                    "turns": result.turns,
                    "duration": getattr(result, "duration", 0),
                    "total_tokens": getattr(result, "total_tokens", 0),
                    "prompt_tokens": getattr(result, "prompt_tokens", 0),
                    "completion_tokens": getattr(result, "completion_tokens", 0),
                    "cached_tokens": getattr(result, "cached_tokens", 0),
                    "interrupted": bool(getattr(result, "interrupted", False)),
                    "cancelled": bool(getattr(result, "cancelled", False)),
                }
        else:
            event = create_tool_complete_event(
                job_id=job_id, content=str(result) if result else ""
            )

        context = tool_completion_context(self, job_id) or deepcopy(
            completion_context or {}
        )
        if isinstance(context.get("origin"), dict):
            context["execution_class"] = "background"
            context["origin"]["execution_class"] = "background"
        event.context.update(context)
        self._on_bg_complete(event)

    def _emit_direct_completion_activity(
        self, job_id: str, result: Any, *, meta=None
    ) -> None:
        """Emit terminal activity immediately when a direct job finishes."""
        # The post-stop sweep already emitted a terminal state for this job.
        if not self._running and meta is None:
            return
        # Termination checkers need the recent result tail to evaluate stopping.
        checker = getattr(self, "_termination_checker", None)
        if checker is not None and hasattr(checker, "record_tool_result"):
            try:
                checker.record_tool_result(result)
            except Exception:  # pragma: no cover — defensive
                pass
        if meta is None:
            meta = self._direct_job_meta.get(job_id, {})
        completion = meta.get("completion_context") or tool_completion_context(
            self, job_id
        )
        kind = meta.get("kind", "subagent" if job_id.startswith("agent_") else "tool")
        _, label = _make_job_label(job_id)
        is_subagent = kind == "subagent"
        done_activity = "subagent_done" if is_subagent else "tool_done"
        error_activity = "subagent_error" if is_subagent else "tool_error"

        # Missing background metadata still contributes a count with zero duration.
        # Use the registry name because job-ID suffixes would create unbounded labels.
        started_at = meta.get("started_at")
        duration_ms = (
            (time.monotonic() - started_at) * 1000.0
            if isinstance(started_at, (int, float))
            else 0.0
        )
        metric_name = meta.get("name") or job_id.rsplit("_", 1)[0]

        if isinstance(result, (asyncio.CancelledError, Exception)):
            interrupted = isinstance(result, asyncio.CancelledError)
            error_text = (
                "User manually interrupted this job." if interrupted else str(result)
            )
            metadata: dict[str, Any] = {
                "job_id": job_id,
                "interrupted": interrupted,
                "cancelled": False,
                "final_state": "interrupted" if interrupted else "error",
                "error": error_text,
            }
            if is_subagent:
                metadata["result"] = error_text
                existing = self.subagent_manager.get_result(job_id)
                if existing is not None:
                    metadata["turns"] = getattr(existing, "turns", 0)
                    metadata["duration"] = getattr(existing, "duration", 0)
                    metadata["total_tokens"] = getattr(existing, "total_tokens", 0)
                    metadata["prompt_tokens"] = getattr(existing, "prompt_tokens", 0)
                    metadata["completion_tokens"] = getattr(
                        existing, "completion_tokens", 0
                    )
                    metadata["cached_tokens"] = getattr(existing, "cached_tokens", 0)
                    metadata["tools_used"] = getattr(existing, "metadata", {}).get(
                        "tools_used", []
                    )
            self.output_router.notify_activity(
                error_activity,
                f"[{label}] {'INTERRUPTED' if interrupted else 'FAILED'}: {error_text}",
                metadata={**metadata, **completion},
            )
            status = "interrupted" if interrupted else "error"
            _emit_completion_metrics(is_subagent, metric_name, status, duration_ms)
            return

        if result is not None and hasattr(result, "error") and result.error:
            output = render_content_text(result.output or "")
            interrupted = bool(getattr(result, "interrupted", False))
            cancelled = bool(getattr(result, "cancelled", False))
            final_state = (
                "interrupted" if interrupted else "cancelled" if cancelled else "error"
            )
            metadata = {
                "job_id": job_id,
                "interrupted": interrupted,
                "cancelled": cancelled,
                "final_state": final_state,
                "error": result.error,
                "result": output,
            }
            if is_subagent:
                metadata["turns"] = getattr(result, "turns", 0)
                metadata["duration"] = getattr(result, "duration", 0)
                metadata["total_tokens"] = getattr(result, "total_tokens", 0)
                metadata["prompt_tokens"] = getattr(result, "prompt_tokens", 0)
                metadata["completion_tokens"] = getattr(result, "completion_tokens", 0)
                metadata["cached_tokens"] = getattr(result, "cached_tokens", 0)
                metadata["tools_used"] = getattr(result, "metadata", {}).get(
                    "tools_used", []
                )
            state_label = (
                "INTERRUPTED" if interrupted else "CANCELLED" if cancelled else "ERROR"
            )
            self.output_router.notify_activity(
                error_activity,
                f"[{label}] {state_label}: {result.error}",
                metadata={**metadata, **completion},
            )
            _emit_completion_metrics(is_subagent, metric_name, final_state, duration_ms)
            return

        output = (
            result.output
            if hasattr(result, "output")
            else str(result) if result else ""
        )
        output = output or ""
        exit_code = getattr(result, "exit_code", 0)
        # ``exit_code=None`` is valid for tools that do not launch a process, so
        # prefer the result's canonical success signal when available.
        succeeded = result.success if hasattr(result, "success") else exit_code == 0
        status = "OK" if succeeded else f"exit={exit_code}"
        preview = (
            result.get_text_output()[:5000]
            if hasattr(result, "get_text_output")
            else str(output)[:5000]
        )
        metadata = {
            "job_id": job_id,
            # Full output for the session event log (recoverable via search_memory);
            # preview stays for UI rendering.
            "output": output,
            "output_preview": preview,
            "exit_code": exit_code,
        }
        if not is_subagent and isinstance(output, list):
            metadata["result"] = content_parts_to_dicts(output)
        if not succeeded:
            metadata["error"] = f"Tool exited with code {exit_code}"
            metadata["final_state"] = "error"
        if is_subagent:
            result_metadata = getattr(result, "metadata", {})
            metadata["result"] = preview
            metadata["subagent"] = result_metadata.get("subagent", metric_name)
            metadata["llm_name"] = result_metadata.get("llm_name", "")
            metadata["model"] = result_metadata.get("model", "")
            metadata["turns"] = getattr(result, "turns", 0)
            metadata["duration"] = getattr(result, "duration", 0)
            metadata["total_tokens"] = getattr(result, "total_tokens", 0)
            metadata["prompt_tokens"] = getattr(result, "prompt_tokens", 0)
            metadata["completion_tokens"] = getattr(result, "completion_tokens", 0)
            metadata["cached_tokens"] = getattr(result, "cached_tokens", 0)
            metadata["tools_used"] = getattr(result, "metadata", {}).get(
                "tools_used", []
            )
        # Preserve preview metadata so event consumers can render modified files.
        tool_metadata = getattr(result, "metadata", None) or {}
        canvas_preview = tool_metadata.get("canvas_preview")
        if canvas_preview:
            metadata["canvas_preview"] = canvas_preview
        session_metadata = tool_metadata.get("session_metadata")
        if isinstance(session_metadata, dict):
            metadata["tool_metadata"] = dict(session_metadata)
        self.output_router.notify_activity(
            done_activity if succeeded else error_activity,
            f"[{label}] {status}",
            metadata={**metadata, **completion},
        )
        _emit_completion_metrics(
            is_subagent,
            metric_name,
            "ok" if succeeded else "error",
            duration_ms,
        )

    def _add_native_results_to_conversation(
        self,
        controller: Controller,
        handle_order: list[str],
        results: dict[str, Any],
        tool_call_ids: dict[str, str],
        *,
        require_pending: bool = False,
    ) -> set[str]:
        """Add completed results as role='tool' messages (native mode)."""
        added = set()
        for job_id in handle_order:
            if job_id not in results:
                continue  # Promoted jobs already have a protocol placeholder.

            result = results[job_id]
            tool_name, _ = _make_job_label(job_id)
            tool_call_id = tool_call_ids.get(job_id, job_id)

            if isinstance(result, (asyncio.CancelledError, Exception)):
                interrupted = isinstance(result, asyncio.CancelledError)
                error_text = (
                    "User manually interrupted this job."
                    if interrupted
                    else str(result)
                )
                prefix = "Interrupted" if interrupted else "Error"
                content = f"{prefix}: {error_text}"
            elif result is not None and hasattr(result, "error") and result.error:
                output = render_content_text(result.output or "")
                interrupted = bool(getattr(result, "interrupted", False))
                cancelled = bool(getattr(result, "cancelled", False))
                prefix = (
                    "Interrupted"
                    if interrupted
                    else "Cancelled" if cancelled else "Error"
                )
                content = f"{prefix}: {result.error}"
                if output:
                    content += f"\n{output}"
            elif result is not None:
                content = result.output if hasattr(result, "output") else str(result)
                content = content or ""
            else:
                content = ""

            if require_pending:
                if append_pending_result(
                    self,
                    job_id,
                    {"tool_call_id": tool_call_id, "name": tool_name},
                    content,
                ):
                    added.add(job_id)
            else:
                existing = controller.conversation.get_messages()
                index = next(
                    (
                        i
                        for i in reversed(range(len(existing)))
                        if any(
                            call.get("id") == tool_call_id
                            for call in (existing[i].tool_calls or [])
                        )
                    ),
                    None,
                )
                if index is not None and any(
                    m.role == "tool" and m.tool_call_id == tool_call_id
                    for m in existing[index + 1 :]
                ):
                    continue
                controller.conversation.append(
                    "tool", content, tool_call_id=tool_call_id, name=tool_name
                )
                added.add(job_id)
        return added

    def _format_text_results(
        self,
        handle_order: list[str],
        results: dict[str, Any],
    ) -> str:
        """Format completed results as text feedback (non-native mode)."""
        result_strs: list[str] = []
        for job_id in handle_order:
            if job_id not in results:
                continue

            result = results[job_id]

            if isinstance(result, (asyncio.CancelledError, Exception)):
                interrupted = isinstance(result, asyncio.CancelledError)
                error_text = (
                    "User manually interrupted this job."
                    if interrupted
                    else str(result)
                )
                result_strs.append(
                    f"## {job_id} - {'INTERRUPTED' if interrupted else 'FAILED'}\n{error_text}"
                )
            elif result is not None:
                output = render_content_text(
                    result.output if hasattr(result, "output") else str(result)
                )
                error = getattr(result, "error", None)
                if error:
                    interrupted = bool(getattr(result, "interrupted", False))
                    cancelled = bool(getattr(result, "cancelled", False))
                    state = (
                        "INTERRUPTED"
                        if interrupted
                        else "CANCELLED" if cancelled else "ERROR"
                    )
                    result_strs.append(f"## {job_id} - {state}\n{error}\n{output}")
                else:
                    exit_code = getattr(result, "exit_code", 0)
                    status = "OK" if exit_code == 0 else f"exit={exit_code}"
                    result_strs.append(f"## {job_id} - {status}\n{output}")

        return "\n\n".join(result_strs) if result_strs else ""


@dataclass(slots=True)
class _TurnResult:
    """Results from a single LLM turn, used internally by the controller loop."""

    handles: dict[str, BackgroundifyHandle] = field(default_factory=dict)
    handle_order: list[str] = field(default_factory=list)
    text_output: list[str] = field(default_factory=list)
    native_mode: bool = False
    native_tool_call_ids: dict[str, str] = field(default_factory=dict)
