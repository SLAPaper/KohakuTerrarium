"""Agent lifecycle helpers.

Centralizes warm-pause and orderly shutdown behavior for agents.
"""

import asyncio
from typing import Any

from kohakuterrarium.core.job import JobState, JobType
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.core.interrupted_history import (
    append_pending_result,
    claim_job_terminal,
)
from kohakuterrarium.core.background_publication import schedule_background_terminal
from kohakuterrarium.core.tool_origin import (
    promote_tool_context,
    tool_completion_context,
)
from kohakuterrarium.core.branch_rerun import drain_history_reruns
from kohakuterrarium.core.branch_selection import (
    drain_history_selections,
    mark_history_selection_stopping,
)
from kohakuterrarium.core.agent_state import (
    close_owner_admission,
    close_session_state_owner,
    submit_internal_session_state,
)
from kohakuterrarium.core.model_switch_runtime import (
    close_retired_model_providers,
    drain_model_switches,
    mark_model_switch_closing,
)
from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


async def wait_for_turn_completion(
    agent: Any, completion: asyncio.Future | None, *, timeout: float = 10.0
) -> None:
    """Join one captured turn without cancelling it when the waiter times out."""
    if completion is None or completion.done():
        return
    if getattr(agent, "_turn_lock_holder", None) is asyncio.current_task():
        raise ConflictError("Cannot wait for the current turn from its own callback")
    try:
        await asyncio.wait_for(asyncio.shield(completion), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise ConflictError(
            "The previous turn is still finishing; retry after it stops"
        ) from exc


class AgentLifecycleMixin:
    """Provide warm-pause and orderly shutdown behavior for an agent."""

    plugins: Any
    compact_manager: Any
    llm: Any
    output_router: Any
    subagent_manager: Any
    trigger_manager: Any
    input: Any
    config: Any
    _running: bool
    _paused: bool
    _shutdown_event: Any

    @property
    def is_processing(self) -> bool:
        """Whether a turn is active, finalizing, or waiting in the inbox."""
        lock = getattr(self, "_processing_lock", None)
        inbox = getattr(self, "_event_inbox", None)
        return bool(
            (lock is not None and lock.locked()) or (inbox is not None and len(inbox))
        )

    async def interrupt_and_wait(self, *, timeout: float = 10.0) -> None:
        """Interrupt the captured turn and wait until its finalization releases it."""
        completion = getattr(self, "_turn_completion", None)
        self.interrupt()
        await wait_for_turn_completion(self, completion, timeout=timeout)

    @property
    def paused(self) -> bool:
        """Return whether the agent is warm-paused and rejecting new turns."""
        return getattr(self, "_paused", False)

    def pause(self) -> None:
        """Idempotently pause admission and triggers while queued events remain pending."""
        if getattr(self, "_paused", False):
            return
        self._paused = True
        self._consumer_resume.clear()
        self.trigger_manager.suspend_all()
        logger.info("Agent paused", agent_name=self.config.name)

    def resume(self) -> None:
        """Idempotently resume triggers and drain events queued during the pause."""
        if not getattr(self, "_paused", False):
            return
        self._paused = False
        self.trigger_manager.resume_all()
        self._consumer_resume.set()
        self._event_inbox.wake()
        logger.info("Agent resumed", agent_name=self.config.name)

    async def stop(self) -> None:
        """Stop all agent modules once, serializing concurrent callers."""
        if getattr(self, "_starting", False):
            raise ConflictError(
                "Cannot stop while the agent is starting; cancel or await start first"
            )
        mark_model_switch_closing(self)
        mark_history_selection_stopping(self)
        close_owner_admission(self)
        stop_lock = getattr(self, "_stop_lock", None)
        if stop_lock is None:
            stop_lock = self._stop_lock = asyncio.Lock()
        async with stop_lock:
            if getattr(self, "_stopped", False):
                return
            await drain_model_switches(self)
            await drain_history_selections(self)
            await drain_history_reruns(self)
            await self._stop_inner()
            self._stopped = True

    async def _stop_inner(self) -> None:
        logger.info("Stopping agent", agent_name=self.config.name)

        if self.plugins:
            await self.plugins.notify("on_agent_stop")
            await self.plugins.unload_all()

        drain_terminals = getattr(
            self.output_router, "drain_terminal_publications", None
        )
        if callable(drain_terminals):
            await drain_terminals()
        await self._finalize_inflight_jobs_for_stop()
        self._running = False
        self._shutdown_event.set()

        consumer_resume = getattr(self, "_consumer_resume", None)
        if consumer_resume is not None:
            consumer_resume.set()
        inbox = getattr(self, "_event_inbox", None)
        if inbox is not None:
            inbox.wake()

        processing = getattr(self, "_processing_task", None)
        if (
            processing is not None
            and not processing.done()
            and processing is not asyncio.current_task()
        ):
            processing.cancel()
            await asyncio.gather(processing, return_exceptions=True)
        turn_lock = getattr(self, "_processing_lock", None)
        if (
            turn_lock is not None
            and getattr(self, "_turn_lock_holder", None) is not asyncio.current_task()
        ):
            try:
                await asyncio.wait_for(turn_lock.acquire(), timeout=10)
                turn_lock.release()
            except asyncio.TimeoutError:  # pragma: no cover - defensive
                logger.warning(
                    "Timed out waiting for turn finalization during stop",
                    agent_name=self.config.name,
                )

        consumer = getattr(self, "_consumer_task", None)
        if (
            consumer is not None
            and not consumer.done()
            and consumer is not asyncio.current_task()
        ):
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

        if hasattr(self, "_mcp_manager") and self._mcp_manager:
            await self._mcp_manager.shutdown()

        await self._cancel_executor_tasks()
        await self.subagent_manager.cancel_all()
        finalizers = getattr(self, "_interruption_tasks", set())
        while finalizers:
            batch = tuple(finalizers)
            await asyncio.shield(asyncio.gather(*batch, return_exceptions=True))
            finalizers.difference_update(batch)
        controller = getattr(self, "controller", None)
        cleanup_inline_files = getattr(controller, "cleanup_inline_files", None)
        if callable(cleanup_inline_files):
            cleanup_inline_files()
        await self.trigger_manager.stop_all()
        await self.input.stop()
        if self.compact_manager:
            await self.compact_manager.cancel()
        if callable(drain_terminals):
            await drain_terminals()
        await close_session_state_owner(self)
        await self.output_router.stop()
        await close_retired_model_providers(self)
        compact_llm = (
            getattr(self.compact_manager, "_llm", None)
            if self.compact_manager
            else None
        )
        if (
            compact_llm is not None
            and compact_llm is not self.llm
            and hasattr(compact_llm, "close")
        ):
            await compact_llm.close()
        await self.llm.close()

    async def _finalize_inflight_jobs_for_stop(self) -> None:
        """Commit interrupted job state on the owner, then publish detached terminals."""
        router = getattr(self, "output_router", None)
        if router is None or not hasattr(router, "notify_activity"):
            return
        if not any(
            source.get_running_jobs()
            for source in (
                getattr(self, "executor", None),
                getattr(self, "subagent_manager", None),
            )
            if source is not None and hasattr(source, "get_running_jobs")
        ):
            return
        plans = await submit_internal_session_state(self, self._prepare_stop_jobs)
        for is_subagent, name, metadata in plans:
            if metadata.get("execution_class") == "background":
                context = {
                    key: value
                    for key, value in metadata.items()
                    if key not in ("job_id", "result")
                }
                event = create_tool_complete_event(
                    job_id=metadata["job_id"],
                    content=metadata.get("result", ""),
                    **context,
                )
                try:
                    schedule_background_terminal(self, event, notify=False)
                except InvalidRequestError as exc:
                    logger.error(
                        "Stop terminal rejected",
                        job_id=metadata["job_id"],
                        error=str(exc),
                    )
            else:
                router.notify_activity(
                    "subagent_error" if is_subagent else "tool_error",
                    f"[{name}] INTERRUPTED: {metadata['error']}",
                    metadata=metadata,
                )

    def _prepare_stop_jobs(self):
        """Capture stop outcomes and pair only current, pending native occurrences."""
        plans = []
        jobs: list[tuple[Any, Any]] = []
        executor = getattr(self, "executor", None)
        if executor is not None and hasattr(executor, "get_running_jobs"):
            store = getattr(executor, "job_store", None)
            jobs.extend((status, store) for status in executor.get_running_jobs())
        manager = getattr(self, "subagent_manager", None)
        if manager is not None and hasattr(manager, "get_running_jobs"):
            store = getattr(manager, "job_store", None)
            jobs.extend((status, store) for status in manager.get_running_jobs())
        seen: set[str] = set()
        for status, store in jobs:
            job_id = getattr(status, "job_id", None)
            if not job_id or job_id in seen or not claim_job_terminal(self, job_id):
                continue
            seen.add(job_id)
            is_subagent = getattr(status, "job_type", None) == JobType.SUBAGENT
            name = getattr(status, "type_name", "") or job_id
            error = "Agent stopped while this job was still running."
            if store is not None and hasattr(store, "update_status"):
                store.update_status(job_id, state=JobState.CANCELLED, error=error)
            metadata: dict[str, Any] = {
                "job_id": job_id,
                "interrupted": True,
                "cancelled": False,
                "final_state": "interrupted",
                "error": error,
            }
            if is_subagent:
                metadata["result"] = error
            meta = getattr(self, "_direct_job_meta", {}).get(job_id) or {}
            if meta.get("tool_call_id"):
                append_pending_result(self, job_id, meta, error)
            handle = getattr(self, "_active_handles", {}).get(job_id)
            if handle is not None and handle.promoted:
                promote_tool_context(self, job_id)
            metadata.update(tool_completion_context(self, job_id))
            plans.append((is_subagent, name, metadata))
        return plans

    async def _cancel_executor_tasks(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is None:
            return
        tasks = [task for task in executor._tasks.values() if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
