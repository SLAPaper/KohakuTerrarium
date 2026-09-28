"""Confirm summary persistence before announcing a completed compaction."""

import asyncio

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.compact_branch import persist_compact_counters
from kohakuterrarium.core.compact_state import apply_compact
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.session.output_handoff import output_binding
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class CompactCommitError(RuntimeError):
    """An applied summary has unconfirmed persistence and requires cold recovery."""


async def commit_summary(manager, captured, summary, metadata, coverage, elide):
    """Join one source-bound owner apply and durable event/counter write."""
    token, agent = object(), manager._agent
    applied = confirmed = False
    payload = {**metadata, **coverage, "summary": summary}

    def apply():
        nonlocal applied
        if manager._agent is not agent:
            return "source_changed"
        require_model_mutation_available(agent)
        reason = apply_compact(manager, captured, summary, elide, coverage)
        if reason is None:
            applied = True
            if agent is not None:
                agent._compact_commit_pending = token
        return reason

    def confirm():
        nonlocal confirmed
        if agent is not None:
            if getattr(agent, "_compact_commit_pending", None) is not token:
                raise CompactCommitError(
                    "Compaction confirmation requires cold recovery"
                )
            agent._compact_commit_pending = None
        confirmed = True

    async with output_binding(captured.sink):
        try:
            reason = await submit_internal_session_state(agent, apply)
            if reason is not None:
                return reason
            if captured.sink is not None:
                receipt = captured.sink.record_compact_complete(payload)
                await asyncio.wrap_future(receipt)
            run = getattr(captured.store, "run", None)
            if callable(run):
                await run(
                    persist_compact_counters,
                    captured.store,
                    manager._agent_name,
                    manager._compact_count,
                    manager._last_compact_time,
                )
            await submit_internal_session_state(agent, confirm)
        except (Exception, asyncio.CancelledError) as exc:
            if not applied:
                raise
            if not confirmed:
                if agent is not None:
                    try:
                        await submit_internal_session_state(
                            agent,
                            lambda: setattr(agent, "_session_attach_uncertain", True),
                        )
                    except (Exception, asyncio.CancelledError):
                        logger.error(
                            "Compact recovery fence was not confirmed", exc_info=True
                        )
                raise CompactCommitError(
                    f"Compaction persistence requires cold recovery: {exc}"
                ) from exc
        manager._splice_ready.set()
        if manager._output_router:
            notify = (
                manager._output_router.notify_confirmed_activity
                if captured.sink is not None
                else manager._output_router.notify_activity
            )
            notify(
                "compact_complete",
                f"Context auto-compact done (round {manager._compact_count})",
                payload,
            )
        return None
