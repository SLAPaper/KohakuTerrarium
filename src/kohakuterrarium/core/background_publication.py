"""Owner-authorized terminal publication, separate from conversation admission."""

import asyncio
from copy import deepcopy

from kohakuterrarium.core.background_origin import completion_identity
from kohakuterrarium.errors import InvalidRequestError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def _owner(agent, callback):
    owner = getattr(agent, "_session_state_owner", None)
    if owner is None or asyncio.current_task() is owner.task:
        return callback()
    return owner.submit_internal(callback)


async def _settle(value):
    if not asyncio.isfuture(value):
        return value
    while True:
        try:
            return await asyncio.shield(value)
        except asyncio.CancelledError:
            if value.cancelled():
                raise RuntimeError(
                    "Accepted publication operation was cancelled"
                ) from None


def _reserve(agent, identity, event, notify):
    ledger = getattr(agent, "_background_terminals", None)
    if ledger is None:
        ledger = agent._background_terminals = {}
    if identity in ledger:
        entry = ledger[identity]
        if entry["state"] == "uncertain":
            entry.update(state="reserved", retry=True)
            return entry
        return None
    entry = {
        "event": event,
        "notify": notify,
        "state": "reserved",
        "persisted": False,
        "error": None,
        "retry": False,
    }
    ledger[identity] = entry
    return entry


def _finish(entry, *, state, persisted=False, error=None):
    entry.update(state=state, persisted=persisted, error=error)


async def _publish(agent, reservation):
    entry = await _settle(reservation)
    if entry is None:
        return
    try:
        receipt = agent._publish_bg_complete(
            entry["event"], consume=False, retry=entry["retry"]
        )
        if receipt is not None:
            await _settle(asyncio.wrap_future(receipt))
        await _settle(
            _owner(
                agent,
                lambda: _finish(
                    entry, state="published", persisted=receipt is not None
                ),
            )
        )
    except Exception as exc:
        await _settle(
            _owner(agent, lambda: _finish(entry, state="uncertain", error=exc))
        )
        logger.error(
            "Background terminal publication unconfirmed",
            job_id=entry["event"].job_id,
            error=str(exc),
        )
        return
    if (
        entry["notify"]
        and agent._running
        and not getattr(agent, "_model_switch_closing", False)
    ):
        asyncio.create_task(agent._process_event(deepcopy(entry["event"])))


def schedule_background_terminal(agent, event, *, notify=None):
    """Capture and reserve a fully classified completion before external effects."""
    captured = deepcopy(event)
    context = captured.context or {}
    origin = context.get("origin")
    scope = origin.get("runtime_scope_id") if isinstance(origin, dict) else None
    identity = completion_identity(captured, scope)
    if identity is None or identity[1] is None:
        raise InvalidRequestError(
            "Background terminal requires full dispatch provenance"
        )
    if notify is None:
        notify = agent._should_notify_controller_on_background_complete(captured.job_id)
    reservation = _owner(agent, lambda: _reserve(agent, identity[1], captured, notify))
    pending = agent.output_router._terminal_publications

    def start():
        task = asyncio.create_task(_publish(agent, reservation))
        pending.add(task)
        task.add_done_callback(done)
        return task

    def done(task):
        pending.discard(task)
        if task.cancelled():
            start()
        elif task.exception() is not None:
            logger.error(
                "Background publication task failed", error=str(task.exception())
            )

    return start()
