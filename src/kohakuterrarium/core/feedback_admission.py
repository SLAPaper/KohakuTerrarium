"""Source-bound receipts for internally assembled text-tool feedback."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.background_origin import completion_identity
from kohakuterrarium.core.controller_generation import _canonical_output
from kohakuterrarium.core.tool_origin import tool_completion_context
from kohakuterrarium.session.feedback_admission import (
    persist_feedback,
    persist_injected_input,
)
from kohakuterrarium.session.output_handoff import output_binding
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackReceipt:
    host: Any
    controller: Any
    conversation: Any
    store: Any
    scope: Any
    events: tuple
    originals: tuple
    coordinates: tuple
    sink: Any
    namespace: str
    view: Any
    origin: list | None = None
    kind: str = "tool_results"


def arm_feedback(agent, event, job_ids):
    """Attach detached terminal facts to the exact internal queue occurrence."""
    deliveries = []
    for job_id in job_ids:
        claim = getattr(agent, "_job_terminal_claims", {}).get(job_id)
        context = tool_completion_context(agent, job_id)
        if (
            not claim
            or claim.get("kind") != "tool"
            or not claim.get("admitted")
            or not isinstance(context.get("origin"), dict)
        ):
            return
        deliveries.append({"job_id": job_id, "origin": deepcopy(context["origin"])})
    _arm(agent, event, deliveries)


def arm_background_feedback(agent, events):
    """Bind admitted queue occurrences to already confirmed publication facts."""
    active = getattr(agent, "_active_event_run", None) or []
    for event in events:
        if not any(envelope.event is event for envelope in active):
            continue
        identity = completion_identity(event, getattr(agent, "_runtime_scope_id", None))
        if identity is None or identity[1] is None:
            continue
        entry = getattr(agent, "_background_terminals", {}).get(identity[1])
        consumed = getattr(agent, "_background_consumption", {}).get(identity[1])
        if (
            not entry
            or entry["state"] != "published"
            or not entry["persisted"]
            or not consumed
            or not consumed[0]
        ):
            continue
        published = entry["event"]
        if (
            {
                key: value
                for key, value in published.context.items()
                if key != "pending_id"
            }
            != {
                key: value
                for key, value in event.context.items()
                if key != "pending_id"
            }
            or published.content != event.content
            or published.job_id != event.job_id
        ):
            raise ConflictError(
                "Background feedback differs from confirmed publication"
            )
        _arm(
            agent,
            event,
            [
                {
                    "job_id": event.job_id,
                    "origin": deepcopy(event.context["origin"]),
                    "mode": "background_terminal",
                    "tool_call_id": event.context.get("tool_call_id"),
                }
            ],
        )


def arm_persisted_input(agent, event, proof):
    """Associate a confirmed constituent with its current claimed occurrence."""
    if any(
        envelope.event is event
        for envelope in (getattr(agent, "_active_event_run", None) or [])
    ):
        _arm(agent, event, [deepcopy(proof)])


async def record_injected_constituent(controller, event, payload):
    """Persist and arm one injected occurrence on a stable output binding."""
    payload, input_type = deepcopy(payload), event.type
    host = getattr(controller, "_state_host", None)

    async def accepted():
        async with output_binding(_canonical_output(controller)):
            receipt = await submit_internal_session_state(
                host, lambda: capture_binding(controller, (event,))
            )
            if receipt is None:
                return False
            try:
                proof = await receipt.store.run(
                    persist_injected_input,
                    receipt.store,
                    receipt.namespace,
                    payload,
                    input_type,
                    *receipt.coordinates,
                )

                def install():
                    validate_feedback(receipt)
                    arm_persisted_input(host, event, proof)

                await submit_internal_session_state(host, install)
            except BaseException:
                await fence_feedback(controller)
                raise
            return True

    return await _join_accepted(asyncio.create_task(accepted()))


def capture_binding(controller, events):
    host = getattr(controller, "_state_host", None)
    sink = _canonical_output(controller)
    if host is None or sink is None:
        return None
    return FeedbackReceipt(
        host,
        controller,
        controller.conversation,
        controller.session_store,
        host._turn_completion,
        tuple(events),
        tuple(deepcopy(event.content) for event in events),
        deepcopy((host._turn_index, host._branch_id, host._parent_branch_path)),
        sink,
        sink._event_key_prefix,
        deepcopy(getattr(host, "_branch_view", None)),
    )


def arm_injected_inputs(agent, events, payloads):
    """Bind claimed user/trigger occurrences to their detached injected payloads."""
    active = getattr(agent, "_active_event_run", None) or []
    for event, payload in zip(events, payloads):
        if event.type in ("user_input", "trigger") and any(
            envelope.event is event for envelope in active
        ):
            _arm(
                agent,
                event,
                [
                    {
                        "mode": "injected_input",
                        "payload": deepcopy(payload),
                        "input_type": event.type,
                    }
                ],
            )


def _arm(agent, event, deliveries):
    candidate = (
        event,
        deepcopy(event.content),
        deliveries,
        agent._turn_completion,
    )
    pending = getattr(agent, "_text_feedback_admission", None) or {}
    existing = pending.get(id(event))
    if existing is not None and existing != candidate:
        raise ConflictError("Feedback occurrence was armed with different evidence")
    pending[id(event)] = candidate
    agent._text_feedback_admission = pending


def clear_feedback(agent, scope):
    outputs = getattr(agent, "_output_feedback_receipts", None)
    if outputs is not None:
        agent._output_feedback_receipts = {
            key: item for key, item in outputs.items() if item[3] is not scope
        } or None
    pending = getattr(agent, "_text_feedback_admission", None)
    if pending is not None:
        agent._text_feedback_admission = {
            key: item for key, item in pending.items() if item[3] is not scope
        } or None


def validate_feedback(receipt):
    host = receipt.host
    if (
        host.controller is not receipt.controller
        or receipt.controller.conversation is not receipt.conversation
        or host.session_store is not receipt.store
        or receipt.controller.session_store is not receipt.store
        or host._turn_completion is not receipt.scope
        or (receipt.scope is not None and receipt.scope.done())
        or _canonical_output(receipt.controller) is not receipt.sink
        or receipt.sink._event_key_prefix != receipt.namespace
        or host.config.name != receipt.namespace
        or getattr(host, "_branch_view", None) != receipt.view
        or (host._turn_index, host._branch_id, host._parent_branch_path)
        != receipt.coordinates
        or any(
            event.content != original
            for event, original in zip(receipt.events, receipt.originals)
        )
    ):
        host._session_attach_uncertain = True
        raise ConflictError("Text feedback admission changed")


async def _prepare(controller, events, content, skip_empty):
    host = getattr(controller, "_state_host", None)

    def capture():
        pending = getattr(host, "_text_feedback_admission", None)
        if pending is None:
            return None
        selected = [pending.pop(id(event)) for event in events if id(event) in pending]
        host._text_feedback_admission = pending or None
        if not events or len(events) != len(selected) or skip_empty:
            return None
        if any(
            scope is not host._turn_completion
            or event is not queued
            or event.content != original
            for event, (queued, original, _, scope) in zip(events, selected)
        ):
            return None
        deliveries = [proof for _, _, proofs, _ in selected for proof in proofs]
        receipt = capture_binding(controller, events)
        if receipt is None:
            return None
        sink = receipt.sink
        if any(
            proof.get("mode") in ("injected_input", "user_pair") for proof in deliveries
        ):
            receipt.kind = "input_group"
        return receipt, sink, deliveries

    captured = await submit_internal_session_state(host, capture)
    if captured is None:
        return None
    receipt, sink, deliveries = captured
    sink._flush_text_segment()
    future = sink._submit_store_write(
        persist_feedback,
        receipt.store,
        receipt.namespace,
        deepcopy(content),
        deliveries,
        *receipt.coordinates,
    )
    receipt.origin = await asyncio.wrap_future(future)
    return receipt


async def fence_feedback(controller):
    """Require cold recovery after an unconfirmed feedback write or live installation."""
    host = getattr(controller, "_state_host", None)
    if host is None:
        return
    owner = getattr(host, "_session_state_owner", None)
    if owner is None or asyncio.current_task() is owner.task:
        host._session_attach_uncertain = True
        return
    try:
        await _join_accepted(
            asyncio.create_task(
                submit_internal_session_state(
                    host, lambda: setattr(host, "_session_attach_uncertain", True)
                )
            )
        )
    except BaseException:
        logger.error("Feedback recovery fence acknowledgement failed", exc_info=True)


async def prepare_feedback(controller, events, content, skip_empty):
    task = asyncio.create_task(_prepare(controller, events, content, skip_empty))
    try:
        return await _join_accepted(task)
    except BaseException:
        if (
            task.done()
            and not task.cancelled()
            and (task.exception() is not None or task.result() is not None)
        ):
            await fence_feedback(controller)
        raise
