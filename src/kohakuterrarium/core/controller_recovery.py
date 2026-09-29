"""Request-stamped recovery commits and provider execution notifications."""

import asyncio
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator

from kohakuterrarium.core.compact_branch import current_path
from kohakuterrarium.core.compact_splice import prefix_fingerprint
from kohakuterrarium.core.conversation_elide import TOOL_FEEDBACK_KIND
from kohakuterrarium.core.conversation_sanitize import normalize_tool_names
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.base import LLMProvider
from kohakuterrarium.llm.generation import provider_identity
from kohakuterrarium.llm.message import create_message
from kohakuterrarium.llm.model_recovery_status import observe_model_recovery
from kohakuterrarium.llm.recovery import drop_last_tool_round
from kohakuterrarium.llm.recovery_scope import RecoveryScope, scoped_chat
from kohakuterrarium.modules.output.event import OutputEvent
from kohakuterrarium.modules.output.router import OutputRouter


@dataclass
class RecoveryCapture:
    controller: object
    conversation: object
    provider: object
    provider_key: tuple
    tool_format: str | None
    store: object
    path: set
    messages: list
    fingerprint: tuple
    dispatch_fingerprint: tuple


def _dispatch_fingerprint(messages):
    """Fingerprint request history except mutable system-prompt content."""
    result = []
    for message in messages:
        data = message.to_dict()
        if data.get("role") == "system":
            data.pop("content", None)
        result.append(repr(data))
    return tuple(result)


def capture_recovery(controller):
    """Capture canonical identities before wire transformations, on the owner."""
    host = getattr(controller, "_state_host", None)
    messages = controller.conversation.get_messages()
    return RecoveryCapture(
        controller,
        controller.conversation,
        controller.llm,
        provider_identity(controller.llm),
        controller.config.tool_format,
        getattr(host, "session_store", None),
        current_path(host),
        messages,
        prefix_fingerprint(messages),
        _dispatch_fingerprint(messages),
    )


def validate_source(capture, *, prefix=True, dispatch=False):
    controller = capture.controller
    host = getattr(controller, "_state_host", None)
    view = getattr(host, "_branch_view", None) or {}
    if (
        controller.conversation is not capture.conversation
        or controller.llm is not capture.provider
        or provider_identity(controller.llm) != capture.provider_key
        or controller.config.tool_format != capture.tool_format
        or getattr(host, "session_store", None) is not capture.store
        or not capture.path.issubset(current_path(host))
        or any(view.get(turn, branch) != branch for turn, branch in capture.path)
    ):
        raise ConflictError("Emergency recovery source changed")
    if prefix:
        live = controller.conversation.get_messages()
        fingerprint = _dispatch_fingerprint if dispatch else prefix_fingerprint
        expected = capture.dispatch_fingerprint if dispatch else capture.fingerprint
        if (
            len(live) < len(capture.messages)
            or any(a is not b for a, b in zip(live, capture.messages))
            or fingerprint(live[: len(capture.messages)]) != expected
        ):
            raise ConflictError("Emergency recovery conversation changed")


async def owner_recovery(controller, callback):
    host = getattr(controller, "_state_host", None)
    if host is None:
        return callback()
    owner = getattr(host, "_session_state_owner", None)
    if owner is None or not owner.is_running:
        raise ConflictError("Hosted recovery requires a running session-state owner")
    if asyncio.current_task() is owner.task:
        return callback()
    pending = owner.submit(callback)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(pending)
            break
        except asyncio.CancelledError:
            if pending.cancelled():
                raise
            cancelled = True
        except BaseException as exc:
            if cancelled:
                raise asyncio.CancelledError() from exc
            raise
    if cancelled:
        raise asyncio.CancelledError()
    return result


def _round(messages):
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")
        ),
        None,
    )
    if start is None:
        raise ConflictError("Emergency recovery has no native round")
    end = start + 1
    while end < len(messages) and messages[end].get("role") == "tool":
        end += 1
    return start, end


def apply_drop(capture, current, recovered):
    """Splice one uniquely proven native round without replacing surviving objects."""
    validate_source(capture)
    dropped, expected = drop_last_tool_round(current)
    if not dropped or expected != recovered:
        raise ConflictError("Emergency recovery is not the expected tool-round splice")
    wire_start, wire_end = _round(current)
    calls = current[wire_start]["tool_calls"]
    ids = [call.get("id") for call in calls]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ConflictError("Emergency recovery call identities are ambiguous")
    if any(
        m.get("tool_call_id") in ids
        or any(c.get("id") in ids for c in m.get("tool_calls") or [])
        for i, m in enumerate(current)
        if not wire_start <= i < wire_end
    ):
        raise ConflictError("Emergency recovery wire call identities are reused")
    canonical = normalize_tool_names([m.to_dict() for m in capture.messages])
    matches = [
        i
        for i, message in enumerate(canonical)
        if message.get("role") == "assistant" and message.get("tool_calls") == calls
    ]
    if len(matches) != 1:
        raise ConflictError(
            "Emergency recovery round cannot be mapped to canonical history"
        )
    start = matches[0]
    end = start + 1
    while end < len(canonical) and canonical[end].get("role") == "tool":
        end += 1
    results = [m.get("tool_call_id") for m in canonical[start + 1 : end]]
    if (
        results != [m.get("tool_call_id") for m in current[wire_start + 1 : wire_end]]
        or len(results) != len(ids)
        or set(results) != set(ids)
    ):
        raise ConflictError(
            "Emergency recovery requires the complete canonical call group"
        )
    if any(
        m.get("tool_call_id") in ids
        or any(c.get("id") in ids for c in m.get("tool_calls") or [])
        for i, m in enumerate(canonical)
        if not start <= i < end
    ):
        raise ConflictError("Emergency recovery call identities are reused")
    conversation = capture.conversation
    if any(
        getattr(m, "tool_call_id", None) in ids
        or any(c.get("id") in ids for c in getattr(m, "tool_calls", None) or [])
        for m in conversation._messages[len(capture.messages) :]
    ):
        raise ConflictError("Emergency recovery has newly appended results")
    placeholder = create_message("user", recovered[wire_start]["content"])
    placeholder.metadata["kind"] = TOOL_FEEDBACK_KIND
    replacement = (
        conversation._messages[:start] + [placeholder] + conversation._messages[end:]
    )
    original, metadata = conversation._messages, deepcopy(conversation._metadata)
    try:
        conversation._messages = replacement
        conversation._metadata.message_count = len(replacement)
        conversation._metadata.total_chars = conversation.get_context_length()
        conversation._metadata.updated_at = datetime.now()
    except BaseException:
        conversation._messages, conversation._metadata = original, metadata
        raise


async def recovery_chat(controller, messages, **kwargs):
    """Bind a canonical request stamp to the provider iterator and its retries."""
    capture = getattr(controller, "_recovery_capture", None)
    if capture is None:
        capture = await owner_recovery(controller, lambda: capture_recovery(controller))
    await owner_recovery(controller, lambda: validate_source(capture, dispatch=True))

    async def commit(current, recovered):
        current, recovered = deepcopy((current, recovered))
        await owner_recovery(
            controller, lambda: apply_drop(capture, current, recovered)
        )

    async def rescue(current):
        nonlocal capture
        host = getattr(controller, "_state_host", None)
        if host is None:
            return None
        manager = getattr(host, "compact_manager", None)
        if (
            manager is None
            or not manager.is_compacting
            or manager._compact_task is asyncio.current_task()
        ):
            return None
        await manager.wait_for_splice()

        def refresh():
            validate_source(capture, prefix=False)
            fresh = capture_recovery(controller)
            wire = deepcopy(controller.conversation.to_messages())
            return fresh, wire

        fresh, wire = await owner_recovery(controller, refresh)
        if not wire or len(wire) >= len(current):
            return None
        capture = fresh
        return wire

    scope = RecoveryScope(
        capture.provider, asyncio.current_task(), messages, commit, rescue
    )
    async with aclosing(
        chat_with_recovery(
            capture.provider,
            controller.output_router,
            messages,
            recovery_scope=scope,
            **kwargs,
        )
    ) as stream:
        async for chunk in stream:
            yield chunk


async def chat_with_recovery(
    llm: LLMProvider,
    output_router: OutputRouter | None,
    messages: list[dict],
    *,
    recovery_scope: RecoveryScope | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Observe only provider execution, never a consumer or inherited child task."""

    async def notify(payload: dict) -> None:
        if output_router is not None:
            await output_router.emit(
                OutputEvent(type="model_recovery", surface="status", payload=payload)
            )

    stream = (
        llm.chat(messages, **kwargs)
        if recovery_scope is None
        else scoped_chat(llm, messages, recovery_scope, **kwargs)
    )
    try:
        while True:
            with observe_model_recovery(notify):
                try:
                    chunk = await anext(stream)
                except StopAsyncIteration:
                    return
            yield chunk
    finally:
        with observe_model_recovery(notify):
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
