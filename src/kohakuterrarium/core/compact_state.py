"""Owner-side compact input capture and validated live summary application."""

import time
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from kohakuterrarium.core.compact_branch import current_path
from kohakuterrarium.core.compact_splice import (
    is_real_user_message,
    prefix_fingerprint,
    select_compact_boundary,
)
from kohakuterrarium.core.compact_text import format_messages_for_summary
from kohakuterrarium.llm.generation import provider_identity


@dataclass(frozen=True)
class CompactInput:
    """Detached request values with owner-only identity guards for its source."""

    controller: Any
    conversation: Any
    controller_provider: tuple
    summary_provider: tuple
    tool_format: str | None
    store: Any
    branch: Any
    boundary: int
    retain_latest_user: bool
    live_turn_count: int
    expected_last: Any
    fingerprint: tuple
    context_length: int
    summary_input: str


def capture_compact(manager) -> CompactInput:
    """Capture summary text, boundary, and source guards in one owner callback."""
    controller = manager._controller
    conversation = controller.conversation
    messages = conversation.get_messages()
    boundary = manager._compact_boundary(messages)
    preferred = select_compact_boundary(messages, manager.config.keep_recent_turns)
    compact_messages = messages[1:boundary]
    live_count = manager.config.keep_recent_turns
    if boundary > preferred:
        live_count = sum(is_real_user_message(m) for m in messages[boundary:])
        live_count += int(any(is_real_user_message(m) for m in compact_messages))
    agent = manager._agent
    branch = (
        SimpleNamespace(
            **deepcopy(
                {
                    name: getattr(agent, name, None)
                    for name in (
                        "_turn_index",
                        "_branch_id",
                        "_parent_branch_path",
                        "_branch_view",
                    )
                }
            )
        )
        if agent is not None
        else None
    )
    return CompactInput(
        controller,
        conversation,
        provider_identity(getattr(controller, "llm", None)),
        provider_identity(manager._llm),
        getattr(getattr(controller, "config", None), "tool_format", None),
        manager._session_store,
        branch,
        boundary,
        boundary > preferred,
        live_count,
        compact_messages[-1] if compact_messages else None,
        prefix_fingerprint(compact_messages),
        conversation.get_context_length(),
        format_messages_for_summary(compact_messages),
    )


def compact_source_matches(manager, captured: CompactInput) -> bool:
    """Match source identity and retained ancestry without rejecting appended turns."""
    controller = manager._controller
    if controller is not captured.controller:
        return False
    if (
        provider_identity(getattr(controller, "llm", None))
        != captured.controller_provider
    ):
        return False
    if provider_identity(manager._llm) != captured.summary_provider:
        return False
    if (
        getattr(getattr(controller, "config", None), "tool_format", None)
        != captured.tool_format
    ):
        return False
    if manager._session_store is not captured.store:
        return False
    agent = manager._agent
    if getattr(agent, "session_store", captured.store) is not captured.store:
        return False
    path = current_path(captured.branch)
    view = getattr(agent, "_branch_view", None)
    if isinstance(view, dict) and any(
        view.get(turn, branch) != branch for turn, branch in path
    ):
        return False
    return path.issubset(current_path(agent))


def apply_compact(manager, captured: CompactInput, summary: str, elide) -> str | None:
    """Validate and apply one summary, its counters, usage reset, and elision."""
    if manager._controller.conversation is not captured.conversation:
        manager._last_compact_time = time.time()
        return "conversation_replaced"
    if not compact_source_matches(manager, captured):
        manager._last_compact_time = time.time()
        return "source_changed"
    conversation = captured.conversation
    original = conversation._messages
    messages = list(original)
    fields = [
        (message, deepcopy((message.content, message.metadata, message.tool_calls)))
        for message in messages
    ]
    metadata = deepcopy(conversation._metadata)
    count, cooldown = manager._compact_count, manager._last_compact_time
    usage = getattr(manager._controller, "_last_usage", None)
    try:
        applied = manager._splice_conversation(
            conversation,
            captured.boundary,
            summary,
            expected_last=captured.expected_last,
            expected_fingerprint=captured.fingerprint,
            retain_latest_user=captured.retain_latest_user,
        )
        if not applied:
            manager._last_compact_time = time.time()
            return "conversation_changed"
        manager._compact_count += 1
        manager._last_compact_time = time.time()
        manager._controller._last_usage = {}
        elide(manager._controller)
        return None
    except BaseException:
        original[:] = messages
        conversation._messages = original
        conversation._metadata = metadata
        for message, (content, message_metadata, tool_calls) in fields:
            message.content, message.metadata, message.tool_calls = (
                content,
                message_metadata,
                tool_calls,
            )
        manager._compact_count, manager._last_compact_time = count, cooldown
        manager._controller._last_usage = usage
        raise
