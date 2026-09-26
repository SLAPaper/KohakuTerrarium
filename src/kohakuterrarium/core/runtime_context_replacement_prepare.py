"""Prepare complete replacement contexts without installing them."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from kohakuterrarium.core.runtime_context import (
    RuntimeContextPreparation,
    RuntimeContextPreparationError,
    RuntimeContextPreparationMode,
)
from kohakuterrarium.core.selected_subagent_history import (
    project_selected_subagents,
    restore_text_subagent_feedback,
)
from kohakuterrarium.llm.message import Message, dicts_to_messages
from kohakuterrarium.session.generation_validation import (
    validate_selected_generation_events,
)
from kohakuterrarium.session.history_feedback import (
    ToolIdentityResolver,
    canonicalize_provider_tool_order,
)
from kohakuterrarium.session.history import (
    InvalidBranchViewError,
    normalize_tool_call_events,
    replay_conversation,
    resolve_branch_view_strict,
    select_live_event_ids,
)

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _baseline_messages(value: object) -> tuple[Message, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, Message) or item.role != "system" for item in value
    ):
        raise RuntimeContextPreparationError(
            "baseline_system_messages must be a finite sequence of system Message"
        )
    return tuple(value)


def _validate_content_part(part: object) -> None:
    if not isinstance(part, Mapping):
        raise ValueError("content parts must be mappings")
    part_type = part.get("type")
    if part_type == "text":
        if not isinstance(part.get("text"), str):
            raise ValueError("text parts require text")
        return
    if part_type == "image_url":
        image = part.get("image_url")
        if not isinstance(image, Mapping) or not isinstance(image.get("url"), str):
            raise ValueError("image parts require a URL")
        if image.get("detail", "low") not in ("auto", "low", "high"):
            raise ValueError("invalid image detail")
        return
    if part_type == "file":
        file_data = part.get("file")
        if not isinstance(file_data, Mapping):
            raise ValueError("file parts require file data")
        for field_name in (
            "path",
            "name",
            "content",
            "mime",
            "data_base64",
            "encoding",
        ):
            value = file_data.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError("file fields must be text")
        if "is_inline" in file_data and not isinstance(file_data["is_inline"], bool):
            raise ValueError("file is_inline must be bool")
        return
    raise ValueError("unsupported content part")


def _validate_message_dict(raw: Mapping[str, Any]) -> None:
    if "role" not in raw or "content" not in raw:
        raise ValueError("message role and content are required")
    if raw["role"] not in _VALID_ROLES:
        raise ValueError("invalid message role")
    content = raw["content"]
    if content is not None and not isinstance(content, (str, list)):
        raise ValueError("message content must be text, parts, or null")
    if isinstance(content, list):
        for part in content:
            _validate_content_part(part)
    if not isinstance(raw.get("metadata", {}), dict):
        raise ValueError("message metadata must be a mapping")
    if raw.get("name") is not None and not isinstance(raw["name"], str):
        raise ValueError("message name must be text")
    if raw.get("tool_call_id") is not None and not isinstance(raw["tool_call_id"], str):
        raise ValueError("tool_call_id must be text")
    tool_calls = raw.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls must be a list")
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise ValueError("tool calls must be mappings")
            call_id = call.get("id")
            call_type = call.get("type", "function")
            function = call.get("function")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("tool call IDs must be nonblank text")
            if call_type != "function":
                raise ValueError("tool call type must be function")
            if function is not None:
                if not isinstance(function, Mapping):
                    raise ValueError("tool call function must be a mapping")
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("tool function names must be nonblank text")
                if not isinstance(arguments, (str, Mapping)):
                    raise ValueError("tool arguments must be text or a mapping")
    if raw["role"] == "tool":
        call_id = raw.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("tool messages require a nonblank tool_call_id")


def _detached_messages(
    value: object,
    *,
    preserve_metadata: bool,
) -> tuple[Message, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("messages must be a finite list or tuple")

    copied: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("message entries must be mappings")
        raw = deepcopy(dict(item))
        _validate_message_dict(raw)
        copied.append(raw)

    messages = tuple(dicts_to_messages(copied))
    for raw, message in zip(copied, messages):
        message.metadata = (
            deepcopy(raw.get("metadata", {})) if preserve_metadata else {}
        )
    return messages


def _with_baseline(
    messages: tuple[Message, ...],
    baseline: tuple[Message, ...],
) -> tuple[Message, ...]:
    if any(message.role == "system" for message in messages):
        return messages
    return baseline + messages


def _create_preparation(
    *,
    agent_id: str,
    runtime_scope_id: str,
    messages: tuple[Message, ...],
    turn_index: int,
    branch_id: int,
    parent_branch_path: Sequence[tuple[int, int]],
    branch_view: Sequence[tuple[int, int]],
    mode: RuntimeContextPreparationMode,
) -> RuntimeContextPreparation:
    return RuntimeContextPreparation.create(
        agent_id=agent_id,
        runtime_scope_id=runtime_scope_id,
        provider_conversation=messages,
        turn_index=turn_index,
        branch_id=branch_id,
        parent_branch_path=tuple(parent_branch_path),
        preparation_mode=mode,
        branch_view=tuple(branch_view),
    )


def _restore_replay_metadata(
    raw_messages: list[dict[str, Any]],
    messages: tuple[Message, ...],
) -> None:
    for raw, message in zip(raw_messages, messages):
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            message.metadata = deepcopy(metadata)


def _validate_selected_tool_events(events: Sequence[Mapping[str, Any]]) -> None:
    """Retain strict failures that replay may otherwise sanitize or deduplicate."""
    announced_calls: set[str] = set()
    dispatched_calls: set[str] = set()
    seen_results: set[str] = set()
    expected: set[str] = set()
    identities = ToolIdentityResolver(events)

    for event in events:
        event_type = event.get("type")
        if event_type == "assistant_tool_calls":
            ids = [
                call.get("id") if isinstance(call, Mapping) else None
                for call in (event.get("tool_calls") or [])
            ]
            if any(
                not isinstance(call_id, str) or not call_id.strip() for call_id in ids
            ):
                raise ValueError(
                    "Invalid structured tool history: call IDs must be nonblank"
                )
            synthetic_after_dispatch = (
                event.get("_synthetic_announce") is True
                and bool(expected)
                and set(ids) == expected
                and set(ids).issubset(dispatched_calls)
            )
            if expected and not synthetic_after_dispatch:
                raise ValueError(
                    "Invalid structured tool history: announcement before prior results"
                )
            if len(ids) != len(set(ids)) or any(
                call_id in announced_calls for call_id in ids
            ):
                raise ValueError("Invalid structured tool history: duplicate call ID")
            expected = set(ids)
            announced_calls.update(ids)
        elif event_type == "tool_call":
            call_id = identities.resolve(event)
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(
                    "Invalid structured tool history: call IDs must be nonblank"
                )
            if call_id in dispatched_calls:
                raise ValueError(
                    f"Invalid structured tool history: duplicate dispatch for {call_id!r}"
                )
            dispatched_calls.add(call_id)
            if call_id not in announced_calls:
                expected.add(call_id)
        elif event_type == "tool_result":
            call_id = identities.resolve(event)
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(
                    "Invalid structured tool history: result IDs must be nonblank"
                )
            if call_id in seen_results:
                raise ValueError(
                    f"Invalid structured tool history: duplicate result for {call_id!r}"
                )
            if call_id not in expected:
                raise ValueError(
                    f"Invalid structured tool history: orphan result {call_id!r}"
                )
            seen_results.add(call_id)
            expected.remove(call_id)
        elif expected and event_type in (
            "user_message",
            "user_input",
            "text_chunk",
            "assistant_tool_calls",
        ):
            raise ValueError(
                "Invalid structured tool history: user or assistant message between "
                "announcement and result"
            )

    if expected:
        raise ValueError(
            "Invalid structured tool history: pending announcement has incomplete results"
        )


def _validate_complete_tool_pairs(messages: Sequence[Message]) -> None:
    """Reject provider-invalid native tool history in a selected sequence."""
    announced_ids: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.tool_calls or []
        if not calls:
            if message.role == "tool":
                raise ValueError("Invalid structured tool history: orphan tool result")
            index += 1
            continue
        if message.role != "assistant":
            raise ValueError(
                "Invalid structured tool history: tool calls require an assistant announcement"
            )

        call_ids: list[str] = []
        for call in calls:
            call_id = call.get("id") if isinstance(call, Mapping) else None
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(
                    "Invalid structured tool history: call IDs must be nonblank"
                )
            if call_id in announced_ids or call_id in call_ids:
                raise ValueError(
                    f"Invalid structured tool history: duplicate call ID {call_id!r}"
                )
            call_ids.append(call_id)
        announced_ids.update(call_ids)

        results: list[str] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role == "tool":
            result_id = messages[cursor].tool_call_id
            if not isinstance(result_id, str) or not result_id.strip():
                raise ValueError(
                    "Invalid structured tool history: result IDs must be nonblank"
                )
            if result_id in results:
                raise ValueError(
                    f"Invalid structured tool history: duplicate result for {result_id!r}"
                )
            results.append(result_id)
            cursor += 1

        if len(results) != len(call_ids) or set(results) != set(call_ids):
            raise ValueError(
                "Invalid structured tool history: each announcement must be immediately "
                "followed by exactly one result for every announced call ID"
            )
        index = cursor


def prepare_selected_branch_context(
    *,
    agent_id: str,
    runtime_scope_id: str,
    raw_events: object,
    branch_view: Mapping[int, int] | None = None,
    baseline_system_messages: Sequence[Message] = (),
    pure_text_only: bool = False,
    require_complete_tool_pairs: bool = False,
) -> RuntimeContextPreparation:
    """Prepare one selected branch, optionally requiring valid native tool pairs."""
    try:
        if raw_events is None or isinstance(raw_events, (str, bytes, Mapping)):
            raise ValueError("raw_events must be an iterable of mappings")
        events: list[dict[str, Any]] = []
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise ValueError("raw event entries must be mappings")
            detached = deepcopy(dict(event))
            if "turn_index" in detached or "branch_id" in detached:
                for field_name in ("turn_index", "branch_id"):
                    coordinate = detached.get(field_name)
                    if (
                        isinstance(coordinate, bool)
                        or not isinstance(coordinate, int)
                        or coordinate <= 0
                    ):
                        raise ValueError(
                            "raw event coordinates must be positive integers"
                        )
            events.append(detached)
        baseline = _baseline_messages(baseline_system_messages)
        selected = resolve_branch_view_strict(events, branch_view)
        live_ids = select_live_event_ids(events, branch_view=selected)
        # Slice 2 supports text/bracket replay only. Reject raw structured/native
        # history before message or Registry candidate materialization.
        for event in events:
            event_id = event.get("event_id")
            if isinstance(event_id, int) and event_id not in live_ids:
                continue
            if pure_text_only and (
                event.get("type") in ("assistant_tool_calls", "tool_result")
                or event.get("role") in ("tool", "tool_result")
                or bool(event.get("tool_calls"))
            ):
                raise ValueError("Only pure-text branch selection is supported")
        selected_view = tuple(sorted(selected.items()))
        projected_events = [
            event
            for event in events
            if not isinstance(event.get("event_id"), int)
            or event["event_id"] in live_ids
        ]
        validate_selected_generation_events(projected_events)
        text_subagents = {}
        if require_complete_tool_pairs:
            projected_events, text_subagents = project_selected_subagents(
                projected_events
            )
        normalized = canonicalize_provider_tool_order(
            normalize_tool_call_events(projected_events)
        )
        raw_messages = replay_conversation(
            normalized,
            branch_view=selected,
            include_metadata=True,
        )
        raw_messages = restore_text_subagent_feedback(raw_messages, text_subagents)
        messages = tuple(dicts_to_messages(raw_messages))
        _restore_replay_metadata(raw_messages, messages)
        messages = _with_baseline(messages, baseline)
        if require_complete_tool_pairs:
            _validate_selected_tool_events(normalized)
            _validate_complete_tool_pairs(messages)
        if pure_text_only and any(
            message.role != "system"
            and (
                message.role not in ("user", "assistant")
                or not isinstance(message.content, str)
                or bool(getattr(message, "tool_calls", None))
                or bool(getattr(message, "tool_call_id", None))
            )
            for message in messages
        ):
            raise ValueError("Only pure-text branch selection is supported")

        if not selected_view:
            turn_index = 0
            branch_id = 0
            parent_path: tuple[tuple[int, int], ...] = ()
        else:
            turn_index, branch_id = selected_view[-1]
            parent_path = selected_view[:-1]
        return _create_preparation(
            agent_id=agent_id,
            runtime_scope_id=runtime_scope_id,
            messages=messages,
            turn_index=turn_index,
            branch_id=branch_id,
            parent_branch_path=parent_path,
            branch_view=selected_view,
            mode=RuntimeContextPreparationMode.NORMALIZED_CANONICAL_REPLAY,
        )
    except InvalidBranchViewError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RuntimeContextPreparationError(str(exc)) from exc


def prepare_saved_snapshot_context(
    *,
    agent_id: str,
    runtime_scope_id: str,
    snapshot_messages: object,
    turn_index: int,
    branch_id: int,
    parent_branch_path: Sequence[tuple[int, int]],
    branch_view: Sequence[tuple[int, int]],
    baseline_system_messages: Sequence[Message] = (),
) -> RuntimeContextPreparation:
    """Prepare a detached persisted snapshot selected by the caller."""
    try:
        messages = _detached_messages(snapshot_messages, preserve_metadata=True)
        messages = _with_baseline(
            messages,
            _baseline_messages(baseline_system_messages),
        )
        return _create_preparation(
            agent_id=agent_id,
            runtime_scope_id=runtime_scope_id,
            messages=messages,
            turn_index=turn_index,
            branch_id=branch_id,
            parent_branch_path=parent_branch_path,
            branch_view=branch_view,
            mode=RuntimeContextPreparationMode.NORMALIZED_CANONICAL_REPLAY,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RuntimeContextPreparationError(str(exc)) from exc


def prepare_provider_replacement_context(
    *,
    agent_id: str,
    runtime_scope_id: str,
    provider_messages: object,
    turn_index: int,
    branch_id: int,
    parent_branch_path: Sequence[tuple[int, int]],
    branch_view: Sequence[tuple[int, int]],
    baseline_system_messages: Sequence[Message] = (),
) -> RuntimeContextPreparation:
    """Prepare a provider-authored emergency replacement conversation."""
    try:
        messages = _detached_messages(provider_messages, preserve_metadata=False)
        messages = _with_baseline(
            messages,
            _baseline_messages(baseline_system_messages),
        )
        return _create_preparation(
            agent_id=agent_id,
            runtime_scope_id=runtime_scope_id,
            messages=messages,
            turn_index=turn_index,
            branch_id=branch_id,
            parent_branch_path=parent_branch_path,
            branch_view=branch_view,
            mode=RuntimeContextPreparationMode.PROVIDER_REPLACEMENT,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RuntimeContextPreparationError(str(exc)) from exc
