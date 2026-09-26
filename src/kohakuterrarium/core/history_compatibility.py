"""Admission fence for raw history; never sanitize, rewrite, or infer routes."""

from collections.abc import Iterable
from typing import Any

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.llm.message import Message

_FIELDS = (
    "backend_key",
    "backend_type",
    "provider_name",
    "model",
    "llm_identifier",
    "tool_format",
)
_ROUTE = ("backend_key", "backend_type")
_NATIVE = (*_ROUTE, "provider_name", "tool_format")
_STANDARD = {"role", "content", "name", "tool_calls", "tool_call_id", "metadata"}


def _source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - set(_FIELDS):
        raise HistoryCompatibilityError("Invalid history generation source")
    if any(
        v is not None and (not isinstance(v, str) or not v.strip())
        for v in value.values()
    ):
        raise HistoryCompatibilityError("Invalid history generation source fact")
    return value


def _origin(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HistoryCompatibilityError("Invalid history metadata")
    if "generation" not in metadata:
        return {}
    generation = metadata["generation"]
    if (
        not isinstance(generation, dict)
        or type(generation.get("schema")) is not int
        or generation["schema"] != 1
        or not isinstance(generation.get("generation_id"), str)
        or not generation["generation_id"].strip()
        or "source" not in generation
    ):
        raise HistoryCompatibilityError("Invalid history generation metadata")
    return _source(generation["source"])


def _raw(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if not isinstance(message, Message):
        raise HistoryCompatibilityError("Invalid raw history message")
    return {
        **message.extra_fields,
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "tool_calls": message.tool_calls,
        "tool_call_id": message.tool_call_id,
        "metadata": message.metadata,
    }


def _extras(message: dict[str, Any]) -> bool:
    return bool(set(message) - _STANDARD) or bool(message.get("name"))


def _native_groups(
    messages: list[dict[str, Any]], *, allow_pending_tail: bool = False
) -> dict[int, int]:
    """Validate raw pairing before private/legacy context can admit a message."""
    groups = {}
    seen = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "tool" or message.get("tool_call_id") is not None:
            raise HistoryCompatibilityError("Orphan native tool result in history")
        calls = message.get("tool_calls")
        if calls is None or calls == []:
            index += 1
            continue
        if message.get("role") != "assistant" or not isinstance(calls, list):
            raise HistoryCompatibilityError(
                "Invalid native tool announcement in history"
            )
        ids = set()
        for call in calls:
            if (
                not isinstance(call, dict)
                or call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not call["id"].strip()
                or call["id"] in seen
                or call["id"] in ids
            ):
                raise HistoryCompatibilityError(
                    "Invalid or duplicate native call ID in history"
                )
            function = call.get("function")
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"].strip()
                or not isinstance(function.get("arguments"), str)
            ):
                raise HistoryCompatibilityError(
                    "Invalid native function call in history"
                )
            ids.add(call["id"])
        seen.update(ids)
        end = index + 1
        pending = ids.copy()
        while end < len(messages) and messages[end].get("role") == "tool":
            result = messages[end]
            call_id = result.get("tool_call_id")
            if (
                not isinstance(call_id, str)
                or call_id not in pending
                or result.get("tool_calls") not in (None, [])
            ):
                raise HistoryCompatibilityError(
                    "Invalid or duplicate native result in history"
                )
            pending.remove(call_id)
            end += 1
        if pending and not (allow_pending_tail and end == len(messages)):
            raise HistoryCompatibilityError("Missing adjacent native result in history")
        groups[index] = end
        index = end
    return groups


def _native_portable(group: list[dict[str, Any]]) -> bool:
    message = group[0]
    return (
        not _extras(message)
        and (message.get("content") is None or isinstance(message.get("content"), str))
        and all(
            set(call) == {"id", "type", "function"}
            and set(call["function"]) == {"name", "arguments"}
            for call in message["tool_calls"]
        )
        and all(
            not (set(result) - _STANDARD)
            and (result.get("name") is None or isinstance(result["name"], str))
            and isinstance(result.get("content"), str)
            for result in group[1:]
        )
    )


def _admit(
    source: dict[str, Any],
    target: dict[str, Any],
    current: dict[str, Any],
    *,
    native: bool,
) -> None:
    fields = _NATIVE if native else _FIELDS

    def differs(left: dict[str, Any]) -> bool:
        return any(
            left.get(k) is not None
            and target.get(k) is not None
            and left[k] != target[k]
            for k in fields
        )

    if differs(source):
        raise HistoryCompatibilityError(
            "Nonportable history context differs from target"
        )
    if any(source.get(k) is not None and target.get(k) is None for k in _ROUTE):
        raise HistoryCompatibilityError("Nonportable history target route is unknown")
    known = all(source.get(k) for k in _ROUTE)
    if known:
        required = _ROUTE if native else (*_ROUTE, "model")
        if any(not source.get(k) or source[k] != target.get(k) for k in required):
            raise HistoryCompatibilityError(
                "Nonportable history requires a known matching route/context"
            )
        if any(source.get(k) is not None and target.get(k) is None for k in fields):
            raise HistoryCompatibilityError(
                "Nonportable history target context is unknown"
            )
    elif differs(current):
        # Switch evidence is not provenance: never stamp it onto legacy messages.
        raise HistoryCompatibilityError(
            "Legacy nonportable history cannot cross a known context change"
        )


def validate_history_compatibility(
    messages: Iterable[Message | dict[str, Any]],
    target_source: dict[str, Any],
    *,
    current_source: dict[str, Any] | None = None,
    allow_pending_tail: bool = False,
) -> str:
    """Return portable/native/private classification or raise before any mutation.

    Legacy provenance retains historical behavior unless declared facts prove a
    relevant change. Declared route facts fail closed against unknown targets.
    """
    target = _source(target_source)
    current = _source(current_source) if current_source is not None else {}
    raw = [_raw(message) for message in messages]
    origins = [_origin(message) for message in raw]
    groups = _native_groups(raw, allow_pending_tail=allow_pending_tail)
    classification = "portable"
    index = 0
    while index < len(raw):
        message = raw[index]
        if (
            message.get("role") in ("system", "user", "assistant")
            and isinstance(message.get("content"), str)
            and not _extras(message)
            and not message.get("tool_calls")
            and not message.get("tool_call_id")
        ):
            index += 1
            continue
        end = groups.get(index, index + 1)
        native = index in groups and _native_portable(raw[index:end])
        for origin in origins[index:end]:
            _admit(origin, target, current, native=native)
        classification = (
            "private" if not native or classification == "private" else "native"
        )
        index = end
    return classification
