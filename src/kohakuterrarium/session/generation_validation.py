"""Validate declared generation authority in an already selected raw chronology."""

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.history_generation import ASSISTANT_EVENTS

_SOURCE_FIELDS = {
    "backend_key",
    "backend_type",
    "provider_name",
    "model",
    "llm_identifier",
    "tool_format",
}


def validate_selected_generation_events(events):
    """Raise HistoryCompatibilityError for malformed authority; never mutate events.

    Callers must select the retained branch/prefix before calling, and call before
    normalization or replay can omit malformed origins or links.
    """
    origins = {}
    active = None
    for event in events:
        kind = event.get("type")
        if kind in {"user_message", "user_input", "processing_start", "processing_end"}:
            active = None
        if kind == "assistant_generation_start":
            key = event.get("generation_id")
            if (
                type(event.get("schema")) is not int
                or event["schema"] != 1
                or not isinstance(key, str)
                or not key.strip()
            ):
                raise HistoryCompatibilityError("Invalid history generation metadata")
            source = event.get("source")
            if not isinstance(source, dict) or set(source) - _SOURCE_FIELDS:
                raise HistoryCompatibilityError("Invalid history generation source")
            if any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in source.values()
            ):
                raise HistoryCompatibilityError(
                    "Invalid history generation source fact"
                )
            if key in origins:
                raise HistoryCompatibilityError("Duplicate selected generation origin")
            origins[key] = event
            active = key
        if kind not in ASSISTANT_EVENTS:
            continue
        if "generation_id" not in event:
            if active is not None:
                raise HistoryCompatibilityError("Missing selected generation link")
            continue
        key = event["generation_id"]
        origin = origins.get(key) if isinstance(key, str) else None
        if (
            origin is None
            or (
                isinstance(origin.get("event_id"), int)
                and isinstance(event.get("event_id"), int)
                and origin["event_id"] >= event["event_id"]
            )
            or any(
                origin.get(field) != event.get(field)
                for field in ("turn_index", "branch_id", "parent_branch_path")
            )
        ):
            raise HistoryCompatibilityError("Invalid selected generation link")
