"""Validate snapshot generation claims against selected persisted origins."""

from kohakuterrarium.core.history_compatibility import _native_groups, _origin
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.history_generation import generation_payload


def validate_resume_snapshot(messages, origins) -> None:
    """Reject malformed or unsupported generation claims without repairing them."""
    for message in messages or []:
        if not isinstance(message, dict):
            raise HistoryCompatibilityError("Invalid raw snapshot message")
        _origin(message)
        generation = message.get("metadata", {}).get("generation")
        if generation is not None:
            declaration = {
                **generation,
                "source": generation_payload(generation)["source"],
            }
            if declaration != origins.get(generation["generation_id"]):
                raise HistoryCompatibilityError("Invalid snapshot generation source")
    _native_groups(messages or [], allow_pending_tail=True)
