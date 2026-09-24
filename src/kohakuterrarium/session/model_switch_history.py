"""Durable switches supply negative evidence, never message provenance."""

from kohakuterrarium.core.history_compatibility import (
    _admit,
    _extras,
    _native_groups,
    _native_portable,
    _origin,
    _raw,
    _source,
)
from kohakuterrarium.errors import HistoryCompatibilityError


def validate_switch_history(
    messages, selected_raw_events, all_switch_events, target_source
):
    """Use selected coordinates and the first subsequent global switch.

    Unlocated private facts conservatively refuse even a switch back: matching
    its old route cannot prove which side of that switch produced the fact.
    """
    raw = [_raw(message) for message in messages]
    selected = list(selected_raw_events)
    switches = []
    for event in all_switch_events:
        if event.get("type") != "model_switch":
            continue
        if (
            type(event.get("schema")) is not int
            or event["schema"] != 1
            or type(event.get("event_id")) is not int
            or event.get("classification") not in ("portable", "native", "private")
        ):
            raise HistoryCompatibilityError("Invalid recorded model switch")
        _source(event.get("from_source"))
        _source(event.get("to_source"))
        switches.append(event)
    switches.sort(key=lambda e: e["event_id"])
    known_switches = [
        e
        for e in switches
        if any(
            e[side].get("backend_key") and e[side].get("backend_type")
            for side in ("from_source", "to_source")
        )
    ]
    announcements = {}
    for event in selected:
        if event.get("type") == "assistant_tool_calls":
            ids = tuple(call.get("id") for call in event.get("tool_calls", []))
            announcements.setdefault(ids, []).append(event.get("event_id"))
    groups = _native_groups(raw)
    index = 0
    while index < len(raw):
        message = raw[index]
        end = groups.get(index, index + 1)
        group = raw[index:end]
        native = index in groups and _native_portable(group)
        if (
            index not in groups
            and message.get("role") in ("system", "user", "assistant")
            and isinstance(message.get("content"), str)
            and not _extras(message)
        ):
            index = end
            continue
        origin = _origin(message)
        # Tool results normally have no generation. The assistant's recorded
        # route was already validated; do not reinterpret it as legacy.
        if origin.get("backend_key") and origin.get("backend_type"):
            index = end
            continue
        if index in groups:
            ids = tuple(call["id"] for call in message["tool_calls"])
            coordinates = announcements.get(ids, [])
        else:
            event_id = message.get("metadata", {}).get("event_id")
            coordinates = [
                e.get("event_id")
                for e in selected
                if e.get("type") == "user_message"
                and message.get("role") == "user"
                and e.get("content") == message.get("content")
                and (event_id is None or e.get("event_id") == event_id)
            ]
        if len(coordinates) != 1 or type(coordinates[0]) is not int:
            if known_switches:
                raise HistoryCompatibilityError(
                    "Ambiguous legacy nonportable history location"
                )
        else:
            switch = next((e for e in switches if e["event_id"] > coordinates[0]), None)
            if switch is not None:
                _admit({}, target_source, switch["from_source"], native=native)
        index = end
