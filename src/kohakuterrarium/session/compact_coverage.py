"""Canonical compact membership and detached replay rules."""

from copy import deepcopy
from typing import NamedTuple

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import (
    EVENT_COORDINATE,
    EVENT_ORIGIN,
    validate_origin,
)

from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

COVERAGE_FIELDS = (
    "compact_coverage_version",
    "coverage_mode",
    "replaced_event_origins",
    "summary_anchor_origin",
    "coverage_reason",
)


class CompactRule(NamedTuple):
    first: int
    last: int
    path: set | None
    summary: str
    members: frozenset | None = None
    anchor: int | None = None
    metadata: dict | None = None


def coordinate(event):
    value = event.get(EVENT_COORDINATE, event.get(EVENT_ORIGIN))
    if value is None:
        return None
    return parse_origin(value)


def parse_origin(value):
    try:
        return validate_origin(value, value[1])
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise HistoryCompatibilityError("Invalid compact event origin") from exc


def coverage_metadata(event):
    """Detach declared coverage fields without synthesizing missing membership."""
    return {key: deepcopy(event[key]) for key in COVERAGE_FIELDS if key in event}


def exact_origins(event):
    """Return exact membership, or None for explicitly legacy coverage."""
    if "compact_coverage_version" not in event:
        if "replaced_event_origins" in event:
            raise HistoryCompatibilityError(
                "Compact origins require a coverage version"
            )
        return None
    if (
        type(event["compact_coverage_version"]) is not int
        or event["compact_coverage_version"] != 1
    ):
        raise HistoryCompatibilityError("Unsupported compact coverage version")
    mode = event.get("coverage_mode")
    if mode == "legacy_fallback":
        if "replaced_event_origins" in event:
            raise HistoryCompatibilityError(
                "Legacy compact cannot declare exact origins"
            )
        return None
    raw = event.get("replaced_event_origins")
    if mode != "exact" or not isinstance(raw, list):
        raise HistoryCompatibilityError("Invalid exact compact coverage")
    parsed = [parse_origin(value) for value in raw]
    if len(set(parsed)) != len(parsed):
        raise HistoryCompatibilityError("Duplicate exact compact origin")
    anchor = event.get("summary_anchor_origin")
    if parsed or anchor is not None:
        parse_origin(anchor)
    return frozenset(parsed)


def make_rule(event, events, path):
    """Resolve canonical membership against this detached event stream."""
    summary = event.get("summary", "") or event.get("summary_text", "")
    origins = exact_origins(event)
    if origins is None:
        if EVENT_ORIGIN in event:
            logger.warning(
                "Imported compact interval has no canonical coverage; retaining raw history"
            )
            return None
        first, last = event.get("replaced_from_event_id"), event.get(
            "replaced_to_event_id"
        )
        if type(first) is not int or type(last) is not int:
            return None
        return CompactRule(first, last, path, summary)
    if not origins:
        return None
    members = frozenset(e["event_id"] for e in events if coordinate(e) in origins)
    if not members:
        return None
    anchor_origin = parse_origin(event["summary_anchor_origin"])
    anchor = next(
        (e["event_id"] for e in events if coordinate(e) == anchor_origin), None
    )
    if anchor is None:
        anchor = next(e["event_id"] for e in events if e.get("event_id") in members)
    return CompactRule(
        min(members),
        max(members),
        path,
        summary,
        members,
        anchor,
        coverage_metadata(event),
    )


def project_exact_prefix(event, prefix):
    """Build an identity-free replay projection without rewriting the durable event."""
    origins = exact_origins(event)
    if origins is None:
        return None
    members = [coordinate(row) for row in prefix if coordinate(row) in origins]
    if not members:
        return None
    projected = deepcopy(event)
    projected.pop(EVENT_ORIGIN, None)
    projected.pop(EVENT_COORDINATE, None)
    projected["replaced_event_origins"] = [
        list(origin) for origin in dict.fromkeys(members)
    ]
    anchor = parse_origin(event["summary_anchor_origin"])
    if not any(coordinate(row) == anchor for row in prefix):
        anchor = members[0]
    projected["summary_anchor_origin"] = list(anchor)
    return projected


def rule_covers(rule, eid):
    if len(rule) > 4 and rule[4] is not None:
        return eid in rule[4]
    return rule[0] <= eid <= rule[1]
