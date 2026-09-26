"""Replay helpers for persisted direct sub-agent feedback."""

from collections.abc import Iterable
from typing import Any

from kohakuterrarium.core.job_label import canonical_tool_name, make_job_label


def clean_tool_name(evt: dict) -> str:
    """Prefer recorded identity, repairing legacy labels or missing names."""
    name = evt.get("tool_name") or evt.get("name", "")
    if isinstance(name, str) and name:
        return canonical_tool_name(name)
    call_id = evt.get("call_id") or evt.get("job_id") or ""
    if call_id:
        tool_name, _ = make_job_label(str(call_id))
        if tool_name:
            return tool_name
    return name if isinstance(name, str) else ""


class ToolIdentityError(ValueError):
    """Selected tool history contains invalid or ambiguous identity records."""


def _identity(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolIdentityError(f"Tool {key} must be a string")
    return value if value.strip() else ""


def provider_call_id(event: dict[str, Any]) -> str:
    """Return the recorded provider id or a legacy call/job identity."""
    values = [_identity(event, key) for key in ("tool_call_id", "call_id", "job_id")]
    return next((value for value in values if value), "")


class ToolIdentityResolver:
    """Resolve selected tool records without ambiguous internal-id attribution."""

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self.providers: set[str] = set()
        self.claims: dict[str, set[str]] = {}
        for event in events:
            if event.get("type") != "tool_call":
                continue
            provider = provider_call_id(event)
            if not provider:
                continue
            if provider in self.providers:
                raise ToolIdentityError(
                    f"Invalid structured tool history: duplicate dispatch for {provider!r}; "
                    "duplicate provider id"
                )
            self.providers.add(provider)
            for key in event_identity_keys(event):
                self.claims.setdefault(key, set()).add(provider)

    def resolve(self, event: dict[str, Any]) -> str:
        recorded = provider_call_id(event)
        explicit = _identity(event, "tool_call_id")
        if explicit or event.get("type") == "tool_call":
            return explicit or recorded
        candidates: set[str] = set()
        for key in event_identity_keys(event):
            candidates.update(self.claims.get(key, ()))
        if len(candidates) > 1:
            raise ToolIdentityError(f"ambiguous tool identity: {recorded!r}")
        return next(iter(candidates), recorded)


def event_identity_keys(event: dict[str, Any]) -> list[str]:
    """Return distinct nonblank provider/call/job identities for an event."""
    keys: list[str] = []
    for key in ("tool_call_id", "call_id", "job_id"):
        value = _identity(event, key)
        if value and value not in keys:
            keys.append(value)
    return keys


def canonicalize_provider_tool_order(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Move unambiguous late tool results beside their announcements.

    Callers must first project a single selected branch. Malformed identities are
    intentionally left untouched so strict replay validation can fail closed.
    """
    announcement_by_id: dict[str, int] = {}
    announcement_ids: dict[int, list[str]] = {}
    result_by_id: dict[str, int] = {}
    call_positions: dict[str, list[int]] = {}
    invalid = False
    identities = ToolIdentityResolver(events)

    for position, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "assistant_tool_calls":
            ids: list[str] = []
            for call in event.get("tool_calls") or []:
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id.strip():
                    invalid = True
                    continue
                if call_id in ids or call_id in announcement_by_id:
                    invalid = True
                ids.append(call_id)
                announcement_by_id[call_id] = position
            announcement_ids[position] = ids
        elif event_type == "tool_call":
            call_id = identities.resolve(event)
            if call_id:
                call_positions.setdefault(call_id, []).append(position)
        elif event_type == "tool_result":
            call_id = identities.resolve(event)
            if not call_id:
                invalid = True
                continue
            if call_id in result_by_id:
                invalid = True
            result_by_id[call_id] = position

    if set(result_by_id) != set(announcement_by_id):
        invalid = True
    for call_id, announcement_position in announcement_by_id.items():
        result_position = result_by_id.get(call_id)
        positions = call_positions.get(call_id, [])
        if result_position is None or result_position < announcement_position:
            invalid = True
        if len(positions) > 1 or (
            positions and result_position is not None and positions[0] > result_position
        ):
            invalid = True
    if invalid:
        return events

    moved_results = set(result_by_id.values())
    ordered: list[dict[str, Any]] = []
    for position, event in enumerate(events):
        if position in moved_results:
            continue
        ordered.append(event)
        for call_id in announcement_ids.get(position, []):
            ordered.append(events[result_by_id[call_id]])
    return ordered


def provider_safe_tool_name(event: dict[str, Any], default: str = "") -> str:
    """Resolve persisted UI labels and job-derived names to provider-safe names."""
    name = event.get("name", "")
    if isinstance(name, str) and name and "[" not in name:
        return name
    call_id = event.get("job_id") or event.get("call_id") or ""
    if call_id:
        tool_name, _ = make_job_label(str(call_id))
        if tool_name:
            return tool_name
    if isinstance(name, str) and "[" in name:
        return name.split("[", 1)[0]
    return name if isinstance(name, str) and name else default


def select_direct_subagent_terminals(
    events: Iterable[dict[str, Any]], live_event_ids: set[int]
) -> tuple[set[str], set[int]]:
    """Associate direct results with selected calls and choose one terminal.

    Result association intentionally happens before result filtering because older
    result rows may omit branch coordinates. Only a nonblank job id linked to a
    selected direct call can inherit that call's branch. For duplicate terminals,
    the last persisted eligible event wins (event order, normally highest event_id).
    """
    events_list = list(events)
    selected_jobs = {
        str(evt.get("job_id"))
        for evt in events_list
        if evt.get("type") == "subagent_call"
        and evt.get("background") is False
        and str(evt.get("job_id") or "").strip()
        and (
            not isinstance(evt.get("event_id"), int)
            or evt["event_id"] in live_event_ids
        )
    }
    terminal_by_job: dict[str, int] = {}
    for position, evt in enumerate(events_list):
        if evt.get("type") != "subagent_result":
            continue
        job_id = str(evt.get("job_id") or "")
        if not job_id.strip() or job_id not in selected_jobs:
            continue
        event_id = evt.get("event_id")
        has_branch = isinstance(evt.get("turn_index"), int) and isinstance(
            evt.get("branch_id"), int
        )
        if has_branch and isinstance(event_id, int) and event_id not in live_event_ids:
            continue
        terminal_by_job[job_id] = position
    return selected_jobs, set(terminal_by_job.values())


def select_background_subagent_terminals(
    events: Iterable[dict[str, Any]], live_event_ids: set[int]
) -> tuple[set[str], set[int]]:
    """Choose the last live, explicitly background terminal for each job.

    Unlike direct-result replay, this history-only recovery path does not infer
    coordinates from a call. A durable integer event id on the selected replay
    lineage is required, so coordinate-less legacy terminals fail closed.
    """
    events_list = list(events)
    background_jobs = {
        str(evt.get("job_id"))
        for evt in events_list
        if evt.get("type") == "subagent_call"
        and evt.get("background") is True
        and str(evt.get("job_id") or "").strip()
        and isinstance(evt.get("event_id"), int)
        and not isinstance(evt.get("event_id"), bool)
        and evt["event_id"] in live_event_ids
    }
    terminal_by_job: dict[str, int] = {}
    for position, evt in enumerate(events_list):
        if evt.get("type") != "subagent_result":
            continue
        origin = evt.get("origin")
        is_background = (
            evt.get("background") is True
            or evt.get("execution_class") == "background"
            or (
                isinstance(origin, dict)
                and origin.get("execution_class") == "background"
            )
        )
        job_id = str(evt.get("job_id") or "")
        event_id = evt.get("event_id")
        has_coordinates = isinstance(evt.get("turn_index"), int) and isinstance(
            evt.get("branch_id"), int
        )
        if (
            not is_background
            or not job_id.strip()
            or job_id not in background_jobs
            or not has_coordinates
            or isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id not in live_event_ids
        ):
            continue
        background_jobs.add(job_id)
        terminal_by_job[job_id] = position
    return background_jobs, set(terminal_by_job.values())


def ineligible_background_tool_terminal_positions(
    events: Iterable[dict[str, Any]], live_event_ids: set[int]
) -> set[int]:
    """Fail closed for explicit-background terminals without an exact selected call."""
    events_list = list(events)
    selected_calls: set[tuple[str, str]] = set()
    for event in events_list:
        origin = event.get("origin")
        is_background = (
            event.get("background") is True
            or event.get("execution_class") == "background"
            or (
                isinstance(origin, dict)
                and origin.get("execution_class") == "background"
            )
        )
        event_id = event.get("event_id")
        if (
            event.get("type") == "tool_call"
            and is_background
            and isinstance(event_id, int)
            and not isinstance(event_id, bool)
            and event_id in live_event_ids
        ):
            call_id = str(event.get("call_id") or "").strip()
            job_id = str(event.get("job_id") or "").strip()
            if call_id:
                selected_calls.add((call_id, job_id))

    rejected: set[int] = set()
    last_eligible: dict[tuple[str, str], int] = {}
    for position, event in enumerate(events_list):
        if event.get("type") != "tool_result" or event.get("_synthetic_resume") is True:
            continue
        origin = event.get("origin")
        explicit_background = (
            event.get("background") is True
            or event.get("execution_class") == "background"
            or (
                isinstance(origin, dict)
                and origin.get("execution_class") == "background"
            )
        )
        identity = (
            str(event.get("call_id") or "").strip(),
            str(event.get("job_id") or "").strip(),
        )
        # A terminal claiming a selected background call must satisfy the same
        # fence even when its own background marker is missing. Unrelated legacy
        # and direct terminals retain replay compatibility.
        if not explicit_background and identity not in selected_calls:
            continue
        event_id = event.get("event_id")
        eligible = (
            bool(identity[0])
            and identity in selected_calls
            and isinstance(event_id, int)
            and not isinstance(event_id, bool)
            and event_id in live_event_ids
            and all(
                isinstance(event.get(key), int) and not isinstance(event.get(key), bool)
                for key in ("turn_index", "branch_id")
            )
        )
        if not eligible:
            rejected.add(position)
            continue
        previous = last_eligible.get(identity)
        if previous is not None:
            rejected.add(previous)
        last_eligible[identity] = position
    return rejected


def tool_result_message(
    event: dict[str, Any], *, rejected: bool, name: str
) -> dict[str, Any] | None:
    if rejected:
        return None
    return {
        "role": "tool",
        "content": event.get("output", "") or "",
        "tool_call_id": provider_call_id(event),
        "name": name,
    }


def subagent_feedback_message(evt: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic tool-results message shared by replay paths."""
    job_id = str(evt.get("job_id") or "")
    output = evt.get("output", "") or ""
    error = evt.get("error")
    status = "OK" if not error and evt.get("success", True) else "ERROR"
    return {
        "role": "user",
        "content": f"## {job_id} - {status}\n{error or output}",
        "metadata": {"kind": "tool_results"},
    }


def background_subagent_feedback_message(
    evt: dict[str, Any],
    background_job_ids: set[str],
    replayed_job_ids: set[str],
) -> dict[str, Any] | None:
    """Build one history-only feedback message for an eligible background result."""
    job_id = str(evt.get("job_id") or "")
    if job_id not in background_job_ids or job_id in replayed_job_ids:
        return None
    replayed_job_ids.add(job_id)
    return subagent_feedback_message(evt)


def direct_subagent_feedback_message(
    evt: dict[str, Any],
    direct_job_ids: set[str],
    replayed_job_ids: set[str],
) -> dict[str, Any] | None:
    """Build one replay feedback message for a direct sub-agent result.

    Result events can omit the direct/background marker, so their matching call
    is also consulted. Duplicate persisted results produce only one message.
    """
    job_id = str(evt.get("job_id") or "")
    is_direct = evt.get("background") is False or job_id in direct_job_ids
    if not is_direct or not job_id or job_id in replayed_job_ids:
        return None

    replayed_job_ids.add(job_id)
    return subagent_feedback_message(evt)


def prepare_subagent_feedback_messages(
    events: Iterable[dict[str, Any]],
    live_event_ids: set[int],
    *,
    materialize_background_terminals: bool = False,
) -> dict[int, dict[str, Any]]:
    """Prepare terminal-position feedback while keeping replay policy isolated."""
    events_list = list(events)
    direct_jobs, direct_positions = select_direct_subagent_terminals(
        events_list, live_event_ids
    )
    feedback_by_position: dict[int, dict[str, Any]] = {}
    replayed_direct: set[str] = set()
    for position in sorted(direct_positions):
        feedback = direct_subagent_feedback_message(
            events_list[position], direct_jobs, replayed_direct
        )
        if feedback is not None:
            feedback_by_position[position] = feedback

    if not materialize_background_terminals:
        return feedback_by_position

    background_jobs, background_positions = select_background_subagent_terminals(
        events_list, live_event_ids
    )
    replayed_background: set[str] = set()
    for position in sorted(background_positions):
        event = events_list[position]
        if str(event.get("job_id") or "") in direct_jobs:
            continue
        feedback = background_subagent_feedback_message(
            event, background_jobs, replayed_background
        )
        if feedback is not None:
            feedback_by_position[position] = feedback
    return feedback_by_position
