"""Reconcile accepted native snapshot calls with selected durable job evidence."""

import copy
import json

from kohakuterrarium.core.history_compatibility import _native_groups
from kohakuterrarium.core.job_label import canonical_tool_name
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.resume_replay import _proven_background_call

_INTERRUPTED = "Interrupted by session resume"
_KINDS = {
    "tool_call": "tool",
    "tool_result": "tool",
    "subagent_call": "subagent",
    "subagent_result": "subagent",
}
_CALL_TYPES = frozenset({"tool_call", "subagent_call"})
_RESULT_TYPES = frozenset({"tool_result", "subagent_result"})


def reconcile_snapshot(base_messages, selected_raw_events, watermark):
    """Return a copied snapshot plus event ids consumed by base calls."""
    if not isinstance(base_messages, list) or not isinstance(selected_raw_events, list):
        raise HistoryCompatibilityError("Invalid snapshot reconciliation input")
    if watermark is not None and type(watermark) is not int:
        raise HistoryCompatibilityError("Invalid snapshot reconciliation watermark")
    _native_groups(base_messages, allow_pending_tail=True)
    announcements = _announcements(base_messages)
    evidence = _collect_evidence(selected_raw_events, announcements)
    _check_consistency(evidence, announcements)
    copied = copy.deepcopy(base_messages)
    _append_missing(copied, announcements, evidence)
    return copied, set(evidence["consumed"])


def _announcements(messages):
    found = {}
    represented = set()
    order = []
    groups = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        for call in calls:
            call_id = call["id"]
            function = call["function"]
            found[call_id] = {
                "name": canonical_tool_name(function["name"]),
                "arguments": function["arguments"],
                "message_index": index,
                "order": len(order),
            }
            order.append(call_id)
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            represented.add(messages[end]["tool_call_id"])
            end += 1
        groups[index] = end
    return {
        "by_id": found,
        "represented": represented,
        "order": order,
        "groups": groups,
    }


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _event_id(event):
    value = event.get("event_id")
    if type(value) is not int or value <= 0:
        raise HistoryCompatibilityError("Selected resume event lacks a durable id")
    return value


def _kind(event):
    event_type = event.get("type")
    kind = _KINDS.get(event_type)
    if kind is None:
        return None
    return kind


def _provider(event):
    for key in ("tool_call_id", "call_id", "job_id"):
        if event.get(key) is not None and not isinstance(event[key], str):
            raise HistoryCompatibilityError("Invalid resume call identity")
    return _text(event.get("tool_call_id")) or _text(event.get("call_id"))


def _job_key(event):
    origin = event.get("origin")
    scope = origin.get("runtime_scope_id") if isinstance(origin, dict) else None
    if scope is not None and not isinstance(scope, str):
        raise HistoryCompatibilityError("Invalid resume scope")
    return scope, _text(event.get("job_id"))


def _recorded_name(event):
    raw = event.get("name")
    if not isinstance(raw, str) or not raw.strip():
        raw = event.get("tool_name")
    if not isinstance(raw, str) or not raw.strip():
        raise HistoryCompatibilityError("Resume evidence lacks a function name")
    return canonical_tool_name(raw.strip())


def _classes(event):
    origin = event.get("origin")
    origin_class = origin.get("execution_class") if isinstance(origin, dict) else None
    found = set()
    for value in (event.get("execution_class"), origin_class):
        if isinstance(value, str) and value:
            found.add(value)
    background = event.get("background")
    if background is True:
        found.add("background")
    elif background is False:
        found.add("direct")
    elif background is not None:
        raise HistoryCompatibilityError("Invalid resume execution flag")
    return found


def _execution_class(event):
    found = _classes(event)
    if len(found) == 1 and next(iter(found)) in {"background", "direct"}:
        return next(iter(found))
    return None


def _reject_synthetic(event):
    marker = event.get("persisted_synthetic_resume", False)
    if marker is not False or (
        "_synthetic_resume" in event and event["_synthetic_resume"] is not False
    ):
        raise HistoryCompatibilityError(
            "Persisted synthetic resume evidence cannot authorize recovery"
        )


def _json_object(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HistoryCompatibilityError("Resume arguments are not JSON") from exc
    elif isinstance(value, dict):
        parsed = value
    else:
        raise HistoryCompatibilityError("Resume arguments are not a JSON object")
    if not isinstance(parsed, dict):
        raise HistoryCompatibilityError("Resume arguments are not a JSON object")
    return parsed


def _bind(jobs, provider, job_id):
    if not job_id:
        return
    previous = jobs.get(job_id)
    if previous is None:
        jobs[job_id] = provider
    elif provider and previous and previous != provider:
        raise HistoryCompatibilityError(
            "Selected resume job is bound to a different provider"
        )
    elif provider and not previous:
        jobs[job_id] = provider


def _collect_evidence(events, announcements):
    ids = announcements["by_id"]
    buckets = {call_id: {"calls": [], "results": []} for call_id in ids}
    consumed = []
    seen = set()
    jobs = {}
    relevant_jobs = {
        _job_key(event)
        for event in events
        if isinstance(event, dict)
        and event.get("type") in _CALL_TYPES
        and _provider(event) in ids
    }
    for event in events:
        if (
            isinstance(event, dict)
            and event.get("type") in _CALL_TYPES
            and _job_key(event) in relevant_jobs
        ):
            _bind(jobs, _provider(event), _job_key(event))
    for event in events:
        if not isinstance(event, dict):
            raise HistoryCompatibilityError("Invalid selected resume event")
        if _kind(event) is None:
            continue
        provider = _provider(event)
        job_id = _text(event.get("job_id"))
        if event.get("type") in _RESULT_TYPES and not _text(event.get("tool_call_id")):
            provider = jobs.get(_job_key(event), provider)
            event = {**event, "tool_call_id": provider}
        if _job_key(event) in relevant_jobs:
            _bind(jobs, provider, _job_key(event))
        if provider not in ids and job_id in ids:
            raise HistoryCompatibilityError(
                "Selected resume job collides with a snapshot provider id"
            )
        matched = provider if provider in ids else ""
        if not matched:
            continue
        _reject_synthetic(event)
        event_id = _event_id(event)
        if event_id in seen:
            raise HistoryCompatibilityError("Duplicate selected resume event id")
        seen.add(event_id)
        if (
            job_id
            and _job_key(event) in jobs
            and jobs[_job_key(event)] not in ("", provider)
        ):
            raise HistoryCompatibilityError(
                "Selected resume identity conflicts with snapshot announcement"
            )
        consumed.append(event_id)
        role = "calls" if event.get("type") in _CALL_TYPES else "results"
        buckets[matched][role].append(event)
    return {"buckets": buckets, "consumed": consumed}


def _origin_agrees(event):
    origin = event.get("origin")
    if not isinstance(origin, dict):
        return
    execution_class = _execution_class(event)
    if origin.get("execution_class") not in (None, execution_class):
        raise HistoryCompatibilityError("Resume origin class disagrees")
    delivery = origin.get("delivery_id")
    scope = origin.get("runtime_scope_id")
    job_id = _text(event.get("job_id"))
    if delivery is None and scope is None:
        return
    if not (
        isinstance(delivery, str)
        and isinstance(scope, str)
        and job_id
        and delivery == f"{scope}:{job_id}"
    ):
        raise HistoryCompatibilityError("Resume origin job identity disagrees")


def _named(event):
    try:
        _recorded_name(event)
    except HistoryCompatibilityError:
        return False
    return True


def _proven_background(event):
    try:
        proven = bool(_proven_background_call(event))
    except Exception:
        proven = False
    if (
        _execution_class(event) == "background"
        and isinstance(event.get("origin"), dict)
        and not proven
    ):
        raise HistoryCompatibilityError("Resume origin is not a proven background call")
    return (
        proven
        and _execution_class(event) == "background"
        and bool(_provider(event))
        and bool(_text(event.get("job_id")))
        and _named(event)
    )


def _direct_identity(event):
    return (
        _execution_class(event) == "direct"
        and event.get("background") is False
        and bool(_provider(event))
        and bool(_text(event.get("job_id")))
        and _named(event)
    )


def _usable(event):
    found = _classes(event)
    if not found or not found <= {"background", "direct"}:
        return False
    _validate_origin(event)
    return _proven_background(event) or _direct_identity(event)


def _validate_origin(event):
    if "origin" not in event:
        return
    origin = event["origin"]
    if not isinstance(origin, dict):
        raise HistoryCompatibilityError("Invalid resume origin")
    probe = copy.deepcopy(event)
    probe["background"] = True
    probe["execution_class"] = "background"
    probe["origin"]["execution_class"] = "background"
    if not _proven_background_call(probe):
        raise HistoryCompatibilityError("Resume origin is not a proven job lineage")
    for key in ("turn_index", "branch_id"):
        if type(event.get(key)) is not int:
            raise HistoryCompatibilityError("Invalid resume event coordinates")


def _check_pair(dispatch, terminal):
    if dispatch["event_id"] >= terminal["event_id"]:
        raise HistoryCompatibilityError("Resume terminal precedes dispatch")
    if _kind(dispatch) != _kind(terminal):
        raise HistoryCompatibilityError("Resume call kind disagrees")
    for key in ("turn_index", "branch_id", "parent_branch_path"):
        if dispatch.get(key) != terminal.get(key):
            raise HistoryCompatibilityError("Resume terminal lineage disagrees")
    origins = [
        {
            key: value
            for key, value in event.get("origin", {}).items()
            if key != "execution_class"
        }
        for event in (dispatch, terminal)
    ]
    if origins[0] != origins[1]:
        raise HistoryCompatibilityError("Resume origin disagrees across evidence")
    classes = (_execution_class(dispatch), _execution_class(terminal))
    if classes[0] != classes[1] and not (
        classes == ("direct", "background")
        and origins[0]
        and _proven_background(terminal)
    ):
        raise HistoryCompatibilityError("Resume class disagrees across evidence")
    if _provider(dispatch) != _provider(terminal):
        raise HistoryCompatibilityError("Resume provider disagrees across evidence")
    if _text(dispatch.get("job_id")) != _text(terminal.get("job_id")):
        raise HistoryCompatibilityError("Resume job disagrees across evidence")
    if not _usable(dispatch) or not _usable(terminal):
        raise HistoryCompatibilityError("Incomplete resume dispatch evidence")


def _check_event(event, announcement, *, represented=False):
    if not represented and _execution_class(event) is None:
        raise HistoryCompatibilityError("Ambiguous resume execution class")
    _origin_agrees(event)
    _validate_origin(event)
    if _recorded_name(event) != announcement["name"]:
        raise HistoryCompatibilityError("Resume function name disagrees")
    for key in ("name", "tool_name"):
        if event.get(key) and canonical_tool_name(event[key]) != announcement["name"]:
            raise HistoryCompatibilityError("Resume function aliases disagree")
    if event.get("call_id") is not None and event["call_id"] not in (
        _provider(event),
        event.get("job_id"),
    ):
        raise HistoryCompatibilityError("Resume call aliases disagree")
    expected = _json_object(announcement["arguments"])
    native_keys = ("tool_call_arguments", "original_tool_call_arguments")
    keys = (
        native_keys
        if any(event.get(key) is not None for key in native_keys)
        else ("args", "tool_start_args")
    )
    for key in keys:
        recorded = event.get(key)
        if recorded is not None and _json_object(recorded) != expected:
            raise HistoryCompatibilityError("Resume arguments disagree")
    if event.get("type") in _RESULT_TYPES and _terminal_content(event) is None:
        raise HistoryCompatibilityError("Resume terminal lacks content")


def _check_consistency(evidence, announcements):
    for call_id, announcement in announcements["by_id"].items():
        bucket = evidence["buckets"][call_id]
        calls = bucket["calls"]
        results = bucket["results"]
        if len(calls) > 1 or len(results) > 1:
            raise HistoryCompatibilityError("Duplicate selected resume evidence")
        dispatch = calls[0] if calls else None
        terminal = results[0] if results else None
        if call_id in announcements["represented"]:
            for event in (dispatch, terminal):
                if event is not None:
                    _check_event(event, announcement, represented=True)
            bucket["dispatch"], bucket["terminal"] = dispatch, terminal
            continue
        if dispatch is not None and terminal is None and not _usable(dispatch):
            _check_event(dispatch, announcement, represented=True)
            dispatch = None
        for event in (dispatch, terminal):
            if event is not None:
                _check_event(event, announcement)
        if dispatch is not None and terminal is not None:
            _check_pair(dispatch, terminal)
        elif terminal is not None:
            raise HistoryCompatibilityError("Resume terminal lacks a complete dispatch")
        bucket["dispatch"] = dispatch
        bucket["terminal"] = terminal


def _terminal_content(event):
    error = event.get("error")
    if isinstance(error, str) and error:
        return error
    output = event.get("output")
    return copy.deepcopy(output) if isinstance(output, (str, list)) else None


def _result_message(call_id, name, content):
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "name": name,
    }


def _append_missing(copied, announcements, evidence):
    pending = []
    for call_id in announcements["order"]:
        if call_id in announcements["represented"]:
            continue
        announcement = announcements["by_id"][call_id]
        bucket = evidence["buckets"][call_id]
        dispatch = bucket["dispatch"]
        terminal = bucket["terminal"]
        content = None
        if terminal is not None and (
            _usable(terminal) or (dispatch is not None and _usable(dispatch))
        ):
            content = _terminal_content(terminal)
        elif terminal is None and dispatch is not None and _proven_background(dispatch):
            content = _INTERRUPTED
        if content is None:
            continue
        pending.append(
            (
                announcement["message_index"],
                announcement["order"],
                _result_message(call_id, announcement["name"], content),
            )
        )
    pending.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for message_index, _order, message in pending:
        copied.insert(announcements["groups"][message_index], message)
