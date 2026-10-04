"""Validate and persist acknowledgements independently of terminal job results."""

from copy import deepcopy

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_coverage import coordinate, parse_origin
from kohakuterrarium.session.event_origin import capture_events


def ack_sources(payload, by_origin):
    """Resolve a declaration and dispatch from a full native acknowledgement proof."""
    declaration_origin = parse_origin(payload.get("declaration_origin"))
    dispatch_origin = parse_origin(payload.get("dispatch_origin"))
    declaration, dispatch = by_origin.get(declaration_origin), by_origin.get(
        dispatch_origin
    )
    origin = payload.get("origin") or {}
    if (
        payload.get("phase") != "ack"
        or not isinstance(origin, dict)
        or not origin.get("runtime_scope_id")
        or not origin.get("delivery_id")
        or declaration is None
        or dispatch is None
        or declaration.get("type") != "assistant_tool_calls"
        or dispatch.get("type") not in ("tool_call", "subagent_call")
        or dispatch.get("job_id") != payload.get("job_id")
        or dispatch.get("tool_call_id") != payload.get("tool_call_id")
        or declaration.get("veto_group_id")
        != f"{origin.get('runtime_scope_id')}:{declaration.get('generation_id')}"
        or not any(
            call.get("id") == payload.get("tool_call_id")
            for call in declaration.get("tool_calls", [])
        )
    ):
        raise HistoryCompatibilityError("Native acknowledgement source is unproven")
    dispatched = dispatch.get("origin") or {}
    if not isinstance(dispatched, dict):
        raise HistoryCompatibilityError("Native acknowledgement dispatch is invalid")
    for row in (dispatch, payload):
        if row.get("type") is not None and any(
            row.get(key) != origin.get(key)
            for key in ("turn_index", "branch_id", "parent_branch_path")
        ):
            raise HistoryCompatibilityError("Native acknowledgement branch conflicts")
    keys = set(origin) | set(dispatched)
    if any(
        origin.get(key) != dispatched.get(key) for key in keys - {"execution_class"}
    ):
        raise HistoryCompatibilityError("Native acknowledgement dispatch conflicts")
    if any(
        declaration.get(key) != origin.get(key)
        for key in ("turn_index", "branch_id", "parent_branch_path")
    ):
        raise HistoryCompatibilityError("Native acknowledgement declaration conflicts")
    if origin.get("generation_id") != declaration.get("generation_id"):
        raise HistoryCompatibilityError("Native acknowledgement generation conflicts")
    return {declaration_origin, dispatch_origin}


def acknowledged_terminal(row, ack, index):
    """Validate the terminal associated with a separately confirmed ACK occurrence."""
    if ack.get("type") != "native_tool_ack":
        raise HistoryCompatibilityError("Acknowledged result requires an ACK event")
    ack_sources(ack, index)
    origin = row.get("origin")
    proof = ack["origin"]
    if (
        row.get("type")
        not in ("tool_result", "subagent_result", "background_ack_result")
        or row.get("job_id") != ack.get("job_id")
        or row.get("tool_call_id") != ack.get("tool_call_id")
        or not isinstance(origin, dict)
        or any(
            origin.get(key) != proof.get(key)
            for key in set(origin) | set(proof)
            if key != "execution_class"
        )
        or any(
            row.get(key) != proof.get(key)
            for key in ("turn_index", "branch_id", "parent_branch_path")
        )
    ):
        raise HistoryCompatibilityError("Acknowledged terminal provenance conflicts")


def project_ack_results(events):
    """Keep acknowledgements paired while replaying later job terminals as feedback."""
    by_origin = {coordinate(row): row for row in events if coordinate(row) is not None}
    acknowledgements, jobs = {}, {}
    for row in events:
        if row.get("type") != "native_tool_ack":
            continue
        ack_sources(row, by_origin)
        proof = row["origin"]
        key = (proof["runtime_scope_id"], proof["delivery_id"])
        if key in acknowledgements:
            raise HistoryCompatibilityError(
                "Duplicate native acknowledgement occurrence"
            )
        acknowledgements[key] = row
        jobs.setdefault(row["job_id"], []).append(key)
    emitted, deferred, result = set(), {}, []
    for row in events:
        proof = row.get("origin") or {}
        key = (proof.get("runtime_scope_id"), proof.get("delivery_id"))
        if row.get("type") in ("tool_result", "subagent_result"):
            if (
                row.get("_synthetic_resume")
                and len(jobs.get(row.get("job_id"), [])) == 1
            ):
                key = jobs[row["job_id"]][0]
            ack = acknowledgements.get(key)
            if ack is not None:
                if row.get("job_id") != ack["job_id"]:
                    raise HistoryCompatibilityError(
                        "Acknowledged terminal job conflicts"
                    )
                projected = {
                    **row,
                    "type": "background_ack_result",
                    "ack_origin": list(coordinate(ack)),
                }
                if not row.get("_synthetic_resume"):
                    acknowledged_terminal(row, ack, by_origin)
                if key not in emitted:
                    deferred.setdefault(key, []).append(projected)
                else:
                    result.append(projected)
                continue
        result.append(row)
        if row.get("type") == "native_tool_ack":
            emitted.add(key)
            result.extend(deferred.pop(key, []))
    return result


def persist_ack(store, namespace, payload):
    """Confirm one source-bound acknowledgement, reusing an identical retry."""
    uid, captured = capture_events(store, unique=False)
    rows = [(row, origin) for name, _, row, origin in captured if name == namespace]
    by_origin = {origin: row for row, origin in rows}
    payload = deepcopy(payload)
    matches = [
        (row, origin)
        for row, origin in rows
        if row.get("type") in ("tool_call", "subagent_call")
        and row.get("job_id") == payload["job_id"]
        and row.get("tool_call_id") == payload["tool_call_id"]
        and (row.get("origin") or {}).get("delivery_id")
        == payload["origin"].get("delivery_id")
    ]
    if len(matches) != 1:
        raise HistoryCompatibilityError("Native acknowledgement dispatch is ambiguous")
    payload["dispatch_origin"] = list(matches[0][1])
    ack_sources(payload, by_origin)
    prior = [
        (row, origin)
        for row, origin in rows
        if row.get("type") == "native_tool_ack"
        and row.get("job_id") == payload["job_id"]
        and row.get("declaration_origin") == payload["declaration_origin"]
    ]
    if prior:
        if len(prior) != 1 or any(
            prior[0][0].get(key) != value for key, value in payload.items()
        ):
            raise HistoryCompatibilityError("Native acknowledgement retry conflicts")
        return list(prior[0][1]), payload
    origin = payload["origin"]
    _, eid = store.append_event(
        namespace,
        "native_tool_ack",
        payload,
        turn_index=origin["turn_index"],
        branch_id=origin["branch_id"],
        parent_branch_path=origin["parent_branch_path"],
    )
    store.flush()
    return [uid, namespace, eid], payload
