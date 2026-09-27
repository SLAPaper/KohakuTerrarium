"""Persist explicitly supplied tool identity and dispatch lineage."""

from copy import deepcopy


def _validate_origin(origin, execution):
    turn, branch = origin.get("turn_index"), origin.get("branch_id")
    if any(type(value) is not int or value < 1 for value in (turn, branch)):
        raise ValueError("Tool origin requires positive turn and branch coordinates")
    path = origin.get("parent_branch_path")
    if not isinstance(path, (list, tuple)):
        raise ValueError("Tool origin requires an ancestry path")
    previous = 0
    for pair in path:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(value) is not int or value < 1 for value in pair)
            or not previous < pair[0] < turn
        ):
            raise ValueError("Tool origin has invalid ancestry")
        previous = pair[0]
    if "origin_path" in origin:
        full_path = origin["origin_path"]
        if not isinstance(full_path, (list, tuple)) or any(
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(value) is not int or value < 1 for value in pair)
            for pair in full_path
        ):
            raise ValueError("Tool origin has an invalid full path")
        normalized = [list(pair) for pair in full_path]
        if normalized != [*[list(p) for p in path], [turn, branch]]:
            raise ValueError("Tool origin path disagrees with its coordinates")
        origin["origin_path"] = normalized
    origin["parent_branch_path"] = [list(pair) for pair in path]
    if origin.get("execution_class") not in (None, execution):
        raise ValueError("Tool origin execution class conflicts with activity")


def _persist_terminal(store, namespace, event_type, payload, retry):
    """Confirm one append, reconciling an uncertain prior write only on retry."""
    origin = payload["origin"]
    if retry:
        matches = [
            event
            for event in store.get_events(namespace)
            if event.get("type") == event_type
            and event.get("job_id") == payload.get("job_id")
            and isinstance(event.get("origin"), dict)
            and event["origin"].get("delivery_id") == origin.get("delivery_id")
            and event["origin"].get("runtime_scope_id")
            == origin.get("runtime_scope_id")
        ]
        if matches:
            if len(matches) != 1 or any(
                matches[0].get(key) != value for key, value in payload.items()
            ):
                raise ValueError("Conflicting persisted terminal for retry")
            store.flush()
            return matches[0]["event_id"]
    result = store.append_event(
        namespace,
        event_type,
        deepcopy(payload),
        turn_index=origin["turn_index"],
        branch_id=origin["branch_id"],
        parent_branch_path=origin["parent_branch_path"],
    )
    store.flush()
    return result


def _persist_veto(store, namespace, payload, declaration, retry):
    """Persist the rejected native request before its unexecuted result."""
    origin = payload["origin"]
    calls = declaration.get("tool_calls", [])
    if (
        len([call for call in calls if call.get("id") == payload.get("tool_call_id")])
        != 1
    ):
        raise ValueError("Native veto requires its original call declaration")
    declaration = deepcopy(declaration)
    existing = [
        event
        for event in store.get_events(namespace)
        if event.get("type") == "assistant_tool_calls"
        and event.get("veto_group_id") == declaration["veto_group_id"]
    ]
    if existing:
        expected = {
            **declaration,
            "turn_index": origin["turn_index"],
            "branch_id": origin["branch_id"],
            "parent_branch_path": origin["parent_branch_path"],
        }
        if len(existing) != 1 or any(
            existing[0].get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Conflicting native veto declaration")
    else:
        store.append_event(
            namespace,
            "assistant_tool_calls",
            declaration,
            turn_index=origin["turn_index"],
            branch_id=origin["branch_id"],
            parent_branch_path=origin["parent_branch_path"],
        )
    return _persist_terminal(store, namespace, "tool_result", payload, retry)


def record_tool_activity(sink, event_type: str, data: dict, metadata: dict):
    """Detach activity provenance and append under its original coordinates."""
    payload = deepcopy(data)
    for key in ("job_id", "tool_call_id", "tool_call_arguments"):
        if key in metadata:
            payload[key] = deepcopy(metadata[key])
    origin = metadata.get("origin")
    if origin is not None and not isinstance(origin, dict):
        raise ValueError("Tool origin must be a mapping")
    origin = deepcopy(origin) if origin is not None else None
    execution = metadata.get("execution_class")
    if execution is None and origin is not None:
        execution = origin.get("execution_class")
    if execution is None and type(metadata.get("background")) is bool:
        execution = "background" if metadata["background"] else "direct"
    if execution not in (None, "direct", "background"):
        raise ValueError("Invalid tool execution class")
    if (
        execution is not None
        and type(metadata.get("background")) is bool
        and metadata["background"] != (execution == "background")
    ):
        raise ValueError("Tool background flag conflicts with execution class")
    if origin is not None:
        _validate_origin(origin, execution)
    if execution in ("direct", "background"):
        payload["execution_class"] = execution
        payload["background"] = execution == "background"
    if origin is None:
        return sink._record(event_type, payload)
    payload["origin"] = origin
    sink._recover_open_text()
    sink._flush_text_segment()
    if metadata.get("_native_veto_declaration") is not None:
        return sink._submit_store_write(
            _persist_veto,
            sink._store,
            sink._event_key_prefix,
            payload,
            deepcopy(metadata["_native_veto_declaration"]),
            bool(metadata.get("_native_veto_retry")),
        )
    if metadata.get("_terminal_publication"):
        return sink._submit_store_write(
            _persist_terminal,
            sink._store,
            sink._event_key_prefix,
            event_type,
            payload,
            bool(metadata.get("_terminal_retry")),
        )
    return sink._append_event_at(
        event_type,
        payload,
        origin.get("turn_index"),
        origin.get("branch_id"),
        deepcopy(origin.get("parent_branch_path")),
    )
