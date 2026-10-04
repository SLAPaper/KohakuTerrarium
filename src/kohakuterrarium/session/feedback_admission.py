"""Persist and project text feedback with its confirmed terminal constituents."""

from copy import deepcopy

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_coverage import coordinate, parse_origin
from kohakuterrarium.session.output_delivery import validate_output_source
from kohakuterrarium.session.event_origin import (
    EVENT_COORDINATE,
    EVENT_ORIGIN,
    LocalOriginIndex,
    capture_events,
    store_uuid,
)


def persist_injected_input(store, namespace, payload, input_type, turn, branch, path):
    """Append one canonical injected-input constituent and confirm its receipt."""
    uid = store_uuid(store)
    _, eid = store.append_event(
        namespace,
        "user_input_injected",
        {**deepcopy(payload), "input_type": input_type},
        turn_index=turn,
        branch_id=branch,
        parent_branch_path=path,
    )
    store.flush()
    return {
        "mode": "injected_input",
        "input_type": input_type,
        "payload": deepcopy(payload),
        "event_origin": [uid, namespace, eid],
    }


def _same_coordinates(source, coordinates):
    paths = [value.get("parent_branch_path") for value in (source, coordinates)]
    return (
        all(
            isinstance(path, (list, tuple))
            and all(isinstance(pair, (list, tuple)) for pair in path)
            for path in paths
        )
        and tuple(map(tuple, paths[0])) == tuple(map(tuple, paths[1]))
        and all(
            source.get(key) == coordinates.get(key)
            for key in ("turn_index", "branch_id")
        )
    )


def _validate_injected(source, proof, coordinates):
    if (
        source is None
        or source.get("type") not in ("user_input_injected", "feedback_source")
        or source.get("input_type") != proof.get("input_type")
        or not isinstance(proof.get("payload"), dict)
        or "content" not in proof["payload"]
        or any(source.get(key) != value for key, value in proof["payload"].items())
        or not _same_coordinates(source, coordinates)
    ):
        raise HistoryCompatibilityError("Injected input admission provenance conflicts")


def _pair_sources(proof, index, coordinates):
    refs = [
        parse_origin(proof.get(key)) for key in ("user_pair_origin", "event_origin")
    ]
    if (
        refs[0] == refs[1]
        or not proof.get("user_pair_id")
        or not isinstance(proof.get("payload"), dict)
        or "content" not in proof["payload"]
    ):
        raise HistoryCompatibilityError("Invalid input-group user pair")
    for ref, kind in zip(refs, ("user_input", "user_message")):
        source = index.get(ref)
        if (
            source is None
            or (
                source.get("feedback_source_type")
                if source.get("type") == "feedback_source"
                else source.get("type")
            )
            != kind
            or source.get("user_pair_id") != proof["user_pair_id"]
            or any(source.get(key) != value for key, value in proof["payload"].items())
            or not _same_coordinates(source, coordinates)
        ):
            raise HistoryCompatibilityError("Input-group user pair conflicts")
    return set(refs)


def persist_feedback(store, namespace, content, deliveries, turn, branch, path):
    uid, captured = capture_events(store, unique=False)
    index = LocalOriginIndex(captured, namespace)
    coordinates = {"turn_index": turn, "branch_id": branch, "parent_branch_path": path}
    sources = []
    deliveries = deepcopy(deliveries)
    inputs = [proof for proof in deliveries if proof.get("mode") == "injected_input"]
    for delivery in deliveries:
        if delivery.get("mode") == "output_delivery":
            sources.append(validate_output_source(delivery, index))
            continue
        if delivery.get("mode") == "user_pair":
            sources.extend(sorted(_pair_sources(delivery, index, coordinates)))
            continue
        if delivery.get("mode") == "injected_input":
            if delivery.get("input_type") not in (
                "user_input",
                "trigger",
            ) or not isinstance(delivery.get("payload"), dict):
                raise HistoryCompatibilityError("Invalid injected input constituent")
            if delivery.get("event_origin") is not None:
                ref = parse_origin(delivery["event_origin"])
                _validate_injected(index.get(ref), delivery, coordinates)
                sources.append(ref)
            continue
        origin = delivery.get("origin") or {}
        if not origin.get("runtime_scope_id") or not origin.get("delivery_id"):
            raise HistoryCompatibilityError("Feedback lacks terminal delivery proof")
        mode = delivery.get("mode")
        background = mode == "background_terminal"
        match mode:
            case "background_terminal":
                types = ("tool_result", "subagent_result")
            case "dispatch_ack":
                types = ("tool_call", "subagent_call")
            case "direct_child":
                types = ("subagent_call", "subagent_result")
            case _:
                types = ("tool_call", "tool_result")
        required = (
            ("tool_call", "subagent_call")
            if mode == "dispatch_ack"
            else ("tool_result", "subagent_result")
        )
        matches = [
            (row, ref)
            for name, _, row, ref in captured
            if name == namespace
            and row.get("type") in types
            and row.get("job_id") == delivery.get("job_id")
            and isinstance(row.get("origin"), dict)
            and row["origin"].get("runtime_scope_id") == origin["runtime_scope_id"]
            and row["origin"].get("delivery_id") == origin["delivery_id"]
        ]
        if len([row for row, _ in matches if row["type"] in required]) != 1:
            raise HistoryCompatibilityError("Feedback terminal is missing or ambiguous")
        if any(
            not _origin_matches(row.get("origin"), delivery)
            or (
                row.get("tool_call_id") != delivery.get("tool_call_id")
                if background
                else bool(row.get("tool_call_id"))
            )
            for row, _ in matches
        ):
            raise HistoryCompatibilityError("Feedback terminal provenance conflicts")
        if mode == "dispatch_ack":
            delivery["dispatch_origin"] = list(matches[0][1])
        for _, ref in matches:
            index[ref]
            sources.append(ref)
    if len(set(sources)) != len(sources):
        raise HistoryCompatibilityError("Feedback sources must be distinct")
    for proof in inputs:
        if proof.get("event_origin") is not None:
            continue
        payload = {**proof["payload"], "input_type": proof["input_type"]}
        _, input_id = store.append_event(
            namespace,
            "user_input_injected",
            payload,
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=path,
        )
        proof["event_origin"] = [uid, namespace, input_id]
        sources.append((uid, namespace, input_id))
    if not sources or len(set(sources)) != len(sources):
        raise HistoryCompatibilityError("Feedback sources must be distinct")
    _, event_id = store.append_event(
        namespace,
        "tool_feedback_admitted",
        {
            "content": deepcopy(content),
            "source_event_origins": [list(ref) for ref in sources],
            "deliveries": deliveries,
            "message_kind": (
                "input_group"
                if inputs
                or any(proof.get("mode") == "user_pair" for proof in deliveries)
                else "tool_results"
            ),
        },
        turn_index=turn,
        branch_id=branch,
        parent_branch_path=path,
    )
    store.flush()
    return [uid, namespace, event_id]


def _origin_matches(origin, proof):
    expected = proof.get("origin")
    if not isinstance(origin, dict) or not isinstance(expected, dict):
        return False
    if proof.get("mode") == "dispatch_ack":
        return (
            all(
                origin.get(key) == expected.get(key)
                for key in set(origin) | set(expected)
                if key != "execution_class"
            )
            and expected.get("execution_class") == "background"
            and origin.get("execution_class") in ("direct", "background")
        )
    return origin == expected


def _source_kind_matches(source, proof):
    kind = source.get("feedback_source_type", source.get("type"))
    if proof.get("mode") == "dispatch_ack":
        return kind in ("tool_call", "subagent_call") and not source.get("tool_call_id")
    if proof.get("mode") == "direct_child":
        return kind in (
            "tool_call",
            "tool_result",
            "subagent_call",
            "subagent_result",
        ) and not source.get("tool_call_id")
    if proof.get("mode") == "background_terminal":
        return source.get("type") in (
            "tool_result",
            "subagent_result",
            "background_ack_result",
            "feedback_source",
        ) and source.get("tool_call_id") == proof.get("tool_call_id")
    return source.get("type") in (
        "tool_call",
        "tool_result",
        "feedback_source",
    ) and not source.get("tool_call_id")


def admission_sources(row, index, *, allow_missing=False):
    """Validate referenced source identities against the admission's recorded proofs."""
    raw = row.get("source_event_origins")
    deliveries = row.get("deliveries")
    if (
        not isinstance(raw, list)
        or not raw
        or not isinstance(deliveries, list)
        or not deliveries
        or any(not isinstance(proof, dict) for proof in deliveries)
    ):
        raise HistoryCompatibilityError("Invalid feedback admission sources")
    refs = [parse_origin(value) for value in raw]
    if len(set(refs)) != len(refs):
        raise HistoryCompatibilityError("Duplicate feedback admission source")
    pair_refs = set()
    for proof in deliveries:
        if proof.get("mode") == "output_delivery":
            member = validate_output_source(proof, index)
            if member not in refs:
                raise HistoryCompatibilityError("Incomplete output feedback admission")
            pair_refs.add(member)
        if isinstance(proof, dict) and proof.get("mode") == "dispatch_ack":
            origin = proof.get("origin")
            if (
                not isinstance(origin, dict)
                or not isinstance(origin.get("runtime_scope_id"), str)
                or not origin["runtime_scope_id"]
                or not isinstance(origin.get("delivery_id"), str)
                or not origin["delivery_id"]
                or origin.get("execution_class") != "background"
            ):
                raise HistoryCompatibilityError(
                    "Invalid dispatch acknowledgement provenance"
                )
            ref = parse_origin(proof.get("dispatch_origin"))
            if ref not in refs:
                raise HistoryCompatibilityError(
                    "Incomplete dispatch acknowledgement admission"
                )
            source = index.get(ref)
            if source is not None and (
                source.get("job_id") != proof.get("job_id")
                or not _origin_matches(source.get("origin"), proof)
                or not _source_kind_matches(source, proof)
            ):
                raise HistoryCompatibilityError(
                    "Dispatch acknowledgement source conflicts"
                )
        if isinstance(proof, dict) and proof.get("mode") == "user_pair":
            members = _pair_sources(proof, index, row)
            if not members.issubset(refs) or row.get("message_kind") != "input_group":
                raise HistoryCompatibilityError("Incomplete input-group user pair")
            pair_refs.update(members)
        if isinstance(proof, dict) and proof.get("mode") == "injected_input":
            if (
                parse_origin(proof.get("event_origin")) not in refs
                or proof.get("input_type") not in ("user_input", "trigger")
                or not isinstance(proof.get("payload"), dict)
                or "content" not in proof["payload"]
                or row.get("message_kind") != "input_group"
            ):
                raise HistoryCompatibilityError("Incomplete injected input admission")
    for ref in refs:
        if ref in pair_refs:
            continue
        source = index.get(ref)
        if source is None:
            if allow_missing:
                continue
            raise HistoryCompatibilityError("Missing feedback admission source")
        input_proofs = [
            proof
            for proof in deliveries
            if isinstance(proof, dict)
            and proof.get("mode") == "injected_input"
            and parse_origin(proof.get("event_origin")) == ref
        ]
        if input_proofs:
            proof = input_proofs[0]
            if (
                len(input_proofs) != 1
                or source.get("type") not in ("user_input_injected", "feedback_source")
                or source.get("input_type") != proof.get("input_type")
                or not isinstance(proof.get("payload"), dict)
                or any(
                    source.get(key) != value for key, value in proof["payload"].items()
                )
                or any(
                    source.get(key) != row.get(key)
                    for key in ("turn_index", "branch_id", "parent_branch_path")
                )
            ):
                raise HistoryCompatibilityError(
                    "Injected input admission provenance conflicts"
                )
            continue
        if source.get("type") not in (
            "tool_call",
            "tool_result",
            "subagent_result",
            "subagent_call",
            "background_ack_result",
            "feedback_source",
        ) or not any(
            isinstance(proof, dict)
            and isinstance(proof.get("origin"), dict)
            and proof["origin"].get("runtime_scope_id")
            and proof["origin"].get("delivery_id")
            and source.get("job_id") == proof.get("job_id")
            and _origin_matches(source.get("origin"), proof)
            and _source_kind_matches(source, proof)
            for proof in deliveries
        ):
            raise HistoryCompatibilityError(
                "Feedback admission source provenance conflicts"
            )
    return set(refs)


def validate_dispatch_terminal(terminal, admission, index):
    """Validate a later text terminal against a recorded dispatch-phase admission."""
    if (
        admission.get("type") != "tool_feedback_admitted"
        or terminal.get("type")
        not in ("tool_result", "subagent_result", "background_dispatch_result")
        or terminal.get("_synthetic_resume")
    ):
        raise HistoryCompatibilityError("Invalid dispatch feedback terminal")
    admission_sources(admission, index)
    if terminal.get("tool_call_id") or not any(
        proof.get("mode") == "dispatch_ack"
        and proof.get("job_id") == terminal.get("job_id")
        and _origin_matches(terminal.get("origin"), proof)
        for proof in admission["deliveries"]
    ):
        raise HistoryCompatibilityError(
            "Dispatch feedback terminal provenance conflicts"
        )


def project_feedback_events(events):
    """Retain feedback admissions while masking their recorded text-mode tool rows."""
    index = {coordinate(row): row for row in events if coordinate(row) is not None}
    covered = set()
    dispatches = {}
    for row in events:
        if row.get("type") != "tool_feedback_admitted":
            continue
        covered.update(admission_sources(row, index, allow_missing=True))
        for proof in row["deliveries"]:
            if proof.get("mode") == "dispatch_ack":
                origin = proof["origin"]
                key = (origin["runtime_scope_id"], origin["delivery_id"])
                prior = dispatches.get(key)
                if prior is not None and coordinate(prior) != coordinate(row):
                    raise HistoryCompatibilityError(
                        "Dispatch acknowledgement has duplicate admissions"
                    )
                dispatches[key] = row
    output = []
    for row in events:
        if row.get("type") == "background_dispatch_result":
            admission = index.get(parse_origin(row.get("dispatch_feedback_origin")))
            if admission is None:
                raise HistoryCompatibilityError("Missing dispatch projection admission")
            validate_dispatch_terminal(row, admission, index)
        if coordinate(row) in covered:
            row = {
                **row,
                "type": "feedback_source",
                "feedback_source_type": row.get(
                    "feedback_source_type", row.get("type")
                ),
                EVENT_COORDINATE: list(coordinate(row)),
            }
            row.pop(EVENT_ORIGIN, None)
        elif row.get("type") in ("tool_result", "subagent_result") and not row.get(
            "_synthetic_resume"
        ):
            origin = row.get("origin") or {}
            admission = dispatches.get(
                (origin.get("runtime_scope_id"), origin.get("delivery_id"))
            )
            if admission is not None:
                validate_dispatch_terminal(row, admission, index)
                row = {
                    **row,
                    "type": "background_dispatch_result",
                    "dispatch_feedback_origin": list(coordinate(admission)),
                }
        output.append(row)
    return output
