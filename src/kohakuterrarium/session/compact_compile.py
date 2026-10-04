"""Compile detached message spans into immutable canonical compact coverage."""

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_coverage import exact_origins, parse_origin
from kohakuterrarium.session.event_origin import LocalOriginIndex, capture_events
from kohakuterrarium.session.feedback_admission import (
    admission_sources,
    validate_dispatch_terminal,
)
from kohakuterrarium.session.native_ack import ack_sources, acknowledged_terminal
from kohakuterrarium.session.history_generation import (
    ASSISTANT_EVENTS,
    generation_payload,
)


class _Unproven(Exception):
    pass


def compile_coverage(store, namespace, prefix, retained):
    """Read one worker-local event snapshot and resolve an actual message prefix."""
    uid, captured = capture_events(store, unique=False)
    rows = [(row, origin) for name, _, row, origin in captured if name == namespace]
    by_origin = LocalOriginIndex(captured, namespace)

    def resolve(message):
        metadata = message.get("metadata") or {}
        inherited = metadata.get("compact_coverage")
        if inherited is not None:
            coverage = exact_origins(inherited)
            if coverage is None:
                raise _Unproven("inherited_legacy_summary")
            return set(coverage)
        role = message.get("role")
        if role == "system":
            return set()
        if role == "user":
            if (
                metadata.get("kind") == "tool_results"
                or metadata.get("feedback_admission_origin") is not None
            ):
                dispatch_proof = metadata.get("dispatch_result_receipt")
                if dispatch_proof is not None:
                    if not isinstance(dispatch_proof, dict):
                        raise HistoryCompatibilityError(
                            "Invalid dispatch result receipt"
                        )
                    origin = parse_origin(dispatch_proof.get("event_origin"))
                    terminal = by_origin.get(origin)
                    admission = by_origin.get(
                        parse_origin(dispatch_proof.get("admission_origin"))
                    )
                    if terminal is None or admission is None:
                        raise HistoryCompatibilityError(
                            "Missing dispatch result source"
                        )
                    validate_dispatch_terminal(terminal, admission, by_origin)
                    if message.get("content") != (
                        terminal.get("output") or terminal.get("error") or ""
                    ):
                        raise HistoryCompatibilityError(
                            "Dispatch result content conflicts"
                        )
                    return {origin}
                proof = metadata.get("background_result_receipt")
                if proof is not None:
                    if not isinstance(proof, dict):
                        raise HistoryCompatibilityError(
                            "Invalid acknowledged result receipt"
                        )
                    origin = parse_origin(proof.get("event_origin"))
                    row = by_origin.get(origin)
                    ack = by_origin.get(parse_origin(proof.get("ack_origin")))
                    if row is None or ack is None or row.get("_synthetic_resume"):
                        raise HistoryCompatibilityError(
                            "Missing acknowledged result source"
                        )
                    acknowledged_terminal(row, ack, by_origin)
                    if message.get("content") != (
                        row.get("output") or row.get("error") or ""
                    ):
                        raise HistoryCompatibilityError(
                            "Acknowledged result content conflicts"
                        )
                    return {origin}
                if metadata.get("feedback_admission_origin") is None:
                    raise _Unproven("unrecorded_feedback_admission")
                origin = parse_origin(metadata["feedback_admission_origin"])
                row = by_origin.get(origin)
                if (
                    row is None
                    or row.get("type") != "tool_feedback_admitted"
                    or row.get("content") != message.get("content")
                    or row.get("message_kind", "tool_results") != metadata.get("kind")
                    or any(
                        row.get(key) != metadata.get(key)
                        for key in ("turn_index", "branch_id")
                    )
                ):
                    raise HistoryCompatibilityError(
                        "Compact feedback admission does not match"
                    )
                members = admission_sources(row, by_origin)
                return {origin, *members}
            if "event_origin" not in metadata:
                raise _Unproven("missing_user_receipt")
            origin = parse_origin(metadata["event_origin"])
            row = by_origin.get(origin)
            if (
                row is None
                or row.get("type") != "user_message"
                or any(
                    row.get(key) != metadata.get(key)
                    for key in ("turn_index", "branch_id")
                )
            ):
                raise HistoryCompatibilityError(
                    "Compact user receipt does not match its event"
                )
            twin = parse_origin(metadata.get("user_pair_origin"))
            other = by_origin.get(twin)
            pair = metadata.get("user_pair_id")
            if (
                not pair
                or other is None
                or other.get("type") != "user_input"
                or any(item.get("user_pair_id") != pair for item in (row, other))
            ):
                raise HistoryCompatibilityError(
                    "Compact user pair receipt does not match"
                )
            return {origin, twin}
        if role == "assistant":
            generation = metadata.get("generation")
            if not isinstance(generation, dict) or not generation.get("generation_id"):
                raise _Unproven("missing_assistant_generation")
            gid = generation["generation_id"]
            if message.get("tool_calls"):
                declaration_origin = metadata.get("native_declaration_origin")
                if declaration_origin is None:
                    raise _Unproven("missing_native_declaration_receipt")
                declaration = by_origin.get(parse_origin(declaration_origin))
                if (
                    declaration is None
                    or declaration.get("type") != "assistant_tool_calls"
                    or declaration.get("generation_id") != gid
                    or declaration.get("tool_calls") != message["tool_calls"]
                ):
                    raise HistoryCompatibilityError(
                        "Native compact declaration does not match"
                    )
            starts = [
                (index, row, origin)
                for index, (row, origin) in enumerate(rows)
                if row.get("type") == "assistant_generation_start"
                and row.get("generation_id") == gid
            ]
            if len(starts) != 1 or generation_payload(
                starts[0][1]
            ) != generation_payload(generation):
                raise HistoryCompatibilityError(
                    "Compact generation is ambiguous or conflicting"
                )
            index, _, start_origin = starts[0]
            following = rows[index + 1 :]
            if not any(
                row.get("type") in ("processing_end", "assistant_generation_start")
                for row, _ in following
            ):
                raise _Unproven("open_assistant_generation")
            members = {
                origin
                for row, origin in rows
                if row.get("generation_id") == gid
                and row.get("type") in ASSISTANT_EVENTS
            }
            members.update(
                origin
                for row, origin in rows
                if row.get("type") == "tool_call"
                and isinstance(row.get("origin"), dict)
                and row["origin"].get("generation_id") == gid
            )
            if not members:
                raise _Unproven("missing_assistant_span")
            return {start_origin, *members}
        if role == "tool":
            delivery = metadata.get("tool_delivery")
            if isinstance(delivery, dict) and delivery.get("phase") == "ack":
                if delivery.get("ack_origin") is None:
                    raise _Unproven("missing_native_ack_receipt")
                origin = parse_origin(delivery["ack_origin"])
                row = by_origin.get(origin)
                if (
                    row is None
                    or row.get("type") != "native_tool_ack"
                    or row.get("output") != message.get("content")
                    or row.get("tool_call_id") != message.get("tool_call_id")
                    or any(
                        row.get(key) != delivery.get(key)
                        for key in ("job_id", "origin", "declaration_origin")
                    )
                ):
                    raise HistoryCompatibilityError(
                        "Native compact acknowledgement conflicts"
                    )
                return {origin, *ack_sources(row, by_origin)}
            if not isinstance(delivery, dict) or delivery.get("phase") != "terminal":
                raise _Unproven("missing_native_result_receipt")
            proof = delivery.get("origin") or {}
            if not proof.get("runtime_scope_id") or not proof.get("delivery_id"):
                raise HistoryCompatibilityError(
                    "Native compact result lacks delivery identity"
                )
            if delivery.get("declaration_origin") is None:
                raise _Unproven("missing_native_declaration_binding")
            declaration = by_origin.get(parse_origin(delivery["declaration_origin"]))
            if (
                declaration is None
                or declaration.get("type") != "assistant_tool_calls"
                or declaration.get("veto_group_id")
                != f"{proof['runtime_scope_id']}:{declaration.get('generation_id')}"
                or not any(
                    call.get("id") == message.get("tool_call_id")
                    for call in declaration.get("tool_calls", [])
                )
                or any(
                    declaration.get(key) != proof.get(key)
                    for key in ("turn_index", "branch_id", "parent_branch_path")
                )
            ):
                raise HistoryCompatibilityError(
                    "Native compact result declaration conflicts"
                )
            matches = [
                (row, origin)
                for row, origin in rows
                if row.get("type") == "tool_result"
                and row.get("job_id") == delivery.get("job_id")
                and isinstance(row.get("origin"), dict)
                and all(
                    row["origin"].get(key) == proof.get(key)
                    for key in ("runtime_scope_id", "delivery_id")
                )
            ]
            if len(matches) != 1:
                raise HistoryCompatibilityError(
                    "Native compact result missing or ambiguous"
                )
            row, origin = matches[0]
            if (
                row.get("tool_call_id") != message.get("tool_call_id")
                or row.get("origin") != proof
            ):
                raise HistoryCompatibilityError(
                    "Native compact result provenance conflicts"
                )
            dispatches = {
                item_origin
                for item, item_origin in rows
                if item.get("type") == "tool_call"
                and item.get("job_id") == delivery["job_id"]
                and item.get("origin") == proof
                and item.get("tool_call_id") == message.get("tool_call_id")
            }
            if len(dispatches) > 1:
                raise HistoryCompatibilityError("Native compact dispatch is ambiguous")
            return {origin, *dispatches}
        raise _Unproven("unsupported_message_role")

    try:
        groups = [resolve(message) for message in prefix]
        covered = set().union(*groups)
        keep = set().union(*(resolve(message) for message in retained))
        for origin in covered | keep:
            if by_origin.get(origin) is None:
                raise HistoryCompatibilityError("Compact coverage reference is missing")
        covered.difference_update(keep)
        if not covered:
            raise _Unproven("empty_effective_prefix")
        anchor = next(
            (origin for _, origin in rows if any(origin in group for group in groups)),
            None,
        )
        if anchor is None:
            raise _Unproven("missing_summary_anchor")
        ordered = [origin for _, origin in rows if origin in covered]
        if len(ordered) != len(covered):
            raise HistoryCompatibilityError(
                "Inherited compact coverage has missing events"
            )
        return {
            "compact_coverage_version": 1,
            "coverage_mode": "exact",
            "replaced_event_origins": [list(origin) for origin in ordered],
            "summary_anchor_origin": list(anchor),
        }
    except _Unproven as exc:
        return {
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "coverage_reason": str(exc),
        }
