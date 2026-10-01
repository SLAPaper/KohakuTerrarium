"""Pure snapshot reconciliation: base announcements authorize recovery."""

import copy

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.resume_reconciliation import reconcile_snapshot

_INTERRUPTED = "Interrupted by session resume"


def _origin(**overrides):
    origin = {
        "runtime_scope_id": "scope-1",
        "execution_class": "background",
        "origin_path": [[1, 2]],
        "delivery_id": "scope-1:job-1",
        "turn_index": 1,
        "branch_id": 2,
        "parent_branch_path": [],
    }
    origin.update(overrides)
    return origin


def _dispatch(
    *,
    event_id,
    kind="tool",
    provider="call-1",
    job="job-1",
    name="bash",
    args=None,
    execution_class="background",
    background=True,
    origin=None,
    extra=None,
):
    event_type = "tool_call" if kind == "tool" else "subagent_call"
    event = {
        "event_id": event_id,
        "type": event_type,
        "tool_call_id": provider,
        "call_id": provider,
        "job_id": job,
        "name": name,
        "tool_name": name,
        "execution_class": execution_class,
        "background": background,
        "turn_index": 1,
        "branch_id": 2,
        "parent_branch_path": [],
    }
    if args is not None:
        event["args"] = args
        event["tool_call_arguments"] = args
    if origin is not False:
        event["origin"] = (
            origin
            if origin is not None
            else _origin(
                delivery_id=f"scope-1:{job}",
                execution_class=execution_class,
            )
        )
    if extra:
        event.update(extra)
    return event


def _terminal(
    *,
    event_id,
    kind="tool",
    provider="call-1",
    job="job-1",
    name="bash",
    output="done",
    error=None,
    execution_class="background",
    background=True,
    origin=None,
):
    event_type = "tool_result" if kind == "tool" else "subagent_result"
    event = {
        "event_id": event_id,
        "type": event_type,
        "tool_call_id": provider,
        "call_id": provider,
        "job_id": job,
        "name": name,
        "tool_name": name,
        "output": output,
        "execution_class": execution_class,
        "background": background,
        "turn_index": 1,
        "branch_id": 2,
        "parent_branch_path": [],
    }
    if error is not None:
        event["error"] = error
    if origin is not False:
        event["origin"] = (
            origin
            if origin is not None
            else _origin(
                delivery_id=f"scope-1:{job}",
                execution_class=execution_class,
            )
        )
    return event


def _assistant(calls, **extra):
    message = {
        "role": "assistant",
        "content": "working",
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
                "opaque_provider": call.get("opaque", {"vendor": True}),
            }
            for call in calls
        ],
        "metadata": {"generation": {"schema": 1}, "summary": "keep"},
        "system_fingerprint": "fp",
    }
    message.update(extra)
    return message


def _tool(call_id, name, content="already"):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def test_nonmonotonic_ids_do_not_reverse_physical_dispatch_order():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    dispatch = _dispatch(event_id=90, args={})
    terminal = _terminal(event_id=7, output="physical tail")
    result, consumed = reconcile_snapshot(base, [dispatch, terminal], 90)
    assert result[-1]["content"] == "physical tail"
    assert consumed == {90, 7}
    with pytest.raises(HistoryCompatibilityError, match="precedes"):
        reconcile_snapshot(base, [terminal, dispatch], 90)


def test_tool_and_subagent_terminals_before_and_after_watermark():
    base = [
        {"role": "system", "content": "sys"},
        _assistant(
            [
                {"id": "call-1", "name": "bash", "arguments": '{"x": 1}'},
                {"id": "call-2", "name": "read", "arguments": '{"p": "a"}'},
            ]
        ),
    ]
    events = [
        _dispatch(
            event_id=3,
            kind="tool",
            provider="call-1",
            job="job-1",
            name="bash",
            args={"x": 1},
        ),
        _terminal(
            event_id=4,
            kind="tool",
            provider="call-1",
            job="job-1",
            name="bash",
            output="ok-tool",
        ),
        _dispatch(
            event_id=8,
            kind="subagent",
            provider="call-2",
            job="job-2",
            name="read",
            args={"p": "a"},
        ),
        _terminal(
            event_id=9,
            kind="subagent",
            provider="call-2",
            job="job-2",
            name="read",
            output="ok-sub",
        ),
    ]
    original = copy.deepcopy(base)
    copied, consumed = reconcile_snapshot(base, events, watermark=5)

    assert base == original
    assert copied[0] == base[0]
    assert copied[1]["tool_calls"] == base[1]["tool_calls"]
    assert copied[1]["metadata"] == base[1]["metadata"]
    assert copied[1]["system_fingerprint"] == "fp"
    assert [item["tool_call_id"] for item in copied[2:]] == ["call-1", "call-2"]
    assert copied[2]["content"] == "ok-tool"
    assert copied[3]["content"] == "ok-sub"
    assert consumed == {3, 4, 8, 9}


def test_partial_multi_call_appends_only_missing_adjacent_results():
    base = [
        _assistant(
            [
                {"id": "call-1", "name": "bash", "arguments": "{}"},
                {"id": "call-2", "name": "read", "arguments": "{}"},
            ]
        ),
        _tool("call-1", "bash", "kept"),
    ]
    events = [
        _dispatch(event_id=1, provider="call-1", job="job-1", name="bash"),
        _terminal(
            event_id=2, provider="call-1", job="job-1", name="bash", output="new"
        ),
        _dispatch(event_id=3, provider="call-2", job="job-2", name="read"),
        _terminal(
            event_id=4, provider="call-2", job="job-2", name="read", output="fresh"
        ),
    ]
    copied, consumed = reconcile_snapshot(base, events, None)
    assert copied[1]["content"] == "kept"
    assert copied[2]["tool_call_id"] == "call-2"
    assert copied[2]["content"] == "fresh"
    assert consumed == {1, 2, 3, 4}


def test_unknown_and_direct_pending_without_terminal_stay_pending():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    events = [
        _dispatch(
            event_id=1,
            execution_class="mystery",
            background=None,
            origin=False,
        )
    ]
    copied, consumed = reconcile_snapshot(base, events, 10)
    assert copied == base
    assert consumed == {1}


def test_missing_raw_dispatch_legacy_pending_remains_pending():
    base = [_assistant([{"id": "legacy", "name": "bash", "arguments": "{}"}])]
    copied, consumed = reconcile_snapshot(base, [], None)
    assert copied == base
    assert consumed == set()


def test_proven_background_without_terminal_synthesizes_ephemeral_error():
    base = [_assistant([{"id": "call-1", "name": "bash[aabbcc]", "arguments": "{}"}])]
    events = [_dispatch(event_id=7, name="bash[aabbcc]", extra={"tool_name": "bash"})]
    copied, consumed = reconcile_snapshot(base, events, 1)
    assert len(copied) == 2
    result = copied[1]
    assert result == {
        "role": "tool",
        "content": _INTERRUPTED,
        "tool_call_id": "call-1",
        "name": "bash",
    }
    assert "origin" not in result and "event_id" not in result
    assert consumed == {7}


def test_raw_args_absent_uses_snapshot_and_semantic_match_accepts():
    base = [
        _assistant([{"id": "call-1", "name": "bash", "arguments": '{"b": 1, "a": 2}'}])
    ]
    events = [
        _dispatch(event_id=2, args={"a": 2, "b": 1}, extra={"args": '{"a":2,"b":1}'}),
        _terminal(event_id=3, output="matched"),
    ]
    copied, consumed = reconcile_snapshot(base, events, None)
    assert copied[1]["content"] == "matched"
    assert consumed == {2, 3}


def test_duplicate_dispatch_or_result_and_synthetic_marker_reject():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    dispatch = _dispatch(event_id=1)
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [dispatch, copy.deepcopy(dispatch)], None)
    terminal = _terminal(event_id=2, output="a")
    other = _terminal(event_id=3, output="b")
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [dispatch, terminal, other], None)
    marked = _terminal(event_id=4, output="x")
    marked["persisted_synthetic_resume"] = True
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [dispatch, marked], None)


def test_foreign_job_provider_and_origin_do_not_authorize_or_conflict():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    foreign = _dispatch(event_id=5, provider="other", job="other-job", name="bash")
    copied, consumed = reconcile_snapshot(base, [foreign], None)
    assert copied == base
    assert consumed == set()

    conflict = _dispatch(event_id=6, provider="other", job="job-1", name="bash")
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [_dispatch(event_id=1), conflict], None)

    disagree = _dispatch(event_id=7, origin=_origin(delivery_id="scope-1:not-job"))
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [disagree], None)


def test_terminal_without_dispatch_never_closes_pending_call():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [_terminal(event_id=2)], None)


@pytest.mark.parametrize("damage", ["kind", "path", "job", "class", "synthetic_int"])
def test_terminal_must_match_dispatch_evidence(damage):
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    dispatch, terminal = _dispatch(event_id=1), _terminal(event_id=2)
    if damage == "kind":
        terminal["type"] = "subagent_result"
    elif damage == "path":
        terminal["origin"]["branch_id"] = 3
        terminal["origin"]["origin_path"] = [[1, 3]]
        terminal["branch_id"] = 3
    elif damage == "job":
        terminal["job_id"] = "other-job"
    elif damage == "class":
        terminal["execution_class"] = "direct"
    else:
        terminal["_synthetic_resume"] = 1
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [dispatch, terminal], None)


def test_direct_origin_and_internal_call_alias_are_supported():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    events = [
        _dispatch(event_id=1, execution_class="direct", background=False),
        _terminal(event_id=2, execution_class="direct", background=False),
    ]
    for event in events:
        event["call_id"] = "job-1"
    copied, consumed = reconcile_snapshot(base, events, 9)
    assert copied[-1]["content"] == "done"
    assert consumed == {1, 2}


def test_corrupt_base_rejected_before_mutation():
    base = [{"role": "assistant", "tool_calls": [{"id": "x", "type": "function"}]}]
    snapshot = copy.deepcopy(base)
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [], None)
    assert base == snapshot


@pytest.mark.parametrize(
    "damage", ["early", "numeric_provider", "call_alias", "name_alias"]
)
def test_malformed_or_reversed_evidence_rejects(damage):
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    dispatch, terminal = _dispatch(event_id=1), _terminal(event_id=2)
    if damage == "early":
        dispatch["event_id"], terminal["event_id"] = 2, 1
    elif damage == "numeric_provider":
        terminal["tool_call_id"] = 13
    elif damage == "call_alias":
        terminal["call_id"] = "foreign"
    else:
        terminal["tool_name"] = "foreign"
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(
            base,
            [terminal, dispatch] if damage == "early" else [dispatch, terminal],
            None,
        )


def test_unrelated_scoped_jobs_do_not_conflict_with_accepted_snapshot():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    a = _dispatch(event_id=1, provider="other-a", job="foreign")
    b = _dispatch(
        event_id=2,
        provider="other-b",
        job="foreign",
        origin=_origin(runtime_scope_id="scope-2", delivery_id="scope-2:foreign"),
    )
    copied, consumed = reconcile_snapshot(base, [a, b], None)
    assert copied == base and consumed == set()


def test_promoted_direct_call_matches_background_terminal_on_same_origin():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    dispatch = _dispatch(event_id=1, execution_class="direct", background=False)
    terminal = _terminal(event_id=2)
    copied, consumed = reconcile_snapshot(base, [dispatch, terminal], 1)
    assert copied[-1]["content"] == "done"
    assert consumed == {1, 2}


def test_execution_args_may_omit_framework_options_from_native_arguments():
    arguments = '{"run_in_background":true}'
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": arguments}])]
    call = _dispatch(event_id=1, args={})
    call["tool_call_arguments"] = arguments
    copied, consumed = reconcile_snapshot(base, [call, _terminal(event_id=2)], 1)
    assert copied[-1]["content"] == "done"
    assert consumed == {1, 2}


def test_conflicting_native_argument_carriers_are_rejected():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    call = _dispatch(
        event_id=1, args={}, extra={"original_tool_call_arguments": '{"forged":true}'}
    )
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [call], 1)


def test_classless_legacy_paired_snapshot_is_preserved():
    base = [
        _assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}]),
        _tool("call-1", "bash", "legacy"),
    ]
    events = [_dispatch(event_id=1, origin=False), _terminal(event_id=2, origin=False)]
    for event in events:
        event.pop("execution_class")
        event.pop("background")
    copied, consumed = reconcile_snapshot(base, events, 2)
    assert copied == base
    assert consumed == {1, 2}


@pytest.mark.parametrize("kind", ["tool", "subagent"])
def test_job_only_terminal_matches_dispatch_and_preserves_multimodal_content(kind):
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    content = [
        {"type": "text", "text": "done"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    dispatch = _dispatch(event_id=1, kind=kind)
    terminal = _terminal(event_id=2, kind=kind, output=content)
    terminal.pop("tool_call_id")
    terminal["call_id"] = "job-1"
    copied, consumed = reconcile_snapshot(base, [dispatch, terminal], 1)
    assert copied[-1]["content"] == content
    assert consumed == {1, 2}


def test_null_extra_alias_cannot_hide_recorded_args_conflict():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    call = _dispatch(
        event_id=1,
        kind="subagent",
        args={"bad": True},
        extra={"original_tool_call_arguments": None},
    )
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, [call], 1)


def test_idempotent_when_results_already_represented():
    base = [
        _assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}]),
        _tool("call-1", "bash", "placeholder"),
    ]
    events = [_dispatch(event_id=1), _terminal(event_id=2, output="later")]
    first, consumed = reconcile_snapshot(base, events, 0)
    second, again = reconcile_snapshot(first, events, 0)
    assert first[1]["content"] == "placeholder"
    assert second == first
    assert consumed == again == {1, 2}


def test_direct_real_terminal_without_full_origin_closes_call():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    events = [
        _dispatch(event_id=1, execution_class="direct", background=False, origin=False),
        _terminal(
            event_id=2,
            execution_class="direct",
            background=False,
            origin=False,
            output="direct-ok",
        ),
    ]
    copied, consumed = reconcile_snapshot(base, events, 99)
    assert copied[1]["content"] == "direct-ok"
    assert consumed == {1, 2}


def _legacy_dispatch(*, event_id, provider, name="bash", internal=None):
    event = _dispatch(
        event_id=event_id,
        provider=provider,
        job=None,
        name=name,
        origin=False,
    )
    event.pop("job_id")
    event.pop("execution_class")
    event.pop("background")
    event["call_id"] = internal or f"{name}_deadbeef"
    return event


def test_legacy_jobless_snapshot_calls_do_not_share_empty_job():
    base = [
        _assistant(
            [
                {"id": "call-1", "name": "bash", "arguments": "{}"},
                {"id": "call-2", "name": "read", "arguments": "{}"},
            ]
        ),
        _tool("call-1", "bash", "a"),
        _tool("call-2", "read", "b"),
    ]
    events = [
        _legacy_dispatch(
            event_id=10, provider="call-legacy", name="skill", internal="skill_aaaa"
        ),
        _legacy_dispatch(
            event_id=11, provider="call-1", name="bash", internal="bash_bbbb"
        ),
        _legacy_dispatch(
            event_id=12, provider="call-2", name="read", internal="read_cccc"
        ),
    ]
    copied, consumed = reconcile_snapshot(base, events, None)
    assert copied == base
    assert consumed == {11, 12}


def test_legacy_jobless_result_pairs_by_internal_call_id():
    base = [
        _assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}]),
        _tool("call-1", "bash", "already"),
    ]
    dispatch = _legacy_dispatch(
        event_id=1, provider="call-1", name="bash", internal="bash_bbbb"
    )
    terminal = _terminal(
        event_id=2,
        provider="call-1",
        job=None,
        origin=False,
        output="legacy-ok",
    )
    terminal.pop("job_id")
    terminal.pop("execution_class")
    terminal.pop("background")
    terminal.pop("tool_call_id")
    terminal["call_id"] = "bash_bbbb"
    copied, consumed = reconcile_snapshot(base, [dispatch, terminal], None)
    assert copied == base
    assert consumed == {1, 2}


def test_same_job_different_tool_call_is_conflict():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    events = [
        _dispatch(event_id=1, provider="call-1", job="job-1"),
        _dispatch(event_id=2, provider="call-9", job="job-1", name="bash"),
    ]
    with pytest.raises(
        HistoryCompatibilityError,
        match="same tool job was recorded under two different tool calls",
    ):
        reconcile_snapshot(base, events, None)


def test_arg_mismatch_and_name_mismatch_reject():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": '{"a": 1}'}])]
    bad_args = [
        _dispatch(event_id=1, args={"a": 2}),
        _terminal(event_id=2, output="no"),
    ]
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, bad_args, None)
    bad_name = [
        _dispatch(event_id=1, name="read", args={"a": 1}),
        _terminal(event_id=2, name="read", output="no"),
    ]
    with pytest.raises(HistoryCompatibilityError):
        reconcile_snapshot(base, bad_name, None)


def test_unproven_background_without_terminal_stays_pending():
    base = [_assistant([{"id": "call-1", "name": "bash", "arguments": "{}"}])]
    event = _dispatch(event_id=1, origin=False)
    copied, consumed = reconcile_snapshot(base, [event], None)
    assert copied == base
    assert consumed == {1}
