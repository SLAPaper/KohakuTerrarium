"""Pure completion identity and prefix policy, including legacy boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.background_origin import (
    completion_identity,
    has_background_provenance,
    lineage_accepts,
)
from kohakuterrarium.core.events import (
    create_tool_complete_event,
    create_user_input_event,
)
from kohakuterrarium.core.tool_origin import capture_tool_context


def _event():
    agent = SimpleNamespace(
        _runtime_scope_id="scope",
        _turn_index=3,
        _branch_id=2,
        _parent_branch_path=[(1, 1)],
    )
    context = capture_tool_context(agent, "job", "background")
    return create_tool_complete_event(job_id="job", content="result", **context)


@pytest.mark.parametrize(
    "field, value",
    [
        ("runtime_scope_id", "other"),
        ("execution_class", "direct"),
        ("origin_path", []),
        ("origin_path", "bad"),
        ("origin_path", [[1, 1], [1, 2]]),
        ("origin_path", [[1, 1], [3, True]]),
        ("origin_path", [[1, 1], [0, 2]]),
        ("origin_path", [[1, 1], [3]]),
        ("origin_path", ["bad"]),
        ("turn_index", True),
        ("branch_id", True),
        ("turn_index", 4),
        ("parent_branch_path", []),
        ("parent_branch_path", None),
        ("delivery_id", "scope:wrong"),
    ],
)
def test_damaged_identity_is_not_a_delivery(field, value):
    event = _event()
    event.context["origin"][field] = value
    assert completion_identity(event, "scope") is None


def test_legacy_and_full_identity_have_distinct_dedup_contracts():
    event = _event()
    original = deepcopy(event.context)
    assert completion_identity(event, "scope") == ([[1, 1], [3, 2]], "scope:job")
    assert event.context == original
    assert has_background_provenance(event)
    for field in ("turn_index", "branch_id", "parent_branch_path", "delivery_id"):
        del event.context["origin"][field]
    assert completion_identity(event, "scope") == ([[1, 1], [3, 2]], None)
    event.context["execution_class"] = "direct"
    assert completion_identity(event, "scope") is None
    event.context = {}
    assert not has_background_provenance(event)
    assert completion_identity(event, "scope") is None
    assert not has_background_provenance(create_user_input_event("normal"))


@pytest.mark.parametrize("scope", [None, "", 1])
def test_no_scope_never_admits(scope):
    assert completion_identity(_event(), scope) is None


@pytest.mark.parametrize("job", [None, "", 1])
def test_no_job_never_admits(job):
    event = _event()
    event.job_id = job
    assert completion_identity(event, "scope") is None


@pytest.mark.parametrize(
    "parents, turn, branch, expected",
    [
        ([(1, 1)], 3, 2, True),
        ([(1, 1), (3, 2)], 5, 1, True),
        ([(1, 1)], 3, 3, False),
        ([], 1, 1, False),
        ([(1, 9), (3, 2)], 5, 1, False),
        (None, 3, 2, False),
        ([(1, 1)], True, 2, False),
        ([(3, 2)], 3, 2, False),
    ],
)
def test_only_origin_or_descendant_is_compatible(parents, turn, branch, expected):
    agent = SimpleNamespace(
        _parent_branch_path=parents, _turn_index=turn, _branch_id=branch
    )
    assert lineage_accepts(agent, [[1, 1], [3, 2]]) is expected
