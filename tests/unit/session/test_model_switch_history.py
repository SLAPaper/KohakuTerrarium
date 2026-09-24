"""Switch controls are negative compatibility evidence, not inferred provenance."""

from copy import deepcopy

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.model_switch_history import validate_switch_history


def _route(model="old", key="route"):
    return {
        "backend_key": key,
        "backend_type": "openai",
        "provider_name": "openai",
        "model": model,
        "tool_format": "native",
    }


def _control():
    return {
        "type": "model_switch",
        "schema": 1,
        "event_id": 5,
        "classification": "portable",
        "from_source": _route(),
        "to_source": _route("new"),
        "to_identifier": "openai/new",
    }


def test_portable_history_stays_unchanged_across_switch():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    before = deepcopy(messages)
    validate_switch_history(messages, [], [_control()], _route("new", "other"))
    assert messages == before


@pytest.mark.parametrize(
    "patch",
    [
        {"schema": 2},
        {"event_id": True},
        {"classification": "invented"},
        {"to_source": {"credential": "must reject"}},
    ],
)
def test_malformed_control_is_not_ignored(patch):
    control = {**_control(), **patch}
    with pytest.raises(HistoryCompatibilityError):
        validate_switch_history([], [], [control], {})


def test_unlocated_private_legacy_history_refuses_even_switch_back():
    messages = [
        {"role": "assistant", "content": "private", "reasoning_state": "opaque"}
    ]
    before = deepcopy(messages)
    with pytest.raises(HistoryCompatibilityError, match="Ambiguous"):
        validate_switch_history(messages, [], [_control()], _route())
    assert messages == before


def test_located_private_user_uses_first_subsequent_control_route():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///retained.png"}}
            ],
            "metadata": {"event_id": 1},
        }
    ]
    selected = [
        {"type": "user_message", "event_id": 1, "content": messages[0]["content"]}
    ]
    before = deepcopy(messages)
    validate_switch_history(messages, selected, [_control()], _route())
    with pytest.raises(HistoryCompatibilityError):
        validate_switch_history(messages, selected, [_control()], _route("new"))
    assert messages == before
