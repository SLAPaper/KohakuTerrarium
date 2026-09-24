"""Admission policy uses raw production messages, not sanitized wire history."""

from copy import deepcopy

import pytest

from kohakuterrarium.core.history_compatibility import (
    validate_history_compatibility as validate,
)
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.llm.message import Message

SOURCE = dict(
    backend_key="a",
    backend_type="openai",
    provider_name="openai",
    model="one",
    llm_identifier="a/one",
    tool_format="native",
)


def metadata(source):
    return {"generation": {"schema": 1, "generation_id": "g", "source": source}}


def native(source=SOURCE):
    return [
        Message(
            "assistant",
            None,
            tool_calls=[
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            metadata=metadata(source) if source is not None else {},
        ),
        Message("tool", "ok", tool_call_id="c"),
    ]


@pytest.mark.parametrize("mode", ["native", "bracket", "xml"])
def test_scalar_text_portable_unchanged(mode):
    messages = [
        Message(role, "exact", metadata=metadata(SOURCE))
        for role in ("system", "user", "assistant")
    ]
    before = deepcopy(messages)
    assert (
        validate(messages, dict(SOURCE, backend_key="b", tool_format=mode))
        == "portable"
    )
    assert messages == before


def test_native_model_change_same_route_allowed():
    assert (
        validate(native(), dict(SOURCE, model="two", llm_identifier="a/two"))
        == "native"
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(backend_key="b"),
        dict(backend_type="other"),
        dict(tool_format="bracket"),
        dict(provider_name="other"),
        dict(backend_key=None),
    ],
)
def test_native_route_or_mode_change_rejected(change):
    with pytest.raises(HistoryCompatibilityError):
        validate(native(), dict(SOURCE, **change))


@pytest.mark.parametrize(
    "extra",
    [
        "reasoning_content",
        "reasoning_details",
        "reasoning",
        "_kt_anthropic_content",
        "_kt_assistant_segments",
        "unknown",
    ],
)
def test_private_exact_context_only(extra):
    messages = [
        Message(
            "assistant",
            "text",
            extra_fields={extra: "opaque"},
            metadata=metadata(SOURCE),
        )
    ]
    before = deepcopy(messages)
    assert validate(messages, SOURCE) == "private"
    assert messages == before
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, dict(SOURCE, model="two"))


def test_media_snapshot_not_converted():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "opaque"}}],
            "metadata": metadata(SOURCE),
        }
    ]
    assert validate(messages, SOURCE) == "private"
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, dict(SOURCE, llm_identifier="a/alias"))


def test_legacy_no_fabricated_origin_and_switch_evidence():
    messages = native(None)
    before = deepcopy(messages)
    assert validate(messages, {}) == "native"
    assert (
        validate(messages, dict(SOURCE, model="two"), current_source=SOURCE) == "native"
    )
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, dict(SOURCE, backend_key="b"), current_source=SOURCE)
    assert messages == before


@pytest.mark.parametrize(
    "generation",
    [
        None,
        {},
        {"schema": True, "generation_id": "g", "source": {}},
        {"schema": 1, "generation_id": "g", "source": None},
        {"schema": 1, "generation_id": "g", "source": {"model": 3}},
    ],
)
def test_invalid_selected_generation_even_portable(generation):
    with pytest.raises(HistoryCompatibilityError):
        validate(
            [Message("assistant", "text", metadata={"generation": generation})], SOURCE
        )


@pytest.mark.parametrize("variant", ["orphan", "duplicate", "missing"])
def test_malformed_native_raw_rejected_on_cross(variant):
    messages = native(None)
    if variant == "orphan":
        messages = messages[1:]
    elif variant == "duplicate":
        messages.append(deepcopy(messages[-1]))
    else:
        messages.pop()
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, dict(SOURCE, backend_key="b"), current_source=SOURCE)


def test_individual_origin_not_overwritten_by_current():
    with pytest.raises(HistoryCompatibilityError):
        validate(
            native(),
            dict(SOURCE, backend_key="b"),
            current_source=dict(SOURCE, backend_key="b"),
        )


@pytest.mark.parametrize("source", [SOURCE, None])
@pytest.mark.parametrize(
    "variant",
    [
        "missing",
        "duplicate_result",
        "orphan",
        "interleaved",
        "duplicate_call",
        "reused_call",
        "blank_id",
        "standalone_id",
        "wrong_role",
        "bad_function",
    ],
)
def test_malformed_native_rejected_before_context(source, variant):
    messages = native(source)
    if variant == "missing":
        messages.pop()
    elif variant == "duplicate_result":
        messages.append(deepcopy(messages[-1]))
    elif variant == "orphan":
        messages = messages[1:]
    elif variant == "interleaved":
        messages.insert(1, Message("user", "interrupt"))
    elif variant == "duplicate_call":
        messages[0].tool_calls *= 2
    elif variant == "reused_call":
        messages += deepcopy(messages)
    elif variant == "blank_id":
        messages[0].tool_calls[0]["id"] = messages[1].tool_call_id = " "
    elif variant == "standalone_id":
        messages = [Message("assistant", "text", tool_call_id="c")]
    elif variant == "wrong_role":
        messages[0].role = "user"
    else:
        messages[0].tool_calls[0]["function"].pop("arguments")
    before = deepcopy(messages)
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, SOURCE if source else {})
    assert messages == before


@pytest.mark.parametrize(
    "variant", ["assistant_extra", "result_extra", "media", "call_extra"]
)
def test_valid_private_native_pair_requires_exact_context(variant):
    messages = native()
    if variant == "assistant_extra":
        messages[0].extra_fields["reasoning_content"] = "opaque"
    elif variant == "result_extra":
        messages[1].extra_fields["private"] = "opaque"
    elif variant == "media":
        messages[0].content = [{"type": "text", "text": "opaque"}]
    else:
        messages[0].tool_calls[0]["function"]["private"] = "opaque"
    before = deepcopy(messages)
    assert validate(messages, SOURCE) == "private"
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, dict(SOURCE, model="two"))
    assert messages == before


def test_pending_background_placeholder_is_a_structurally_complete_result():
    messages = native()
    messages[1].content = "Tool running in background; result pending."
    assert validate(messages, SOURCE) == "native"


@pytest.mark.parametrize("private", [False, True])
def test_native_result_metadata_cannot_bypass_validation(private):
    messages = native()
    if private:
        messages[0].extra_fields["private"] = True
    messages[1].metadata = {"generation": {}}
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, SOURCE)


@pytest.mark.parametrize("key", ["backend_key", "backend_type"])
@pytest.mark.parametrize("private", [False, True])
def test_partial_known_route_fails_closed_when_target_fact_unknown(key, private):
    messages = native({key: SOURCE[key]})
    if private:
        messages[0].extra_fields["private"] = True
    with pytest.raises(HistoryCompatibilityError):
        validate(messages, {})
    assert validate(messages, {key: SOURCE[key]}) == (
        "private" if private else "native"
    )


def test_partial_facts_prove_change_without_inferring_vendor():
    assert validate(native({"llm_identifier": "selector"}), {}) == "native"
    with pytest.raises(HistoryCompatibilityError):
        validate(native({"backend_key": "a"}), {"backend_key": "b"})
