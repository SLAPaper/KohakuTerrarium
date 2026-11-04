"""Unit tests for ``builtins/outputs/stdout.py`` — resume history summary."""

from contextlib import closing

import pytest

from kohakuterrarium.builtins.outputs.stdout import (
    PrefixedStdoutOutput,
    StdoutOutput,
    _group_resume_events,
)
from kohakuterrarium.builtins.outputs.stdout_history import _render_resume_text
from kohakuterrarium.session.resume.history import ResumeHistory
from kohakuterrarium.session.store import SessionStore

from tests.helpers.event_payload_reads import track_event_payload_reads


def test_group_resume_events_renders_multimodal_user_input_as_text():
    events = [
        {
            "type": "user_input",
            "content": [{"text": "hello from web", "type": "text"}],
        },
        {"type": "text", "content": "answer"},
    ]

    turns = _group_resume_events(events)

    assert turns[0]["user"] == "hello from web"
    assert turns[0]["text"] == "answer"


def test_group_resume_events_keeps_plain_string_content():
    events = [
        {"type": "user_input", "content": "plain"},
        {"type": "text", "content": "reply"},
    ]

    turns = _group_resume_events(events)

    assert turns[0]["user"] == "plain"
    assert turns[0]["text"] == "reply"


async def test_on_resume_prints_text_not_repr(capsys):
    output = StdoutOutput()
    events = [
        {
            "type": "user_input",
            "content": [{"text": "hello from web", "type": "text"}],
        },
    ]

    await output.on_resume(events)

    out = capsys.readouterr().out
    assert "You: hello from web" in out
    assert "'type': 'text'" not in out


@pytest.mark.parametrize("output_type", [StdoutOutput, PrefixedStdoutOutput])
async def test_prepare_resume_reads_only_display_payloads(
    tmp_path, monkeypatch, capsys, output_type
):
    path = tmp_path / "display.kohakutr"
    reads = track_event_payload_reads(monkeypatch, path)
    with closing(SessionStore(path)) as store:

        def record(kind, data, turn=1, branch=1, *, legacy=False):
            key, event_id = store.append_event(
                "a",
                kind,
                data,
                turn_index=turn,
                branch_id=branch,
                parent_branch_path=[] if turn == 1 else [[1, 1]],
            )
            if legacy:
                event = store.events[key]
                event.pop("parent_branch_path")
                store.events[key] = event
            return key, event_id

        user_key, _ = record(
            "user_input", {"content": [{"type": "text", "text": "hello from web"}]}
        )
        record("user_message", {"content": "hello from web"})
        _, anchor = record("assistant_round_start", {})
        record(
            "tool_call",
            {"call_id": "c", "name": "read", "args": {"path": "x" * 10000}},
        )
        text_key, _ = record("text_chunk", {"content": " " + "x" * 260 + " "})
        record(
            "assistant_message",
            {
                "round_start_event_id": anchor,
                "message": {
                    "role": "assistant",
                    "content": "canonical-only" * 1000,
                    "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "read"}}
                    ],
                },
            },
        )
        record("tool_result", {"call_id": "c", "output": "unused" * 10000})
        record(
            "tool_call",
            {"call_id": "pending", "name": "bash", "args": {"command": "unused"}},
        )
        record("subagent_call", {"job_id": "child", "task": "unused" * 10000})
        record("user_input", {"content": "hidden"}, 2, 1)
        record("text", {"content": "hidden"}, 2, 1)
        record("user_input", {"content": "visible"}, 2, 2, legacy=True)
        for _ in range(2):
            record("text_chunk", {"content": "shown"}, 2, 2, legacy=True)
        record("trigger_fired", {"channel": "wake", "sender": "clock"}, 2, 2)
        record("text", {"content": "trigger answer"}, 2, 2)
        store.flush()
        keys = {key.decode() for key in store.events.keys()}
        history = ResumeHistory(store, "a", store.max_event_id("a"))
        expected = (
            "\n--- Resumed session (3 turns) ---\n"
            "You: hello from web\n"
            f"Assistant: [used read, bash] {'x' * 200}...\n"
            "You: visible\nAssistant: shown\n"
            "You: [trigger: wake from clock]\nAssistant: trigger answer\n"
            "--- End of history ---\n\n"
        )
        assert await history.projection(_render_resume_text) == expected
        store.append_event("a", "user_input", {"event_id": 1, "content": "future"})
        reads.clear()
        reads.batch_sizes.clear()
        output = output_type()
        prepared = await output.prepare_resume(history)
        assert prepared == expected
        assert set(reads) == keys
        assert len(reads) == len(keys) + 2
        assert reads.count(user_key) == reads.count(text_key) == 2
        await output.deliver_resume(prepared)
        assert capsys.readouterr().out == expected
