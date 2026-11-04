"""Bounded previews preserve the original stdout truncation semantics."""

import random

from kohakuterrarium.builtins.outputs.stdout_history import (
    _group_resume_events,
    _render_resume_text,
)


def test_chunked_whitespace_and_truncation_match_complete_text_preview():
    randomizer = random.Random(42)
    samples = [
        " " * 900,
        "a" * 200 + " " * 400,
        "a" * 200 + " " * 400 + "b",
        "  " + "word " * 200 + "  ",
        "\n\ttext\n",
        "x" * 10000,
    ]
    for content in samples:
        for _ in range(8):
            cuts = sorted(
                {
                    0,
                    len(content),
                    *(randomizer.randrange(len(content) + 1) for _ in range(12)),
                }
            )
            events = [{"type": "user_input", "content": "question"}]
            events.extend(
                {"type": "text_chunk", "content": content[a:b]}
                for a, b in zip(cuts, cuts[1:])
                if a != b
            )
            preview = content.strip()
            preview = preview[:200] + ("..." if len(preview) > 200 else "")
            assert f"Assistant: {preview}\n" in _render_resume_text(
                events, deduplicated=True
            )
            turns = _group_resume_events(events, deduplicated=True)
            assert len(turns[0]["text"]) <= 201


def test_projection_bounds_user_text_and_preserves_tool_names():
    events = [
        {"type": "user_input", "content": "u" * 2000},
        {"type": "tool_call", "name": "bash"},
        {"type": "tool_call", "name": "bash"},
        {"type": "text", "content": "result"},
    ]
    rendered = _render_resume_text(events)
    assert "You: " + "u" * 120 + "...\n" in rendered
    assert "Assistant: [used bash] result\n" in rendered
    assert _group_resume_events(events)[0]["tools"] == ["bash"]
