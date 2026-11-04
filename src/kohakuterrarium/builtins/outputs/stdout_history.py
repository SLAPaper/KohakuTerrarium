"""Project stdout history directly into bounded turn previews."""

import re

from kohakuterrarium.llm.message import content_display_text
from kohakuterrarium.session.history.replay import (
    dedupe_adjacent_duplicate_events,
    select_live_event_ids,
)

TEXT_PREVIEW_LIMIT = 200
_NONSPACE = re.compile(r"\S")


def _resume_turn(user: str = "") -> dict:
    """Create one bounded stdout turn."""
    return {
        "user": user[:121],
        "text": "",
        "has_text": False,
        "truncated": False,
        "tools": [],
    }


def _append_resume_text(turn: dict, content: str) -> None:
    """Retain a stripped preview and evidence of meaningful text beyond it."""
    if not content:
        return
    turn["has_text"] = True
    if turn["truncated"]:
        return
    start = 0
    if not turn["text"]:
        first = _NONSPACE.search(content)
        if first is None:
            return
        start = first.start()
    room = TEXT_PREVIEW_LIMIT + 1 - len(turn["text"])
    turn["text"] += content[start : start + room]
    if len(turn["text"]) > TEXT_PREVIEW_LIMIT:
        turn["truncated"] = (
            not turn["text"][-1].isspace()
            or _NONSPACE.search(content, start + room) is not None
        )


def _group_resume_events(
    events: list[dict], *, deduplicated: bool = False
) -> list[dict]:
    """Group live-branch session events into condensed resume turns."""
    if not events:
        return []
    events = events if deduplicated else dedupe_adjacent_duplicate_events(events)
    live_ids = select_live_event_ids(events)
    turns: list[dict] = []
    current = _resume_turn()

    for evt in events:
        etype = evt.get("type", "")
        eid = evt.get("event_id")
        if isinstance(eid, int) and eid not in live_ids:
            continue
        if etype == "user_input":
            if current["user"] or current["has_text"]:
                turns.append(current)
            # Multimodal content lists render as display text, not Python repr.
            current = _resume_turn(content_display_text(evt.get("content", "")))
        elif etype == "trigger_fired":
            if current["user"] or current["has_text"]:
                turns.append(current)
            channel = evt.get("channel", "")
            sender = evt.get("sender", "")
            current = _resume_turn(f"[trigger: {channel} from {sender}]")
        elif etype in ("text", "text_chunk"):
            _append_resume_text(current, evt.get("content", ""))
        elif etype == "tool_call":
            name = evt.get("name", "tool")
            if name not in current["tools"]:
                current["tools"].append(name)

    if current["user"] or current["has_text"]:
        turns.append(current)
    return turns


def _render_resume_text(events: list[dict], *, deduplicated: bool = False) -> str:
    turns = _group_resume_events(events, deduplicated=deduplicated)
    if not turns:
        return ""
    parts = [f"\n--- Resumed session ({len(turns)} turns) ---\n"]
    for turn in turns:
        if turn["user"]:
            user_preview = turn["user"][:120]
            if len(turn["user"]) > 120:
                user_preview += "..."
            parts.append(f"You: {user_preview}\n")
        if turn["has_text"]:
            text_preview = turn["text"][:TEXT_PREVIEW_LIMIT]
            if turn["truncated"]:
                text_preview += "..."
            else:
                text_preview = text_preview.rstrip()
            tools_str = ""
            if turn["tools"]:
                tools_str = f" [used {', '.join(turn['tools'])}]"
            parts.append(f"Assistant:{tools_str} {text_preview}\n")
    parts.append("--- End of history ---\n\n")
    return "".join(parts)
