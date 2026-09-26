"""Selected-branch conversation replay, including generation grouping."""

from typing import Any, Iterable

from kohakuterrarium.session.history_generation import GenerationReplay
from kohakuterrarium.session.history_support import (
    _clean_tool_name as clean_tool_name,
    _covered_by_replace as covered_by_replace,
    _rule_priority as rule_priority,
    compact_path_from_event,
    dedupe_adjacent_duplicate_events,
    index_parent_paths,
    resolve_selected_branches,
    select_live_event_ids,
)


def replay_selected_events(
    events: Iterable[dict[str, Any]],
    *,
    branch_view: dict[int, int] | None = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild an OpenAI-shape message list from the event log.

    Events are branch-filtered and emitted in provider-ready message order.
    Consecutive text chunks coalesce, tool announcements pair with tool results,
    system prompts remain ordered, and compaction ranges become one summary.
    Unknown observability events are ignored.
    """
    events_list = dedupe_adjacent_duplicate_events(events)
    # Nested branch ancestry determines the live event set and the selected
    # branch projection compaction rules are validated against.
    live_ids = select_live_event_ids(events_list, branch_view=branch_view)
    provider_ids = {
        evt.get("call_id") or evt.get("job_id"): evt["tool_call_id"]
        for evt in events_list
        if evt.get("type") == "tool_call"
        and evt.get("tool_call_id")
        and (not isinstance(evt.get("event_id"), int) or evt["event_id"] in live_ids)
    }
    selected = resolve_selected_branches(
        events_list, index_parent_paths(events_list), branch_view
    )

    # Compaction summaries replace every covered source event. Each
    # compact_replace may carry a ``compact_path`` so a replacement recorded
    # on one branch never deletes the sibling branch's events.
    replaced_rules: list[tuple[int, int, set[tuple[int, int]] | None, str]] = []
    for evt in events_list:
        if evt.get("type") not in ("compact_replace", "compact_complete"):
            continue
        frm = evt.get("replaced_from_event_id")
        to = evt.get("replaced_to_event_id")
        if isinstance(frm, int) and isinstance(to, int):
            replaced_rules.append(
                (
                    frm,
                    to,
                    compact_path_from_event(evt),
                    evt.get("summary", "") or evt.get("summary_text", ""),
                )
            )
    # A single replay projects ONE linear branch, so at most one compaction
    # summary is live: the deepest compaction on the lineage matching the
    # selected branch. Same-lineage rounds nest (the deeper one wins); a
    # sibling branch's compaction shares only a shorter prefix and must not
    # stack on top of this branch's own summary. Newest wins on an exact tie.
    if replaced_rules:
        selected_pairs = {(turn, branch) for turn, branch in selected.items()}
        # Drop sibling/incompatible rules (rank 0) before selecting: their
        # summaries carry sibling-specific content that must not leak into
        # this branch's replay.
        eligible = [
            r for r in replaced_rules if rule_priority(r, selected_pairs)[0] > 0
        ]
        replaced_rules = (
            [
                max(
                    enumerate(eligible),
                    key=lambda item: (
                        *rule_priority(item[1], selected_pairs),
                        item[0],
                    ),
                )[1]
            ]
            if eligible
            else []
        )
    # Summaries are emitted where the replaced content sat, not at the
    # compact_replace event's own stream position (a compact runs after the
    # live tail it preserves). pending summaries are ordered by replaced_from
    # and flushed when the live stream reaches the first event past a range.
    pending_rules: list[tuple[int, int, set[tuple[int, int]] | None, str]] = sorted(
        replaced_rules, key=lambda r: r[0]
    )

    messages: list[dict[str, Any]] = []
    text_buf: list[str] = []
    generations = GenerationReplay(events_list, live_ids, include_metadata)
    text_generation = None
    # A turn/branch owns exactly one user_message. Graph merge and
    # migration can copy the same event twice; duplicate user messages
    # would break turn-targeted edit resolution and pollute the view.
    seen_user_messages: set[tuple[int, int]] = set()

    def _flush_text() -> None:
        if not text_buf:
            return
        content = "".join(text_buf)
        if content:
            messages.append(
                generations.attach(
                    {"role": "assistant", "content": content}, text_generation
                )
            )
        text_buf.clear()

    def _flush_pending_summaries(
        eid: int, turn_index: object, branch_id: object
    ) -> None:
        # Scan every pending rule, not just the front one: a rule whose range
        # has not passed and whose path does not intersect this branch (e.g. a
        # sibling-branch compact) must not block a later rule that DOES apply —
        # otherwise that summary is injected late or lost entirely.
        i = 0
        while i < len(pending_rules):
            frm, to, _path, summary = pending_rules[i]
            if frm > eid:
                break
            # Inject the summary only when the current branch actually has an
            # event covered by this rule; a rule whose path does not intersect
            # the selected branch must NOT inject a spurious summary. Rules
            # without a path (legacy migration data) cover the whole range.
            if covered_by_replace(eid, turn_index, branch_id, [pending_rules[i]]):
                # Flush buffered assistant text that precedes the replaced
                # range before injecting the summary (a covered event does not
                # otherwise delimit the text buffer).
                _flush_text()
                if summary:
                    messages.append({"role": "assistant", "content": summary})
                pending_rules.pop(i)
                continue
            if eid > to:
                # Range passed with no covered event on this branch — drop.
                pending_rules.pop(i)
                continue
            # Range not passed and event not covered — keep pending and scan
            # the next rule; it may apply to this branch right now.
            i += 1

    for evt in events_list:
        etype = evt.get("type", "")
        eid = evt.get("event_id")

        # Synthetic events without identifiers bypass branch filtering.
        if isinstance(eid, int) and eid not in live_ids:
            continue

        if isinstance(eid, int):
            # A compact covers id range [replaced_from..replaced_to]; its
            # summary is inserted where the covered content sat, i.e. as soon
            # as the stream reaches a covered event on this branch.
            _flush_pending_summaries(eid, evt.get("turn_index"), evt.get("branch_id"))

        if etype != "compact_replace" and covered_by_replace(
            eid,
            evt.get("turn_index"),
            evt.get("branch_id"),
            replaced_rules,
        ):
            continue

        if etype == "compact_replace":
            # Handled by _flush_pending_summaries above; never emitted inline.
            continue

        if etype == "assistant_generation_start":
            _flush_text()
            generations.boundary += 1
            continue

        if etype in ("text_chunk", "text"):
            key = generations.key(evt)
            if text_buf and text_generation != key:
                _flush_text()
            text_generation = key
            chunk = evt.get("content", "")
            if isinstance(chunk, str):
                text_buf.append(chunk)
            continue

        if etype in (
            "compact_start",
            "compact_complete",
            "compact_skipped",
            "compact_decision",
            "background_result",
            "token_usage",
            "turn_token_usage",
            "cache_stats",
            "scratchpad_write",
            "plugin_hook_timing",
            "processing_start",
            "processing_complete",
        ):
            continue

        # Structural events delimit streamed assistant text.
        _flush_text()

        if etype == "user_message":
            ti = evt.get("turn_index")
            bi = evt.get("branch_id")
            if isinstance(ti, int) and isinstance(bi, int):
                if (ti, bi) in seen_user_messages:
                    continue
                seen_user_messages.add((ti, bi))
            message = {"role": "user", "content": evt.get("content", "")}
            if include_metadata:
                message["metadata"] = {
                    "event_id": evt.get("event_id"),
                    "turn_index": evt.get("turn_index"),
                    "branch_id": evt.get("branch_id"),
                }
            messages.append(message)
        elif etype == "assistant_tool_calls":
            tool_calls = evt.get("tool_calls") or []
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and not messages[-1].get("tool_calls")
                and generations.matches(messages[-1], evt)
            ):
                messages[-1]["tool_calls"] = tool_calls
            else:
                messages.append(
                    generations.attach(
                        {
                            "role": "assistant",
                            "content": evt.get("content", ""),
                            "tool_calls": tool_calls,
                        },
                        generations.key(evt),
                    )
                )
        elif etype == "tool_result":
            job_id = evt.get("call_id", "") or evt.get("job_id", "")
            messages.append(
                {
                    "role": "tool",
                    "content": evt.get("output", "") or "",
                    "tool_call_id": evt.get("tool_call_id")
                    or provider_ids.get(job_id, job_id),
                    "name": clean_tool_name(evt),
                }
            )
        elif etype == "system_prompt_set":
            messages.append({"role": "system", "content": evt.get("content", "")})
        elif etype == "assistant_reasoning":
            generation_key = generations.key(evt)
            linked = "generation_id" in evt
            if linked and (
                not isinstance(generation_key, str)
                or generation_key not in generations.origins
            ):
                continue
            reasoning = {
                key: evt[key]
                for key in (
                    "reasoning_content",
                    "reasoning_summary",
                    "reasoning_details",
                    "reasoning",
                    "_kt_assistant_segments",
                )
                if evt.get(key) not in (None, "", [], {})
            }
            target = next(
                (
                    message
                    for message in reversed(messages)
                    if message.get("role") == "assistant"
                    and generations.matches(message, evt)
                ),
                None,
            )
            if target is None and linked and reasoning:
                target = generations.attach(
                    {"role": "assistant", "content": ""}, generation_key
                )
                messages.append(target)
            if target is not None:
                target.update(reasoning)

    _flush_text()
    return messages
