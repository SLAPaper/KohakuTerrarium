"""Detached edit targeting and retained-context preparation."""

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from kohakuterrarium.core.agent_message_history import resolve_edit_message_index
from kohakuterrarium.core.agent_raw_history import reload_raw_prefix_for_target
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.message_locator import (
    is_turn_user_message,
    user_message_indices_for_content,
)
from kohakuterrarium.core.runtime_context_replacement_prepare import (
    prepare_selected_branch_context,
)
from kohakuterrarium.llm.message import normalize_content_parts
from kohakuterrarium.session.event_origin import EVENT_COORDINATE, EVENT_ORIGIN
from kohakuterrarium.session.generation_validation import (
    validate_selected_generation_events,
)
from kohakuterrarium.session.history import (
    resolve_branch_view_strict,
    select_live_event_ids,
)
from kohakuterrarium.session.model_switch_history import validate_switch_history
from kohakuterrarium.session.raw_history import (
    UserMessageSelector,
    select_raw_history_prefix,
)


@dataclass(frozen=True)
class EditRequest:
    message_idx: int = -1
    new_content: Any = ""
    turn_index: int | None = None
    user_position: int | None = None
    branch_view: dict | None = None
    target: UserMessageSelector | None = None
    regenerate: bool = False


class _EventSnapshot:
    def __init__(self, events):
        self.events = events

    def get_events(self, _name):
        return self.events


def _message_turn(message, selected_events, *, persisted):
    """Resolve a retained user message by identity or unique legacy content."""
    metadata = message.metadata or {}
    turn = metadata.get("turn_index")
    if "turn_index" in metadata and (type(turn) is not int or turn <= 0):
        raise ValueError("Invalid edit target turn metadata")
    if not persisted:
        return turn
    candidates = [
        event for event in selected_events if event.get("type") == "user_message"
    ]
    if turn is not None:
        candidates = [event for event in candidates if event.get("turn_index") == turn]
        if "branch_id" in metadata:
            candidates = [
                event
                for event in candidates
                if event.get("branch_id") == metadata["branch_id"]
            ]
        if "event_origin" in metadata:
            candidates = [
                event
                for event in candidates
                if event.get(EVENT_COORDINATE, event.get(EVENT_ORIGIN))
                == metadata["event_origin"]
            ]
        elif "event_id" in metadata:
            candidates = [
                event
                for event in candidates
                if event.get("event_id") == metadata["event_id"]
            ]
    else:
        candidates = [
            event
            for event in candidates
            if user_message_indices_for_content([message], event.get("content"))
        ]
    if len(candidates) != 1 or type(candidates[0].get("turn_index")) is not int:
        raise ValueError("Edit target has no unique selected user event")
    return candidates[0]["turn_index"]


def prepare_edit(store, prefix, messages, coordinates, config, source, request):
    """Resolve an edit against detached messages and a single persisted event read."""
    events = list(store.get_events(prefix)) if store is not None else []
    conv = Conversation(replace(config, max_messages=0))
    for message in messages:
        conv.append_message(message)
    turn, branch, path = coordinates
    candidate = SimpleNamespace(
        session_store=_EventSnapshot(events) if store is not None else None,
        config=SimpleNamespace(name=prefix),
        controller=SimpleNamespace(conversation=conv),
        _turn_index=turn,
        _branch_id=branch,
        _parent_branch_path=list(path),
    )
    view = request.branch_view
    if view is None:
        view = dict(path)
        if turn > 0 and branch > 0:
            view[turn] = branch
    live_ids = select_live_event_ids(events, branch_view=view)
    selected_events = [e for e in events if e.get("event_id") in live_ids]
    target_turn = request.turn_index
    user_position = request.user_position
    rebuilt = False
    target = request.target
    if target is None and request.branch_view and store is not None:
        resolve_branch_view_strict(events, request.branch_view)
        users = [
            event for event in selected_events if event.get("type") == "user_message"
        ]
        matches = (
            [event for event in users if event.get("turn_index") == target_turn]
            if target_turn is not None
            else []
        )
        if (
            target_turn is None
            and user_position is not None
            and 0 <= user_position < len(users)
        ):
            matches = [users[user_position]]
        if len(matches) == 1:
            selected_target = matches[0]
            target = UserMessageSelector(
                selected_target["event_id"],
                selected_target["turn_index"],
                selected_target["branch_id"],
            )
    if target is not None:
        raw = select_raw_history_prefix(
            events, selector=target, branch_view=request.branch_view
        )
        resolve_branch_view_strict(events, raw.branch_view)
        view = raw.branch_view
        selected_events = [*raw.events, raw.target]
        validate_selected_generation_events(selected_events)
        reload_raw_prefix_for_target(
            candidate, target, branch_view=request.branch_view, strict_pairs=True
        )
        target_turn = target.turn_index
        user_position = None
        rebuilt = True
    elif request.branch_view and store is not None:
        preparation = prepare_selected_branch_context(
            agent_id=prefix,
            runtime_scope_id="detached-edit",
            raw_events=events,
            branch_view=request.branch_view,
            baseline_system_messages=tuple(m for m in messages if m.role == "system"),
        )
        conv = Conversation(replace(config, max_messages=0))
        for message in preparation.provider_conversation:
            conv.append_message(message)
        candidate.controller.conversation = conv
        candidate._parent_branch_path = list(preparation.parent_branch_path)
        live_ids = select_live_event_ids(events, branch_view=request.branch_view)
        selected_events = [e for e in events if e.get("event_id") in live_ids]
        rebuilt = True
    conv = candidate.controller.conversation
    msgs = conv.get_messages()
    idx = resolve_edit_message_index(
        candidate,
        msgs,
        request.message_idx,
        turn_index=target_turn,
        user_position=user_position,
        branch_view=view,
    )
    if idx is None or not is_turn_user_message(msgs[idx]):
        return None
    if target_turn is None:
        target_turn = _message_turn(
            msgs[idx], selected_events, persisted=store is not None
        )
    if target_turn is None:
        target_turn = turn
    retained_events = [
        event
        for event in selected_events
        if not isinstance(event.get("turn_index"), int)
        or event["turn_index"] < target_turn
    ]
    if (
        store is not None
        and target is None
        and any(
            event.get("type") in ("subagent_call", "subagent_result")
            for event in retained_events
        )
    ):
        targets = [
            event
            for event in selected_events
            if event.get("type") == "user_message"
            and event.get("turn_index") == target_turn
        ]
        if len(targets) != 1:
            raise ValueError(
                "Retained subagent history requires a unique raw edit target"
            )
        raw_target = targets[0]
        target = UserMessageSelector(
            raw_target["event_id"], target_turn, raw_target["branch_id"]
        )
        validate_selected_generation_events(retained_events)
        reload_raw_prefix_for_target(
            candidate, target, branch_view=view, strict_pairs=True
        )
        conv = candidate.controller.conversation
        msgs = conv.get_messages()
        idx = resolve_edit_message_index(
            candidate, msgs, -1, turn_index=target_turn, branch_view=view
        )
        if idx is None:
            raise ValueError("Rebuilt history lost its canonical edit target")
        rebuilt = True
    content = deepcopy(msgs[idx].content if request.regenerate else request.new_content)
    normalized = normalize_content_parts(content)
    if normalized is None:
        normalized = content if isinstance(content, str) else ""
    retained = tuple(msgs[:idx])
    admission = [*retained, {"role": "user", "content": content}]
    validate_history_compatibility(admission, source, current_source=source)
    validate_switch_history(retained, retained_events, events, source)
    new_branch = (
        max(
            [branch if target_turn == turn else 0]
            + [
                e["branch_id"]
                for e in events
                if e.get("turn_index") == target_turn
                and isinstance(e.get("branch_id"), int)
            ]
        )
        + 1
    )
    parent = tuple((t, b) for t, b in candidate._parent_branch_path if t < target_turn)
    conv.truncate_from(idx)
    return {
        "candidate": conv,
        "messages": retained,
        "index": idx,
        "rebuilt": rebuilt,
        "preserve_baseline": not any(
            e.get("type") == "system_prompt_set" for e in selected_events
        ),
        "turn": target_turn,
        "branch": new_branch,
        "path": parent,
        "source_view": dict(view),
        "content": content,
        "normalized": normalized,
    }
