"""Resolve persisted branch selection against raw event ancestry."""

from typing import Any

from kohakuterrarium.errors import SessionError
from kohakuterrarium.session.generation_validation import (
    validate_selected_generation_events,
)
from kohakuterrarium.session.history_generation import generation_payload
from kohakuterrarium.session.history import (
    InvalidBranchViewError,
    index_parent_paths,
    resolve_branch_view_strict,
    resolve_selected_branches,
    select_live_event_ids,
)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def persisted_branch_view(store: Any, agent_name: str) -> dict[int, int] | None:
    """Parse a selected snapshot's sparse branch view.

    ``None`` means the snapshot has no view (legacy tag or no tag). A view
    key that is present but not a coherent positive-int map is a claimed
    selection and fails closed — resume must not fall through to latest.
    """
    branch = _snapshot_branch(store, agent_name)
    if branch is None or "branch_view" not in branch:
        return None
    raw = branch.get("branch_view")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SessionError(
            f"snapshot_branch for {agent_name!r} has a non-mapping branch_view"
        )
    view: dict[int, int] = {}
    for raw_turn, raw_branch in raw.items():
        if not _positive_int(raw_turn) or not _positive_int(raw_branch):
            raise SessionError(
                f"snapshot_branch for {agent_name!r} has a non-integer branch_view"
            )
        view[int(raw_turn)] = int(raw_branch)
    return view or None


def _snapshot_branch(store: Any, agent_name: str) -> dict | None:
    """Return the persisted snapshot tag, or None when it was never written."""
    key = f"{agent_name}:snapshot_branch"
    try:
        state = store.state
        if key not in state:
            return None
        branch = state.get(key)
    except (KeyError, TypeError):
        return None
    if not isinstance(branch, dict):
        raise SessionError(f"snapshot_branch for {agent_name!r} is not a mapping")
    return branch


def resolve_resume_branch_view(
    events: list[dict],
    store: Any,
    agent_name: str,
) -> dict[int, int] | None:
    """Resolve the selected raw map resume must use everywhere.

    A snapshot that records ``branch_view`` is a claimed selection. The
    sparse overrides are expanded against actual raw ancestry BEFORE any
    normalize/replay, and the expanded map must contain the snapshot's
    terminal coordinates. Inconsistent view or ancestry raises
    :class:`SessionError` instead of silently restoring latest.

    ``None`` means "no selected view": callers keep latest-compatible
    selection. Legacy tags without a view stay on that path.
    """
    persisted = persisted_branch_view(store, agent_name)
    if persisted is None:
        return None
    try:
        selected = resolve_branch_view_strict(events, persisted)
    except InvalidBranchViewError as exc:
        raise SessionError(
            f"snapshot_branch for {agent_name!r} does not match raw ancestry: {exc}"
        ) from exc
    branch = _snapshot_branch(store, agent_name) or {}
    turn = branch.get("turn_index")
    branch_id = branch.get("branch_id")
    if _positive_int(turn) and _positive_int(branch_id):
        if selected.get(int(turn)) != int(branch_id):
            raise SessionError(
                f"snapshot_branch for {agent_name!r} selects "
                f"({turn}, {branch_id}) but the view resolves to "
                f"{selected.get(int(turn))!r}"
            )
    parent = branch.get("parent_branch_path")
    if isinstance(parent, (list, tuple)):
        for item in parent:
            if not (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and _positive_int(item[0])
                and _positive_int(item[1])
            ):
                continue
            resolved = selected.get(int(item[0]))
            if resolved is not None and resolved != int(item[1]):
                raise SessionError(
                    f"snapshot_branch for {agent_name!r} parent path "
                    f"disagrees with the resolved view at turn {item[0]}"
                )
    return selected


def validate_selected_raw_events(
    events: list[dict],
    branch_view: dict[int, int] | None,
) -> dict[str, dict]:
    """Check generation provenance and return the selected origin declarations.

    Runs before normalize/replay, which can drop the links this validates.
    History without generation metadata is a no-op.
    """
    if not events:
        return {}
    live_ids = select_live_event_ids(events, branch_view=branch_view)
    selected = [
        event
        for event in events
        if not (
            isinstance(event.get("event_id"), int)
            and not isinstance(event.get("event_id"), bool)
            and event["event_id"] not in live_ids
        )
    ]
    validate_selected_generation_events(selected)
    return {
        event["generation_id"]: generation_payload(event)
        for event in selected
        if event.get("type") == "assistant_generation_start"
    }


def apply_branch_coordinates(
    agent: Any,
    selected: dict[int, int],
) -> None:
    """Install one resolved view as turn, branch, parent path, and view."""
    if not selected:
        return
    max_turn = max(selected)
    agent._turn_index = max_turn
    agent._branch_id = selected[max_turn]
    agent._parent_branch_path = [
        (turn, selected[turn]) for turn in sorted(selected) if turn < max_turn
    ]
    agent._branch_view = dict(selected)


def latest_selected_branches(events: list[dict]) -> dict[int, int]:
    """Latest compatible live subtree (no persisted override)."""
    events_list = list(events)
    return resolve_selected_branches(events_list, index_parent_paths(events_list), None)
