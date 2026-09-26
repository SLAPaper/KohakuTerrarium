"""Branch selection and compaction helpers shared by history and replay."""

import bisect
import json
from collections.abc import Hashable, Mapping
from typing import Any, Iterable

from kohakuterrarium.core.job_label import canonical_tool_name, make_job_label

# Parent paths identify the selected branches of earlier turns. Legacy events
# derive this ancestry from event order when no explicit path was persisted.


def _coerce_path(raw: Any) -> tuple[tuple[int, int], ...]:
    """Normalize a parent_branch_path payload into a tuple of pairs.

    Accept JSON-friendly lists or tuples of integer pairs. Invalid or missing
    input produces an empty path.
    """
    if not raw:
        return ()
    out: list[tuple[int, int]] = []
    try:
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t, b = item
                if isinstance(t, int) and isinstance(b, int):
                    out.append((t, b))
    except TypeError:
        return ()
    return tuple(out)


def index_parent_paths(
    events_list: list[dict[str, Any]],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Map each event_id → its parent_branch_path.

    Explicit paths take precedence. Legacy events inherit the latest branch seen
    for each earlier turn at that point in event order.
    Public: shared by replay selection and ``session.resume`` branch-state
    restore.
    """
    paths: dict[int, tuple[tuple[int, int], ...]] = {}
    latest_by_turn: dict[int, int] = {}
    for evt in events_list:
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        eid = evt.get("event_id")
        explicit = _coerce_path(evt.get("parent_branch_path"))
        if isinstance(eid, int):
            if explicit:
                paths[eid] = explicit
            elif isinstance(ti, int):
                paths[eid] = tuple(
                    sorted(
                        ((t, b) for t, b in latest_by_turn.items() if t < ti),
                        key=lambda p: p[0],
                    )
                )
        if isinstance(ti, int) and isinstance(bi, int):
            prev = latest_by_turn.get(ti, 0)
            if bi > prev:
                latest_by_turn[ti] = bi
    return paths


def _path_matches(
    parent_path: tuple[tuple[int, int], ...],
    selected: dict[int, int],
) -> bool:
    """Return whether a parent path is compatible with selected branches.

    Path turns absent from ``selected`` remain unconstrained until resolution
    reaches them.
    """
    for t, b in parent_path:
        if t in selected and selected[t] != b:
            return False
    return True


def resolve_selected_branches(
    events_list: list[dict[str, Any]],
    parent_paths: dict[int, tuple[tuple[int, int], ...]],
    branch_view: dict[int, int] | None,
) -> dict[int, int]:
    """Pick a live branch for each turn while respecting nested paths.

    Turns resolve in ascending order. Valid overrides win; otherwise the highest
    compatible branch is selected. Turns with no compatible branch are omitted
    from the live subtree.
    Public: shared by replay selection and ``session.resume`` branch-state
    restore.
    """
    branches_by_turn: dict[int, list[tuple[int, int]]] = {}
    for evt in events_list:
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        eid = evt.get("event_id")
        if not isinstance(ti, int) or not isinstance(bi, int):
            continue
        path = parent_paths.get(eid, ()) if isinstance(eid, int) else ()
        bucket = branches_by_turn.setdefault(ti, [])
        if not any(b == bi for _, b in bucket):
            bucket.append((path, bi))

    selected: dict[int, int] = {}
    override = dict(branch_view or {})
    for ti in sorted(branches_by_turn.keys()):
        candidates = [
            (path, bi)
            for path, bi in branches_by_turn[ti]
            if _path_matches(path, selected)
        ]
        if not candidates:
            continue
        if ti in override:
            requested = override[ti]
            match = next(
                ((path, bi) for path, bi in candidates if bi == requested), None
            )
            if match is not None:
                selected[ti] = match[1]
                continue
        selected[ti] = max(bi for _, bi in candidates)
    return selected


class InvalidBranchViewError(ValueError):
    """Raised when a requested branch view cannot identify a coherent path."""


def _coerce_branch_view(branch_view: dict[int, int] | None) -> dict[int, int]:
    """Normalize branch selections without accepting lossy values."""
    if branch_view is None:
        return {}
    if not isinstance(branch_view, dict):
        raise InvalidBranchViewError("branch_view must be a mapping")

    selected: dict[int, int] = {}
    for raw_turn, raw_branch in branch_view.items():
        values = (raw_turn, raw_branch)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, str))
            or (
                isinstance(value, str)
                and (not value.isdigit() or str(int(value)) != value)
            )
            for value in values
        ):
            raise InvalidBranchViewError("branch_view keys and values must be integers")
        turn_index = int(raw_turn)
        branch_id = int(raw_branch)
        if turn_index < 1 or branch_id < 1:
            raise InvalidBranchViewError("turn indices and branch ids must be positive")
        selected[turn_index] = branch_id
    return selected


def resolve_branch_view_strict(
    events: Iterable[dict[str, Any]],
    branch_view: dict[int, int] | None,
) -> dict[int, int]:
    """Validate a branch view and return its authoritative branch projection."""
    events_list = list(events)
    requested = _coerce_branch_view(branch_view)
    parent_paths = index_parent_paths(events_list)

    pairs: set[tuple[int, int]] = set()
    pair_paths: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    candidates: dict[int, list[tuple[tuple[tuple[int, int], ...], int]]] = {}
    for evt in events_list:
        if evt.get("type") not in ("user_message", "user_input"):
            continue
        try:
            pair = (int(evt.get("turn_index")), int(evt.get("branch_id")))
        except (TypeError, ValueError):
            continue
        event_id = evt.get("event_id")
        path = (
            parent_paths.get(event_id, ())
            if isinstance(event_id, int)
            else _coerce_path(evt.get("parent_branch_path"))
        )
        pairs.add(pair)
        pair_paths.setdefault(pair, path)
        candidate = (path, pair[1])
        if candidate not in candidates.setdefault(pair[0], []):
            candidates[pair[0]].append(candidate)

    for pair in requested.items():
        if pair not in pairs:
            raise InvalidBranchViewError(
                f"branch {pair[1]} does not exist for turn {pair[0]}"
            )

    constraints = dict(requested)
    pending = list(requested.items())
    visited: set[tuple[int, int]] = set()
    while pending:
        pair = pending.pop()
        if pair in visited:
            continue
        visited.add(pair)
        for parent_turn, parent_branch in pair_paths.get(pair, ()):
            existing = constraints.get(parent_turn)
            if existing is not None and existing != parent_branch:
                raise InvalidBranchViewError(
                    f"branch {pair[1]} at turn {pair[0]} is incompatible with "
                    f"branch {existing} at turn {parent_turn}"
                )
            parent_pair = (parent_turn, parent_branch)
            if parent_pair not in pairs:
                raise InvalidBranchViewError(
                    f"branch {pair[1]} at turn {pair[0]} references missing "
                    f"branch {parent_branch} at turn {parent_turn}"
                )
            if existing is None:
                constraints[parent_turn] = parent_branch
                pending.append(parent_pair)

    selected = dict(constraints)
    for turn_index in sorted(candidates):
        required = selected.get(turn_index)
        compatible = [
            branch_id
            for path, branch_id in candidates[turn_index]
            if _path_matches(path, selected)
            and (required is None or branch_id == required)
        ]
        if required is not None and not compatible:
            raise InvalidBranchViewError(
                f"branch {required} at turn {turn_index} is incompatible with the view"
            )
        if compatible:
            selected[turn_index] = max(compatible)
    return selected


def project_branch_metadata(
    events: Iterable[dict[str, Any]],
    branch_view: dict[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Project branch choices, ancestry, and the selected coherent path."""
    events_list = list(events)
    selected = resolve_branch_view_strict(events_list, branch_view)
    parent_paths = index_parent_paths(events_list)
    branches: dict[int, dict[int, set[tuple[tuple[int, int], ...]]]] = {}

    for evt in events_list:
        if evt.get("type") not in ("user_message", "user_input"):
            continue
        try:
            turn_index = int(evt.get("turn_index"))
            branch_id = int(evt.get("branch_id"))
        except (TypeError, ValueError):
            continue
        event_id = evt.get("event_id")
        path = (
            parent_paths.get(event_id, ())
            if isinstance(event_id, int)
            else _coerce_path(evt.get("parent_branch_path"))
        )
        branches.setdefault(turn_index, {}).setdefault(branch_id, set()).add(path)

    return {
        turn_index: {
            "branches": [
                {
                    "branch_id": branch_id,
                    "parent_branch_paths": [
                        [
                            [parent_turn, parent_branch]
                            for parent_turn, parent_branch in path
                        ]
                        for path in sorted(paths)
                    ],
                    "selected": selected.get(turn_index) == branch_id,
                }
                for branch_id, paths in sorted(branches_by_id.items())
            ],
            "latest": max(branches_by_id),
            "selected": selected.get(turn_index),
        }
        for turn_index, branches_by_id in sorted(branches.items())
    }


def collect_branch_metadata(
    events: Iterable[dict[str, Any]],
    *,
    branch_view: dict[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Extract per-turn branch metadata from an event stream.

    Events lacking turn or branch identifiers are ignored. With ``branch_view``,
    only branches compatible with selected prior ancestry are reported, so
    navigator counts reflect the visible subtree.
    """
    events_list = list(events)
    parent_paths = index_parent_paths(events_list)
    selected = resolve_selected_branches(events_list, parent_paths, branch_view)

    out: dict[int, dict[str, Any]] = {}
    for evt in events_list:
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        eid = evt.get("event_id")
        if not isinstance(ti, int) or not isinstance(bi, int):
            continue
        path = parent_paths.get(eid, ()) if isinstance(eid, int) else ()
        # Navigator counts must remain local to the selected ancestry.
        prior_selected = {t: b for t, b in selected.items() if t < ti}
        if not _path_matches(path, prior_selected):
            continue
        bucket = out.setdefault(
            ti, {"branches": [], "latest_branch": 0, "events_by_branch": {}}
        )
        if bi not in bucket["events_by_branch"]:
            bucket["events_by_branch"][bi] = []
            bucket["branches"].append(bi)
        if isinstance(eid, int):
            bucket["events_by_branch"][bi].append(eid)
        if bi > bucket["latest_branch"]:
            bucket["latest_branch"] = bi
    for bucket in out.values():
        bucket["branches"].sort()
    return out


def collect_user_groups(
    events: Iterable[dict[str, Any]],
    *,
    branch_view: dict[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-turn grouping of branches by ``user_message`` content.

    Identical user content groups response regenerations together; distinct user
    content represents edited alternatives. Each turn reports its groups and the
    selected group index.
    """
    events_list = list(events)
    meta = collect_branch_metadata(events_list, branch_view=branch_view)
    parent_paths = index_parent_paths(events_list)
    selected = resolve_selected_branches(events_list, parent_paths, branch_view)
    contents: dict[int, dict[int, str]] = {}
    for evt in events_list:
        if evt.get("type") not in ("user_message", "user_input"):
            continue
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        if not isinstance(ti, int) or not isinstance(bi, int):
            continue
        contents.setdefault(ti, {})
        if bi not in contents[ti]:
            c = evt.get("content")
            contents[ti][bi] = c if isinstance(c, str) else str(c)
    out: dict[int, dict[str, Any]] = {}
    for ti, info in meta.items():
        groups: list[dict[str, Any]] = []
        for branch in info["branches"]:
            content = contents.get(ti, {}).get(branch, "")
            existing = next((g for g in groups if g["content"] == content), None)
            if existing is None:
                groups.append({"content": content, "branches": [branch]})
            else:
                existing["branches"].append(branch)
        sel = selected.get(ti)
        sel_idx = next((i for i, g in enumerate(groups) if sel in g["branches"]), 0)
        out[ti] = {"groups": groups, "selected_group_idx": sel_idx}
    return out


def select_live_event_ids(
    events: Iterable[dict[str, Any]],
    *,
    branch_view: dict[int, int] | None = None,
) -> set[int]:
    """Return the event_ids that belong to the live subtree.

    Live events belong to the selected branch and compatible prior ancestry.
    Legacy or non-state events without branch metadata remain live. Without an
    override, the latest compatible branch is selected at every turn.
    """
    events_list = list(events)
    parent_paths = index_parent_paths(events_list)
    selected = resolve_selected_branches(events_list, parent_paths, branch_view)

    # ``selected`` only contains turns present in ``branches_by_turn`` (see
    # ``resolve_selected_branches``), so only its prefix up to the maximal
    # constrained turn in each parent path matters. Building that prefix per
    # event costs O(longest path) instead of rebuilding the full
    # ``{t: b for t < ti}`` dict per event.
    # ``selected`` only contains turns present in ``branches_by_turn`` (see
    # ``resolve_selected_branches``), so only its prefix up to the maximal
    # constrained turn in each parent path matters. ``selected_turns`` is
    # sorted, and the clamped bound lets bisect slice the needed prefix
    # directly — the per-event cost is O(path length), not O(#turns).
    selected_turns = sorted(selected.keys())
    live: set[int] = set()
    for evt in events_list:
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        eid = evt.get("event_id")
        if not isinstance(eid, int):
            continue
        if not isinstance(ti, int) or not isinstance(bi, int):
            live.add(eid)
            continue
        if selected.get(ti) != bi:
            continue
        path = parent_paths.get(eid, ())
        if path:
            bound = min(ti, max(t for t, _ in path) + 1)
            idx = bisect.bisect_left(selected_turns, bound)
            prefix = {t: selected[t] for t in selected_turns[:idx]}
            if not _path_matches(path, prefix):
                continue
        live.add(eid)
    return live


def _event_signature_value(value: Any) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def dedupe_adjacent_duplicate_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse identical adjacent persisted events.

    Duplicate sink attachment can persist equivalent neighboring rows with distinct
    identifiers and timestamps. Replay consumers collapse them while the raw log
    remains unchanged.
    """
    out: list[dict[str, Any]] = []
    previous_signature: tuple[tuple[str, Hashable], ...] | None = None
    for evt in events:
        signature = tuple(
            sorted(
                (k, _event_signature_value(v))
                for k, v in evt.items()
                if k not in {"event_id", "ts"}
            )
        )
        if signature == previous_signature:
            continue
        out.append(evt)
        previous_signature = signature
    return out


def _clean_tool_name(evt: dict) -> str:
    """Prefer recorded identity, repairing legacy labels or missing names."""
    name = evt.get("tool_name") or evt.get("name", "")
    if isinstance(name, str) and name:
        return canonical_tool_name(name)
    call_id = evt.get("call_id") or evt.get("job_id") or ""
    if call_id:
        tool_name, _ = make_job_label(str(call_id))
        if tool_name:
            return tool_name
    return name if isinstance(name, str) else ""


def compact_path_from_event(evt: Mapping[str, Any]) -> set[tuple[int, int]] | None:
    """Parse a compact event's ``compact_path`` (list of [turn, branch]).

    Returns ``None`` when absent — the legacy v1_to_v2 migration product has
    no path, so replay falls back to the global id-range behaviour for it.
    Public: shared by ``session.history`` (replay) and
    ``core.agent_raw_history`` (edit reload baseline selection).
    """
    raw = evt.get("compact_path")
    if not isinstance(raw, (list, tuple)):
        return None
    out: set[tuple[int, int]] = set()
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                out.add((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    return out or None


def compact_replay_events(events, compacts, view):
    """Keep summary anchors without synthesizing calls hidden by the baseline."""
    selected = set(view.items())
    rules = [
        (
            event["replaced_from_event_id"],
            event["replaced_to_event_id"],
            compact_path_from_event(event),
            event.get("summary", ""),
        )
        for event in compacts
        if type(event.get("replaced_from_event_id")) is int
        and type(event.get("replaced_to_event_id")) is int
    ]
    eligible = [rule for rule in rules if _rule_priority(rule, selected)[0] > 0]
    if not eligible:
        return events
    _, rule = max(
        enumerate(eligible),
        key=lambda item: (*_rule_priority(item[1], selected), item[0]),
    )
    return [
        (
            {**event, "type": "text_chunk", "content": ""}
            if event.get("type") in ("tool_call", "tool_result", "assistant_tool_calls")
            and _covered_by_replace(
                event.get("event_id"),
                event.get("turn_index"),
                event.get("branch_id"),
                [rule],
            )
            else event
        )
        for event in events
    ]


def _covered_by_replace(
    eid: object,
    turn_index: object,
    branch_id: object,
    rules: list[tuple[int, int, set[tuple[int, int]] | None, str]],
) -> bool:
    """Whether event ``eid`` is covered by a compaction replacement rule.

    A rule is ``(replaced_from, replaced_to, compact_path_or_None, summary)``.
    With a path, only events on that path are covered (branch-aware); without
    a path (legacy migration data) the whole id range is covered.
    """
    if not isinstance(eid, int):
        return False
    for frm, to, path, _summary in rules:
        if not (frm <= eid <= to):
            continue
        if path is None:
            return True
        if isinstance(turn_index, int) and isinstance(branch_id, int):
            if (turn_index, branch_id) in path:
                return True
    return False


def _rule_priority(
    rule: tuple[int, int, set[tuple[int, int]] | None, str],
    selected_pairs: set[tuple[int, int]],
) -> tuple[int, int]:
    """Rank a compaction rule against a selected branch projection.

    Rank 2: a path-carrying rule whose ENTIRE path lies on the selected
    lineage (``path ⊆ selected``) — the summary covers only this branch's own
    turns, and a deeper path (more turns) is authoritative for the prefix it
    covers. Rank 1: a legacy pathless rule (fallback baseline). Rank 0: a
    sibling/incompatible rule whose summary covers turns outside the selected
    branch — its summary carries sibling-specific content, not just the shared
    ancestor, so it must never leak into this branch's replay.
    """
    _frm, _to, path, _summary = rule
    if path is None:
        return (1, 0)
    if not path.issubset(selected_pairs):
        return (0, 0)
    return (2, len(path))
