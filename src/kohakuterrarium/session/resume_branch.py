"""Branch-matching helpers for resume (kept small for the file-size guard).

Split out of :mod:`resume` so it stays under the line limit. These helpers
decide whether a saved conversation snapshot still matches the branch the
agent resumes onto, or must be rebuilt from the event log.
"""

from typing import Any

from kohakuterrarium.core.conversation_elide import TOOL_FEEDBACK_KIND
from kohakuterrarium.session.history import select_live_event_ids
from kohakuterrarium.session.resume_admission import validate_resume_snapshot
from kohakuterrarium.session.resume_reconciliation import reconcile_snapshot
from kohakuterrarium.session.resume_replay import replay_resume_history
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def snapshot_has_turn_metadata(snapshot: list[dict]) -> bool:
    """Return whether the snapshot carries valid user turn metadata.

    Edit/regenerate targeting resolves the edited turn through message
    metadata; legacy snapshots saved without it force content matching,
    which is ambiguous when a turn's wording repeats. Such snapshots are
    backfilled from the event log so targeting stays deterministic.

    A metadata entry is only "valid" when ``turn_index`` and ``branch_id``
    are positive ints — the exact shape every writer produces (backfill,
    replay, live snapshot). A non-int (corrupt/legacy-typed) value would be
    trusted here but fail edit targeting, so it must trigger backfill too.
    A non-dict entry is malformed data that ``_build_conversation`` would
    crash on, so it must also trigger backfill/replay rather than being
    trusted.
    """
    return all(
        isinstance(m, dict)
        and (
            m.get("role") != "user"
            or (
                isinstance(m.get("metadata"), dict)
                and m["metadata"].get("kind") == TOOL_FEEDBACK_KIND
            )
            or (
                isinstance(m.get("metadata"), dict)
                and isinstance(m["metadata"].get("turn_index"), int)
                and m["metadata"]["turn_index"] > 0
                and isinstance(m["metadata"].get("branch_id"), int)
                and m["metadata"]["branch_id"] > 0
            )
        )
        for m in snapshot
    )


def backfill_turn_metadata(
    snapshot: list[dict],
    events: list[dict],
    branch_view: dict[int, int] | None = None,
) -> list[dict]:
    """Backfill user-turn metadata onto a legacy metadata-less snapshot.

    The snapshot is the canonical persisted state (compaction exists only
    there), so it is trusted verbatim: a full replay would resurrect the
    pre-compact history and drop snapshot-only in-flight messages. Turn
    identity is recovered from the live ``user_message`` events so edit
    targeting stays deterministic.

    Snapshot user messages hold the most recent turns verbatim (compaction
    summarizes the prefix and keeps only the live zone), so they map to the
    LAST ``user_message`` events, not the first ones.
    """
    live_ids = set(select_live_event_ids(events, branch_view=branch_view))
    meta_by_pos: list[dict] = []
    pos_by_key: dict[tuple[int, int], int] = {}
    type_by_key: dict[tuple[int, int], str] = {}
    for evt in events:
        if evt.get("type") not in ("user_message", "user_input"):
            continue
        if isinstance(evt.get("event_id"), int) and evt["event_id"] not in live_ids:
            continue
        ti = evt.get("turn_index")
        bi = evt.get("branch_id")
        if not isinstance(ti, int) or not isinstance(bi, int):
            continue
        key = (ti, bi)
        etype = evt.get("type", "")
        meta = {
            # Only a user_message carries an editable event_id; a legacy
            # user_input's id is NOT valid for edit targeting
            # (select_raw_history_prefix rejects non-user_message targets),
            # so leave it unset when the chosen source is user_input.
            "event_id": evt.get("event_id") if etype == "user_message" else None,
            "turn_index": ti,
            "branch_id": bi,
        }
        if key in pos_by_key:
            # A turn usually carries BOTH user_input (written first) and
            # user_message (the canonical editable event). Edit targeting
            # requires event_id to point at a user_message, so if we recorded
            # a user_input first, replace it with the user_message event_id.
            # A duplicate user_message must NOT overwrite the first one.
            if etype == "user_message" and type_by_key[key] != "user_message":
                meta_by_pos[pos_by_key[key]] = meta
                type_by_key[key] = etype
            continue
        type_by_key[key] = etype
        pos_by_key[key] = len(meta_by_pos)
        meta_by_pos.append(meta)
    user_messages = [
        m
        for m in snapshot
        if isinstance(m, dict)
        and m.get("role") == "user"
        and not (
            isinstance(m.get("metadata"), dict)
            and m["metadata"].get("kind") == TOOL_FEEDBACK_KIND
        )
    ]
    tail_meta = meta_by_pos[-len(user_messages) :] if user_messages else []
    out: list[dict] = []
    for msg in snapshot:
        if not isinstance(msg, dict):
            # Malformed snapshot entry (corrupt data): keep it verbatim so
            # backfill never crashes here; resume's downstream build drops or
            # tolerates it rather than trusting a broken message shape.
            out.append(msg)
            continue
        m = dict(msg)
        raw_metadata = m.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        is_canonical_user = (
            m.get("role") == "user" and metadata.get("kind") != TOOL_FEEDBACK_KIND
        )
        if is_canonical_user and tail_meta:
            meta = dict(metadata)
            # Position-aligned: every user message consumes one tail slot so
            # later messages keep their correct turn. Replace the canonical
            # identity fields whenever turn or branch is not a positive int
            # (missing, corrupt, or legacy-typed) — a partial/valid-looking
            # value must not survive and later fail edit targeting.
            ti = meta.get("turn_index")
            bi = meta.get("branch_id")
            if not (isinstance(ti, int) and ti > 0 and isinstance(bi, int) and bi > 0):
                meta["turn_index"] = tail_meta[0]["turn_index"]
                meta["branch_id"] = tail_meta[0]["branch_id"]
                if tail_meta[0].get("event_id") is not None:
                    meta["event_id"] = tail_meta[0]["event_id"]
                else:
                    meta.pop("event_id", None)
                m["metadata"] = meta
            tail_meta = tail_meta[1:]
        out.append(m)
    return out


def is_path_prefix(sub: list[tuple[int, int]], full: list[tuple[int, int]]) -> bool:
    """Whether ``sub`` is a strict/equal prefix of ``full``."""
    return len(sub) <= len(full) and full[: len(sub)] == sub


def _safe_branch_path(raw: Any) -> list[tuple[int, int]]:
    """Parse a persisted branch path defensively.

    Malformed entries (int, string, wrong length, non-int coords) are
    skipped so resume never crashes on corrupt state.
    """
    out: list[tuple[int, int]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            t, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        out.append((t, b))
    return out


def snapshot_mismatches_branch(
    store: Any,
    agent: Any,
    agent_name: str,
    *,
    branch_view: dict[int, int] | None = None,
) -> bool:
    """Whether the saved snapshot belongs to a different branch than the one
    resume lands on.

    A snapshot tagged with a branch that is an ANCESTOR of the target branch
    is still usable — resume appends the post-snapshot tail. Only a snapshot
    whose path diverges (a sibling branch) must be discarded and rebuilt.

    When ``branch_view`` is the resolved selected map, the target path is
    that map — never the agent's pre-restore coordinates, which still point
    at latest until restore runs.
    """
    try:
        key = f"{agent_name}:snapshot_branch"
        if key not in store.state:
            return False
        branch = store.state.get(key)
    except (KeyError, TypeError):
        return False
    if not isinstance(branch, dict):
        return False  # caller fail-closes claimed-but-unreadable tags
    ti = branch.get("turn_index")
    bi = branch.get("branch_id")
    if not isinstance(ti, int) or not isinstance(bi, int) or ti <= 0 or bi <= 0:
        return False
    if branch_view:
        turns = sorted(branch_view)
        a_ti = turns[-1]
        a_bi = branch_view[a_ti]
        a_ppath = [(turn, branch_view[turn]) for turn in turns if turn < a_ti]
        # A compacted ancestor is on the saved path but not in the live view.
        # Compare only turns the view still contains; dropping those ancestors
        # is not a sibling mismatch and must not discard the snapshot.
        viewed = set(branch_view)
        snapshot_parent = [
            pair
            for pair in _safe_branch_path(branch.get("parent_branch_path"))
            if pair[0] in viewed
        ]
        snapshot_path = snapshot_parent + [(ti, bi)]
        agent_path = _safe_branch_path(a_ppath) + [(a_ti, a_bi)]
        return not is_path_prefix(snapshot_path, agent_path)
    else:
        a_ti = getattr(agent, "_turn_index", None)
        a_bi = getattr(agent, "_branch_id", None)
        a_ppath = getattr(agent, "_parent_branch_path", None) or []
        # An agent with no live selection (core code can set turn/branch to 0)
        # has no branch to compare against — trust the snapshot.
        if (
            not isinstance(a_ti, int)
            or not isinstance(a_bi, int)
            or a_ti <= 0
            or a_bi <= 0
        ):
            return False
    snapshot_path = _safe_branch_path(branch.get("parent_branch_path")) + [(ti, bi)]
    agent_path = _safe_branch_path(a_ppath) + [(a_ti, a_bi)]
    return not is_path_prefix(snapshot_path, agent_path)


def replayed_messages_for(
    store: Any,
    agent_name: str,
    events: list[dict] | None = None,
    branch_view: dict[int, int] | None = None,
) -> list[dict]:
    """Replay the selected live subtree from the event log (branch-aware).

    Used by resume when the saved snapshot belongs to a different branch.
    ``events`` is the caller's shared raw read; without it the store is
    read here. ``branch_view`` pins the replay to the resolved selection
    instead of the latest compatible subtree.
    """
    if events is None:
        try:
            events = list(store.get_events(agent_name))
        except Exception:  # pragma: no cover - defensive
            return []
    if not events:
        return []
    return replay_resume_history(events, branch_view)


def load_conversation_with_replay_fallback(
    store: Any,
    agent_name: str,
    events: list[dict] | None = None,
    branch_view: dict[int, int] | None = None,
    *,
    origins: dict | None = None,
) -> list[dict] | None:
    """Load the conversation snapshot and replay events when it is stale.

    Post-snapshot events are appended when branch ancestry is unchanged; new
    branch forks require a full replay to preserve coherent selection.
    ``events`` (when given) is the caller's single raw read of the event
    log, shared across the resume helpers so one scan serves them all;
    without it the store is read here.

    ``branch_view`` is the resolved selected map (sparse overrides already
    expanded). It filters backfill, tail replay, and full replay so a
    non-latest selection never loads the latest coordinates. ``None`` keeps
    latest-compatible selection, including legacy snapshots that have no view.

    A compacted snapshot is kept when its own turn metadata still lies on
    the selected view, even if raw events for the compacted prefix are gone.
    The view existing is not itself a reason to discard that snapshot.

    Deliberately NO counter-based freshness shortcut: the persisted event
    counter is NOT an upper bound of the event table. ``append_event``
    flushes the events cache BEFORE ``persist_event_counter`` runs, and
    that persist swallows failures (session/store.py), so the stored
    counter can lag the table by any amount. A snapshot watermarked at a
    lagging counter would then pass a ``cached_up_to >= counter`` check
    while newer events exist, silently dropping the tail replay.
    """
    snapshot = store.load_conversation(agent_name)
    if events is None:
        events = store.get_events(agent_name)
    if origins is not None and snapshot is not None:
        return _load_validated_snapshot(
            store, agent_name, snapshot, events, branch_view, origins
        )
    if not events:
        return snapshot
    last_event_id = 0
    for evt in events:
        eid = evt.get("event_id")
        if isinstance(eid, int) and eid > last_event_id:
            last_event_id = eid
    try:
        cached_up_to = store.state.get(f"{agent_name}:snapshot_event_id")
    except (KeyError, TypeError):
        cached_up_to = None
    if snapshot is not None and isinstance(cached_up_to, int):
        if cached_up_to >= last_event_id:
            if snapshot_has_turn_metadata(snapshot):
                return snapshot
            logger.info(
                "Legacy snapshot lacks turn metadata — backfill",
                agent=agent_name,
            )
            return backfill_turn_metadata(snapshot, events, branch_view)
        # Compaction exists only in the snapshot, so replay just its normalized tail.
        tail = [
            evt
            for evt in events
            if isinstance(evt.get("event_id"), int) and evt["event_id"] > cached_up_to
        ]
        # A new post-snapshot branch can supersede earlier turns; appending it
        # to the snapshot would retain incompatible history. With a resolved
        # view, "fork" means the tail (or the snapshot's own coordinates)
        # leaves that view — not merely "some branch id > 1 exists", which is
        # exactly the non-latest selection this resume must keep.
        if branch_view is not None:
            tail_has_forks = _selection_leaves_view(snapshot, tail, branch_view)
        else:
            pre_pairs = {
                (evt.get("turn_index"), evt.get("branch_id"))
                for evt in events
                if isinstance(evt.get("event_id"), int)
                and evt["event_id"] <= cached_up_to
            }
            tail_has_forks = any(
                isinstance(evt.get("branch_id"), int)
                and evt["branch_id"] > 1
                and (evt.get("turn_index"), evt["branch_id"]) not in pre_pairs
                for evt in tail
            )
        if not tail_has_forks:
            appended = replay_resume_history(tail, branch_view)
            # A legacy snapshot portion must be backfilled too, otherwise the
            # resumed conversation becomes a mix of metadata-less (snapshot)
            # and metadata-bearing (tail) user messages. Backfill uses only
            # events up to the snapshot watermark so the tail's user turns do
            # not shift the mapping.
            base = snapshot
            if not snapshot_has_turn_metadata(snapshot):
                pre_events = [
                    evt
                    for evt in events
                    if isinstance(evt.get("event_id"), int)
                    and evt["event_id"] <= cached_up_to
                ]
                base = backfill_turn_metadata(snapshot, pre_events, branch_view)
            logger.info(
                "Resume appended post-snapshot tail",
                agent=agent_name,
                snapshot_event_id=cached_up_to,
                last_event_id=last_event_id,
                appended=len(appended),
            )
            return list(base) + appended
        logger.info(
            "Post-snapshot tail contains branch forks — full replay",
            agent=agent_name,
            snapshot_event_id=cached_up_to,
        )
    if snapshot is not None and cached_up_to is None:
        if snapshot_has_turn_metadata(snapshot):
            return snapshot
        logger.info(
            "Legacy snapshot lacks turn metadata — backfill",
            agent=agent_name,
        )
        return backfill_turn_metadata(snapshot, events, branch_view)
    replayed = replay_resume_history(events, branch_view)
    if replayed:
        logger.info(
            "Resume rebuilt conversation via replay",
            agent=agent_name,
            snapshot_event_id=cached_up_to,
            last_event_id=last_event_id,
            messages=len(replayed),
        )
        return replayed
    return snapshot


def _load_validated_snapshot(store, name, snapshot, events, view, origins):
    watermark = store.state.get(f"{name}:snapshot_event_id")
    selected_ids = select_live_event_ids(events, branch_view=view)
    selected = [
        event
        for event in events
        if not isinstance(event.get("event_id"), int)
        or event["event_id"] in selected_ids
    ]
    tail = [
        event
        for event in selected
        if type(watermark) is int
        and type(event.get("event_id")) is int
        and event["event_id"] > watermark
    ]
    if view is not None:
        leaves_view = _selection_leaves_view(snapshot, tail, view)
    else:
        prior = {
            (event.get("turn_index"), event.get("branch_id"))
            for event in selected
            if type(watermark) is int
            and type(event.get("event_id")) is int
            and event["event_id"] <= watermark
        }
        leaves_view = any(
            type(event.get("branch_id")) is int
            and event["branch_id"] > 1
            and (event.get("turn_index"), event["branch_id"]) not in prior
            for event in tail
        )
    if leaves_view:
        return replay_resume_history(events, view)
    validate_resume_snapshot(snapshot, origins)
    base, consumed = reconcile_snapshot(snapshot, selected, watermark)
    if not snapshot_has_turn_metadata(base):
        prior_events = [
            event
            for event in selected
            if watermark is None
            or type(event.get("event_id")) is int
            and event["event_id"] <= watermark
        ]
        base = backfill_turn_metadata(base, prior_events, view)
    remaining = [event for event in tail if event.get("event_id") not in consumed]
    if remaining:
        base = list(base) + replay_resume_history(remaining, view)
    validate_resume_snapshot(base, origins)
    return base


def _selection_leaves_view(
    snapshot: list[dict],
    tail: list[dict],
    branch_view: dict[int, int],
) -> bool:
    """Whether snapshot or tail coordinates leave the resolved selection.

    A snapshot turn absent from the view (compacted prefix whose raw events
    are gone) is not a fork. A snapshot turn the view selects differently,
    or a tail event on another branch of a viewed turn, is.
    """
    for message in snapshot:
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("kind") == TOOL_FEEDBACK_KIND:
            continue
        turn = metadata.get("turn_index")
        branch = metadata.get("branch_id")
        if not isinstance(turn, int) or not isinstance(branch, int):
            continue
        selected = branch_view.get(turn)
        if selected is not None and selected != branch:
            return True
    for event in tail:
        turn = event.get("turn_index")
        branch = event.get("branch_id")
        if not isinstance(turn, int) or not isinstance(branch, int):
            continue
        selected = branch_view.get(turn)
        if selected is not None and selected != branch:
            return True
        if selected is None and branch > 1:
            # A brand-new turn after the selected tip is a continuation
            # (branch 1). A sibling (>1) of an unviewed turn is a fork.
            return True
    return False
