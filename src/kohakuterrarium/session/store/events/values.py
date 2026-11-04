"""Private event-value iteration within a store-owned read operation (blocking)."""

from collections.abc import Iterator, Sequence
from typing import Any

from kohakuterrarium.session.history.ancestry import BranchAncestry
from kohakuterrarium.session.vault.io import read_values


def _iter_event_values(
    table: Any,
    keys: Sequence[Any],
    since_event_id: int | None = None,
    *,
    shared_paths: bool = False,
) -> Iterator[dict]:
    """Read supplied physical keys, logging and skipping unreadable values."""
    tree = BranchAncestry() if shared_paths else None
    for _, evt in read_values(
        table, keys, what="event", max_chunk_size=64 if shared_paths else None
    ):
        if since_event_id is not None:
            eid = evt.get("event_id") if isinstance(evt, dict) else None
            if not isinstance(eid, int) or eid <= since_event_id:
                continue
        if tree is not None and isinstance(evt, dict):
            path = evt.get("parent_branch_path")
            if isinstance(path, (list, tuple)) and path:
                try:
                    evt["parent_branch_path"] = tree.view(tree.intern(path))
                except ValueError:
                    pass
        yield evt
