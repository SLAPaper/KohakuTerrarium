"""Read shared ancestry internally and expand paths at public boundaries."""

from kohakuterrarium.session.history.ancestry import SharedPath
from kohakuterrarium.session.history.payloads import read_descriptors


def read_restore_events(
    store,
    namespace: str,
    *,
    before_sequence: int | None = None,
    legacy_keys: frozenset[str] | None = None,
) -> list[dict]:
    """Build temporary lightweight descriptors over bounded payload pages."""
    return read_descriptors(store, namespace, before_sequence, legacy_keys)


def materialize_event_paths(events: list[dict]) -> list[dict]:
    """Expose ordinary mutable path lists to external history consumers."""
    return [
        (
            {**event, "parent_branch_path": path.tree.path(path.leaf)}
            if isinstance(path := event.get("parent_branch_path"), SharedPath)
            else dict(event)
        )
        for event in events
    ]
