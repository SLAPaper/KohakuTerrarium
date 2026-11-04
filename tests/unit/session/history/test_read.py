"""Shared paths belong to an explicit read and expand at public boundaries."""

from contextlib import closing

from kohakuterrarium.session.history.ancestry import SharedPath
from kohakuterrarium.session.history.read import (
    read_restore_events,
    materialize_event_paths,
)
from kohakuterrarium.session.store import SessionStore


def test_shared_restore_read_does_not_change_public_event_reads(tmp_path):
    with closing(SessionStore(tmp_path / "paths.kohakutr")) as store:
        for i in range(3):
            store.append_event(
                "a",
                "text_chunk",
                {"content": str(i)},
                turn_index=2,
                branch_id=1,
                parent_branch_path=[(1, 1)],
            )
        internal = read_restore_events(store, "a")
        paths = [e["parent_branch_path"] for e in internal]
        assert all(isinstance(path, SharedPath) for path in paths)
        assert len({id(path.tree) for path in paths}) == 1
        external = materialize_event_paths(internal)
        assert external == store.get_events("a")
        assert all(type(e["parent_branch_path"]) is list for e in external)
        external[0]["parent_branch_path"].append([9, 9])
        assert list(paths[0]) == [(1, 1)]
