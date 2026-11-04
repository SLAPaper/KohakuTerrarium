"""Retire native cache guards without interpreting opaque session State."""

import sqlite3
from contextlib import closing

from kohakuvault import KVault

from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.store.lifecycle.obsolete import (
    remove_obsolete_resume_guards,
)


def names(path):
    with closing(sqlite3.connect(path)) as connection:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}


def test_readonly_preserves_guards_and_writable_open_preserves_all_state(tmp_path):
    path = tmp_path / "obsolete.kohakutr"
    state = {
        "a:recovery_context": [{"role": "user", "content": "legacy value"}],
        "a:recovery_receipt": {"version": 1},
        "a:history_projection:stdout-v3": {"text": "preview"},
        "plugin:example:recovery_context": {"plugin_data": "keep"},
        "plugin:example:recovery_receipt": {"plugin_data": "keep"},
        "run:recovery_context": {"state": "running", "turn_id": "t"},
        "jobs:reaped:recovery_context": ["job-id"],
        "attach_seq:recovery_context": 7,
        "a:history_projection:turn_count": 7,
        "a:branch_snapshot_messages:"
        + "a" * 32: [{"role": "user", "content": "snapshot"}],
    }
    with closing(SessionStore(path)) as store:
        store.meta["agents"] = ["a", "run", "recovery_context"]
        store.append_event("a", "user_message", {"content": "archived"})
        for key, value in state.items():
            store.state[key] = value
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE kt_event_revisions (prefix BLOB)")
        connection.execute("CREATE TABLE kt_value_revisions (key BLOB)")
        connection.execute("CREATE TABLE kt_event_sequences (prefix BLOB)")
        connection.execute(
            "CREATE TRIGGER kt_event_insert AFTER INSERT ON events BEGIN INSERT INTO kt_event_revisions VALUES(NEW.key); END"
        )
        connection.execute("CREATE TABLE user_audit (key BLOB)")
        connection.execute(
            "CREATE TRIGGER user_event_insert AFTER INSERT ON events BEGIN INSERT INTO user_audit VALUES(NEW.key); END"
        )
    before = names(path)
    with closing(SessionStore(path, read_only=True)) as store:
        remove_obsolete_resume_guards(store)
        assert {key: store.state[key] for key in state} == state
        assert store.get_events("a")[0]["content"] == "archived"
    assert names(path) == before
    for _ in range(2):
        with closing(SessionStore(path)) as store:
            assert {key: store.state[key] for key in state} == state
            assert store.get_events("a")[0]["content"] == "archived"
            store.append_event("a", "activity", {"detail": "continued"})
    assert not {
        "kt_event_insert",
        "kt_event_revisions",
        "kt_value_revisions",
        "kt_event_sequences",
    } & names(path)
    assert {"user_audit", "user_event_insert"} <= names(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT count(*) FROM user_audit").fetchone()[0] == 2


def test_guard_cleanup_performs_no_kvault_record_reads_or_writes(tmp_path, monkeypatch):
    with closing(SessionStore(tmp_path / "no-record-io.kohakutr")) as store:
        store.state["a:recovery_context"] = "opaque value"
        calls = []

        def observe(method):
            def recorded(*args, **kwargs):
                calls.append(method.__name__)
                return method(*args, **kwargs)

            return recorded

        with monkeypatch.context() as observation:
            for method in (
                "keys",
                "get",
                "get_many",
                "__getitem__",
                "__setitem__",
                "__delitem__",
                "put_many",
            ):
                observation.setattr(KVault, method, observe(getattr(KVault, method)))
            remove_obsolete_resume_guards(store)
            assert calls == []
        assert store.state["a:recovery_context"] == "opaque value"
