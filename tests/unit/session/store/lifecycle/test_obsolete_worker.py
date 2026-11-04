"""Offline DDL never creates a missing archive or changes unrelated objects."""

import sqlite3
from contextlib import closing

import pytest

from kohakuterrarium.session.store.lifecycle.obsolete_worker import retire_guards


def test_offline_cleanup_is_exact_idempotent_and_does_not_create_files(tmp_path):
    path = tmp_path / "offline.db"
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        retire_guards(path)
    assert not path.exists()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE kt_event_revisions (prefix BLOB)")
        db.execute("CREATE TABLE user_audit (key BLOB)")
    retire_guards(path)
    retire_guards(path)
    with closing(sqlite3.connect(path)) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    assert "user_audit" in names
    assert "kt_event_revisions" not in names
