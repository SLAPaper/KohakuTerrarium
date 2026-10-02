"""Read-only session snapshots use the real SQLite and KohakuVault codecs."""

import sqlite3
import threading
from contextlib import closing

import pytest

from kohakuterrarium.session import readonly_view
from kohakuterrarium.session.errors import SessionNotReadyError
from kohakuterrarium.session.readonly_view import SessionReadView
from kohakuterrarium.session.store import SessionStore


def test_live_reads_execute_on_the_store_worker(tmp_path, monkeypatch):
    with closing(SessionStore(tmp_path / "affinity.kohakutr")) as store:
        _populate(store)
        worker = store.submit(threading.get_ident).result(timeout=3)
        table = store.meta
        observed = []

        class CheckedTable:
            def __getitem__(self, key):
                observed.append(threading.get_ident())
                assert observed[-1] == worker
                return table[key]

            def flush_cache(self):
                observed.append(threading.get_ident())
                assert observed[-1] == worker
                return table.flush_cache()

            def keys(self, **kwargs):
                return table.keys(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(store, "meta", CheckedTable())
            with SessionReadView(store.path) as reader:
                assert reader.get("meta", "nested")["items"][-1] == "汉字"
                assert dict(reader.items("meta"))["session_id"] == "sid"
        assert observed and set(observed) == {worker}


def test_cold_reader_reserves_path_before_native_writer_open(tmp_path, monkeypatch):
    path = tmp_path / "cold-lease.kohakutr"
    with closing(SessionStore(path)) as store:
        _populate(store)
    original = SessionStore._open_tables

    def forbidden_open(_store):
        raise AssertionError("native writer reached a file held by cold SQLite")

    with SessionReadView(path) as reader:
        with monkeypatch.context() as patch:
            patch.setattr(SessionStore, "_open_tables", forbidden_open)
            with pytest.raises(SessionNotReadyError):
                SessionStore(path, writer_lock=True)
        assert reader.get("meta", "session_id") == "sid"
    with closing(SessionStore(path, writer_lock=True)) as reopened:
        assert reopened._open_tables.__func__ is original
        assert reopened.meta["session_id"] == "sid"


def test_decoder_close_failure_releases_cold_reader_lease(tmp_path, monkeypatch):
    path = tmp_path / "decoder-close.kohakutr"
    with closing(SessionStore(path)) as store:
        _populate(store)
    reader = SessionReadView(path)
    decoder = reader._decoder
    original = decoder.close

    def failed_close():
        original()
        raise OSError("decoder close failed")

    monkeypatch.setattr(decoder, "close", failed_close)
    with pytest.raises(OSError, match="decoder close failed"):
        reader.close()
    with closing(SessionStore(path, writer_lock=True)) as reopened:
        assert reopened.meta["session_id"] == "sid"


def test_closed_live_view_has_a_defined_error(tmp_path):
    with closing(SessionStore(tmp_path / "closed-view.kohakutr")) as store:
        reader = SessionReadView(store.path)
        reader.close()
        with pytest.raises(SessionNotReadyError, match="closed"):
            reader.get("meta", "session_id")
        with pytest.raises(SessionNotReadyError, match="closed"):
            list(reader.items("events"))


def test_live_view_after_store_close_never_falls_back(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "closed-live.kohakutr")
    _populate(store)
    with SessionReadView(store.path) as reader:
        store.close()
        monkeypatch.setattr(readonly_view, "sqlite3", _ForbiddenSqlite())
        with pytest.raises(SessionNotReadyError):
            reader.get("meta", "session_id")
        with pytest.raises(SessionNotReadyError):
            list(reader.items("events"))


class _ForbiddenSqlite:
    """Stand-in for ``sqlite3`` that fails if the view opens its own connection."""

    def connect(self, *args, **kwargs):
        raise AssertionError(
            "a second SQLite library must not open a live session file"
        )


def _populate(store):
    store.init_meta("sid", "agent", "", "", ["alice"])
    store.meta["nested"] = {"items": [None, True, 4.5, "汉字"]}
    store.append_event("discovered", "user_input", {"content": "new agent"})
    store.append_event("alice:attached:helper:1", "text", {"content": "private"})
    store.append_event("alice", "user_input", {"content": "first"})
    store.append_event("alice", "user_input", {"content": "second"})
    store.save_conversation("alice", [{"role": "user", "content": "second"}])


def test_view_of_a_live_store_reads_through_it(tmp_path, monkeypatch):
    path = tmp_path / "snapshot %20 #.kohakutr"
    with closing(SessionStore(path)) as store:
        _populate(store)
        monkeypatch.setattr(readonly_view, "sqlite3", _ForbiddenSqlite())
        with SessionReadView(path.as_uri()) as reader:
            meta = reader.load_meta()
            assert meta["agents"] == ["alice", "discovered"]
            assert meta["nested"] == {"items": [None, True, 4.5, "汉字"]}
            assert reader.load_meta(discover_agents=False)["agents"] == ["alice"]
            store.meta["nested"] = {"items": ["new value"]}
            assert reader.get("meta", "nested") == {"items": ["new value"]}
            assert reader.get("state", "missing", "default") == "default"
            assert reader.get("conversation", "alice") == [
                {"role": "user", "content": "second"}
            ]
            assert reader.get("meta", "missing") is None
            assert (
                list(reader.items("events", prefix="discovered:e"))[0][1]["content"]
                == "new agent"
            )
            with pytest.raises(ValueError, match="unsupported session table"):
                reader.get("other", "key")
            with pytest.raises(ValueError, match="unsupported session table"):
                list(reader.items("other"))


def test_live_view_lists_only_the_requested_prefix_in_key_order(tmp_path, monkeypatch):
    path = tmp_path / "live.kohakutr"
    with closing(SessionStore(path)) as store:
        _populate(store)
        monkeypatch.setattr(readonly_view, "sqlite3", _ForbiddenSqlite())
        with SessionReadView(path) as reader:
            items = list(reader.items("events", prefix="alice:e"))
            assert [event["content"] for _, event in items] == ["first", "second"]
            assert [key for key, _ in items] == sorted(key for key, _ in items)
            assert all(key.startswith("alice:e") for key, _ in items)


def test_view_of_a_closed_store_reads_a_sqlite_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "closed.kohakutr"
    with closing(SessionStore(path)) as store:
        _populate(store)
    opened = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append(target)
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(readonly_view.sqlite3, "connect", spy)
    with SessionReadView(path) as reader:
        assert reader.get("meta", "nested") == {"items": [None, True, 4.5, "汉字"]}
        assert [
            event["content"] for _, event in reader.items("events", prefix="alice:e")
        ] == ["first", "second"]
        assert reader.get("conversation", "alice") == [
            {"role": "user", "content": "second"}
        ]
    assert len(opened) == 1
    assert opened[0].endswith("?mode=ro")


def test_view_switches_to_a_snapshot_after_the_live_store_closes(tmp_path):
    path = tmp_path / "later.kohakutr"
    store = SessionStore(path)
    _populate(store)
    with SessionReadView(path) as reader:
        assert reader.get("meta", "nested") == {"items": [None, True, 4.5, "汉字"]}
    store.close()
    with SessionReadView(path) as reader:
        assert reader.load_meta()["agents"] == ["alice", "discovered"]


def test_view_backs_off_while_a_store_is_still_opening_the_file(tmp_path, monkeypatch):
    path = tmp_path / "opening.kohakutr"
    SessionStore(path).close()
    outcome = {}
    real_open = SessionStore._open_tables

    def open_and_read(self):
        monkeypatch.setattr(readonly_view, "sqlite3", _ForbiddenSqlite())
        try:
            SessionReadView(path)
        except SessionNotReadyError as exc:
            outcome["error"] = exc
        real_open(self)

    monkeypatch.setattr(SessionStore, "_open_tables", open_and_read)
    SessionStore(path).close()
    assert isinstance(outcome["error"], SessionNotReadyError)


def test_missing_view_does_not_create_database(tmp_path):
    path = tmp_path / "missing.kohakutr"
    with pytest.raises(FileNotFoundError):
        SessionReadView(path)
    assert not path.exists()
