"""Offline rebuild: preservation, explicit vector policy, admission and recovery."""

import hashlib
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pytest

import kohakuterrarium.session.memory.rebuild as maintenance
import kohakuterrarium.session.store.events.values as event_values
import kohakuterrarium.studio.sessions.memory.build as builder
from kohakuterrarium.cli._main import COMMANDS, _build_parser
from kohakuterrarium.cli.memory import rebuild_memory_cli
from kohakuterrarium.errors import SessionLockedError
from kohakuterrarium.session.memory.embedding import BaseEmbedder
from kohakuterrarium.session.memory.index import SessionMemory
from kohakuterrarium.session.memory.rebuild import (
    RebuildRefused,
    inspect_index,
    rebuild_session_memory,
)
from kohakuterrarium.session.memory.source_events import sourced_events
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.store.lifecycle.lock import (
    acquire_writer_lock,
    release_writer_lock,
)
from kohakuterrarium.session.store.lifecycle.version import SessionFormatError


class Embedder(BaseEmbedder):
    dimensions = 4

    def encode(self, texts):
        return np.ones((len(texts), self.dimensions), dtype=np.float32)


@contextmanager
def opened_store(path, *, readonly=False):
    store = SessionStore.open_readonly(path) if readonly else SessionStore(path)
    try:
        yield store
    finally:
        store.close(update_status=False)


def seed(path, *, vectors=False):
    with opened_store(path) as store:
        store.init_meta("s", "agent", "", str(path.parent), ["alice", "bob", "empty"])
        with closing(
            SessionMemory(store.path, Embedder() if vectors else None)
        ) as index:
            for agent in ("alice", "bob"):
                store.append_event(
                    agent,
                    "user_input",
                    {
                        "content": f"SOURCE_{agent.upper()}_MEMORY",
                        "event_id": 50 if agent == "alice" else 51,
                    },
                )
                # Old indexes have no physical source metadata.
                index.index_events(agent, store.get_events(agent))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(path):
    with opened_store(path, readonly=True) as store:
        return store.get_all_events()


def backups(path):
    return list(path.parent.glob(f"{path.name}.before-memory-rebuild-*.bak"))


def test_cli_rebuilds_all_agents_preserves_history_and_preferences(tmp_path, capsys):
    path = tmp_path / "legacy.kohakutr"
    seed(path)
    preferences = {"provider": "model2vec", "model": "@base"}
    with opened_store(path) as store:
        store.state["embedding_config"] = preferences
        store.append_event(
            "alice", "text", {"content": "MIRRORED_REPLY", "event_id": 10}
        )
    original = events(path)
    args = _build_parser().parse_args(["memory", "rebuild", str(path)])
    assert COMMANDS[args.command](args) == 0
    assert "across 3 agents" in capsys.readouterr().out
    (backup,) = backups(path)
    assert events(path) == events(backup) == original
    assert inspect_index(backup).fts_rows == 2
    with (
        opened_store(path, readonly=True) as store,
        closing(SessionMemory(str(path))) as index,
    ):
        assert store.state["embedding_config"] == preferences
        assert not index.has_vectors
        for marker in ("SOURCE_ALICE_MEMORY", "SOURCE_BOB_MEMORY", "MIRRORED_REPLY"):
            (hit,) = index.search(marker, mode="fts")
            assert store.events[hit.source_ref]["content"] == marker
            assert hit.source_fingerprint
        assert not index.search("SOURCE_ALICE_MEMORY", mode="fts", agent="empty")


def test_vector_choice_dry_run_and_drop_all_dimensions(tmp_path, capsys, monkeypatch):
    path = tmp_path / "old-index.kohakutr"
    seed(path, vectors=True)
    other = Embedder()
    other.dimensions = 3
    with (
        opened_store(path) as store,
        closing(SessionMemory(store.path, other)) as index,
    ):
        for name in ("alice", "bob"):
            index.index_events(name, store.get_events(name))
        # A stale derived row not present in the current event projection.
        index._fts.insert(
            "OLD_UNSELECTED_TOOL_RESULT",
            {"agent": "alice", "type": "tool", "block_id": "old"},
        )
    original = events(path)
    before = digest(path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert rebuild_memory_cli(str(path)) == 1
    assert "--vectors rebuild" in capsys.readouterr().out
    assert rebuild_memory_cli(str(path), dry_run=True) == 0
    assert digest(path) == before and not backups(path)
    with pytest.raises(RebuildRefused, match="Existing vectors"):
        rebuild_session_memory(path)
    result = rebuild_session_memory(path, vectors="drop")
    assert events(path) == events(result.backup) == original
    assert inspect_index(result.backup).vector_rows == ((3, 2), (4, 2))
    assert not inspect_index(path).vectors_present
    with (
        opened_store(path, readonly=True) as store,
        closing(SessionMemory(str(path))) as index,
    ):
        assert store.state["embedding_config"] == {"provider": "none"}
        assert not index.has_vectors
        assert not index.search("OLD_UNSELECTED_TOOL_RESULT", mode="fts")
        assert index.search("SOURCE_BOB_MEMORY", mode="fts")[0].source_ref
    with closing(SessionMemory(str(result.backup))) as index:
        assert index.search("OLD_UNSELECTED_TOOL_RESULT", mode="fts")


def test_interrupted_keyword_rebuild_keeps_authority_and_recovers(
    tmp_path, monkeypatch
):
    path = tmp_path / "interrupted.kohakutr"
    seed(path)
    original = events(path)
    finish = SessionMemory._finish_batch

    def interrupted(self, agent, kind, batch):
        # Fail at publication, after writes, so partial rows really exist.
        if agent == "bob" and kind == "keywords":
            raise RuntimeError("injected index update failure")
        return finish(self, agent, kind, batch)

    with monkeypatch.context() as patch:
        patch.setattr(SessionMemory, "_finish_batch", interrupted)
        with pytest.raises(RebuildRefused, match="injected index update failure"):
            rebuild_session_memory(path)
    (backup,) = backups(path)
    assert events(path) == events(backup) == original
    assert inspect_index(backup).fts_rows == 2
    with closing(SessionMemory(str(path))) as index:
        assert index.search("SOURCE_ALICE_MEMORY", mode="fts")
        assert not index.search("SOURCE_BOB_MEMORY", mode="fts")
    rebuild_session_memory(path)
    assert events(path) == original
    with closing(SessionMemory(str(path))) as index:
        assert index.search("SOURCE_BOB_MEMORY", mode="fts")


def test_model_load_and_encode_failures_are_retryable(tmp_path, monkeypatch, capsys):
    path = tmp_path / "vector-retry.kohakutr"
    seed(path, vectors=True)
    with opened_store(path) as store:
        store.state["embedding_config"] = {"provider": "model2vec", "model": None}
    original = events(path)
    before = digest(path)
    attempts = []

    class FailingEmbedder(Embedder):
        def encode(self, texts):
            raise RuntimeError("encoder unavailable")

    class LockedEmbedder(Embedder):
        def encode(self, texts):
            # The external model call occurs inside offline writer ownership.
            with pytest.raises(SessionLockedError):
                acquire_writer_lock(str(path))
            return super().encode(texts)

    def provision(config):
        attempts.append(config)
        assert config["provider"] == "model2vec" and config.get("model") is None
        if len(attempts) == 1:
            raise ImportError("model unavailable")
        return FailingEmbedder() if len(attempts) == 2 else LockedEmbedder()

    monkeypatch.setattr(maintenance, "create_embedder", provision)
    assert rebuild_memory_cli(str(path), vectors="rebuild") == 1
    assert "model unavailable" in capsys.readouterr().out
    assert digest(path) == before
    with pytest.raises(RebuildRefused, match="encoder unavailable"):
        rebuild_session_memory(path, vectors="rebuild")
    assert events(path) == original
    assert inspect_index(path).vector_rows == ((4, 2),)
    with closing(SessionMemory(str(path), Embedder())) as index:
        assert index.search("SOURCE_ALICE_MEMORY", mode="fts")
        assert not index.search("SOURCE_ALICE_MEMORY", mode="semantic")
    result = rebuild_session_memory(path, vectors="rebuild")
    assert result.fts_rows == result.vector_rows == 2
    assert events(result.backup) == events(path) == original
    with closing(SessionMemory(str(path), Embedder())) as index:
        (hit,) = index.search("SOURCE_BOB_MEMORY", mode="semantic", agent="bob")
        assert hit.content == "SOURCE_BOB_MEMORY" and hit.source_ref


def test_active_native_handles_refuse_maintenance_then_allow_retry(tmp_path):
    path = tmp_path / "active.kohakutr"
    seed(path)
    original = events(path)
    before = digest(path)
    for open_handle in (
        lambda: opened_store(path),
        lambda: opened_store(path, readonly=True),
        lambda: closing(SessionMemory(str(path))),
    ):
        with open_handle() as active:
            if isinstance(active, SessionStore):
                active.state.get("embedding_config")
            with pytest.raises(RebuildRefused):
                inspect_index(path)
            with pytest.raises(RebuildRefused):
                rebuild_session_memory(path)
            assert not backups(path)
    # The read-only handles may have left SQLite sidecars. Retry without
    # requiring any particular sidecar cleanup implementation.
    assert digest(path) == before
    assert inspect_index(path).fts_rows == 2
    result = rebuild_session_memory(path)
    assert result.fts_rows == 2 and events(path) == original


def test_failed_backup_prevents_index_mutation(tmp_path, monkeypatch):
    path = tmp_path / "backup-failure.kohakutr"
    seed(path)
    before = digest(path)

    def fail(*args):
        raise RebuildRefused("Backup verification failed")

    monkeypatch.setattr(maintenance, "_backup", fail)
    with pytest.raises(RebuildRefused, match="Backup verification failed"):
        rebuild_session_memory(path)
    assert digest(path) == before
    assert inspect_index(path).fts_rows == 2


@pytest.mark.parametrize("failure", ["malformed", "incomplete_read"])
def test_bad_source_cannot_publish_empty_replacement(tmp_path, monkeypatch, failure):
    path = tmp_path / "bad-source.kohakutr"
    seed(path)
    if failure == "malformed":
        with opened_store(path) as store:
            store.events["alice:e000001"] = "not an event"
    else:
        real_read = event_values.read_values

        def incomplete(*args, **kwargs):
            if kwargs.get("strict"):
                return iter(())
            return real_read(*args, **kwargs)

        monkeypatch.setattr(event_values, "read_values", incomplete)
        with opened_store(path) as store:
            assert sourced_events(store, "alice") == store.get_events("alice")
    before = digest(path)
    with pytest.raises(RebuildRefused, match="Cannot read or project"):
        rebuild_session_memory(path)
    assert digest(path) == before
    assert inspect_index(path).fts_rows == 2
    assert len(backups(path)) == 1


@pytest.mark.parametrize("vector_only", [False, True])
def test_missing_indexed_namespace_is_not_erased(tmp_path, vector_only):
    path = tmp_path / "missing-namespace.kohakutr"
    seed(path, vectors=vector_only)
    with opened_store(path) as store, closing(SessionMemory(str(path))) as index:
        if vector_only:
            for row, _, _ in index._fts.search("SOURCE_BOB_MEMORY", k=5):
                index._fts.delete(row)
        del store.events["bob:e000000"]
    before = digest(path)
    with pytest.raises(RebuildRefused, match="Indexed Agent .*bob.*no physical source"):
        rebuild_session_memory(path, vectors="drop" if vector_only else "auto")
    assert digest(path) == before
    assert len(backups(path)) == 1


def test_original_archive_format_gate_allows_retry(tmp_path):
    path = tmp_path / "session.kohakutr"
    seed(path)
    future = Path(f"{path}.v3")
    future.write_bytes(b"future sentinel")
    before = digest(path)
    with pytest.raises(SessionFormatError, match="format v3"):
        rebuild_session_memory(path)
    assert digest(path) == before and future.read_bytes() == b"future sentinel"
    assert not backups(path)
    future.unlink()
    assert rebuild_session_memory(path).fts_rows == 2


def test_symlink_archive_families_and_writer_locks(tmp_path):
    path = tmp_path / "actual.kohakutr"
    seed(path)
    alias = tmp_path / "alias.kohakutr"
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("file symlinks unavailable")
    original = events(path)
    for target in (alias, path):
        future = Path(f"{target}.v3")
        future.write_bytes(b"future sentinel")
        with pytest.raises(SessionFormatError, match="format v3"):
            rebuild_session_memory(alias)
        future.unlink()
        held = acquire_writer_lock(str(target))
        try:
            with pytest.raises(SessionLockedError):
                rebuild_session_memory(alias)
        finally:
            release_writer_lock(held)
    assert events(path) == original and alias.samefile(path)
    assert rebuild_session_memory(alias).fts_rows == 2
    assert alias.is_symlink() and alias.samefile(path)
    assert events(path) == original


def test_committed_wal_is_preserved_in_original_and_backup(tmp_path):
    path = tmp_path / "committed-wal.kohakutr"
    seed(path)
    code = (
        "import os, sqlite3, sys; "
        "db=sqlite3.connect(sys.argv[1]); "
        "db.execute('PRAGMA journal_mode=WAL'); "
        "db.execute('PRAGMA wal_autocheckpoint=0'); "
        "db.execute('CREATE TABLE preserved_probe (value TEXT)'); "
        "db.execute(\"INSERT INTO preserved_probe VALUES ('committed')\"); "
        "db.commit(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    assert Path(f"{path}-wal").stat().st_size > 0
    result = rebuild_session_memory(path)
    for file in (path, result.backup):
        with closing(sqlite3.connect(file)) as db:
            assert db.execute("SELECT value FROM preserved_probe").fetchone() == (
                "committed",
            )


def test_offline_embedding_builder_produces_source_bound_hits(tmp_path, monkeypatch):
    path = tmp_path / "builder.kohakutr"
    with opened_store(path) as store:
        store.init_meta("s", "agent", "", str(tmp_path), ["alice"])
        store.append_event("alice", "user_input", {"content": "SOURCE_ALICE_MEMORY"})
    monkeypatch.setattr(builder, "create_embedder", lambda *_args: Embedder())
    builder.build_index(path, provider="model2vec")
    with (
        opened_store(path, readonly=True) as store,
        closing(SessionMemory(str(path), Embedder())) as index,
    ):
        for mode in ("fts", "semantic"):
            (hit,) = index.search("SOURCE_ALICE_MEMORY", mode=mode)
            assert store.events[hit.source_ref]["content"] == hit.content
            assert hit.source_fingerprint
