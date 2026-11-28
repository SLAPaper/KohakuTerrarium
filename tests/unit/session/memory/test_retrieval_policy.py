"""Ranked retrieval excludes search echoes and reports incomplete candidate windows."""

from types import SimpleNamespace

import numpy as np

import kohakuterrarium.builtins.tools.search_memory as memory_tool
import kohakuterrarium.session.memory.index as index_mod
from kohakuterrarium.core.session import Session
from kohakuterrarium.modules.tool.base import ToolContext
from kohakuterrarium.session.memory.embedding import BaseEmbedder
from kohakuterrarium.session.memory.index import SessionMemory
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.studio.sessions.memory.search import search_session_memory

MARKER = "31.933066093773366"


class EchoPreferredEmbedder(BaseEmbedder):
    """Put echoes first in vector ranking, independently of production ranking."""

    dimensions = 4

    def encode(self, texts):
        return np.array(
            [
                (
                    np.ones(4, dtype=np.float32)
                    if "search_memory" in text
                    else -np.ones(4, dtype=np.float32)
                )
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_one(self, query):
        return np.ones(4, dtype=np.float32)


def seed_legacy_echoes(memory, count, *, query=MARKER, vectors=False):
    # Explicitly write the historical index shape. Do not derive the legacy
    # fixture from the current extractor (which now excludes these events).
    for i in range(count):
        text = f"[{'tool' if i % 2 else 'result'}:search_memory] {query}"
        meta = {
            "agent": "alice",
            "type": "tool",
            "tool_name": "search_memory",
            "block_id": f"legacy-echo-{i}",
        }
        memory._fts.insert(text, meta)
        if vectors:
            memory._vec.insert(np.ones(4, dtype=np.float32), {**meta, "content": text})


def test_new_and_legacy_echoes_do_not_displace_real_results(tmp_path):
    memory = SessionMemory(str(tmp_path / "echo.kohakutr"), EchoPreferredEmbedder())
    try:
        original = f"[result:bash] Actual benchmark speed {MARKER} tokens per second"
        rows = [
            {
                "type": "tool_result",
                "name": "bash",
                "output": original.removeprefix("[result:bash] "),
            },
            {
                "type": "tool_call",
                "name": "search_memory",
                "args": {"query": "NEVER_RECORDED"},
            },
            {
                "type": "tool_result",
                "name": "search_memory",
                "output": "NEVER_RECORDED",
            },
        ]
        memory.index_events("alice", rows)
        stats = memory.get_stats()
        assert stats["fts_blocks"] == stats["vec_blocks"] == 1
        assert not memory.search("NEVER_RECORDED", mode="fts")
        for mode in ("fts", "semantic", "hybrid", "auto"):
            hits = memory.search(MARKER, mode=mode, k=5)
            assert [hit.content for hit in hits] == [original]
            assert hits.warning is None
        seed_legacy_echoes(memory, 80, vectors=True)
        # The evidence really does rank behind echoes in the native FTS/vec0
        # corpus; testing exclusion after top-k alone would miss the defect.
        assert all(
            meta["tool_name"] == "search_memory"
            for _, _, meta in memory._fts.search(MARKER, k=5)
        )
        for mode in ("fts", "semantic", "hybrid", "auto"):
            hits = memory.search(MARKER, mode=mode, k=1)
            assert [hit.content for hit in hits] == [original]
            assert hits.warning is None
            hits = memory.search(MARKER, mode=mode, k=5)
            assert [hit.content for hit in hits] == [original]
            assert hits.warning is None
    finally:
        memory.close()


def test_native_vector_limit_returns_uncertainty_not_backend_error(tmp_path):
    memory = SessionMemory(str(tmp_path / "vec0.kohakutr"), EchoPreferredEmbedder())
    try:
        # 4096 is the native sqlite-vec limit, not a mocked algorithm budget.
        for i in range(4096):
            memory._vec.insert(
                np.ones(4, dtype=np.float32),
                {
                    "content": f"echo {i}",
                    "agent": "alice",
                    "tool_name": "search_memory",
                    "type": "tool",
                    "block_id": f"echo:{i}",
                },
            )
        memory._vec.insert(
            -np.ones(4, dtype=np.float32),
            {
                "content": "historical evidence",
                "agent": "alice",
                "tool_name": "bash",
                "type": "tool",
                "block_id": "actual",
            },
        )
        hits = memory.search("evidence", mode="semantic", k=5)
        assert hits == [] and hits.warning
        assert "may be incomplete" in hits.warning
    finally:
        memory.close()


async def test_incomplete_fts_is_reported_by_tool_and_studio(tmp_path, monkeypatch):
    path = tmp_path / "warning.kohakutr"
    store = SessionStore(path)
    store.init_meta("s", "agent", "", str(tmp_path), ["alice"])
    try:
        store.append_event("alice", "user_input", {"content": "No genuine match"})
        memory = SessionMemory(store.path)
        try:
            memory.index_events("alice", store.get_events("alice"))
            seed_legacy_echoes(memory, 5)
        finally:
            memory.close()
        ctx = ToolContext(
            agent_name="alice",
            session=Session(key="s"),
            agent=SimpleNamespace(session_store=store, config=None),
            working_dir=None,
        )
        monkeypatch.setattr(index_mod, "_MAX_RANKED_CANDIDATES", 3)
        result = await memory_tool.SearchMemoryTool().execute(
            {"query": MARKER, "mode": "fts"}, context=ctx
        )
        assert result.error is None
        assert "may be incomplete" in result.output
        assert (
            "No eligible hits" in result.output
            and "No results found." not in result.output
        )
    finally:
        store.close(update_status=False)
    response = await search_session_memory(path, q=MARKER, mode="fts", k=5)
    assert response["count"] == 0 and "may be incomplete" in response["warning"]
    monkeypatch.setattr(index_mod, "_MAX_RANKED_CANDIDATES", 20)
    complete = await search_session_memory(path, q=MARKER, mode="fts", k=5)
    assert complete["count"] == 0 and not complete.get("warning")


async def test_hybrid_keeps_keyword_evidence_and_reports_semantic_exhaustion(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "hybrid.kohakutr")
    store.init_meta("s", "agent", "", str(tmp_path), ["alice"])
    try:
        store.append_event(
            "alice", "user_input", {"content": f"ACTUAL_EVIDENCE_{MARKER}"}
        )
        memory = SessionMemory(store.path, EchoPreferredEmbedder())
        try:
            memory.index_events("alice", store.get_events("alice"))
            seed_legacy_echoes(memory, 6, vectors=True)
            monkeypatch.setattr(index_mod, "_MAX_VECTOR_CANDIDATES", 3)
            hits = memory.search(MARKER, mode="hybrid", k=5)
            assert len(hits) == 1 and "ACTUAL_EVIDENCE" in hits[0].content
            assert "Semantic candidate limit" in hits.warning
        finally:
            memory.close()
        ctx = ToolContext(
            agent_name="alice",
            session=Session(key="s"),
            agent=SimpleNamespace(session_store=store, config=None),
            working_dir=None,
        )
        monkeypatch.setattr(
            memory_tool, "create_embedder", lambda _: EchoPreferredEmbedder()
        )
        result = await memory_tool.SearchMemoryTool().execute(
            {"query": MARKER, "mode": "hybrid", "k": 5}, context=ctx
        )
        assert result.error is None and "ACTUAL_EVIDENCE" in result.output
        assert "Semantic candidate limit" in result.output
        assert "Embedding unavailable" not in result.output
        # FTS may return a partial subset as well, rather than failing the query.
        monkeypatch.setattr(index_mod, "_MAX_RANKED_CANDIDATES", 3)
        result = await memory_tool.SearchMemoryTool().execute(
            {"query": MARKER, "mode": "fts", "k": 5}, context=ctx
        )
        assert result.error is None and "may be incomplete" in result.output
        assert "tool:search_memory" not in result.output
    finally:
        store.close(update_status=False)
