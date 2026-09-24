"""Durable controls outrank the legacy model-selection cache by exact namespace."""

import asyncio
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.agent_selection import (
    load_model_selection,
    restore_selections,
)
from kohakuterrarium import Agent
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.errors import SessionError
from kohakuterrarium.session.store import SessionStore


def _payload(identifier):
    return {
        "schema": 1,
        "classification": "portable",
        "from_source": {},
        "to_source": {},
        "to_identifier": identifier,
    }


@pytest.mark.parametrize("peer", ["alice2", "alice:child", "host:alice"])
async def test_durable_model_control_beats_stale_cache_without_namespace_leak(
    tmp_path, peer
):
    store = SessionStore(tmp_path / "selection.kohakutr")
    agent = SimpleNamespace(config=SimpleNamespace(name="alice"))
    try:

        def setup():
            store.state["alice:model_selection"] = "openai/old"
            store.append_control_event(
                "alice", "model_switch", _payload("openai/current")
            )
            store.append_control_event(peer, "model_switch", _payload("openai/peer"))

        await store.run(setup)
        assert await store.run(load_model_selection, agent, store) == "openai/current"
        attached = SimpleNamespace(
            config=SimpleNamespace(name="alice"),
            _session_output=SimpleNamespace(_event_key_prefix=peer),
        )
        assert await store.run(load_model_selection, attached, store) == "openai/peer"
    finally:
        await asyncio.to_thread(store.close)


async def test_resume_rejects_malformed_control_instead_of_using_default_provider(
    tmp_path,
):
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: alice\nsystem_prompt: Reply\ninput: {type: none}\noutput: {type: stdout}\n"
    )
    store = SessionStore(tmp_path / "bad-resume.kohakutr")
    agent = await Agent.build(config, llm=ScriptedLLM(["wrong default"]), io="headless")
    try:
        await store.run(
            store.append_control_event, "alice", "model_switch", {"schema": 999}
        )
        with pytest.raises(SessionError):
            await inject_saved_state_async(agent, store, "alice")
        assert agent.llm.call_count == 0
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


async def test_unavailable_committed_model_refuses_resume_instead_of_fallback(tmp_path):
    store = SessionStore(tmp_path / "unavailable.kohakutr")

    def unavailable(selector):
        raise ValueError("profile is unavailable")

    agent = SimpleNamespace(
        config=SimpleNamespace(name="alice"), switch_model=unavailable, plugins=None
    )
    try:
        await store.run(
            store.append_control_event,
            "alice",
            "model_switch",
            _payload("openai/removed"),
        )
        with pytest.raises(SessionError, match="committed"):
            await store.run(restore_selections, agent, store)
    finally:
        await asyncio.to_thread(store.close)


async def test_malformed_control_does_not_fall_back_to_cached_choice(tmp_path):
    store = SessionStore(tmp_path / "malformed.kohakutr")
    agent = SimpleNamespace(config=SimpleNamespace(name="alice"))
    try:

        def setup():
            store.state["alice:model_selection"] = "openai/old"
            store.append_control_event("alice", "model_switch", {"schema": 999})

        await store.run(setup)
        with pytest.raises(SessionError):
            await store.run(load_model_selection, agent, store)
    finally:
        await asyncio.to_thread(store.close)
