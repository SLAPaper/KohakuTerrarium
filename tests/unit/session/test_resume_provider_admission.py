"""Cold history admission precedes publication of the effective provider and state."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.core import agent_model
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.compact import CompactManager
from kohakuterrarium.errors import HistoryCompatibilityError, SessionError
from kohakuterrarium.llm.generation import provider_source
from kohakuterrarium.session import resume_provider
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


class RoutedLLM(ScriptedLLM):
    provider_name = "openai"
    backend_type = "openai"

    def __init__(self, route):
        super().__init__(["unused"])
        self.backend_key = route
        self.model = "test-model"
        self._llm_identifier = route + "/test-model"
        self.closed = 0

    async def close(self):
        self.closed += 1


def _seed(store, source, mode):
    store.append_event(
        "saved",
        "user_message",
        {"content": "question"},
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    )
    origin = {"schema": 1, "generation_id": "g1", "source": source}
    store.append_event(
        "saved",
        "assistant_generation_start",
        deepcopy(origin),
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    )
    answer = {
        "role": "assistant",
        "content": "saved answer",
        "metadata": {"generation": origin},
    }
    if mode == "private":
        answer["reasoning_content"] = "private reasoning"
    elif mode in ("pending", "complete"):
        answer["tool_calls"] = [
            {
                "id": "pending",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ]
    tail = (
        [{"role": "tool", "tool_call_id": "pending", "name": "read", "content": "done"}]
        if mode == "complete"
        else []
    )
    store.save_conversation(
        "saved",
        [
            {"role": "system", "content": "saved system"},
            {
                "role": "user",
                "content": "question",
                "metadata": {"turn_index": 1, "branch_id": 1},
            },
            answer,
            *tail,
        ],
    )
    store.state["saved:snapshot_event_id"] = store.max_event_id("saved")


@pytest.mark.parametrize(
    "selection",
    [
        "legacy_missing",
        "committed_missing",
        "malformed",
        "legacy_resolve_error",
        "committed_resolve_error",
        "legacy_create_error",
        "committed_create_error",
    ],
)
async def test_saved_selector_failures_preserve_fresh_state(
    tmp_path, monkeypatch, selection
):
    original = RoutedLLM("original")
    agent = await Agent.build(AgentConfig(name="fresh"), llm=original, io="headless")
    store = SessionStore(str(tmp_path / "selection.kohakutr"))
    try:
        await store.run(_seed, store, provider_source(original, "bracket"), "portable")
        if selection.startswith("committed"):
            await store.run(
                store.append_event,
                "saved",
                "model_switch",
                {
                    "schema": 1,
                    "classification": "portable",
                    "from_source": {},
                    "to_source": {},
                    "to_identifier": "unavailable",
                },
            )
        else:
            await store.run(
                lambda: store.state.__setitem__(
                    "saved:model_selection",
                    {} if selection == "malformed" else "unavailable",
                )
            )
        monkeypatch.setattr(agent_model, "resolve_controller_llm", lambda *a, **k: None)
        if selection.endswith("error"):

            def unavailable(*args, **kwargs):
                raise RuntimeError("provider unavailable")

            if "create" in selection:
                monkeypatch.setattr(
                    agent_model,
                    "resolve_controller_llm",
                    lambda *a, **k: SimpleNamespace(),
                )
                monkeypatch.setattr(
                    agent_model, "profile_to_identifier", lambda profile: "unavailable"
                )
                monkeypatch.setattr(
                    agent_model, "create_llm_from_profile_name", unavailable
                )
            else:
                monkeypatch.setattr(agent_model, "resolve_controller_llm", unavailable)
        before = agent.controller.conversation
        if selection.startswith("legacy"):
            await inject_saved_state_async(agent, store, "saved")
            assert agent.llm is original
            assert (
                agent.controller.conversation.get_system_message().content
                == "saved system"
            )
        else:
            with pytest.raises(SessionError):
                await inject_saved_state_async(agent, store, "saved")
            assert agent.config.name == "fresh"
            assert agent.controller.conversation is before
            assert agent.llm is original
        assert original.call_count == original.closed == 0
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("failure", ["publish", "retire"])
async def test_publish_failure_rolls_back_bindings_before_candidate_cleanup(
    tmp_path, monkeypatch, failure
):
    original, candidate = RoutedLLM("original"), RoutedLLM("saved-route")
    agent = await Agent.build(
        AgentConfig(name="fresh", tool_format="native"), llm=original, io="headless"
    )
    store = SessionStore(str(tmp_path / "publish.kohakutr"))
    try:
        await store.run(_seed, store, provider_source(candidate, "native"), "portable")
        await store.run(
            lambda: store.state.__setitem__("saved:model_selection", "saved-route")
        )
        _factory(monkeypatch, candidate)
        before = (
            agent.controller.conversation,
            agent.registry._tools,
            agent.registry._tool_infos,
            agent.executor._tools,
            agent._llm_identifier,
        )
        publish = resume_provider._publish
        retire = resume_provider._remember_retired
        agent.compact_manager = CompactManager()
        agent.compact_manager._llm = original
        original_compact = agent.compact_manager._llm
        original_limit = agent.compact_manager.config.max_tokens
        candidate_compact = RoutedLLM("candidate-compact")
        candidate_compact._profile_max_context = original_limit + 100
        monkeypatch.setattr(
            resume_provider, "_candidate_compact", lambda *args: candidate_compact
        )
        plugin_context = agent.plugins._load_context
        original_plugin_model = (
            plugin_context.model if plugin_context is not None else None
        )
        before_retired = list(getattr(agent, "_retired_model_providers", []))

        def fail_publish(*args, **kwargs):
            publish(*args, **kwargs)
            raise RuntimeError("publish failure")

        def fail_retire(*args, **kwargs):
            retire(*args, **kwargs)
            raise RuntimeError("retire failure")

        monkeypatch.setattr(
            resume_provider,
            "_publish" if failure == "publish" else "_remember_retired",
            fail_publish if failure == "publish" else fail_retire,
        )
        with pytest.raises(RuntimeError, match="failure"):
            await inject_saved_state_async(agent, store, "saved")
        assert (
            agent.llm is original
            and agent.controller.llm is original
            and agent.subagent_manager.llm is original
        )
        assert agent.config.name == "fresh" and agent._turn_index == 0
        assert (
            agent.controller.conversation,
            agent.registry._tools,
            agent.registry._tool_infos,
            agent.executor._tools,
            agent._llm_identifier,
        ) == before
        assert agent.compact_manager._llm is original_compact
        assert agent.compact_manager.config.max_tokens == original_limit
        assert list(getattr(agent, "_retired_model_providers", [])) == before_retired
        if plugin_context is not None:
            assert plugin_context.model == original_plugin_model
        assert (
            candidate_compact.closed == candidate.closed == 1 and original.closed == 0
        )
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


def _factory(monkeypatch, provider):
    monkeypatch.setattr(
        agent_model, "resolve_controller_llm", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: provider
    )
    monkeypatch.setattr(
        agent_model, "profile_to_identifier", lambda profile: provider._llm_identifier
    )


@pytest.mark.parametrize("mode", ["private", "pending"])
@pytest.mark.parametrize("saved_selection", [False, True])
async def test_incompatible_provider_rejects_before_install(
    tmp_path, monkeypatch, mode, saved_selection
):
    original, candidate = RoutedLLM("original"), RoutedLLM("other")
    agent = await Agent.build(
        AgentConfig(name="fresh", tool_format="native"), llm=original, io="headless"
    )
    store = SessionStore(str(tmp_path / "admission.kohakutr"))
    try:
        source = provider_source(original if saved_selection else candidate, "native")
        await store.run(_seed, store, source, mode)
        if saved_selection:
            await store.run(
                lambda: store.state.__setitem__("saved:model_selection", "other")
            )
            _factory(monkeypatch, candidate)
        conversation = agent.controller.conversation
        snapshot = conversation.to_json()
        coordinates = (
            agent._turn_index,
            agent._branch_id,
            deepcopy(agent._parent_branch_path),
        )
        with pytest.raises(HistoryCompatibilityError):
            await inject_saved_state_async(agent, store, "saved")
        assert agent.config.name == "fresh"
        assert agent.controller.conversation is conversation
        assert conversation.to_json() == snapshot
        assert (
            agent._turn_index,
            agent._branch_id,
            agent._parent_branch_path,
        ) == coordinates
        assert agent.llm is original and agent.controller.llm is original
        assert original.call_count == candidate.call_count == 0
        assert original.closed == 0
        if saved_selection:
            assert candidate.closed == 1
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("mode", ["private", "pending", "portable", "complete"])
async def test_saved_provider_is_the_one_admitted_and_published(
    tmp_path, monkeypatch, mode
):
    original, candidate = RoutedLLM("original"), RoutedLLM("saved-route")
    agent = await Agent.build(
        AgentConfig(name="fresh", tool_format="native"), llm=original, io="headless"
    )
    store = SessionStore(str(tmp_path / "accepted.kohakutr"))
    try:
        await store.run(_seed, store, provider_source(candidate, "native"), mode)
        await store.run(
            lambda: store.state.__setitem__("saved:model_selection", "saved-route")
        )
        _factory(monkeypatch, candidate)
        await inject_saved_state_async(agent, store, "saved")
        assert agent.llm is candidate and agent.controller.llm is candidate
        assert agent.subagent_manager.llm is candidate
        assert (
            agent.controller.conversation.get_system_message().content == "saved system"
        )
        assert agent.config.name == "saved"
        assert candidate.closed == original.closed == 0
        assert candidate.call_count == 0
        if mode == "pending":
            assert any(
                m.tool_calls for m in agent.controller.conversation.get_messages()
            )
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)
