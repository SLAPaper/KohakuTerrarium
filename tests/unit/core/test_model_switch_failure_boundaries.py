"""Model-switch failure boundaries preserve usable state or fail closed."""

import asyncio

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.core import agent_model
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError, SessionError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.fixture
async def model_agent(monkeypatch):
    old = ScriptedLLM(["old answer"])
    new = ScriptedLLM(["new answer"])
    monkeypatch.setattr(agent_model, "create_llm_from_profile_name", lambda name: new)
    agent = await Agent.build(
        AgentConfig(name="alice", system_prompt="Reply"), llm=old, io="headless"
    )
    try:
        yield agent, old, new
    finally:
        await agent.stop()


async def test_no_store_live_switch_keeps_in_memory_contract(model_agent):
    agent, old, new = model_agent
    await agent.start()
    assert agent.session_store is None
    assert (
        await agent.switch_model_async("openai/gpt-5.4-nano") == "openai/gpt-5.4-nano"
    )
    assert (await agent.run("continue", timeout=5)).text == "new answer"
    assert old.call_count == 0


async def test_post_commit_publication_fault_fences_live_turn(
    model_agent, tmp_path, monkeypatch
):
    agent, old, new = model_agent
    store = SessionStore(tmp_path / "publish.kohakutr")
    agent.attach_session_store(store)
    await agent.start()

    def fail_after_commit(*args):
        raise RuntimeError("publication fault")

    monkeypatch.setattr(
        "kohakuterrarium.core.model_switch_runtime._publish", fail_after_commit
    )
    try:
        with pytest.raises(RuntimeError, match="publication fault"):
            await agent.switch_model_async("openai/gpt-5.4-nano")
        assert agent._model_switch_uncertain
        controls = await store.run(store.get_events, "alice")
        assert sum(event["type"] == "model_switch" for event in controls) == 1
        result = await agent.run("blocked", timeout=5, raise_on_error=False)
        assert not result.ok
        assert old.call_count == new.call_count == 0
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


async def test_cancel_before_internal_task_start_releases_reservation(model_agent):
    agent, old, new = model_agent
    await agent.start()
    request = asyncio.create_task(agent.switch_model_async("openai/gpt-5.4-nano"))
    await asyncio.sleep(0)
    internal = agent._model_switch_task
    assert internal.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not agent._processing_lock.locked()
    assert not agent._model_switch_pending
    assert (await agent.run("continue old", timeout=5)).text == "old answer"


async def test_bootstrap_preparation_failure_keeps_old_binding_and_closes_candidate(
    model_agent, monkeypatch
):
    agent, old, new = model_agent
    closed = []

    async def close():
        closed.append(new)

    def fail_callback(*args):
        raise RuntimeError("callback preparation failed")

    monkeypatch.setattr(new, "close", close)
    monkeypatch.setattr(new, "on_emergency_drop", fail_callback, raising=False)
    with pytest.raises(RuntimeError, match="callback preparation failed"):
        agent.switch_model("openai/gpt-5.4-nano")
    assert agent.llm is old
    assert agent.controller.llm is old
    assert agent.subagent_manager.llm is old
    await agent.stop()
    assert closed == [new]


async def test_same_tool_set_switch_refreshes_model_template_and_plugin_prose(
    model_agent,
):
    agent, old, new = model_agent
    old.model, new.model = "old-model", "new-model"
    agent.config.system_prompt = "Model {{ model }}"

    class NewModelProse(BasePlugin):
        name = "new_model_prose"
        applies_to = {"model_patterns": ["new-model"]}

        def get_prompt_content(self, context):
            return "NEW-MODEL-ONLY"

    await agent.add_plugin(NewModelProse())
    await agent.start()
    agent.refresh_system_prompt()
    assert "NEW-MODEL-ONLY" not in agent.get_system_prompt()
    assert "Model old-model" in agent.get_system_prompt()
    await agent.switch_model_async("openai/gpt-5.4-nano")
    prompt = agent.get_system_prompt()
    assert "Model new-model" in prompt
    assert "NEW-MODEL-ONLY" in prompt


@pytest.mark.parametrize("phase", ["options", "control"])
async def test_definite_precommit_session_error_does_not_fence(
    model_agent, tmp_path, monkeypatch, phase
):
    agent, old, new = model_agent
    store = SessionStore(tmp_path / "rejected.kohakutr")
    agent.attach_session_store(store)
    await agent.start()

    def reject(*args, **kwargs):
        raise SessionError("definite precommit rejection")

    target = "_read_options" if phase == "options" else "append_control_event"
    try:
        with monkeypatch.context() as patch:
            patch.setattr(f"kohakuterrarium.core.model_switch_runtime.{target}", reject)
            with pytest.raises(SessionError, match="definite precommit rejection"):
                await agent.switch_model_async("openai/gpt-5.4-nano")
        assert not getattr(agent, "_model_switch_uncertain", False)
        assert agent.llm is old
        assert (await agent.run("still usable", timeout=5)).text == "old answer"
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


async def test_bootstrap_switch_cannot_bypass_uncertainty(model_agent):
    agent, old, new = model_agent
    agent._model_switch_uncertain = True
    with pytest.raises(ConflictError, match="uncertain"):
        agent.switch_model("openai/gpt-5.4-nano")
    assert agent.llm is old
