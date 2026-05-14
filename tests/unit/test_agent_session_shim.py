import asyncio
from types import SimpleNamespace

import pytest

import kohakuterrarium.serving.agent_session as agent_session_module
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.serving.agent_session import AgentSession


class _FakeAgent:
    def __init__(self, config=None, **kwargs) -> None:
        self.config = config or SimpleNamespace(name="fake")
        self.init_kwargs = kwargs
        self.is_running = False
        self.llm = SimpleNamespace(model="test/model")
        self.executor = SimpleNamespace(_working_dir="C:/work")
        self.tools = ["read"]
        self.subagents = ["worker"]
        self.output_router = SimpleNamespace(default_output=object())
        self.output_handler = None
        self.replace_default = None
        self.injected: list[tuple[object, str]] = []
        self.start_calls = 0
        self.stop_calls = 0

    def set_output_handler(self, handler, replace_default: bool = False) -> None:
        self.output_handler = handler
        self.replace_default = replace_default
        if replace_default:
            self.output_router.default_output = handler

    async def start(self) -> None:
        self.is_running = True
        self.start_calls += 1

    async def stop(self) -> None:
        self.is_running = False
        self.stop_calls += 1

    async def inject_input(self, message, source: str = "programmatic") -> None:
        self.injected.append((message, source))
        self.output_handler("hello ")
        await asyncio.sleep(0)
        self.output_handler("world")


@pytest.mark.asyncio
async def test_agent_session_chat_streams_and_restores_default_output():
    agent = _FakeAgent()
    await agent.start()
    old_default = agent.output_router.default_output
    session = AgentSession(agent)

    chunks = [chunk async for chunk in session.chat("hi")]

    assert chunks == ["hello ", "world"]
    assert agent.injected == [("hi", "chat")]
    assert agent.replace_default is True
    assert agent.output_router.default_output is old_default


@pytest.mark.asyncio
async def test_agent_session_from_config_uses_none_input_and_starts(monkeypatch):
    monkeypatch.setattr(agent_session_module, "Agent", _FakeAgent)

    session = await AgentSession.from_config(
        AgentConfig(name="shim"),
        llm_override="test/override",
        pwd="C:/project",
    )

    assert session.agent.start_calls == 1
    assert session.agent.init_kwargs["llm_override"] == "test/override"
    assert session.agent.init_kwargs["pwd"] == "C:/project"
    assert isinstance(
        session.agent.init_kwargs["input_module"],
        agent_session_module.NoneInput,
    )


@pytest.mark.asyncio
async def test_agent_session_status_and_stop():
    agent = _FakeAgent()
    session = await AgentSession.from_agent(agent)

    status = session.get_status()
    await session.stop()

    assert status["running"] is True
    assert status["name"] == "fake"
    assert status["model"] == "test/model"
    assert status["tools"] == ["read"]
    assert status["subagents"] == ["worker"]
    assert status["pwd"] == "C:/work"
    assert agent.stop_calls == 1
