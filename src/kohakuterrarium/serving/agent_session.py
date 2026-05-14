"""Legacy AgentSession compatibility wrapper."""

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from kohakuterrarium.builtins.inputs.none import NoneInput
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config_types import AgentConfig


class AgentSession:
    """Small compatibility facade around a standalone :class:`Agent`."""

    def __init__(self, agent: Agent, agent_id: str | None = None) -> None:
        self.agent = agent
        self.agent_id = agent_id or f"agent_{uuid4().hex[:8]}"
        self._running = bool(getattr(agent, "is_running", False))

    @classmethod
    async def from_path(
        cls,
        config_path: str,
        *,
        llm_override: str | None = None,
        pwd: str | None = None,
    ) -> "AgentSession":
        agent = Agent.from_path(
            config_path,
            input_module=NoneInput(),
            llm_override=llm_override,
            pwd=pwd,
        )
        return await cls.from_agent(agent)

    @classmethod
    async def from_config(
        cls,
        config: AgentConfig,
        *,
        llm_override: str | None = None,
        pwd: str | None = None,
    ) -> "AgentSession":
        agent = Agent(
            config,
            input_module=NoneInput(),
            llm_override=llm_override,
            pwd=pwd,
        )
        return await cls.from_agent(agent)

    @classmethod
    async def from_agent(cls, agent: Agent) -> "AgentSession":
        session = cls(agent)
        await session.start()
        return session

    async def start(self) -> None:
        if self._running and getattr(self.agent, "is_running", False):
            return
        await self.agent.start()
        self._running = True

    async def stop(self) -> None:
        if not self._running and not getattr(self.agent, "is_running", False):
            return
        self._running = False
        await self.agent.stop()

    async def chat(self, message: str | list[dict]) -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        router = getattr(self.agent, "output_router", None)
        old_default = getattr(router, "default_output", None)
        self.agent.set_output_handler(queue.put_nowait, replace_default=True)

        inject_task = asyncio.create_task(
            self.agent.inject_input(message, source="chat")
        )
        try:
            while not inject_task.done():
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
            while not queue.empty():
                yield queue.get_nowait()
            await inject_task
        finally:
            if router is not None:
                router.default_output = old_default

    def get_status(self) -> dict:
        llm = getattr(self.agent, "llm", None)
        llm_config = getattr(llm, "config", None)
        model = getattr(llm, "model", "") or getattr(llm_config, "model", "")
        executor = getattr(self.agent, "executor", None)
        return {
            "agent_id": self.agent_id,
            "name": self.agent.config.name,
            "running": self._running and getattr(self.agent, "is_running", False),
            "model": model,
            "tools": self.agent.tools,
            "subagents": self.agent.subagents,
            "pwd": str(getattr(executor, "_working_dir", "")),
        }


__all__ = ["AgentSession"]
