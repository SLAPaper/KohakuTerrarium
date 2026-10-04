"""Leave one confirmed generation's durable partial text after process exit."""

import asyncio
import os
import sys
from pathlib import Path

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


class PartialProvider(ScriptedLLM):
    def __init__(self):
        super().__init__(["unused"])
        self.partial = asyncio.Event()

    async def chat(self, messages, **kwargs):
        self.call_log.append(self._normalize_messages(messages))
        self.call_count += 1
        yield "P" * 600
        self.partial.set()
        await asyncio.Event().wait()


async def main(root):
    provider = PartialProvider()
    engine = Terrarium(session_dir=root / "sessions")
    await engine.__aenter__()
    creature = await engine.add_creature(
        AgentConfig(name="crash_recovery", tool_format="native"),
        llm=provider,
        pwd=str(root),
        io="headless",
        start=True,
    )
    asyncio.create_task(creature.run("original question", timeout=30))
    await asyncio.wait_for(provider.partial.wait(), 10)
    await creature.agent._session_output.drain()
    store = creature.agent.session_store
    await store.run(store.checkpoint, strict=True)
    os._exit(7)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
