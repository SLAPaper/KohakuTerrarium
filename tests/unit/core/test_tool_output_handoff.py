"""Tool result artifacts select an admitted store after the tool itself completes."""

import asyncio
import threading
from pathlib import Path

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.tool_output_persistence import normalize_completed_tool
from kohakuterrarium.llm.message import ImagePart
from kohakuterrarium.modules.tool.base import BaseTool, ToolResult
from kohakuterrarium.terrarium import session_coord
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_accepted_artifact_write_joins_cancellation_before_copy(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    class ImageTool(BaseTool):
        tool_name = "image_job"
        description = "Image product"

        async def _execute(self, args, **kwargs):
            return ToolResult(output="unused")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        source = alice.agent.session_store
        write = type(source).write_artifact
        paths = []

        def held(store, filename, data):
            if store is source:
                assert threading.get_ident() == source._affinity_worker_ident
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            paths.append((store, filename, bytes(data)))
            return write(store, filename, data)

        monkeypatch.setattr(type(source), "write_artifact", held)
        original = ToolResult(
            output=[ImagePart(url="data:image/png;base64,aW1hZ2UtYnl0ZXM=")]
        )
        pending = asyncio.create_task(
            normalize_completed_tool(
                alice.agent,
                ImageTool(),
                original,
                max_output=0,
                job_id="image_job_accepted",
            )
        )
        moving = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            original.output[0].url = "data:image/png;base64,Y2hhbmdlZA=="
            pending.cancel()
            moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
            await alice.agent._session_state_owner.submit_internal(lambda: None)
            assert not pending.done() and not moving.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 5)
            await asyncio.wait_for(moving, 8)
            assert paths[0][2] == b"image-bytes"
            destination = alice.agent.session_store
            artifact = Path(destination.path).with_suffix(".artifacts") / paths[0][1]
            assert artifact.read_bytes() == b"image-bytes"
            assert source._closed
        finally:
            release.set()
            await asyncio.gather(
                pending, *([moving] if moving else []), return_exceptions=True
            )


async def test_tool_image_finishing_during_copy_persists_on_destination_worker(
    tmp_path, monkeypatch
):
    tool_entered, finish_tool, returned, copying = (asyncio.Event() for _ in range(4))
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class ImageTool(BaseTool):
        tool_name = "image_job"
        description = "Return an image after a barrier"

        async def _execute(self, args, **kwargs):
            tool_entered.set()
            await finish_tool.wait()
            returned.set()
            return ToolResult(
                output=[ImagePart(url="data:image/png;base64,aW1hZ2UtYnl0ZXM=")]
            )

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            tools=[ImageTool()],
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent, source = alice.agent, alice.agent.session_store
        job = await agent.executor.submit("image_job", {}, is_direct=True)
        await asyncio.wait_for(tool_entered.wait(), 3)
        capture = session_coord.capture_runtime_transfer
        first = True

        def held(store, names):
            nonlocal first
            captured = capture(store, names)
            if first and store is source:
                first = False
                loop.call_soon_threadsafe(copying.set)
                assert release.wait(6)
            return captured

        write = type(source).write_artifact
        writes = []

        def observed(store, name, data):
            writes.append((store, threading.get_ident(), bytes(data)))
            return write(store, name, data)

        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held)
        monkeypatch.setattr(type(source), "write_artifact", observed)
        moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        try:
            await asyncio.wait_for(copying.wait(), 3)
            finish_tool.set()
            await asyncio.wait_for(returned.wait(), 3)
            await agent._session_state_owner.submit_internal(lambda: None)
            release.set()
            await asyncio.wait_for(moving, 8)
            result = await agent.executor.wait_for(job, timeout=3)
            assert result.error is None
            destination = agent.session_store
            assert writes == [
                (destination, destination._affinity_worker_ident, b"image-bytes")
            ]
            assert not result.output[0].url.startswith("data:")
            artifacts = result.metadata["session_metadata"]["artifacts"]
            assert len(artifacts) == 1
        finally:
            finish_tool.set()
            release.set()
            await asyncio.gather(moving, return_exceptions=True)
            await agent.executor.wait_for(job, timeout=3)
