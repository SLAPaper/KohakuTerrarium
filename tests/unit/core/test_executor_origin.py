"""Executor delivery context is registered before tool execution and detached."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium.core.executor import Executor
from kohakuterrarium.core.job import JobStore
from kohakuterrarium.modules.tool.base import BaseTool, ToolResult


class HeldTool(BaseTool):
    tool_name = "held"
    description = "Deterministic held execution"

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def _execute(self, args, **kwargs):
        self.entered.set()
        await self.release.wait()
        if args.get("fail"):
            raise RuntimeError("failed")
        return ToolResult(output="done")


@pytest.mark.parametrize("direct", [False, True])
async def test_authoritative_context_updates_are_detached_and_direct_is_silent(direct):
    deliveries = []
    executor = Executor(on_complete=deliveries.append)
    tool = HeldTool()
    executor.register_tool(tool)
    job = await executor.submit(
        "held", {}, is_direct=direct, completion_context={"origin": {"branch_id": 1}}
    )
    task = executor.get_task(job)
    try:
        await asyncio.wait_for(tool.entered.wait(), 3)
        status = executor.job_store.get_status(job)
        status.context = {"origin": {"branch_id": 2}, "execution_class": "background"}
        tool.release.set()
        await task
        assert len(deliveries) == (0 if direct else 1)
        if not direct:
            assert deliveries[0].context["origin"] == {"branch_id": 2}
        detached = executor.job_store.completion_context(job)
        detached["origin"]["branch_id"] = 99
        assert executor.job_store.get_status(job).context["origin"]["branch_id"] == 2
    finally:
        tool.release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_zero_retention_keeps_completion_origin():
    deliveries = []
    executor = Executor(
        job_store=JobStore(max_completed=0), on_complete=deliveries.append
    )
    tool = HeldTool()
    executor.register_tool(tool)
    expected = {"tool_call_id": "provider", "origin": {"branch_id": 1}}
    job = await executor.submit("held", {}, completion_context=expected)
    task = executor.get_task(job)
    tool.release.set()
    await task
    assert executor.job_store.get_status(job) is None
    assert deliveries[0].context["origin"] == expected["origin"]
    assert deliveries[0].context["tool_call_id"] == "provider"


async def test_context_cannot_override_terminal_outcome():
    executor = Executor()
    tool = HeldTool()
    executor.register_tool(tool)
    with pytest.raises(ValueError):
        await executor.submit(
            "held", {}, job_id="reserved", completion_context={"error": "forged"}
        )
    assert executor.get_task("reserved") is None
    assert executor.job_store.get_status("reserved") is None


@pytest.mark.parametrize(
    "mode", ["success", "error", "cancel", "cancel_before_entry", "manual_rejection"]
)
async def test_completion_context_survives_all_terminal_paths(mode):
    arrived = asyncio.Event()
    deliveries = []

    def deliver(event):
        deliveries.append(event)
        arrived.set()

    executor = Executor(on_complete=deliver)
    tool = HeldTool()
    if mode == "manual_rejection":
        tool.require_manual_read = True
    executor.register_tool(tool)
    context = {
        "origin": {"turn_index": 2, "branch_id": 1, "parent_branch_path": [[1, 1]]},
        "execution_class": "background",
        "tool_call_id": "provider",
    }
    expected = deepcopy(context)
    task = None
    try:
        job = await executor.submit(
            "held",
            {"fail": mode == "error"},
            job_id="held_job",
            completion_context=context,
        )
        task = executor.get_task(job)
        assert executor.job_store.get_status(job).context == expected
        context["origin"]["branch_id"] = 99
        if mode == "cancel_before_entry":
            task.cancel()
        elif mode == "manual_rejection":
            pass
        else:
            await asyncio.wait_for(tool.entered.wait(), 3)
            if mode == "cancel":
                task.cancel()
            else:
                tool.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(arrived.wait(), 3)
        assert len(deliveries) == 1
        for key, value in expected.items():
            assert deliveries[0].context[key] == value
        deliveries[0].context["origin"]["branch_id"] = 123
        assert executor.job_store.get_status(job).context == expected
    finally:
        tool.release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
