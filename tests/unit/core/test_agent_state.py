"""First session-state owner lifecycle slice on a real Agent."""

import asyncio

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: state\nsystem_prompt: offline\ninput: {type: none}\n"
        "output: {type: stdout}\n",
        encoding="utf-8",
    )
    return path


async def _agent(tmp_path, *, plugins=None):
    return await Agent.build(
        str(_config(tmp_path)),
        llm=ScriptedLLM(["ok"]),
        io="headless",
        pwd=tmp_path,
        plugins=plugins,
    )


def _owner_tasks():
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("session_state_owner") and not task.done()
    ]


class BoomPlugin(BasePlugin):
    name = "boom"


class BoomInput:
    """Fail input.start after the owner is expected to be running."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self._tui = None

    async def start(self):
        self.started = True
        raise RuntimeError("input start failed")

    async def stop(self):
        self.stopped = True


async def test_start_owner_exists_before_input_start(tmp_path):
    agent = await _agent(tmp_path)
    assert agent._session_state_owner is None
    seen = {}
    real_start = agent.input.start

    async def start():
        owner = agent._session_state_owner
        seen["running"] = (
            owner is not None and owner.task is not None and not owner.task.done()
        )
        await real_start()

    agent.input.start = start
    try:
        await agent.start()
        assert seen["running"] is True
        assert agent._session_state_owner.task is not None
        assert not agent._session_state_owner.task.done()
        assert agent.controller._session_state_owner is agent._session_state_owner
    finally:
        await agent.stop()


async def test_submit_callback_runs_on_owner_task(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    try:
        seen = []

        def callback():
            seen.append(asyncio.current_task())
            return "ran"

        result = await agent._submit_session_state(callback)
        assert result == "ran"
        assert seen == [agent._session_state_owner.task]
    finally:
        await agent.stop()


async def test_helper_from_owner_task_runs_inline_without_recursion(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    try:
        seen = []

        def nested():
            seen.append(asyncio.current_task())
            return "inline"

        def outer():
            result = agent._run_session_state(nested)
            assert not asyncio.isfuture(result)
            return result

        assert await agent._submit_session_state(outer) == "inline"
        assert seen == [agent._session_state_owner.task]
    finally:
        await agent.stop()


@pytest.mark.parametrize("inline", ["bootstrap", "owner"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_inline_callback_rejects_async_work(tmp_path, inline, wrapped):
    agent = await _agent(tmp_path)
    calls = []

    async def callback():
        calls.append("must not run")

    operation = (lambda: callback()) if wrapped else callback
    try:
        if inline == "owner":
            await agent.start()
            with pytest.raises(InvalidRequestError):
                await agent._submit_session_state(
                    lambda: agent._run_session_state(operation)
                )
        else:
            with pytest.raises(InvalidRequestError):
                agent._run_session_state(operation)
        assert calls == []
    finally:
        await agent.stop()


async def test_running_agent_does_not_mask_dead_owner(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    owner = agent._session_state_owner
    try:
        owner.task.cancel()
        await asyncio.gather(owner.task, return_exceptions=True)
        with pytest.raises(ConflictError, match="owner"):
            await agent.start()
        assert agent._session_state_owner is owner
    finally:
        await agent.stop()


async def test_owner_callback_rejects_coroutine(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    try:

        async def callback():
            return "nope"

        with pytest.raises(InvalidRequestError, match="synchronous"):
            agent._run_session_state(callback)
    finally:
        await agent.stop()


@pytest.mark.parametrize("started", [False, True])
async def test_post_stop_submit_fails_before_mutation(tmp_path, started):
    agent = await _agent(tmp_path)
    if started:
        await agent.start()
    await agent.stop()
    mutated = []

    def callback():
        mutated.append("mutated")

    with pytest.raises((InvalidRequestError, ConflictError)):
        agent._run_session_state(callback)
    assert mutated == []


async def test_input_start_failure_closes_owner_and_keeps_error(tmp_path):
    agent = await _agent(tmp_path)
    boom = BoomInput()
    agent.input = boom
    with pytest.raises(RuntimeError, match="input start failed"):
        await agent.start()
    owner = agent._session_state_owner
    assert owner is None or owner.task is None or owner.task.done()
    assert agent._running is False
    consumer = getattr(agent, "_consumer_task", None)
    assert consumer is None or consumer.done()
    assert boom.stopped is True
    assert _owner_tasks() == []


async def test_plugin_load_failure_closes_owner_and_consumer(tmp_path):
    agent = await _agent(tmp_path, plugins=[BoomPlugin()])

    async def fail_load():
        raise RuntimeError("plugin load failed")

    agent._load_plugins = fail_load
    with pytest.raises(RuntimeError, match="plugin load failed"):
        await agent.start()
    owner = agent._session_state_owner
    assert owner is None or owner.task is None or owner.task.done()
    assert agent._running is False
    assert agent._consumer_task is None
    assert _owner_tasks() == []


async def test_accepted_callback_cancel_preserves_cause_and_runs_once(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    owner = agent._session_state_owner
    queued = asyncio.Event()
    calls = []

    class GatedQueue(type(owner._queue)):
        def put(self, item):
            super().put(item)
            queued.set()

    gated = GatedQueue()
    while True:
        try:
            gated.put(owner._queue.get_nowait())
        except Exception:
            break
    queued.clear()
    owner._queue = gated

    def callback():
        calls.append("ran")
        raise RuntimeError("callback failed")

    waiter = asyncio.create_task(agent._submit_session_state(callback))
    await queued.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await waiter
    await asyncio.sleep(0)
    assert calls == ["ran"]
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "callback failed" in str(raised.value.__cause__)
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task is not owner.task
        and task.get_name() != f"event_consumer_{agent.config.name}"
    ]
    assert pending == []
    await agent.stop()


async def test_stop_during_start_rejects_before_teardown(tmp_path):
    agent = await _agent(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    real_start = agent.input.start

    async def start():
        entered.set()
        await release.wait()
        await real_start()

    agent.input.start = start
    starting = asyncio.create_task(agent.start())
    await entered.wait()
    try:
        with pytest.raises(ConflictError, match="starting"):
            await agent.stop()
        assert agent._session_state_owner.task is not None
        assert not agent._session_state_owner.task.done()
        assert agent._running is False
    finally:
        release.set()
        await starting
        await agent.stop()


async def test_stop_closes_owner_after_producers_and_before_output(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    order = []
    real_input_stop = agent.input.stop
    real_output_stop = agent.output_router.stop
    real_close = agent._session_state_owner.close

    async def input_stop():
        order.append("input")
        await real_input_stop()

    async def output_stop():
        order.append("output")
        await real_output_stop()

    async def owner_close():
        order.append("owner")
        await real_close()

    agent.input.stop = input_stop
    agent.output_router.stop = output_stop
    agent._session_state_owner.close = owner_close
    await agent.stop()
    assert order.index("input") < order.index("owner") < order.index("output")
    assert (
        agent._session_state_owner.task is None
        or agent._session_state_owner.task.done()
    )


async def test_start_reentry_and_running_are_idempotent(tmp_path):
    agent = await _agent(tmp_path)
    await agent.start()
    owner = agent._session_state_owner
    await agent.start()
    assert agent._session_state_owner is owner
    entered = asyncio.Event()
    release = asyncio.Event()

    original_stop = agent._stop_inner

    async def block_stop():
        entered.set()
        await release.wait()
        await original_stop()

    agent._stop_inner = block_stop
    stopping = asyncio.create_task(agent.stop())
    await entered.wait()
    try:
        with pytest.raises(ConflictError, match="stopping"):
            await agent.start()
    finally:
        release.set()
        await stopping
