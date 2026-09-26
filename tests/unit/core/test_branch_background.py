"""Only fully classified work can survive branch selection."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.branch_background import background_work_allows_selection
from kohakuterrarium.core.event_inbox import EventEnvelope, EventInbox
from kohakuterrarium.core.events import (
    create_tool_complete_event,
    create_user_input_event,
)
from kohakuterrarium.core.job import JobState, JobStatus, JobStore, JobType
from kohakuterrarium.core.tool_origin import capture_tool_context


def _fixture():
    agent = SimpleNamespace(
        _runtime_scope_id="scope",
        _turn_index=1,
        _branch_id=1,
        _parent_branch_path=[],
        _event_inbox=EventInbox(),
        _trigger_backlog_stash=[],
    )
    context = capture_tool_context(agent, "job", "background")
    store = JobStore()
    status = JobStatus(
        job_id="job",
        job_type=JobType.TOOL,
        type_name="tool",
        state=JobState.RUNNING,
        context=context,
    )
    store.register(status)
    source = SimpleNamespace(
        job_store=store, _tasks={}, get_running_jobs=store.get_running_jobs
    )
    agent.executor = source
    agent.subagent_manager = None
    return agent, source, status


@pytest.mark.parametrize(
    "damage",
    [
        "direct",
        "legacy",
        "scope",
        "job",
        "kind",
        "partial",
        "pending_unknown",
        "pending_count",
        "input",
        "legacy_event",
        "unknown_stash",
    ],
)
def test_unproved_work_blocks_selection(damage):
    agent, source, status = _fixture()
    assert background_work_allows_selection(agent)
    if damage == "direct":
        status.context["execution_class"] = "direct"
    elif damage == "legacy":
        status.context = {}
    elif damage == "scope":
        status.context["origin"]["runtime_scope_id"] = "wrong"
    elif damage == "job":
        status.context["origin"]["delivery_id"] = "scope:other"
    elif damage == "kind":
        status.job_type = "unknown"
    elif damage == "partial":
        for key in ("turn_index", "branch_id", "parent_branch_path", "delivery_id"):
            status.context["origin"].pop(key)
    elif damage == "pending_unknown":
        source._tasks["other"] = SimpleNamespace(done=lambda: False)
    elif damage == "pending_count":
        source.get_pending_count = lambda: 1
    elif damage == "input":
        agent._event_inbox.put(EventEnvelope(create_user_input_event("input")))
    elif damage == "legacy_event":
        agent._event_inbox.put(
            EventEnvelope(create_tool_complete_event(job_id="job", content="legacy"))
        )
    else:
        agent._trigger_backlog_stash.append(create_user_input_event("stashed"))
    assert not background_work_allows_selection(agent)


@pytest.mark.parametrize("subagent", [False, True])
async def test_terminal_status_with_pending_cleanup_remains_provable(subagent):
    agent, source, status = _fixture()
    if subagent:
        status.job_type = JobType.SUBAGENT
        agent.subagent_manager, agent.executor = source, None
    gate = asyncio.Event()
    task = asyncio.create_task(gate.wait())
    source._tasks["job"] = task
    try:
        assert background_work_allows_selection(agent)
        status.state = JobState.DONE
        assert background_work_allows_selection(agent)
        event = create_tool_complete_event(
            job_id="job", content="result", **deepcopy(status.context)
        )
        agent._event_inbox.put(EventEnvelope(event))
        assert background_work_allows_selection(agent)
        status.context = {}
        assert not background_work_allows_selection(agent)
    finally:
        gate.set()
        await task
