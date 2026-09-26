"""Prove that outstanding work can survive a local branch selection."""

from types import SimpleNamespace

from kohakuterrarium.core.background_origin import completion_identity
from kohakuterrarium.core.job import JobType


def _classified(event, scope):
    if getattr(event, "type", None) not in ("tool_complete", "subagent_output"):
        return False
    context = getattr(event, "context", None)
    if not isinstance(context, dict) or context.get("execution_class") != "background":
        return False
    identity = completion_identity(event, scope)
    return identity is not None and identity[1] is not None


def _status_is_background(status, scope):
    if getattr(status, "job_type", None) not in (JobType.TOOL, JobType.SUBAGENT):
        return False
    event = SimpleNamespace(
        type="tool_complete", job_id=status.job_id, context=status.context
    )
    return _classified(event, scope)


def background_work_allows_selection(agent):
    """Allow only fully identified background jobs and queued terminal events."""
    scope = getattr(agent, "_runtime_scope_id", None)
    for source in (
        getattr(agent, "executor", None),
        getattr(agent, "subagent_manager", None),
    ):
        if source is None:
            continue
        running = getattr(source, "get_running_jobs", None)
        statuses = running() if callable(running) else []
        if any(not _status_is_background(status, scope) for status in statuses):
            return False
        tasks = getattr(source, "_tasks", {})
        pending_ids = [job_id for job_id, task in tasks.items() if not task.done()]
        store = getattr(source, "job_store", None)
        for job_id in pending_ids:
            status = store.get_status(job_id) if store is not None else None
            if status is None or not _status_is_background(status, scope):
                return False
        pending_count = getattr(source, "get_pending_count", None)
        if callable(pending_count) and pending_count() != len(pending_ids):
            return False
    inbox = getattr(agent, "_event_inbox", None)
    if inbox is not None and len(inbox):
        if not hasattr(inbox, "has_event") or inbox.has_event(
            lambda event: not _classified(event, scope)
        ):
            return False
    return not getattr(agent, "_trigger_backlog_stash", None)
