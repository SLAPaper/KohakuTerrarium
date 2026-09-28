"""Detached tool dispatch provenance and terminal activity context."""

from copy import deepcopy
from uuid import uuid4


def capture_tool_context(agent, job_id, execution_class, tool_call_id=None):
    """Capture dispatch coordinates before the executor starts a task."""
    scope = getattr(agent, "_runtime_scope_id", None)
    if not scope:
        scope = agent._runtime_scope_id = str(uuid4())
    parents = [list(pair) for pair in getattr(agent, "_parent_branch_path", [])]
    turn = getattr(agent, "_turn_index", 0)
    branch = getattr(agent, "_branch_id", 0)
    origin = {
        "runtime_scope_id": scope,
        "origin_path": [*deepcopy(parents), [turn, branch]],
        "turn_index": turn,
        "branch_id": branch,
        "parent_branch_path": parents,
        "branch_view": {str(t): b for t, b in parents},
        "delivery_id": f"{scope}:{job_id}",
        "execution_class": execution_class,
    }
    generation = getattr(getattr(agent, "controller", None), "_active_generation", None)
    if isinstance(generation, dict) and generation.get("generation_id"):
        origin["generation_id"] = generation["generation_id"]
    context = {"origin": origin, "execution_class": execution_class}
    if tool_call_id:
        context["tool_call_id"] = tool_call_id
    return context


def _job_status(agent, job_id):
    for attribute in ("executor", "subagent_manager"):
        store = getattr(getattr(agent, attribute, None), "job_store", None)
        status = store.get_status(job_id) if store is not None else None
        if status is not None:
            return status
    return None


def promote_tool_context(agent, job_id):
    """Change delivery class while preserving the original lineage and identity."""
    context = tool_completion_context(agent, job_id)
    if not isinstance(context.get("origin"), dict):
        return
    context["execution_class"] = "background"
    context["origin"]["execution_class"] = "background"
    status = _job_status(agent, job_id)
    if status is not None:
        status.context = deepcopy(context)
    meta = getattr(agent, "_direct_job_meta", {}).get(job_id)
    if meta is not None:
        meta["completion_context"] = context


def tool_completion_context(agent, job_id):
    """Read current job delivery metadata, retaining the direct-handle fallback."""
    status = _job_status(agent, job_id)
    if status is not None:
        return deepcopy(status.context)
    meta = getattr(agent, "_direct_job_meta", {}).get(job_id, {})
    return deepcopy(meta.get("completion_context", {}))
