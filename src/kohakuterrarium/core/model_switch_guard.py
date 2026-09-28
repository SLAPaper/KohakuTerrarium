"""Reject mutations that conflict with an accepted model replacement."""

import asyncio

from kohakuterrarium.errors import ConflictError


def require_topology_mutation_available(agent) -> None:
    task = getattr(agent, "_topology_handoff_task", None)
    if task is None:
        return
    try:
        caller = asyncio.current_task()
    except RuntimeError:
        caller = None
    if caller is not task:
        raise ConflictError("Graph handoff is awaiting completion")


def require_model_mutation_available(agent, *, allow_compact_pending=False) -> None:
    """Refuse changes to prepared model state before touching live collaborators."""
    require_topology_mutation_available(agent)
    if getattr(agent, "_session_attach_uncertain", False):
        raise ConflictError(
            "Session attachment rollback failed; cold recovery required"
        )
    if (
        not allow_compact_pending
        and getattr(agent, "_compact_commit_pending", None) is not None
    ):
        raise ConflictError(
            "Compaction confirmation is pending; failed commits require cold recovery"
        )
    if getattr(agent, "_session_attach_pending", False):
        raise ConflictError("Session attachment is awaiting completion")
    if getattr(agent, "_history_rerun_uncertain", False):
        raise ConflictError(
            "Regeneration durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_history_rerun_pending", False):
        raise ConflictError("Regeneration is awaiting persistence")
    if getattr(agent, "_history_selection_uncertain", False):
        raise ConflictError(
            "Branch selection durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_history_selection_pending", False):
        raise ConflictError("Branch selection is awaiting persistence")
    if getattr(agent, "_model_switch_uncertain", False):
        raise ConflictError(
            "Model selection durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_model_switch_pending", False):
        raise ConflictError("Model switch is awaiting persistence")
