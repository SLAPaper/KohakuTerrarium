"""Reject mutations that conflict with an accepted model replacement."""

from kohakuterrarium.errors import ConflictError


def require_model_mutation_available(agent) -> None:
    """Refuse changes to prepared model state before touching live collaborators."""
    if getattr(agent, "_session_attach_uncertain", False):
        raise ConflictError(
            "Session attachment rollback failed; cold recovery required"
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
