"""Reject mutations that conflict with an accepted model replacement."""

from kohakuterrarium.errors import ConflictError


def require_model_mutation_available(agent) -> None:
    """Refuse changes to prepared model state before touching live collaborators."""
    if getattr(agent, "_model_switch_uncertain", False):
        raise ConflictError(
            "Model selection durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_model_switch_pending", False):
        raise ConflictError("Model switch is awaiting persistence")
