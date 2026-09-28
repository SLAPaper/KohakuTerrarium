"""Publish response origins at controller chat boundaries."""

from copy import deepcopy
from typing import Any

from kohakuterrarium.errors import ConflictError, SessionError
from kohakuterrarium.llm.generation import generation_payload, provider_identity

_CANONICAL_OUTPUT = "_kt_canonical_session_output"


def _canonical_output(controller: Any) -> Any | None:
    """The router-attached SessionOutput for this controller's store, if any."""
    store = getattr(controller, "session_store", None)
    router = getattr(controller, "output_router", None)
    if store is None:
        return None
    if router is None:
        raise SessionError("Session store has no canonical generation router")
    targets = [getattr(router, "default_output", None)]
    targets.extend(getattr(router, "_secondary_outputs", []) or [])
    sinks = [
        target
        for target in targets
        if getattr(target, "_store", None) is store
        and getattr(type(target), _CANONICAL_OUTPUT, False)
    ]
    if len(sinks) != 1:
        raise SessionError(
            "Session store requires exactly one canonical generation sink"
        )
    return sinks[0]


async def start_generation(controller: Any) -> dict[str, Any]:
    """Persist one origin and install it only after that write completes."""
    controller._active_generation = None
    identity = provider_identity(controller.llm)
    tool_format = controller.config.tool_format
    payload = generation_payload(controller.llm, controller.config.tool_format)
    captured = deepcopy(payload)
    sink = _canonical_output(controller)
    if sink is not None:
        await sink._begin_generation(captured)
        sink = _canonical_output(controller)
    router = getattr(controller, "output_router", None)
    notify = getattr(router, "_notify_generation_observers", None)
    if callable(notify):
        notify(deepcopy(captured))
    if (
        provider_identity(controller.llm) != identity
        or controller.config.tool_format != tool_format
    ):
        if sink is not None:
            sink._generation_id = sink._generation_coordinates = None
        raise ConflictError("Provider changed before generation chat")
    controller._active_generation = captured
    return captured
