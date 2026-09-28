"""Source-bound completion receipts for named-output feedback occurrences."""

import asyncio
from copy import deepcopy
from uuid import uuid4

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.controller_generation import _canonical_output
from kohakuterrarium.core.feedback_admission import fence_feedback
from kohakuterrarium.llm.generation import provider_identity
from kohakuterrarium.session.output_delivery import persist_output_delivery
from kohakuterrarium.session.output_handoff import output_binding


def completion_payload(completion):
    return {
        "target": completion.target,
        "content": deepcopy(completion.content),
        "success": completion.success,
        "error": completion.error,
        "completed_at": completion.timestamp.isoformat(),
        "feedback_line": completion.to_feedback_line(),
    }


async def route_output_feedback(agent, controller, event):
    """Route external output outside admission, then confirm its finite receipt."""
    event = deepcopy(event)

    def capture():
        return (
            agent._turn_completion,
            deepcopy(controller._active_generation),
            provider_identity(controller.llm),
            controller.output_router,
            _canonical_output(controller),
        )

    scope, generation, provider, router, captured_sink = (
        await submit_internal_session_state(agent, capture)
    )
    try:
        completion = await router.route(event)
    except BaseException:
        await fence_feedback(controller)
        raise
    if captured_sink is None:
        return
    if completion is None:
        await fence_feedback(controller)
        raise ConflictError("Hosted output did not return a completion receipt")
    payload = {**completion_payload(completion), "output_id": str(uuid4())}

    def validate(sink):
        if (
            agent.controller is not controller
            or agent._turn_completion is not scope
            or (scope is not None and scope.done())
            or controller._active_generation != generation
            or provider_identity(controller.llm) != provider
            or controller.output_router is not router
            or _canonical_output(controller) is not sink
            or agent.session_store is not sink._store
        ):
            raise ConflictError("Output feedback generation changed")
        if completion_payload(completion) != {
            key: value for key, value in payload.items() if key != "output_id"
        }:
            raise ConflictError("Output completion changed before receipt")

    async def confirm():
        async with output_binding(captured_sink) as sink:
            await submit_internal_session_state(agent, lambda: validate(sink))
            proof = await sink._store.run(
                persist_output_delivery,
                sink._store,
                sink._event_key_prefix,
                generation,
                payload,
            )

            def install():
                validate(sink)
                registry = getattr(agent, "_output_feedback_receipts", None)
                if registry is None:
                    registry = agent._output_feedback_receipts = {}
                registry[id(completion)] = (
                    completion,
                    deepcopy(payload),
                    proof,
                    scope,
                )

            await submit_internal_session_state(agent, install)

    task = asyncio.create_task(confirm())
    try:
        await _join_accepted(task)
    except BaseException:
        if task.cancelled() or (task.done() and task.exception() is not None):
            await fence_feedback(controller)
        raise


def take_output_proofs(agent, completions):
    """Consume only confirmed, unchanged router completion occurrences."""
    registry = getattr(agent, "_output_feedback_receipts", None) or {}
    selected = [registry.pop(id(item), None) for item in completions]
    if any(item is None for item in selected):
        return None
    proofs = []
    for completion, (original, payload, proof, scope) in zip(completions, selected):
        if (
            completion is not original
            or scope is not agent._turn_completion
            or completion_payload(completion)
            != {key: value for key, value in payload.items() if key != "output_id"}
        ):
            raise ConflictError("Output feedback occurrence changed")
        proofs.append(deepcopy(proof))
    return proofs
