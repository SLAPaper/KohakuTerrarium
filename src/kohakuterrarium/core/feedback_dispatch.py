"""Dispatch-phase proofs carried by text acknowledgements into composed feedback."""

from copy import deepcopy
from dataclasses import dataclass

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.feedback_admission import _arm
from kohakuterrarium.core.tool_origin import tool_completion_context
from kohakuterrarium.core.output_feedback import take_output_proofs


@dataclass(frozen=True)
class DispatchAcknowledgement:
    content: str
    proof: dict


def append_dispatch_ack(agent, job_id, content):
    """Capture one text acknowledgement at its owner-side dispatch occurrence."""
    if getattr(agent, "_bg_dispatch_acks", None) is None:
        agent._bg_dispatch_acks = []
    origin = tool_completion_context(agent, job_id).get("origin")
    if not isinstance(origin, dict):
        agent._bg_dispatch_acks.append(content)
        return
    agent._bg_dispatch_acks.append(
        DispatchAcknowledgement(
            content,
            {"mode": "dispatch_ack", "job_id": job_id, "origin": deepcopy(origin)},
        )
    )


async def queue_dispatch_ack(agent, job_id, content):
    await submit_internal_session_state(
        agent, lambda: append_dispatch_ack(agent, job_id, content)
    )


def arm_dispatch_feedback(agent, event, job_ids, acknowledgements, outputs=()):
    """Authorize an assembled feedback occurrence without claiming ACKs are terminals."""
    if any(not isinstance(ack, DispatchAcknowledgement) for ack in acknowledgements):
        return
    deliveries = take_output_proofs(agent, outputs)
    if deliveries is None:
        return
    for job_id in job_ids:
        claim = getattr(agent, "_job_terminal_claims", {}).get(job_id)
        origin = tool_completion_context(agent, job_id).get("origin")
        if (
            not claim
            or claim.get("kind") not in ("tool", "subagent")
            or not claim.get("admitted")
            or not isinstance(origin, dict)
        ):
            return
        proof = {"job_id": job_id, "origin": deepcopy(origin)}
        if claim["kind"] == "subagent":
            proof["mode"] = "direct_child"
        deliveries.append(proof)
    deliveries.extend(deepcopy(ack.proof) for ack in acknowledgements)
    if deliveries:
        _arm(agent, event, deliveries)
