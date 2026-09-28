"""Persist native acknowledgement proofs before their owner-side conversation append."""

import asyncio
from copy import deepcopy

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.interrupted_history import (
    append_pending_result,
    pending_result_slot,
)
from kohakuterrarium.core.tool_origin import tool_completion_context
from kohakuterrarium.session.native_ack import persist_ack
from kohakuterrarium.session.output_handoff import output_binding
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


async def append_ack(agent, job_id, provider_id, name, content):
    """Join one finite acknowledgement capture, durable receipt, and live install."""
    meta = {"tool_call_id": provider_id, "name": name, "history_phase": "ack"}
    content = deepcopy(content)

    async def accepted():
        async with output_binding(getattr(agent, "_session_output", None)) as sink:

            def _append_background_ack_capture():
                slot = pending_result_slot(agent, job_id, meta)
                if not slot:
                    return None
                conversation, call_id, label, declaration = slot
                origin = tool_completion_context(agent, job_id).get("origin")
                proof = declaration.metadata.get("native_declaration_origin")
                if sink is None or proof is None or not origin:
                    return append_pending_result(agent, job_id, meta, content)
                payload = {
                    "job_id": job_id,
                    "tool_call_id": call_id,
                    "name": label,
                    "output": content,
                    "phase": "ack",
                    "origin": deepcopy(origin),
                    "declaration_origin": deepcopy(proof),
                }
                return (
                    conversation,
                    declaration,
                    payload,
                    agent.controller,
                    agent._turn_completion,
                )

            captured = await submit_internal_session_state(
                agent, _append_background_ack_capture
            )
            if captured is None or isinstance(captured, bool):
                return bool(captured)
            conversation, declaration, payload, controller, scope = captured
            store = sink._store

            def install():
                if (
                    agent.controller is not controller
                    or agent.session_store is not store
                    or controller.session_store is not store
                    or agent._turn_completion is not scope
                    or controller.conversation is not conversation
                ):
                    raise ConflictError(
                        "Native acknowledgement binding changed before install"
                    )
                slot = pending_result_slot(agent, job_id, meta)
                if not slot or slot[3] is not declaration:
                    raise ConflictError(
                        "Native acknowledgement slot changed before install"
                    )
                if (
                    tool_completion_context(agent, job_id).get("origin")
                    != payload["origin"]
                ):
                    raise ConflictError(
                        "Native acknowledgement delivery changed before install"
                    )
                return append_pending_result(
                    agent, job_id, {**meta, "ack_origin": ack_origin}, content
                )

            try:
                ack_origin, payload = await store.run(
                    persist_ack, store, sink._event_key_prefix, payload
                )
                return await submit_internal_session_state(agent, install)
            except Exception:
                try:
                    await submit_internal_session_state(
                        agent, lambda: setattr(agent, "_session_attach_uncertain", True)
                    )
                except Exception:
                    logger.error(
                        "Native acknowledgement recovery fence was not confirmed",
                        exc_info=True,
                    )
                raise

    return await _join_accepted(asyncio.create_task(accepted()))
