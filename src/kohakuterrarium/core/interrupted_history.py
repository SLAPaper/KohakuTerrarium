"""Claim terminal outcomes and pair results with their current native occurrence."""

from copy import deepcopy
from types import SimpleNamespace

from kohakuterrarium.core.background_origin import completion_identity, lineage_accepts
from kohakuterrarium.core.message_locator import is_turn_user_message
from kohakuterrarium.core.tool_origin import tool_completion_context


def claim_job_terminal(agent, job_id):
    """Reserve one direct completion, stop, or interruption outcome on the owner."""
    finalized = getattr(agent, "_job_terminal_claims", None)
    if finalized is None:
        finalized = agent._job_terminal_claims = {}
    if job_id in finalized:
        return False
    finalized[job_id] = None
    return True


def pending_result_slot(agent, job_id, meta):
    """Locate a missing result in the current native block and captured lineage."""
    conversation = getattr(getattr(agent, "controller", None), "conversation", None)
    if conversation is None:
        return False
    provider_id = meta.get("tool_call_id") or job_id
    messages = conversation.get_messages()
    index = next(
        (
            i
            for i in reversed(range(len(messages)))
            if any(
                call.get("id") == provider_id for call in (messages[i].tool_calls or [])
            )
        ),
        None,
    )
    if index is None:
        return False
    tail = messages[index + 1 :]
    if any(
        message.role != "tool" or message.tool_call_id == provider_id
        for message in tail
    ):
        return False
    context = tool_completion_context(agent, job_id) or meta.get(
        "completion_context", {}
    )
    if any(key in context for key in ("origin", "execution_class")):
        origin = context.get("origin")
        if (
            not isinstance(origin, dict)
            or context.get("execution_class") not in ("direct", "background")
            or origin.get("execution_class") != context["execution_class"]
        ):
            return False
        probe = deepcopy(context)
        probe["execution_class"] = probe["origin"]["execution_class"] = "background"
        identity = completion_identity(
            SimpleNamespace(job_id=job_id, context=probe),
            getattr(agent, "_runtime_scope_id", None),
        )
        if (
            identity is None
            or identity[1] is None
            or not lineage_accepts(agent, identity[0])
        ):
            return False
        user = next(
            (m for m in reversed(messages[:index]) if is_turn_user_message(m)), None
        )
        if user is not None:
            for key in ("turn_index", "branch_id"):
                value = user.metadata.get(key)
                if value is not None and value != origin[key]:
                    return False
    call = next(
        call for call in messages[index].tool_calls if call.get("id") == provider_id
    )
    name = call.get("function", {}).get("name") or meta.get("name")
    return conversation, provider_id, name


def append_pending_result(agent, job_id, meta, content):
    """Append only a missing result in the last pending native call block."""
    slot = pending_result_slot(agent, job_id, meta)
    if not slot:
        return False
    conversation, provider_id, name = slot
    original = conversation._messages
    snapshot = list(original)
    metadata = deepcopy(conversation._metadata)
    try:
        conversation.append("tool", content, tool_call_id=provider_id, name=name)
    except BaseException:
        original[:] = snapshot
        conversation._messages = original
        conversation._metadata = metadata
        raise
    return True
