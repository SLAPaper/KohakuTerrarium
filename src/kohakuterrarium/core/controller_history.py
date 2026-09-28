"""Short owner commits for hosted controller conversation state."""

import asyncio
from copy import deepcopy

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.controller_generation import _canonical_output
from kohakuterrarium.core.controller_recovery import capture_recovery
from kohakuterrarium.core.conversation_elide import TOOL_FEEDBACK_KIND
from kohakuterrarium.core.events import EventType
from kohakuterrarium.core.feedback_admission import (
    fence_feedback,
    prepare_feedback,
    validate_feedback,
)
from kohakuterrarium.core.native_declaration import commit_native_declaration
from kohakuterrarium.core.primary_user_identity import event_content, take_primary_user
from kohakuterrarium.session.output_handoff import output_binding


async def commit_controller_state(controller, callback):
    """Use the host owner, retaining inline behavior for standalone controllers."""
    host = getattr(controller, "_state_host", None)
    owner = getattr(host, "_session_state_owner", None)
    if owner is None or asyncio.current_task() is owner.task:
        return await submit_internal_session_state(host, callback)
    return await _join_accepted(
        asyncio.create_task(submit_internal_session_state(host, callback))
    )


async def capture_turn_messages(controller, events, content, skip_empty):
    """Append admitted input metadata and detach one provider request snapshot."""
    content = deepcopy(content)
    events = tuple(events)

    async with output_binding(_canonical_output(controller)):
        return await _capture_turn_messages(controller, events, content, skip_empty)


async def _capture_turn_messages(controller, events, content, skip_empty):
    contents = [event_content(event) for event in events]
    identities = [
        deepcopy(event.context.get("_history_user_identity"))
        for event in events
        if event.context.get("rerun") and event.context.get("edited")
    ]
    feedback = bool(events) and all(
        event.type == EventType.TOOL_COMPLETE for event in events
    )

    receipt = await prepare_feedback(controller, events, content, skip_empty)

    def capture():
        if receipt is not None:
            validate_feedback(receipt)
        handled, primary_identity = take_primary_user(
            getattr(controller, "_state_host", None),
            controller,
            events,
            contents,
            skip_empty,
        )
        admitted_identities = [primary_identity] if handled else identities
        if not skip_empty:
            message = controller.conversation.append("user", content)
            for identity in admitted_identities:
                if isinstance(identity, dict):
                    message.metadata.update(
                        {
                            key: identity[key]
                            for key in ("event_id", "turn_index", "branch_id")
                            if key in identity
                        }
                    )
            if handled and primary_identity:
                for key in ("event_origin", "user_pair_origin", "user_pair_id"):
                    if key in primary_identity:
                        message.metadata[key] = deepcopy(primary_identity[key])
            if feedback or receipt is not None:
                message.metadata["kind"] = (
                    receipt.kind if receipt is not None else TOOL_FEEDBACK_KIND
                )
                if receipt is not None:
                    message.metadata.update(
                        turn_index=receipt.coordinates[0],
                        branch_id=receipt.coordinates[1],
                    )
                    message.metadata["feedback_admission_origin"] = deepcopy(
                        receipt.origin
                    )
        controller._recovery_capture = capture_recovery(controller)
        return deepcopy(controller.conversation.to_messages())

    try:
        return await commit_controller_state(controller, capture)
    except BaseException:
        if receipt is not None:
            await fence_feedback(controller)
        raise


async def append_assistant(controller, content, **kwargs):
    """Install detached assistant content before native execution dispatch."""
    content, kwargs = deepcopy((content, kwargs))
    if (
        kwargs.get("tool_calls")
        and getattr(controller, "_state_host", None) is not None
        and getattr(controller, "session_store", None) is not None
    ):
        await commit_native_declaration(controller, content, kwargs)
        return
    await commit_controller_state(
        controller,
        lambda: controller.conversation.append("assistant", content, **kwargs),
    )


async def capture_rewrite_target(controller):
    def capture():
        message = controller.conversation.get_last_assistant_message()
        return message, message.get_text_content() if message else ""

    return await commit_controller_state(controller, capture)


async def apply_rewrite(controller, message, original, rewritten):
    def apply():
        if (
            controller.conversation.get_last_assistant_message() is not message
            or message.get_text_content() != original
        ):
            return False
        message.content = rewritten
        return True

    return await commit_controller_state(controller, apply)
