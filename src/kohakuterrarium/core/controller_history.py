"""Short owner commits for hosted controller conversation state."""

from copy import deepcopy

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.controller_recovery import capture_recovery
from kohakuterrarium.core.conversation_elide import TOOL_FEEDBACK_KIND
from kohakuterrarium.core.events import EventType


async def commit_controller_state(controller, callback):
    """Use the host owner, retaining inline behavior for standalone controllers."""
    host = getattr(controller, "_state_host", None)
    return await submit_internal_session_state(host, callback)


async def capture_turn_messages(controller, events, content, skip_empty):
    """Append admitted input metadata and detach one provider request snapshot."""
    content = deepcopy(content)
    identities = [
        deepcopy(event.context.get("_history_user_identity"))
        for event in events
        if event.context.get("rerun") and event.context.get("edited")
    ]
    feedback = bool(events) and all(
        event.type == EventType.TOOL_COMPLETE for event in events
    )

    def capture():
        if not skip_empty:
            message = controller.conversation.append("user", content)
            for identity in identities:
                if isinstance(identity, dict):
                    message.metadata.update(
                        {
                            key: identity[key]
                            for key in ("event_id", "turn_index", "branch_id")
                            if key in identity
                        }
                    )
            if feedback:
                message.metadata["kind"] = TOOL_FEEDBACK_KIND
        controller._recovery_capture = capture_recovery(controller)
        return deepcopy(controller.conversation.to_messages())

    return await commit_controller_state(controller, capture)


async def append_assistant(controller, content, **kwargs):
    """Install detached assistant content before native execution dispatch."""
    content, kwargs = deepcopy((content, kwargs))
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
