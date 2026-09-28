"""One-turn receipts linking ordinary user messages to accepted session writes."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.message import content_parts_to_dicts


@dataclass
class PrimaryUserReceipt:
    store: Any
    controller: Any
    controller_store: Any
    conversation: Any
    event: Any
    envelope: Any
    scope: Any
    namespace: str
    turn: int
    branch: int
    path: Any
    view: Any
    payload: dict
    eligible: bool
    event_id: int | None = None
    persisted_identity: dict | None = None
    armed: bool = False


def event_content(event):
    """Detach the serialized user content without adding provider context."""
    content = (
        content_parts_to_dicts(event.content)
        if hasattr(event, "is_multimodal") and event.is_multimodal()
        else (event.content or "")
    )
    return deepcopy(content)


def capture_primary_user(agent, event, pending_id):
    """Capture one write's source and optional singleton-admission identity."""
    run = getattr(agent, "_active_event_run", None) or []
    controller = agent.controller
    payload = {"content": event_content(event)}
    if pending_id:
        payload["pending_id"] = pending_id
    return PrimaryUserReceipt(
        agent.session_store,
        controller,
        controller.session_store,
        controller.conversation,
        event,
        run[0] if len(run) == 1 and run[0].event is event else None,
        getattr(agent, "_turn_completion", None),
        agent.config.name,
        agent._turn_index,
        agent._branch_id,
        deepcopy(agent._parent_branch_path),
        deepcopy(getattr(agent, "_branch_view", None)),
        payload,
        len(run) == 1
        and run[0].event is event
        and controller.session_store is agent.session_store,
    )


def _validate(agent, receipt):
    run = getattr(agent, "_active_event_run", None) or []
    if (
        agent.session_store is not receipt.store
        or agent.controller is not receipt.controller
        or receipt.controller.session_store is not receipt.controller_store
        or (
            receipt.controller_store is not None
            and receipt.controller_store is not receipt.store
        )
        or receipt.controller.conversation is not receipt.conversation
        or getattr(agent, "_turn_completion", None) is not receipt.scope
        or (receipt.scope is not None and receipt.scope.done())
        or agent.config.name != receipt.namespace
        or (agent._turn_index, agent._branch_id) != (receipt.turn, receipt.branch)
        or agent._parent_branch_path != receipt.path
        or getattr(agent, "_branch_view", None) != receipt.view
        or (
            receipt.eligible
            and (
                len(run) != 1
                or run[0] is not receipt.envelope
                or run[0].event is not receipt.event
            )
        )
    ):
        raise ConflictError("Primary user history binding changed")


def install_primary_user(agent, receipt, event_id):
    """Install a confirmed receipt only on its captured runtime binding."""
    _validate(agent, receipt)
    if receipt.scope is None:
        return
    if isinstance(event_id, dict):
        receipt.persisted_identity = deepcopy(event_id)
        event_id = event_id.get("event_id")
    if type(event_id) is not int or event_id <= 0:
        raise ConflictError("Invalid primary user history receipt")
    receipt.event_id = event_id
    agent._primary_user_receipt = receipt


def arm_primary_user(agent, controller):
    """Arm only a fresh controller queue occurrence."""
    receipt = getattr(agent, "_primary_user_receipt", None)
    if receipt is None:
        return
    _validate(agent, receipt)
    receipt.armed = (
        receipt.eligible
        and controller is receipt.controller
        and not controller.has_pending_events()
    )


def take_primary_user(agent, controller, events, contents, skip_empty):
    """Consume a receipt once, returning admission ownership and safe metadata."""
    receipt = getattr(agent, "_primary_user_receipt", None)
    if receipt is None:
        return False, None
    agent._primary_user_receipt = None
    _validate(agent, receipt)
    if (
        skip_empty
        or not receipt.armed
        or controller is not receipt.controller
        or len(events) != 1
        or events[0] is not receipt.event
        or receipt.event.type != "user_input"
        or receipt.event.context.get("rerun")
        or receipt.event.context.get("edited")
        or contents != [receipt.payload["content"]]
        or event_content(receipt.event) != receipt.payload["content"]
    ):
        return True, None
    return True, {
        **deepcopy(receipt.persisted_identity or {}),
        "event_id": receipt.event_id,
        "turn_index": receipt.turn,
        "branch_id": receipt.branch,
    }


def clear_primary_user(agent, scope):
    """Release the current turn's receipt without touching a successor turn."""
    receipt = getattr(agent, "_primary_user_receipt", None)
    if receipt is not None and receipt.scope is scope:
        agent._primary_user_receipt = None
