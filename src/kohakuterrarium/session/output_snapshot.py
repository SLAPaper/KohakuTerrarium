"""Detached turn-snapshot inputs captured by the session-state owner."""

from copy import deepcopy

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.compact_branch import snapshot_branch_view


def _capture(agent):
    messages, branch_tag, state = None, None, {}
    if agent is not None and hasattr(agent, "controller"):
        messages = agent.controller.conversation.snapshot_messages(
            preserve_pending_tail=True
        )
    turn = getattr(agent, "_turn_index", None)
    branch = getattr(agent, "_branch_id", None)
    if type(turn) is int and turn > 0 and type(branch) is int and branch > 0:
        branch_tag = {
            "turn_index": turn,
            "branch_id": branch,
            "parent_branch_path": getattr(agent, "_parent_branch_path", None),
            "branch_view": snapshot_branch_view(agent),
        }
    session = getattr(agent, "session", None)
    pad = getattr(session, "scratchpad", None)
    if hasattr(pad, "to_dict"):
        state["scratchpad"] = pad.to_dict()
    return deepcopy((messages, branch_tag, state))


async def capture_turn_snapshot(agent):
    """Capture memory only; store submission and persistence remain outside the owner."""
    return await submit_internal_session_state(agent, lambda: _capture(agent))
