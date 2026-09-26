"""Editable user identity excludes retained child feedback messages."""

from types import SimpleNamespace

from kohakuterrarium.core.agent_message_history import resolve_edit_message_index
from kohakuterrarium.core.message_locator import (
    is_turn_user_message,
    user_message_indices_for_content,
    user_message_indices_for_turn,
)
from kohakuterrarium.llm.message import Message


def test_child_feedback_is_not_a_user_locator_candidate():
    feedback = Message(
        role="user", content="same", metadata={"kind": "tool_results", "turn_index": 2}
    )
    user = Message(role="user", content="same", metadata={"turn_index": 2})
    messages = [Message(role="system", content="system"), feedback, user]
    assert not is_turn_user_message(feedback)
    assert is_turn_user_message(user)
    assert user_message_indices_for_turn(messages, 2) == [2]
    assert user_message_indices_for_content(messages, "same") == [2]
    agent = SimpleNamespace(session_store=None)
    assert resolve_edit_message_index(agent, messages, -1, user_position=0) == 2
