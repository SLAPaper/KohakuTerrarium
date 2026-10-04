"""Message edit / regenerate / rewind mixin for Agent.

Modify past messages, regenerate responses, and replay conversation branches.
"""

import asyncio
from uuid import uuid4

from kohakuterrarium.core.agent_lifecycle import wait_for_turn_completion
from kohakuterrarium.core.agent_message_history import (
    live_user_turns as _live_user_turns,
    max_branch_id_for_turn as _max_branch_id_for_turn,
    previous_branch_user_content as _previous_branch_user_content,
    reload_conversation_under_branch_view as _reload_branch_view,
    resolve_edit_message_index as _resolve_edit_message_index,
    turn_index_for_user_position as _turn_index_for_user_position,
    user_message_content_for_turn as _user_message_content_for_turn,
)
from kohakuterrarium.core.branch_edit_prepare import EditRequest
from kohakuterrarium.core.branch_rerun import request_rerun
from kohakuterrarium.core.branch_rewind import rewind
from kohakuterrarium.core.branch_selection import select_branch as _select_branch
from kohakuterrarium.core.events import EventType, TriggerEvent
from kohakuterrarium.core.interrupt_regeneration import (
    completed_regeneration_result,
    get_regeneration_status as _get_regeneration_status,
    interrupt_and_regenerate as _interrupt_and_regenerate,
    request_regeneration as _request_regeneration,
)
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.message import normalize_content_parts
from kohakuterrarium.session.history import replay_conversation
from kohakuterrarium.session.raw_history import (
    UserMessageSelector,
    append_user_event_pair,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class AgentMessagesMixin:
    """Message edit / regenerate / rewind operations."""

    async def interrupt_and_regenerate(
        self,
        *,
        timeout: float = 10.0,
        turn_index: int | None = None,
        branch_view: dict[int, int] | None = None,
        request_id: str | None = None,
        target: UserMessageSelector | None = None,
    ) -> dict:
        """Return the admitted branch receipt after a source-pinned rerun succeeds."""
        return await _interrupt_and_regenerate(
            self,
            timeout=timeout,
            turn_index=turn_index,
            branch_view=branch_view,
            request_id=request_id,
            target=target,
        )

    def request_regeneration(
        self,
        *,
        timeout: float = 10.0,
        turn_index: int | None = None,
        branch_view: dict[int, int] | None = None,
        request_id: str | None = None,
        target: UserMessageSelector | None = None,
    ) -> str:
        """Submit a non-blocking regeneration request and return its token."""
        return _request_regeneration(
            self,
            timeout=timeout,
            turn_index=turn_index,
            branch_view=branch_view,
            request_id=request_id,
            target=target,
        )

    def get_regeneration_status(self, request_id: str) -> dict:
        """Query the latest retained request without treating acceptance as success."""
        return _get_regeneration_status(self, request_id)

    async def select_branch(self, branch_view: dict[int, int] | None = None) -> int:
        """Select a persisted branch without regenerating. Local idle only."""
        return await _select_branch(self, branch_view)

    async def regenerate_last_response(
        self,
        *,
        turn_index: int | None = None,
        branch_view: dict[int, int] | None = None,
        request_id: str | None = None,
        target: UserMessageSelector | None = None,
    ) -> None:
        """Regenerate an assistant response.

        With ``turn_index=None`` (default): re-runs the conversation
        tail's last turn. With ``turn_index`` set: re-runs that
        specific turn (creates a new branch under the current
        viewing subtree). The latter is the path "click retry on an
        old assistant message" — without a turn_index parameter the
        backend always defaults to the tail and the click silently
        targets the wrong message.

        ``branch_view`` lets a caller retry on a NON-LATEST branch.
        Without it the agent's in-memory conversation reflects the
        last run (latest subtree) and a retry on an older branch
        would target the wrong message. Frontend passes the user's
        current ``branchViewByTab`` selection through.

        Uses current model/settings — which may differ from when the
        original response was generated. Opens a new ``branch_id`` for
        the resolved ``turn_index`` so the original branch is preserved
        and addressable via the ``<x/N>`` navigator.
        """
        if target is not None or turn_index is not None:
            await request_rerun(
                self,
                append_user_event_pair,
                request_id=request_id,
                edit=EditRequest(
                    turn_index=turn_index,
                    branch_view=branch_view,
                    target=target,
                    regenerate=True,
                ),
            )
            return

        await self._regenerate_tail_response(request_id=request_id)

    async def regenerate_last_response_result(
        self,
        *,
        turn_index: int | None = None,
        branch_view: dict[int, int] | None = None,
        request_id: str | None = None,
        target: UserMessageSelector | None = None,
    ) -> dict:
        """Return a fixed settled receipt, or raise a failed-turn error; requires local idle."""
        edit = (
            EditRequest(
                turn_index=turn_index,
                branch_view=branch_view,
                target=target,
                regenerate=True,
            )
            if target is not None or turn_index is not None
            else None
        )
        outcome, receipt = await request_rerun(
            self,
            append_user_event_pair,
            request_id=request_id or uuid4().hex,
            edit=edit,
            return_receipt=True,
        )
        return completed_regeneration_result(outcome, receipt)

    async def _regenerate_tail_response(self, *, request_id: str | None) -> None:
        """Persist tail regeneration before publishing its context and rerun."""
        await request_rerun(self, append_user_event_pair, request_id=request_id)

    async def edit_and_rerun(
        self,
        message_idx: int,
        new_content: str,
        *,
        turn_index: int | None = None,
        user_position: int | None = None,
        branch_view: dict[int, int] | None = None,
        request_id: str | None = None,
        target: UserMessageSelector | None = None,
    ) -> bool:
        """Prepare an edit independently, persist it, then publish and rerun."""
        return await request_rerun(
            self,
            append_user_event_pair,
            request_id=request_id,
            edit=EditRequest(
                message_idx, new_content, turn_index, user_position, branch_view, target
            ),
        )

    async def edit_and_rerun_result(
        self,
        message_idx,
        new_content,
        *,
        turn_index=None,
        user_position=None,
        branch_view=None,
        request_id=None,
        target=None,
    ) -> dict | None:
        """Return immutable edit coordinates only after successful rerun settlement."""
        outcome, receipt = await request_rerun(
            self,
            append_user_event_pair,
            request_id=request_id or uuid4().hex,
            edit=EditRequest(
                message_idx, new_content, turn_index, user_position, branch_view, target
            ),
            return_receipt=True,
        )
        return completed_regeneration_result(outcome, receipt) if receipt else None

    async def _edit_and_rerun_result(self, message_idx, new_content, **options):
        """Compatibility alias for the public settled-result entrypoint."""
        return await self.edit_and_rerun_result(message_idx, new_content, **options)

    async def rewind_to(self, message_idx: int) -> None:
        """Drop messages from ``message_idx`` onward without re-running."""
        await rewind(self, message_idx)

    def _get_message_mutation_lock(self) -> asyncio.Lock:
        """Return the per-agent lock guarding destructive history mutations."""
        lock = getattr(self, "_message_mutation_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._message_mutation_lock = lock
        return lock

    async def _wait_for_history_finalization(self) -> None:
        """Wait for cancellation or finalization, never for an actively generating turn."""
        if (
            getattr(self, "_interrupt_requested", False)
            or getattr(self, "_processing_task", None) is None
        ):
            await wait_for_turn_completion(
                self, getattr(self, "_turn_completion", None)
            )

    def _ensure_history_mutation_idle(self) -> None:
        """Reject destructive history changes while another turn can observe it."""
        require_model_mutation_available(self)
        processing_lock = getattr(self, "_processing_lock", None)
        inbox = getattr(self, "_event_inbox", None)
        turn_active = processing_lock is not None and processing_lock.locked()
        input_pending = inbox is not None and len(inbox) > 0
        if turn_active or input_pending:
            raise ConflictError(
                "Cannot mutate conversation history while a turn is active"
            )

    def _ensure_rerun_available(self) -> None:
        """Reject reruns before mutation when the event loop cannot admit them."""
        if getattr(self, "_paused", False):
            raise ConflictError("Cannot rerun conversation while the agent is paused")
        if getattr(self, "_running", True) is False:
            raise ConflictError(
                "Cannot rerun conversation while the agent is not running"
            )

    async def _rerun_from_last(self, new_user_content: str | list = "") -> None:
        """Trigger an LLM turn from the current conversation state.

        Empty content regenerates from the existing user message; non-empty
        content records an edit. Normalization is required because downstream
        context formatting recognizes content-part objects rather than raw
        multimodal dictionaries.
        """
        edited = bool(new_user_content)
        normalised = normalize_content_parts(new_user_content)
        if normalised is None:
            normalised = new_user_content if isinstance(new_user_content, str) else ""
        event = TriggerEvent(
            type=EventType.USER_INPUT,
            content=normalised,
            context={
                "rerun": True,
                "edited": edited,
                "request_id": getattr(self, "_branch_request_id", None),
            },
            stackable=False,
        )
        await self._process_event(event)

    def _resolve_edit_message_index(
        self,
        msgs: list[object],
        message_idx: int,
        *,
        turn_index: int | None = None,
        user_position: int | None = None,
        branch_view: dict[int, int] | None = None,
    ) -> int | None:
        """Resolve by turn metadata or exact unique legacy-content matching."""
        return _resolve_edit_message_index(
            self,
            msgs,
            message_idx,
            turn_index=turn_index,
            user_position=user_position,
            branch_view=branch_view,
        )

    def _user_position_for_turn_index(
        self,
        turn_index: int,
        *,
        branch_view: dict[int, int] | None = None,
    ) -> int | None:
        """Return the visible user-position for a live turn_index."""
        for pos, candidate in enumerate(self._live_user_turns(branch_view=branch_view)):
            if candidate == turn_index:
                return pos
        return None

    def _live_user_turns(
        self,
        *,
        branch_view: dict[int, int] | None = None,
    ) -> list[int]:
        """Return live user turn_index values in visible order.

        "Live" must match what the user actually SEES — the freshest
        subtree of the branch tree. Older subtrees that have been
        orphaned by a higher-up edit/retry must NOT contribute (their
        turns inflate positions and make ``_user_position_for_turn_index``
        resolve to the wrong message).

        Defers to ``select_live_event_ids`` from ``session/history.py``
        so this stays in lock-step with the replay logic.
        """
        return _live_user_turns(self, branch_view=branch_view)

    def _turn_index_for_user_position(
        self,
        user_position: int,
        *,
        branch_view: dict[int, int] | None = None,
    ) -> int | None:
        """Return the ``turn_index`` of the ``user_position``-th live
        user_message event, or ``None`` if it cannot be resolved.

        Live = belonging to the chosen branch of its turn under
        ``branch_view`` (or the latest subtree when ``branch_view``
        is ``None``).
        """
        return _turn_index_for_user_position(
            self,
            user_position,
            branch_view=branch_view,
        )

    def _max_branch_id_for_turn(self, turn_index: int) -> int:
        """Return the largest ``branch_id`` recorded for ``turn_index``,
        or ``0`` if no branch yet exists."""
        return _max_branch_id_for_turn(self, turn_index)

    def _user_message_content_for_turn(
        self,
        turn_index: int,
        *,
        branch_view: dict[int, int] | None = None,
    ):
        """Return the ``user_message`` content recorded at the chosen
        branch of ``turn_index`` under ``branch_view``, or ``None``.

        Used by ``regenerate_last_response(turn_index=…)`` so retry
        clicks on a non-tail turn carry the same content forward into
        the new branch — semantically equivalent to "edit to identical
        content." When ``branch_view`` is given the lookup respects
        the user's current subtree (otherwise it picks the latest
        branch globally).
        """
        return _user_message_content_for_turn(
            self,
            turn_index,
            branch_view=branch_view,
        )

    def _reload_conversation_under_branch_view(
        self,
        branch_view: dict[int, int],
    ) -> None:
        """Replay ``branch_view`` and align the conversation and agent state.

        Branch selection changes only the displayed view, so runtime state must
        be reseated before an edit or retry can resolve the intended message.
        """
        _reload_branch_view(
            self,
            branch_view,
            replay=replay_conversation,
        )

    def _previous_branch_user_content(self):
        """Return the ``user_message`` content recorded for the most
        recent prior branch of ``self._turn_index``, or ``None`` if no
        such event is found.

        Used by ``regenerate_last_response`` to seed the new branch's
        ``user_message`` event with the same wording as the original
        branch (pure regen does not change the user message).
        """
        return _previous_branch_user_content(self)
