"""Non-blocking context compaction with atomic summary splicing."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from kohakuterrarium.core.agent_state import (
    submit_internal_session_state,
    submit_session_state,
)
from kohakuterrarium.core.compact_branch import (
    read_agent_events,
    build_compact_metadata,
    persist_compacted_async,
)
from kohakuterrarium.core.compact_splice import (
    select_compact_boundary,
    splice_conversation,
)
from kohakuterrarium.core.compact_state import apply_compact, capture_compact
from kohakuterrarium.core.compact_text import (
    COMPACT_PROMPT,
    extract_message_text,
)
from kohakuterrarium.core.conversation_elide import elide_after_compact
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.single_flight import SingleFlightDispatch, SingleFlightLease
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

# Defaults for auto-compaction.
# max_tokens is the model's context window size. Set this to match
# the model you are using (e.g. 272000 for gpt-5.4, 1000000 for Gemini 3).
# threshold triggers compaction when prompt_tokens reaches this fraction
# of max_tokens. If context somehow exceeds max_tokens, emergency truncation.
DEFAULT_MAX_TOKENS = 256_000
DEFAULT_THRESHOLD = 0.80  # compact when prompt_tokens >= 80% of max_tokens
DEFAULT_TARGET = 0.50  # aim for 50% of max_tokens after compact
DEFAULT_KEEP_RECENT = 8  # keep last 8 turns raw (not summarized)

# Usage from an LLM round that STARTED before a splice can land long
# after it (long rounds outlive the cooldown). Within this window after
# a splice, a trigger must be corroborated by the live conversation.
STALE_USAGE_WINDOW_SECONDS = 900.0


@dataclass
class CompactConfig:
    """Configuration for auto-compaction."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    threshold: float = DEFAULT_THRESHOLD
    target: float = DEFAULT_TARGET
    keep_recent_turns: int = DEFAULT_KEEP_RECENT
    enabled: bool = True
    cooldown_seconds: float = 30.0
    # If set, use a different model for summarization (cheaper/faster)
    compact_model: str | None = None


class CompactManager:
    """Manage non-blocking context compaction after LLM calls."""

    def __init__(self, config: CompactConfig | None = None):
        self.config = config or CompactConfig()
        self._dispatch = SingleFlightDispatch()
        self._compact_task: asyncio.Task | None = None
        self._splice_ready = asyncio.Event()
        self._splice_ready.set()
        self._active_lease: SingleFlightLease | None = None
        self._last_compact_time: float = 0
        self._compact_count: int = 0
        self._last_summary_error: str = ""
        # Why ``trigger_compact`` last returned ``False``. Read by the
        # ``/compact`` slash command (``builtins/user_commands/compact.py``)
        # to surface a precise reason instead of the generic "busy"
        # message. One of: ``""`` (success or never called),
        # ``"no_controller"``, ``"too_short"``, ``"busy"``.
        self._last_skip_reason: str = ""
        # References set by agent
        self._controller: Any = None
        self._agent: Any = None
        self._llm: Any = None
        self._session_store: Any = None
        self._output_router: Any = None
        self._agent_name: str = ""
        # Plugin manager (set by agent via _init_compact_manager when
        # plugins are wired up). Used for on_compact_start veto + the
        # on_compact_end notification. ``None`` = no-op.
        self._plugins: Any = None

    @property
    def is_compacting(self) -> bool:
        return self._dispatch.is_running

    def should_compact(self, prompt_tokens: int = 0) -> bool:
        """Check if compaction should be triggered.

        Uses prompt_tokens from the last LLM call. Token-based only,
        no character estimation fallback.
        """
        if not self.config.enabled or self.is_compacting:
            return False
        if self._last_compact_time:
            elapsed = time.time() - self._last_compact_time
            if elapsed < self.config.cooldown_seconds:
                return False

        if prompt_tokens <= 0:
            return False
        if prompt_tokens < self.config.max_tokens * self.config.threshold:
            return False
        return not self._usage_is_stale_after_splice()

    def _usage_is_stale_after_splice(self) -> bool:
        """True when a claimed over-threshold usage can't be justified
        by the live conversation shortly after a splice (the usage came
        from a round that started pre-splice)."""
        if not self._last_compact_time or not self._controller:
            return False
        if time.time() - self._last_compact_time > STALE_USAGE_WINDOW_SECONDS:
            return False
        try:
            messages = self._controller.conversation.get_messages()
        except Exception:
            return False
        chars = sum(len(extract_message_text(m) or "") for m in messages)
        # ``chars // 2`` deliberately over-estimates tokens so only a
        # conversation FAR below threshold vetoes the trigger.
        return chars // 2 < self.config.max_tokens * self.config.threshold

    def trigger_compact(self) -> bool:
        """Dispatch one background compaction and return whether it started.

        Busy or insufficient-history requests return ``False`` without starting
        a job. Dispatched jobs emit ``compact_start`` and then either
        ``compact_complete`` or ``compact_skipped``.
        """
        require_model_mutation_available(self._agent)
        if not self._controller:
            self._last_skip_reason = "no_controller"
            return False

        # Pre-check size. The same logic is replicated inside _run_compact
        # for the early-return path; we check here too so we can emit a
        # clean ``compact_skipped`` without flipping the compacting flag
        # or misleadingly showing a "compacting..." banner.
        messages = self._controller.conversation.get_messages()
        boundary = self._compact_boundary(messages)
        if boundary <= 1:
            if self._output_router:
                self._output_router.notify_activity(
                    "compact_skipped",
                    "Not enough context to compact — nothing to summarize",
                    metadata={
                        "reason": "too_short",
                        "round": self._compact_count + 1,
                    },
                )
                # Wave B additive ``compact_decision`` — records that
                # the manager saw the pre-check and chose to skip.
                self._output_router.notify_activity(
                    "compact_decision",
                    f"[{self._agent_name}] skipped (too_short)",
                    metadata={
                        "reason": "too_short",
                        "tokens_before": 0,
                        "tokens_after": 0,
                        "skipped": True,
                    },
                )
            logger.info(
                "Compact skipped — conversation too short",
                agent=self._agent_name,
                message_count=len(messages),
                boundary=boundary,
            )
            self._last_skip_reason = "too_short"
            return False

        lease = self._dispatch.try_acquire()
        if lease is None:
            logger.debug(
                "Compact trigger ignored — already running",
                agent=self._agent_name,
            )
            self._last_skip_reason = "busy"
            if self._output_router:
                self._output_router.notify_activity(
                    "compact_decision",
                    f"[{self._agent_name}] ignored (busy)",
                    metadata={
                        "reason": "busy",
                        "tokens_before": 0,
                        "tokens_after": 0,
                        "skipped": True,
                    },
                )
            return False

        self._active_lease = lease

        # Emit compact_start immediately (before background task)
        if self._output_router:
            self._output_router.notify_activity(
                "compact_start",
                f"Compacting context (round {self._compact_count + 1})",
                metadata={"round": self._compact_count + 1},
            )
            # Wave B ``compact_decision`` — triggered path.
            self._output_router.notify_activity(
                "compact_decision",
                f"[{self._agent_name}] triggered",
                metadata={
                    "reason": "threshold",
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "skipped": False,
                },
            )

        self._last_skip_reason = ""
        self._splice_ready.clear()
        self._compact_task = asyncio.create_task(self._run_compact(lease))
        logger.info(
            "Auto-compact triggered",
            agent=self._agent_name,
            compact_count=self._compact_count + 1,
        )
        return True

    def _emit_compact_skipped(self, reason: str, message: str) -> None:
        """Terminal for a started round that did not complete. Every
        ``compact_start`` must be closed by exactly one terminal
        (``compact_complete`` or ``compact_skipped``) — FE / TUI / rich
        CLI all keep a "compacting…" indicator open until one lands."""
        if self._output_router:
            self._output_router.notify_activity(
                "compact_skipped",
                message,
                metadata={
                    "reason": reason,
                    "round": self._compact_count + 1,
                },
            )

    async def _run_compact(self, lease: SingleFlightLease | None = None) -> None:
        """Background compaction task."""
        if lease is None:
            lease = self._dispatch.try_acquire()
            if lease is None:
                logger.debug(
                    "Compact run ignored — already running",
                    agent=self._agent_name,
                )
                return
        self._active_lease = lease
        self._splice_ready.clear()
        terminal_sent = False
        abort_reason = "aborted"
        try:
            captured = await submit_session_state(
                self._agent, lambda: capture_compact(self)
            )
            conversation = captured.conversation
            boundary = captured.boundary
            if boundary <= 1:
                logger.debug("Not enough messages to compact")
                self._emit_compact_skipped(
                    "nothing_to_compact",
                    "Nothing to compact",
                )
                terminal_sent = True
                return

            context_length = captured.context_length
            if self._plugins is not None:
                proceed = await self._plugins.should_proceed(
                    "on_compact_start",
                    context_length=context_length,
                )
                if not proceed:
                    self._emit_compact_skipped(
                        "plugin_veto",
                        "Compaction vetoed by plugin",
                    )
                    terminal_sent = True
                    logger.info(
                        "Compact vetoed by plugin(s)",
                        agent=self._agent_name,
                    )
                    await submit_internal_session_state(
                        self._agent,
                        lambda: setattr(self, "_last_compact_time", time.time()),
                    )
                    return

            summary = await self._summarize(captured.summary_input)

            if not summary:
                logger.warning(
                    "Compact summarization failed — aborting to preserve context"
                )
                if self._output_router:
                    self._output_router.notify_activity(
                        "processing_error",
                        "[CompactError] Summarization failed — context preserved unchanged",
                        metadata={
                            "error_type": "CompactError",
                            "error": self._last_summary_error
                            or "Summarization LLM call failed. Context was NOT modified.",
                        },
                    )
                self._emit_compact_skipped(
                    "summary_failed",
                    "Summarization failed — context preserved unchanged",
                )
                terminal_sent = True
                return

            reason = await submit_internal_session_state(
                self._agent,
                lambda: apply_compact(self, captured, summary, elide_after_compact),
            )
            if reason is not None:
                logger.warning(
                    "Compact source changed while summarizing",
                    agent=self._agent_name,
                    reason=reason,
                )
                self._emit_compact_skipped(
                    reason, "Compact source changed while summarizing"
                )
                terminal_sent = True
                return
            self._splice_ready.set()

            if self._output_router:
                run = getattr(self._session_store, "run", None)
                if callable(run):
                    # Scan on the affinity thread, ordered after queued writes.
                    events = await run(
                        read_agent_events, self._session_store, self._agent_name
                    )
                else:
                    events = read_agent_events(self._session_store, self._agent_name)
                metadata = build_compact_metadata(
                    captured.branch,
                    events,
                    captured.live_turn_count,
                    compact_round=self._compact_count,
                    summary=summary,
                    messages_compacted=boundary - 1,
                )
                self._output_router.notify_activity(
                    "compact_complete",
                    f"Context auto-compact done (round {self._compact_count})",
                    metadata=metadata,
                )
                if callable(run):
                    await run(lambda: None)
            terminal_sent = True

            if self._session_store:
                await persist_compacted_async(
                    self._session_store,
                    self._agent_name,
                    self._agent,
                    conversation,
                    self._compact_count,
                    self._last_compact_time,
                )

            logger.info(
                "Auto-compact complete",
                agent=self._agent_name,
                messages_compacted=boundary - 1,
                compact_round=self._compact_count,
            )

            # Fire on_compact_end callback on all plugins (observation
            # only — veto applies pre-compact only).
            if self._plugins is not None:
                await self._plugins.notify(
                    "on_compact_end",
                    summary=summary,
                    messages_removed=boundary - 1,
                )

        except asyncio.CancelledError:
            abort_reason = "cancelled"
            logger.info("Compact cancelled", agent=self._agent_name)
        except Exception as e:
            abort_reason = "error"
            logger.error("Compact failed", agent=self._agent_name, error=str(e))
        finally:
            self._splice_ready.set()
            if not terminal_sent:
                try:
                    self._emit_compact_skipped(
                        abort_reason,
                        f"Compaction {abort_reason}",
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            self._dispatch.release(lease)
            if self._active_lease is lease:
                self._active_lease = None
            self._compact_task = None

    def _compact_boundary(self, messages: list) -> int:
        usage = getattr(self._controller, "_last_usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        return select_compact_boundary(
            messages,
            self.config.keep_recent_turns,
            target_tokens=int(self.config.max_tokens * self.config.target),
            prompt_tokens=prompt_tokens,
            summary_tokens=self._summary_max_tokens(),
        )

    def _summary_max_tokens(self) -> int:
        """Return a conservative output cap for the summarization request.

        Compact runs near the model's context ceiling. Reusing the
        profile's full normal ``max_output`` budget can make the summary
        request exceed the context window even when the regular turn just
        fit. Keep the summary budget intentionally small.
        """
        max_context = max(int(self.config.max_tokens or DEFAULT_MAX_TOKENS), 1024)
        return max(512, min(4096, max_context // 64))

    async def _summarize(self, text: str) -> str:
        """Call LLM to produce a structured summary."""
        if not self._llm:
            self._last_summary_error = (
                "No LLM available for compaction. Context was NOT modified."
            )
            return ""

        prompt_messages = [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": f"Summarize this conversation:\n\n{text}"},
        ]
        summary_max_tokens = self._summary_max_tokens()

        try:
            self._last_summary_error = ""
            result = ""
            async for chunk in self._llm.chat(
                prompt_messages,
                stream=True,
                max_tokens=summary_max_tokens,
            ):
                result += chunk
            return result.strip()
        except Exception as e:
            self._last_summary_error = (
                f"Summarization LLM call failed: {e}. Context was NOT modified."
            )
            logger.error(
                "Summarization LLM call failed",
                error=str(e),
                summary_max_tokens=summary_max_tokens,
            )
            return ""

    def _splice_conversation(
        self,
        conversation: Any,
        boundary: int,
        summary: str,
        expected_last: Any = None,
        expected_fingerprint: tuple | None = None,
        retain_latest_user: bool = False,
    ) -> bool:
        """Atomic splice — see :func:`compact_splice.splice_conversation`."""
        return splice_conversation(
            conversation,
            boundary,
            summary,
            self._compact_count + 1,
            expected_last=expected_last,
            expected_fingerprint=expected_fingerprint,
            retain_latest_user=retain_latest_user,
        )

    async def wait_for_splice(self) -> None:
        """Wait until the current summary is applied or skipped, not persisted."""
        if (
            self._compact_task is not None
            and self._compact_task is not asyncio.current_task()
        ):
            await self._splice_ready.wait()

    async def wait_for_current(self) -> None:
        """Join the full round externally; turn-held callers wait for the splice."""
        current = asyncio.current_task()
        agent = self._agent
        if agent is not None and current in (
            getattr(agent, "_processing_task", None),
            getattr(agent, "_turn_lock_holder", None),
        ):
            await self.wait_for_splice()
            return
        task = self._compact_task
        if task is not None and not task.done():
            try:
                await asyncio.shield(task)
            except Exception:
                pass

    async def cancel(self) -> None:
        """Cancel any running compaction."""
        if self._compact_task and not self._compact_task.done():
            self._compact_task.cancel()
            try:
                await self._compact_task
            except (asyncio.CancelledError, Exception):
                pass
        self._splice_ready.set()
        self._dispatch.force_release()
        self._active_lease = None
        self._compact_task = None
