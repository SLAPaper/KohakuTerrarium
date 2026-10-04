"""Single event consumer + batch-turn driver (queue-first event system).

Single-consumer event batching and serialized turn execution for agents.

Each inbox claim is segmented into maximal stackable runs, with one turn driven
per run. The processing lock allows shutdown to join an in-flight turn.
"""

import asyncio
from copy import deepcopy

from kohakuterrarium.core.agent_mid_turn import _to_serializable_content
from kohakuterrarium.core.agent_runtime_tools import _make_job_label
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.agent_turn_control import (
    SettledProcessingFailure,
    finalize_turn,
    withdraw_turn_events,
)
from kohakuterrarium.core.background_admission import admit_background_run
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.event_inbox import EventEnvelope, TurnOutcome
from kohakuterrarium.core.feedback_admission import (
    arm_background_feedback,
    arm_persisted_input,
    record_injected_constituent,
    clear_feedback,
)
from kohakuterrarium.core.metrics_hook import metrics
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.shutdown_callbacks import stop_wait_ineligible
from kohakuterrarium.core.pending_input import pending_id_of
from kohakuterrarium.core.primary_user_identity import (
    arm_primary_user,
    capture_primary_user,
    clear_primary_user,
    install_primary_user,
)
from kohakuterrarium.core.regeneration_reservation import wait_regeneration_admission
from kohakuterrarium.session.raw_history import append_user_event_pair
from kohakuterrarium.modules.output.event import OutputEvent
from kohakuterrarium.skills.hints import inject_skill_path_hint
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

# Settle window (in cooperative yields) applied after a wake before the
# claim, so producers whose event reaches the inbox across several
# create_task / done-callback hops still land in the SAME batch. A
# background TOOL completion reaches the inbox in ~1 hop
# (``_run_tool`` → ``_on_bg_complete`` → ``create_task(_process_event)``);
# a backgroundified SUB-AGENT completion in ~3 (``_on_task_done``
# done-callback → ``create_task(_on_backgroundify_complete)`` →
# ``create_task(_process_event)``). A single yield would coalesce only
# the shallower path, letting a simultaneous sub-agent completion slip to
# a second turn. This budget spans that gap with margin. It is NOT a poll:
# a FIXED, bounded number of immediate reschedules — no timed sleep and no
# drain-until-empty loop — so a steady event stream can never extend it.
_COALESCE_HOPS = 3


def _is_fresh_user_input(event) -> bool:
    """A user_input that opens a NEW user turn (not a regen/edit rerun,
    whose branch was pre-incremented by its caller)."""
    ctx = event.context or {}
    return event.type == "user_input" and not ctx.get("rerun")


def segment_stackable(batch: list[EventEnvelope]) -> list[list[EventEnvelope]]:
    """Split a claimed batch into maximal stackable runs.

    An awaited request or non-stackable event (startup / shutdown / error /
    Drive / rerun) becomes its own singleton run and sole primary. A maximal
    span of other stackable events becomes one run folded into one turn.
    Mirrors the controller's break-on-non-stackable batching.
    """
    runs: list[list[EventEnvelope]] = []
    cur: list[EventEnvelope] = []
    for env in batch:
        if (
            env.event.stackable
            and env.capture is None
            and not env.event.context.get("await_turn")
        ):
            cur.append(env)
        else:
            if cur:
                runs.append(cur)
                cur = []
            runs.append([env])
    if cur:
        runs.append(cur)
    return runs


class AgentEventLoopMixin:
    """The single consumer + batch-turn driver mixed into ``Agent``."""

    def _flush_trigger_backlog_stash(self) -> None:
        """Place a trigger's stashed backlog immediately after its primary.

        The synchronous flush admits no scheduling point, preserving adjacency
        and order in the inbox.
        """
        stash = getattr(self, "_trigger_backlog_stash", None)
        if not stash:
            return
        for event in stash:
            self._event_inbox.put(EventEnvelope(event))
        stash.clear()

    async def _run_event_consumer(self) -> None:
        """Drain claimed input into turns under pause and mutation admission gates."""
        batch: list[EventEnvelope] = []
        try:
            while self._running:
                await self._consumer_resume.wait()
                if not self._running:
                    break
                await self._event_inbox.wait_nonempty()
                if not self._running:
                    break
                # Settle window: yield a small FIXED number of times so
                # simultaneous producers whose event reaches the inbox across
                # several create_task / done-callback hops (bg tool vs
                # sub-agent completion) accumulate, THEN claim them all in one
                # drain → ONE turn. Yielding BEFORE the drain keeps the inbox
                # non-empty throughout (no spurious-empty window). Bounded —
                # never a poll (see ``_COALESCE_HOPS``).
                for _ in range(_COALESCE_HOPS):
                    await asyncio.sleep(0)
                if not self._running:
                    break
                # Paused between the wake and here — re-park (invariant: the
                # queue may hold events while paused; they drain on resume).
                if not self._consumer_resume.is_set():
                    continue
                admission = getattr(self, "_history_rerun_ready", None)
                if admission is not None:
                    await admission.wait()
                if not self._running:
                    break
                if not self._consumer_resume.is_set():
                    continue
                claim = getattr(self, "_regeneration_reservation", None)
                if claim is not None:
                    await claim.dispatch_ready.wait()
                batch = self._event_inbox.drain_all()
                for run in segment_stackable(batch):
                    await self._consumer_resume.wait()
                    if not self._running:
                        self._reject_run(run, status="rejected")
                        continue
                    try:
                        if not await wait_regeneration_admission(self, run):
                            for envelope in run:
                                self._event_inbox.put(envelope)
                            continue
                        await self._run_interruptible_turn(run)
                    except asyncio.CancelledError:
                        self._reject_run(run, status="interrupted")
                        raise
                    except Exception:
                        logger.error("Turn crashed in event consumer", exc_info=True)
                        self._reject_run(run, status="error")
        except asyncio.CancelledError:
            pass
        finally:
            self._reject_run(batch)
            self._drain_and_reject_inbox()

    async def _run_interruptible_turn(self, run: list[EventEnvelope]) -> None:
        """Keep cancellation of one turn separate from the ingress consumer."""
        self._active_turn_run = run
        self._interrupt_requested = False
        self._turn_phase = "admission"
        task = asyncio.create_task(self._run_turn_for_batch(run))
        self._active_turn_task = task
        try:
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError as cancellation:
                task.cancel()
                try:
                    await _join_accepted(task)
                except BaseException as exc:
                    raise cancellation from exc
                raise
            try:
                task.result()
            except asyncio.CancelledError:
                self._resolve_run(
                    run,
                    status="interrupted",
                    interrupted=any(
                        env.event.context.get("interrupted_by_user") for env in run
                    ),
                )
        finally:
            if getattr(self, "_active_turn_task", None) is task:
                self._active_turn_task = None
                self._active_turn_run = None
                self._turn_phase = None

    @stop_wait_ineligible
    async def _run_turn_for_batch(self, run: list[EventEnvelope]) -> None:
        """Run a claimed batch under source admission and attach turn-scoped captures."""
        run = [env for env in run if not env.withdrawn]
        if not run:
            return
        async with self._processing_lock:
            ready = getattr(self, "_topology_handoff_ready", None)
            if ready is not None:
                await ready.wait()
            self._turn_lock_holder = asyncio.current_task()
            try:
                run = await admit_background_run(
                    self, [env for env in run if not env.withdrawn]
                )
                run = [env for env in run if not env.withdrawn]
                if run:
                    await self._run_admitted_batch(run)
            finally:
                self._turn_lock_holder = None
                self._active_turn_stackable = True

    async def _run_admitted_batch(self, run: list[EventEnvelope]) -> None:
        """Drive a lineage-admitted batch while holding the processing lock."""
        events = [env.event for env in run]
        primary = events[0]
        captures = [env.capture for env in run if env.capture is not None]
        completion = asyncio.get_running_loop().create_future()
        self._turn_completion = completion
        self._active_event_run = run
        self._active_event_captures = captures
        for cap in captures:
            self.output_router.add_secondary(cap)
        interrupted = False
        cancelled = False
        try:
            self._turn_phase = "begin"
            self._active_turn_stackable = primary.stackable
            if not self._running:
                self._reject_run(run, status="rejected")
                return
            require_model_mutation_available(
                self, regeneration_completion=run[0].future
            )
            await self._begin_batch(events)
            if self._interrupt_requested:
                raise asyncio.CancelledError()
            cancelled = await self._process_batch_with_controller(
                events, self.controller
            )
            interrupted = bool(primary.context.get("interrupted_by_user")) or any(
                env.withdrawn for env in run
            )
            if interrupted:
                self._flush_trigger_backlog_stash()
        except asyncio.CancelledError:
            cancelled = True
            interrupted = bool(primary.context.get("interrupted_by_user"))
            if self._turn_phase != "finalizing":
                withdraw_turn_events(self.controller, events)
                await finalize_turn(self, primary, self.controller, [])
            if (
                asyncio.current_task() is not getattr(self, "_active_turn_task", None)
                or not self._interrupt_requested
            ):
                raise
        finally:
            try:
                await _join_accepted(
                    asyncio.create_task(
                        submit_internal_session_state(
                            self,
                            lambda: (
                                clear_primary_user(self, completion),
                                clear_feedback(self, completion),
                            ),
                        )
                    )
                )
            finally:
                for cap in captures:
                    self.output_router.remove_secondary(cap)
                self._active_event_run = None
                self._active_event_captures = None
                if self._turn_completion is completion:
                    self._turn_completion = None
                if not completion.done():
                    completion.set_result(None)
        self._resolve_run(
            run,
            status="interrupted" if interrupted or cancelled else "ok",
            interrupted=bool(primary.context.get("interrupted_by_user")),
        )

    def _resolve_run(
        self, run: list[EventEnvelope], *, status: str, interrupted: bool = False
    ) -> None:
        """Resolve awaited envelopes with their turn outcome and primary participation."""
        for idx, env in enumerate(run):
            fut = env.future
            if fut is None or fut.done():
                continue
            fut.set_result(
                TurnOutcome(
                    status=status,
                    was_primary=(idx == 0),
                    interrupted_by_user=interrupted,
                )
            )

    def _reject_run(
        self, run: list[EventEnvelope], *, status: str = "rejected"
    ) -> None:
        """Resolve a run's futures as non-primary rejection (stop / crash)."""
        for env in run:
            fut = env.future
            if fut is not None and not fut.done():
                fut.set_result(TurnOutcome(status=status, was_primary=False))

    def _drain_and_reject_inbox(self) -> None:
        """On consumer exit, reject every leftover awaiting future so no
        ``run`` / ``run_event`` caller hangs after the agent stopped."""
        for env in self._event_inbox.drain_all():
            fut = env.future
            if fut is not None and not fut.done():
                fut.set_result(TurnOutcome(status="rejected", was_primary=False))

    async def _begin_batch(self, events: list) -> None:
        """Publish primary and folded input identity, output and plugin lifecycle."""
        primary = events[0]
        is_rerun = bool(primary.context.get("rerun")) if primary.context else False
        is_edited = bool(primary.context.get("edited")) if primary.context else False
        is_pure_rerun = is_rerun and not is_edited

        # A fresh (non-rerun) user_input introduces a NEW user turn — advance
        # even when it is FOLDED behind a non-user primary (e.g. a background
        # completion claimed in the same batch), so the folded message is
        # stamped under the new (turn_index, branch_id), not the previous
        # turn's stale ids. A rerun primary keeps its pre-incremented branch.
        if not is_rerun and any(_is_fresh_user_input(e) for e in events):
            self._advance_turn_for_user_input()
        if primary.type == "user_input" and not is_rerun:
            if self.session_store is not None:
                await self._record_primary_user_input(primary)

        if primary.type == "trigger":
            await self._record_folded_user_event(primary)
        if (
            primary.type == "user_input"
            and self.output_router is not None
            and not is_pure_rerun
        ):
            content = (
                primary.get_text_content()
                if hasattr(primary, "is_multimodal") and primary.is_multimodal()
                else (primary.content or "")
            )
            await self.output_router.emit(
                OutputEvent(type="user_input", content=content)
            )

        if self.plugins is not None:
            await self.plugins.notify("on_event", event=primary)
        if primary.type == "user_input" or any(
            _is_fresh_user_input(event) for event in events
        ):
            inject_skill_path_hint(self)

        # Folded events get their own injected record + queued-banner-clear
        # frame (only user-facing kinds — background completions get the
        # combined delivery banner below instead).
        for evt in events[1:]:
            if evt.type in ("user_input", "trigger"):
                await self._record_folded_user_event(evt)

        self._emit_batch_background_banner(events)

    def _advance_turn_for_user_input(self) -> None:
        """Bump turn_index + allocate a collision-safe branch_id for a fresh
        user turn (was inline in ``_process_event``)."""
        if self._turn_index > 0 and self._branch_id > 0:
            self._parent_branch_path = list(self._parent_branch_path)
            self._parent_branch_path.append((self._turn_index, self._branch_id))
        self._turn_index += 1
        helper = getattr(self, "_max_branch_id_for_turn", None)
        existing_max = helper(self._turn_index) if helper else 0
        self._branch_id = existing_max + 1 if existing_max > 0 else 1

    async def _record_primary_user_input(self, event) -> None:
        """Append the ``user_input`` + ``user_message`` session events for a
        fresh (non-rerun) user turn."""
        receipt = await _join_accepted(
            asyncio.create_task(
                submit_internal_session_state(
                    self,
                    lambda: capture_primary_user(self, event, pending_id_of(event)),
                )
            )
        )
        event_id = await _join_accepted(
            asyncio.create_task(
                receipt.store.run(
                    append_user_event_pair,
                    receipt.store,
                    receipt.namespace,
                    deepcopy(receipt.payload),
                    receipt.turn,
                    receipt.branch,
                    [tuple(p) for p in receipt.path],
                    with_receipt=True,
                )
            )
        )

        def install():
            install_primary_user(self, receipt, event_id)
            if not receipt.eligible and isinstance(event_id, dict):
                arm_persisted_input(
                    self,
                    event,
                    {
                        "mode": "user_pair",
                        **deepcopy(event_id),
                        "payload": deepcopy(receipt.payload),
                    },
                )

        await _join_accepted(
            asyncio.create_task(submit_internal_session_state(self, install))
        )

    async def _record_folded_user_event(self, evt) -> None:
        """Record + notify one folded user_input/trigger event so the FE
        pops its queued banner and replay shows it as its own bubble."""
        content = _to_serializable_content(self._resolve_injected_content(evt))
        pending_id = pending_id_of(evt)
        payload = {
            "content": content,
            **({"pending_id": pending_id} if pending_id else {}),
        }
        if not await record_injected_constituent(self.controller, evt, payload):
            await self._record_injected_input_event(content, pending_id=pending_id)
        metadata = {
            "content": content,
            "turn_index": self._turn_index,
            "branch_id": self._branch_id,
        }
        if pending_id:
            metadata["pending_id"] = pending_id
        self.output_router.notify_activity(
            "user_input_injected",
            "",
            metadata=metadata,
        )
        await asyncio.sleep(0)

    def _emit_batch_background_banner(self, events: list) -> None:
        """Emit ONE ``background_result`` activity for every background
        completion in the batch — ``delivered: a, b, c`` instead of one
        banner per completion. Carries per-event metadata in ``jobs`` and
        keeps singular ``label`` / ``job_id`` for un-updated renderers."""
        jobs: list[dict] = []
        labels: list[str] = []
        kinds: set[str] = set()
        for evt in events:
            if evt.type not in ("tool_complete", "subagent_output"):
                continue
            job_id = getattr(evt, "job_id", "") or ""
            if not job_id:
                continue
            kind = (
                "subagent"
                if evt.type == "subagent_output" or job_id.startswith("agent_")
                else "tool"
            )
            _, label = _make_job_label(job_id)
            jobs.append({"job_id": job_id, "kind": kind, "label": label})
            labels.append(label)
            kinds.add(kind)
        if not jobs:
            return
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        noun = "result" if len(labels) == 1 else "results"
        self.output_router.notify_activity(
            "background_result",
            f"background {noun} delivered: " + ", ".join(labels),
            metadata={
                "labels": labels,
                "jobs": jobs,
                "kind": kind,
                "count": len(labels),
                "label": labels[0] if len(labels) == 1 else ", ".join(labels),
                "job_id": jobs[0]["job_id"],
                "turn_index": self._turn_index,
                "branch_id": self._branch_id,
            },
        )

    def _notify_drive_turn(self, event) -> None:
        """Mark a turn that a Drive delivery started, for every output surface."""
        ctx = event.context or {}
        projected = ctx.get("drive") if isinstance(ctx.get("drive"), dict) else {}
        drive_id = str(ctx.get("drive_id") or "")
        kind = str(ctx.get("drive_kind") or projected.get("kind") or "drive")
        reason = str(ctx.get("delivery_reason") or event.type.removeprefix("drive_"))
        objective = str(projected.get("objective") or "")[:240]
        detail = f"[drive] {kind} {drive_id} ({reason})"
        if objective:
            detail += f": {objective}"
        self.output_router.notify_activity(
            "drive_turn",
            detail,
            metadata={
                "drive_id": drive_id,
                "drive_kind": kind,
                "delivery_reason": reason,
                "delivery_id": str(ctx.get("delivery_id") or ""),
                "objective": objective,
                "event_type": event.type,
                "turn_index": self._turn_index,
                "branch_id": self._branch_id,
            },
        )

    async def _process_batch_with_controller(self, events: list, controller) -> bool:
        """Drive combined controller input and settle output before reporting its outcome."""
        primary = events[0]
        self._prepare_processing_cycle(primary, controller)
        await _join_accepted(
            asyncio.create_task(
                submit_internal_session_state(
                    self,
                    lambda: (
                        arm_primary_user(self, controller),
                        arm_background_feedback(self, events),
                    ),
                )
            )
        )
        for event in events:
            await controller.push_event(event)
        await self.output_router.emit(
            OutputEvent(
                type="processing_start",
                payload={"request_id": self._branch_request_id},
            )
        )
        if primary.type.startswith("drive_"):
            self._notify_drive_turn(primary)

        if any(env.withdrawn for env in getattr(self, "_active_event_run", None) or []):
            self._interrupt_active_turn()
        if self._interrupt_requested:
            raise asyncio.CancelledError()
        all_round_text: list[str] = []
        self._turn_phase = "controller"
        loop_task = asyncio.create_task(
            self._run_controller_loop(controller, all_round_text)
        )
        self._processing_task = loop_task
        cancelled = False
        failure = None
        try:
            await loop_task
        except asyncio.CancelledError:
            cancelled = True
            logger.info("Processing cancelled by interrupt")
            self.output_router.notify_activity(
                "interrupt", "[system] Processing interrupted"
            )
        except Exception as e:
            failure = e
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error("Processing error", error_type=error_type, error=error_msg)
            self.output_router.notify_activity(
                "processing_error",
                f"[{error_type}] {error_msg}",
                metadata={"error_type": error_type, "error": error_msg},
            )
            metrics.observe_error("controller")
        finally:
            self._processing_task = None
        shutdown = getattr(self, "_shutdown_task", None)
        if (
            self._interrupt_requested
            and (shutdown is None or shutdown.done())
            and not any(
                env.withdrawn for env in getattr(self, "_active_event_run", None) or []
            )
        ):
            primary.context["interrupted_by_user"] = True
        await finalize_turn(self, primary, controller, all_round_text)
        if failure is not None:
            raise SettledProcessingFailure(str(failure)) from failure
        return cancelled
