"""Own one fixed-boundary history replay for a resumed output."""

from dataclasses import dataclass
from typing import Any

from kohakuterrarium.modules.output.base import wants_resume_history
from kohakuterrarium.session.history.read import (
    read_restore_events,
    materialize_event_paths,
)
from kohakuterrarium.session.history.replay import (
    dedupe_adjacent_duplicate_events,
    normalize_resumable_events,
    prepare_replay_events,
)


@dataclass
class ResumeHistory:
    """A frozen append boundary for a deferred history consumer."""

    store: Any
    namespace: str
    watermark: int
    claimed: bool = False
    before_sequence: int | None = None
    legacy_keys: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.before_sequence is None:
            self.before_sequence, self.legacy_keys = (
                self.store.capture_event_read_boundary(self.namespace)
            )

    def _read(self):
        events = read_restore_events(
            self.store,
            self.namespace,
            before_sequence=self.before_sequence,
            legacy_keys=self.legacy_keys,
        )
        return [
            event
            for event in events
            if not isinstance(event.get("event_id"), int)
            or event["event_id"] <= self.watermark
        ]

    def read(self, shared_paths: bool = False) -> list[dict]:
        """Materialize a detached legacy callback's complete history list."""
        normalized = normalize_resumable_events(
            dedupe_adjacent_duplicate_events(self._read())
        )
        return normalized if shared_paths else materialize_event_paths(normalized)

    def _projection(self, render, normalize: bool) -> str:
        events = dedupe_adjacent_duplicate_events(self._read())
        if normalize:
            events = prepare_replay_events(events, resumable=True)
        return render(events, deduplicated=True)

    async def projection(self, render, *, normalize: bool = True) -> str:
        """Prepare bounded display text on the store worker."""
        return await self.store.run(self._projection, render, normalize)

    async def events(self, shared_paths: bool = False) -> list[dict]:
        """Prepare legacy output events on the owning store worker."""
        return await self.store.run(self.read, shared_paths)


def stage_output_history(
    agent: Any,
    store: Any,
    namespace: str,
    events: list[dict],
):
    """Stage a lightweight append boundary without retaining the restore plan."""
    if not events:
        agent._resume_history = None
        return
    agent._resume_history = ResumeHistory(
        store,
        namespace,
        max(
            (e["event_id"] for e in events if isinstance(e.get("event_id"), int)),
            default=0,
        ),
    )


async def take_output_history(
    agent: Any, *, shared_paths: bool = False
) -> list[dict] | None:
    """Consume one manual terminal replay, retaining failed reads for retry."""
    source = getattr(agent, "_resume_history", None)
    if source is None or source.claimed:
        return None
    source.claimed = True
    try:
        events = await source.events(shared_paths=shared_paths)
        if getattr(agent, "_resume_history", None) is source:
            agent._resume_history = None
        return events
    finally:
        source.claimed = False


async def replay_output_history(agent: Any) -> None:
    """Deliver history to the current consumer without concurrent claims."""
    source = getattr(agent, "_resume_history", None)
    if source is None or source.claimed:
        return
    if not wants_resume_history(agent.output_router.default_output):
        return
    source.claimed = True
    try:
        if (
            await agent.output_router.replay_history(source)
            and getattr(agent, "_resume_history", None) is source
        ):
            agent._resume_history = None
    finally:
        source.claimed = False
