"""Event reads and agent discovery for ``SessionStore``."""

from collections.abc import Iterator
from contextlib import closing
from typing import Any

from kohakuterrarium.session.store.events.key_index import EventKeyIndex
from kohakuterrarium.session.history.replay import (
    dedupe_adjacent_duplicate_events,
    normalize_resumable_events,
)
from kohakuterrarium.session.store.events.discovery import (
    discover_agents,
    discover_attached_agents,
)
from kohakuterrarium.session.store.events.values import _iter_event_values
from kohakuterrarium.session.store.events.cursor import event_rows
from kohakuterrarium.session.vault.io import key_text, read_prefix, scan_keys


class EventReadMixin:
    """Read persisted events back by agent, id cursor, or resume normalization."""

    # Owned by the events append part.
    _event_keys: EventKeyIndex
    _event_seq: dict[str, int]
    global_max_event_id: int
    events: Any

    def get_events(self, agent: str, since_event_id: int | None = None) -> list[dict]:
        """Get events for an agent, ordered by sequence.

        Returns list of event dicts with keys sorted chronologically. With
        ``since_event_id``, only events whose ``event_id`` exceeds it are
        returned (cursor-style incremental reads).
        """
        return list(self._iter_events(agent, since_event_id))

    def _iter_events(
        self,
        agent: str,
        since_event_id: int | None = None,
        *,
        shared_paths: bool = False,
        before_sequence: int | None = None,
        legacy_keys: frozenset[str] | None = None,
    ) -> Iterator[dict]:
        """Iterate values in physical-key order inside the owning store operation."""
        self._flush_events_cache()
        prefix = f"{agent}:e"
        keys = []
        for key in scan_keys(self.events, prefix):
            text = key_text(key)
            owner, suffix = text.rsplit(":e", 1)
            if owner != agent:
                continue
            if suffix.isascii() and suffix.isdigit():
                if before_sequence is not None and int(suffix) >= before_sequence:
                    continue
            elif legacy_keys is not None and text not in legacy_keys:
                continue
            keys.append(key)
        return _iter_event_values(
            self.events, keys, since_event_id, shared_paths=shared_paths
        )

    def event_sequence_boundary(self, agent: str, *, at_least: int = 0) -> int:
        """Capture the append bound, ratcheting past any observed legacy rows."""
        sequence = self._event_seq.get(agent, 0)
        if at_least > sequence:
            self._event_seq[agent] = sequence = at_least
        return sequence

    def capture_event_read_boundary(self, agent: str) -> tuple[int, frozenset[str]]:
        """Freeze legacy keys when native recovery metadata is unavailable."""
        self._flush_events_cache()
        sequence = self.event_sequence_boundary(agent)
        legacy = set()
        for key in scan_keys(self.events, f"{agent}:e"):
            text = key_text(key)
            owner, suffix = text.rsplit(":e", 1)
            if owner != agent:
                continue
            if suffix.isascii() and suffix.isdigit():
                sequence = max(sequence, int(suffix) + 1)
            else:
                legacy.add(text)
        return self.event_sequence_boundary(agent, at_least=sequence), frozenset(legacy)

    def get_last_turn_event(self, agent: str) -> dict | None:
        """Seek backward in bounded pages to the latest turn-bearing event."""
        self._flush_events_cache()
        with closing(event_rows(self, agent, reverse=True)) as rows:
            for _, event in rows:
                turn = event.get("turn_index") if isinstance(event, dict) else None
                if isinstance(turn, int) and turn > 0:
                    return event
        return None

    def max_event_id(self, agent: str) -> int:
        """Return the session-global max ``event_id`` (monotonic, per-session).

        Local events take ids from ``_global_event_id`` and mirrored events
        ratchet it upward, so the counter is the authoritative maximum and
        this is O(1). ``event_id`` is a session-global (not per-agent) cursor,
        so ``agent`` is accepted for API symmetry with ``get_events`` and
        intentionally unused.
        """
        del agent
        return self.global_max_event_id

    def get_event_by_id(self, agent: str, event_id: int) -> dict | None:
        """Fetch a current event using a lazily built, per-agent key index.

        The first lookup scans historical values; subsequent lookups read only
        the matching payload. Appends update locators, including mirrored ids.
        """
        self._flush_events_cache()
        return self._event_keys.get(self.events, agent, event_id)

    def get_resumable_events(
        self,
        agent: str,
        *,
        live_job_ids: set[str] | None = None,
        since_event_id: int | None = None,
    ) -> list[dict]:
        """Get agent events normalized for resume/history replay.

        ``live_job_ids`` (when provided) lists currently-in-flight job
        ids; those jobs are NOT synthesized as interrupted. Callers
        serving a live session's history should pass the running jobs;
        resume-time callers (post-restart, every unfinished job IS dead)
        leave it ``None``.
        """
        events = dedupe_adjacent_duplicate_events(
            self.get_events(agent, since_event_id=since_event_id)
        )
        return normalize_resumable_events(events, live_job_ids=live_job_ids)

    def get_all_events(self) -> list[tuple[str, dict]]:
        """Get ALL events across all agents, sorted by timestamp.

        Returns list of (key, event_dict) tuples.
        """
        self.events.flush_cache()
        all_events = [
            (key_text(key), evt)
            for key, evt in read_prefix(self.events, what="event in get_all_events")
        ]
        all_events.sort(key=lambda x: x[1].get("ts", 0))
        return all_events

    def discover_agents_from_events(self) -> list[str]:
        """Scan event keys, returning agent names in first-seen order.

        Framework namespaces and attached-agent namespaces are excluded so
        resume does not rebuild them as standalone creatures.
        """
        return discover_agents(self._event_key_names())

    def discover_attached_agents(self) -> list[dict[str, Any]]:
        """Return attached-agent namespaces from events in first-seen order."""
        return discover_attached_agents(self._event_key_names())

    def _event_key_names(self) -> list[str]:
        """Event keys as text; buffered keys are flushed so they are visible."""
        self._flush_events_cache()
        return [key_text(key) for key in scan_keys(self.events)]
