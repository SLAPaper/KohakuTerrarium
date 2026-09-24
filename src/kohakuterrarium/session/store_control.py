"""Uncached commits for model-selection intent, not a multi-writer transaction."""

import threading
import time
from contextlib import contextmanager
from copy import deepcopy

from kohakuvault import KVault

from kohakuterrarium.errors import HistoryCompatibilityError, SessionError
from kohakuterrarium.session.model_switch_history import validate_switch_history
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class AmbiguousControlCommitError(SessionError):
    """A failed acknowledgement whose durable control cannot be confirmed."""


@contextmanager
def control_events(store):
    """Fresh handles have no write-back cache; retain KVault's own encoding."""
    writer = KVault(store._path, table="events")
    try:
        writer.enable_auto_pack()
        yield writer
    finally:
        try:
            writer.close()
        except Exception:
            logger.warning("Control handle close failed", exc_info=True)


def read_model_selection(store, namespace: str) -> str | None:
    """Read confirmed controls for an exact recorder namespace."""
    if not isinstance(getattr(store, "_path", None), str):
        return None
    prefix = f"{namespace}:e"
    latest = None
    with control_events(store) as events:
        for key in events.keys(prefix=prefix, limit=2**31 - 1):
            text = key.decode() if isinstance(key, bytes) else key
            if not text[len(prefix) :].isdigit():
                continue
            event = events[key]
            if not isinstance(event, dict) or event.get("type") != "model_switch":
                continue
            if (
                type(event.get("schema")) is not int
                or event["schema"] != 1
                or type(event.get("event_id")) is not int
                or event["event_id"] <= 0
                or event.get("classification") not in ("portable", "native", "private")
                or not isinstance(event.get("to_identifier"), str)
                or not event["to_identifier"].strip()
            ):
                raise SessionError("Invalid recorded model switch")
            try:
                validate_switch_history([], [], [event], {})
            except HistoryCompatibilityError as exc:
                raise SessionError("Invalid recorded model switch") from exc
            if latest is None or event["event_id"] > latest["event_id"]:
                latest = event
    return latest["to_identifier"] if latest is not None else None


def append_control_event(
    self,
    agent,
    event_type,
    data,
    *,
    turn_index=None,
    spawned_in_turn=None,
    branch_id=None,
    parent_branch_path=None,
):
    """Commit a model-selection control on the store worker, returning its identity."""
    if threading.get_ident() != self._affinity_worker_ident:
        raise RuntimeError(
            "Control commits must run on the SessionStore affinity worker"
        )
    if event_type != "model_switch":
        raise ValueError("Unsupported control event")
    if self._readonly:
        raise SessionError("Cannot commit control to a read-only store")
    key = f"{agent}:e{self._next_event_seq(agent):06d}"
    event_id = self._next_global_event_id()
    payload = deepcopy(data)
    payload.update(type=event_type, event_id=event_id)
    payload.setdefault("ts", time.time())
    if turn_index is not None:
        payload["turn_index"] = turn_index
    if spawned_in_turn is not None:
        payload["spawned_in_turn"] = spawned_in_turn
    elif turn_index is not None:
        payload.setdefault("spawned_in_turn", turn_index)
    if branch_id is not None:
        payload["branch_id"] = branch_id
    if parent_branch_path is not None:
        payload["parent_branch_path"] = [list(p) for p in parent_branch_path]
    self.state["counters:max_event_id"] = event_id
    try:
        with control_events(self) as writer:
            writer[key] = payload
    except Exception as write_error:
        try:
            with control_events(self) as reader:
                committed = reader.get(key)
        except Exception as read_error:
            raise AmbiguousControlCommitError(
                f"Ambiguous model_switch commit at {key}; cannot confirm durability"
            ) from read_error
        if committed is None:
            raise write_error
        if committed != payload:
            raise AmbiguousControlCommitError(
                f"Ambiguous model_switch payload at {key}"
            ) from write_error
    for callback in tuple(self._event_subscribers):
        try:
            callback(key, deepcopy(payload))
        except Exception:
            logger.warning("Control subscriber failed", exc_info=True)
    return key, event_id
