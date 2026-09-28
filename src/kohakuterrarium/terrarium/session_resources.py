"""Ownership of newly opened stores during failed session transitions."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from uuid import uuid4

from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)
_OPENED = ContextVar("kt_transition_stores", default=None)


def archive_unpublished_store(store):
    """Preserve a closed, newly-created failed target and its artifacts off its retry path."""
    if not store._closed:
        raise RuntimeError("Cannot archive an open session target")
    path = Path(store.path)
    archive = path.parent / ".failed-handoffs" / uuid4().hex
    archive.mkdir(parents=True, exist_ok=False)
    artifacts = path.with_name(f"{path.stem}.artifacts")
    if artifacts.exists():
        artifacts.rename(archive / artifacts.name)
    target = path.rename(archive / path.name)
    logger.warning("Unpublished handoff target archived", path=str(target))
    return target


def track_opened_store(store):
    opened = _OPENED.get()
    if opened is not None:
        opened.append(store)
    return store


def unpublished_store_factory(operation):
    """Close factory-created handles if construction fails before returning them."""

    @wraps(operation)
    def create(*args, **kwargs):
        parent = _OPENED.get()
        opened = []
        token = _OPENED.set(opened)
        try:
            return operation(*args, **kwargs)
        except BaseException:
            for store in reversed(opened):
                try:
                    store.close(update_status=False)
                except Exception:
                    logger.error(
                        "Unpublished session store close failed", exc_info=True
                    )
            raise
        finally:
            _OPENED.reset(token)
            if parent is not None:
                parent.extend(opened)

    return create


def retain_store(engine, store):
    """Keep a failed-transition handle reachable until engine shutdown."""
    retained = getattr(engine, "_retained_session_stores", None)
    if retained is None:
        retained = engine._retained_session_stores = set()
    retained.add(store)


def defer_store_close(engine, store):
    """Keep an uncertain binding open without transferring handle ownership."""
    holds = getattr(engine, "_session_store_close_holds", None)
    if holds is None:
        holds = engine._session_store_close_holds = set()
    holds.add(store)


def release_failed_store(engine, store):
    """Close unbound handles without replacing the original transition error."""
    bound = store in engine._session_stores.values() or any(
        getattr(creature.agent, "session_store", None) is store
        for creature in engine._creatures.values()
    )
    if bound or store in getattr(engine, "_session_store_close_holds", ()):
        retain_store(engine, store)
        return
    try:
        store.close(update_status=False)
    except Exception:
        retain_store(engine, store)
        logger.error("Failed transition store cleanup failed", exc_info=True)


@contextmanager
def transition_stores(engine):
    """Track handles created by one synchronous topology transition."""
    opened = []
    token = _OPENED.set(opened)
    try:
        yield
    except BaseException:
        for store in reversed(opened):
            release_failed_store(engine, store)
        raise
    finally:
        _OPENED.reset(token)


def close_retained_stores(engine):
    retained = getattr(engine, "_retained_session_stores", set())
    for store in list(retained):
        try:
            store.close(update_status=False)
        except Exception:
            logger.error("Retained transition store close failed", exc_info=True)
        else:
            retained.discard(store)
    getattr(engine, "_session_store_close_holds", set()).clear()
