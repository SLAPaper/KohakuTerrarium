"""Registry of session files held open by a live ``SessionStore`` in this process.

Two SQLite library copies (Python's ``sqlite3`` and KohakuVault's) must not open
one file in the same process. Readers ask this registry first and read through
the live store when it holds the file.
"""

import os
import threading
import weakref
from pathlib import Path
from typing import Any

from kohakuterrarium.session.errors import SessionNotReadyError
from kohakuterrarium.utils.fs_path import coerce_fs_path

_lock = threading.Lock()
_open: dict[str, list["weakref.ReferenceType[Any]"]] = {}
_cold_readers: dict[str, int] = {}


def _key(path: str | Path) -> str:
    return os.path.normcase(str(coerce_fs_path(path).resolve()))


def register_open_store(path: str | Path, store: Any) -> None:
    """Record ``store`` as holding ``path`` open."""
    key = _key(path)
    with _lock:
        if _cold_readers.get(key, 0):
            raise SessionNotReadyError(f"Session has an active cold reader: {path}")
        _open.setdefault(key, []).append(weakref.ref(store))


def unregister_open_store(path: str | Path, store: Any) -> None:
    """Forget ``store``; other handles on the same path stay registered."""
    key = _key(path)
    with _lock:
        remaining = [ref for ref in _open.get(key, []) if ref() not in (None, store)]
        if remaining:
            _open[key] = remaining
        else:
            _open.pop(key, None)


def reserve_read_source(path: str | Path) -> Any | None:
    """Choose a live store or reserve a cold connection before either library opens."""
    key = _key(path)
    with _lock:
        for ref in _open.get(key, []):
            store = ref()
            if store is not None:
                return store
        _cold_readers[key] = _cold_readers.get(key, 0) + 1
    return None


def release_cold_reader(path: str | Path) -> None:
    """Release a cold-reader reservation after its SQLite handle closes."""
    key = _key(path)
    with _lock:
        remaining = _cold_readers.get(key, 0) - 1
        if remaining > 0:
            _cold_readers[key] = remaining
        else:
            _cold_readers.pop(key, None)


def live_store_for(path: str | Path) -> Any | None:
    """Return a store that holds ``path`` open in this process, or ``None``."""
    with _lock:
        for ref in _open.get(_key(path), []):
            store = ref()
            if store is not None:
                return store
    return None
