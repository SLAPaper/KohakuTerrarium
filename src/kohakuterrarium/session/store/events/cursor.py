"""Page physical event keys inside an owning synchronous store operation."""

import sqlite3
from contextlib import closing
from pathlib import Path

from kohakuterrarium.session.vault.io import key_text, read_values


def connect(store, *, writable: bool = False) -> sqlite3.Connection:
    """Open the native archive without creating missing files."""
    path = getattr(store, "_resume_index_path", store.path)
    mode = "rw" if writable else "ro"
    return sqlite3.connect(Path(path).resolve().as_uri() + f"?mode={mode}", uri=True)


def event_rows(
    store,
    namespace: str,
    *,
    after: str = "",
    reverse: bool = False,
    strict: bool = False,
):
    """Yield exact-namespace rows in bounded physical-key pages."""
    prefix = f"{namespace}:e".encode()
    upper = prefix[:-1] + bytes([prefix[-1] + 1])
    order = "DESC" if reverse else "ASC"
    with closing(connect(store)) as connection:
        cursor = connection.execute(
            f"SELECT key FROM events WHERE key >= ? AND key < ? AND key > ? ORDER BY key {order}",
            (prefix, upper, after.encode()),
        )
        while keys := cursor.fetchmany(64):
            batch = [
                key for (key,) in keys if key_text(key).rsplit(":e", 1)[0] == namespace
            ]
            yield from read_values(
                store.events, batch, what="event cursor", strict=strict
            )
