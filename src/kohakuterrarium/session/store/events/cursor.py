"""Native key scans keep live archives inside KohakuVault's SQLite library."""

from kohakuterrarium.session.vault.io import key_text, read_values


def event_keys(store, namespace, *, after="", reverse=False):
    """Stream exact-namespace keys; reverse scans retain keys, never bodies."""
    keys = (
        key
        for key in store.events.keys(prefix=f"{namespace}:e", batch_size=64)
        if (text := key_text(key)) > after and text.rsplit(":e", 1)[0] == namespace
    )
    yield from reversed(list(keys)) if reverse else keys


def event_rows(store, namespace, *, after="", reverse=False, strict=False):
    """Read one event at a time, including a potentially oversized value."""
    for key in event_keys(store, namespace, after=after, reverse=reverse):
        yield from read_values(store.events, [key], what="event cursor", strict=strict)
