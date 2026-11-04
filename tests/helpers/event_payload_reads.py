"""Observe event payload reads across fresh handles without replacing storage."""

from pathlib import Path

from kohakuvault import KVault


class EventPayloadReads(list):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []


def track_event_payload_reads(monkeypatch, path: Path) -> EventPayloadReads:
    """Record requested event keys for one archive, including native cursor reads."""
    handles, reads = [], EventPayloadReads()
    original_init = KVault.__init__
    original_many = KVault.get_many
    original_item = KVault.__getitem__
    original_get = KVault.get
    original_items = KVault.items

    def opened(vault, source, *args, **kwargs):
        original_init(vault, source, *args, **kwargs)
        if kwargs.get("table") == "events" and Path(source).resolve() == path.resolve():
            handles.append(vault)

    def record(table, keys):
        if any(table is handle for handle in handles):
            reads.batch_sizes.append(len(keys))
            reads.extend(
                key.decode() if isinstance(key, bytes) else key for key in keys
            )

    def many(table, keys, *args, **kwargs):
        record(table, keys)
        return original_many(table, keys, *args, **kwargs)

    def item(table, key):
        record(table, [key])
        return original_item(table, key)

    def get(table, key, *args, **kwargs):
        record(table, [key])
        return original_get(table, key, *args, **kwargs)

    def items(table, *args, **kwargs):
        for key, value in original_items(table, *args, **kwargs):
            record(table, [key])
            yield key, value

    monkeypatch.setattr(KVault, "__init__", opened)
    monkeypatch.setattr(KVault, "get_many", many)
    monkeypatch.setattr(KVault, "__getitem__", item)
    monkeypatch.setattr(KVault, "get", get)
    monkeypatch.setattr(KVault, "items", items)
    return reads
