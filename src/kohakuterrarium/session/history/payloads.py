"""Build temporary event descriptors with bounded, borrowed payload reads."""

from collections.abc import Mapping
from sys import intern

from kohakuterrarium.session.history.ancestry import BranchAncestry
from kohakuterrarium.session.history.signature import (
    SignatureValues,
    event_signature,
    _event_signature_value,
)
from kohakuterrarium.session.store.events.cursor import event_keys
from kohakuterrarium.session.vault.io import key_text, read_values


def event_pages(store, namespace, before_sequence=None, legacy_keys=None):
    """Read one payload per page without mixing SQLite libraries in the owner."""
    for key in event_keys(store, namespace):
        text = key_text(key)
        suffix = text.rsplit(":e", 1)[1]
        if suffix.isascii() and suffix.isdigit():
            if before_sequence is not None and int(suffix) >= before_sequence:
                continue
        elif legacy_keys is not None and text not in legacy_keys:
            continue
        yield [key]


class EventPayloads:
    """Borrow at most one decoded event for a temporary restore read."""

    def __init__(self, table):
        self.table = table
        self._values = {}

    def read(self, key):
        if key not in self._values:
            self.release()
            self._values = dict(
                read_values(self.table, [key], what="restore payload", strict=True)
            )
        return self._values[key]

    def release(self):
        self._values = {}


class EventDescriptor(Mapping):
    """Borrow an immutable event, retaining small fields and payload locators."""

    __slots__ = ("_reader", "_key", "_header", "_fields", "_signature_hash")

    def __init__(self, reader, key, header, fields, _signature_hash):
        self._reader = reader
        self._key = key
        self._header = header
        self._fields = fields
        self._signature_hash = _signature_hash

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, key):
        if key in self._header:
            return self._header[key]
        if key not in self._fields:
            raise KeyError(key)
        return self._reader.read(self._key)[key]


def read_descriptors(store, namespace, before_sequence=None, legacy_keys=None):
    """Decode pages, release bodies, and retain a shared ancestry descriptor list."""
    store._flush_events_cache()
    reader, tree, events = EventPayloads(store.events), BranchAncestry(), []
    encode = SignatureValues(_event_signature_value)
    for keys in event_pages(store, namespace, before_sequence, legacy_keys):
        for key, event in read_values(store.events, keys, what="restore event"):
            if not isinstance(event, dict):
                continue
            header = {
                intern(field) if isinstance(field, str) else field: value
                for field, value in event.items()
                if value is None
                or type(value) in (bool, int, float)
                or isinstance(value, str)
                and len(value) <= 256
            }
            path = event.get("parent_branch_path")
            if isinstance(path, (list, tuple)):
                try:
                    header["parent_branch_path"] = tree.view(tree.intern(path))
                    event["parent_branch_path"] = header["parent_branch_path"]
                except ValueError:
                    pass
            if isinstance(header.get("type"), str):
                header["type"] = intern(header["type"])
            try:
                fingerprint = hash(event_signature(event, encode))
            except (TypeError, ValueError):
                fingerprint = None
            fields = tuple(
                intern(field) if isinstance(field, str) else field for field in event
            )
            events.append(EventDescriptor(reader, key, header, fields, fingerprint))
        event = path = None
    return events
