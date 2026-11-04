"""Bound path-signature reuse to one event iterator."""

import json
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

from kohakuterrarium.session.history.ancestry import SharedPath


class SignatureValues:
    """Reuse encoded immutable paths without retaining unbounded prefixes."""

    def __init__(
        self,
        encode: Callable[[Any], Hashable],
        *,
        path_limit: int = 32,
        byte_limit: int = 1 << 20,
    ):
        self._encode = encode
        self._path_limit = max(0, path_limit)
        self._byte_limit = max(0, byte_limit)
        self._paths: OrderedDict[tuple, tuple[str, int]] = OrderedDict()
        self._bytes = 0

    def __call__(self, value: Any) -> Hashable:
        if type(value) is not SharedPath:
            return self._encode(value)
        key = value.tree, value.leaf
        if key in self._paths:
            self._paths.move_to_end(key)
            return self._paths[key][0]
        encoded = self._encode(value)
        if not isinstance(encoded, str) or not self._path_limit:
            return encoded
        size = len(encoded.encode("utf-8"))
        if size <= self._byte_limit:
            self._paths[key] = encoded, size
            self._bytes += size
            while len(self._paths) > self._path_limit or self._bytes > self._byte_limit:
                _, (_, dropped) = self._paths.popitem(last=False)
                self._bytes -= dropped
        return encoded


def _event_signature_value(value: Any) -> Hashable:
    if isinstance(value, SharedPath):
        # Preserve the public list representation's signature, including legacy
        # collisions with malformed string paths. Only the preceding signature
        # is retained, so this boundary does not accumulate expanded prefixes.
        return json.dumps(
            [list(pair) for pair in value], sort_keys=True, separators=(",", ":")
        )
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def event_signature(event, encode):
    """Compare replay payloads independently of identity and timestamp."""
    return tuple(
        sorted(
            (key, encode(value))
            for key, value in event.items()
            if key not in {"event_id", "ts"}
        )
    )
