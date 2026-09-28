"""Detached runtime values and immutable artifacts in a session handoff."""

from copy import deepcopy
from pathlib import Path

from kohakuterrarium.session.artifacts import resolve_artifact_relpath
from kohakuterrarium.session.event_origin import import_events
from kohakuterrarium.session.snapshot_coverage import (
    capture_snapshot_transfer,
    install_snapshot_transfer,
)

_STATE_KEYS = (
    "scratchpad",
    "plugin_selection",
    "triggers",
    "compact_count",
    "last_compact_time",
    "turn_count",
    "token_usage",
    "native_tool_options",
    "tool_options",
    "plugin_options",
    "model_selection",
    "open_text",
)


def _artifact_root(store):
    path = Path(store.path)
    return path.parent / f"{path.stem}.artifacts"


def _capture_artifacts(store):
    root = _artifact_root(store)
    if not root.exists():
        return {}
    if root.is_symlink() or root.resolve() != root.parent.resolve() / root.name:
        raise ValueError("Session artifact root cannot be a link during transfer")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Session artifact links cannot escape the transfer root")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def capture_runtime_transfer(store, owned_names=None):
    """Capture raw history, snapshots, state presence, and artifact bytes on the source."""
    batch, snapshots = capture_snapshot_transfer(store, owned_names)
    state = {
        name: {
            key: deepcopy(store.state[f"{name}:{key}"])
            for key in _STATE_KEYS
            if f"{name}:{key}" in store.state
        }
        for name in snapshots
    }
    return batch, snapshots, state, _capture_artifacts(store)


def _validate_artifacts(store, files):
    root = _artifact_root(store)
    if root.is_symlink() or root.resolve() != root.parent.resolve() / root.name:
        raise ValueError("Session artifact root cannot be a link during transfer")
    for filename, data in files.items():
        path = root / resolve_artifact_relpath(filename)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("Session artifact target escapes its directory")
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise ValueError(f"Conflicting session artifact: {filename}")


def install_runtime_transfer(store, captured):
    """Validate artifact collisions, import immutable events, then replace owned state."""
    batch, snapshots, state, files = captured
    _validate_artifacts(store, files)
    receipts = import_events(store, batch)
    install_snapshot_transfer(store, batch, snapshots, receipts)
    for name, values in state.items():
        for key in _STATE_KEYS:
            target = f"{name}:{key}"
            if key in values:
                store.state[target] = deepcopy(values[key])
            else:
                store.state.pop(target, None)
    for filename, data in files.items():
        path = _artifact_root(store) / resolve_artifact_relpath(filename)
        if not path.exists():
            store.write_artifact(filename, data)
    store.flush()
    return receipts
