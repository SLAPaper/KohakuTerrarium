"""Canonical generation-origin persistence for SessionOutput."""

import asyncio
from copy import deepcopy

from kohakuterrarium.errors import SessionError
from kohakuterrarium.llm.generation import GENERATION_SOURCE_KEYS
from kohakuterrarium.session.output_handoff import admit_output
from kohakuterrarium.session.text_recovery import prove_generation_origin

ASSISTANT_EVENTS = frozenset(
    {
        "text",
        "text_chunk",
        "assistant_reasoning",
        "assistant_image",
        "assistant_tool_calls",
    }
)


def stamp_generation(sink, event_type, data):
    """Detach assistant records and attach the confirmed origin."""
    copied = deepcopy(data)
    if event_type in ASSISTANT_EVENTS and sink._generation_id is not None:
        copied["generation_id"] = sink._generation_id
    return copied


@admit_output
async def begin_generation(sink, payload):
    """Confirm one append-plus-flush affinity unit before publishing its origin."""
    source = payload["source"]
    captured = {
        "schema": payload["schema"],
        "generation_id": payload["generation_id"],
        "source": {key: source.get(key) for key in GENERATION_SOURCE_KEYS},
    }
    sink._flush_text_segment()
    await sink.drain()
    sink._generation_payload = None
    sink._generation_proof = None
    sink._generation_id = None
    sink._generation_coordinates = None
    store, namespace = sink._store, sink._event_key_prefix
    turn, branch = sink._current_turn_branch()
    path = deepcopy(sink._current_parent_path() or [])

    def persist():
        _, event_id = store.append_event(
            namespace,
            "assistant_generation_start",
            deepcopy(captured),
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=deepcopy(path),
        )
        store.flush()
        return prove_generation_origin(
            store, namespace, captured, turn, branch, path, event_id=event_id
        )

    future = sink._submit_store_write(persist)
    try:
        proof = await asyncio.shield(asyncio.wrap_future(future))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise SessionError(f"Cannot persist assistant_generation_start: {exc}") from exc
    sink._generation_id = captured["generation_id"]
    sink._generation_coordinates = (turn, branch, path)
    sink._generation_payload = deepcopy(captured)
    sink._generation_proof = deepcopy(proof)
