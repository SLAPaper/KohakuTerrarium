"""Destination-worker generation proof for a live output binding transfer."""

from copy import deepcopy

from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.text_recovery import prove_generation_origin


def _capture(sink):
    payload = getattr(sink, "_generation_payload", None)
    coordinates = getattr(sink, "_generation_coordinates", None)
    if payload is None:
        return None
    if coordinates is None:
        raise ConflictError("Generation transfer has no source coordinates")
    return deepcopy((payload, coordinates, getattr(sink, "_generation_proof", None)))


async def prepare_generation_handoff(agent, source, destination):
    """Prepare a detached target-store proof without vault access on the owner."""
    captured = await submit_internal_session_state(agent, lambda: _capture(source))
    if captured is None:
        return
    payload, (turn, branch, path), old_proof = captured
    proof = await destination._store.run(
        prove_generation_origin,
        destination._store,
        destination._event_key_prefix,
        payload,
        turn,
        branch,
        path,
    )
    if (
        old_proof is not None
        and proof["declaration_origin"] != old_proof["declaration_origin"]
    ):
        raise ConflictError("Generation transfer changed canonical declaration origin")
    destination._generation_proof = proof


def validate_generation_handoff(source, destination):
    """Validate the prepared source fields immediately before binding publication."""
    captured = _capture(source)
    if captured is None:
        destination._generation_proof = None
        return
    payload, (turn, branch, path), old_proof = captured
    proof = getattr(destination, "_generation_proof", None)
    if not isinstance(proof, dict) or any(
        proof.get(key) != value
        for key, value in {
            "schema": payload["schema"],
            "generation_id": payload["generation_id"],
            "source": payload["source"],
            "turn_index": turn,
            "branch_id": branch,
            "parent_branch_path": [list(pair) for pair in (path or [])],
            "namespace": destination._event_key_prefix,
        }.items()
    ):
        raise ConflictError("Generation transfer source changed before publication")
    if (
        old_proof is not None
        and proof["declaration_origin"] != old_proof["declaration_origin"]
    ):
        raise ConflictError("Generation transfer origin changed before publication")
