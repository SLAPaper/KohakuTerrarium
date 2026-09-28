"""Confirm native declarations on the source store before execution dispatch."""

import asyncio
from copy import deepcopy
from uuid import uuid4

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.controller_generation import _canonical_output
from kohakuterrarium.llm.generation import provider_identity
from kohakuterrarium.session.event_origin import store_uuid
from kohakuterrarium.session.output_handoff import output_binding


def _persist(store, namespace, payload, turn, branch, path):
    uid = store_uuid(store)
    _, event_id = store.append_event(
        namespace,
        "assistant_tool_calls",
        deepcopy(payload),
        turn_index=turn,
        branch_id=branch,
        parent_branch_path=path,
    )
    store.flush()
    return [uid, namespace, event_id]


async def append_native_declaration(controller, content, kwargs):
    """Persist a detached declaration and commit its matching live announcement."""
    async with output_binding(_canonical_output(controller)):
        await _append_native_declaration(controller, content, kwargs)


async def _append_native_declaration(controller, content, kwargs):
    host = getattr(controller, "_state_host", None)

    def capture():
        sink = _canonical_output(controller)
        generation = deepcopy(kwargs.get("metadata", {}).get("generation"))
        if sink is None or not generation:
            return None
        if not getattr(host, "_runtime_scope_id", None):
            host._runtime_scope_id = str(uuid4())
        return (
            sink,
            controller.conversation,
            provider_identity(controller.llm),
            deepcopy(generation),
            getattr(host, "_runtime_scope_id", None),
            deepcopy(sink._generation_coordinates),
            controller.session_store,
            host._turn_completion,
            sink._event_key_prefix,
            deepcopy(getattr(host, "_branch_view", None)),
            deepcopy((host._turn_index, host._branch_id, host._parent_branch_path)),
        )

    captured = await submit_internal_session_state(host, capture)
    if captured is None:
        await submit_internal_session_state(
            host, lambda: controller.conversation.append("assistant", content, **kwargs)
        )
        return
    (
        sink,
        conversation,
        provider,
        generation,
        scope,
        coordinates,
        store,
        turn_scope,
        namespace,
        view,
        branch,
    ) = captured
    if (
        not scope
        or coordinates is None
        or len(coordinates) != 3
        or any(type(value) is not int or value <= 0 for value in coordinates[:2])
        or not isinstance(coordinates[2], list)
    ):
        raise ConflictError(
            "Native declaration requires captured generation coordinates"
        )
    payload = {
        "content": "",
        "tool_calls": deepcopy(kwargs["tool_calls"]),
        "generation_id": generation["generation_id"],
        "veto_group_id": f"{scope}:{generation['generation_id']}",
    }
    sink._recover_open_text()
    sink._flush_text_segment()
    receipt = sink._submit_store_write(
        _persist, store, namespace, payload, *coordinates
    )
    origin = await asyncio.wrap_future(receipt)

    def install():
        if (
            host.controller is not controller
            or controller.conversation is not conversation
            or host.session_store is not store
            or controller.session_store is not store
            or _canonical_output(controller) is not sink
            or provider_identity(controller.llm) != provider
            or controller._active_generation != generation
            or host._runtime_scope_id != scope
            or host._turn_completion is not turn_scope
            or (turn_scope is not None and turn_scope.done())
            or sink._event_key_prefix != namespace
            or host.config.name != namespace
            or getattr(host, "_branch_view", None) != view
            or (host._turn_index, host._branch_id, host._parent_branch_path) != branch
        ):
            raise ConflictError("Native declaration source changed before announcement")
        metadata = deepcopy(kwargs.get("metadata", {}))
        metadata["native_declaration_origin"] = origin
        conversation.append("assistant", content, **{**kwargs, "metadata": metadata})

    await submit_internal_session_state(host, install)


async def commit_native_declaration(controller, content, kwargs):
    await _join_accepted(
        asyncio.create_task(append_native_declaration(controller, content, kwargs))
    )
