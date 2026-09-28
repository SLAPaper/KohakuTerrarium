"""Admit detached tool-result normalization before choosing its artifact store."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.tool_output import normalize_tool_result
from kohakuterrarium.llm.message import ImagePart, normalize_content_parts
from kohakuterrarium.session.output_handoff import output_binding
from kohakuterrarium.modules.tool.media_policy import resolve_media_policy


async def normalize_completed_tool(host, tool, result, *, max_output, job_id):
    """Normalize completed output on its current store worker, not the tool lifetime."""
    captured = SimpleNamespace(
        output=deepcopy(result.output), metadata=deepcopy(result.metadata)
    )
    descriptor = SimpleNamespace(
        tool_name=tool.tool_name,
        media_policy=resolve_media_policy(tool, captured.metadata),
    )

    parts = normalize_content_parts(captured.output)
    needs_store = (
        descriptor.media_policy.persist
        and isinstance(parts, list)
        and any(
            isinstance(part, ImagePart) and part.url.startswith("data:image/")
            for part in parts
        )
    )

    async def accepted():
        if not needs_store:
            return await asyncio.to_thread(
                normalize_tool_result,
                descriptor,
                captured,
                max_output=max_output,
                job_id=job_id,
            )
        async with output_binding(getattr(host, "_session_output", None)) as sink:
            store = (
                sink._store
                if sink is not None
                else getattr(host, "session_store", None)
            )
            kwargs = {
                "max_output": max_output,
                "job_id": job_id,
                "artifact_store": store,
            }
            run = getattr(store, "run", None)
            if callable(run):
                return await run(normalize_tool_result, descriptor, captured, **kwargs)
            return normalize_tool_result(descriptor, captured, **kwargs)

    return await _join_accepted(asyncio.create_task(accepted()))
