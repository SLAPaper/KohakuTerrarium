"""Detached child history writes and accepted affinity receipt settlement."""

import asyncio
from copy import deepcopy


async def settle(receipt, accept=None):
    """Join an accepted worker operation before propagating caller cancellation."""
    pending = asyncio.wrap_future(receipt)
    cancelled = False
    while True:
        try:
            value = await asyncio.shield(pending)
            break
        except asyncio.CancelledError:
            if pending.cancelled():
                raise RuntimeError("Accepted child persistence was cancelled") from None
            cancelled = True
        except BaseException as exc:
            if cancelled:
                raise asyncio.CancelledError() from exc
            raise
    if accept is not None:
        accept(value)
    if cancelled:
        raise asyncio.CancelledError()
    return value


def bind_children(manager, store, parent):
    """Rebind retained children and reserve identities for newly persisted jobs."""
    manager._session_store, manager._parent_name = store, parent
    for job in manager._jobs.values():
        child = job.subagent
        child._session_store, child._parent_name = store, parent
        if child._run_index is None and child._run_receipt is None:
            child._run_receipt = store.submit(
                store.next_subagent_run, parent, child.config.name
            )


def result_metadata(result):
    return deepcopy(
        {
            "success": result.success,
            "error": result.error,
            "interrupted": result.interrupted,
            "cancelled": result.cancelled,
            "turns": result.turns,
            "duration": result.duration,
            "total_tokens": result.total_tokens,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
        }
    )


async def save_result(child, result):
    """Persist a child snapshot captured before selecting its current store worker."""
    if child._session_store is None:
        return
    if child._run_index is None:
        receipt = child._run_receipt
        if receipt is None:
            store = child._session_store
            receipt = child._run_receipt = store.submit(
                store.next_subagent_run, child._parent_name, child.config.name
            )
        await settle(receipt, lambda value: setattr(child, "_run_index", value))
    store = child._session_store
    messages = child.conversation.to_messages()
    meta = result_metadata(result)
    meta.update(
        {
            "job_id": child._job_id,
            "task": messages[1].get("content", "") if len(messages) > 1 else "",
            "tools_used": deepcopy(result.metadata.get("tools_used", [])),
            "output_preview": result.output[:500],
        }
    )
    payload = dict(
        parent=child._parent_name,
        name=child.config.name,
        run=child._run_index,
        meta=meta,
        conv_json=child.conversation.to_json(),
    )
    receipt = store.submit(store.save_subagent, **payload)
    await settle(receipt, lambda _: setattr(child, "_canonical_saved", True))


def patch_result(store, parent, job_id, values):
    """Update one canonical child result on its store's affinity worker."""
    row = store.find_subagent_run(job_id, parent=parent)
    if row is None:
        raise RuntimeError(f"Canonical child run missing for {job_id}")
    meta = dict(row.get("meta") or {})
    meta.update(values)
    store.save_subagent(
        parent=row["parent"], name=row["name"], run=row["run"], meta=meta
    )
