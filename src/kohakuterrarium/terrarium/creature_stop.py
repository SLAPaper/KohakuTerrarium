"""Settle host input and restoration cleanup after accepted agent shutdown."""

import asyncio

from kohakuterrarium.core.branch_selection import _join_accepted


async def _cleanup(creature, restoration):
    await creature._reap_input_task()
    if restoration is not None:
        await asyncio.gather(restoration, return_exceptions=True)


async def settle_host_stop(creature, restoration):
    """Join host cleanup even when the stop caller was cancelled."""
    task = asyncio.create_task(_cleanup(creature, restoration))
    await _join_accepted(task)
