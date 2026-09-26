"""Short owner commits for background-result consumption dispositions."""

import asyncio
from copy import deepcopy

from kohakuterrarium.core.background_origin import (
    completion_identity,
    has_background_provenance,
    lineage_accepts,
)
from kohakuterrarium.core.event_inbox import TurnOutcome


def _admit(agent, events, token):
    accepted = []
    ledger = getattr(agent, "_background_consumption", None)
    if ledger is None:
        ledger = agent._background_consumption = {}
    for event in events:
        if not has_background_provenance(event):
            accepted.append(True)
            continue
        identity = completion_identity(event, getattr(agent, "_runtime_scope_id", None))
        if identity is None or not getattr(agent, "_running", False):
            accepted.append(False)
            continue
        path, delivery = identity
        if delivery is not None and delivery in ledger:
            accepted.append(False)
            continue
        admitted = lineage_accepts(agent, path)
        if delivery is not None:
            ledger[delivery] = (admitted, token)
        accepted.append(admitted)
    return accepted


def _release(agent, token):
    ledger = getattr(agent, "_background_consumption", {})
    for delivery, (_, owner_token) in list(ledger.items()):
        if owner_token is token:
            del ledger[delivery]


async def _join(future):
    cancelled = None
    while True:
        try:
            return await asyncio.shield(future), cancelled
        except asyncio.CancelledError as exc:
            if future.cancelled():
                raise
            cancelled = exc


def _reject(envelopes):
    for envelope in envelopes:
        future = envelope.future
        if future is not None and not future.done():
            future.set_result(TurnOutcome(status="rejected", was_primary=False))


async def admit_background_run(agent, envelopes):
    """Filter claimed envelopes under the turn lock before any delivery effects."""
    if not any(has_background_provenance(env.event) for env in envelopes):
        return envelopes
    owner = getattr(agent, "_session_state_owner", None)
    token = object()
    submitted = False
    try:
        events = [
            deepcopy(env.event) if has_background_provenance(env.event) else env.event
            for env in envelopes
        ]
        if owner is None:
            verdicts = [not has_background_provenance(event) for event in events]
        else:
            future = owner.submit_internal(lambda: _admit(agent, events, token))
            submitted = True
            verdicts, cancelled = await _join(future)
            if cancelled is not None:
                raise cancelled
            if not getattr(agent, "_running", False):
                _, cancelled = await _join(
                    owner.submit_internal(lambda: _release(agent, token))
                )
                if cancelled is not None:
                    raise cancelled
                verdicts = [not has_background_provenance(event) for event in events]
    except BaseException as exc:
        _reject(envelopes)
        if submitted:
            try:
                await _join(owner.submit_internal(lambda: _release(agent, token)))
            except BaseException as cleanup_error:
                raise exc from cleanup_error
        raise
    accepted = []
    for envelope, event, admitted in zip(envelopes, events, verdicts):
        if admitted:
            envelope.event = event
            accepted.append(envelope)
        else:
            _reject([envelope])
    return accepted
