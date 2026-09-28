"""Preserve live delivery reservations across repository handoffs."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from kohakuterrarium.terrarium.drive.errors import DriveConflictError


@dataclass(eq=False)
class DeliveryReceipt:
    """One physical admission and its confirmed repository destination."""

    delivery: Any
    dispatcher: Any
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    destination: Any = None
    failure: Exception | None = None
    admission: Any = None
    settlement: Any = None
    completed: bool = False

    def __post_init__(self):
        self.ready.set()


class DeliveryHandoffBook:
    """Own admission reservations and finite repository-application receipts."""

    def __init__(self):
        self._receipts: dict[str, DeliveryReceipt] = {}
        self._failed: dict[str, DeliveryReceipt] = {}
        self._paused: set[Any] = set()
        self._phases: dict[asyncio.Future, Any] = {}

    def register(self, dispatcher, delivery):
        if dispatcher._repo in self._paused:
            raise DriveConflictError("Drive repository handoff is pending")
        if delivery.delivery_id in self._receipts:
            raise DriveConflictError("Drive delivery already has a live admission")
        receipt = DeliveryReceipt(delivery, dispatcher)
        self._receipts[delivery.delivery_id] = receipt
        return receipt

    def requires_recovery(self, repo):
        return any(
            receipt.dispatcher._repo is repo
            or (receipt.destination is not None and receipt.destination._repo is repo)
            for receipt in self._failed.values()
        )

    def reserved(self, repo, *, drive_id=None, assignee=None):
        return any(
            (receipt.destination or receipt.dispatcher)._repo is repo
            and (drive_id is None or receipt.delivery.drive_id == drive_id)
            and (assignee is None or receipt.delivery.assignee_creature_id == assignee)
            for receipt in self._receipts.values()
        )

    @asynccontextmanager
    async def phase(self, receipt):
        await receipt.ready.wait()
        if receipt.failure is not None:
            raise receipt.failure
        dispatcher = receipt.dispatcher
        if dispatcher._repo in self._paused:
            raise DriveConflictError("Drive delivery destination is not confirmed")
        completed = asyncio.get_running_loop().create_future()
        self._phases[completed] = dispatcher._repo
        try:
            yield dispatcher
        finally:
            self._phases.pop(completed, None)
            completed.set_result(None)

    async def pause(self, repositories):
        self._paused.update(repositories)
        for receipt in self._receipts.values():
            if receipt.dispatcher._repo in repositories:
                receipt.ready.clear()
        phases = [f for f, repo in self._phases.items() if repo in repositories]
        if phases:
            await asyncio.gather(*(asyncio.shield(f) for f in phases))

    def prepare(self, repositories, destinations):
        """Validate exact delivery identities before publishing successor routes."""
        routes = {}
        for dispatcher, deliveries in destinations:
            for delivery in deliveries:
                routes.setdefault(delivery.delivery_id, []).append(
                    (dispatcher, delivery)
                )
        prepared = []
        for receipt in self._receipts.values():
            if receipt.dispatcher._repo not in repositories:
                continue
            matches = routes.get(receipt.delivery.delivery_id, [])
            if len(matches) != 1:
                raise DriveConflictError("Live Drive delivery has no unique successor")
            dispatcher, delivery = matches[0]
            if delivery.logical_key() != receipt.delivery.logical_key():
                raise DriveConflictError(
                    "Live Drive delivery successor identity changed"
                )
            if delivery.assignee_creature_id != receipt.delivery.assignee_creature_id:
                raise DriveConflictError(
                    "Live Drive delivery successor assignee changed"
                )
            prepared.append((receipt, dispatcher))
        for receipt, dispatcher in prepared:
            receipt.destination = dispatcher
        return prepared

    def publish(self, repositories, prepared):
        for receipt, dispatcher in prepared:
            receipt.dispatcher = dispatcher
            receipt.destination = None
        self._paused.difference_update(repositories)
        for receipt, _ in prepared:
            receipt.ready.set()

    def complete(self, receipt):
        receipt.completed = True
        if self._receipts.get(receipt.delivery.delivery_id) is receipt:
            del self._receipts[receipt.delivery.delivery_id]

    def fail(self, receipt, error):
        if receipt.completed:
            return
        receipt.failure = receipt.failure or DriveConflictError(
            f"Drive delivery is unconfirmed; cold recovery required: {error}"
        )
        self._paused.add(receipt.dispatcher._repo)
        self._receipts.pop(receipt.delivery.delivery_id, None)
        self._failed[receipt.delivery.delivery_id] = receipt
        receipt.ready.set()

    def abort(self, repositories=None):
        for receipt in list(self._receipts.values()):
            if receipt.ready.is_set() or (
                repositories is not None
                and receipt.dispatcher._repo not in repositories
            ):
                continue
            self.fail(receipt, "repository handoff was not confirmed")
