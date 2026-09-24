"""Test-only composition probes, not a production Agent or persistence protocol."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.session.store import SessionStore

_WAIT = 5


class _Probe:
    def __init__(self, store, owner):
        self.store = store
        self.owner = owner
        self.published = []
        self.threads = {}
        self.submitted = asyncio.Event()
        self.cancel_deliveries = asyncio.Queue()

    async def append(self, namespace, payload):
        def capture():
            self.threads["capture"] = threading.get_ident()
            return deepcopy(payload)

        captured = await self.owner.submit(capture)

        def persist():
            self.threads["persist"] = threading.get_ident()
            return self.store.append_event(namespace, "probe", deepcopy(captured))

        accepted = self.store.submit(persist)
        self.submitted.set()
        key, event_id = await asyncio.shield(asyncio.wrap_future(accepted))

        def publish():
            self.threads["publish"] = threading.get_ident()
            self.published.append((event_id, captured))
            return key

        return await self.owner.submit_internal(publish)

    async def join(self, work):
        cancelled = False
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                cancelled = True
                self.cancel_deliveries.put_nowait(None)
            except Exception:
                break
        if cancelled:
            try:
                work.result()
            except BaseException as exc:
                raise asyncio.CancelledError from exc
            raise asyncio.CancelledError
        return work.result()


def _gate(store):
    entered, release = threading.Event(), threading.Event()
    append = store.append_event

    def gated(*args, **kwargs):
        entered.set()
        if not release.wait(_WAIT):
            raise TimeoutError("unreleased write barrier")
        return append(*args, **kwargs)

    store.append_event = gated
    return entered, release


async def test_detached_capture_and_owner_responsive_during_store_write(tmp_path):
    store = SessionStore(tmp_path / "capture.kohakutr")
    owner = SessionStateOwner(name="capture")
    owner.start()
    probe = _Probe(store, owner)
    entered, release = _gate(store)
    work = None
    try:
        payload = {"content": {"nested": ["original"]}}
        work = asyncio.create_task(probe.append("left", payload))
        assert await asyncio.to_thread(entered.wait, _WAIT)
        payload["content"]["nested"].append("later mutation")
        assert (
            await asyncio.wait_for(owner.submit(lambda: "responsive"), _WAIT)
            == "responsive"
        )
        assert not work.done() and not probe.published
        release.set()
        await asyncio.wait_for(work, _WAIT)
        records = await store.run(store.get_events, "left")
        assert records[0]["content"] == {"nested": ["original"]}
        assert probe.published == [
            (records[0]["event_id"], {"content": {"nested": ["original"]}})
        ]
        assert (
            probe.threads["capture"]
            == probe.threads["publish"]
            == threading.get_ident()
        )
        assert probe.threads["persist"] == store._affinity_worker_ident
        assert probe.threads["persist"] != threading.get_ident()
    finally:
        release.set()
        if work is not None:
            await asyncio.gather(work, return_exceptions=True)
        await owner.close()
        await asyncio.to_thread(store.close)


async def test_negative_control_cancelled_wait_leaves_write_without_publish(tmp_path):
    store = SessionStore(tmp_path / "naive.kohakutr")
    owner = SessionStateOwner(name="naive")
    owner.start()
    probe = _Probe(store, owner)
    entered, release = _gate(store)
    work = None
    try:
        work = asyncio.create_task(probe.append("left", {"content": "accepted"}))
        assert await asyncio.to_thread(entered.wait, _WAIT)
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        release.set()
        records = await store.run(store.get_events, "left")
        assert [row["content"] for row in records] == ["accepted"]
        assert probe.published == []
    finally:
        release.set()
        if work is not None:
            await asyncio.gather(work, return_exceptions=True)
        await owner.close()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("cancels", [1, 3])
async def test_join_finishes_accepted_write_then_reraises_delivered_cancellation(
    tmp_path, cancels
):
    store = SessionStore(tmp_path / "join.kohakutr")
    owner = SessionStateOwner(name="join")
    owner.start()
    probe = _Probe(store, owner)
    entered, release = _gate(store)
    work = waiter = None
    try:
        work = asyncio.create_task(probe.append("left", {"content": "accepted"}))
        assert await asyncio.to_thread(entered.wait, _WAIT)
        waiter = asyncio.create_task(probe.join(work))
        # This callback runs after the already-scheduled waiter has entered join.
        for _ in range(cancels):
            asyncio.get_running_loop().call_soon(waiter.cancel)
            await asyncio.wait_for(probe.cancel_deliveries.get(), _WAIT)
            assert not waiter.done() and not work.done()
        owner.close_admission()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, _WAIT)
        records = await store.run(store.get_events, "left")
        assert len(records) == len(probe.published) == 1
        assert records[0]["content"] == "accepted"
        assert probe.published[0][0] == records[0]["event_id"]
        assert work.done() and not work.cancelled()
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (work, waiter) if task is not None),
            return_exceptions=True,
        )
        await owner.close()
        await asyncio.to_thread(store.close)


async def test_cancel_before_task_starts_accepts_no_owner_or_store_work(tmp_path):
    store = SessionStore(tmp_path / "before.kohakutr")
    owner = SessionStateOwner(name="before")
    owner.start()
    probe = _Probe(store, owner)
    try:
        work = asyncio.create_task(probe.append("left", {"content": "not accepted"}))
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert await store.run(store.get_events, "left") == []
        assert probe.threads == {} and probe.published == []
    finally:
        await owner.close()
        await asyncio.to_thread(store.close)


async def test_cancelled_join_retains_write_failure_as_cause(tmp_path):
    store = SessionStore(tmp_path / "cancel-failure.kohakutr")
    owner = SessionStateOwner(name="cancel-failure")
    owner.start()
    probe = _Probe(store, owner)
    entered, release = threading.Event(), threading.Event()
    work = waiter = None
    try:

        def fail(*args, **kwargs):
            entered.set()
            if not release.wait(_WAIT):
                raise TimeoutError("unreleased failure barrier")
            raise OSError("accepted write failed")

        store.append_event = fail
        work = asyncio.create_task(probe.append("left", {"content": "failure"}))
        assert await asyncio.to_thread(entered.wait, _WAIT)
        waiter = asyncio.create_task(probe.join(work))
        asyncio.get_running_loop().call_soon(waiter.cancel)
        await asyncio.wait_for(probe.cancel_deliveries.get(), _WAIT)
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(waiter, _WAIT)
        assert isinstance(caught.value.__cause__, OSError)
        assert str(caught.value.__cause__) == "accepted write failed"
        assert probe.published == []
        assert await store.run(store.get_events, "left") == []
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (work, waiter) if task is not None),
            return_exceptions=True,
        )
        await owner.close()
        await asyncio.to_thread(store.close)


async def test_failed_write_has_no_owner_publication(tmp_path):
    store = SessionStore(tmp_path / "failure.kohakutr")
    owner = SessionStateOwner(name="failure")
    owner.start()
    probe = _Probe(store, owner)
    try:

        def fail(*args, **kwargs):
            raise OSError("critical write failed")

        store.append_event = fail
        with pytest.raises(OSError, match="critical write"):
            await probe.append("left", {"content": "must not publish"})
        assert probe.published == []
        assert await store.run(store.get_events, "left") == []
    finally:
        await owner.close()
        await asyncio.to_thread(store.close)


async def test_two_owners_keep_namespaces_while_store_serializes_submissions(tmp_path):
    store = SessionStore(tmp_path / "shared.kohakutr")
    owners = [SessionStateOwner(name=name) for name in ("left", "right")]
    for owner in owners:
        owner.start()
    probes = [_Probe(store, owner) for owner in owners]
    entered, release = _gate(store)
    tasks = []
    try:
        for name, probe in zip(("left", "right"), probes):
            tasks.append(asyncio.create_task(probe.append(name, {"content": name})))
            await asyncio.wait_for(probe.submitted.wait(), _WAIT)
        assert await asyncio.to_thread(entered.wait, _WAIT)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), _WAIT)
        rows = [await store.run(store.get_events, name) for name in ("left", "right")]
        assert [row[0]["content"] for row in rows] == ["left", "right"]
        assert rows[0][0]["event_id"] < rows[1][0]["event_id"]
        assert probes[0].threads["persist"] == probes[1].threads["persist"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        for owner in owners:
            await owner.close()
        await asyncio.to_thread(store.close)
