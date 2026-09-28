"""Persistent runtime state and artifacts follow session copy with deletion semantics."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.builtins.tools.scratchpad_tool import ScratchpadTool
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.scratchpad import Scratchpad
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium.session_coord import _copy_session_into
from kohakuterrarium.terrarium import session_runtime_state


def test_declared_creature_without_history_still_moves_runtime_state(tmp_path):
    source, target = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    try:

        def prepare():
            source.meta["agents"] = ["alice"]
            source.state["alice:scratchpad"] = {"preconfigured": "value"}
            source.state["alice:plugin_selection"] = "[]"

        source.submit(prepare).result()
        _copy_session_into(source, target)
        assert target.submit(target.load_scratchpad, "alice").result() == {
            "preconfigured": "value"
        }
        assert (
            target.submit(lambda: target.state["alice:plugin_selection"]).result()
            == "[]"
        )
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


async def test_hosted_scratchpad_tool_mutates_on_owner_and_persists_before_handoff(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(
                [
                    "[/scratchpad]@@action=set\n@@key=saved\n@@value=yes[scratchpad/]",
                    "done",
                ]
            ),
            tools=[ScratchpadTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        observed, notifications = [], []
        notify = agent.output_router.notify_activity

        def notification(*args, **kwargs):
            if args and args[0] == "scratchpad_write":
                notifications.append(asyncio.current_task())
            return notify(*args, **kwargs)

        monkeypatch.setattr(agent.output_router, "notify_activity", notification)
        setter = Scratchpad.set

        def record(pad, key, value):
            if pad is agent.session.scratchpad:
                observed.append(asyncio.current_task())
            return setter(pad, key, value)

        monkeypatch.setattr(Scratchpad, "set", record)
        assert (await creature.run("store it")).ok
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert observed == [agent._session_state_owner.task], [
            row.get("error") for row in rows if row["type"] == "tool_result"
        ]
        assert await agent.session_store.run(
            agent.session_store.load_scratchpad, "alice"
        ) == {"saved": "yes"}
        assert notifications and all(
            task is not agent._session_state_owner.task for task in notifications
        )


def test_runtime_values_artifacts_and_source_deletions_follow_copy(tmp_path):
    source, target = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    values = {
        "scratchpad": {"key": "value"},
        "plugin_selection": "[]",
        "triggers": [],
        "compact_count": 4,
        "last_compact_time": 18.5,
        "turn_count": 6,
        "token_usage": {"input_tokens": 23},
        "native_tool_options": {"strict": False},
        "tool_options": {},
        "plugin_options": {},
        "model_selection": "chosen",
    }
    try:

        def prepare():
            source.append_event("alice", "probe", {})
            for key, value in values.items():
                source.state[f"alice:{key}"] = value
            source.write_artifact("alice/chart.png", b"image-data")

        source.submit(prepare).result()
        target.submit(
            lambda: target.state.__setitem__("bob:scratchpad", {"other": "retained"})
        ).result()
        _copy_session_into(source, target)
        assert (
            target.submit(
                lambda: {key: target.state.get(f"alice:{key}") for key in values}
            ).result()
            == values
        )
        assert (
            target.submit(
                lambda: (target.artifacts_dir / "alice/chart.png").read_bytes()
            ).result()
            == b"image-data"
        )
        source.submit(lambda: source.state.pop("alice:scratchpad")).result()
        _copy_session_into(source, target)
        assert (
            target.submit(lambda: "alice:scratchpad" in target.state).result() is False
        )
        assert target.submit(lambda: target.state["bob:scratchpad"]).result() == {
            "other": "retained"
        }
        path = target.path
        target.close(update_status=False)
        target = SessionStore(path)
        assert (
            target.submit(lambda: target.state["alice:plugin_selection"]).result()
            == "[]"
        )
        assert (
            target.submit(
                lambda: (target.artifacts_dir / "alice/chart.png").read_bytes()
            ).result()
            == b"image-data"
        )
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_runtime_transfer_retries_after_artifact_write_failure_on_correct_workers(
    tmp_path, monkeypatch
):
    source, target = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    capture, install = (
        session_runtime_state.capture_runtime_transfer,
        session_runtime_state.install_runtime_transfer,
    )
    seen = []
    try:
        source.submit(source.append_event, "alice", "probe", {}).result()
        source.submit(source.write_artifact, "file.bin", b"bytes").result()
        source.submit(
            lambda: source.state.__setitem__("alice:scratchpad", {"x": "source"})
        ).result()

        def capturing(store, names):
            seen.append(("capture", threading.get_ident(), store))
            return capture(store, names)

        def installing(store, values):
            seen.append(("install", threading.get_ident(), store))
            return install(store, values)

        monkeypatch.setattr(
            "kohakuterrarium.terrarium.session_coord.capture_runtime_transfer",
            capturing,
        )
        monkeypatch.setattr(
            "kohakuterrarium.terrarium.session_coord.install_runtime_transfer",
            installing,
        )
        writer = type(target).write_artifact

        def fail(store, *args):
            if store is target:
                raise OSError("artifact write failed")
            return writer(store, *args)

        with monkeypatch.context() as patch:
            patch.setattr(type(target), "write_artifact", fail)
            with pytest.raises(OSError, match="artifact write failed"):
                _copy_session_into(source, target)
        _copy_session_into(source, target)
        assert len(target.submit(target.get_events, "alice").result()) == 1
        assert target.submit(lambda: target.state["alice:scratchpad"]).result() == {
            "x": "source"
        }
        assert (
            target.submit(
                lambda: (target.artifacts_dir / "file.bin").read_bytes()
            ).result()
            == b"bytes"
        )
        assert seen and all(
            ident == store._affinity_worker_ident and ident != threading.get_ident()
            for _, ident, store in seen
        )
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_artifact_conflict_does_not_overwrite_target_or_import_history(tmp_path):
    source, target = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    try:
        source.submit(source.append_event, "alice", "probe", {}).result()
        source.submit(source.write_artifact, "chart.png", b"source").result()
        target.submit(target.write_artifact, "chart.png", b"target").result()
        with pytest.raises(ValueError, match="artifact"):
            _copy_session_into(source, target)
        assert (
            target.submit(
                lambda: (target.artifacts_dir / "chart.png").read_bytes()
            ).result()
            == b"target"
        )
        assert target.submit(target.get_events, "alice").result() == []
    finally:
        source.close(update_status=False)
        target.close(update_status=False)
