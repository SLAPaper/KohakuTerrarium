"""The mounted model picker uses the confirmed asynchronous live-switch protocol."""

import asyncio
import threading

import pytest
from kohakuvault import KVault
from textual.app import App
from textual.worker import WorkerCancelled

from kohakuterrarium import Terrarium
from kohakuterrarium.builtins.tui.widgets.model_picker_modal import ModelPickerModal
from kohakuterrarium.core import agent_model
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


class PickerApp(App):
    def __init__(self, agent):
        super().__init__()
        self.picker = ModelPickerModal(agent)
        self.dismissals = []

    def on_mount(self):
        self.push_screen(self.picker, self.dismissals.append)


@pytest.fixture
def picker_profiles(monkeypatch):
    monkeypatch.setattr(
        ModelPickerModal,
        "_load_entries",
        lambda self: [{"provider": "openai", "name": "gpt-5.4-nano"}],
    )
    replacement = ScriptedLLM(["new provider reply"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    return replacement


async def test_apply_closes_unaccepted_coroutine_when_worker_dispatch_fails(
    tmp_path, monkeypatch, picker_profiles
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM([]), io="headless", start=True
        )
        app = PickerApp(creature.agent)
        pending = []

        def failed_dispatch(coroutine, **options):
            pending.append(coroutine)
            raise RuntimeError("worker manager rejected dispatch")

        async with app.run_test(size=(110, 45)) as pilot:
            await pilot.pause()
            app.picker.query_one("#picker-tabs").active = "tab-openai"
            await pilot.pause()
            monkeypatch.setattr(app.picker, "run_worker", failed_dispatch)
            try:
                with pytest.raises(RuntimeError, match="rejected dispatch"):
                    app.picker._apply()
                assert pending[0].cr_frame is None
                assert not app.picker._applying and not app.dismissals
                assert creature.agent.llm is not picker_profiles
            finally:
                for coroutine in pending:
                    coroutine.close()


@pytest.mark.parametrize("started", [False, True])
async def test_actual_apply_handler_switches_live_and_startup_bindings(
    tmp_path, picker_profiles, started
):
    old = ScriptedLLM(["old provider reply"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=old, io="headless", start=started
        )
        app = PickerApp(creature.agent)
        async with app.run_test(size=(110, 45)) as pilot:
            await pilot.pause()
            app.picker.query_one("#picker-tabs").active = "tab-openai"
            await pilot.pause()
            assert app.picker._highlighted_entry()["name"] == "gpt-5.4-nano"
            app.picker._apply()
            await asyncio.wait_for(app.workers.wait_for_complete(), 3)
            await pilot.pause()
            assert app.dismissals == [True]
            assert app.picker._current_identifier == "openai/gpt-5.4-nano"
            assert creature.agent.llm is picker_profiles
            assert creature.agent.controller.llm is picker_profiles
        if not started:
            await creature.start()
        assert (
            await creature.run("after switch", timeout=3)
        ).text == "new provider reply"
        assert picker_profiles.call_count == 1 and old.call_count == 0


async def test_apply_reports_durable_failure_without_dismissing_or_changing_binding(
    tmp_path, monkeypatch, picker_profiles
):
    old = ScriptedLLM(["old provider reply"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=old, io="headless", start=True
        )
        put = KVault.put

        def fail_control(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                raise OSError("durable confirmation rejected")
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", fail_control)
        app = PickerApp(creature.agent)
        async with app.run_test(size=(110, 45)) as pilot:
            await pilot.pause()
            app.picker.query_one("#picker-tabs").active = "tab-openai"
            await pilot.pause()
            assert app.picker._highlighted_entry()["name"] == "gpt-5.4-nano"
            app.picker._apply()
            await asyncio.wait_for(app.workers.wait_for_complete(), 3)
            await pilot.pause()
            assert "durable confirmation rejected" in str(
                app.picker.query_one("#picker-status").render()
            )
            assert not app.dismissals
            assert creature.agent.llm is old and creature.agent.controller.llm is old
        assert (
            await creature.run("continue old", timeout=3)
        ).text == "old provider reply"
        assert picker_profiles.call_count == 0


async def test_picker_rejects_live_fence_without_provider_switch_or_false_dismiss(
    tmp_path, picker_profiles
):
    old = ScriptedLLM(["old provider reply"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=old, io="headless", start=True
        )
        agent = creature.agent
        await agent._session_state_owner.submit_internal(
            lambda: setattr(agent, "_history_selection_uncertain", True)
        )
        app = PickerApp(agent)
        async with app.run_test(size=(110, 45)) as pilot:
            await pilot.pause()
            app.picker.query_one("#picker-tabs").active = "tab-openai"
            await pilot.pause()
            app.picker._apply()
            await asyncio.wait_for(app.workers.wait_for_complete(), 3)
            assert "uncertain" in str(app.picker.query_one("#picker-status").render())
            assert not app.dismissals and agent.llm is old
            assert agent._history_selection_uncertain
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert not any(row["type"] == "model_switch" for row in rows)
        assert picker_profiles.call_count == 0


async def test_apply_duplicate_rejected_and_cancelled_widget_still_settles_control(
    tmp_path, monkeypatch, picker_profiles
):
    entered, release = threading.Event(), threading.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM([]), io="headless", start=True
        )
        store, controls = creature.agent.session_store, []
        put = KVault.put

        def held_control(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                controls.append("model_switch")
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test did not release durable control")
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", held_control)
        app = PickerApp(creature.agent)
        try:
            async with app.run_test(size=(110, 45)) as pilot:
                await pilot.pause()
                app.picker.query_one("#picker-tabs").active = "tab-openai"
                await pilot.pause()
                assert app.picker._highlighted_entry()["name"] == "gpt-5.4-nano"
                app.picker._apply()
                assert await asyncio.to_thread(entered.wait, 2)
                app.picker._apply()
                assert "already in progress" in str(
                    app.picker.query_one("#picker-status").render()
                )
                await pilot.press("escape")
                await pilot.pause()
                assert app.dismissals == [False]
                assert controls == ["model_switch"]
                release.set()
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(app.workers.wait_for_complete(), 3)
                await pilot.pause()
                assert app.dismissals == [False]
                assert creature.agent.llm is picker_profiles
                assert not creature.agent._model_switch_pending
                rows = await store.run(store.get_events, "alice")
                assert sum(row["type"] == "model_switch" for row in rows) == 1
        finally:
            release.set()
        assert (
            await creature.run("after dismissed switch", timeout=3)
        ).text == "new provider reply"
