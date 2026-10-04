"""Known owner publication remains confirmed when its waiting operation is cancelled."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("operation", ["selection", "rerun", "edit"])
@pytest.mark.parametrize("cancel_target", ["caller", "operation"])
@pytest.mark.parametrize("cancel_timing", ["future_pending", "publish_confirmed"])
async def test_cancel_after_confirmed_publish_does_not_fence_consistent_history(
    tmp_path, monkeypatch, operation, cancel_target, cancel_timing
):
    provider = ScriptedLLM(["original", "second", "rerun reply", "continued"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("question", timeout=3)
        await agent.regenerate_last_response()
        owner, execute = agent._session_state_owner, agent._session_state_owner._execute
        caller, publications = None, []
        submit = owner.submit_internal

        def is_publication(callback):
            symbols = callback.__code__.co_names
            return (
                callback.__name__ == "publish"
                if operation == "selection"
                else any(name in symbols for name in ("_commit", "_commit_edit"))
            )

        def cancel_waiter():
            target = (
                caller
                if cancel_target == "caller"
                else getattr(
                    agent,
                    (
                        "_history_selection_task"
                        if operation == "selection"
                        else "_history_rerun_task"
                    ),
                )
            )
            target.cancel()

        def cancel_before_owner_settles(callback):
            future = submit(callback)
            if is_publication(callback):
                assert not future.done()
                cancel_waiter()
            return future

        def cancel_after_publish(callback, future):
            execute(callback, future)
            if (
                is_publication(callback)
                and not future.cancelled()
                and future.exception() is None
            ):
                publications.append(agent._branch_id)
                if cancel_timing == "publish_confirmed":
                    cancel_waiter()

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_execute", cancel_after_publish)
            if cancel_timing == "future_pending":
                patch.setattr(owner, "submit_internal", cancel_before_owner_settles)
            match operation:
                case "selection":
                    pending = agent.select_branch({1: 1})
                case "rerun":
                    pending = agent.regenerate_last_response()
                case "edit":
                    index = next(
                        i
                        for i, message in enumerate(
                            agent.controller.conversation.get_messages()
                        )
                        if message.role == "user"
                    )
                    pending = agent.edit_and_rerun(index, "edited")
            caller = asyncio.create_task(pending)
            result = (
                await asyncio.wait_for(
                    asyncio.gather(caller, return_exceptions=True), 3
                )
            )[0]
        assert isinstance(result, asyncio.CancelledError)
        assert publications == [1 if operation == "selection" else 3]
        assert not getattr(agent, "_history_selection_uncertain", False)
        assert not getattr(agent, "_history_rerun_uncertain", False)
        if operation != "selection":
            completion = agent._branch_rerun_completion
            assert (
                await asyncio.wait_for(asyncio.shield(completion), 3)
            ).status == "ok"
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert (
                sum(
                    row.get("type") == "user_message" and row.get("branch_id") == 3
                    for row in rows
                )
                == 1
            )
            assert provider.call_count == 3
        assert (await creature.run("continue after cancellation", timeout=3)).ok


async def test_failed_worker_save_still_fences_history_until_cold_recovery(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original", "second"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("question", timeout=3)
        await agent.regenerate_last_response()
        store = agent.session_store
        original = store.save_conversation

        def partial_save(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("snapshot written but confirmation failed")

        with monkeypatch.context() as patch:
            patch.setattr(store, "save_conversation", partial_save)
            with pytest.raises(OSError, match="confirmation failed"):
                await agent.select_branch({1: 1})
        assert agent._history_selection_uncertain
        assert agent._branch_id == 2
        with pytest.raises(ConflictError, match="uncertain"):
            await agent.select_branch({1: 1})
        assert provider.call_count == 2
