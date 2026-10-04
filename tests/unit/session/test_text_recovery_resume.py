"""Hard-crash text recovery enters the first provider context and survives resuming twice."""

import os
import subprocess
import sys
from pathlib import Path

from kohakuterrarium import Terrarium
from kohakuterrarium.session.reader import SessionReader
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_confirmed_generation_partial_text_survives_two_public_resumes(tmp_path):
    helper = (
        Path(__file__).resolve().parents[2] / "helpers" / "text_recovery_crash_child.py"
    )
    result = subprocess.run(
        [sys.executable, str(helper), str(tmp_path)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 7, result.stderr
    paths = list((tmp_path / "sessions").glob("*.kohakutr"))
    assert len(paths) == 1
    path = paths[0]
    with SessionReader(path) as reader:
        slot = reader._store.state.get("crash_recovery:open_text")
        assert isinstance(slot, dict) and slot["text"] == "P" * 600
    first_provider = ScriptedLLM(["continued"])
    first = await Terrarium.resume(path, llm=first_provider)
    async with first:
        creature = first.list_creatures()[0]
        agent = creature.agent
        assert any(
            m.content == "P" * 600
            for m in agent.controller.conversation.get_messages()
            if m.role == "assistant"
        )
        assert (await creature.run("continue", timeout=5)).text == "continued"
        assert any(
            m["role"] == "assistant" and m["content"] == "P" * 600
            for m in first_provider.last_messages
        )
        store = agent.session_store
        rows = await store.run(store.get_events, "crash_recovery")
        recovered = [row for row in rows if row.get("finalize") == "recovered"]
        assert len(recovered) == 1
        assert recovered[0]["generation_id"] == slot["generation"]["generation_id"]
        assert recovered[0]["parent_branch_path"] == slot["parent_branch_path"]
        assert await store.run(store.state.get, "crash_recovery:open_text") in (
            None,
            "",
        )
    second_provider = ScriptedLLM(["after second resume"])
    second = await Terrarium.resume(path, llm=second_provider)
    async with second:
        creature = second.list_creatures()[0]
        assert (await creature.run("next", timeout=5)).text == "after second resume"
        assert (
            sum(
                m["role"] == "assistant" and m["content"] == "P" * 600
                for m in second_provider.last_messages
            )
            == 1
        )
        store = creature.agent.session_store
        rows = await store.run(store.get_events, "crash_recovery")
        assert len([row for row in rows if row.get("finalize") == "recovered"]) == 1
