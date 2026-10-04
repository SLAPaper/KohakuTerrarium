"""Worker dispatch preserves failed rerun settlement and its committed branch receipt."""

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.interrupt_regeneration import RegenerationTurnError
from kohakuterrarium.laboratory._internal.app import AppMessage
from kohakuterrarium.laboratory.adapters.terrarium_runtime import (
    TerrariumRuntimeAdapter,
)
from kohakuterrarium.terrarium.remote_service import _maybe_raise
from kohakuterrarium.testing.llm import ScriptedLLM


class Registrar:
    client_id = None

    def register_app_extension(self, namespace, handler):
        self.handler = handler


class FailedRerun(ScriptedLLM):
    async def chat(self, messages, **kwargs):
        if self.call_count:
            self.call_count += 1
            raise RuntimeError("worker rerun failed")
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("operation", ["regenerate", "edit_message"])
async def test_worker_rerun_failure_is_error_with_applied_receipt(tmp_path, operation):
    provider = FailedRerun(["original"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        await creature.run("question", timeout=3)
        adapter = TerrariumRuntimeAdapter(engine, Registrar())
        body = {"creature_id": creature.creature_id, "request_id": "worker-rerun"}
        if operation == "edit_message":
            body.update(
                msg_idx=next(
                    i
                    for i, m in enumerate(
                        creature.agent.controller.conversation.get_messages()
                    )
                    if m.role == "user"
                ),
                content="edited",
            )
        response = await adapter._dispatch(
            AppMessage(
                namespace=adapter.NAMESPACE,
                type=operation,
                body=body,
                sender_node="controller",
                request_id="rpc",
                in_reply_to=None,
            )
        )
        assert "status" not in response
        assert response["error"]["kind"] == "regeneration_failed"
        assert response["error"]["receipt"]["status"] == "error"
        assert response["error"]["receipt"]["branch_id"] == 2
        assert response["error"]["receipt"]["turn_index"] == 1
        assert response["error"]["receipt"]["request_id"] == "worker-rerun"
        with pytest.raises(RegenerationTurnError) as raised:
            _maybe_raise(response)
        assert raised.value.receipt == response["error"]["receipt"]
        assert raised.value.may_have_run
        assert provider.call_count == 2
