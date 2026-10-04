"""Adapter unit ops — companion suite for :class:`TerrariumRuntimeAdapter`.

Companion to `test_terrarium_runtime_adapter.py` (topology / lifecycle /
channel ops); split out to keep each file under the 600-line cap. Same
harness: a real :class:`Terrarium` engine via `TestTerrariumBuilder`,
a fake `LabRegistrar`, and `creature.agent` stubbed per-op where the
test-builder's `_FakeAgent` lacks the production agent surface.
"""

from types import SimpleNamespace

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.laboratory._internal.app import AppMessage
from kohakuterrarium.laboratory.adapters.terrarium_runtime import (
    TerrariumRuntimeAdapter,
)
from kohakuterrarium.session.raw_history import UserMessageSelector
from kohakuterrarium.testing.terrarium import TestTerrariumBuilder


class _FakeNode:
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.registered = {}
        self.unregistered = []

    def register_app_extension(self, ns, handler):
        self.registered[ns] = handler

    def unregister_app_extension(self, ns):
        self.unregistered.append(ns)
        self.registered.pop(ns, None)


def _msg(type_, body=None, sender="ctrl") -> AppMessage:
    return AppMessage(
        namespace=TerrariumRuntimeAdapter.NAMESPACE,
        type=type_,
        body=body or {},
        sender_node=sender,
        request_id=None,
        in_reply_to=None,
    )


async def _make_adapter():
    engine = await (
        TestTerrariumBuilder()
        .with_creature("alice", responses=["hi"])
        .with_creature("bob")
        .with_channel("chat")
        .with_connection("alice", "bob", channel="chat")
        .build()
    )
    return TerrariumRuntimeAdapter(engine, _FakeNode())


# ── connect / disconnect ────────────────────────────────────────


class TestConnectDisconnect:
    async def test_disconnect_then_connect_round_trip(self):
        adapter = await _make_adapter()
        try:
            # alice↔bob start connected on "chat"; disconnect splits the
            # graph and the result names the affected channel.
            out = await adapter._dispatch(
                _msg(
                    "disconnect",
                    {
                        "sender_id": "alice",
                        "receiver_id": "bob",
                        "channel": "chat",
                    },
                )
            )
            assert out["result"]["delta_kind"] == "split"
            assert "chat" in out["result"]["channels"]
            # Reconnecting merges the two singleton graphs back together.
            out = await adapter._dispatch(
                _msg(
                    "connect",
                    {
                        "sender_id": "alice",
                        "receiver_id": "bob",
                        "channel": "chat",
                    },
                )
            )
            assert out["result"]["delta_kind"] == "merge"
            assert out["result"]["channel"] == "chat"
        finally:
            await adapter._engine.shutdown()


# ── per-creature control ops ────────────────────────────────────


def _stub_agent_control(creature):
    """Replace ``creature.agent`` with a control-surface stub.

    Records ``interrupt`` and exposes deterministic job stores so the
    adapter's control ops can be exercised without a live LLM agent.
    """
    state = {"interrupted": False}
    running = [SimpleNamespace(to_dict=lambda: {"id": "j1", "kind": "tool"})]

    async def _ex_cancel(jid):
        return jid == "exec-job"

    async def _sa_cancel(jid):
        return jid == "sa-job"

    async def _interrupt_and_wait():
        state["interrupted"] = True

    creature.agent = SimpleNamespace(
        is_running=False,
        interrupt_and_wait=_interrupt_and_wait,
        interrupt=lambda: state.__setitem__("interrupted", True),
        executor=SimpleNamespace(get_running_jobs=lambda: running, cancel=_ex_cancel),
        subagent_manager=SimpleNamespace(
            get_running_jobs=lambda: [], cancel=_sa_cancel
        ),
        _interrupt_direct_job=lambda jid: jid == "direct-job",
        _promote_handle=lambda jid: jid == "promote-me",
    )
    return state


class TestControlOps:
    async def test_interrupt_conflict_retains_retryable_error_kind(self):
        adapter = await _make_adapter()
        try:

            async def finishing():
                raise ConflictError("The previous turn is still finishing")

            adapter._engine.get_creature("alice").agent.interrupt_and_wait = finishing
            out = await adapter._dispatch(_msg("interrupt", {"creature_id": "alice"}))
            assert out["error"]["kind"] == "conflict"
            assert "still finishing" in out["error"]["message"]
        finally:
            await adapter._engine.shutdown()

    async def test_interrupt_calls_agent_interrupt(self):
        adapter = await _make_adapter()
        try:
            state = _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(_msg("interrupt", {"creature_id": "alice"}))
            assert out == {}
            # The op's whole job is to forward to agent.interrupt().
            assert state["interrupted"] is True
        finally:
            await adapter._engine.shutdown()

    async def test_list_jobs_merges_executor_and_subagent_jobs(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(_msg("list_jobs", {"creature_id": "alice"}))
            assert out["jobs"] == [{"id": "j1", "kind": "tool"}]
        finally:
            await adapter._engine.shutdown()

    async def test_stop_job_via_direct_interrupt(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(
                _msg("stop_job", {"creature_id": "alice", "job_id": "direct-job"})
            )
            assert out == {"cancelled": True}
        finally:
            await adapter._engine.shutdown()

    async def test_stop_job_via_executor_cancel(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(
                _msg("stop_job", {"creature_id": "alice", "job_id": "exec-job"})
            )
            assert out == {"cancelled": True}
        finally:
            await adapter._engine.shutdown()

    async def test_stop_job_via_subagent_cancel(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(
                _msg("stop_job", {"creature_id": "alice", "job_id": "sa-job"})
            )
            assert out == {"cancelled": True}
        finally:
            await adapter._engine.shutdown()

    async def test_stop_job_unknown_reports_not_cancelled(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            out = await adapter._dispatch(
                _msg("stop_job", {"creature_id": "alice", "job_id": "ghost"})
            )
            assert out == {"cancelled": False}
        finally:
            await adapter._engine.shutdown()

    async def test_promote_job_reports_outcome(self):
        adapter = await _make_adapter()
        try:
            _stub_agent_control(adapter._engine.get_creature("alice"))
            ok = await adapter._dispatch(
                _msg(
                    "promote_job",
                    {"creature_id": "alice", "job_id": "promote-me"},
                )
            )
            assert ok == {"promoted": True}
            miss = await adapter._dispatch(
                _msg("promote_job", {"creature_id": "alice", "job_id": "no"})
            )
            assert miss == {"promoted": False}
        finally:
            await adapter._engine.shutdown()

    async def test_control_op_on_unknown_creature_is_not_hosted(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(_msg("interrupt", {"creature_id": "ghost"}))
            assert out["error"]["kind"] == "creature_not_hosted"
        finally:
            await adapter._engine.shutdown()


# ── per-creature chat ops ───────────────────────────────────────


class TestChatOps:
    async def test_chat_history_returns_history_payload(self):
        adapter = await _make_adapter()
        try:
            adapter._engine.get_creature("alice").agent.is_processing = False
            out = await adapter._dispatch(
                _msg("chat_history", {"creature_id": "alice"})
            )
            # The op wraps the engine's chat-history view under "history".
            assert "history" in out
        finally:
            await adapter._engine.shutdown()

    async def test_chat_branches_returns_branches_payload(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(
                _msg("chat_branches", {"creature_id": "alice"})
            )
            assert "branches" in out
        finally:
            await adapter._engine.shutdown()

    async def test_chat_history_unknown_creature_not_hosted(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(
                _msg("chat_history", {"creature_id": "ghost"})
            )
            assert out["error"]["kind"] == "creature_not_hosted"
        finally:
            await adapter._engine.shutdown()

    async def test_regenerate_forwards_turn_args_to_agent(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []
            receipt = {
                "status": "completed",
                "request_id": "receipt-req-1",
                "turn_index": 3,
                "branch_id": 1,
                "parent_branch_path": [],
            }

            async def _regen(**kwargs):
                calls.append(kwargs)
                return receipt

            legacy_calls = []

            async def _legacy_regen(*a, **kw):
                legacy_calls.append((a, kw))

            creature.agent = SimpleNamespace(
                is_running=False,
                interrupt_and_regenerate=_regen,
                regenerate_last_response=_legacy_regen,
            )
            out = await adapter._dispatch(
                _msg(
                    "regenerate",
                    {
                        "creature_id": "alice",
                        "turn_index": 3,
                        "branch_view": "v",
                    },
                )
            )
            assert out == receipt
            assert out["status"] == "completed"
            assert out["turn_index"] == 3
            assert out["branch_id"] == 1
            assert out["request_id"] == "receipt-req-1"
            # The turn/branch selectors are passed through; service generates request_id when omitted.
            assert len(calls) == 1
            assert calls[0]["turn_index"] == 3
            assert calls[0]["branch_view"] == "v"
            assert isinstance(calls[0]["request_id"], str) and calls[0]["request_id"]
            assert legacy_calls == []
        finally:
            await adapter._engine.shutdown()

    async def test_regenerate_returns_opened_branch_for_navigator_promotion(self):
        # The frontend's chevron navigator <N/M> needs to promote
        # immediately on Retry / Save&Rerun — it can't wait for the
        # post-turn history resync. The dispatcher surfaces the receipt
        # from interrupt_and_regenerate with its opened branch_id and turn_index.
        # Stub agent sets sentinel mutable attributes to prove they do not drive the DTO.
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []

            async def _regen(**kwargs):
                calls.append(kwargs)
                return {
                    "status": "completed",
                    "request_id": kwargs["request_id"],
                    "turn_index": 3,
                    "branch_id": 2,
                    "parent_branch_path": [],
                }

            legacy_calls = []

            async def _legacy_regen(*a, **kw):
                legacy_calls.append((a, kw))

            creature.agent = SimpleNamespace(
                is_running=False,
                interrupt_and_regenerate=_regen,
                regenerate_last_response=_legacy_regen,
                _turn_index=999,
                _branch_id=999,
            )
            out = await adapter._dispatch(
                _msg(
                    "regenerate",
                    {
                        "creature_id": "alice",
                        "turn_index": 3,
                        "branch_view": None,
                    },
                )
            )
            assert out["status"] == "completed"
            assert out["turn_index"] == 3
            assert out["branch_id"] == 2
            assert out["request_id"] == calls[0]["request_id"]
            assert isinstance(out["request_id"], str) and out["request_id"]
            assert len(calls) == 1
            assert calls[0]["turn_index"] == 3
            assert calls[0]["branch_view"] is None
            assert legacy_calls == []
        finally:
            await adapter._engine.shutdown()

    async def test_regenerate_forwards_target_selector_to_agent(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []
            receipt = {
                "status": "completed",
                "request_id": "target-tok",
                "turn_index": 3,
                "branch_id": 1,
                "parent_branch_path": [],
            }

            async def _regen(**kwargs):
                calls.append(kwargs)
                return receipt

            creature.agent = SimpleNamespace(
                is_running=False,
                interrupt_and_regenerate=_regen,
                regenerate_last_response=lambda *a, **kw: None,
            )
            out = await adapter._dispatch(
                _msg(
                    "regenerate",
                    {
                        "creature_id": "alice",
                        "turn_index": 3,
                        "branch_view": {"1": 1},
                        "target": {
                            "event_id": 42,
                            "turn_index": 3,
                            "branch_id": 0,
                        },
                    },
                )
            )
            assert out == receipt
            assert len(calls) == 1
            assert calls[0]["target"] == UserMessageSelector(
                event_id=42, turn_index=3, branch_id=0
            )
        finally:
            await adapter._engine.shutdown()

    async def test_edit_message_returns_edited_flag(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []

            async def _edit(msg_idx, content, **kw):
                calls.append((msg_idx, content, kw))
                if msg_idx == 2:
                    return {
                        "status": "completed",
                        "request_id": kw["request_id"],
                        "turn_index": 1,
                        "branch_id": 2,
                        "parent_branch_path": [],
                    }
                return None

            legacy_calls = []

            async def _legacy_edit(*a, **kw):
                legacy_calls.append((a, kw))

            creature.agent = SimpleNamespace(
                is_running=False,
                _edit_and_rerun_result=_edit,
                edit_and_rerun=_legacy_edit,
            )
            out = await adapter._dispatch(
                _msg(
                    "edit_message",
                    {
                        "creature_id": "alice",
                        "msg_idx": 2,
                        "content": "fixed",
                    },
                )
            )
            assert out["status"] == "completed"
            assert out["request_id"] == calls[0][2]["request_id"]
            assert isinstance(out["request_id"], str) and out["request_id"]
            assert len(calls) == 1
            assert calls[0][0] == 2
            assert calls[0][1] == "fixed"
            assert calls[0][2]["turn_index"] is None
            assert calls[0][2]["user_position"] is None
            assert calls[0][2]["branch_view"] is None
            assert legacy_calls == []
        finally:
            await adapter._engine.shutdown()

    async def test_edit_message_returns_branch_id_when_agent_assigns_one(self):
        # Same rationale as the regenerate case above — the navigator
        # promotion is gated on receiving the new branch_id back from
        # the worker, so the dispatcher must surface it.
        # Sentinel mutable coordinates on agent prove they do not drive the DTO.
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []

            async def _edit(msg_idx, content, **kw):
                calls.append((msg_idx, content, kw))
                return {
                    "status": "completed",
                    "request_id": kw["request_id"],
                    "turn_index": 4,
                    "branch_id": 3,
                    "parent_branch_path": [],
                }

            legacy_calls = []

            async def _legacy_edit(*a, **kw):
                legacy_calls.append((a, kw))

            creature.agent = SimpleNamespace(
                is_running=False,
                _edit_and_rerun_result=_edit,
                edit_and_rerun=_legacy_edit,
                _turn_index=999,
                _branch_id=999,
            )
            out = await adapter._dispatch(
                _msg(
                    "edit_message",
                    {
                        "creature_id": "alice",
                        "msg_idx": 2,
                        "content": "fixed",
                    },
                )
            )
            assert out["status"] == "completed"
            assert out["turn_index"] == 4
            assert out["branch_id"] == 3
            assert out["request_id"] == calls[0][2]["request_id"]
            assert len(calls) == 1
            assert calls[0][0] == 2
            assert calls[0][1] == "fixed"
            assert legacy_calls == []
        finally:
            await adapter._engine.shutdown()

    async def test_edit_message_forwards_target_selector_to_agent(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []
            receipt = {
                "status": "completed",
                "request_id": "edit-target-tok",
                "turn_index": 2,
                "branch_id": 1,
                "parent_branch_path": [],
            }

            async def _edit(msg_idx, content, **kw):
                calls.append((msg_idx, content, kw))
                return receipt

            creature.agent = SimpleNamespace(
                is_running=False,
                _edit_and_rerun_result=_edit,
                edit_and_rerun=lambda *a, **kw: None,
            )
            out = await adapter._dispatch(
                _msg(
                    "edit_message",
                    {
                        "creature_id": "alice",
                        "msg_idx": 2,
                        "content": "fixed",
                        "target": {
                            "event_id": 99,
                            "turn_index": 2,
                            "branch_id": 0,
                        },
                    },
                )
            )
            assert out == receipt
            assert len(calls) == 1
            assert calls[0][2]["target"] == UserMessageSelector(
                event_id=99, turn_index=2, branch_id=0
            )
        finally:
            await adapter._engine.shutdown()

    async def test_edit_message_failure_does_not_leak_branch_metadata(self):
        # When the agent refuses the edit (wrong target, etc.),
        # _edit_and_rerun_result returns None which service maps to ValueError.
        # The invalid error reply must NOT carry stale turn_index or branch_id —
        # the caller receives kind="invalid" without leaking branch metadata.
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            calls = []

            async def _edit(msg_idx, content, **kw):
                calls.append((msg_idx, content, kw))
                return None

            legacy_calls = []

            async def _legacy_edit(*a, **kw):
                legacy_calls.append((a, kw))

            creature.agent = SimpleNamespace(
                is_running=False,
                _edit_and_rerun_result=_edit,
                edit_and_rerun=_legacy_edit,
                _turn_index=7,
                _branch_id=9,
            )
            out = await adapter._dispatch(
                _msg(
                    "edit_message",
                    {
                        "creature_id": "alice",
                        "msg_idx": 2,
                        "content": "fixed",
                    },
                )
            )
            assert out["error"]["kind"] == "invalid"
            assert "cannot be edited" in out["error"]["message"]
            assert "status" not in out
            assert "turn_index" not in out
            assert "branch_id" not in out
            assert len(calls) == 1
            assert calls[0][0] == 2
            assert calls[0][1] == "fixed"
            assert legacy_calls == []
        finally:
            await adapter._engine.shutdown()

    async def test_rewind_forwards_index_to_agent(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            rewound = []

            async def _rewind(idx):
                rewound.append(idx)

            creature.agent = SimpleNamespace(is_running=False, rewind_to=_rewind)
            out = await adapter._dispatch(
                _msg("rewind", {"creature_id": "alice", "msg_idx": 5})
            )
            assert out == {}
            assert rewound == [5]
        finally:
            await adapter._engine.shutdown()


# ── per-creature state ops ──────────────────────────────────────


class TestStateOps:
    async def test_scratchpad_read_and_patch(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            store = {"k": "v"}

            class _Scratch:
                def to_dict(self):
                    return dict(store)

                def set(self, key, val):
                    store[key] = val

                def delete(self, key):
                    store.pop(key, None)

            creature.agent = SimpleNamespace(is_running=False, scratchpad=_Scratch())
            out = await adapter._dispatch(
                _msg("get_scratchpad", {"creature_id": "alice"})
            )
            assert out == {"scratchpad": {"k": "v"}}
            patched = await adapter._dispatch(
                _msg(
                    "patch_scratchpad",
                    {"creature_id": "alice", "updates": {"k2": "v2"}},
                )
            )
            # The patch is applied and the fresh full scratchpad returned.
            assert patched["scratchpad"] == {"k": "v", "k2": "v2"}
        finally:
            await adapter._engine.shutdown()

    async def test_get_env_returns_env_payload(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(_msg("get_env", {"creature_id": "alice"}))
            # agent_env yields a dict; the op wraps it under "env".
            assert isinstance(out["env"], dict)
        finally:
            await adapter._engine.shutdown()

    async def test_list_triggers_returns_trigger_list(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            from datetime import datetime

            info = SimpleNamespace(
                trigger_id="t1",
                trigger_type="timer",
                running=True,
                created_at=datetime(2026, 1, 1, 12, 0, 0),
            )
            creature.agent = SimpleNamespace(
                is_running=False,
                trigger_manager=SimpleNamespace(list=lambda: [info]),
            )
            out = await adapter._dispatch(
                _msg("list_triggers", {"creature_id": "alice"})
            )
            # Each trigger is flattened into a serialisable dict.
            assert out["triggers"] == [
                {
                    "trigger_id": "t1",
                    "trigger_type": "timer",
                    "running": True,
                    "created_at": "2026-01-01T12:00:00",
                }
            ]
        finally:
            await adapter._engine.shutdown()

    async def test_list_triggers_no_manager_returns_empty(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            creature.agent = SimpleNamespace(is_running=False)
            out = await adapter._dispatch(
                _msg("list_triggers", {"creature_id": "alice"})
            )
            assert out == {"triggers": []}
        finally:
            await adapter._engine.shutdown()

    async def test_get_working_dir_and_set_working_dir(self, tmp_path):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            wd = {"path": "/old"}

            class _Workspace:
                def get(self):
                    return wd["path"]

                def set(self, new):
                    # set_working_dir mutates the workspace and echoes
                    # the resolved path back.
                    wd["path"] = new
                    return new

            creature.agent = SimpleNamespace(is_running=False, workspace=_Workspace())
            target = str(tmp_path)
            out = await adapter._dispatch(
                _msg(
                    "set_working_dir",
                    {"creature_id": "alice", "new_path": target},
                )
            )
            assert out == {"working_dir": target}
            # A follow-up read reflects the mutation.
            got = await adapter._dispatch(
                _msg("get_working_dir", {"creature_id": "alice"})
            )
            assert got == {"working_dir": target}
        finally:
            await adapter._engine.shutdown()

    async def test_get_system_prompt_returns_helper_payload(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(
                _msg("get_system_prompt", {"creature_id": "alice"})
            )
            # agent_system_prompt returns a dict that the op returns as-is.
            assert isinstance(out, dict)
        finally:
            await adapter._engine.shutdown()

    async def test_native_tool_inventory_and_options(self):
        adapter = await _make_adapter()
        try:
            out = await adapter._dispatch(
                _msg("native_tool_inventory", {"creature_id": "alice"})
            )
            assert isinstance(out["inventory"], list)
            opts = await adapter._dispatch(
                _msg("get_native_tool_options", {"creature_id": "alice"})
            )
            assert isinstance(opts["options"], dict)
        finally:
            await adapter._engine.shutdown()

    async def test_set_native_tool_options_forwards_values(self):
        adapter = await _make_adapter()
        try:
            creature = adapter._engine.get_creature("alice")
            received = {}

            def _set_opts(tool, values):
                received["tool"] = tool
                received["values"] = values
                return {"applied": True}

            creature.agent = SimpleNamespace(
                is_running=False,
                set_native_tool_option=lambda *a, **k: None,
            )
            # Patch the helper indirection: the adapter calls
            # agent_set_native_tool_options(agent, tool, values); stub the
            # agent surface that helper reaches.
            import kohakuterrarium.laboratory.adapters.terrarium_runtime as mod

            orig = mod.agent_set_native_tool_options
            mod.agent_set_native_tool_options = lambda ag, tool, values: _set_opts(
                tool, values
            )
            try:
                out = await adapter._dispatch(
                    _msg(
                        "set_native_tool_options",
                        {
                            "creature_id": "alice",
                            "tool": "bash",
                            "values": {"timeout": 5},
                        },
                    )
                )
            finally:
                mod.agent_set_native_tool_options = orig
            assert out == {"options": {"applied": True}}
            assert received == {"tool": "bash", "values": {"timeout": 5}}
        finally:
            await adapter._engine.shutdown()
