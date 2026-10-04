"""Local compact compilation verifies retained sources before set subtraction."""

from contextlib import closing

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_compile import compile_coverage
from kohakuterrarium.session.event_origin import store_uuid
from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize("collision_in_retained", [False, True])
def test_inherited_compact_source_ambiguity_is_not_hidden_by_retention(
    tmp_path, collision_in_retained
):
    with closing(SessionStore(tmp_path / "compact-collision.kohakutr")) as store:

        def observe():
            store.append_event(
                "alice", "text_chunk", {"event_id": 7, "content": "answer"}
            )
            store.append_event(
                "alice", "plugin_hook_timing", {"event_id": 7, "hook": "on_agent_stop"}
            )
            store.append_event(
                "alice", "text_chunk", {"event_id": 8, "content": "other answer"}
            )
            uid = store_uuid(store)

            def inherited(event_id):
                return {
                    "role": "assistant",
                    "content": "summary",
                    "metadata": {
                        "compact_coverage": {
                            "compact_coverage_version": 1,
                            "coverage_mode": "exact",
                            "replaced_event_origins": [[uid, "alice", event_id]],
                            "summary_anchor_origin": [uid, "alice", event_id],
                        }
                    },
                }

            ambiguous, unique = inherited(7), inherited(8)
            return compile_coverage(
                store,
                "alice",
                [unique, ambiguous] if collision_in_retained else [ambiguous],
                [ambiguous] if collision_in_retained else [],
            )

        with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
            store.submit(observe).result()
