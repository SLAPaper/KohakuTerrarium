"""Rebuild agents from session files and restore their persisted runtime state."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kohakuterrarium.builtins.inputs import create_builtin_input
from kohakuterrarium.builtins.outputs import create_builtin_output
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.agent_selection import restore_selections
from kohakuterrarium.core.config_serde import unpack_agent_config
from kohakuterrarium.core.conversation_elide import (
    elide_stale_tool_results,
    estimate_tokens,
)
from kohakuterrarium.errors import ConflictError, SessionError, SessionNotResumableError
from kohakuterrarium.modules.input.base import InputModule
from kohakuterrarium.modules.output.base import OutputModule
from kohakuterrarium.packages.resolve import resolve_any_path
from kohakuterrarium.session.history import (
    dedupe_adjacent_duplicate_events,
    normalize_resumable_events,
)
from kohakuterrarium.session.resume_open import (
    # Re-exported for callers that still import it from this module.
    detect_session_type as detect_session_type,  # noqa: F401
    open_store_with_migration as _open_store_with_migration,
    preflight_legacy_workspace,
)
from kohakuterrarium.session.resume_branch import (
    load_conversation_with_replay_fallback as _load_conversation_with_replay_fallback,
    replayed_messages_for,
    snapshot_mismatches_branch,
)
from kohakuterrarium.session.resume_admission import validate_resume_snapshot
from kohakuterrarium.session.resume_provider import prepare_resume_provider
from kohakuterrarium.session.resume_selection import (
    apply_branch_coordinates,
    latest_selected_branches,
    resolve_resume_branch_view,
    validate_selected_raw_events,
)
from kohakuterrarium.session.resume_build import (
    build_conversation as _build_conversation,
)
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.text_recovery import recover_record
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

IO_MODES = ("cli", "plain", "tui")


def _mark_conversation_open(store: SessionStore) -> None:
    """Persist the UI lifecycle marker when the store supports it."""
    setter = getattr(store, "set_conversation_open", None)
    if callable(setter):
        setter(True)


def _create_io_modules(
    mode: str,
) -> tuple[InputModule, OutputModule]:
    """Create input and output modules for a given IO mode.

    Rich ``cli`` modules must be supplied by the caller because importing their
    higher-level dependencies here would create a package cycle.
    """
    match mode:
        case "plain":
            return create_builtin_input("cli", {}), create_builtin_output("stdout", {})
        case "tui":
            return create_builtin_input("tui", {}), create_builtin_output("tui", {})
        case _:
            raise ValueError(
                f"Unknown IO mode: {mode}. Use one of {IO_MODES} "
                "(``cli`` mode must be constructed by the caller and "
                "passed via ``input_module`` / ``output_module``)."
            )


def _restore_turn_branch_state(
    agent,
    store: SessionStore,
    agent_name: str,
    events: list[dict] | None = None,
    branch_view: dict[int, int] | None = None,
) -> None:
    """Set turn / branch / parent-path / view on the agent from saved events.

    ``branch_view`` is the resolved selected map (already expanded against
    raw ancestry). When it is set, those coordinates are installed as-is so
    a non-latest selection cannot fall back to the latest live subtree.
    Without it, resume picks the latest compatible subtree, matching
    ``replay_conversation``'s default. ``events`` is the caller's shared
    raw read; omit it to read the store here.
    """
    if branch_view is not None:
        apply_branch_coordinates(agent, branch_view)
        logger.debug(
            "Turn/branch state restored from selected view",
            agent=agent_name,
            turn_index=getattr(agent, "_turn_index", None),
            branch_id=getattr(agent, "_branch_id", None),
        )
        return
    if events is None:
        try:
            events = store.get_events(agent_name)
        except Exception as e:
            logger.warning(
                "Failed to read events for turn/branch restore",
                error=str(e),
                exc_info=True,
            )
            return
    selected = latest_selected_branches(list(events))
    if not selected:
        return
    apply_branch_coordinates(agent, selected)
    logger.debug(
        "Turn/branch state restored",
        agent=agent_name,
        turn_index=getattr(agent, "_turn_index", None),
        branch_id=getattr(agent, "_branch_id", None),
        parent_path_len=len(getattr(agent, "_parent_branch_path", []) or []),
    )


def align_agent_name(agent, agent_name: str) -> None:
    """Force ``agent`` to identify as ``agent_name`` after resume.

    Rebuilding can generate a different runtime name, which would split reads and
    writes across namespaces. All subsystem name caches are aligned to the saved
    namespace.
    """
    if getattr(agent, "config", None) is not None:
        agent.config.name = agent_name
    executor = getattr(agent, "executor", None)
    if executor is not None and hasattr(executor, "_agent_name"):
        executor._agent_name = agent_name
    trigger_manager = getattr(agent, "trigger_manager", None)
    if trigger_manager is not None and hasattr(trigger_manager, "_agent_name"):
        trigger_manager._agent_name = agent_name
    compact_manager = getattr(agent, "compact_manager", None)
    if compact_manager is not None and hasattr(compact_manager, "_agent_name"):
        compact_manager._agent_name = agent_name


def _apply_restore_elision(agent: Any) -> None:
    """Re-apply tool-result elision after restoring/rebuilding a conversation.

    Rebuilds restore tool outputs elided during live turns, so re-apply
    elision when the estimated prompt is already past the compact threshold
    (prevents the first resumed LLM call from overflowing). Elision is a
    compact companion: it only fires under real pressure.
    """
    controller = getattr(agent, "controller", None)
    if controller is None:
        return
    config = getattr(controller, "config", None)
    if config is None or not getattr(config, "elide_tool_results", False):
        return
    compact = getattr(agent, "compact_manager", None)
    compact_max = (
        compact.config.max_tokens
        if compact is not None
        and compact.config.enabled
        and getattr(compact.config, "max_tokens", 0)
        else 0
    )
    if compact_max and estimate_tokens(controller.conversation) >= int(
        compact_max * compact.config.threshold
    ):
        elide_stale_tool_results(controller.conversation)


def _reapply_options(agent: Any, agent_name: str, attribute: str) -> None:
    options = getattr(agent, attribute, None)
    if options is None:
        return
    try:
        options.apply()
    except Exception as exc:  # pragma: no cover - resume continues without options
        message = f"Failed to reapply {attribute.replace('_', ' ')}"
        logger.warning(message, agent=agent_name, error=str(exc))


def _reject_running_injection(agent: Any) -> None:
    """Refuse injection once the session-state owner has started.

    Injection mutates conversation and branch coordinates. A started owner
    is already committing that state; doing it from the store worker races
    the owner. The check is local — it must not await the owner.
    """
    owner = getattr(agent, "_session_state_owner", None)
    if owner is not None and getattr(owner, "has_started", False):
        raise ConflictError(
            "Saved-state injection is only allowed before Agent start; "
            "build a fresh Agent to resume"
        )


def inject_saved_state(agent, store: SessionStore, agent_name: str) -> None:
    """Restore identity, conversation, branch state, scratchpad, and triggers.

    Future writes remain in the saved namespace, and interrupted events are
    queued for the rebuilt agent's resume flow.

    A persisted ``snapshot_branch.branch_view`` is the selected raw map.
    Snapshot acceptance, replay/backfill/tail, coordinates, and
    ``agent._branch_view`` all use that same map. A non-latest selection
    never loads the latest coordinates. A view that disagrees with raw
    ancestry raises :class:`SessionError`.
    """
    _reject_running_injection(agent)
    recover_record(store, agent_name)
    # ONE raw read of the event log feeds every consumer below (conversation
    # replay, branch restore, mismatch replay) — not the old 3-4 scans.
    events = store.get_events(agent_name)
    # Resolve before normalize/replay. Generation provenance is checked on
    # this same selected raw slice, still before normalize drops links.
    branch_view = resolve_resume_branch_view(events, store, agent_name)
    origins = validate_selected_raw_events(events, branch_view)
    coordinates = SimpleNamespace(
        _turn_index=getattr(agent, "_turn_index", 0),
        _branch_id=getattr(agent, "_branch_id", 0),
        _parent_branch_path=list(getattr(agent, "_parent_branch_path", [])),
        _branch_view=dict(getattr(agent, "_branch_view", {}) or {}),
    )
    _restore_turn_branch_state(
        coordinates, store, agent_name, events, branch_view=branch_view
    )
    if snapshot_mismatches_branch(
        store, coordinates, agent_name, branch_view=branch_view
    ):
        logger.info(
            "Snapshot belongs to another branch — replaying target branch",
            agent=agent_name,
        )
        saved_messages = replayed_messages_for(store, agent_name, events, branch_view)
    else:
        saved_messages = _load_conversation_with_replay_fallback(
            store, agent_name, events, branch_view, origins=origins
        )
    validate_resume_snapshot(saved_messages, origins)
    conversation = _build_conversation(saved_messages) if saved_messages else None
    provider = prepare_resume_provider(
        agent, store, agent_name, saved_messages, events, branch_view
    )
    provider.publish(agent)
    align_agent_name(agent, agent_name)
    agent._turn_index = coordinates._turn_index
    agent._branch_id = coordinates._branch_id
    agent._parent_branch_path = coordinates._parent_branch_path
    agent._branch_view = coordinates._branch_view
    if conversation is not None:
        agent.controller.conversation = conversation
        _apply_restore_elision(agent)
        logger.info(
            "Conversation restored", agent=agent_name, messages=len(saved_messages)
        )

    pad_data = store.load_scratchpad(agent_name)
    if pad_data:
        legacy_native_options = pad_data.get("__native_tool_options__")
        if legacy_native_options:
            agent.session.scratchpad.set(
                "__native_tool_options__", legacy_native_options
            )
        visible_count = 0
        for k, v in pad_data.items():
            if k.startswith("__") and k.endswith("__"):
                continue
            agent.session.scratchpad.set(k, v)
            visible_count += 1
        logger.info("Scratchpad restored", agent=agent_name, keys=visible_count)

    _reapply_options(agent, agent_name, "native_tool_options")
    _reapply_options(agent, agent_name, "tool_options")

    # Same dedupe→normalize pipeline as SessionStore.get_resumable_events,
    # computed from the shared read instead of a second full table scan.
    resume_events = normalize_resumable_events(dedupe_adjacent_duplicate_events(events))
    if resume_events:
        agent._pending_resume_events = resume_events
        logger.info("Resume events loaded", agent=agent_name, count=len(resume_events))

    saved_triggers = store.load_triggers(agent_name)
    if saved_triggers:
        agent._pending_resume_triggers = saved_triggers
        logger.info(
            "Resumable triggers loaded",
            agent=agent_name,
            count=len(saved_triggers),
        )

    try:
        restore_selections(agent, store, restore_model=False)
    except SessionError:
        raise
    except Exception:  # pragma: no cover - resume continues without selections
        logger.warning(
            "selection restore failed",
            agent=agent_name,
            exc_info=True,
        )


def _rebuild_agent(
    *,
    config_path: str,
    config_snapshot: dict[str, Any],
    llm: Any,
    io_kwargs: dict[str, Any],
    pwd: str | None = None,
) -> Agent:
    """Build the ``Agent`` from saved meta.

    Prefer a resolvable config path, then fall back to the embedded snapshot for
    cross-node or inline-spawn sessions. Non-strict construction lets users open
    saved history even when the original model profile is unavailable.
    """
    if config_path:
        try:
            path_obj = resolve_any_path(config_path)
        except (FileNotFoundError, ValueError):
            path_obj = None
        if path_obj is not None and path_obj.exists():
            return Agent.from_path(
                str(path_obj), llm=llm, pwd=pwd, strict=False, **io_kwargs
            )
    if not config_snapshot:
        # Without a snapshot, callers must deploy the original config before retrying.
        raise FileNotFoundError(
            f"Agent config folder not found at {config_path!r} and the "
            "session has no config_snapshot to rebuild from"
        )
    cfg = unpack_agent_config(config_snapshot)
    return Agent(cfg, llm=llm, pwd=pwd, strict=False, **io_kwargs)


def resume_agent(
    session_path: str | Path,
    pwd_override: str | None = None,
    io_mode: str | None = None,
    llm: Any = None,
    *,
    input_module: InputModule | None = None,
    output_module: OutputModule | None = None,
    mark_conversation_open: bool = True,
) -> tuple[Agent, SessionStore]:
    """Resume a standalone agent and return it with its writable store.

    Explicit input or output modules override ``io_mode``. The caller owns the
    resumed agent loop and must close the returned store.
    """
    pwd_override = preflight_legacy_workspace(session_path, pwd_override)
    store = _open_store_with_migration(session_path, writer_lock=True)
    try:
        return _resume_agent_from_open_store(
            store,
            session_path,
            pwd_override=pwd_override,
            io_mode=io_mode,
            llm=llm,
            input_module=input_module,
            output_module=output_module,
            mark_conversation_open=mark_conversation_open,
        )
    except BaseException:
        # Any post-open failure must release the writer lock before propagating.
        try:
            store.close(update_status=False)
        except Exception:
            logger.warning(
                "resume_agent: closing store after failed resume failed",
                exc_info=True,
            )
        raise


def _rebuild_resumed_agent(
    store: SessionStore,
    session_path: str | Path,
    *,
    pwd_override: str | None,
    io_mode: str | None,
    llm: Any,
    input_module: InputModule | None,
    output_module: OutputModule | None,
) -> tuple[Agent, str]:
    """Rebuild an agent from an open store, up to state injection.

    Returns ``(agent, saved_agent_name)``. Everything after this point —
    injection, status marking, store attachment — is kept separate so the
    async resume flow can dispatch the injection onto the store's affinity
    thread without duplicating the rebuild checks.
    """
    meta = store.load_meta()

    # Missing type metadata follows detection's agent default for partial mirrors.
    config_type = meta.get("config_type")
    if config_type not in (None, "", "agent"):
        raise ValueError(
            f"Session config_type is {config_type!r}, not 'agent'. "
            "Resume the saved file via "
            "`Terrarium.resume(path)` / `engine.adopt_session(path)` "
            "(see kohakuterrarium.terrarium.resume.resume_into_engine) "
            "which dispatches between the agent and terrarium rebuild "
            "paths."
        )

    config_path = meta.get("config_path", "")
    config_snapshot = meta.get("config_snapshot") or {}
    if not config_path and not config_snapshot:
        raise ValueError("Session has no config_path or config_snapshot in metadata")

    # Pass workspace explicitly; process-wide directory changes race other sessions.
    saved_pwd = meta.get("pwd")
    pwd = pwd_override or saved_pwd
    if not (pwd and os.path.isdir(pwd)):
        source = "override" if pwd_override else "saved"
        raise SessionNotResumableError(
            f"The {source} working directory is missing or invalid: {pwd!r}. "
            "Choose a replacement directory or open the session history."
        )

    # Explicit module instances take precedence over the mode shortcut.
    io_kwargs: dict[str, Any] = {}
    if input_module is not None or output_module is not None:
        if input_module is not None:
            io_kwargs["input_module"] = input_module
        if output_module is not None:
            io_kwargs["output_module"] = output_module
    elif io_mode:
        inp, out = _create_io_modules(io_mode)
        io_kwargs["input_module"] = inp
        io_kwargs["output_module"] = out

    # Resolution order is caller override, saved profile, then provider default.
    effective_llm = llm
    if not effective_llm:
        try:
            effective_llm = store.state.get(
                f"{meta.get('agents', ['agent'])[0]}:llm_profile"
            )
        except (KeyError, Exception):
            pass

    # Embedded snapshots support inline-spawn and cross-node resume.
    agent = _rebuild_agent(
        config_path=config_path,
        config_snapshot=config_snapshot,
        llm=effective_llm,
        io_kwargs=io_kwargs,
        pwd=pwd,
    )
    agent_name = meta.get("agents", [agent.config.name])[0]
    return agent, agent_name


def _attach_resumed_agent(
    agent: Agent,
    store: SessionStore,
    session_path: str | Path,
    agent_name: str,
    *,
    mark_conversation_open: bool,
) -> tuple[Agent, SessionStore]:
    """Mark the conversation open and attach the store for continued turns."""
    # Continued turns append to the same session file.
    if mark_conversation_open:
        _mark_conversation_open(store)
        store.update_status("running")
    agent.attach_session_store(store)

    logger.info("Agent resumed", agent=agent_name, session=str(session_path))
    return agent, store


def _resume_agent_from_open_store(
    store: SessionStore,
    session_path: str | Path,
    *,
    pwd_override: str | None,
    io_mode: str | None,
    llm: Any,
    input_module: InputModule | None,
    output_module: OutputModule | None,
    mark_conversation_open: bool,
) -> tuple[Agent, SessionStore]:
    """Rebuild and rehydrate an agent from an already-open session store."""
    agent, agent_name = _rebuild_resumed_agent(
        store,
        session_path,
        pwd_override=pwd_override,
        io_mode=io_mode,
        llm=llm,
        input_module=input_module,
        output_module=output_module,
    )

    inject_saved_state(agent, store, agent_name)

    return _attach_resumed_agent(
        agent,
        store,
        session_path,
        agent_name,
        mark_conversation_open=mark_conversation_open,
    )
