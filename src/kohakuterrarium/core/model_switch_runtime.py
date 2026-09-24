"""Live model switches confirm durable intent before publishing bindings.

Bootstrap ``switch_model`` stays synchronous and rejects a started or stopped
agent. Live callers use ``switch_model_async``: one switch lock serializes
requests, the processing lock is held for the whole switch, and the control
event is awaited before any binding is published. Ambiguous commits stay
fenced until the parent refuses further mutation.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
from typing import Any

from kohakuterrarium.core.agent_native_tools import NATIVE_TOOL_OPTIONS_STATE_SUFFIX
from kohakuterrarium.core.agent_selection import persist_model_selection
from kohakuterrarium.core.agent_tool_options import TOOL_OPTIONS_STATE_SUFFIX
from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.model_switch_prepare import prepare
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.generation import provider_source
from kohakuterrarium.session.model_switch_history import validate_switch_history
from kohakuterrarium.session.store_control import (
    AmbiguousControlCommitError,
    append_control_event,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

_BUSY = "Cannot switch model while an agent turn is processing"
_STOPPING = "Cannot switch model while the agent is stopping"
_STOPPED = "Cannot switch model after the agent has stopped"
_CLOSING = "Cannot switch model while the agent is closing"
_LIVE = (
    "Cannot switch_model while the agent is live; "
    "await agent.switch_model_async(profile) instead"
)


def _model_api() -> Any:
    """Resolve the model mixin's provider factory surface."""
    return importlib.import_module("kohakuterrarium.core.agent_model")


def mark_model_switch_closing(agent: Any) -> None:
    """Refuse new switch requests. Call at stop entry, before awaiting."""
    agent._model_switch_closing = True


async def drain_model_switches(agent: Any) -> None:
    """Wait for every accepted switch. Call before plugin unload."""
    task = getattr(agent, "_model_switch_task", None)
    if task is None or task.done():
        return
    try:
        await _join_accepted(task)
    except Exception:
        logger.warning("Accepted model switch failed during stop", exc_info=True)


def _live(agent: Any) -> bool:
    return bool(getattr(agent, "_running", False)) or bool(
        getattr(agent, "_started", False)
    )


def _stopped(agent: Any) -> bool:
    return bool(getattr(agent, "_stopped", False))


def _closing(agent: Any) -> bool:
    return _stopped(agent) or bool(getattr(agent, "_model_switch_closing", False))


def reject_new_model_switch(agent: Any) -> None:
    """Busy, re-entrant, and closed checks. Must not create a switch task."""
    require_model_mutation_available(agent)
    if _closing(agent):
        raise ConflictError(_STOPPED if _stopped(agent) else _CLOSING)
    if getattr(agent, "_model_sensitive_extension_active", 0):
        raise ConflictError("Plugin initialization is still running")
    compact = getattr(agent, "compact_manager", None)
    if compact is not None and compact.is_compacting:
        raise ConflictError("Compaction is still running")
    stop_lock = getattr(agent, "_stop_lock", None)
    if stop_lock is not None and stop_lock.locked():
        raise ConflictError(_STOPPING)
    current = asyncio.current_task()
    lock = getattr(agent, "_processing_lock", None)
    holder = getattr(agent, "_turn_lock_holder", None)
    if holder is not None and holder is current:
        raise ConflictError(_BUSY)
    if lock is not None and lock.locked() and holder is not current:
        raise ConflictError(_BUSY)
    processing = getattr(agent, "_processing_task", None)
    if (
        processing is not None
        and not processing.done()
        and processing is not current
        and holder is not current
    ):
        raise ConflictError(_BUSY)


async def request_model_switch(agent: Any, profile_name: str) -> str:
    """Serialize live switches. Cancelling the waiter does not cancel the task."""
    reject_new_model_switch(agent)
    gate = getattr(agent, "_model_switch_request_lock", None)
    if gate is None:
        gate = agent._model_switch_request_lock = asyncio.Lock()
    async with gate:
        reject_new_model_switch(agent)
        existing = getattr(agent, "_model_switch_task", None)
        if existing is not None and not existing.done():
            raise ConflictError("A model switch is already in progress")
        lock = getattr(agent, "_processing_lock", None)
        if lock is not None:
            await lock.acquire()
        agent._model_switch_pending = True
        try:
            task = asyncio.create_task(
                _run_switch(agent, profile_name),
                name=f"model_switch_{agent.config.name}",
            )
        except BaseException:
            agent._model_switch_pending = False
            if lock is not None:
                lock.release()
            raise
        agent._model_switch_task = task
    try:
        return await _join_accepted(task)
    finally:
        if getattr(agent, "_model_switch_task", None) is task and task.done():
            agent._model_switch_task = None
        agent._model_switch_pending = False
        if lock is not None:
            lock.release()


async def _join_accepted(task: asyncio.Task) -> Any:
    """Finish accepted work before propagating waiter cancellation or its cause."""
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    try:
        result = task.result()
    except BaseException as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def _run_switch(agent: Any, profile_name: str) -> str:
    """Hold the turn lock across snapshot, commit, and publish."""
    lock = getattr(agent, "_processing_lock", None)
    if lock is None:
        return await _switch_locked(agent, profile_name)
    previous = getattr(agent, "_turn_lock_holder", None)
    agent._turn_lock_holder = asyncio.current_task()
    try:
        return await _switch_locked(agent, profile_name)
    finally:
        if getattr(agent, "_turn_lock_holder", None) is asyncio.current_task():
            agent._turn_lock_holder = previous


async def _switch_locked(agent: Any, profile_name: str) -> str:
    agent._model_switch_pending = True
    created: list[Any] = []
    published = False
    committed = False
    try:
        api = _model_api()
        profile = api.resolve_controller_llm({}, llm=profile_name)
        if profile is None:
            raise ValueError(f"Model profile not found: {profile_name}")
        identifier = api.profile_to_identifier(profile)
        new_llm = api.create_llm_from_profile_name(profile_name)
        created.append(new_llm)
        messages, turn, branch, path = _snapshot(agent)
        tool_format = getattr(
            getattr(agent.controller, "config", None), "tool_format", None
        )
        from_source = provider_source(agent.llm, tool_format)
        to_source = provider_source(new_llm, tool_format)
        classification = validate_history_compatibility(
            messages, to_source, current_source=from_source
        )
        validate_switch_history(messages, [], [], to_source)
        options = await _persisted_options(agent)
        bundle = prepare(agent, profile_name, new_llm, identifier, options)
        compact_llm = _candidate_compact(agent, profile_name, new_llm)
        if compact_llm is not None and compact_llm is not new_llm:
            created.append(compact_llm)
        _bind_callbacks(agent, new_llm)
        payload = _control_payload(
            classification=classification,
            from_source=from_source,
            to_source=to_source,
            from_identifier=getattr(agent, "_llm_identifier", "") or None,
            to_identifier=identifier,
        )
        await _commit(agent, payload, turn, branch, path)
        committed = True
        previous = _capture_previous(agent)
        _publish(agent, bundle, compact_llm, identifier)
        published = True
        agent._model_switch_uncertain = False
        _remember_retired(agent, previous, retained=(new_llm, compact_llm))
        _notify(agent, identifier, new_llm)
        await _persist_legacy(agent, identifier)
        return identifier
    except asyncio.CancelledError:
        if not published:
            await _cleanup(created, agent)
        raise
    except AmbiguousControlCommitError:
        agent._model_switch_uncertain = True
        if not published:
            await _cleanup(created, agent)
        raise
    except Exception:
        if committed and not published:
            agent._model_switch_uncertain = True
        if not published:
            await _cleanup(created, agent)
        raise
    finally:
        agent._model_switch_pending = False


def _snapshot(agent: Any) -> tuple[list[Any], Any, Any, Any]:
    conversation = getattr(agent.controller, "conversation", None)
    get_messages = getattr(conversation, "get_messages", None)
    messages = copy.deepcopy(list(get_messages())) if callable(get_messages) else []
    output = getattr(agent, "_session_output", None)
    turn = branch = path = None
    if output is not None:
        turn, branch = output._current_turn_branch()
        parent = output._current_parent_path()
        path = [list(pair) for pair in parent] if parent is not None else None
    return messages, turn, branch, path


async def _persisted_options(agent: Any) -> dict[str, Any]:
    store = getattr(agent, "session_store", None)
    name = getattr(getattr(agent, "config", None), "name", None)
    if store is None or not name or not hasattr(store, "run"):
        return {}

    return await store.run(_read_options, store, name)


def _read_options(store: Any, name: str) -> dict[str, Any]:
    state = getattr(store, "state", None)
    if state is None:
        return {}
    return {
        "native": copy.deepcopy(
            state.get(f"{name}:{NATIVE_TOOL_OPTIONS_STATE_SUFFIX}") or {}
        ),
        "local": copy.deepcopy(state.get(f"{name}:{TOOL_OPTIONS_STATE_SUFFIX}") or {}),
    }


def _candidate_compact(agent: Any, selector: str, new_llm: Any) -> Any:
    manager = getattr(agent, "compact_manager", None)
    build = getattr(agent, "_build_compact_llm", None)
    if not manager or not callable(build):
        return None
    staging = copy.copy(agent)
    staging._llm_selector = selector
    staging.llm = new_llm
    return staging._build_compact_llm(manager.config)


def _bind_callbacks(agent: Any, new_llm: Any) -> None:
    if getattr(agent, "compact_manager", None) is not None:
        try:
            new_llm._overflow_rescue = agent._compact_overflow_rescue
        except (AttributeError, TypeError):
            logger.debug("Provider does not accept the overflow-rescue hook")
    cache_key = _cache_key(agent)
    if cache_key and hasattr(new_llm, "prompt_cache_key"):
        new_llm.prompt_cache_key = cache_key
    callback = getattr(agent, "_on_provider_emergency_drop", None)
    if callback is not None and hasattr(new_llm, "on_emergency_drop"):
        new_llm.on_emergency_drop(callback)
    commit = getattr(agent, "_commit_provider_emergency_drop", None)
    if commit is not None and hasattr(new_llm, "set_emergency_drop_commit"):
        new_llm.set_emergency_drop_commit(commit)


def _control_payload(**facts: Any) -> dict[str, Any]:
    return {
        "schema": 1,
        "classification": facts["classification"],
        "from_source": facts["from_source"],
        "to_source": facts["to_source"],
        "from_identifier": facts["from_identifier"],
        "to_identifier": facts["to_identifier"],
    }


def _namespace(agent: Any) -> str:
    return (
        getattr(getattr(agent, "_session_output", None), "_event_key_prefix", None)
        or agent.config.name
    )


async def _commit(
    agent: Any, payload: dict[str, Any], turn: Any, branch: Any, path: Any
) -> None:
    store = getattr(agent, "session_store", None)
    if store is None:
        return
    await store.run(
        append_control_event,
        store,
        _namespace(agent),
        "model_switch",
        payload,
        turn_index=turn,
        branch_id=branch,
        parent_branch_path=path,
    )


def _publish(agent: Any, bundle: Any, compact_llm: Any, identifier: str) -> None:
    bundle.publish(agent)
    plugins = getattr(agent, "plugins", None)
    load_context = getattr(plugins, "_load_context", None)
    if load_context is not None:
        load_context.model = identifier
    manager = getattr(agent, "compact_manager", None)
    if manager is not None and compact_llm is not None:
        manager._llm = compact_llm
        compact_max = getattr(compact_llm, "_profile_max_context", 0)
        if compact_max:
            manager.config.max_tokens = compact_max


def _cache_key(agent: Any) -> Any:
    key = getattr(agent, "_session_prompt_cache_key", None)
    if key:
        return key
    return getattr(agent, "_session_id", None)


def _notify(agent: Any, identifier: str, new_llm: Any) -> None:
    router = getattr(agent, "output_router", None)
    notify = getattr(router, "notify_activity", None)
    if not callable(notify):
        return
    model_name = getattr(new_llm, "model", identifier)
    new_max = getattr(new_llm, "_profile_max_context", 0)
    manager = getattr(agent, "compact_manager", None)
    compact_at = 0
    if manager and new_max:
        compact_at = int(new_max * manager.config.threshold)
    try:
        notify(
            "session_info",
            f"Model switched to {identifier}",
            metadata={
                "model": model_name,
                "llm_name": identifier,
                "agent_name": getattr(getattr(agent, "config", None), "name", ""),
                "session_id": getattr(agent, "_session_id", ""),
                "max_context": new_max,
                "compact_threshold": compact_at,
            },
        )
    except Exception:
        logger.warning("Model switch notification failed", exc_info=True)
    logger.info(
        "Model switched",
        agent_name=getattr(getattr(agent, "config", None), "name", ""),
        identifier=identifier,
        model=model_name,
    )


async def _persist_legacy(agent: Any, identifier: str) -> None:
    store = getattr(agent, "session_store", None)
    name = getattr(getattr(agent, "config", None), "name", None)
    if store is None or not name:
        return
    key = f"{name}:model_selection"

    def write():
        store.state[key] = identifier

    try:
        await store.run(write)
    except Exception:
        logger.warning("Legacy model selection cache failed", exc_info=True)


def _capture_previous(agent: Any) -> tuple[Any, Any]:
    return (
        getattr(agent, "llm", None),
        getattr(getattr(agent, "compact_manager", None), "_llm", None),
    )


def _remember_retired(
    agent: Any, previous: tuple[Any, Any], *, retained: tuple[Any, ...]
) -> None:
    """Queue old providers. Do not close them during the hot switch."""
    retained_ids = {id(item) for item in retained if item is not None}
    current_ids = {id(item) for item in _capture_previous(agent) if item is not None}
    retired = getattr(agent, "_retired_model_providers", None)
    if retired is None:
        retired = agent._retired_model_providers = []
    seen = {id(item) for item in retired}
    for provider in previous:
        if provider is None:
            continue
        identity = id(provider)
        if identity in retained_ids or identity in current_ids or identity in seen:
            continue
        retired.append(provider)
        seen.add(identity)


async def _cleanup(created: list[Any], agent: Any) -> None:
    protected = {id(item) for item in _capture_previous(agent) if item is not None}
    for provider in created:
        if provider is None or id(provider) in protected:
            continue
        protected.add(id(provider))
        await _close_one(provider)


async def _close_one(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("Failed to close unused model provider", exc_info=True)


async def close_retired_model_providers(agent: Any) -> None:
    """Close providers retired by confirmed switches. Current aliases stay open."""
    cleanup = getattr(agent, "_model_provider_cleanup_tasks", ())
    if cleanup:
        await _join_accepted(asyncio.gather(*cleanup, return_exceptions=True))
        agent._model_provider_cleanup_tasks = []
    retired = list(getattr(agent, "_retired_model_providers", ()) or ())
    agent._retired_model_providers = []
    protected = {id(item) for item in _capture_previous(agent) if item is not None}
    seen: set[int] = set()
    for provider in retired:
        if provider is None or id(provider) in protected or id(provider) in seen:
            continue
        seen.add(id(provider))
        await _close_one(provider)


def bootstrap_switch_model(agent: Any, profile_name: str) -> str:
    """Synchronous pre-start switch. Rejects a live or closed agent first."""
    require_model_mutation_available(agent)
    if _live(agent):
        raise ConflictError(_LIVE)
    if _closing(agent):
        raise ConflictError(_STOPPED if _stopped(agent) else _CLOSING)
    return _bootstrap_mutate(agent, profile_name)


def _bootstrap_mutate(agent: Any, profile_name: str) -> str:
    """Prepare a cold model replacement before changing bootstrap bindings."""
    api = _model_api()
    profile = api.resolve_controller_llm({}, llm=profile_name)
    if profile is None:
        raise ValueError(f"Model profile not found: {profile_name}")
    new_llm = api.create_llm_from_profile_name(profile_name)
    created = [new_llm]
    publication_started = False
    previous = _capture_previous(agent)
    try:
        conversation = getattr(agent.controller, "conversation", None)
        get_messages = getattr(conversation, "get_messages", None)
        messages = list(get_messages()) if callable(get_messages) else []
        tool_format = getattr(
            getattr(agent.controller, "config", None), "tool_format", None
        )
        to_source = provider_source(new_llm, tool_format)
        validate_history_compatibility(
            messages, to_source, current_source=provider_source(agent.llm, tool_format)
        )
        validate_switch_history(messages, [], [], to_source)
        identifier = api.profile_to_identifier(profile)
        options = _read_options(
            getattr(agent, "session_store", None), agent.config.name
        )
        bundle = prepare(agent, profile_name, new_llm, identifier, options)
        compact_llm = _candidate_compact(agent, profile_name, new_llm)
        if compact_llm is not None and compact_llm is not new_llm:
            created.append(compact_llm)
        _bind_callbacks(agent, new_llm)
        publication_started = True
        _publish(agent, bundle, compact_llm, identifier)
    except BaseException:
        if publication_started:
            agent._model_switch_uncertain = True
            _remember_retired(agent, previous, retained=())
        _schedule_cleanup(created, agent)
        raise
    _remember_retired(agent, previous, retained=(new_llm, compact_llm))
    persist_model_selection(agent, identifier)
    _notify(agent, identifier, new_llm)
    return identifier


def _schedule_cleanup(created: list[Any], agent: Any) -> None:
    """Close unpublished providers, tracking loop-bound cleanup until stop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_cleanup(created, agent))
        return
    tasks = getattr(agent, "_model_provider_cleanup_tasks", None)
    if tasks is None:
        tasks = agent._model_provider_cleanup_tasks = []
    tasks.append(loop.create_task(_cleanup(created, agent)))
