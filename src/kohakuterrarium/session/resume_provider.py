"""Stage effective cold-resume providers before publishing saved history."""

from dataclasses import dataclass, replace

from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.model_switch_prepare import prepare
from kohakuterrarium.core.model_switch_runtime import (
    _bind_callbacks,
    _candidate_compact,
    _capture_previous,
    _model_api,
    _publish,
    _read_options,
    _remember_retired,
    _schedule_cleanup,
)
from kohakuterrarium.errors import SessionError
from kohakuterrarium.llm.generation import provider_source
from kohakuterrarium.session.history import select_live_event_ids
from kohakuterrarium.session.model_switch_history import validate_switch_history
from kohakuterrarium.session.store_control import read_model_selection
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)
_MISSING = object()


@dataclass
class ResumeProvider:
    bundle: object
    compact: object
    previous: tuple
    created: list

    def publish(self, agent):
        """Bind the exact staged providers without replacing saved system content."""
        if self.bundle is None:
            return
        saved = []
        retired = list(getattr(agent, "_retired_model_providers", []))
        had_retired = hasattr(agent, "_retired_model_providers")
        try:
            saved = _publication_state(agent, self.bundle)
            _publish(agent, self.bundle, self.compact, self.bundle.identifier)
            _remember_retired(
                agent, self.previous, retained=(self.bundle.llm, self.compact)
            )
        except BaseException:
            for target, key, value in reversed(saved):
                if value is _MISSING:
                    if hasattr(target, key):
                        delattr(target, key)
                else:
                    setattr(target, key, value)
            if had_retired:
                agent._retired_model_providers = retired
            elif hasattr(agent, "_retired_model_providers"):
                delattr(agent, "_retired_model_providers")
            self.discard(agent)
            raise

    def discard(self, agent):
        _schedule_cleanup(self.created, agent)


def _publication_state(agent, bundle):
    fields = [
        (agent, "_llm_selector"),
        (agent, "_llm_identifier"),
        (agent, "llm"),
        (agent.controller, "llm"),
        (agent.registry, "_tools"),
        (agent.registry, "_tool_infos"),
        (agent.executor, "_tools"),
        (agent, "_auto_provider_tools"),
    ]
    manager = getattr(agent, "compact_manager", None)
    if manager is not None:
        fields.extend(((manager, "_llm"), (manager.config, "max_tokens")))
    load_context = getattr(getattr(agent, "plugins", None), "_load_context", None)
    if load_context is not None:
        fields.append((load_context, "model"))
    for name, values in bundle.option_state.items():
        fields.extend((getattr(agent, name), key) for key in values)
    if getattr(agent, "subagent_manager", None) is not None:
        fields.append((agent.subagent_manager, "llm"))
    return [(target, key, getattr(target, key, _MISSING)) for target, key in fields]


def _resolve_candidate(selector, committed):
    api = _model_api()
    try:
        profile = api.resolve_controller_llm({}, llm=selector)
        if profile is None:
            raise SessionError("Saved provider is unavailable")
        identifier = api.profile_to_identifier(profile)
        return api.create_llm_from_profile_name(selector), identifier
    except Exception as exc:
        if committed is not None:
            raise SessionError("Cannot restore committed model selection") from exc
        logger.warning(
            "Unavailable legacy model selection ignored",
            selector=selector,
            error=str(exc),
        )
        return None


def prepare_resume_provider(agent, store, namespace, messages, events, view):
    """Resolve saved intent and admit detached history against its final provider."""
    committed = read_model_selection(store, namespace)
    selector = (
        committed
        if committed is not None
        else store.state.get(f"{namespace}:model_selection")
    )
    if selector is not None and not isinstance(selector, str):
        raise SessionError("Invalid saved model selection")
    selector = selector.strip() if selector else None
    staged = ResumeProvider(None, None, _capture_previous(agent), [])
    current = getattr(agent, "llm", None)
    target = current
    try:
        if selector:
            resolved = _resolve_candidate(selector, committed)
            if resolved is not None:
                target, identifier = resolved
                staged.created.append(target)
                staged.bundle = prepare(
                    agent, selector, target, identifier, _read_options(store, namespace)
                )
                staged.bundle = replace(staged.bundle, system_prompt=None)
                staged.compact = _candidate_compact(agent, selector, target)
                if staged.compact is not None and staged.compact is not target:
                    staged.created.append(staged.compact)
                _bind_callbacks(agent, target)
        tool_format = getattr(
            getattr(agent.controller, "config", None), "tool_format", None
        )
        source = provider_source(target, tool_format)
        validate_history_compatibility(
            messages or [], source, current_source=source, allow_pending_tail=True
        )
        selected_ids = select_live_event_ids(events, branch_view=view)
        selected = [
            event
            for event in events
            if not isinstance(event.get("event_id"), int)
            or event["event_id"] in selected_ids
        ]
        validate_switch_history(
            messages or [], selected, events, source, allow_pending_tail=True
        )
        return staged
    except BaseException:
        staged.discard(agent)
        raise
