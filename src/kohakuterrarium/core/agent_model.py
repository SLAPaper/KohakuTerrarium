"""Agent LLM-profile switching + canonical-identifier helpers.

Provide model switching and canonical LLM identifiers for agents.
Live switches confirm a control event before publishing; the synchronous
method is bootstrap and resume only.
"""

from typing import Any

from kohakuterrarium.bootstrap.llm import (
    create_llm_from_profile_name as create_llm_from_profile_name,
)
from kohakuterrarium.core.model_switch_runtime import (
    bootstrap_switch_model,
    request_model_switch,
)
from kohakuterrarium.core.provider_tools import (
    sync_provider_tools as sync_provider_tools,
)
from kohakuterrarium.llm.profiles import (
    profile_to_identifier,
    resolve_controller_llm,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class AgentModelMixin:
    """Switch an agent's model and report its canonical identifier."""

    # These annotations describe state supplied by ``Agent.__init__``. A stub
    # for ``_build_compact_llm`` would shadow ``AgentCompactMixin`` through MRO.
    llm: Any
    controller: Any
    compact_manager: Any
    output_router: Any
    config: Any
    _llm_selector: str | None
    _llm_identifier: str

    def switch_model(self, profile_name: str) -> str:
        """Switch before the agent is live.

        Once ``start`` has run, callers must await :meth:`switch_model_async`.
        A started or stopped agent is rejected before any resolve or mutation.
        """
        return bootstrap_switch_model(self, profile_name)

    async def switch_model_async(self, profile_name: str) -> str:
        """Confirm durable model intent, then publish the prepared bindings."""
        return await request_model_switch(self, profile_name)

    def llm_identifier(self) -> str:
        """Return the canonical ``provider/name[@variations]`` for the
        currently-bound LLM profile.

        Populated on every :meth:`switch_model` call. On startup the
        agent goes through the bootstrap path instead, so first access
        resolves from config and caches the result — the banner,
        ``/model`` command, and frontend pill all see the identifier
        the user can paste back into ``/model``.
        """
        if self._llm_identifier:
            return self._llm_identifier
        controller_data: dict[str, Any] = {
            "llm": self._llm_selector or self.config.llm_profile or None,
            "model": self.config.model,
            "provider": self.config.provider,
            "variation_selections": dict(self.config.variation_selections or {}),
        }
        controller_data = {k: v for k, v in controller_data.items() if v}
        profile = resolve_controller_llm(controller_data)
        if profile is None:
            self._llm_identifier = getattr(self.llm, "model", "")
        else:
            self._llm_identifier = profile_to_identifier(profile)
        return self._llm_identifier
