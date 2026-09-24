"""Allowlisted source facts for one assistant response attempt."""

from typing import Any
from uuid import uuid4

GENERATION_SOURCE_KEYS = (
    "llm_identifier",
    "backend_key",
    "backend_type",
    "provider_name",
    "model",
    "tool_format",
)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def provider_source(provider: Any, tool_format: str | None) -> dict[str, Any]:
    """Snapshot declared provider facts without inferring profile aliases."""
    model = _string(getattr(provider, "model", None))
    if model is None:
        model = _string(getattr(getattr(provider, "config", None), "model", None))
    return {
        "llm_identifier": _string(getattr(provider, "_llm_identifier", None)),
        "backend_key": _string(getattr(provider, "backend_key", None)),
        "backend_type": _string(getattr(provider, "backend_type", None)),
        "provider_name": _string(getattr(provider, "provider_name", None)),
        "model": model,
        "tool_format": _string(tool_format),
    }


def provider_identity(provider: Any) -> tuple[Any, ...]:
    """Identity compared before chat so a swapped provider cannot keep the stamp."""
    source = provider_source(provider, None)
    return (
        id(provider),
        source["llm_identifier"],
        source["backend_key"],
        source["backend_type"],
        source["provider_name"],
        source["model"],
    )


def generation_payload(provider: Any, tool_format: str | None) -> dict[str, Any]:
    """Create one response attempt with an allowlisted source snapshot."""
    return {
        "schema": 1,
        "generation_id": str(uuid4()),
        "source": provider_source(provider, tool_format),
    }
