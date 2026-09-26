"""Generation provenance projection shared by durable output and replay."""

from copy import deepcopy
from typing import Any

from kohakuterrarium.llm.generation import GENERATION_SOURCE_KEYS

SOURCE_KEYS = GENERATION_SOURCE_KEYS
ASSISTANT_EVENTS = {
    "text",
    "text_chunk",
    "assistant_reasoning",
    "assistant_image",
    "assistant_tool_calls",
}


def generation_payload(value: dict) -> dict:
    """Detach only origin facts; never persist arbitrary provider configuration."""
    source = value.get("source")
    source = source if isinstance(source, dict) else {}
    return {
        "schema": deepcopy(value.get("schema")),
        "generation_id": deepcopy(value.get("generation_id")),
        "source": {
            key: source.get(key) if isinstance(source.get(key), str) else None
            for key in SOURCE_KEYS
        },
    }


def recovered_generation(store: Any, prefix: str) -> dict:
    """Recover the origin preceding the orphan slot, not the resumed provider."""
    for event in reversed(store.get_events(prefix)):
        if event.get("type") in {
            "processing_end",
            "processing_start",
            "user_message",
            "user_input",
        }:
            break
        if event.get("type") == "assistant_generation_start":
            generation_id = event.get("generation_id")
            return {"generation_id": generation_id} if generation_id is not None else {}
    return {}


class GenerationReplay:
    """Keep grouping internal even when provider messages omit metadata."""

    def __init__(self, events: list[dict], live_ids: set[int], metadata: bool):
        self.origins = {
            event["generation_id"]: generation_payload(event)
            for event in events
            if event.get("type") == "assistant_generation_start"
            and isinstance(event.get("generation_id"), str)
            and (
                not isinstance(event.get("event_id"), int)
                or event["event_id"] in live_ids
            )
        }
        self.metadata = metadata
        self.groups: dict[int, Any] = {}
        self.boundary = 0

    def key(self, event: dict) -> Any:
        return event.get("generation_id", ("legacy", self.boundary))

    def attach(self, message: dict, key: Any) -> dict:
        self.groups[id(message)] = key
        if self.metadata and isinstance(key, str) and key in self.origins:
            message.setdefault("metadata", {})["generation"] = deepcopy(
                self.origins[key]
            )
        return message

    def matches(self, message: dict, event: dict) -> bool:
        return self.groups.get(id(message), ("legacy", self.boundary)) == self.key(
            event
        )
