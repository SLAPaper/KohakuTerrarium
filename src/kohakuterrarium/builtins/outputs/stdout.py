"""Plain stdout output with streaming and resume-history support."""

import sys
from typing import TextIO

from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.builtins.outputs.stdout_history import (
    _group_resume_events as _group_resume_events,
    _render_resume_text,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def _write_safe(stream: TextIO, text: str) -> None:
    """Write text without letting console encoding mismatches crash streaming."""
    try:
        stream.write(text)
        return
    except UnicodeEncodeError:
        pass

    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, errors="replace").decode(encoding))


class StdoutOutput(BaseOutputModule):
    """Write complete and streamed assistant output to stdout."""

    def __init__(
        self,
        *,
        prefix: str = "",
        suffix: str = "\n",
        stream_suffix: str = "",
        flush_on_stream: bool = True,
    ):
        """Configure output framing and streaming flush behavior."""
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix
        self.stream_suffix = stream_suffix
        self.flush_on_stream = flush_on_stream
        self._streaming = False
        self._has_output = False

    async def _on_start(self) -> None:
        """Log that stdout output is ready."""
        logger.debug("Stdout output started")

    async def _on_stop(self) -> None:
        """Log that stdout output has stopped."""
        logger.debug("Stdout output stopped")

    async def write(self, content: str) -> None:
        """Write one complete output message."""
        if not content:
            return

        output = ""
        if not self._has_output and self.prefix:
            output += self.prefix

        output += content + self.suffix

        _write_safe(sys.stdout, output)
        sys.stdout.flush()

        self._has_output = True
        self._streaming = False

    async def write_stream(self, chunk: str) -> None:
        """Write one streaming chunk without closing the message."""
        if not chunk:
            return

        if not self._streaming and not self._has_output and self.prefix:
            _write_safe(sys.stdout, self.prefix)

        _write_safe(sys.stdout, chunk + self.stream_suffix)

        if self.flush_on_stream:
            sys.stdout.flush()

        self._streaming = True
        self._has_output = True

    async def flush(self) -> None:
        """Flush stdout and add suffix if streaming."""
        if self._streaming:
            _write_safe(sys.stdout, self.suffix)
        sys.stdout.flush()
        self._streaming = False

    def reset(self) -> None:
        """Reset prefix and streaming state for the next turn."""
        self._has_output = False
        self._streaming = False

    def on_activity_with_metadata(
        self, activity_type: str, detail: str, metadata: dict
    ) -> None:
        """Present side-conversation notices independently of assistant framing."""
        if activity_type not in ("btw_started", "btw_result"):
            return
        job_id = metadata.get("job_id", "")
        status = "error" if metadata.get("error") else "side conversation"
        text = detail or str(metadata.get("error") or "")
        _write_safe(sys.stdout, f"\n[btw {job_id} · {status}]\n{text}\n[/btw]\n")
        sys.stdout.flush()

    async def on_resume(self, events: list[dict]) -> None:
        """Render bounded turn previews from legacy replay events."""
        await self._deliver_history_text(_render_resume_text(events))

    async def prepare_resume(self, history):
        """Prepare bounded history text on the store worker."""
        callback = getattr(self.on_resume, "__func__", self.on_resume)
        if callback is not StdoutOutput.on_resume:
            return await super().prepare_resume(history)
        return await history.projection(_render_resume_text, normalize=False)

    async def deliver_resume(self, prepared) -> None:
        """Deliver prepared text or retain a subclass's legacy callback."""
        if isinstance(prepared, str):
            await self._deliver_history_text(prepared)
        else:
            await self.on_resume(prepared)

    async def _deliver_history_text(self, text: str) -> None:
        if text:
            _write_safe(sys.stdout, text)
            sys.stdout.flush()


class PrefixedStdoutOutput(StdoutOutput):
    """Stdout output with a configurable speaker prefix."""

    def __init__(
        self,
        prefix: str = "Assistant: ",
        **kwargs,
    ):
        super().__init__(prefix=prefix, **kwargs)

    async def write_with_prefix(self, content: str, prefix: str | None = None) -> None:
        """Write one message with an optional temporary prefix."""
        old_prefix = self.prefix
        if prefix is not None:
            self.prefix = prefix

        await self.write(content)

        self.prefix = old_prefix
