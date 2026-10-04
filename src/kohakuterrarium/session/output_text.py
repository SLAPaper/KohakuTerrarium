"""Bound streamed-text persistence and orphan recovery for session outputs."""

from copy import deepcopy
from uuid import uuid4

from kohakuterrarium.errors import SessionError
from kohakuterrarium.session.text_recovery import build_record, recover_record


def materialize_segment(store, namespace, segment):
    """Resolve a detached streamed segment against its durable declaration."""
    return build_record(
        store,
        namespace,
        segment["text"],
        segment["generation"],
        segment["turn_index"],
        segment["branch_id"],
        segment["parent_branch_path"],
        segment["segment_id"],
        segment["chunk_seq"],
        ephemeral_proof=segment.get("declaration_proof"),
    )


def persist_segment(store, namespace, segment, finalize):
    """Use the same receipt for normal finalization and crash recovery."""
    return recover_record(
        store,
        namespace,
        materialize_segment(store, namespace, segment),
        finalize=finalize,
        ephemeral_proof=segment.get("declaration_proof"),
    )


class SessionTextMixin:
    """Coalesce source-bound text without performing store I/O on the producer."""

    def _check_text_durability(self):
        error = getattr(self, "_text_durability_error", None)
        if error is not None:
            raise SessionError(
                "Text persistence is unresolved; cold recovery required"
            ) from error
        for future in self._critical_writes:
            if future.done():
                try:
                    future.result()
                except Exception as exc:
                    self._text_durability_error = exc
                    raise

    def _ingest_text(self, chunk):
        self._recover_open_text()
        self._check_text_durability()
        if not chunk:
            return
        if not self._open_text._buf and self._generation_payload is not None:
            turn, branch, path = self._generation_coordinates
            self._open_text.bind_record(
                {
                    "segment_id": str(uuid4()),
                    "generation": deepcopy(self._generation_payload),
                    "declaration_proof": deepcopy(
                        getattr(self, "_generation_proof", None)
                    ),
                    "turn_index": turn,
                    "branch_id": branch,
                    "parent_branch_path": deepcopy(path),
                    "chunk_seq": self._chunk_seq,
                }
            )
        bound = self._open_text._bound_record is not None
        receipt = self._open_text.append(chunk)
        if bound and receipt is not None:
            self._critical_writes.add(receipt)

    def _recover_open_text(self):
        if self._recovered_open_text:
            return
        self._recovered_open_text = True
        record = self._prepared_recovery
        self._prepared_recovery = None
        receipt = self._submit_store_write(
            recover_record, self._store, self._event_key_prefix, record
        )
        self._critical_writes.add(receipt)

    def _flush_text_segment(self, *, finalize=None):
        self._check_text_durability()
        text, segment = self._open_text.take_bound()
        if not text:
            return
        if segment is None:
            recorded = self._append_text_chunk(text, finalize=finalize)
            self._open_text.clear(after=recorded)
        else:
            recorded = self._submit_store_write(
                persist_segment, self._store, self._event_key_prefix, segment, finalize
            )
            self._critical_writes.add(recorded)
            self._chunk_seq += 1

    def _append_text_chunk(self, content, *, finalize=None):
        data = {"content": content, "chunk_seq": self._chunk_seq}
        if finalize:
            data["finalize"] = finalize
        self._chunk_seq += 1
        return self._append_event("text_chunk", data)
