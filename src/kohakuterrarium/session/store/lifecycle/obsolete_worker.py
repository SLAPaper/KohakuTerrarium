"""Standalone DDL cleanup: never open Python SQLite beside live KohakuVault."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

_TRIGGERS = (
    {"kt_event_insert", "kt_event_update", "kt_event_delete"}
    | {
        f"kt_{table}_{action}"
        for table in ("state", "conversation")
        for action in ("insert", "update", "delete")
    }
    | {"kt_event_sequence_insert", "kt_event_sequence_update"}
)
_TABLES = ("kt_event_revisions", "kt_value_revisions", "kt_event_sequences")


def retire_guards(path):
    """Remove exact native cache objects; never interpret State values."""
    with (
        closing(
            sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True)
        ) as connection,
        connection,
    ):
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('trigger','table')"
            )
        }
        for trigger in _TRIGGERS & names:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        for table in _TABLES:
            if table in names:
                connection.execute(f'DROP TABLE "{table}"')


if __name__ == "__main__":
    retire_guards(sys.argv[1])
