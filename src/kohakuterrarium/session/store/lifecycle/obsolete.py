"""Remove native guards from the retired cold-resume cache."""

from contextlib import closing

from kohakuterrarium.session.store.events.cursor import connect

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


def remove_obsolete_resume_guards(store) -> None:
    """Remove only the retired cache's native triggers and revision tables."""
    if store._readonly:
        return
    with closing(connect(store, writable=True)) as connection, connection:
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
