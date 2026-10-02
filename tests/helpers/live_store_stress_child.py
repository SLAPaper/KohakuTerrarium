"""Child process: read a session file while a live store in this process writes it.

Runs in its own process because the failure it guards against is a native crash.
Prints ``LIVE-STORE-OK <events read>`` when every read succeeded.
"""

import sys
import tempfile
import threading
import time
from itertools import islice
from pathlib import Path

from kohakuterrarium.session.readonly_view import SessionReadView
from kohakuterrarium.session.store import SessionStore

READS = 200
PAGE = 500


def main(root: Path) -> None:
    path = root / "live.kohakutr"
    store = SessionStore(str(path))
    store.init_meta("live", "agent", "cfg", "/tmp", ["a"])
    stop = threading.Event()

    errors = []

    def append(i):
        store.meta["last_active"] = i
        store.append_event("a", "user_input", {"content": f"message {i}"})

    def write() -> None:
        i = 0
        try:
            while not stop.is_set():
                store.submit(append, i).result(timeout=5)
                i += 1
                time.sleep(0.0005)
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    seen = 0
    try:
        for _ in range(READS):
            with SessionReadView(path) as view:
                meta = view.load_meta()
                assert meta["agents"] == ["a"], meta
                events = view.items("events", prefix="a:e")
                seen = max(seen, sum(1 for _ in islice(events, PAGE)))
    finally:
        stop.set()
        writer.join()
        store.close()
    assert not errors, errors
    assert seen > 0
    print("LIVE-STORE-OK", seen)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()))
