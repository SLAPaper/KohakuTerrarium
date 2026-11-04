"""Remove native guards from the retired cold-resume cache."""

import subprocess
from pathlib import Path

from kohakuterrarium.utils.interpreter import python_command
from kohakuterrarium.utils.mobile_sandbox import is_mobile_profile


def remove_obsolete_resume_guards(store) -> None:
    """Remove only the retired cache's native triggers and revision tables."""
    if store._readonly or is_mobile_profile():
        return
    subprocess.run(
        [
            *python_command(),
            "-I",
            str(Path(__file__).with_name("obsolete_worker.py")),
            str(Path(store.path).resolve()),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
