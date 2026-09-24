"""Subprocess probe for the public worker deadline boundary; no CFD claims."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


def invoke(tool, inputs, context):
    case = Path(inputs["case_dir"])
    case.mkdir(parents=True, exist_ok=True)
    (case / "worker.pid").write_text(str(os.getpid()))
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(case / 'child.pid')!r}).write_text(str(os.getpid())); "
        "time.sleep(2); "
        f"pathlib.Path({str(case / 'late-artifact.txt')!r}).write_text('orphan wrote after deadline')"
    )
    subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    time.sleep(10)
    raise AssertionError("outer deadline should terminate this worker")
