"""No module may use a name it never defines or imports.

Four production 500s in two days came from this one shape — a name that resolves fine at
import and raises on a request:

  * `Segment.faceswap_enabled` in list_jobs, after the column was dropped  -> GET /jobs
  * `_resolve_trigger` deleted as collateral, call site left behind        -> POST segments
  * `Worker` used in get_stats but never imported                          -> GET /stats
  * the same `Worker`, a second time, because the "fix" was never committed

Ruff's F821 catches all of them. Two things about HOW it is wired here, both learned the
expensive way:

NO skipif. The first version of this file skipped when ruff was absent, ruff was not a
declared dependency, and it therefore skipped on CI for the very commit it was written to
guard — which shipped `NameError: name 'Worker' is not defined` to production and took the
dashboard down for an hour. A guard that excuses itself when its tool is missing is not a
guard, so a missing ruff is a FAILURE here.

Run from pytest, not a CI lint step. Coverage that depends on a workflow file staying
configured can disappear without a test going red.
"""

import shutil
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).parent.parent / "app"


def _ruff() -> str | None:
    beside_python = Path(sys.executable).parent / "ruff"
    return shutil.which("ruff") or (str(beside_python) if beside_python.exists() else None)


def test_no_undefined_names_anywhere_in_app():
    assert _ruff() is not None, (
        "ruff is not installed, so undefined names are not being checked at all. It is in "
        "requirements-dev.txt and CI installs it — a missing ruff means the environment is "
        "wrong, not that this check is optional."
    )
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--output-format", "concise", str(APP)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "undefined name(s) — these raise NameError at request time, not at import, so the "
        "module loads and only the endpoint breaks:\n" + (result.stdout or result.stderr)
    )
