"""Using SLSQP, perform a VaryRun for a given scenario."""

from process.main import VaryRun
from pathlib import Path
import process
from shutil import copy
import sys

scenario = sys.argv[1]
PROCESS_DIR = Path(process.__file__).parent.parent
INPUT_PATH = PROCESS_DIR / "tests/regression/scenarios/" / scenario / "IN.DAT"
CONF_PATH = Path("run_process.conf")

copy(INPUT_PATH, Path("ref_IN.DAT"))

vary_run = VaryRun(str(CONF_PATH), solver="scipy")
