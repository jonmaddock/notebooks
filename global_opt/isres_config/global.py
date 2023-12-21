from process.main import SingleRun
from pathlib import Path

input_path = Path("generic_demo_IN.DAT")

single_run = SingleRun(str(input_path), solver_config="solver.toml")
single_run.run()
