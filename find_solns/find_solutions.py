import process
from pathlib import Path
# from process.io import plot_solutions as ps
from process.io.find_solutions import find_solutions

conf_path = Path("find_solutions.conf")
find_solutions(conf_path, solver_config="solver.toml")
