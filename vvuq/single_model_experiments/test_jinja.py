from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import numpy as np

# Render a simple template
cwd = Path(__file__).parent
environment = Environment(loader=FileSystemLoader(cwd))
template = environment.get_template("test.jinja")
content = template.render(what="world")

output_path = cwd / "output.txt"
with open(output_path, mode="w+") as file:
    file.write(content)

# # Render an input file template
# boundu = np.zeros(20)
# boundl = np.ones(20)
# fimp = np.ones(20)
# boundu[1] = 10.0
# params = {"boundu": boundu, "boundl": boundl, "fimp": fimp}
# template = environment.get_template("baseline_2018.jinja")
# content = template.render(params)

# rendered_in = cwd / "rendered_IN.DAT"
# with open(rendered_in, "w+") as file:
#     file.write(content)
