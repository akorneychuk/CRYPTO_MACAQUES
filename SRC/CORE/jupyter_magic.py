import os
import sys

from SRC.CORE._CONSTANTS import project_root_dir

sys.path.insert(0, [path for path in sys.path if 'SRC' in path][0].replace("/SRC", ""))

import ipynbname
from IPython.core.magic import (Magics, magics_class, line_magic)
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from SRC.CORE.debug_utils import printmd, get_current_notebook_name
from SRC.LIBRARIES.new_utils import run_notebook_cell


@magics_class
class CustomMagics(Magics):    
    @line_magic
    def set_env(self, line):
        var, value = line.split('=', 1)

        if var in os.environ:
            if os.environ[var] == value:
                print(f"env: {var}={os.environ[var]}")
            else:
                printmd(f"ENV: {var}={value} >> ***{os.environ[var]}***")
        else:
            os.environ[var] = value
            print(f"env: {var}={os.environ[var]}")

    @magic_arguments()
    @argument('param1', type=int, help="Cell num")
    @argument('param2', type=str, nargs='?', default=None, help="Notebook name")
    @line_magic
    def run_cell_before(self, line):
        args = parse_argstring(self.run_cell_before, line)
        cell_num = args.param1
        notebook_name = args.param2 if args.param2 is not None else get_current_notebook_name()
        notebook_path = ipynbname.path()
        rel_path = str(notebook_path).replace(f"{project_root_dir()}/SRC/", "").replace(f"/{notebook_name}.ipynb", "")

        run_notebook_cell(int(cell_num), name=notebook_name, rel_path=rel_path)