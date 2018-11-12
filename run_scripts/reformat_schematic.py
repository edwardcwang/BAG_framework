# -*- coding: utf-8 -*-

"""Reformat BAG2 schematic generator files to BAG3.

NOTE: This is an alpha script, please double check your results.
"""

from typing import Tuple

import os
import glob
import argparse
from itertools import islice

repl_header = r'''# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB


# noinspection PyPep8Naming
class {lib_name}__{cell_name}(Module):
    """Module for library {lib_name} cell {cell_name}.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             '{cell_name}.yaml'))

    def __init__(self, database, lib_name, params, used_names, **kwargs):
        # type: (ModuleDB, str, Dict[str, Any], Set[str], **Any) -> None
        Module.__init__(self, self.yaml_file, database, lib_name, params, used_names, **kwargs)
'''


def parse_options() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description='Convert BAG3 schematic generators to BAG3.')
    parser.add_argument('root_path', type=str,
                        help='path to schematic generator files.')
    parser.add_argument('lib_name', type=str,
                        help='schematic library name.')

    args = parser.parse_args()
    return args.root_path, args.lib_name


def main() -> None:
    root_path, lib_name = parse_options()
    os.chdir(root_path)
    for fname in glob.iglob('*.py'):
        if fname == '__init__.py':
            continue

        cell_name = fname[:-3]
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_header = repl_header.format(lib_name=lib_name, cell_name=cell_name)
        with open(fname, 'w') as f:
            f.write(new_header)
            start_write = False
            for l in lines:
                if start_write:
                    f.write(l)
                else:
                    tmp = l.lstrip()
                    if '.__init__(' in tmp:
                        start_write = True


if __name__ == '__main__':
    main()
