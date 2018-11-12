# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

import os
import pkg_resources

from bag.design.module import {{ module_name }}
from bag.design.database import ModuleDB


# noinspection PyPep8Naming
class {{ lib_name }}__{{ cell_name }}({{ module_name }}):
    """design module for {{ lib_name }}__{{ cell_name }}.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             '{{ cell_name }}.yaml'))

    def __init__(self, database, lib_name, params, used_names, **kwargs):
        # type: (ModuleDB, str, Dict[str, Any], Set[str], **Any) -> None
        {{ module_name }}.__init__(self, self.yaml_file, database, lib_name, params, used_names, **kwargs)
