# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Set

import os
import pkg_resources

from bag.design.module import Module
<<<<<<< HEAD:bag/interface/templates/Module.pytemp

if TYPE_CHECKING:
    from bag.design.database import ModuleDB
=======
>>>>>>> master:bag/interface/templates/Module.pyi


# noinspection PyPep8Naming
class {{ lib_name }}__{{ cell_name }}(Module):
    """Module for library {{ lib_name }} cell {{ cell_name }}.

    Fill in high level description here.
    """
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             '{{ cell_name }}.yaml'))


<<<<<<< HEAD:bag/interface/templates/Module.pytemp
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             '{{ cell_name }}.yaml'))

    def __init__(self, database, lib_name, params, used_names, **kwargs):
        # type: (ModuleDB, str, Dict[str, Any], Set[str], **Any) -> None
        Module.__init__(self, self.yaml_file, database, lib_name, params, used_names, **kwargs)
=======
    def __init__(self, database, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, self.yaml_file, parent=parent, prj=prj, **kwargs)
>>>>>>> master:bag/interface/templates/Module.pyi

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
        )

    def design(self):
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        pass
