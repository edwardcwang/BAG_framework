# -*- coding: utf-8 -*-

"""This module defines the design database class.
"""

from typing import TYPE_CHECKING, TypeVar, Dict, Optional, Any, Sequence, Type

import importlib

from pybag.enum import DesignOutput

from ..util.cache import MasterDB

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.core import TechInfo
    from .module import Module

    ModuleType = TypeVar('ModuleType', bound=Module)


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is a subclass of MasterDB that defines some extra properties/function
    aliases to make creating schematics easier.

    Parameters
    ----------
    tech_info : TechInfo
        the TechInfo instance.
    lib_name : str
        the cadence library to put all generated templates in.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated schematic name prefix.
    name_suffix : str
        generated schematic name suffix.
    """

    def __init__(self, tech_info, lib_name, prj=None, name_prefix='', name_suffix=''):
        # type: (TechInfo, str, Optional[BagProject], str, str) -> None
        MasterDB.__init__(self, lib_name, prj=prj, name_prefix=name_prefix, name_suffix=name_suffix)

        self._tech_info = tech_info

    @classmethod
    def get_schematic_class(cls, lib_name, cell_name):
        # type: (str, str) -> Type[ModuleType]
        """Get the Python class object for the given schematic.

        Parameters
        ----------
        lib_name : str
            schematic library name.
        cell_name : str
            schematic cell name.

        Returns
        -------
        sch_cls : Type[ModuleType]
            the schematic class.
        """
        module_name = lib_name + '.schematic.' + cell_name
        try:
            sch_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError('Cannot find Python module {} for schematic generator {}__{}.  '
                              'Is it on your PYTHONPATH?'.format(module_name, lib_name, cell_name))
        cls_name = lib_name + '__' + cell_name
        if not hasattr(sch_module, cls_name):
            raise ImportError('Cannot find schematic generator class {} '
                              'in module {}'.format(cls_name, module_name))
        return getattr(sch_module, cls_name)

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def instantiate_schematic(self, design, top_cell_name=None, output=DesignOutput.SCHEMATIC,
                              **kwargs):
        # type: (Module, Optional[str], DesignOutput, **Any) -> None
        """Alias for instantiate_master(), with default output type of SCHEMATIC.
        """
        self.instantiate_master(output, design, top_cell_name, **kwargs)

    def batch_schematic(self,
                        design_list,  # type: Sequence[Module]
                        name_list=None,  # type: Optional[Sequence[Optional[str]]]
                        lib_name='',  # type: str
                        debug=False,  # type: bool
                        rename_dict=None,  # type: Optional[Dict[str, str]]
                        output=DesignOutput.SCHEMATIC,  # type: DesignOutput
                        **kwargs  # type: Any
                        ):
        # type: (...) -> None
        """Alias for batch_output(), with default output type of SCHEMATIC.
        """
        self.batch_output(output, design_list, name_list=name_list, lib_name=lib_name,
                          debug=debug, rename_dict=rename_dict, **kwargs)
