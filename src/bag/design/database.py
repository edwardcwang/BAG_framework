# -*- coding: utf-8 -*-

"""This module defines the design database class.
"""

from typing import TYPE_CHECKING, TypeVar, Dict, Optional, Any, Type, Sequence

import time
import importlib

from ..util.cache import MasterDB, DesignOutput

try:
    import pybag
except ImportError:
    pybag = None

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.core import TechInfo
    from .module import Module

    ModuleType = TypeVar('ModuleType', bound=Module)


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is responsible for keeping track of module libraries and
    creating new modules.

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
        MasterDB.__init__(self, lib_name, name_prefix=name_prefix, name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info

    def create_masters_in_db(self, output, lib_name, content_list, debug=False, **kwargs):
        # type: (DesignOutput, str, Sequence[Any], bool, **Any) -> None
        start = time.time()
        if output is DesignOutput.SCHEMATIC:
            if self._prj is None:
                raise ValueError('BagProject is not defined.')

            self._prj.instantiate_schematic(lib_name, content_list)
        elif output is DesignOutput.NETLIST:
            if pybag is None:
                raise ValueError('Cannot find pybag C extension; check your LD_LIBRARY_PATH.')

            fname = kwargs['fname']
            cell_map = kwargs['cell_map']
            inc_list = kwargs['includes']
            fmt = kwargs.get('format', 'cdl')
            flat = kwargs.get('flat', True)
            shell = kwargs.get('shell', False)

            pybag.implement_netlist(content_list, cell_map, inc_list, fmt, fname,
                                    flat, shell)
        elif output is DesignOutput.YAML:
            if pybag is None:
                raise ValueError('Cannot find pybag C extension; check your LD_LIBRARY_PATH.')

            fname = kwargs['fname']

            pybag.implement_yaml(content_list, fname)
        else:
            raise ValueError('Unsupported output type: {}'.format(output))
        end = time.time()

        if debug:
            print('schematic instantiation took %.4g seconds' % (end - start))

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
