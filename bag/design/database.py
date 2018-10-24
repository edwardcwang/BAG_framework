# -*- coding: utf-8 -*-

"""This module defines the design database class.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Any, Type, Set, Sequence

from ..util.cache import MasterDB

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.core import TechInfo
    from .module import Module


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is responsible for keeping track of module libraries and
    creating new modules.

    Parameters
    ----------
    lib_defs : str
        path to the design library definition file.
    tech_info : TechInfo
        the TechInfo instance.
    sch_exc_libs : List[str]
        list of libraries that are excluded from import.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    lib_path : str
        path to create generated library in.
    """

    def __init__(self, lib_defs, tech_info, sch_exc_libs, prj=None, name_prefix='',
                 name_suffix='', lib_path=''):
        # type: (str, TechInfo, List[str], Optional[BagProject], str, str, str) -> None
        MasterDB.__init__(self, '', lib_defs=lib_defs, name_prefix=name_prefix,
                          name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info
        self._exc_libs = set(sch_exc_libs)
        self.lib_path = lib_path

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[Module], str, Dict[str, Any], Set[str], **Any) -> Module
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[Module]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs : Any
            optional arguments for the generator.

        Returns
        -------
        master : Module
            the non-finalized generated instance.
        """
        kwargs = kwargs.copy()
        kwargs['lib_name'] = lib_name
        kwargs['params'] = params
        kwargs['used_names'] = used_cell_names
        # noinspection PyTypeChecker
        return gen_cls(self, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False, output='', **kwargs):
        # type: (str, Sequence[Any], bool, str, Any) -> None
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        if output == 'schematic':
            self._prj.instantiate_schematic(lib_name, content_list, lib_path=self.lib_path)
        elif output == 'netlist':
            self._prj.instantiate_netlist(lib_name, content_list, **kwargs)
        else:
            raise ValueError('Unsupported output type: {}'.format(output))

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def is_lib_excluded(self, lib_name):
        # type: (str) -> bool
        """Returns true if the given schematic library does not contain generators.

        Parameters
        ----------
        lib_name : str
            library name

        Returns
        -------
        is_excluded : bool
            True if given library is excluded.
        """
        return lib_name in self._exc_libs
