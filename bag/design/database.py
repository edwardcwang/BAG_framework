# -*- coding: utf-8 -*-

"""This module defines the design database class.
"""

from typing import TYPE_CHECKING, TypeVar, Dict, Optional, Any, Type, Set, Sequence

import time

from ..util.cache import MasterDB

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
    **kwargs :
        additional arguments.
    """

    # noinspection PyUnusedLocal
    def __init__(self, tech_info, lib_name, prj=None, name_prefix='',
                 name_suffix='', **kwargs):
        # type: (TechInfo, str, Optional[BagProject], str, str, **Any) -> None
        MasterDB.__init__(self, lib_name, name_prefix=name_prefix,
                          name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[ModuleType], str, Dict[str, Any], Set[str], **Any) -> ModuleType
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[ModuleType]
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
        master : ModuleType
            the non-finalized generated instance.
        """
        return gen_cls(self, lib_name, params, used_cell_names, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False, output='', **kwargs):
        # type: (str, Sequence[Any], bool, str, Any) -> None
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        start = time.time()
        if output == 'schematic':
            self._prj.instantiate_schematic(lib_name, content_list)
        elif output == 'netlist':
            self._prj.instantiate_netlist(lib_name, content_list, **kwargs)
        else:
            raise ValueError('Unsupported output type: {}'.format(output))
        end = time.time()

        if debug:
            print('layout instantiation took %.4g seconds' % (end - start))

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def new_schematic(self, gen_cls, params=None, **kwargs):
        # type: (Type[ModuleType], Optional[Dict[str, Any]], **Any) -> ModuleType
        """Create a new template.

        Parameters
        ----------
        gen_cls : Type[ModuleType]
            the generator class.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.
        **kwargs : Any
            optional template parameters.

        Returns
        -------
        sch : ModuleType
            the new schematic instance.
        """
        return self.new_master(gen_cls, params=params, **kwargs)

    def instantiate_schematic(self, prj, design, top_cell_name=None, debug=False, rename_dict=None):
        # type: (BagProject, Module, Optional[str], bool, Optional[Dict[str, str]]) -> None
        """Instantiate the layout of the given :class:`~bag.layout.template.TemplateBase`.

        Parameters
        ----------
        prj : BagProject
            the BagProject instance.
        design : Module
            the schematic to instantiate.
        top_cell_name : Optional[str]
            name of the top level cell.  If None, a default name is used.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        """
        self.batch_schematic(prj, [design], [top_cell_name], debug=debug, rename_dict=rename_dict)

    def batch_schematic(self,
                        prj,  # type: BagProject
                        design_list,  # type: Sequence[Module]
                        name_list=None,  # type: Optional[Sequence[Optional[str]]]
                        lib_name='',  # type: str
                        debug=False,  # type: bool
                        rename_dict=None,  # type: Optional[Dict[str, str]]
                        ):
        # type: (...) -> None
        """Instantiate all given templates.

        Parameters
        ----------
        prj : BagProject
            the BagProject instance.
        design_list : Sequence[Module]
            list of schematics to instantiate.
        name_list : Optional[Sequence[Optional[str]]]
            list of schematic names.  If not given, default names will be used.
        lib_name : str
            Library to create the masters in.  If empty or None, use default library.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        """
        self._prj = prj
        self.instantiate_masters(design_list, name_list=name_list, lib_name=lib_name,
                                 debug=debug, rename_dict=rename_dict)
