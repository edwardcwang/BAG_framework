# -*- coding: utf-8 -*-

"""This module defines DbAccess, the base class for CAD database manipulation.
"""

from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Sequence, Any, Union

import os
import abc
import traceback

from jinja2 import Template

import bag.io
from ..verification import make_checker

if TYPE_CHECKING:
    from ..verification import Checker
    from ..design.module import ModuleDB


def dict_to_item_list(table):
    """Given a Python dictionary, convert to sorted item list.

    Parameters
    ----------
    table : dict[str, any]
        a Python dictionary where the keys are strings.

    Returns
    -------
    assoc_list : list[(str, str)]
        the sorted item list representation of the given dictionary.
    """
    return [[key, table[key]] for key in sorted(table.keys())]


def format_inst_map(inst_map):
    """Given instance map from DesignModule, format it for database changes.

    Parameters
    ----------
    inst_map : Dict[str, Any]
        the instance map created by DesignModule.

    Returns
    -------
    ans : List[(str, Any)]
        the database change instance map.
    """
    ans = []
    for old_inst_name, rinst_list in inst_map.items():
        new_rinst_list = [dict(name=rinst['name'],
                               lib_name=rinst['lib_name'],
                               cell_name=rinst['cell_name'],
                               params=dict_to_item_list(rinst['params']),
                               term_mapping=dict_to_item_list(rinst['term_mapping']),
                               ) for rinst in rinst_list]
        ans.append([old_inst_name, new_rinst_list])
    return ans


class DbAccess(object, metaclass=abc.ABCMeta):
    """A class that manipulates the CAD database.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for DbAccess.
    db_config : Dict[str, Any]
        the database configuration dictionary.
    """

    def __init__(self, tmp_dir, db_config):
        # type: (str, Dict[str, Any]) -> None
        """Create a new DbAccess object.
        """
        self.tmp_dir = bag.io.make_temp_dir('dbTmp', parent_dir=tmp_dir)
        self.db_config = db_config
        self.exc_libs = set(db_config['schematic']['exclude_libraries'])
        # noinspection PyBroadException
        try:
            check_kwargs = self.db_config['checker'].copy()
            check_kwargs['tmp_dir'] = self.tmp_dir
            self.checker = make_checker(**check_kwargs)  # type: Optional[Checker]
        except Exception:
            stack_trace = traceback.format_exc()
            print('*WARNING* error creating Checker:\n%s' % stack_trace)
            print('*WARNING* LVS/RCX will be disabled.')
            self.checker = None  # type: Optional[Checker]

        # set default lib path
        self._default_lib_path = self.get_default_lib_path(db_config)

    @classmethod
    def get_default_lib_path(cls, db_config):
        # type: (Dict[str, Any]) -> str
        lib_path_fallback = os.path.abspath('.')
        default_lib_path = os.path.abspath(db_config.get('default_lib_path', lib_path_fallback))
        if not os.path.isdir(default_lib_path):
            default_lib_path = lib_path_fallback

        return default_lib_path

    @classmethod
    def get_python_template(cls, lib_name, cell_name, primitive_table):
        # type: (str, str, Dict[str, str]) -> str
        """Returns the default Python Module template for the given schematic.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        primitive_table : Dict[str, str]
            a dictionary from primitive cell name to module template file name.

        Returns
        -------
        template : str
            the default Python Module template.
        """
        param_dict = dict(lib_name=lib_name, cell_name=cell_name)
        if lib_name == 'BAG_prim':
            if cell_name in primitive_table:
                # load template from user defined file
                content = bag.io.read_file(primitive_table[cell_name])
            else:
                if cell_name.startswith('nmos4_') or cell_name.startswith('pmos4_'):
                    # transistor template
                    module_name = 'MosModuleBase'
                elif cell_name == 'res_ideal':
                    # ideal resistor template
                    module_name = 'ResIdealModuleBase'
                elif cell_name == 'res_metal':
                    module_name = 'ResMetalModule'
                elif cell_name == 'cap_ideal':
                    # ideal capacitor template
                    module_name = 'CapIdealModuleBase'
                elif cell_name.startswith('res_'):
                    # physical resistor template
                    module_name = 'ResPhysicalModuleBase'
                else:
                    raise Exception('Unknown primitive cell: %s' % cell_name)

                content = bag.io.read_resource(__name__,
                                               os.path.join('templates', 'PrimModule.pytemp'))
                param_dict['module_name'] = module_name
        else:
            # use default empty template.
            content = bag.io.read_resource(__name__, os.path.join('templates', 'Module.pytemp'))

        return Template(content).render(**param_dict)

    @property
    def default_lib_path(self):
        # type: () -> str
        """Returns the default directory to create new libraries in.

        Returns
        -------
        lib_path : str
            directory to create new libraries in.
        """
        return self._default_lib_path

    @abc.abstractmethod
    def close(self):
        # type: () -> None
        """Terminate the database server gracefully.
        """
        pass

    @abc.abstractmethod
    def get_cells_in_library(self, lib_name):
        # type: (str) -> List[str]
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : List[str]
            a list of cells in the library
        """
        return []

    @abc.abstractmethod
    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : str
            the library name.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        pass

    @abc.abstractmethod
    def create_implementation(self, lib_name, template_list, change_list, lib_path=''):
        # type: (str, Sequence[Any], Sequence[Any], str) -> None
        """Create implementation of a design in the CAD database.

        Parameters
        ----------
        lib_name : str
            implementation library name.
        template_list : Sequence[Any]
            a list of schematic templates to copy to the new library.
        change_list : Sequence[Any]
            a list of changes to be performed on each copied templates.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        pass

    @abc.abstractmethod
    def configure_testbench(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[str, List[str], Dict[str, str], Dict[str, str]]
        """Update testbench state for the given testbench.

        This method fill in process-specific information for the given testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        cur_env : str
            the current simulation environment.
        envs : List[str]
            a list of available simulation environments.
        parameters : Dict[str, str]
            a list of testbench parameter values, represented as string.
        outputs : Dict[str, str]
            a dictionary of output expressions
        """
        return "", [], {}, {}

    @abc.abstractmethod
    def get_testbench_info(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]
        """Returns information about an existing testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library.
        tb_cell : str
            testbench cell.

        Returns
        -------
        cur_envs : List[str]
            the current simulation environments.
        envs : List[str]
            a list of available simulation environments.
        parameters : Dict[str, str]
            a list of testbench parameter values, represented as string.
        outputs : Dict[str, str]
            a list of testbench output expressions.
        """
        return [], [], {}, {}

    @abc.abstractmethod
    def update_testbench(self,  # type: DbAccess
                         lib,  # type: str
                         cell,  # type: str
                         parameters,  # type: Dict[str, str]
                         sim_envs,  # type: Sequence[str]
                         config_rules,  # type: Sequence[List[str]]
                         env_parameters,  # type: Sequence[List[Tuple[str, str]]]
                         ):
        # type: (...) -> None
        """Update the given testbench configuration.

        Parameters
        ----------
        lib : str
            testbench library.
        cell : str
            testbench cell.
        parameters : Dict[str, str]
            testbench parameters.
        sim_envs : Sequence[str]
            list of enabled simulation environments.
        config_rules : Sequence[List[str]]
            config view mapping rules, list of (lib, cell, view) rules.
        env_parameters : Sequence[List[Tuple[str, str]]]
            list of param/value list for each simulation environment.
        """
        pass

    @abc.abstractmethod
    def instantiate_layout_pcell(self, lib_name, cell_name, view_name,
                                 inst_lib, inst_cell, params, pin_mapping):
        # type: (str, str, str, str, str, Dict[str, Any], Dict[str, str]) -> None
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        view_name : str
            layout view name, default is "layout".
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : Dict[str, Any]
            the parameter dictionary.
        pin_mapping: Dict[str, str]
            the pin mapping dictionary.
        """
        pass

    @abc.abstractmethod
    def instantiate_layout(self, lib_name, view_name, via_tech, layout_list):
        # type: (str, str, str, Sequence[Any]) -> None
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : str
            layout library name.
        view_name : str
            layout view name.
        via_tech : str
            via technology library name.
        layout_list : Sequence[Any]
            a list of layouts to create
        """
        pass

    @abc.abstractmethod
    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        pass

    @abc.abstractmethod
    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        # type: (str, str, str, Optional[str], **kwargs) -> None
        """Create a schematic from a netlist.

        This is mainly used to create extracted schematic from an extracted netlist.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : Optional[str]
            schematic view name.  The default value is implemendation dependent.
        **kwargs
            additional implementation-dependent arguments.
        """
        pass

    @abc.abstractmethod
    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **kwargs) -> None
        """Create a verilog view for mix-signal simulation.

        Parameters
        ----------
        verilog_file : str
            the verilog file name.
        lib_name : str
            library name.
        cell_name : str
            cell name.
        **kwargs
            additional implementation-dependent arguments.
        """
        pass

    @abc.abstractmethod
    def import_sch_cellview(self, lib_name, cell_name, dsn_db, new_lib_path):
        # type: (str, str, ModuleDB, str) -> None
        """Recursively import the given schematic and symbol cellview.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        dsn_db : ModuleDB
            the design database object.
        new_lib_path: str
            location to import new libraries to.
        """
        pass

    @abc.abstractmethod
    def import_design_library(self, lib_name, dsn_db, new_lib_path):
        # type: (str, ModuleDB, str) -> None
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        dsn_db : ModuleDB
            the design database object.
        new_lib_path: str
            location to import new libraries to.
        """
        pass

    def _process_rcx_output(self, netlist, log_fname, lib_name, cell_name, create_schematic):
        if create_schematic:
            if netlist is None:
                return False, log_fname
            if netlist:
                # create schematic only if netlist name is not empty.
                self.create_schematic_from_netlist(netlist, lib_name, cell_name)
            return True, log_fname
        else:
            return netlist, log_fname

    async def async_run_lvs(self, lib_name, cell_name, **kwargs):
        # type: (str, str, **kwargs) -> Tuple[bool, str]
        """A coroutine for running LVS.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs :
            optional keyword arguments.  See Checker class for details.
            LVS parameters should be specified as lvs_params.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        if self.checker is None:
            raise Exception('LVS/RCX is disabled.')

        kwargs['params'] = kwargs.pop('lvs_params', None)
        return await self.checker.async_run_lvs(lib_name, cell_name, **kwargs)

    async def async_run_rcx(self,  # type: DbAccess
                            lib_name,  # type: str
                            cell_name,  # type: str
                            create_schematic=True,  # type: bool
                            **kwargs  # type: **kwargs
                            ):
        # type: (...) -> Tuple[Union[bool, Optional[str]], str]
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If create_schematic is True, this method will run RCX, then if it succeeds,
        create a schematic of the extracted netlist in the database.  It then returns
        a boolean value which will be True if RCX succeeds.

        If create_schematic is False, this method will run RCX, then return a string
        which is the extracted netlist filename. If RCX failed, None will be returned
        instead.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
            override RCX parameter values.
        create_schematic : bool
            True to automatically create extracted schematic in database if RCX
            is successful and it is supported.
        **kwargs :
            optional keyword arguments.  See Checker class for details.
            RCX parameters should be specified as rcx_params.

        Returns
        -------
        value : Union[bool, Optional[str]]
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        kwargs['params'] = kwargs.pop('rcx_params', None)
        netlist, log_fname = await self.checker.async_run_rcx(lib_name, cell_name, **kwargs)

        return self._process_rcx_output(netlist, log_fname, lib_name, cell_name, create_schematic)

    async def async_export_layout(self, lib_name, cell_name, out_file, *args, **kwargs):
        # type: (str, str, str, *args, **kwargs) -> str
        """export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        *args :
            optional list arguments.
        **kwargs :
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        if self.checker is None:
            raise Exception('layout export is disabled.')

        return await self.checker.async_export_layout(lib_name, cell_name, out_file,
                                                      *args, **kwargs)

    def instantiate_schematic(self, lib_name, content_list, lib_path=''):
        """Create the given schematics in CAD database.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        content_list : Sequence[Any]
            list of schematics to create.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        template_list, change_list = [], []
        for content in content_list:
            if content is not None:
                master_lib, master_cell, impl_cell, pin_map, inst_map, new_pins = content

                # add to template list
                template_list.append([master_lib, master_cell, impl_cell])

                # construct change object
                change = dict(
                    name=impl_cell,
                    pin_map=dict_to_item_list(pin_map),
                    inst_list=format_inst_map(inst_map),
                    new_pins=new_pins,
                )
                change_list.append(change)

        self.create_implementation(lib_name, template_list, change_list, lib_path=lib_path)
