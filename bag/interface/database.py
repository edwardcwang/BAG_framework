# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""This module defines DbAccess, the base class for CAD database manipulation.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import abc
import traceback

from jinja2 import Template
import yaml
import networkx as nx
from future.utils import with_metaclass

import bag.io
from ..verification import make_checker


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


def format_inst_map(inst_map, concrete_lib_name):
    """Given instance map from DesignModule, format it for database changes.

    Parameters
    ----------
    inst_map : dict[str, any]
        the instance map created by DesignModule.
    concrete_lib_name : str
        name of the concrete schematic library.

    Returns
    -------
    ans : list[(str, any)]
        the database change instance map.
    """
    ans = []
    for old_inst_name, rinst_list in inst_map.items():
        new_rinst_list = [dict(name=rinst['name'],
                               lib_name=rinst['lib_name'] or concrete_lib_name,
                               cell_name=rinst['cell_name'],
                               params=dict_to_item_list(rinst['params']),
                               term_mapping=dict_to_item_list(rinst['term_mapping']),
                               ) for rinst in rinst_list]
        ans.append([old_inst_name, new_rinst_list])
    return ans


def get_python_template(lib_name, cell_name, primitive_table):
    """Returns the default Python Module template for the given schematic.

    Parameters
    ----------
    lib_name : str
        the library name.
    cell_name : str
        the cell name.
    primitive_table : dict[str, str]
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
            elif cell_name == 'cap_ideal':
                # ideal capacitor template
                module_name = 'CapIdealModuleBase'
            elif cell_name.startswith('res_'):
                # physical resistor template
                module_name = 'ResPhysicalModuleBase'
            else:
                raise Exception('Unknown primitive cell: %s' % cell_name)

            content = bag.io.read_resource(__name__, os.path.join('templates', 'PrimModule.pytemp'))
            param_dict['module_name'] = module_name
    else:
        # use default empty template.
        content = bag.io.read_resource(__name__, os.path.join('templates', 'Module.pytemp'))

    return Template(content).render(**param_dict)


class DbAccess(with_metaclass(abc.ABCMeta, object)):
    """A class that manipulates the CAD database.

    Parameters
    ----------
    tmp_dir : string
        temporary file directory for DbAccess.
    db_config : dict[string, any]
        the database configuration dictionary.
    """

    def __init__(self, tmp_dir, db_config):
        """Create a new DbAccess object.
        """
        self.tmp_dir = bag.io.make_temp_dir('dbTmp', parent_dir=tmp_dir)
        self.db_config = db_config
        try:
            check_kwargs = self.db_config['checker'].copy()
            check_kwargs['tmp_dir'] = self.tmp_dir
            self.checker = make_checker(**check_kwargs)
        except:
            stack_trace = traceback.format_exc()
            print('*WARNING* error creating Checker:\n%s' % stack_trace)
            print('*WARNING* LVS/RCX will be disabled.')
            self.checker = None

        # set default lib path
        lib_path_fallback = os.path.abspath('.')
        self._default_lib_path = self.db_config.get('default_lib_path', lib_path_fallback)
        if not os.path.isdir(self._default_lib_path):
            self._default_lib_path = lib_path_fallback

    @property
    def default_lib_path(self):
        """Returns the default directory to create new libraries in.

        Returns
        -------
        lib_path : string
            directory to create new libraries in.
        """
        return self._default_lib_path

    @abc.abstractmethod
    def close(self):
        """Terminate the database server gracefully.
        """
        pass

    @abc.abstractmethod
    def parse_schematic_template(self, lib_name, cell_name):
        """Parse the given schematic template.

        Parameters
        ----------
        lib_name : str
            name of the library.
        cell_name : str
            name of the cell.

        Returns
        -------
        template : str
            the content of the netlist structure file.
        """
        return ""

    @abc.abstractmethod
    def get_cells_in_library(self, lib_name):
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : list[str]
            a list of cells in the library
        """
        return []

    @abc.abstractmethod
    def create_library(self, lib_name, lib_path=''):
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : string
            the library name.
        lib_path : string
            directory to create the library in.  If Empty, use default location.
        """
        pass

    @abc.abstractmethod
    def create_implementation(self, lib_name, template_list, change_list, lib_path=''):
        """Create implementation of a design in the CAD database.

        Parameters
        ----------
        lib_name : str
            implementation library name.
        template_list : list
            a list of schematic templates to copy to the new library.
        change_list :
            a list of changes to be performed on each copied templates.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        pass

    @abc.abstractmethod
    def instantiate_testbench(self, tb_lib, tb_cell, targ_lib, dut_lib, dut_cell, new_lib_path=''):
        """Create a new process-specific testbench based on a testbench template.

        Copies the testbench template to another library, replace the device-under-test (DUT),
        then fill in process-specific information.

        Parameters
        ----------
        tb_lib : str
            testbench template library name.
        tb_cell : str
            testbench cell name.
        targ_lib : str
            the process-specific testbench library name.
        dut_lib : str
            DUT library name.
        dut_cell : str
            DUT cell name.
        new_lib_path : str
            path to put targ_lib if it does not exist.  If Empty, use default location.

        Returns
        -------
        cur_env : str
            the current simulation environment.
        envs : list[str]
            a list of available simulation environments.
        parameters : dict[str, str]
            a list of testbench parameter values, represented as string.
        """
        return "", [], {}

    @abc.abstractmethod
    def get_testbench_info(self, tb_lib, tb_cell):
        """Returns information about an existing testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library.
        tb_cell : str
            testbench cell.

        Returns
        -------
        cur_envs : list[str]
            the current simulation environments.
        envs : list[str]
            a list of available simulation environments.
        parameters : dict[str, str]
            a list of testbench parameter values, represented as string.
        outputs : dict[str, str]
            a list of testbench output expressions.
        """
        return [], [], {}, {}

    @abc.abstractmethod
    def update_testbench(self, lib, cell, parameters, sim_envs, config_rules):
        """Update the given testbench configuration.

        Parameters
        ----------
        lib : str
            testbench library.
        cell : str
            testbench cell.
        parameters : dict[str, str]
            testbench parameters.
        sim_envs : list[str]
            list of enabled simulation environments.
        config_rules : list[list[str]]
            config view mapping rules, list of (lib, cell, view) rules.
        """
        pass

    @abc.abstractmethod
    def instantiate_layout_pcell(self, lib_name, cell_name, view_name,
                                 inst_lib, inst_cell, params, pin_mapping):
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
        params : dict[str, any]
            the parameter dictionary.
        pin_mapping: dict[str, str]
            the pin mapping dictionary.
        """
        pass

    @abc.abstractmethod
    def instantiate_layout(self, lib_name, view_name, via_tech, layout_list):
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : str
            layout library name.
        view_name : str
            layout view name.
        via_tech : str
            via technology library name.
        layout_list : list[any]
            a list of layouts to create
        """
        pass

    @abc.abstractmethod
    def release_write_locks(self, lib_name, cell_view_list):
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : string
            the library name.
        cell_view_list : List[(string, string)]
            list of cell/view name tuples.
        """
        pass

    @abc.abstractmethod
    def run_lvs(self, lib_name, cell_name, sch_view, lay_view, lvs_params,
                block=True, callback=None):
        """Run LVS on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : str
            schematic view name.  Default is 'schematic'.
        lay_view : str
            layout view name.  Default is 'layout'.
        lvs_params : dict[str, any]
            override LVS parameter values.
        block : bool
            If True, wait for LVS to finish.  Otherwise, return
            a ID you can use to query LVS status later.
        callback : callable or None
            If given, this function will be called with the LVS success flag
            and process return code when LVS finished.

        Returns
        -------
        value : bool or string
            If block is True, returns the LVS success flag.  Otherwise,
            return a LVS ID you can use to query LVS status later.
        log_fname : str
            name of the LVS log file.
        """
        return False

    @abc.abstractmethod
    def run_rcx(self, lib_name, cell_name, sch_view, lay_view, rcx_params,
                block=True, callback=None, create_schematic=True):
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If block is True and create_schematic is True, this method will run RCX,
        then if it succeeds, create a schematic of the extracted netlist in the
        database.  It then returns a boolean value which will be True if
        RCX succeeds.

        If block is True and create_schematic is False, this method will run
        RCX, then return a string which is the extracted netlist filename.
        If RCX failed, None will be returned instead.

        If block is False, this method will submit a RCX job and return a string
        RCX ID which you can use to query RCX status later.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : str
            schematic view name.  Default is 'schematic'.
        lay_view : str
            layout view name.  Default is 'layout'.
        rcx_params : dict[str, any]
            override RCX parameter values.
        block : bool
            If True, wait for RCX to finish.  Otherwise, return
            a ID you can use to query RCX status later.
        callback : callable or None
            If given, this function will be called with the RCX netlist filename
            and process return code when RCX finished.
        create_schematic : bool
            True to automatically create extracted schematic in database if RCX
            is successful and it is supported.

        Returns
        -------
        value : bool or string
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        return False

    @abc.abstractmethod
    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        """Create a schematic from a netlist.

        This is mainly used to create extracted schematic from an extracted netlist.

        Parameters
        ----------
        netlist : string
            the netlist file name.
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : string or None
            schematic view name.  The default value is implemendation dependent.
        kwargs : dict[string, any]
            additional implementation-dependent arguments.
        """
        pass

    def import_design_library(self, lib_name, dsn_db, new_lib_path):
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        dsn_db : :class:`bag.design.Database`
            the design database object.
        new_lib_path: str
            location to import new libraries to.
        """
        imported_cells = set()
        for cell_name in self.get_cells_in_library(lib_name):
            self._import_design(lib_name, cell_name, imported_cells, dsn_db, new_lib_path)

    def _import_design(self, lib_name, cell_name, imported_cells, dsn_db, new_lib_path):
        """Recursive helper for import_design_library.
        """
        # check if we already imported this schematic
        key = '%s__%s' % (lib_name, cell_name)
        if key in imported_cells:
            return
        imported_cells.add(key)

        # create root directory if missing
        root_path = dsn_db.get_library_path(lib_name)
        if root_path is None:
            root_path = new_lib_path
            dsn_db.append_library(lib_name, new_lib_path)

        package_path = os.path.join(root_path, lib_name)
        python_file = os.path.join(package_path, '%s.py' % cell_name)
        yaml_file = os.path.join(package_path, 'netlist_info', '%s.yaml' % cell_name)
        yaml_dir = os.path.dirname(yaml_file)
        if not os.path.exists(yaml_dir):
            os.makedirs(yaml_dir)
            bag.io.write_file(os.path.join(package_path, '__init__.py'), '\n',
                              mkdir=False)

        # update netlist file
        content = self.parse_schematic_template(lib_name, cell_name)
        sch_info = yaml.load(content)
        try:
            bag.io.write_file(yaml_file, content)
        except IOError:
            print('Warning: cannot write to %s.' % yaml_file)

        # generate new design module file if necessary.
        if not os.path.exists(python_file):
            content = get_python_template(lib_name, cell_name,
                                          self.db_config.get('prim_table', {}))
            bag.io.write_file(python_file, content + '\n', mkdir=False)

        # recursively import all children
        for inst_name, inst_attrs in sch_info['instances'].items():
            inst_lib_name = inst_attrs['lib_name']
            inst_cell_name = inst_attrs['cell_name']
            self._import_design(inst_lib_name, inst_cell_name, imported_cells, dsn_db,
                                new_lib_path)

    def implement_design(self, lib_name, module, lib_path=''):
        """Implement the given design.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        module : :class:`bag.design.Module`
            the design module to create schematics for.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        hierarchy_graph = module.hierarchy_graph

        # sort the netlist graph in reverse order.
        nodes = nx.topological_sort(hierarchy_graph, reverse=True)

        # create template_list and change_list
        template_list = []
        change_list = []
        for n in nodes:
            master_lib = n.get_lib_name()
            master_cell = n.get_cell_name()
            attrs = hierarchy_graph.node[n]
            concrete_cell_name = attrs['concrete_cell_name']
            pin_map = attrs['pin_map']
            inst_map = attrs['inst_map']

            # add to template list
            template_list.append([master_lib, master_cell, concrete_cell_name])

            # construct change object
            change = dict(
                name=concrete_cell_name,
                pin_map=dict_to_item_list(pin_map),
                inst_list=format_inst_map(inst_map, lib_name),
            )
            change_list.append(change)

        self.create_implementation(lib_name, template_list, change_list, lib_path=lib_path)
