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

"""This module defines base design module class and primitive design classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import abc
from future.utils import with_metaclass
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any, Type, Set, Sequence, Callable

from bag import float_to_si_string
from bag.io import read_yaml, fix_string
from bag.util.cache import DesignMaster, MasterDB

if TYPE_CHECKING:
    from bag.core import BagProject
    from bag.layout.core import TechInfo


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
    """
    def __init__(self, lib_defs, tech_info, sch_exc_libs, prj=None, name_prefix='', name_suffix=''):
        # type: (str, TechInfo, List[str], Optional[BagProject], str, str) -> None
        super(ModuleDB, self).__init__('', lib_defs=lib_defs, name_prefix=name_prefix, name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info
        self._exc_libs = set(sch_exc_libs)

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[Module], str, Dict[str, Any], Set[str], **kwargs) -> Module
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
        **kwargs
            optional arguments for the generator.

        Returns
        -------
        master : Module
            the non-finalized generated instance.
        """
        kwargs = kwargs.copy()
        kwargs['prj'] = self._prj
        kwargs['lib_name'] = lib_name
        kwargs['params'] = params
        kwargs['used_cell_names'] = used_cell_names
        return gen_cls(self, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False):
        # type: (str, Sequence[Any], bool) -> None
        """Create the masters in the design database.

        Parameters
        ----------
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        debug : bool
            True to print debug messages
        """
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        raise ValueError('Not implemented yet')

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info


class SchInstance(object):
    """A class representing a schematic instance.

    Parameters
    ----------
    database : ModuleDB
        the schematic generator database.
    lib_name : str
        the instance master library name.
    cell_name : str
        the instance master cell name.
    inst_name : str
        name of this instance.
    static : bool
        True if the instance master is static.
    """
    def __init__(self, database, lib_name, cell_name, inst_name, static=False):
        # type: (MasterDB, str, str, str, bool) -> None
        self._db = database
        self._inst_name = inst_name
        self._master = None
        self.lib_name = lib_name
        self.cell_name = cell_name
        self.static = static
        self.term_mapping = {}
        self.parameters = {}

    def change_master(self, lib_name, cell_name, static=False):
        # type: (str, str, bool) -> None
        """Change the master associated with this instance.

        All instance parameters and terminal mappings will be reset.

        Parameters
        ----------
        lib_name : str
            the new master library name.
        cell_name : str
            the new master cell name.
        static : bool
            True if the instance master is static.
        """
        self._master = None
        self.lib_name = lib_name
        self.cell_name = cell_name
        self.static = static
        self.parameters.clear()
        self.term_mapping.clear()

    def design_specs(self, **kwargs):
        self.design(**kwargs)

    def design(self, **kwargs):
        pass


class ModuleID(object):
    """A class that uniquely identifies a module.

    This class is used to identify a unique module instance.

    Parameters
    ----------
    lib_name : str
        the module library.
    cell_name : str
        the module cell.
    iden_str : str
        the module identification string.
    """

    def __init__(self, lib_name, cell_name, iden_str):
        self.lib_name = lib_name
        self.cell_name = cell_name
        self.iden_str = iden_str

    def __hash__(self):
        ans = 17
        ans = ans * 31 + hash(self.lib_name)
        ans = ans * 31 + hash(self.cell_name)
        return ans * 31 + hash(self.iden_str)

    def __eq__(self, other):
        return (self.lib_name == getattr(other, 'lib_name', None) and
                self.cell_name == getattr(other, 'cell_name', None) and
                self.iden_str == getattr(other, 'iden_str', None))

    def get_lib_name(self):
        """Returns the master template library name"""
        return self.lib_name

    def get_cell_name(self):
        """Returns the master template cell name"""
        return self.cell_name


class Module(with_metaclass(abc.ABCMeta, DesignMaster)):
    """The base class of all schematic generators.  This represents a schematic master.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_fname : str
        the netlist information file name.  If this is a generic module, this should be None.
    parent : None
        deprecated parameter.  Not used.
    prj : Optional[BagProject]
        the BagProject instance.  Used to implement design.

    Attributes
    ----------
    parameters : dict[str, any]
        the design parameters dictionary.
    instances : dict[str, None or :class:`~bag.design.Module` or list[:class:`~bag.design.Module`]]
        the instance dictionary.
    """

    # noinspection PyUnusedLocal
    def __init__(self, database, yaml_fname, parent=None, prj=None, **kwargs):
        # type: (ModuleDB, str, None, Optional[BagProject], **kwargs) -> None

        lib_name = kwargs['lib_name']
        params = kwargs['params']
        used_names = kwargs['used_names']
        design_fun = kwargs['design_fun']

        self.prj = prj
        self.tech_info = database.tech_info
        self.instances = {}
        self.pin_map = {}
        self.new_pins = []
        self.parameters = {}

        self._yaml_fname = os.path.abspath(yaml_fname)
        self.sch_info = read_yaml(self._yaml_fname)

        self._orig_lib_name = self.sch_info['lib_name']
        self._orig_cell_name = self.sch_info['cell_name']
        self._design_fun = design_fun

        # create initial instances and populate instance map
        for inst_name, inst_attr in self.sch_info['instances'].items():
            lib_name = inst_attr['lib_name']
            cell_name = inst_attr['cell_name']
            self.instances[inst_name] = SchInstance(database, lib_name, cell_name, inst_name, static=False)

        # fill in pin map
        for pin in self.sch_info['pins']:
            self.pin_map[pin] = pin

        # initialize schematic master
        super(Module, self).__init__(database, lib_name, params, used_names)

    @abc.abstractmethod
    def design(self, **kwargs):
        """To be overridden by subclasses to design this module.

        To design instances of this module, you can
        call their :meth:`.design` method or any other ways you coded.

        To modify schematic structure, call:

        :meth:`.rename_pin`

        :meth:`.delete_instance`

        :meth:`.replace_instance_master`

        :meth:`.reconnect_instance_terminal`

        :meth:`.array_instance`
        """
        pass

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # invoke design function
        fun = getattr(self, self._design_fun)
        fun(**self.params)

        # backwards compatibility
        if self.key is None:
            self.params.clear()
            self.params.update(self.parameters)
            self.update_master_info()

        # call super finalize routine
        super(Module, self).finalize()

    @classmethod
    def get_params_info(cls):
        # type: () -> Optional[Dict[str, str]]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return None

    def get_master_basename(self):
        # type: () -> str
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self._orig_cell_name

    def get_content(self, lib_name, rename_fun):
        # type: (str, Callable[str, str]) -> Any
        """Returns the content of this master instance.

        Parameters
        ----------
        lib_name : str
            the library to create the design masters in.
        rename_fun : Callable[str, str]
            a function that renames design masters.

        Returns
        -------
        content : Any
            the master content data structure.
        """
        return None

    def is_primitive(self):
        # type: () -> bool
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG and schematic primitives.  This method prevents
        the module from being copied during design implementation.  Custom subclasses should
        not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return False

    def should_delete_instance(self):
        # type: () -> bool
        """Returns True if this instance should be deleted based on its parameters.

        This method is mainly used to delete 0 finger or 0 width transistors.  However,
        You can override this method if there exists parameter settings which corresponds
        to an empty schematic.

        Returns
        -------
        delete : bool
            True if parent should delete this instance.
        """
        return False

    def rename_pin(self, old_pin, new_pin):
        # type: (str, str) -> None
        """Renames an input/output pin of this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        old_pin : str
            the old pin name.
        new_pin : str
            the new pin name.
        """
        self.pin_map[old_pin] = new_pin

    def add_pin(self, new_pin, pin_type):
        # type: (str, str) -> None
        """Adds a new pin to this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        new_pin : str
            the new pin name.
        pin_type : str
            the new pin type.  We current support "input", "output", or "inputOutput"
        """
        self.new_pins.append([new_pin, pin_type])

    def remove_pin(self, remove_pin):
        # type: (str) -> None
        """Removes a pin from this schematic.

        Parameters
        ----------
        remove_pin : str
            the pin to remove.
        """
        self.rename_pin(remove_pin, '')

    def delete_instance(self, inst_name):
        # type: (str) -> None
        """Delete the instance with the given name.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.
        """
        self.instances[inst_name] = []

    def replace_instance_master(self, inst_name, lib_name, cell_name, static=False, index=None):
        # type: (str, str, str, bool, Optional[int]) -> None
        """Replace the master of the given instance.

        NOTE: all terminal connections will be reset.  Call reconnect_instance_terminal() to modify
        terminal connections.

        Parameters
        ----------
        inst_name : str
            the child instance to replace.
        lib_name : str
            the new library name.
        cell_name : str
            the new cell name.
        static : bool
            True if we're replacing instance with a static schematic instead of a design module.
        index : Optional[int]
            If index is not None and the child instance has been arrayed, this is the instance array index
            that we are replacing.
            If index is None, the entire child instance (whether arrayed or not) will be replaced by
            a single new instance.
        """
        if inst_name not in self.instances:
            raise ValueError('Cannot find instance with name: %s' % inst_name)

        # check if this is arrayed
        if index is not None and isinstance(self.instances[inst_name], list):
            self.instances[inst_name][index].change_master(lib_name, cell_name, static=static)
        else:
            self.instances[inst_name] = SchInstance(self.master_db, lib_name, cell_name, inst_name, static=static)

    def reconnect_instance_terminal(self, inst_name, term_name, net_name, index=None):
        """Reconnect the instance terminal to a new net.

        Parameters
        ----------
        inst_name : str
            the child instance to modify.
        term_name : Union[str, List[str]]
            the instance terminal name to reconnect.
            If a list is given, it is applied to each arrayed instance.
        net_name : Union[str, List[str]]
            the net to connect the instance terminal to.
            If a list is given, it is applied to each arrayed instance.
        index : Optional[int]
            If not None and the given instance is arrayed, will only modify terminal
            connection for the instance at the given index.
            If None and the given instance is arrayed, all instances in the array
            will be reconnected.
        """
        if index is not None:
            # only modify terminal connection for one instance in the array
            if isinstance(term_name, str) and isinstance(net_name, str):
                self.instances[inst_name][index].term_mapping[term_name] = net_name
            else:
                raise ValueError('If index is not None, both term_name and net_name must be string.')
        else:
            # modify terminal connection for all instances in the array
            cur_inst_list = self.instances[inst_name]
            if isinstance(cur_inst_list, SchInstance):
                cur_inst_list = [cur_inst_list]

            num_insts = len(cur_inst_list)
            if not isinstance(term_name, list) and not isinstance(term_name, tuple):
                if not isinstance(term_name, str):
                    raise ValueError('term_name = %s must be string.' % term_name)
                term_name = [term_name] * num_insts
            else:
                if len(term_name) != num_insts:
                    raise ValueError('term_name length = %d != %d' % (len(term_name), num_insts))

            if not isinstance(net_name, list) and not isinstance(net_name, tuple):
                if not isinstance(net_name, str):
                    raise ValueError('net_name = %s must be string.' % net_name)
                net_name = [net_name] * num_insts
            else:
                if len(net_name) != num_insts:
                    raise ValueError('net_name length = %d != %d' % (len(net_name), num_insts))

            for inst, tname, nname in zip(cur_inst_list, term_name, net_name):
                inst.term_mapping[term_name] = net_name

    def array_instance(self, inst_name, inst_name_list, term_list=None, same=False):
        # type: (str, List[str], Optional[List[Dict[str, str]]], bool) -> None
        """Replace the given instance by an array of instances.

        This method will replace self.instances[inst_name] by a list of
        Modules.  The user can then design each of those modules.

        Parameters
        ----------
        inst_name : str
            the instance to array.
        inst_name_list : List[str]
            a list of the names for each array item.
        term_list : Optional[List[Dict[str, str]]]
            a list of modified terminal connections for each array item.  The keys are
            instance terminal names, and the values are the net names to connect
            them to.  Only terminal connections different than the parent instance
            should be listed here.
            If None, assume terminal connections are not changed.
        same : bool
            Deprecated feature.  No longer supported.
        """
        if same:
            raise ValueError('same flag no longer supported.  See developer.')

        num_inst = len(inst_name_list)
        if not term_list:
            term_list = [{} for _ in range(num_inst)]
        if num_inst != len(term_list):
            msg = 'len(inst_name_list) = %d != len(term_list) = %d'
            raise ValueError(msg % (num_inst, len(term_list)))

        orig_inst = self.instances[inst_name]
        if not isinstance(orig_inst, SchInstance):
            orig_inst = orig_inst[0]

        lib_name, cell_name, static = orig_inst.lib_name, orig_inst.cell_name, orig_inst.static
        new_inst_list = []
        for iname, iterm in zip(inst_name_list, term_list):
            cur_inst = SchInstance(self.master_db, lib_name, cell_name, iname, static=static)
            cur_inst.term_mapping.update(iterm)
            new_inst_list.append(cur_inst)

        self.instances[inst_name] = new_inst_list

    def design_dc_bias_sources(self, vbias_dict, ibias_dict, vinst_name, iinst_name, define_vdd=True):
        # type: (Optional[Dict[str, List[str]]], Optional[Dict[str, List[str]]], str, str, bool) -> None
        """Convenience function for generating DC bias sources.

        Given DC voltage/current bias sources information, array the given voltage/current bias sources
        and configure the voltage/current.

        Each bias dictionary is a dictionary from bias source name to a 3-element list.  The first two
        elements are the PLUS/MINUS net names, respectively, and the third element is the DC
        voltage/current value as a string or float. A variable name can be given to define a testbench
        parameter.

        Parameters
        ----------
        vbias_dict : Optional[Dict[str, List[str]]]
            the voltage bias dictionary.  None or empty to disable.
        ibias_dict : Optional[Dict[str, List[str]]]
            the current bias dictionary.  None or empty to disable.
        vinst_name : str
            the DC voltage source instance name.
        iinst_name : str
            the DC current source instance name.
        define_vdd : bool
            True to include a supply voltage source connected to VDD/VSS, with voltage value 'vdd'.
        """
        if define_vdd:
            vbias_dict = vbias_dict.copy()
            # make sure VDD is always included
            name = 'SUP'
            counter = 1
            while name in vbias_dict:
                name = 'SUP%d' % counter
                counter += 1

            vbias_dict[name] = ['VDD', 'VSS', 'vdd']

        for bias_dict, name_template, param_name, inst_name in \
                ((vbias_dict, 'V%s', 'vdc', vinst_name), (ibias_dict, 'I%s', 'idc', iinst_name)):
            if bias_dict:
                name_list, term_list, val_list = [], [], []
                for name in sorted(bias_dict.keys()):
                    pname, nname, bias_val = bias_dict[name]
                    term_list.append(dict(PLUS=pname, MINUS=nname))
                    name_list.append(name_template % name)
                    if isinstance(bias_val, str):
                        val_list.append(bias_val)
                    elif isinstance(bias_val, int) or isinstance(bias_val, float):
                        val_list.append(float_to_si_string(bias_val))
                    else:
                        raise ValueError('value %s of type %s not supported' % (bias_val, type(bias_val)))

                self.array_instance(inst_name, name_list, term_list=term_list)
                for inst, val in zip(self.instances[inst_name], val_list):
                    inst.parameters[param_name] = val
            else:
                self.delete_instance(inst_name)

    def design_dummy_transistors(self, dum_info, inst_name, vdd_name, vss_name):
        # type: (List[Tuple[Any]], str, str, str) -> None
        """Convenience function for generating dummy transistor schematic.

        Given dummy information (computed by AnalogBase) and a BAG transistor instance,
        this method generates dummy schematics by arraying and modifying the BAG
        transistor instance.

        Parameters
        ----------
        dum_info : List[Tuple[Any]]
            the dummy information data structure.
        inst_name : str
            the BAG transistor instance name.
        vdd_name : str
            VDD net name.  Used for PMOS dummies.
        vss_name : str
            VSS net name.  Used for NMOS dummies.
        """
        if not dum_info:
            self.delete_instance(inst_name)
        else:
            num_arr = len(dum_info)
            arr_name_list = ['XDUMMY%d' % idx for idx in range(num_arr)]
            self.array_instance(inst_name, arr_name_list)

            for idx, ((mos_type, w, lch, th, s_net, d_net), fg) in enumerate(dum_info):
                if mos_type == 'pch':
                    cell_name = 'pmos4_standard'
                    sup_name = vdd_name
                else:
                    cell_name = 'nmos4_standard'
                    sup_name = vss_name
                s_name = s_net if s_net else sup_name
                d_name = d_net if d_net else sup_name

                self.replace_instance_master(inst_name, 'BAG_prim', cell_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'G', sup_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'B', sup_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'D', d_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'S', s_name, index=idx)
                self.instances[inst_name][idx].design(w=w, l=lch, nf=fg, intent=th)

    def get_primitives_descendants(self, cur_inst_name):
        """Internal method.  Returns a dictionary of all BAG primitive descendants of this module.

        Parameters
        ----------
        cur_inst_name : string
            name of the current module instance.

        Returns
        -------
        prim_dict : dict[str, :class:`bag.design.Module`]
            a dictionary from primitive's absolute path name to the corresponding Module.
        """
        prim_dict = {}
        for inst_name, rinst_list in self.instance_map.items():
            module_list = self.instances[inst_name]
            if not isinstance(module_list, list):
                module_list = [module_list]

            for modul, rinst in zip(module_list, rinst_list):
                name = '%s.%s' % (cur_inst_name, rinst['name'])
                if modul.is_primitive():
                    prim_dict[name] = modul
                else:
                    sub_dict = modul.get_primitives_descendants(name)
                    prim_dict.update(sub_dict)
        return prim_dict

    def update_bag_primitives(self):
        """Internal method.

        Update schematic parameters and cell name for bag primitives.
        As a side effect, also check that every declared parameters are set.

        """
        for param, val in self.parameters.items():
            if val is None:
                raise Exception('Parameter %s unset.' % param)

        for key, rinst_list in self.instance_map.items():
            module_list = self.instances[key]
            if not isinstance(module_list, list):
                module_list = [module_list]

            # Traverse list backward so we can remove instances
            for idx in range(len(rinst_list) - 1, -1, -1):
                rinst = rinst_list[idx]
                modul = module_list[idx]
                modul.update_bag_primitives()
                if modul.should_delete_instance():
                    # check if we need to delete this instance based on its parameters
                    del rinst_list[idx]
                else:
                    # update parameter/cell name information.
                    rinst['params'] = modul.get_schematic_parameters()
                    rinst['cell_name'] = modul.get_cell_name_from_parameters()

    def update_structure(self, lib_name, top_cell_name='', prefix='', suffix='', used_names=None):
        """Update the generated schematic structure.

        This function should be called after :func:`.design` to prepare this design module
        for implementation.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the generated schematics.
        top_cell_name : str
            the cell name of the top level design.
        prefix : str
            prefix to add to cell names.
        suffix : str
            suffix to add to cell names.
        used_names : set[str] or None
            if given, all names in this set will not be used.
        """
        # Error checking: update can only be called on top level.
        if self.parent is not None:
            raise Exception('Can only call update_structure() on top level Module.')

        # update bag primitives
        self.update_bag_primitives()

        # Create the instance hierarchy graph
        self.hierarchy_graph = nx.MultiDiGraph()
        used_concrete_names = used_names or set()
        self.populate_hierarchy_graph(lib_name, self.hierarchy_graph, used_concrete_names,
                                      top_cell_name, prefix, suffix, 'XTOP')

    def implement_design(self, lib_name, top_cell_name='', prefix='', suffix='', lib_path='', erase=False):
        """Implement this design module in the given library.

        If the given library already exists, this method will not delete or override
        any pre-existing cells in that library.

        If you use this method, you do not need to call update_structure(),
        as this method calls it for you.

        This method only works if BagProject is given.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the generated schematics.
        top_cell_name : str
            the cell name of the top level design.
        prefix : str
            prefix to add to cell names.
        suffix : str
            suffix to add to cell names.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        erase : bool
            True to erase old cellviews.
        """
        if not self.prj:
            raise Exception('BagProject is not given.')

        if not erase:
            used_names = set(self.prj.get_cells_in_library(lib_name))
        else:
            used_names = set()
        self.update_structure(lib_name, top_cell_name=top_cell_name, prefix=prefix,
                              suffix=suffix, used_names=used_names)

        self.prj.implement_design(lib_name, self, lib_path=lib_path)

    def populate_hierarchy_graph(self, lib_name, hier_graph, used_concrete_names,
                                 cur_cell_name, prefix, suffix, inst_identifier):
        """Internal method.  Populate the instance hierarchy graph.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the generated schematics.
        hier_graph : :class:`networkx.MultiDiGraph`
            a multi-edge directed graph representing physical design hierarchy.
        used_concrete_names : set[str]
            a set of used concrete schematic cell names.
        cur_cell_name : str
            the cell name of the current design.
        prefix : str
            prefix to add to cell names.
        suffix : str
            suffix to add to cell names.
        inst_identifier : str
            instance identifier string

        Returns
        -------
        cell_id : :class:`bag.design.module.ModuleID`
            the PyNetlistCell representing this module.
        """
        if self.is_primitive():
            self._generated_lib_name = self.lib_name
            self._generated_cell_name = self.get_cell_name_from_parameters()
            return None

        if not cur_cell_name:
            cur_cell_name = self._cell_name

        cell_id = ModuleID(self.lib_name, self.cell_name, self.get_iden_str())
        if cell_id not in hier_graph:
            # find new unsed name
            new_concrete_cell_name = prefix + cur_cell_name + suffix
            if new_concrete_cell_name in used_concrete_names:
                new_concrete_cell_name = '%s%s_%s%s' % (prefix, cur_cell_name, inst_identifier, suffix)
            counter = 0
            while new_concrete_cell_name in used_concrete_names:
                counter += 1
                new_concrete_cell_name = '%s%s_%s_%d%s' % (prefix, cur_cell_name, inst_identifier, counter, suffix)

            # update hierarchy graph and set generated lib/cell name
            self._generated_lib_name = lib_name
            self._generated_cell_name = new_concrete_cell_name

            used_concrete_names.add(new_concrete_cell_name)
            hier_graph.add_node(cell_id, concrete_cell_name=new_concrete_cell_name,
                                pin_map=self.pin_map, new_pins=self.new_pins)

            concrete_inst_map = {}
            for inst_name, rinst_list in sorted(self.instance_map.items()):
                module_list = self.instances[inst_name]
                if not isinstance(module_list, list):
                    module_list = [module_list]

                child_inst_id = '%s_%s' % (inst_identifier, inst_name)
                concrete_rinst_list = []
                for modul, rinst in zip(module_list, rinst_list):
                    child_cell_name = modul.cell_name
                    child_id = modul.populate_hierarchy_graph(lib_name, hier_graph, used_concrete_names,
                                                              child_cell_name, prefix, suffix, child_inst_id)
                    if not modul.is_primitive():
                        hier_graph.add_edge(cell_id, child_id)
                        concrete_lib_name = None
                        concrete_cell_name = hier_graph.node[child_id]['concrete_cell_name']
                    else:
                        # bag primitive
                        concrete_lib_name = modul.lib_name
                        concrete_cell_name = rinst['cell_name']
                    concrete_rinst = dict(
                        name=rinst['name'],
                        lib_name=concrete_lib_name,
                        cell_name=concrete_cell_name,
                        params=rinst['params'],
                        term_mapping=rinst['term_mapping'],
                    )
                    concrete_rinst_list.append(concrete_rinst)
                concrete_inst_map[inst_name] = concrete_rinst_list

            hier_graph.node[cell_id]['inst_map'] = concrete_inst_map

        return cell_id


class MosModuleBase(Module):
    """The base design class for the bag primitive transistor.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    yaml_file : str
        the netlist information file name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    """

    def __init__(self, database, yaml_file, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, yaml_file, parent=parent, prj=prj, **kwargs)
        self.parameters['w'] = None
        self.parameters['l'] = None
        self.parameters['nf'] = None
        self.parameters['intent'] = None

    def get_iden_str(self):
        """Internal Method.  Returns a string that uniquely identifies this module.

        BAG primitives should override this method to return a string based on its parameters.

        Returns
        -------
        iden_str : str
            an identification string for this module.
        """
        # check all parameters are set.
        for key, val in self.parameters.items():
            if val is None:
                raise Exception('Parameter %s unset' % key)

        w_str = self.parameters['w']
        l_str = self.parameters['l']
        nf = self.parameters['nf']
        intent = self.parameters['intent']

        return '%s__%s(%s,%s,%d,%s)' % (self.lib_name, self.cell_name, w_str, l_str, nf, intent)

    def design(self, w=1e-6, l=60e-9, nf=1, intent='standard'):
        """Create a transistor with the given parameters.

        Parameters
        ----------
        w : float or int
            the width/number of fins of this transsitor.
        l : float
            the channel length of this transistor.
        nf : int
            number of fingers of this transistor.
        intent : str
            the design intent of this transistor.  In other words,
            the threshold voltage flavor.
        """
        w_res = self.tech_info.tech_params['mos']['width_resolution']
        l_res = self.tech_info.tech_params['mos']['length_resolution']
        self.parameters['w'] = float_to_si_string(round(w / w_res) * w_res)
        self.parameters['l'] = float_to_si_string(round(l / l_res) * l_res)
        self.parameters['nf'] = nf
        self.parameters['intent'] = intent

    def get_schematic_parameters(self):
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : dict[str, str]
            the schematic parameter dictionary.
        """
        return {'w': self.parameters['w'],
                'l': self.parameters['l'],
                'nf': '%d' % self.parameters['nf'],
                }

    def get_cell_name_from_parameters(self):
        """Returns new cell name based on parameters.

        NOTE: This method is only used by BAG primitives.  This method
        enables a BAG primitive to change the cell master based on
        design parameters (e.g. change transistor instance based on the
        intent parameter).  Custom subclasses should not override this
        method.

        Returns
        -------
        cell : str
            the cell name based on parameters.
        """
        mos_type = self.cell_name.split('_')[0]
        return '%s_%s' % (mos_type, self.parameters['intent'])

    def is_primitive(self):
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG primitives.  This method prevents
        BAG primitives from being copied during design implementation.  Custom
        subclasses should not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return True

    def should_delete_instance(self):
        """Returns True if this instance should be deleted based on its parameters.

        This method is mainly used to delete 0 finger or 0 width transistors.  However,
        You can override this method if there exists parameter settings which corresponds
        to an empty schematic.

        Returns
        -------
        delete : bool
            True if parent should delete this instance.
        """
        return self.parameters['nf'] == 0 or self.parameters['w'] == 0


class GenericModule(Module):
    """A generic module.  Used to represent schematic instances that aren't imported into Python.

    Schematic parameters can be set by modifying the self.parameters dictionary.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    lib_name : str
        the library name.
    cell_name : str
        the cell name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    **kwargs :
        additional arguments.
    """

    def __init__(self, database, lib_name, cell_name, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, None, parent=parent, prj=prj, lib_name=lib_name, cell_name=cell_name, **kwargs)

    def is_primitive(self):
        return True

    def design(self):
        pass

    def get_schematic_parameters(self):
        return self.parameters.copy()


class ResIdealModuleBase(Module):
    """The base design class for an ideal resistor.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    yaml_file : str
        the netlist information file name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    """

    def __init__(self, database, yaml_file, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, yaml_file, parent=parent, prj=prj, **kwargs)
        self.parameters['res'] = None

    def get_iden_str(self):
        """Internal Method.  Returns a string that uniquely identifies this module.

        BAG primitives should override this method to return a string based on its parameters.

        Returns
        -------
        iden_str : str
            an identification string for this module.
        """
        # check all parameters are set.
        for key, val in self.parameters.items():
            if val is None:
                raise Exception('Parameter %s unset' % key)

        res_str = self.parameters['res']

        return '%s__%s(%s)' % (self.lib_name, self.cell_name, res_str)

    def design(self, res=1e3):
        """Create an ideal resistor.

        Parameters
        ----------
        res : float
            the resistance, in Ohms.
        """
        self.parameters['res'] = float_to_si_string(res)

    def get_schematic_parameters(self):
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : dict[str, str]
            the schematic parameter dictionary.
        """
        return {'res': self.parameters['res'],
                }

    def is_primitive(self):
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG primitives.  This method prevents
        BAG primitives from being copied during design implementation.  Custom
        subclasses should not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return True


class ResPhysicalModuleBase(Module):
    """The base design class for a real resistor parameterized by width and length.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    yaml_file : str
        the netlist information file name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    """

    def __init__(self, database, yaml_file, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, yaml_file, parent=parent, prj=prj, **kwargs)
        self.parameters['w'] = None
        self.parameters['l'] = None
        self.parameters['intent'] = None

    def get_iden_str(self):
        """Internal Method.  Returns a string that uniquely identifies this module.

        BAG primitives should override this method to return a string based on its parameters.

        Returns
        -------
        iden_str : str
            an identification string for this module.
        """
        # check all parameters are set.
        for key, val in self.parameters.items():
            if val is None:
                raise Exception('Parameter %s unset' % key)

        w_str = self.parameters['w']
        l_str = self.parameters['l']
        intent = self.parameters['intent']

        return '%s__%s(%s,%s,%s)' % (self.lib_name, self.cell_name, w_str, l_str, intent)

    def design(self, w=1e-6, l=1e-6, intent='standard'):
        """Create a physical resistor.

        Parameters
        ----------
        w : float
            width of the resistor, in meters.
        l : float
            length of the resistor, in meters.
        intent : str
            the resistor design intent, i.e. the type of resistor to use.
        """
        self.parameters['w'] = float_to_si_string(w)
        self.parameters['l'] = float_to_si_string(l)
        self.parameters['intent'] = intent

    def get_schematic_parameters(self):
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : dict[str, str]
            the schematic parameter dictionary.
        """
        return {'w': self.parameters['w'],
                'l': self.parameters['l'],
                }

    def get_cell_name_from_parameters(self):
        """Returns new cell name based on parameters.

        NOTE: This method is only used by BAG primitives.  This method
        enables a BAG primitive to change the cell master based on
        design parameters (e.g. change transistor instance based on the
        intent parameter).  Custom subclasses should not override this
        method.

        Returns
        -------
        cell : str
            the cell name based on parameters.
        """
        return 'res_%s' % self.parameters['intent']

    def is_primitive(self):
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG primitives.  This method prevents
        BAG primitives from being copied during design implementation.  Custom
        subclasses should not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return True


class ResMetalModule(Module):
    """The base design class for a metal resistor.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    yaml_file : str
        the netlist information file name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    """

    param_list = ['w', 'l', 'layer']

    def __init__(self, database, yaml_file, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, yaml_file, parent=parent, prj=prj, **kwargs)
        for par in self.param_list:
            self.parameters[par] = None

    def design(self, w=1e-6, l=1e-6, layer=1):
        """Create a metal resistor.

        Parameters
        ----------
        w : float
            the resistor width, in meters.
        l: float
            the resistor length, in meters.
        layer : int
            the metal layer ID.
        """
        local_dict = locals()
        for name in self.param_list:
            if name not in local_dict:
                raise ValueError('Parameter %s not specified.' % name)
            self.parameters[name] = local_dict[name]

        # get technology parameters
        tech_dict = self.tech_info.tech_params['res_metal']
        lib_name = tech_dict['lib_name']
        l_name = tech_dict['l_name']
        w_name = tech_dict['w_name']
        cell_name = tech_dict['cell_table'][layer]

        self.replace_instance_master('R0', lib_name, cell_name, static=True)
        self.instances['R0'].parameters[l_name] = float_to_si_string(l, precision=6)
        self.instances['R0'].parameters[w_name] = float_to_si_string(w, precision=6)
        self.instances['R0'].parameters[w_name] = float_to_si_string(w, precision=6)
        for key, val in tech_dict['others'].items():
            if isinstance(val, float):
                val = float_to_si_string(val, precision=6)
            elif isinstance(val, int):
                val = '%d' % val
            elif isinstance(val, bool) or isinstance(val, str):
                pass
            else:
                raise ValueError('unsupported type: %s' % type(val))

            self.instances['R0'].parameters[key] = val


class CapIdealModuleBase(Module):
    """The base design class for an ideal capacitor.

    Parameters
    ----------
    database : :class:`bag.design.Database`
        the design database object.
    yaml_file : str
        the netlist information file name.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`bag.BagProject` or None
        the BagProject instance.  Used to implement design.
    """

    def __init__(self, database, yaml_file, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, yaml_file, parent=parent, prj=prj, **kwargs)
        self.parameters['cap'] = None

    def get_iden_str(self):
        """Internal Method.  Returns a string that uniquely identifies this module.

        BAG primitives should override this method to return a string based on its parameters.

        Returns
        -------
        iden_str : str
            an identification string for this module.
        """
        # check all parameters are set.
        for key, val in self.parameters.items():
            if val is None:
                raise Exception('Parameter %s unset' % key)

        cap_str = self.parameters['cap']

        return '%s__%s(%s)' % (self.lib_name, self.cell_name, cap_str)

    def design(self, cap=1e-12):
        """Create an ideal capacitor.

        Parameters
        ----------
        cap : float
            the capacitance, in Farads.
        """
        self.parameters['cap'] = float_to_si_string(cap)

    def get_schematic_parameters(self):
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : dict[str, str]
            the schematic parameter dictionary.
        """
        return {'cap': self.parameters['cap'],
                }

    def is_primitive(self):
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG primitives.  This method prevents
        BAG primitives from being copied during design implementation.  Custom
        subclasses should not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return True
