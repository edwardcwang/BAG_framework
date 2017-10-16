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
from typing import List, Dict, Optional, Tuple, Any

import networkx as nx

from bag import float_to_si_string
import bag.io


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


class Module(with_metaclass(abc.ABCMeta, object)):
    """The base class of all design classes.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    database : :class:`~bag.design.Database`
        the design database object.
    yaml_fname : Optional[str]
        the netlist information file name.  If this is a generic module, this should be None.
    parent : :class:`bag.design.Module`
        the parent of this design module.  None if this is the top level design.
    prj : :class:`~bag.core.BagProject` or None
        the BagProject instance.  Used to implement design.

    Attributes
    ----------
    parameters : dict[str, any]
        the design parameters dictionary.
    instances : dict[str, None or :class:`~bag.design.Module` or list[:class:`~bag.design.Module`]]
        the instance dictionary.
    """

    def __init__(self, database, yaml_fname, parent=None, prj=None, **kwargs):
        self.prj = prj
        self.database = database
        self.tech_info = database.tech_info
        self.parent = parent
        self.hierarchy_graph = None
        self.parameters = {}
        self.instances = {}
        self.instance_map = {}
        self.pin_map = {}
        self._generated_lib_name = None
        self._generated_cell_name = None
        self.orig_instances = {}
        self.new_pins = []

        if yaml_fname is not None:
            self._yaml_fname = os.path.abspath(yaml_fname)
            self.sch_info = bag.io.read_yaml(self._yaml_fname)

            self._lib_name = self.sch_info['lib_name']
            self._cell_name = self.sch_info['cell_name']

            # populate instance map
            for inst_name, inst_attr in self.sch_info['instances'].items():
                lib_name = inst_attr['lib_name']
                cell_name = inst_attr['cell_name']
                modul = self.database.make_design_module(lib_name, cell_name, parent=self, prj=prj)
                rinst = dict(name=inst_name,
                             cell_name=cell_name,
                             params={},
                             term_mapping={},
                             )
                self.instances[inst_name] = modul
                self.instance_map[inst_name] = [rinst, ]
                self.orig_instances[inst_name] = (lib_name, cell_name)

            # fill in pin map
            for pin in self.sch_info['pins']:
                self.pin_map[pin] = pin
        else:
            # This is a dummy module.
            self._yaml_fname = None
            self.sch_info = None
            self._lib_name = kwargs['lib_name']
            self._cell_name = kwargs['cell_name']

    @property
    def generated_lib_name(self):
        """The generated instance library name"""
        return self._generated_lib_name

    @property
    def generated_cell_name(self):
        """The sgenerated instance cell name"""
        return self._generated_cell_name

    @property
    def lib_name(self):
        """The schematic generator library name"""
        return self._lib_name

    @property
    def cell_name(self):
        """The schematic generator cell name"""
        return self._cell_name

    @abc.abstractmethod
    def design(self):
        """To be overridden by subclasses to design this module.

        To design instances of this module, you can
        call their :meth:`.design` method or any other ways you coded.

        To modify schematic structure, call:

        :meth:`.rename_pin`

        :meth:`.delete_instance`

        :meth:`.replace_instance_master`

        :meth:`.reconnect_instance_terminal`

        :meth:`.restore_instance`

        :meth:`.array_instance`
        """
        pass

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
        return {}

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
        return self.cell_name

    def is_primitive(self):
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
        """Removes a pin from this schematic.

        Parameters
        ----------
        remove_pin : str
            the pin to remove.
        """
        self.rename_pin(remove_pin, '')

    def delete_instance(self, inst_name):
        """Delete the instance with the given name.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.
        """
        self.instances[inst_name] = None
        self.instance_map[inst_name] = []

    def replace_instance_master(self, inst_name, lib_name, cell_name, static=False, index=None):
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

        new_module = self.database.make_design_module(lib_name, cell_name, parent=self, static=static)

        # check if this is arrayed
        if index is not None and isinstance(self.instances[inst_name], list):
            rinst_name = self.instance_map[inst_name][index]['name']
            rinst = dict(name=rinst_name, cell_name=cell_name, params={}, term_mapping={})
            self.instances[inst_name][index] = new_module
            self.instance_map[inst_name][index] = rinst
        else:
            rinst = dict(name=inst_name, cell_name=cell_name, params={}, term_mapping={})
            self.instances[inst_name] = new_module
            self.instance_map[inst_name] = [rinst, ]

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
                self.instance_map[inst_name][index]['term_mapping'][term_name] = net_name
            else:
                raise ValueError('If index is not None, both term_name and net_name must be string.')
        else:
            # modify terminal connection for all instances in the array
            rinst_list = self.instance_map[inst_name]
            if not isinstance(term_name, list) and not isinstance(term_name, tuple):
                if not isinstance(term_name, str):
                    raise ValueError('term_name = %s must be string.' % term_name)
                term_name = [term_name] * len(rinst_list)
            else:
                if len(term_name) != len(rinst_list):
                    raise ValueError('term_name length = %d != %d' % (len(term_name), len(rinst_list)))

            if not isinstance(net_name, list) and not isinstance(net_name, tuple):
                if not isinstance(net_name, str):
                    raise ValueError('net_name = %s must be string.' % net_name)
                net_name = [net_name] * len(rinst_list)
            else:
                if len(net_name) != len(rinst_list):
                    raise ValueError('net_name length = %d != %d' % (len(net_name), len(rinst_list)))

            for rinst, tname, nname in zip(rinst_list, term_name, net_name):
                rinst['term_mapping'][tname] = nname

    def restore_instance(self, inst_name):
        """Restore a instance to the original template state.

        This method is useful if you decide to un-array an instance, recover a deleted
        instance, or so on.

        Parameters
        ----------
        inst_name : str
            the instance to restore.
        """
        lib_name, cell_name = self.orig_instances[inst_name]
        modul = self.database.make_design_module(lib_name, cell_name, parent=self)
        rinst = dict(name=inst_name,
                     cell_name=cell_name,
                     params={},
                     term_mapping={},
                     )

        self.instances[inst_name] = modul
        self.instance_map[inst_name] = [rinst, ]

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
            True if all modules in this array is identical.
        """
        num_inst = len(inst_name_list)
        if not term_list:
            term_list = [{} for _ in range(num_inst)]
        if num_inst != len(term_list):
            msg = 'len(inst_name_list) = %d != len(term_list) = %d'
            raise ValueError(msg % (num_inst, len(term_list)))

        orig_module = self.instances[inst_name]
        lib_name, cell_name = orig_module.lib_name, orig_module.cell_name
        rinst_list = []
        module_list = []
        if not same:
            for iname, iterm in zip(inst_name_list, term_list):
                modul = self.database.make_design_module(lib_name, cell_name, parent=self)
                rinst = dict(name=iname,
                             cell_name=cell_name,
                             params={},
                             term_mapping=iterm,
                             )
                rinst_list.append(rinst)
                module_list.append(modul)
        else:
            module_list = [orig_module] * num_inst
            rinst_list = [dict(name=iname, cell_name=cell_name, params={}, term_mapping=iterm)
                          for iname, iterm in zip(inst_name_list, term_list)]

        self.instances[inst_name] = module_list
        self.instance_map[inst_name] = rinst_list

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

    def get_iden_str(self):
        """Internal Method.  Returns a string that uniquely identifies this module.

        BAG primitives should override this method to return a string based on its parameters.

        Returns
        -------
        iden_str : str
            an identification string for this module.
        """
        # Note old way fails because it doesn't differentiate between two block with
        # identical sub-blocks but different connections.

        # prim_dict = self.get_primitives_descendants('XTOP')
        # return '\n'.join('%s:%s' % (n, prim_dict[n].get_iden_str()) for n in sorted(prim_dict.keys()))
        return repr(self.to_immutable_id(self.parameters))

    @classmethod
    def to_immutable_id(cls, val):
        """Convert the given object to an immutable type for use as keys in dictionary.
        """
        # python 2/3 compatibility: convert raw bytes to string
        val = bag.io.fix_string(val)

        if val is None or isinstance(val, int) or isinstance(val, str) or isinstance(val, float):
            return val
        elif isinstance(val, list) or isinstance(val, tuple):
            return tuple((cls.to_immutable_id(item) for item in val))
        elif isinstance(val, dict):
            return tuple(((k, cls.to_immutable_id(val[k])) for k in sorted(val.keys())))
        else:
            raise Exception('Unrecognized value %s with type %s' % (str(val), type(val)))

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
