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


"""This module implements all CAD database manipulations using skill commands.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import yaml
from typing import List, Dict, Optional, Any, Tuple

from jinja2 import Template

import bag.io
from .database import DbAccess

calibre_tmp = bag.io.read_resource(bag.__name__, os.path.join('virtuoso_files', 'calibreview_setup.pytemp'))

try:
    import cybagoa
except ImportError:
    cybagoa = None


def _dict_to_pcell_params(table):
    """Convert given parameter dictionary to pcell parameter list format.

    Parameters
    ----------
    table : dict[str, any]
        the parameter dictionary.

    Returns
    -------
    param_list : list[any]
        the Pcell parameter list
    """
    param_list = []
    for key, val in table.items():
        # python 2/3 compatibility: convert raw bytes to string.
        val = bag.io.fix_string(val)
        if isinstance(val, float):
            param_list.append([key, "float", val])
        elif isinstance(val, str):
            # unicode string
            param_list.append([key, "string", val])
        elif isinstance(val, int):
            param_list.append([key, "int", val])
        else:
            raise Exception('Unsupported parameter %s with type: %s' % (key, type(val)))

    return param_list


def to_skill_list_str(pylist):
    """Convert given python list to a skill list string.

    Parameters
    ----------
    pylist : list[str]
        a list of string.

    Returns
    -------
    ans : str
        a string representation of the equivalent skill list.

    """
    content = ' '.join(('"%s"' % val for val in pylist))
    return "'( %s )" % content


def _handle_reply(reply):
    """Process the given reply."""
    if isinstance(reply, dict):
        if reply.get('type') == 'error':
            if 'data' not in reply:
                raise Exception('Unknown reply format: %s' % reply)
            raise VirtuosoException(reply['data'])
        else:
            try:
                return reply['data']
            except Exception:
                raise Exception('Unknown reply format: %s' % reply)
    else:
        raise Exception('Unknown reply format: %s' % reply)


class VirtuosoException(Exception):
    """Exception raised when Virtuoso returns an error."""

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class SkillInterface(DbAccess):
    """Skill interface between bag and Virtuoso.

    This class sends all bag's database and simulation operations to
    an external Virtuoso process, then get the result from it.

    Parameters
    ----------
    dealer : :class:`bag.interface.ZMQDealer`
        the socket used to communicate with :class:`~bag.interface.SkillOceanServer`.
    tmp_dir : string
        temporary file directory for DbAccess.
    db_config : dict[str, any]
        the database configuration dictionary.
    """

    def __init__(self, dealer, tmp_dir, db_config):
        """Initialize a new SkillInterface object.
        """
        DbAccess.__init__(self, tmp_dir, db_config)
        self.handler = dealer

    def close(self):
        """Terminate the database server gracefully.
        """
        self.handler.send_obj(dict(type='exit'))
        self.handler.close()

    def _eval_skill(self, expr, input_files=None, out_file=None):
        # type: (str, Optional[Dict[str, Any]], Optional[str]) -> str
        """Send a request to evaluate the given skill expression.

        Because Virtuoso has a limit on the input/output data (< 4096 bytes),
        if your input is large, you need to write it to a file and have
        Virtuoso open the file to parse it.  Similarly, if you expect a
        large output, you need to make Virtuoso write the result to the
        file, then read it yourself.  The parameters input_files and
        out_file help you achieve this functionality.

        For example, if you need to evaluate "skill_fun(arg fname)", where
        arg is a file containing the list [1 2 3], and fname is the output
        file name, you will call this function with:

        expr = "skill_fun({arg} {fname})"
        input_files = { "arg": [1 2 3] }
        out_file = "fname"

        the bag server will then a temporary file for arg and fname, write
        the list [1 2 3] into the file for arg, call Virtuoso, then read
        the output file fname and return the result.

        Parameters
        ----------
        expr : string
            the skill expression to evaluate.
        input_files : dict[string, any] or None
            A dictionary of input files content.
        out_file : string or None
            the output file name argument in expr.

        Returns
        -------
        result : str
            a string representation of the result.

        Raises
        ------
        :class: `.VirtuosoException` :
            if virtuoso encounters errors while evaluating the expression.
        """
        request = dict(
            type='skill',
            expr=expr,
            input_files=input_files,
            out_file=out_file,
        )

        self.handler.send_obj(request)
        reply = self.handler.recv_obj()
        return _handle_reply(reply)

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
        exc_libs = to_skill_list_str(self.db_config['schematic']['exclude_libraries'])
        cmd = 'parse_cad_sch( "%s" "%s" %s {netlist_info} )' % (lib_name, cell_name, exc_libs)
        return self._eval_skill(cmd, out_file='netlist_info')

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
        cmd = 'get_cells_in_library_file( "%s" {cell_file} )' % lib_name
        return self._eval_skill(cmd, out_file='cell_file').split()

    def create_library(self, lib_name, lib_path=''):
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : string
            the library name.
        lib_path : string
            directory to create the library in.  If Empty, use default location.
        """
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']
        return self._eval_skill('create_or_erase_library("%s" "%s" "%s" nil)' % (lib_name, tech_lib, lib_path))

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
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']

        if cybagoa is not None and self.db_config['schematic'].get('use_cybagoa', False):
            cds_lib_path = os.environ.get('CDS_LIB_PATH', './cds.lib')
            sch_name = 'schematic'
            sym_name = 'symbol'
            encoding = bag.io.get_encoding()
            # release write locks
            cell_view_list = []
            for _, _, cell_name in template_list:
                cell_view_list.append((cell_name, sch_name))
                cell_view_list.append((cell_name, sym_name))
            self.release_write_locks(lib_name, cell_view_list)

            # create library in case it doesn't exist
            self.create_library(lib_name, lib_path)

            # write schematic
            with cybagoa.PyOASchematicWriter(cds_lib_path, lib_name, encoding) as writer:
                for temp_info, change_info in zip(template_list, change_list):
                    sch_cell = cybagoa.PySchCell(temp_info[0], temp_info[1], temp_info[2], encoding)
                    for old_pin, new_pin in change_info['pin_map']:
                        sch_cell.rename_pin(old_pin, new_pin)
                    for inst_name, rinst_list in change_info['inst_list']:
                        sch_cell.add_inst(inst_name, lib_name, rinst_list)
                    writer.add_sch_cell(sch_cell)
                writer.create_schematics(sch_name, sym_name)

            copy = 'nil'
        else:
            copy = "'t"

        in_files = {'template_list': template_list,
                    'change_list': change_list}
        sympin = to_skill_list_str(self.db_config['schematic']['sympin'])
        ipin = to_skill_list_str(self.db_config['schematic']['ipin'])
        opin = to_skill_list_str(self.db_config['schematic']['opin'])
        iopin = to_skill_list_str(self.db_config['schematic']['iopin'])
        simulators = to_skill_list_str(self.db_config['schematic']['simulators'])
        cmd = ('create_concrete_schematic( "%s" "%s" "%s" {template_list} '
               '{change_list} %s %s %s %s %s %s)' % (lib_name, tech_lib, lib_path,
                                                     sympin, ipin, opin, iopin, simulators, copy))

        return self._eval_skill(cmd, input_files=in_files)

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
        new_lib_path = new_lib_path or self.default_lib_path

        tb_config = self.db_config['testbench']

        cmd = ('instantiate_testbench("{tb_lib}" "{tb_cell}" "{targ_lib}" "{dut_lib}" "{dut_cell}" ' +
               '"{config_libs}" "{config_views}" "{config_stops}" ' +
               '"{default_corner}" "{corner_file}" {def_files} ' +
               '"{tech_lib}" "{new_lib_path}" {result_file})')
        cmd = cmd.format(tb_lib=tb_lib,
                         tb_cell=tb_cell,
                         targ_lib=targ_lib,
                         dut_lib=dut_lib,
                         dut_cell=dut_cell,
                         config_libs=tb_config['config_libs'],
                         config_views=tb_config['config_views'],
                         config_stops=tb_config['config_stops'],
                         default_corner=tb_config['default_env'],
                         corner_file=tb_config['env_file'],
                         def_files=to_skill_list_str(tb_config['def_files']),
                         tech_lib=self.db_config['schematic']['tech_lib'],
                         new_lib_path=new_lib_path,
                         result_file='{result_file}')
        output = yaml.load(self._eval_skill(cmd, out_file='result_file'))
        return tb_config['default_env'], output['corners'], output['parameters'], output['outputs']

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
        cmd = 'get_testbench_info("{tb_lib}" "{tb_cell}" {result_file})'
        cmd = cmd.format(tb_lib=tb_lib,
                         tb_cell=tb_cell,
                         result_file='{result_file}')
        output = yaml.load(self._eval_skill(cmd, out_file='result_file'))
        return output['enabled_corners'], output['corners'], output['parameters'], output['outputs']

    def update_testbench(self, lib, cell, parameters, sim_envs, config_rules, env_parameters):
        # type: (str, str, Dict[str, str], List[str], List[List[str]], List[List[Tuple[str, str]]]) -> None
        """Update the given testbench configuration.

        Parameters
        ----------
        lib : str
            testbench library.
        cell : str
            testbench cell.
        parameters : Dict[str, str]
            testbench parameters.
        sim_envs : List[str]
            list of enabled simulation environments.
        config_rules : List[List[str]]
            config view mapping rules, list of (lib, cell, view) rules.
        env_parameters : List[List[Tuple[str, str]]]
            list of param/value list for each simulation environment.
        """

        cmd = 'modify_testbench("%s" "%s" {conf_rules} {run_opts} {sim_envs} {params} {env_params})' % (lib, cell)
        in_files = {'conf_rules': config_rules,
                    'run_opts': [],
                    'sim_envs': sim_envs,
                    'params': list(parameters.items()),
                    'env_params': list(zip(sim_envs, env_parameters)),
                    }
        self._eval_skill(cmd, input_files=in_files)

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
        # create library in case it doesn't exist
        self.create_library(lib_name)

        # convert parameter dictionary to pcell params list format
        param_list = _dict_to_pcell_params(params)

        cmd = ('create_layout_with_pcell( "%s" "%s" "%s" "%s" "%s"'
               '{params} {pin_mapping} )' % (lib_name, cell_name,
                                             view_name, inst_lib, inst_cell))
        in_files = {'params': param_list, 'pin_mapping': list(pin_mapping.items())}
        return self._eval_skill(cmd, input_files=in_files)

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
        # create library in case it doesn't exist
        self.create_library(lib_name)

        # convert parameter dictionary to pcell params list format
        new_layout_list = []
        for info_list in layout_list:
            new_inst_list = []
            for inst in info_list[1]:
                if 'params' in inst:
                    inst = inst.copy()
                    inst['params'] = _dict_to_pcell_params(inst['params'])
                new_inst_list.append(inst)

            new_info_list = info_list[:]
            new_info_list[1] = new_inst_list
            new_layout_list.append(new_info_list)

        cmd = 'create_layout( "%s" "%s" "%s" {layout_list} )' % (lib_name, view_name, via_tech)
        in_files = {'layout_list': new_layout_list}
        return self._eval_skill(cmd, input_files=in_files)

    def release_write_locks(self, lib_name, cell_view_list):
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : string
            the library name.
        cell_view_list : List[(string, string)]
            list of cell/view name tuples.
        """
        cmd = 'release_write_locks( "%s" {cell_view_list} )' % lib_name
        in_files = {'cell_view_list': cell_view_list}
        return self._eval_skill(cmd, input_files=in_files)

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
        if self.checker is None:
            raise Exception('LVS/RCX is disabled.')

        return self.checker.run_lvs(lib_name, cell_name, sch_view, lay_view, lvs_params,
                                    block=block, callback=callback)

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
        if self.checker is None:
            raise Exception('LVS/RCX is disabled.')

        value, log_fname = self.checker.run_rcx(lib_name, cell_name, sch_view, lay_view, rcx_params,
                                                block=block, callback=callback)
        if block:
            if create_schematic:
                if value is None:
                    return False, log_fname
                self.create_schematic_from_netlist(value, lib_name, cell_name)
                return True, log_fname
            else:
                return value, log_fname
        else:
            return value, log_fname

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
        calview_config = self.db_config['calibreview']
        cell_map = calview_config['cell_map']
        sch_view = sch_view or calview_config['view_name']

        # create calibre view config file
        content = Template(calibre_tmp).render(netlist_file=netlist,
                                               lib_name=lib_name,
                                               cell_name=cell_name,
                                               calibre_cellmap=cell_map,
                                               view_name=sch_view)
        with bag.io.open_temp(prefix='calview', dir=self.tmp_dir, delete=False) as f:
            fname = f.name
            f.write(content)

        # delete old calibre view
        cmd = 'delete_cellview( "%s" "%s" "%s" )' % (lib_name, cell_name, sch_view)
        self._eval_skill(cmd)
        # make extracted schematic
        cmd = 'mgc_rve_load_setup_file( "%s" )' % fname
        self._eval_skill(cmd)

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
        # delete old verilog view
        cmd = 'delete_cellview( "%s" "%s" "verilog" )' % (lib_name, cell_name)
        self._eval_skill(cmd)
        cmd = 'schInstallHDL("%s" "%s" "verilog" "%s" t)' % (lib_name, cell_name, verilog_file)
        self._eval_skill(cmd)
