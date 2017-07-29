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

"""This is the core bag module.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import string
import importlib
from typing import List, Optional, Union

# noinspection PyPackageRequirements
import yaml

from . import interface
from . import design
from .layout.core import DummyTechInfo
from .io import read_file, sim_data


def _parse_yaml_file(fname):
    """Parse YAML file with environment variable substitution.

    Parameters
    ----------
    fname : str
        yaml file name.

    Returns
    -------
    table : dict[string, any]
        the yaml file as a dictionary.
    """
    content = read_file(fname)
    # substitute environment variables
    content = string.Template(content).substitute(os.environ)
    return yaml.load(content)


def _get_port_number(port_file):
    """Read the port number from the given port file.

    Parameters
    ----------
    port_file : str
        a file containing the communication port number.

    Returns
    -------
    port : int or None
        the port number if reading is successful.
    msg : str
        Empty string on success, the error message on failure.
    """
    port_file = os.path.basename(port_file)
    if 'BAG_WORK_DIR' not in os.environ:
        return None, 'Environment variable BAG_WORK_DIR not defined'

    work_dir = os.environ['BAG_WORK_DIR']
    if not os.path.isdir(work_dir):
        return None, '$BAG_WORK_DIR = %s is not a directory' % work_dir

    # read port number
    port_file = os.path.join(work_dir, port_file)
    if not os.path.isfile(port_file):
        return None, 'Cannot find port file.'

    port = int(read_file(port_file))
    return port, ''


def _import_class_from_str(class_str):
    """Given a Python class string, convert it to the Python class.

    Parameters
    ----------
    class_str : str
        a Python class string/

    Returns
    -------
    py_class : class
        a Python class.
    """
    sections = class_str.split('.')

    module_str = '.'.join(sections[:-1])
    class_str = sections[-1]
    modul = importlib.import_module(module_str)
    return getattr(modul, class_str)


class Testbench(object):
    """A class that represents a testbench instance.

    Parameters
    ----------
    sim : :class:`bag.interface.simulator.SimAccess`
        The SimAccess instance used to issue simulation commands.
    db : :class:`bag.interface.database.DbAccess`
        The DbAccess instance used to update testbench schematic.
    lib : str
        testbench library.
    cell : str
        testbench cell.
    parameters : dict[str, str]
        the simulation parameter dictionary.  The values are string representation
        of actual parameter values.
    env_list : list[str]
        list of defined simulation environments.
    default_envs : list[str]
        the selected simulation environments.
    outputs : dict[str, str]
        default output expressions
    Attributes
    ----------
    lib : str
        testbench library.
    cell : str
        testbench cell.
    save_dir : str
        directory containing the last simulation data.
    """

    def __init__(self, sim, db, lib, cell, parameters, env_list, default_envs, outputs):
        """Create a new testbench instance.
        """
        self.sim = sim
        self.db = db
        self.lib = lib
        self.cell = cell
        self.parameters = parameters
        self.env_parameters = {}
        self.env_list = env_list
        self.sim_envs = default_envs
        self.config_rules = {}
        self.outputs = outputs
        self.save_dir = None
        self.sim_id = None

    def get_defined_simulation_environments(self):
        """Return a list of defined simulation environments"""
        return self.env_list

    def get_current_simulation_environments(self):
        """Returns a list of simulation environments this testbench will simulate."""
        return self.sim_envs

    def add_output(self, var, expr):
        """Add an output expression to be recorded and exported back to python.

        Parameters
        ----------
        var : str
            output variable name.
        expr : str
            the output expression.
        """
        if var in sim_data.illegal_var_name:
            raise ValueError('Variable name %s is illegal.' % var)
        self.outputs[var] = expr

    def set_parameter(self, name, val, precision=6):
        """Sets the value of the given simulation parameter.

        Parameters
        ----------
        name : str
            parameter name.
        val : int or float
            parameter value
        precision : int
            the parameter value will be rounded to this precision.
        """
        param_config = dict(type='single', value=val)
        self.parameters[name] = self.sim.format_parameter_value(param_config, precision)

    def set_env_parameter(self, name, val_list, precision=6):
        # type: (str, List[float], int) -> None
        """Configure the given parameter to have different value across simulation environments.

        Parameters
        ----------
        name : str
            the parameter name.
        val_list : List[float]
            the parameter values for each simulation environment.  the order of the simulation
            environments can be found in self.sim_envs
        precision : int
            the parameter value will be rounded to this precision.
        """
        if len(self.sim_envs) != len(val_list):
            raise ValueError('env parameter must have %d values.' % len(self.sim_envs))

        default_val = None
        for env, val in zip(self.sim_envs, val_list):
            if env not in self.env_parameters:
                cur_dict = {}
                self.env_parameters[env] = cur_dict
            else:
                cur_dict = self.env_parameters[env]

            param_config = dict(type='single', value=val)
            cur_val = self.sim.format_parameter_value(param_config, precision)
            if default_val is None:
                default_val = cur_val
            cur_dict[name] = self.sim.format_parameter_value(param_config, precision)
        self.parameters[name] = default_val

    def set_sweep_parameter(self, name, precision=6, **kwargs):
        """Set to sweep the given parameter.

        To set the sweep values directly:

        tb.set_sweep_parameter('var', values=[1.0, 5.0, 10.0])

        To set a linear sweep with start/stop/step (inclusive start and stop):

        tb.set_sweep_parameter('var', start=1.0, stop=9.0, step=4.0)

        To set a logarithmic sweep with points per decade (inclusive start and stop):

        tb.set_sweep_parameter('var', start=1.0, stop=10.0, num_decade=3)

        Parameters
        ----------
        name : str
            parameter name.
        precision : int
            the parameter value will be rounded to this precision.
        **kwargs :
            the sweep parameters.  Refer to the above for example calls.
        """
        if 'values' in kwargs:
            param_config = dict(type='list', values=kwargs['values'])
        elif 'start' in kwargs and 'stop' in kwargs:
            start = kwargs['start']
            stop = kwargs['stop']
            if 'step' in kwargs:
                step = kwargs['step']
                param_config = dict(type='linstep', start=start, stop=stop, step=step)
            elif 'num_decade' in kwargs:
                num = kwargs['num_decade']
                param_config = dict(type='decade', start=start, stop=stop, num=num)
            else:
                raise Exception('Unsupported sweep arguments: %s' % kwargs)
        else:
            raise Exception('Unsupported sweep arguments: %s' % kwargs)

        self.parameters[name] = self.sim.format_parameter_value(param_config, precision)

    def set_simulation_environments(self, env_list):
        """Enable the given list of simulation environments.

        If more than one simulation environment is specified, then a sweep
        will be performed.

        Parameters
        ----------
        env_list : list[str]
        """
        self.sim_envs = env_list

    def set_simulation_view(self, lib_name, cell_name, sim_view):
        """Set the simulation view of the given design.

        For simulation, each design may have multiple views, such as schematic,
        veriloga, extracted, etc.  This method lets you choose which view to
        use for netlisting.  the given design can be the top level design or
        an intermediate instance.

        Parameters
        ----------
        lib_name : str
            design library name.
        cell_name : str
            design cell name.
        sim_view : str
            the view to simulate with.
        """
        key = '%s__%s' % (lib_name, cell_name)
        self.config_rules[key] = sim_view

    def update_testbench(self):
        """Commit the testbench changes to the CAD database.
        """
        config_list = []
        for key, view in self.config_rules.items():
            lib, cell = key.split('__')
            config_list.append([lib, cell, view])

        env_params = []
        for env in self.sim_envs:
            if env in self.env_parameters:
                val_table = self.env_parameters[env]
                env_params.append(list(val_table.items()))
        self.db.update_testbench(self.lib, self.cell, self.parameters, self.sim_envs, config_list,
                                 env_params)

    def run_simulation(self, precision=6, sim_tag=None, block=True, callback=None):
        """Run simulation.

        Parameters
        ----------
        precision : int
            the floating point number precision.
        sim_tag : str or None
            optional description for this simulation run.
        block : bool
            If True, wait for the simulation to finish.  Otherwise, return
            a simulation ID you can use to query simulation status later.
        callback : callable or None
            If given, this function will be called with the save directory
            and process return code when the simulation finished.

        Returns
        -------
        value : str or None
            Either the save directory path or the simulation ID.  If simulation
            is cancelled, return None.
        """
        if self.sim_id is not None:
            raise ValueError('An simulation is currently running with this testbench.')

        retval = self.sim.run_simulation(self.lib, self.cell, self.outputs,
                                         precision=precision, sim_tag=sim_tag,
                                         block=block, callback=callback)
        if block:
            self.save_dir = retval
        else:
            self.sim_id = retval

        return retval

    def wait(self, timeout=None, cancel_timeout=10.0):
        # type: (Optional[float], float) -> Optional[str]
        """Wait for the simulation to finish, then return the results.

        Parameters
        ----------
        timeout : Optional[float]
            number of seconds to wait.  If None, waits indefinitely.
        cancel_timeout : float
            number of seconds to wait for simulation cancellation.

        Returns
        -------
        save_dir : Optional[str]
            the save directory.  None if the simulation is cancelled.
        """
        if self.sim_id is None:
            raise ValueError('No simulation is associated with this testbench.')

        save_dir, retcode = self.sim.wait(self.sim_id, timeout=timeout, cancel_timeout=cancel_timeout)
        if retcode is not None:
            self.save_dir = save_dir
        else:
            save_dir = None

        self.sim_id = None

        return save_dir

    def load_sim_results(self, hist_name, precision=6, block=True, callback=None):
        """Load previous simulation data.

        Parameters
        ----------
        hist_name : str
            the simulation history name.
        precision : int
            the floating point number precision.
        block : bool
            If True, wait for the process to finish.  Otherwise, return
            a process ID you can use to query process status later.
        callback : callable or None
            If given, this function will be called with the save directory
            and process return code when the process finished.

        Returns
        -------
        value : str or None
            Either the save directory path or the simulation ID.  If simulation
            is cancelled, return None.
        """
        retval = self.sim.load_sim_results(self.lib, self.cell, hist_name, self.outputs,
                                           precision=precision, block=block, callback=callback)
        if block:
            self.save_dir = retval

        return retval


class BagProject(object):
    """The main bag controller class.

    This class mainly stores all the user configurations, and issue
    high level bag commands.

    Parameters
    ----------
    bag_config_path : str or None
        the bag configuration file path.  If None, will attempt to read from
        environment variable BAG_CONFIG_PATH.
    port : Optional[int]
        the BAG server process port number.  If not given, will read from port file.

    Attributes
    ----------
    bag_config : dict[string, any]
        the BAG configuration parameters dictionary.
    tech_info : bag.layout.core.TechInfo
        the BAG process technology class.
    """

    def __init__(self, bag_config_path=None, port=None):
        if bag_config_path is None:
            if 'BAG_CONFIG_PATH' not in os.environ:
                raise Exception('BAG_CONFIG_PATH not defined.')
            bag_config_path = os.environ['BAG_CONFIG_PATH']

        self.bag_config = _parse_yaml_file(bag_config_path)
        bag_tmp_dir = os.environ.get('BAG_TEMP_DIR', None)

        # get port files
        if port is None:
            socket_config = self.bag_config['socket']
            if 'port_file' in socket_config:
                port, msg = _get_port_number(socket_config['port_file'])
                if msg:
                    print('*WARNING* %s' % msg)

        # create ZMQDealer object
        dealer_kwargs = {}
        dealer_kwargs.update(self.bag_config['socket'])
        del dealer_kwargs['port_file']

        # create TechInfo instance
        tech_params = _parse_yaml_file(self.bag_config['tech_config_path'])
        if 'class' in tech_params:
            tech_cls = _import_class_from_str(tech_params['class'])
            self.tech_info = tech_cls(tech_params)
        else:
            # just make a default tech_info object as place holder.
            print('*WARNING*: No TechInfo class defined.  Using a dummy version.')
            self.tech_info = DummyTechInfo(tech_params)

        # create design module database.
        sch_exc_libs = self.bag_config['database']['schematic']['exclude_libraries']
        self.dsn_db = design.Database(self.bag_config['lib_defs'], self.tech_info, sch_exc_libs)

        if port is not None:
            # make DbAccess instance.
            dealer = interface.ZMQDealer(port, **dealer_kwargs)
            db_cls = _import_class_from_str(self.bag_config['database']['class'])
            self.impl_db = db_cls(dealer, bag_tmp_dir, self.bag_config['database'])
        else:
            self.impl_db = None

        # make SimAccess instance.
        sim_cls = _import_class_from_str(self.bag_config['simulation']['class'])
        self.sim = sim_cls(bag_tmp_dir, self.bag_config['simulation'])

    def close_bag_server(self):
        """Close the BAG database server."""
        if self.impl_db is not None:
            self.impl_db.close()
            self.impl_db = None

    def close_sim_server(self):
        """Close the BAG simulation server."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def import_design_library(self, lib_name):
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        new_lib_path = self.bag_config['new_lib_path']
        self.impl_db.import_design_library(lib_name, self.dsn_db, new_lib_path)

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
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.get_cells_in_library(lib_name)

    def create_library(self, lib_name, lib_path=''):
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : string
            the library name.
        lib_path : string
            directory to create the library in.  If Empty, use default location.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.create_library(lib_name, lib_path=lib_path)

    def create_design_module(self, lib_name, cell_name, **kwargs):
        """Create a new top level design module for the given schematic template

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        kwargs : dict[str, any]
            optional parameters.

        Returns
        -------
        dsn : :class:`bag.design.Module`
            the DesignModule correspodning to the design template.
        """
        return self.dsn_db.make_design_module(lib_name, cell_name, parent=None, prj=self, **kwargs)

    def implement_design(self, lib_name, design_module, lib_path=''):
        """Implement the given design.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        design_module : bag.Design.module.Module
            the design module to create schematics for.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.implement_design(lib_name, design_module, lib_path=lib_path)

    def configure_testbench(self, tb_lib, tb_cell):
        """Update testbench state for the given testbench.

        This method fill in process-specific information for the given testbench, then returns
        a testbench object which you can use to control simulation.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        tb : :class:`bag.core.Testbench`
            the :class:`~bag.core.Testbench` instance.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')
        if self.sim is None:
            raise Exception('SimAccess is not set up.')

        c, clist, params, outputs = self.impl_db.configure_testbench(tb_lib, tb_cell)
        return Testbench(self.sim, self.impl_db, tb_lib, tb_cell, params, clist, [c], outputs)

    def load_testbench(self, tb_lib, tb_cell):
        """Loads a testbench from the database.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        tb : :class:`bag.core.Testbench`
            the :class:`~bag.core.Testbench` instance.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')
        if self.sim is None:
            raise Exception('SimAccess is not set up.')

        cur_envs, all_envs, params, outputs = self.impl_db.get_testbench_info(tb_lib, tb_cell)
        return Testbench(self.sim, self.impl_db, tb_lib, tb_cell, params, all_envs, cur_envs, outputs)

    def load_sim_results(self, lib, cell, hist_name, outputs, precision=6):
        """Load previous simulation data."""
        save_dir = self.sim.load_sim_results(lib, cell, hist_name, outputs,
                                             precision=precision)

        import bag.data
        results = bag.data.load_sim_results(save_dir)
        return results

    def instantiate_layout_pcell(self, lib_name, cell_name, inst_lib, inst_cell, params,
                                 pin_mapping=None, view_name='layout'):
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : dict[str, any]
            the parameter dictionary.
        pin_mapping: dict[str, str]
            the pin renaming dictionary.
        view_name : str
            layout view name, default is "layout".
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        pin_mapping = pin_mapping or {}
        self.impl_db.instantiate_layout_pcell(lib_name, cell_name, view_name,
                                              inst_lib, inst_cell, params, pin_mapping)

    def instantiate_layout(self, lib_name, view_name, via_tech, layout_list):
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : string
            layout library name.
        view_name : string
            layout view name.
        via_tech : string
            via technology name.
        layout_list : list[any]
            a list of layouts to create
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.instantiate_layout(lib_name, view_name, via_tech, layout_list)

    def release_write_locks(self, lib_name, cell_view_list):
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : string
            the library name.
        cell_view_list : List[(string, string)]
            list of cell/view name tuples.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        self.impl_db.release_write_locks(lib_name, cell_view_list)

    def run_lvs(self, lib_name, cell_name, sch_view='schematic',
                lay_view='layout', lvs_params=None,
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
        lvs_params : dict[str, any] or None
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
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        lvs_params = lvs_params or {}
        return self.impl_db.run_lvs(lib_name, cell_name, sch_view, lay_view, lvs_params,
                                    block=block, callback=callback)

    def run_rcx(self, lib_name, cell_name, sch_view='schematic',
                lay_view='layout', rcx_params=None,
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
        rcx_params : dict[str, any] or None
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
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        rcx_params = rcx_params or {}
        return self.impl_db.run_rcx(lib_name, cell_name, sch_view, lay_view, rcx_params,
                                    block=block, callback=callback,
                                    create_schematic=create_schematic)

    def wait_lvs_rcx(self, job_id, timeout=None, cancel_timeout=10.0):
        # type: (str, Optional[float], float) -> Optional[Union[bool, str]]
        """Wait for the given LVS/RCX job to finish, then return the result.

        If ``timeout`` is None, waits indefinitely.  Otherwise, if after
        ``timeout`` seconds the simulation is still running, a
        :class:`concurrent.futures.TimeoutError` will be raised.
        However, it is safe to catch this error and call wait again.

        If Ctrl-C is pressed before the job is finished or before timeout
        is reached, the job will be cancelled.

        Parameters
        ----------
        job_id : str
            the job ID.
        timeout : float or None
            number of seconds to wait.  If None, waits indefinitely.
        cancel_timeout : float
            number of seconds to wait for job cancellation.

        Returns
        -------
        result : Optional[Union[bool, str]]
            the job result.  None if the job is cancelled.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.wait_lvs_rcx(job_id, timeout=timeout, cancel_timeout=cancel_timeout)

    def cancel(self, job_id, timeout=None):
        # type: (str, Optional[float]) -> Optional[Union[bool, str]]
        """Cancel the given LVS/RCX job.

        If the process haven't started, this method prevents it from started.
        Otherwise, we first send a SIGTERM signal to kill the process.  If
        after ``timeout`` seconds the process is still alive, we will send a
        SIGKILL signal.  If after another ``timeout`` seconds the process is
        still alive, an Exception will be raised.

        Parameters
        ----------
        job_id : str
            the process ID to cancel.
        timeout : float or None
            number of seconds to wait for cancellation.  If None, use default
            timeout.

        Returns
        -------
        output : Optional[Union[bool, str]]
            output of the job if it successfully terminates.
            Otherwise, return None.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.cancel(job_id, timeout=timeout)

    def done(self, job_id):
        # type: (str) -> bool
        """Returns True if the given process finished or is cancelled successfully.

        Parameters
        ----------
        job_id : str
            the process ID.

        Returns
        -------
        done : bool
            True if the process is cancelled or completed.
        """
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.done(job_id)

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
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        return self.impl_db.create_schematic_from_netlist(netlist, lib_name, cell_name,
                                                          sch_view=sch_view, **kwargs)

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
        if self.impl_db is None:
            raise Exception('BAG Server is not set up.')

        verilog_file = os.path.abspath(verilog_file)
        if not os.path.isfile(verilog_file):
            raise ValueError('%s is not a file.' % verilog_file)

        return self.impl_db.create_verilog_view(verilog_file, lib_name, cell_name, **kwargs)
