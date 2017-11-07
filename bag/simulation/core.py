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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
import importlib
import itertools
import os
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, List, Iterable, Sequence

import yaml

from bag import float_to_si_string
from bag.io import read_yaml, open_file, load_sim_results, save_sim_results
from bag.layout import RoutingGrid, TemplateDB

if TYPE_CHECKING:
    from bag.core import BagProject, Testbench


class MeasurementManager(with_metaclass(abc.ABCMeta, object)):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.

    Parameters
    ----------
    data_dir : str
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    dsn_name : str
        DUT cell name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    """
    def __init__(self, data_dir, meas_name, impl_lib, dsn_name, specs):
        # type: (str, str, str, str, Dict[str, Any]) -> None
        self.data_dir = os.path.abspath(data_dir)
        self.impl_lib = impl_lib
        self.dsn_name = dsn_name
        self.meas_name = meas_name
        self.specs = specs

    @abc.abstractmethod
    def get_testbench_info(self, state, prev_output):
        # type: (str, Optional[Dict[str, Any]]) -> Tuple[str, str, Optional[Dict[str, Any]]]
        """Get information about the next testbench.

        Parameters
        ----------
        state : str
            the current FSM state.
        prev_output : Optional[Dict[str, Any]]
            the previous post-processing output.

        Returns
        -------
        tb_name : str
            cell name of the next testbench.
        tb_type : str
            the next testbench type.
        sch_params : Optional[Dict[str, Any]]
            the next testbench schematic parameters.  If we are reusing an existing
            testbench, this should be None.
        """
        return '', '', None

    @abc.abstractmethod
    def setup_testbench(self, state, tb):
        # type: (str, Testbench) -> None
        """Configure the simulation state of the given testbench.

        No need to call update_testbench(); it is called for you.

        Parameters
        ----------
        state : str
            the current FSM state.
        tb : Testbench
            the simulation Testbench instance.
        """
        pass

    @abc.abstractmethod
    def process_output(self, state, data):
        # type: (str, Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : Dict[str, Any]
            simulation data dictionary.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_state : str
            the next FSM state.
        output : Dict[str, Any]
            a dictionary containing post-processed data.
        """
        return False, '', {}

    async def async_measure_performance(self, prj):
        # type: (BagProject) -> Dict[str, Any]
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        prj : BagProject
            the BagProject instance.

        Returns
        -------
        output : Dict[str, Any]
            the last dictionary returned by process_output().
        """
        cur_state = 'init'
        prev_output = None
        done = False

        while not done:
            # create and setup testbench
            tb_name, tb_type, tb_sch_params = self.get_testbench_info(cur_state, prev_output)
            if tb_sch_params is None:
                tb = prj.load_testbench(self.impl_lib, tb_name)
            else:
                tb = self._create_tb_schematic(prj, tb_name, tb_type, tb_sch_params)
            self.setup_testbench(cur_state, tb)
            tb.update_testbench()

            # run simulation and save raw result
            save_dir = await tb.async_run_simulation(sim_tag=cur_state)
            cur_results = load_sim_results(save_dir)
            root_dir = os.path.join(self.data_dir, tb_name)
            os.makedirs(root_dir, exist_ok=True)
            save_sim_results(cur_results, os.path.join(root_dir, '%s.hdf5' % cur_state))

            # process and save simulation data
            done, cur_state, prev_output = self.process_output(cur_state, cur_results)
            with open_file(os.path.join(root_dir, '%s.yaml' % cur_state), 'w') as f:
                yaml.dump(prev_output, f)

        return prev_output

    def get_default_tb_sch_params(self, tb_type):
        # type: (str) -> Dict[str, Any]
        """Helper method to return a default testbench schematic parameters dictionary.

        This method loads default values from specification file, the fill in dut_lib
        and dut_cell for you.

        Parameters
        ----------
        tb_type : str
            the testbench type.

        Returns
        -------
        sch_params : Dict[str, Any]
            the default schematic parameters dictionary.
        """
        tb_specs = self.specs['testbenches'][tb_type]
        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self.impl_lib
        tb_params['dut_cell'] = self.dsn_name
        return tb_params

    def _create_tb_schematic(self, prj, tb_name, tb_type, tb_sch_params):
        # type: (BagProject, str, str, Dict[str, Any]) -> Testbench
        """Helper method to create a testbench schematic.

        Parmaeters
        ----------
        prj : BagProject
            the BagProject instance.
        tb_name : str
            the testbench cell name.
        tb_type : str
            the testbench type.
        tb_sch_params : Dict[str, Any]
            the testbench schematic parameters dictionary.

        Returns
        -------
        tb : Testbench
            the simulation Testbench instance.
        """
        tb_specs = self.specs['testbenches'][tb_type]
        tb_lib = tb_specs['tb_lib']
        tb_cell = tb_specs['tb_cell']

        tb_sch = prj.create_design_module(tb_lib, tb_cell)
        tb_sch.design(**tb_sch_params)
        tb_sch.implement_design(self.impl_lib, top_cell_name=tb_name)

        return prj.configure_testbench(tb_lib, tb_cell)


class DesignManager(object):
    """A class that manages instantiating design instances and running simulations.

    This class provides various methods to allow you to sweep design parameters
    and generate multiple instances at once.  It also provides methods for running
    simulations and helps you interface with TestbenchManager instances.

    Parameters
    ----------
    prj : Optional[BagProject]
        The BagProject instance.
    spec_file : str
        the specification file name or the data directory.
    """

    def __init__(self, prj, spec_file):
        # type: (Optional[BagProject], str) -> None
        self.prj = prj
        self._specs = None

        if os.path.isfile(spec_file):
            self._specs = read_yaml(spec_file)
            root_dir = os.path.abspath(self._specs['root_dir'])
            save_spec_file = os.path.join(root_dir, 'specs.yaml')
        elif os.path.isdir(spec_file):
            root_dir = os.path.abspath(spec_file)
            save_spec_file = spec_file = os.path.join(root_dir, 'specs.yaml')
            self._specs = read_yaml(spec_file)
        else:
            raise ValueError('%s is neither data directory or specification file.' % spec_file)

        self._swp_var_list = tuple(sorted(self._specs['sweep_params'].keys()))

        # save root_dir as absolute path, in this way everything will still work
        # if the user start python from a different directory.
        self._specs['root_dir'] = root_dir
        os.makedirs(root_dir, exist_ok=True)
        with open_file(save_spec_file, 'w') as f:
            yaml.dump(self._specs, f)

    @classmethod
    def load_state(cls, prj, root_dir):
        # type: (BagProject, str) -> DesignManager
        """Create the DesignManager instance corresponding to data in the given directory."""
        return cls(prj, root_dir)

    @classmethod
    def get_testbench_name(cls, dsn_name, tb_type):
        # type: (str, str) -> str
        """Returns the testbench cell name.

        Parameters
        ----------
        dsn_name : str
            design cell name.
        tb_type : str
            testbench type.

        Returns
        -------
        tb_name : str
            testbench cell name
        """
        return '%s_TB_%s' % (dsn_name, tb_type)

    @classmethod
    def get_wrapper_name(cls, dut_name, wrapper_name):
        # type: (str, str) -> str
        """Returns the wrapper cell name corresponding to the given DUT."""
        return '%s_WRAPPER_%s' % (dut_name, wrapper_name)

    @property
    def specs(self):
        # type: () -> Dict[str, Any]
        """Return the specification dictionary."""
        return self._specs

    @property
    def swp_var_list(self):
        # type: () -> Tuple[str, ...]
        return self._swp_var_list

    def modify_tb_schematic(self, tb_type, tb_sch_params):
        # type: (str, Dict[str, Any]) -> None
        """Perform any modifications necessary on the testbench schematic parameters.

        Parameters
        ----------
        tb_type : str
            the testbench type.
        tb_sch_params : Dict[str, Any]
            the testbench schematic parameters dictionary.
        """
        pass

    def modify_tb_setup(self, tb_type, tb):
        # type: (str, Testbench) -> None
        """Perform any modifications necessary on the testbench simulation setup.

        Parameters
        ----------
        tb_type : str
            the testbench type.
        tb : Testbench
            the Testbench instance.
        """
        pass

    def post_simulation_procedure(self, tb_type, tb_manager, results):
        # type: (str, TestbenchManager, Dict[str, Any]) -> None
        """Perform any task necessary after the given simulation finished.

        Parameters
        ----------
        tb_type : str
            the testbench type;
        tb_manager : TestbenchManager
            the TestbenchManager instance.
        results : Dict[str, Any]
            the simulation results.
        """
        pass

    def get_swp_var_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid sweep variable values.

        Parameter
        ---------
        var : str
            the sweep variable name.

        Returns
        -------
        val_list : List[Any]
            the sweep values of the given variable.
        """
        return self.specs['sweep_params'][var]

    def get_combinations_iter(self):
        # type: () -> Iterable[Tuple[Any, ...]]
        """Returns an iterator of schematic parameter combinations we sweep over.

        Returns
        -------
        combo_iter : Iterable[Tuple[Any, ...]]
            an iterator of tuples of schematic parameters values that we sweep over.
        """

        swp_par_dict = self.specs['sweep_params']
        return itertools.product(*(swp_par_dict[var] for var in self.swp_var_list))

    def make_tdb(self):
        # type: () -> TemplateDB
        """Create and return a new TemplateDB object.

        Returns
        -------
        tdb : TemplateDB
            the TemplateDB object.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        target_lib = self.specs['impl_lib']
        grid_specs = self.specs['routing_grid']
        layers = grid_specs['layers']
        spaces = grid_specs['spaces']
        widths = grid_specs['widths']
        bot_dir = grid_specs['bot_dir']

        routing_grid = RoutingGrid(self.prj.tech_info, layers, spaces, widths, bot_dir)
        tdb = TemplateDB('', routing_grid, target_lib, use_cybagoa=True)
        return tdb

    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['layout_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    def create_dut_schematics(self, sch_params_list, cell_name_list, gen_wrappers=True):
        # type: (Sequence[Dict[str, Any]], Sequence[str], bool) -> None
        dut_lib = self.specs['dut_lib']
        dut_cell = self.specs['dut_cell']
        impl_lib = self.specs['impl_lib']
        wrapper_list = self.specs.get('dut_wrappers', [])

        inst_list, name_list = [], []
        for sch_params, cur_name in zip(sch_params_list, cell_name_list):
            dsn = self.prj.create_design_module(dut_lib, dut_cell)
            dsn.design(**sch_params)
            inst_list.append(dsn)
            name_list.append(cur_name)
            if gen_wrappers:
                for wrapper_config in wrapper_list:
                    wrapper_name = wrapper_config['name']
                    wrapper_lib = wrapper_config['lib']
                    wrapper_cell = wrapper_config['cell']
                    wrapper_params = wrapper_config['params'].copy()
                    wrapper_params['dut_lib'] = impl_lib
                    wrapper_params['dut_cell'] = cur_name
                    dsn = self.prj.create_design_module(wrapper_lib, wrapper_cell)
                    dsn.design(**wrapper_params)
                    inst_list.append(dsn)
                    inst_list.append(self.get_wrapper_name(cur_name, wrapper_name))

        self.prj.batch_schematic(impl_lib, inst_list, name_list=name_list)

    def create_dut_layouts(self, lay_params_list, cell_name_list, temp_db):
        # type: (Sequence[Dict[str, Any]], Sequence[str], TemplateDB) -> Sequence[Dict[str, Any]]
        """Create multiple layouts"""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        cls_package = self.specs['layout_package']
        cls_name = self.specs['layout_class']

        lay_module = importlib.import_module(cls_package)
        temp_cls = getattr(lay_module, cls_name)

        temp_list, sch_params_list = [], []
        for lay_params in lay_params_list:
            template = temp_db.new_template(params=lay_params, temp_cls=temp_cls, debug=False)
            temp_list.append(template)
            sch_params_list.append(template.sch_params)
        temp_db.batch_layout(self.prj, temp_list, cell_name_list)
        return sch_params_list

    def get_design_name(self, dsn_params):
        # type: (Dict[str, Any]) -> str
        """Returns the name of the design with the given parameters."""
        dsn_basename = self.specs['dsn_basename']
        try:
            combo_list = [dsn_params[key] for key in self.swp_var_list]
            return self.get_instance_name(dsn_basename, combo_list)
        except KeyError:
            for key in self.swp_var_list:
                if key not in dsn_params:
                    raise ValueError('Unspecified design parameter: %s' % key)
            raise ValueError('something is wrong...')

    def get_instance_name(self, name_base, combo_list):
        # type: (str, Sequence[Any, ...]) -> str
        """Generate cell names based on sweep parameter values."""
        suffix = ''
        for var, val in zip(self.swp_var_list, combo_list):
            if isinstance(val, str):
                suffix += '_%s_%s' % (var, val)
            elif isinstance(val, int):
                suffix += '_%s_%d' % (var, val)
            elif isinstance(val, float):
                suffix += '_%s_%s' % (var, float_to_si_string(val))
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name_base + suffix

    def get_data_file_name(self, dsn_name, tb_type):
        # type: (str, str) -> str
        """Returns the simulation data file name.

        Parameters
        ----------
        dsn_name : str
            design cell name.
        tb_type : str
            testbench type.

        Returns
        -------
        data_fname : str
            simulation data file name.
        """
        return os.path.join(self.specs['root_dir'], dsn_name, '%s.hdf5' % tb_type)

    def test_layout(self, gen_sch=True):
        # type: (bool) -> None
        """Create a test schematic and layout for debugging purposes"""

        sweep_params = self.specs['sweep_params']
        dsn_name = self.specs['dsn_basename'] + '_TEST'

        val_list = tuple((sweep_params[key][0] for key in self.swp_var_list))
        lay_params = self.get_layout_params(val_list)

        temp_db = self.make_tdb()
        print('create test layout')
        sch_params_list = self.create_dut_layouts([lay_params], [dsn_name], temp_db)

        if gen_sch:
            print('create test schematic')
            self.create_dut_schematics(sch_params_list, [dsn_name], gen_wrappers=False)
        print('done')

    def create_designs(self):
        # type: () -> None
        """Create DUT schematics/layouts, run LVS/RCX if necessary.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        impl_lib = self.specs['impl_lib']
        dsn_basename = self.specs['dsn_basename']
        view_name = self.specs['view_name']
        rcx_params = self.specs.get('rcx_params', {})

        extract = (view_name != 'schematic')
        temp_db = self.make_tdb()

        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_basename, combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)

        print('creating all layouts')
        sch_params_list = self.create_dut_layouts(lay_params_list, dsn_name_list, temp_db)
        print('creating all schematics')
        self.create_dut_schematics(sch_params_list, dsn_name_list, gen_wrappers=True)

        if extract:
            job_info_list = []
            for dsn_name, sch_params, combo_list in zip(dsn_name_list, sch_params_list, combo_list_list):
                print('start lvs job for %s' % dsn_name)
                lvs_id, lvs_log = self.prj.run_lvs(impl_lib, dsn_name, block=False)
                job_info_list.append([lvs_id, lvs_log])

            num_dsns = len(dsn_name_list)
            # start RCX jobs
            for idx in range(num_dsns):
                lvs_id, lvs_log = job_info_list[idx]
                dsn_name = dsn_name_list[idx]
                print('wait for %s LVS to finish' % dsn_name)
                lvs_passed = self.prj.wait_lvs_rcx(lvs_id)
                if not lvs_passed:
                    print('ERROR: LVS died for %s, cancelling rest of the jobs...' % dsn_name)
                    print('LVS log file: %s' % lvs_log)
                    for cancel_idx in range(len(job_info_list)):
                        self.prj.cancel(job_info_list[cancel_idx][0])
                    raise Exception('oops, LVS died for %s.' % dsn_name)
                print('%s LVS passed.  start RCX' % dsn_name)
                rcx_id, rcx_log = self.prj.run_rcx(impl_lib, dsn_name, block=False, rcx_params=rcx_params)
                job_info_list[idx][0] = rcx_id
                job_info_list[idx][1] = rcx_log

            # finish RCX jobs.
            for idx in range(num_dsns):
                dsn_name = dsn_name_list[idx]
                rcx_id, rcx_log = job_info_list[idx]
                print('wait for %s RCX to finish' % dsn_name)
                rcx_passed = self.prj.wait_lvs_rcx(rcx_id)
                if not rcx_passed:
                    print('ERROR: RCX died for %s, cancelling rest of the jobs...' % dsn_name)
                    print('RCX log file: %s' % rcx_log)
                    for cancel_idx in range(len(job_info_list)):
                        self.prj.cancel(job_info_list[cancel_idx][0])
                    raise Exception('oops, RCX died for %s.' % dsn_name)
                print('%s RCX passed.' % dsn_name)

        print('design generation done.')

    def create_tb_manager(self, dsn_name, tb_type):
        # type: (str, str) -> TestbenchManager
        """Create the TestbenchManager object for the given design and testbench type.

        Parameters
        ----------
        dsn_name : str
            the design cell name.
        tb_type : str
            the testbench type.

        Returns
        -------
        tb_manager : TestbenchManager
            the TestbenchManager instance.
        """
        impl_lib = self.specs['impl_lib']
        tb_specs = self.specs['testbench'][tb_type]

        cls_package = tb_specs['package']
        cls_name = tb_specs['class']
        tb_module = importlib.import_module(cls_package)
        tb_cls = getattr(tb_module, cls_name)

        data_fname = self.get_data_file_name(dsn_name, tb_type)
        tb_name = self.get_testbench_name(dsn_name, tb_type)
        tb_manager = tb_cls(self.prj, data_fname, impl_lib, dsn_name, tb_name, tb_specs)

        return tb_manager

    def run_simulations(self, tb_list, **kwargs):
        # type: (Sequence[str], **kwargs) -> Dict[str, Dict[str, Dict[str, Any]]]
        """Run given simulations on all designs.

        Parameters
        ----------
        tb_list : Sequence[str]
            list of simulations to run, in that order.
        **kwargs
            Optional arguments for wait() method of Testbench.

        Returns
        -------
        results : Dict[str, Dict[str, Dict[str, Any]]]
            A nested dictionary from design name and testbench type to
            performance summary dictionaries.  None results mean a
            simulation error occurred.
        """
        dsn_basename = self.specs['dsn_basename']
        dsn_name_list = [self.get_instance_name(dsn_basename, combo_list)
                         for combo_list in self.get_combinations_iter()]

        results = {dsn_name: {} for dsn_name in dsn_name_list}

        tb_type = tb_list[0]
        tb_manager_list = []
        for dsn_name in dsn_name_list:
            tb_manager = self.create_tb_manager(dsn_name, tb_type)
            tb_manager.run_simulation(tb_sch_fun=self.modify_tb_schematic,
                                      tb_setup_fun=self.modify_tb_setup,
                                      tb_type=tb_type)
            tb_manager_list.append(tb_manager)

        err_msg = 'Simulation Error for design %s, testbench %s.  Abort further simulations for this design'

        for next_tb_type in itertools.islice(tb_list, 1, None):
            next_tb_manager_list = []
            for dsn_name, tb_manager in zip(dsn_name_list, tb_manager_list):
                if tb_manager is None:
                    next_tb_manager = None
                    summary = None
                else:
                    sim_results = tb_manager.wait(**kwargs)
                    if sim_results is None:
                        print(err_msg % (dsn_name, tb_type))
                        next_tb_manager = None
                        summary = None
                    else:
                        self.post_simulation_procedure(tb_type, tb_manager, sim_results)
                        summary = tb_manager.get_performance_summary(sim_results)

                        next_tb_manager = self.create_tb_manager(dsn_name, next_tb_type)
                        next_tb_manager.run_simulation(tb_sch_fun=self.modify_tb_schematic,
                                                       tb_setup_fun=self.modify_tb_setup,
                                                       tb_type=next_tb_type)

                results[dsn_name][tb_type] = summary
                next_tb_manager_list.append(next_tb_manager)

            tb_manager_list = next_tb_manager_list
            tb_type = next_tb_type

        for dsn_name, tb_manager in zip(dsn_name_list, tb_manager_list):
            if tb_manager is None:
                summary = None
            else:
                sim_results = tb_manager.wait(**kwargs)
                if sim_results is None:
                    print(err_msg % (dsn_name, tb_type))
                    summary = None
                else:
                    self.post_simulation_procedure(tb_type, tb_manager, sim_results)
                    summary = tb_manager.get_performance_summary(sim_results)

            results[dsn_name][tb_type] = summary

        return results
