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
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, List, Iterable, Sequence, Callable

import yaml

from bag import float_to_si_string
from bag.io import read_yaml, open_file, load_sim_results, save_sim_results, load_sim_file
from bag.layout import RoutingGrid, TemplateDB

if TYPE_CHECKING:
    from bag.core import BagProject, Testbench


class TestbenchManager(with_metaclass(abc.ABCMeta, object)):
    """A class that manages testbench generation, setup, simulation, and data post-processing.

    Parameters
    ----------
    prj : Optional[BagProject]
        The BagProject instance.
    data_fname : str
        Simulation data file name.
    impl_lib : str
        implementation library name.
    dsn_name : str
        DUT cell name.
    tb_name : str
        testbench cell name.
    specs : Dict[str, Any]
        the testbench specification dictionary.
    """
    def __init__(self, prj, data_fname, impl_lib, dsn_name, tb_name, specs):
        # type: (Optional[BagProject], str, str, str, str, Dict[str, Any]) -> None
        self.prj = prj
        self.data_fname = os.path.abspath(data_fname)
        self.impl_lib = impl_lib
        self.dsn_name = dsn_name
        self.tb_name = tb_name
        self.specs = specs
        self._tb = None  # type: Optional[Testbench]

    @abc.abstractmethod
    def configure_testbench(self, tb):
        # type: (Testbench) -> None
        """Configure the testbench simulation state.

        Parameters
        ----------
        tb : Testbench
            the testbench object.
        """
        pass

    def _create_tb_schematic(self, setup_fun=None):
        # type: (Optional[Callable[[Dict[str, Any]], None]]) -> None
        """Creates the testbench schematic.

        Parameters
        ----------
        setup_fun : Optional[Callable[[Dict[str, Any]], None]]
            an optional function that modifies the testbench schematic parameters.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        tb_lib = self.specs['tb_lib']
        tb_cell = self.specs['tb_cell']
        if 'sch_params' in self.specs:
            tb_params = self.specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self.impl_lib
        tb_params['dut_cell'] = self.dsn_name
        if setup_fun is not None:
            setup_fun(tb_params)

        tb_sch = self.prj.create_design_module(tb_lib, tb_cell)
        tb_sch.design(**tb_params)
        tb_sch.implement_design(self.impl_lib, top_cell_name=self.tb_name, erase=True)

    def _setup_testbench(self, setup_fun=None):
        # type: (Optional[Callable[[Testbench], None]]) -> Testbench
        """Setup simulation state of the testbench.

        Parameters
        ----------
        setup_fun : Optional[Callable[[Testbench], None]]
            an optional function that modifies testbench simulation setup.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        tb = self.prj.configure_testbench(self.impl_lib, self.tb_name)
        self.configure_testbench(tb)
        setup_fun(tb)
        tb.update_testbench()
        return tb

    def run_simulation(self,  # type: TestbenchManager
                       tb_sch_fun=None,  # type: Optional[Callable[[Dict[str, Any]], None]]
                       tb_setup_fun=None,  # type: Optional[Callable[[Testbench], None]]
                       ):
        # type: (...) -> None
        """Create testbench, setup simulation, then start simulation

        Parameters
        ----------
        tb_sch_fun : Optional[Callable[[Dict[str, Any]], None]]
            an optional function that modifies the testbench schematic parameters.
        tb_setup_fun : Optional[Callable[[Testbench], None]]
            an optional function that modifies testbench simulation setup.

        Returns
        -------
        tag_name : str
            the simulation tag name.
        tb : Testbench
            the simulation Testbench object.
        """
        if self._tb is not None:
            raise ValueError('A simulation may already be running.')

        self._create_tb_schematic(setup_fun=tb_sch_fun)
        self._tb = self._setup_testbench(setup_fun=tb_setup_fun)
        self._tb.run_simulation(sim_tag=self.tb_name, block=False)

    def _record_results(self, data):
        # type: (Dict[str, Any]) -> None
        """Record simulation results to file."""
        os.makedirs(os.path.dirname(self.data_fname), exist_ok=True)
        save_sim_results(data, self.data_fname)

    def wait(self, **kwargs):
        # type: (**kwargs) -> Dict[str, Any]
        """Wait for testbench simulation to finish, then return the results.

        Parameters
        ----------
        **kwargs
            Optional arguments for wait() method of Testbench.

        Returns
        -------
        results : Dict[str, Any]
            Simulation results dictionary.

        """
        if self._tb is None:
            raise ValueError('No simulation is running')

        save_dir = self._tb.wait(**kwargs)
        if save_dir is not None:
            # noinspection PyBroadException
            try:
                cur_results = load_sim_results(save_dir)
            except Exception:
                print('Error when loading results for %s' % self.tb_name)
                cur_results = None
        else:
            cur_results = None

        self._record_results(cur_results)
        return cur_results

    def load_sim_results(self):
        # type: () -> Dict[str, Any]
        """Load previously saved simulation results.

        Returns
        -------
        results : Dict[str, Any]
            the simulation results dictionary.
        """
        return load_sim_file(self.data_fname)


class SimulationManager(with_metaclass(abc.ABCMeta, object)):
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
        # type: (BagProject, str) -> SimulationManager
        """Create the SimulationManager instance corresponding to data in the given directory."""
        return cls(prj, root_dir)

    @classmethod
    def _get_wrapper_name(cls, dut_name, wrapper_name):
        # type: (str, str) -> str
        """Returns the wrapper cell name corresponding to the given DUT."""
        return '%s_%s' % (dut_name, wrapper_name)

    @classmethod
    def _get_testbench_name(cls, dut_name, tb_type):
        # type: (str, str) -> str
        """Returns the testbench cell name corresponding to the given DUT."""
        return '%s_%s' % (dut_name, tb_type)

    @property
    def specs(self):
        # type: () -> Dict[str, Any]
        """Return the specification dictionary."""
        return self._specs

    @property
    def swp_var_list(self):
        # type: () -> Tuple[str, ...]
        return self._swp_var_list

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
                    inst_list.append(self._get_wrapper_name(cur_name, wrapper_name))

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
        dsn_name_base = self.specs['dsn_name_base']
        try:
            combo_list = [dsn_params[key] for key in self.swp_var_list]
            return self.get_instance_name(dsn_name_base, combo_list)
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

    def test_layout(self, gen_sch=True):
        # type: (bool) -> None
        """Create a test schematic and layout for debugging purposes"""

        sweep_params = self.specs['sweep_params']
        dsn_name = self.specs['dsn_name_base'] + '_TEST'

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
        dsn_name_base = self.specs['dsn_name_base']
        view_name = self.specs['view_name']
        rcx_params = self.specs.get('rcx_params', {})

        extract = (view_name != 'schematic')
        temp_db = self.make_tdb()

        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, combo_list)
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
