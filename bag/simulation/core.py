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
from bag.io import read_yaml, open_file, load_sim_results, save_sim_results, load_sim_file
from bag.layout import RoutingGrid, TemplateDB

if TYPE_CHECKING:
    from bag.core import BagProject, Testbench


class SimulationManager(with_metaclass(abc.ABCMeta, object)):
    """A class that provide simple methods for running simulations.

    This class allows you to run simulations with schematic parameter sweeps.

    For now, this class will overwrite existing data, so please backup if you need to.

    Parameters
    ----------
    prj : Optional[BagProject]
        The BagProject instance.
    spec_file : str
        the specification file name or the data directory.  The specification file
        contains the following entries:

        impl_lib :
            the implementation library name.
        dut_lib :
            the DUT schematic library name.
        dut_cell :
            the DUT schematic cell name.
        layout_package :
            the XBase layout package name.
        layout_class :
            the XBase layout class name.
        sweep_params :
            a dictionary of schematic parameters to sweep and their values.
        dsn_name_base :
            base cell name for generated DUT instances.
        sim_envs :
            list of simulation environment names.
        routing_grid :
            the Layout RoutingGrid specification.
        rcx_params :
            RCX parameters dictionary.  Optional.
        root_dir :
            directory to save all simulation results in.

        <tb_type> :
            parameters for testbench <tb_type>.  Contains the following entries:

            tb_lib :
                testbench library name.
            tb_cell :
                testbench cell name.
            tb_name_base :
                base cell name for generated testbench schematics.
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
    def load_simulation_state(cls, prj, root_dir):
        # type: (BagProject, str) -> SimulationManager
        """Create the SimulationManager instance corresponding to data in the given directory."""
        return cls(prj, os.path.join(root_dir, 'specs.yaml'))

    @property
    def specs(self):
        # type: () -> Dict[str, Any]
        """Return the specification dictionary."""
        return self._specs

    @property
    def swp_var_list(self):
        # type: () -> Tuple[str, ...]
        return self._swp_var_list

    @abc.abstractmethod
    def configure_tb(self, tb_type, tb, val_list):
        # type: (str, Testbench, Tuple[Any, ...]) -> None
        """Setup the testbench with the given sweep parameter values."""
        pass

    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['layout_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    # noinspection PyUnusedLocal
    def get_wrapper_params(self, impl_lib, dsn_cell_name, sch_params):
        # type: (str, str, Dict[str, Any]) -> Dict[str, Any]
        """Returns the schematic wrapper parameters dictionary given library/cell/Module instance."""
        if 'wrapper_params' in self.specs:
            wrapper_params = self.specs['wrapper_params'].copy()
        else:
            wrapper_params = {}
        wrapper_params['dut_lib'] = impl_lib
        wrapper_params['dut_cell'] = dsn_cell_name
        return wrapper_params

    def get_tb_sch_params(self, tb_type, impl_lib, dsn_cell_name, val_list):
        # type: (str, str, str, Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the testbench schematic parameters dictionary from the given sweep parameter values."""
        tb_specs = self.specs[tb_type]
        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}
        tb_params['dut_lib'] = impl_lib
        tb_params['dut_cell'] = dsn_cell_name
        return tb_params

    def get_swp_var_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid sweep variable values."""
        return self.specs['sweep_params'][var]

    def get_combinations_iter(self):
        # type: () -> Iterable[Tuple[Any, ...]]
        """Returns an iterator of schematic parameter combinations we sweep over."""

        swp_par_dict = self.specs['sweep_params']
        return itertools.product(*(swp_par_dict[var] for var in self.swp_var_list))

    def make_tdb(self):
        # type: () -> TemplateDB
        """Create and return a new TemplateDB object."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        target_lib = self.specs['impl_lib']
        grid_specs = self.specs['routing_grid']
        layers = grid_specs['layers']
        spaces = grid_specs['spaces']
        widths = grid_specs['widths']
        bot_dir = grid_specs['bot_dir']

        routing_grid = RoutingGrid(self.prj.tech_info, layers, spaces, widths, bot_dir)
        tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
        return tdb

    def create_dut_sch(self, sch_params, dsn_cell_name, wrapper_name=''):
        # type: (Dict[str, Any], str, str) -> None
        """Create a new DUT schematic."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        dut_lib = self.specs['dut_lib']
        dut_cell = self.specs['dut_cell']
        impl_lib = self.specs['impl_lib']
        wrapper_cell = self.specs.get('wrapper_cell', '')

        dsn = self.prj.create_design_module(dut_lib, dut_cell)
        dsn.design(**sch_params)
        dsn.implement_design(impl_lib, top_cell_name=dsn_cell_name, erase=True)

        # create wrapper schematic if it exists
        if wrapper_name and wrapper_cell:
            wrapper_lib = self.specs['wrapper_lib']
            wrapper_params = self.get_wrapper_params(impl_lib, dsn_cell_name, sch_params)
            wrapper_dsn = self.prj.create_design_module(wrapper_lib, wrapper_cell)
            wrapper_dsn.design(**wrapper_params)
            wrapper_dsn.implement_design(impl_lib, top_cell_name=wrapper_name, erase=True)

    def create_tb_sch(self, tb_type, dsn_cell_name, tb_name, val_list):
        # type: (str, str, str, Tuple[Any, ...]) -> None
        """Create a new testbench schematic of the given type with the given DUT and testbench cell name."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        impl_lib = self.specs['impl_lib']
        tb_specs = self.specs[tb_type]
        tb_lib = tb_specs['tb_lib']
        tb_cell = tb_specs['tb_cell']

        tb_sch_params = self.get_tb_sch_params(tb_type, impl_lib, dsn_cell_name, val_list)
        tb_sch = self.prj.create_design_module(tb_lib, tb_cell)
        tb_sch.design(**tb_sch_params)
        tb_sch.implement_design(impl_lib, top_cell_name=tb_name, erase=True)

    def create_layout(self, lay_params_list, cell_name_list, temp_db):
        # type: (List[Dict[str, Any]], List[str], TemplateDB) -> List[Dict[str, Any]]
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
        sch_params = self.create_layout([lay_params], [dsn_name], temp_db)[0]

        if gen_sch:
            self.create_dut_sch(sch_params, dsn_name)

    def create_designs(self, tb_type=''):
        # type: (str, bool) -> None
        """Create DUT schematics, and run LVS/RCX, then simulate testbench if a testbench type is given."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        impl_lib = self.specs['impl_lib']
        dsn_name_base = self.specs['dsn_name_base']
        view_name = self.specs['view_name']
        wrapper_name_base = dsn_name_base + '_WRAPPER'
        rcx_params = self.specs.get('rcx_params', {})

        extract = (view_name != 'schematic')
        temp_db = self.make_tdb()

        # make layouts
        dsn_name_list, wrap_name_list, lay_params_list, combo_list_list = [], [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, combo_list)
            wrapper_name = self.get_instance_name(wrapper_name_base, combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            wrap_name_list.append(wrapper_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)

        print('creating all layouts')
        sch_params_list = self.create_layout(lay_params_list, dsn_name_list, temp_db)

        dsn_info_list = []
        job_info_list = []
        for dsn_name, wrapper_name, sch_params, combo_list in \
                zip(dsn_name_list, wrap_name_list, sch_params_list, combo_list_list):
            print('create schematic for %s' % dsn_name)
            self.create_dut_sch(sch_params, dsn_name, wrapper_name=wrapper_name)

            dsn_info_list.append((dsn_name, combo_list))
            if extract:
                print('start lvs job')
                lvs_id, lvs_log = self.prj.run_lvs(impl_lib, dsn_name, block=False)
                job_info_list.append([lvs_id, lvs_log])

        num_dsns = len(dsn_info_list)
        # start RCX jobs
        if extract:
            for idx in range(num_dsns):
                lvs_id, lvs_log = job_info_list[idx]
                dsn_name = dsn_info_list[idx][0]
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

        # finish RCX jobs.  Start testbench jobs if necessary
        sim_info_list = []
        for idx in range(num_dsns):
            dsn_name, val_list = dsn_info_list[idx]
            if extract:
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

            if tb_type:
                sim_info_list.append(self._run_tb_sim(tb_type, val_list))

        if sim_info_list:
            self.save_sim_data(tb_type, sim_info_list, dsn_info_list)
            print('characterization done.')

    def setup_testbench(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> Tuple[str, Testbench]
        """Create testbench of the given type and run simulation."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        impl_lib = self.specs['impl_lib']
        tb_specs = self.specs[tb_type]
        dsn_name_base = self.specs['dsn_name_base']
        wrapper_cell = self.specs.get('wrapper_cell', '')

        if wrapper_cell:
            wrapper_name_base = dsn_name_base + '_WRAPPER'
            dut_name = self.get_instance_name(wrapper_name_base, val_list)
        else:
            dut_name = self.get_instance_name(dsn_name_base, val_list)

        tb_name_base = tb_specs['tb_name_base']

        tb_name = self.get_instance_name(tb_name_base, val_list)
        print('create testbench %s' % tb_name)
        self.create_tb_sch(tb_type, dut_name, tb_name, val_list)

        tb = self.prj.configure_testbench(impl_lib, tb_name)

        self.configure_tb(tb_type, tb, val_list)
        tb.update_testbench()

        return tb_name, tb

    def run_simulations(self, tb_type, overwrite=True):
        # type: (str, bool) -> None
        """Create the given testbench type for all DUTs and run simulations in parallel."""
        dsn_name_base = self.specs['dsn_name_base']

        dsn_info_list = []
        sim_info_list = []
        for val_list in self.get_combinations_iter():
            save_data_path = self._get_data_path(tb_type, val_list)
            if overwrite or not os.path.isfile(save_data_path):
                dsn_name = self.get_instance_name(dsn_name_base, val_list)
                dsn_info_list.append((dsn_name, val_list))
                sim_info_list.append(self._run_tb_sim(tb_type, val_list))

        self.save_sim_data(tb_type, sim_info_list, dsn_info_list)
        print('simulation done.')

    def _run_tb_sim(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> Tuple[str, Testbench]
        """Create testbench of the given type and run simulation."""
        tb_name, tb = self.setup_testbench(tb_type, val_list)
        tb.run_simulation(sim_tag=tb_name, block=False)
        return tb_name, tb

    def save_sim_data(self, tb_type, sim_info_list, dsn_info_list):
        # type: (str, List[Tuple[str, Testbench]], List[Tuple[str, Tuple[Any, ...]]]) -> None
        """Save the simulation results to HDF5 files."""
        for (tb_name, tb), (_, val_list) in zip(sim_info_list, dsn_info_list):
            print('wait for %s simulation to finish' % tb_name)
            save_dir = tb.wait()
            print('%s simulation done.' % tb_name)
            if save_dir is not None:
                try:
                    cur_results = load_sim_results(save_dir)
                except Exception:
                    print('Error when loading results for %s' % tb_name)
                    cur_results = None
            else:
                cur_results = None

            if cur_results is not None:
                self.record_results(cur_results, tb_type, val_list)

    def _get_data_path(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> str
        """Returns the save file name."""
        root_dir = self.specs['root_dir']
        tb_specs = self.specs[tb_type]

        tb_name_base = tb_specs['tb_name_base']
        tb_name = self.get_instance_name(tb_name_base, val_list)
        results_dir = os.path.join(root_dir, tb_type)

        return os.path.join(results_dir, '%s.hdf5' % tb_name)

    def record_results(self, data, tb_type, val_list):
        # type: (Dict[str, Any], str, Tuple[Any, ...]) -> None
        """Record simulation results to file."""
        save_data_path = self._get_data_path(tb_type, val_list)
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        save_sim_results(data, save_data_path)

    def get_sim_results(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> Dict[str, Any]
        """Return simulation results corresponding to the given schematic parameters."""
        return load_sim_file(self._get_data_path(tb_type, val_list))
