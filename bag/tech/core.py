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

"""This module contains commonly used technology related classes and functions.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import os
import abc
import itertools
import importlib
from typing import List, Union, Tuple, Dict, Any, Optional, Set, Iterable, Sequence

import yaml
import numpy as np
import h5py
import openmdao.api as omdao

from bag import float_to_si_string
from bag.core import BagProject, Testbench
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB
from bag.data import load_sim_results, save_sim_results, load_sim_file
from ..math.interpolate import interpolate_grid
from bag.math.dfun import VectorDiffFunction, DiffFunction
from ..mdao.core import GroupBuilder
from ..io import fix_string, to_bytes, read_yaml, open_file


def _equal(a, b, rtol, atol):
    """Returns True if a == b.  a and b are both strings, floats or numpy arrays."""
    # python 2/3 compatibility: convert raw bytes to string
    a = fix_string(a)
    b = fix_string(b)

    if isinstance(a, str):
        return a == b
    return np.allclose(a, b, rtol=rtol, atol=atol)


def _equal_list(a, b, rtol, atol):
    """Returns True if a == b.  a and b are list of strings/floats/numpy arrays."""
    if len(a) != len(b):
        return False
    for a_item, b_item in zip(a, b):
        if not _equal(a_item, b_item, rtol, atol):
            return False
    return True


def _index_in_list(item_list, item, rtol, atol):
    """Returns index of item in item_list, with tolerance checking for floats."""
    for idx, test in enumerate(item_list):
        if _equal(test, item, rtol, atol):
            return idx
    return -1


def _in_list(item_list, item, rtol, atol):
    """Returns True if item is in item_list, with tolerance checking for floats."""
    return _index_in_list(item_list, item, rtol, atol) >= 0


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
        wrapper_params = self.specs['wrapper_params'].copy()
        wrapper_params['dut_lib'] = impl_lib
        wrapper_params['dut_cell'] = dsn_cell_name
        return wrapper_params

    def get_tb_sch_params(self, tb_type, impl_lib, dsn_cell_name, val_list):
        # type: (str, str, str, Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the testbench schematic parameters dictionary from the given sweep parameter values."""
        tb_params = self.specs[tb_type]['sch_params'].copy()
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
            wrapper_params = self.get_wrapper_params(impl_lib, dsn_cell_name, sch_params)
            wrapper_dsn = self.prj.create_design_module(dut_lib, wrapper_cell)
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

    def create_designs(self, tb_type='', extract=True):
        # type: (str, bool) -> None
        """Create DUT schematics, and run LVS/RCX, then simulate testbench if a testbench type is given."""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        impl_lib = self.specs['impl_lib']
        dsn_name_base = self.specs['dsn_name_base']
        wrapper_name_base = dsn_name_base + '_WRAPPER'
        rcx_params = self.specs.get('rcx_params', {})

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
                    for cancel_idx in range(len(job_info_list)):
                        self.prj.cancel(job_info_list[cancel_idx][0])
                    raise Exception('oops, LVS died for %s.  See LVS log file %s' % (dsn_name, lvs_log))
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
                    for cancel_idx in range(len(job_info_list)):
                        self.prj.cancel(job_info_list[cancel_idx][0])
                    raise Exception('oops, RCX died for %s.  See RCX log file %s' % (dsn_name, rcx_log))
                print('%s RCX passed.' % dsn_name)

            if tb_type:
                sim_info_list.append(self._run_tb_sim(tb_type, val_list))

        if sim_info_list:
            self.save_sim_data(tb_type, sim_info_list, dsn_info_list)
            print('characterization done.')

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

        print('start simulation for %s' % tb_name)
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


class CircuitCharacterization(with_metaclass(abc.ABCMeta, SimulationManager)):
    """A class that handles characterization of a circuit.

    This class sweeps schematic parameters and run a testbench with a single analysis.
    It will then save the simulation data in a format CharDB understands.

    For now, this class will overwrite existing data, so please backup if you need to.

    Parameters
    ----------
    prj : BagProject
        the BagProject instance.
    spec_file : str
        the SimulationManager specification file.
    tb_type : str
        the testbench type name.  The parameter dictionary corresponding to this
        testbench should have the following entries (in addition to those required
        by Simulation Manager:

        outputs :
            list of testbench output names to save.
        constants :
            constant values used to identify this simulation run.
        sweep_params:
            a dictionary from testbench parameters to (start, stop, num_points)
            sweep tuple.

    compression : str
        HDF5 compression method.
    """

    def __init__(self, prj, spec_file, tb_type, compression='gzip'):
        super(CircuitCharacterization, self).__init__(prj, spec_file)
        self._compression = compression
        self._outputs = self.specs[tb_type]['outputs']
        self._constants = self.specs[tb_type]['constants']
        self._sweep_params = self.specs[tb_type]['sweep_params']

    def record_results(self, data, tb_type, val_list):
        # type: (Dict[str, Any], str, Tuple[Any, ...]) -> None
        """Record simulation results to file.

        Override implementation in SimulationManager in order to save data
        in a format that CharDB understands.
        """
        env_list = self.specs['sim_envs']

        tb_specs = self.specs[tb_type]
        results_dir = tb_specs['results_dir']

        os.makedirs(results_dir, exist_ok=True)
        fname = os.path.join(results_dir, 'data.hdf5')

        with h5py.File(fname, 'w') as f:
            for key, val in self._constants.items():
                f.attrs[key] = val
            for key, val in self._sweep_params.items():
                f.attrs[key] = val

            for env in env_list:
                env_result, sweep_list = self._get_env_result(data, env)

                grp = f.create_group('%d' % len(f))
                for key, val in zip(self.swp_var_list, val_list):
                    grp.attrs[key] = val
                # h5py workaround: explicitly store strings as encoded unicode data
                grp.attrs['env'] = to_bytes(env)
                grp.attrs['sweep_params'] = [to_bytes(swp) for swp in sweep_list]

                for name, val in env_result.items():
                    grp.create_dataset(name, data=val, compression=self._compression)

    def get_sim_results(self, tb_type, val_list):
        # type: (str, Tuple[Any, ...]) -> Dict[str, Any]
        # TODO: implement this.
        raise NotImplementedError('not implemented yet.')

    def _get_env_result(self, sim_results, env):
        """Extract results from a given simulation environment from the given data.

        all output sweep parameter order and data shape must be the same.

        Parameters
        ----------
        sim_results : dict[string, any]
            the simulation results dictionary
        env : str
            the target simulation environment

        Returns
        -------
        results : dict[str, any]
            the results from a given simulation environment.
        sweep_list : list[str]
            a list of sweep parameter order.
        """
        if 'corner' not in sim_results:
            # no corner sweep anyways
            results = {output: sim_results[output] for output in self._outputs}
            sweep_list = sim_results['sweep_params'][self._outputs[0]]
            return results, sweep_list

        corner_list = sim_results['corner'].tolist()
        results = {}
        # we know all sweep order and shape is the same.
        test_name = self._outputs[0]
        sweep_list = list(sim_results['sweep_params'][test_name])
        shape = sim_results[test_name].shape
        # make numpy array slice index list
        index_list = [slice(0, l) for l in shape]
        if 'corner' in sweep_list:
            idx = sweep_list.index('corner')
            index_list[idx] = corner_list.index(env)
            del sweep_list[idx]

        # store outputs in results
        for output in self._outputs:
            results[output] = sim_results[output][index_list]

        return results, sweep_list


class CharDB(with_metaclass(abc.ABCMeta, object)):
    """The abstract base class of a database of characterization data.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    root_dir : str
        path to the root characterization data directory.  Supports environment variables.
    constants : Dict[str, Any]
        constants dictionary.
    discrete_params : List[str]
        a list of parameters that should take on discrete values.
    init_params : Dict[str, Any]
        a dictionary of initial parameter values.  All parameters should be specified,
        and None should be used if the parameter value is not set.
    env_list : List[str]
        list of simulation environments to consider.
    update : bool
        By default, CharDB saves and load post-processed data directly.  If update is True,
        CharDB will update the post-process data from raw simulation data. Defaults to
        False.
    rtol : float
        relative tolerance used to compare constants/sweep parameters/sweep attributes.
    atol : float
        relative tolerance used to compare constants/sweep parameters/sweep attributes.
    compression : str
        HDF5 compression method.  Used only during post-processing.
    method : str
        interpolation method.
    opt_package : str
        default Python optimization package.  Supports 'scipy' or 'pyoptsparse'.  Defaults
        to 'scipy'.
    opt_method : str
        default optimization method.  Valid values depends on the optimization package.
        Defaults to 'SLSQP'.
    opt_settings : Optional[Dict[str, Any]]
        optimizer specific settings.
    """

    def __init__(self,  # type: CharDB
                 root_dir,  # type: str
                 constants,  # type: Dict[str, Any]
                 discrete_params,  # type: List[str]
                 init_params,  # type: Dict[str, Any]
                 env_list,  # type: List[str]
                 update=False,  # type: bool
                 rtol=1e-5,  # type: float
                 atol=1e-18,  # type: float
                 compression='gzip',  # type: str
                 method='spline',  # type: str
                 opt_package='scipy',  # type: str
                 opt_method='SLSQP',  # type: str
                 opt_settings=None,  # type: Optional[Dict[str, Any]]
                 **kwargs  # type: **kwargs
                 ):
        # type: (...) -> None

        root_dir = os.path.abspath(os.path.expandvars(root_dir))

        if not os.path.isdir(root_dir):
            # error checking
            raise ValueError('Directory %s not found.' % root_dir)
        if 'env' in discrete_params:
            discrete_params.remove('env')

        if opt_settings is None:
            opt_settings = {}
        else:
            pass

        if opt_method == 'IPOPT' and not opt_settings:
            # set default IPOPT settings
            opt_settings['option_file_name'] = ''

        self._discrete_params = discrete_params
        self._params = init_params.copy()
        self._env_list = env_list
        self._config = dict(opt_package=opt_package,
                            opt_method=opt_method,
                            opt_settings=opt_settings,
                            rtol=rtol,
                            atol=atol,
                            method=method,
                            )

        cache_fname = self.get_cache_file(root_dir, constants)
        if not os.path.isfile(cache_fname) or update:
            sim_fname = self.get_sim_file(root_dir, constants)
            results = self._load_sim_data(sim_fname, constants, discrete_params)
            sim_data, total_params, total_values, self._constants = results
            self._data = self.post_process_data(sim_data, total_params, total_values, self._constants)

            # save to cache
            with h5py.File(cache_fname, 'w') as f:
                for key, val in self._constants.items():
                    f.attrs[key] = val
                sp_grp = f.create_group('sweep_params')
                # h5py workaround: explicitly store strings as encoded unicode data
                sp_grp.attrs['sweep_order'] = [to_bytes(swp) for swp in total_params]
                for par, val_list in zip(total_params, total_values):
                    if val_list.dtype.kind == 'U':
                        # unicode array, convert to raw bytes array
                        val_list = val_list.astype('S')
                    sp_grp.create_dataset(par, data=val_list, compression=compression)
                data_grp = f.create_group('data')
                for name, data_arr in self._data.items():
                    data_grp.create_dataset(name, data=data_arr, compression=compression)
        else:
            # load from cache
            with h5py.File(cache_fname, 'r') as f:
                self._constants = dict(iter(f.attrs.items()))
                sp_grp = f['sweep_params']
                total_params = [fix_string(swp) for swp in sp_grp.attrs['sweep_order']]
                total_values = [self._convert_hdf5_array(sp_grp[par][()]) for par in total_params]
                data_grp = f['data']
                self._data = {name: data_grp[name][()] for name in data_grp}

        # change axes location so discrete parameters are at the start of sweep_params
        env_disc_params = ['env'] + discrete_params
        for idx, dpar in enumerate(env_disc_params):
            if total_params[idx] != dpar:
                # swap
                didx = total_params.index(dpar)
                ptmp = total_params[idx]
                vtmp = total_values[idx]
                total_params[idx] = total_params[didx]
                total_values[idx] = total_values[didx]
                total_params[didx] = ptmp
                total_values[didx] = vtmp
                for key, val in self._data.items():
                    self._data[key] = np.swapaxes(val, idx, didx)

        sidx = len(self._discrete_params) + 1
        self._cont_params = total_params[sidx:]
        self._cont_values = total_values[sidx:]
        self._discrete_values = total_values[1:sidx]
        self._env_values = total_values[0]

        # get lazy function table.
        shape = [total_values[idx].size for idx in range(len(env_disc_params))]

        fun_name_iter = itertools.chain(iter(self._data.keys()), self.derived_parameters())
        # noinspection PyTypeChecker
        self._fun = {name: np.full(shape, None, dtype=object) for name in fun_name_iter}

    @staticmethod
    def _convert_hdf5_array(arr):
        # type: (np.ndarray) -> np.ndarray
        """Check if raw bytes array, if so convert to unicode array."""
        if arr.dtype.kind == 'S':
            return arr.astype('U')
        return arr

    def _load_sim_data(self,  # type: CharDB
                       fname,  # type: str
                       constants,  # type: Dict[str, Any]
                       discrete_params  # type: List[str]
                       ):
        # type: (...) -> Tuple[Dict[str, np.ndarray], List[str], List[np.ndarray], Dict[str, Any]]
        """Returns the simulation data.

        Parameters
        ----------
        fname : str
            the simulation filename.
        constants : Dict[str, Any]
            the constants dictionary.
        discrete_params : List[str]
            a list of parameters that should take on discrete values.

        Returns
        -------
        data_dict : Dict[str, np.ndarray]
            a dictionary from output name to data as numpy array.
        master_attrs : List[str]
            list of attribute name for each dimension of numpy array.
        master_values : List[np.ndarray]
            list of attribute values for each dimension.
        file_constants : Dict[str, Any]
            the constants dictionary in file.
        """
        if not os.path.exists(fname):
            raise ValueError('Simulation file %s not found.' % fname)

        rtol, atol = self.get_config('rtol'), self.get_config('atol')  # type: float

        master_attrs = None
        master_values = None
        master_dict = None
        file_constants = None
        with h5py.File(fname, 'r') as f:
            # check constants is consistent
            for key, val in constants.items():
                if not _equal(val, f.attrs[key], rtol, atol):
                    raise ValueError('sim file attr %s = %s != %s' % (key, f.attrs[key], val))

            # simple error checking.
            if len(f) == 0:
                raise ValueError('simulation file has no data.')

            # check that attributes sweep forms regular grid.
            attr_table = {}
            for gname in f:
                grp = f[gname]
                for key, val in grp.attrs.items():
                    # convert raw bytes to unicode
                    # python 2/3 compatibility: convert raw bytes to string
                    val = fix_string(val)

                    if key != 'sweep_params':
                        if key not in attr_table:
                            attr_table[key] = []
                        val_list = attr_table[key]
                        if not _in_list(val_list, val, rtol, atol):
                            val_list.append(val)

            expected_len = 1
            for val in attr_table.values():
                expected_len *= len(val)

            if expected_len != len(f):
                raise ValueError('Attributes of f does not form complete sweep. '
                                 'Expect length = %d, but actually = %d.' % (expected_len, len(f)))

            # check all discrete parameters in attribute table.
            for disc_par in discrete_params:
                if disc_par not in attr_table:
                    raise ValueError('Discrete attribute %s not found' % disc_par)

            # get attribute order
            attr_order = sorted(attr_table.keys())
            # check all non-discrete attribute value list lies on regular grid
            attr_values = [np.array(sorted(attr_table[attr])) for attr in attr_order]
            for attr, aval_list in zip(attr_order, attr_values):
                if attr not in discrete_params and attr != 'env':
                    test_vec = np.linspace(aval_list[0], aval_list[-1], len(aval_list), endpoint=True)
                    if not np.allclose(test_vec, aval_list, rtol=rtol, atol=atol):
                        raise ValueError('Attribute %s values do not lie on regular grid' % attr)

            # consolidate all data into one giant numpy array.
            # first compute numpy array shape
            test_grp = f['0']
            sweep_params = [fix_string(tmpvar) for tmpvar in test_grp.attrs['sweep_params']]

            # get constants dictionary
            file_constants = {}
            for key, val in f.attrs.items():
                if key not in sweep_params:
                    file_constants[key] = val

            master_attrs = attr_order + sweep_params
            swp_values = [np.linspace(f.attrs[var][0], f.attrs[var][1], f.attrs[var][2],
                                      endpoint=True) for var in sweep_params]  # type: List[np.array]
            master_values = attr_values + swp_values
            master_shape = [len(val_list) for val_list in master_values]
            master_index = [slice(0, n) for n in master_shape]
            master_dict = {}
            for gname in f:
                grp = f[gname]
                # get index of the current group in the giant array.
                # Note: using linear search to compute index now, but attr_val_list should be small.
                for aidx, (attr, aval_list) in enumerate(zip(attr_order, attr_values)):
                    master_index[aidx] = _index_in_list(aval_list, grp.attrs[attr], rtol, atol)

                for output in grp:
                    dset = grp[output]
                    if output not in master_dict:
                        master_dict[output] = np.empty(master_shape, dtype=dset.dtype)
                    master_dict[output][master_index] = dset

        return master_dict, master_attrs, master_values, file_constants

    def __getitem__(self, param):
        # type: (str) -> Any
        """Returns the given parameter value.

        Parameters
        ----------
        param : str
            parameter name.

        Returns
        -------
        val : Any
            parameter value.
        """
        return self._params[param]

    def __setitem__(self, key, value):
        # type: (str, Any) -> None
        """Sets the given parameter value.

        Parameters
        ----------
        key : str
            parameter name.
        value : Any
            parameter value.  None to unset.
        """
        rtol, atol = self.get_config('rtol'), self.get_config('atol')

        if key in self._discrete_params:
            if value is not None:
                idx = self._discrete_params.index(key)
                if not _in_list(self._discrete_values[idx], value, rtol, atol):
                    raise ValueError('Cannot set discrete variable %s value to %s' % (key, value))
        elif key in self._cont_params:
            if value is not None:
                idx = self._cont_params.index(key)
                val_list = self._cont_values[idx]
                if value < val_list[0] or value > val_list[-1]:
                    raise ValueError('Variable %s value %s out of bounds.' % (key, value))
        else:
            raise ValueError('Unknown variable %s.' % key)

        self._params[key] = value

    def get_config(self, name):
        # type: (str) -> Any
        """Returns the configuration value.

        Parameters
        ----------
        name : str
            configuration name.

        Returns
        -------
        val : Any
            configuration value.
        """
        return self._config[name]

    def set_config(self, name, value):
        # type: (str, Any) -> None
        """Sets the configuration value.

        Parameters
        ----------
        name : str
            configuration name.
        value : Any
            configuration value.
        """
        if name not in self._config:
            raise ValueError('Unknown configuration %s' % name)
        self._config[name] = value

    @property
    def env_list(self):
        # type: () -> List[str]
        """The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list):
        # type: (List[str]) -> None
        """Sets the list of simulation environments to consider."""
        self._env_list = new_env_list

    @classmethod
    def get_sim_file(cls, root_dir, constants):
        # type: (str, Dict[str, Any]) -> str
        """Returns the simulation data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : Dict[str, Any]
            constants dictionary.

        Returns
        -------
        fname : str
            the simulation data file name.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def get_cache_file(cls, root_dir, constants):
        # type: (str, Dict[str, Any]) -> str
        """Returns the post-processed characterization data file name.

        Parameters
        ----------
        root_dir : str
            absolute path to the root characterization data directory.
        constants : Dict[str, Any]
            constants dictionary.

        Returns
        -------
        fname : str
            the post-processed characterization data file name.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def post_process_data(cls, sim_data, sweep_params, sweep_values, constants):
        # type: (Dict[str, np.ndarray], List[str], List[np.ndarray], Dict[str, Any]) -> Dict[str, np.ndarray]
        """Postprocess simulation data.

        Parameters
        ----------
        sim_data : Dict[str, np.ndarray]
            the simulation data as a dictionary from output name to numpy array.
        sweep_params : List[str]
            list of parameter name for each dimension of numpy array.
        sweep_values : List[np.ndarray]
            list of parameter values for each dimension.
        constants : Dict[str, Any]
            the constants dictionary.

        Returns
        -------
        data : Dict[str, np.ndarray]
            a dictionary of post-processed data.
        """
        raise NotImplementedError('Not implemented')

    @classmethod
    def derived_parameters(cls):
        # type: () -> List[str]
        """Returns a list of derived parameters."""
        return []

    @classmethod
    def compute_derived_parameters(cls, fdict):
        # type: (Dict[str, DiffFunction]) -> Dict[str, DiffFunction]
        """Compute derived parameter functions.

        Parameters
        ----------
        fdict : Dict[str, DiffFunction]
            a dictionary from core parameter name to the corresponding function.

        Returns
        -------
        deriv_dict : Dict[str, DiffFunction]
            a dictionary from derived parameter name to the corresponding function.
        """
        return {}

    def _get_function_index(self, **kwargs):
        # type: (**kwargs) -> List[int]
        """Returns the function index corresponding to given discrete parameter values.

        simulation environment index will be set to 0

        Parameters
        ----------
        **kwargs :
            discrete parameter values.

        Returns
        -------
        fidx_list : List[int]
            the function index.
        """
        rtol, atol = self.get_config('rtol'), self.get_config('atol')

        fidx_list = [0]
        for par, val_list in zip(self._discrete_params, self._discrete_values):
            val = kwargs.get(par, self[par])
            if val is None:
                raise ValueError('Parameter %s value not specified' % par)

            val_idx = _index_in_list(val_list, val, rtol, atol)
            if val_idx < 0:
                raise ValueError('Discrete parameter %s have illegal value %s' % (par, val))
            fidx_list.append(val_idx)

        return fidx_list

    def _get_function_helper(self, name, fidx_list):
        # type: (str, Union[List[int], Tuple[int]]) -> DiffFunction
        """Helper method for get_function()

        Parameters
        ----------
        name : str
            name of the function.
        fidx_list : Union[List[int], Tuple[int]]
            function index.

        Returns
        -------
        fun : DiffFunction
            the interpolator function.
        """
        # get function table index
        fidx_list = tuple(fidx_list)
        ftable = self._fun[name]
        if ftable[fidx_list] is None:
            if name in self._data:
                # core parameter
                char_data = self._data[name]

                # get scale list and data index
                scale_list = []
                didx = list(fidx_list)  # type: List[Union[int, slice]]
                for vec in self._cont_values:
                    scale_list.append((vec[0], vec[1] - vec[0]))
                    didx.append(slice(0, vec.size))

                # make interpolator.
                cur_data = char_data[didx]
                method = self.get_config('method')
                ftable[fidx_list] = interpolate_grid(scale_list, cur_data, method=method, extrapolate=True)
            else:
                # derived parameter
                core_fdict = {fn: self._get_function_helper(fn, fidx_list) for fn in self._data}
                deriv_fdict = self.compute_derived_parameters(core_fdict)
                for fn, deriv_fun in deriv_fdict.items():
                    self._fun[fn][fidx_list] = deriv_fun

        return ftable[fidx_list]

    def get_function(self, name, env='', **kwargs):
        # type: (str, str, **kwargs) -> Union[VectorDiffFunction, DiffFunction]
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs :
            dictionary of discrete parameter values.

        Returns
        -------
        output : Union[VectorDiffFunction, DiffFunction]
            the output vector function.
        """
        fidx_list = self._get_function_index(**kwargs)
        if not env:
            fun_list = []
            for env in self.env_list:
                occur_list = np.where(self._env_values == env)[0]
                if occur_list.size == 0:
                    raise ValueError('environment %s not found.')
                env_idx = occur_list[0]
                fidx_list[0] = env_idx
                fun_list.append(self._get_function_helper(name, fidx_list))
            return VectorDiffFunction(fun_list)
        else:
            occur_list = np.where(self._env_values == env)[0]
            if occur_list.size == 0:
                raise ValueError('environment %s not found.')
            env_idx = occur_list[0]
            fidx_list[0] = env_idx
            return self._get_function_helper(name, fidx_list)

    def get_fun_sweep_params(self):
        # type: () -> Tuple[List[str], List[Tuple[float, float]]]
        """Returns interpolation function sweep parameter names and values.

        Returns
        -------
        sweep_params : List[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        return self._cont_params, [(vec[0], vec[-1]) for vec in self._cont_values]

    def _get_fun_arg(self, **kwargs):
        # type: (**kwargs) -> np.ndarray
        """Make numpy array of interpolation function arguments."""
        val_list = []
        for par in self._cont_params:
            val = kwargs.get(par, self[par])
            if val is None:
                raise ValueError('Parameter %s value not specified.' % par)
            val_list.append(val)

        return np.array(val_list)

    def query(self, **kwargs):
        # type: (**kwargs) -> Dict[str, np.ndarray]
        """Query the database for the values associated with the given parameters.

        All parameters must be specified.

        Parameters
        ----------
        **kwargs :
            parameter values.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        results = {}
        arg = self._get_fun_arg(**kwargs)
        for name in self._data:
            fun = self.get_function(name, **kwargs)
            results[name] = fun(arg)

        for var in itertools.chain(self._discrete_params, self._cont_params):
            results[var] = kwargs.get(var, self[var])

        results.update(self.compute_derived_parameters(results))

        return results

    def minimize(self,  # type: CharDB
                 objective,  # type: str
                 define=None,  # type: List[Tuple[str, int]]
                 cons=None,  # type: Dict[str, Dict[str, float]]
                 vector_params=None,  # type: Set[str]
                 debug=False,  # type: bool
                 **kwargs  # type: **kwargs
                 ):
        # type: (...) -> Dict[str, Union[np.ndarray, float]]
        """Find operating point that minimizes the given objective.

        Parameters
        ----------
        objective : str
            the objective to minimize.  Must be a scalar.
        define : List[Tuple[str, int]]
            list of expressions to define new variables.  Each
            element of the list is a tuple of string and integer.  The string
            contains a python assignment that computes the variable from
            existing ones, and the integer indicates the variable shape.

            Note that define can also be used to enforce relationships between
            existing variables.  Using transistor as an example, defining
            'vgs = vds' will force the vgs of vds of the transistor to be
            equal.
        cons : Dict[str, Dict[str, float]]
            a dictionary from variable name to constraints of that variable.
            see OpenMDAO documentations for details on constraints.
        vector_params : Set[str]
            set of input variables that are vector instead of scalar.  An input
            variable is a vector if it can change across simulation environments.
        debug : bool
            True to enable debugging messages.  Defaults to False.
        **kwargs :
            known parameter values.

        Returns
        -------
        results : Dict[str, Union[np.ndarray, float]]
            the results dictionary.
        """
        cons = cons or {}
        fidx_list = self._get_function_index(**kwargs)
        builder = GroupBuilder()

        params_ranges = dict(zip(self._cont_params,
                                 ((vec[0], vec[-1]) for vec in self._cont_values)))
        # add functions
        fun_name_iter = itertools.chain(iter(self._data.keys()), self.derived_parameters())
        for name in fun_name_iter:
            fun_list = []
            for idx, env in enumerate(self.env_list):
                fidx_list[0] = idx
                fun_list.append(self._get_function_helper(name, fidx_list))

            builder.add_fun(name, fun_list, self._cont_params, params_ranges,
                            vector_params=vector_params)

        # add expressions
        for expr, ndim in define:
            builder.add_expr(expr, ndim)

        # update input bounds from constraints
        input_set = builder.get_inputs()
        var_list = builder.get_variables()

        for name in input_set:
            if name in cons:
                setup = cons[name]
                if 'equals' in setup:
                    eq_val = setup['equals']
                    builder.set_input_limit(name, equals=eq_val)
                else:
                    vmin = vmax = None
                    if 'lower' in setup:
                        vmin = setup['lower']
                    if 'upper' in setup:
                        vmax = setup['upper']
                    builder.set_input_limit(name, lower=vmin, upper=vmax)

        # build the group and make the problem
        grp, input_bounds = builder.build()

        top = omdao.Problem()
        top.root = grp

        opt_package = self.get_config('opt_package')  # type: str
        opt_settings = self.get_config('opt_settings')

        if opt_package == 'scipy':
            driver = top.driver = omdao.ScipyOptimizer()
            print_opt_name = 'disp'
        elif opt_package == 'pyoptsparse':
            driver = top.driver = omdao.pyOptSparseDriver()
            print_opt_name = 'print_results'
        else:
            raise ValueError('Unknown optimization package: %s' % opt_package)

        driver.options['optimizer'] = self.get_config('opt_method')
        driver.options[print_opt_name] = debug
        driver.opt_settings.update(opt_settings)

        # add constraints
        constants = {}
        for name, setup in cons.items():
            if name not in input_bounds:
                # add constraint
                driver.add_constraint(name, **setup)

        # add inputs
        for name in input_set:
            eq_val, lower, upper, ndim = input_bounds[name]
            val = kwargs.get(name, self[name])  # type: float
            if val is None:
                val = eq_val
            comp_name = 'comp__%s' % name
            if val is not None:
                val = np.atleast_1d(np.ones(ndim) * val)
                constants[name] = val
                top.root.add(comp_name, omdao.IndepVarComp(name, val=val), promotes=[name])
            else:
                avg = (lower + upper) / 2.0
                span = upper - lower
                val = np.atleast_1d(np.ones(ndim) * avg)
                top.root.add(comp_name, omdao.IndepVarComp(name, val=val), promotes=[name])
                driver.add_desvar(name, lower=lower, upper=upper, adder=-avg, scaler=1.0 / span)
                # driver.add_desvar(name, lower=lower, upper=upper)

        # add objective and setup
        driver.add_objective(objective)
        top.setup(check=debug)

        # somehow html file is not viewable.
        if debug:
            omdao.view_model(top, outfile='CharDB_debug.html')

        # set constants
        for name, val in constants.items():
            top[name] = val

        top.run()

        results = {var: kwargs.get(var, self[var]) for var in self._discrete_params}
        for var in var_list:
            results[var] = top[var]

        return results
