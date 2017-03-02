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


"""This module implements bag's interaction with an ocean simulator.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import time
# noinspection PyCompatibility
from concurrent.futures import TimeoutError

from jinja2 import Template

import bag.io
from .simulator import SimProcessManager

run_script = bag.io.read_resource(bag.__name__, os.path.join('virtuoso_files', 'run_simulation.ocn'))
load_script = bag.io.read_resource(bag.__name__, os.path.join('virtuoso_files', 'load_results.ocn'))


class OceanInterface(SimProcessManager):
    """This class handles interaction with Ocean simulators.

    Parameters
    ----------
    tmp_dir : string
        temporary file directory for SimAccess.
    sim_config : dict[string, any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        """Initialize a new SkillInterface object.
        """
        SimProcessManager.__init__(self, tmp_dir, sim_config)

    def _launch_ocean(self, basename, save_dir, script_fname, log_fname, block, callback):
        """Private helper function that launches ocean process."""
        # get the simulation command.
        sim_kwargs = self.sim_config['kwargs']
        ocn_cmd = sim_kwargs['command']
        env = sim_kwargs.get('env', None)
        cwd = sim_kwargs.get('cwd', None)
        sim_cmd = [ocn_cmd, '-nograph', '-replay', script_fname, '-log', log_fname]

        # setup callback
        if callback is not None:
            def callback_wrapper(future):
                exc = future.exception()
                if exc is None:
                    # process exited normally
                    retcode = future.result()
                else:
                    retcode = None
                callback(save_dir, retcode)
        else:
            callback_wrapper = None

        # launch process
        sim_id = self.new_sim_process(sim_cmd, save_dir, basename=basename, env=env,
                                      cwd=cwd, callback=callback_wrapper)

        self.add_log(sim_id, log_fname)

        if not block:
            # return simulation ID.
            return sim_id

        # wait for simulation to finish.
        update_timeout_s = self.sim_config['update_timeout_ms'] / 1e3
        cancel_timeout_s = self.sim_config.get('cancel_timeout_ms', 1e4) / 1e3

        done = False
        tstart = time.time()
        print('Waiting for simulation %s to finish.  You may press Ctrl-C to abort.' % sim_id)
        while not done:
            try:
                self.wait(sim_id, timeout=update_timeout_s, cancel_timeout=cancel_timeout_s)
                done = True
            except TimeoutError:
                tcur = time.time()
                print('Simulation Running (%.1f s),  Press Ctrl-C to abort.' % (tcur - tstart))

        tcur = time.time()
        print('Simulation took %.1f s' % (tcur - tstart))
        return save_dir

    def run_simulation(self, tb_lib, tb_cell, outputs, precision=6, sim_tag=None,
                       block=True, callback=None):
        """Simulate the given testbench.

        If block is True, returns the save directory.  Otherwise, returns a simulation
        ID that can be used to query simulation status.

        In blocking mode, you can press Ctrl-C to cancel the simulation.

        Parameters
        ----------
        tb_lib : string
            testbench library name.
        tb_cell : string
            testbench cell name.
        outputs : dict[string, string]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.
        sim_tag : string or None
            a descriptive tag describing this simulation run.
        block : bool
            If True, wait for the simulation to finish.  Otherwise, return
            a simulation ID you can use to query simulation status later.
        callback : callable or None
            If given, this function will be called with the save directory
            and process return code when the simulation finished.  Process
            return code will be None if an exception is raised by the
            process.

        Returns
        -------
        value : str or None
            Either the save directory path or the simulation ID.  If simulation
            is cancelled, return None.
        """
        sim_tag = sim_tag or 'BagSim'
        job_options = self.sim_config['job_options']
        init_file = self.sim_config['init_file']
        view = self.sim_config['view']
        state = self.sim_config['state']

        # format job options as skill list of string
        job_opt_str = "'( "
        for key, val in job_options.items():
            job_opt_str += '"%s" "%s" ' % (key, val)
        job_opt_str += " )"

        # create temporary save directory and log/script names
        save_dir = bag.io.make_temp_dir(prefix='%s_data' % sim_tag, parent_dir=self.tmp_dir)
        log_fname = os.path.join(save_dir, 'ocn_output.log')
        script_fname = os.path.join(save_dir, 'run.ocn')

        # setup ocean simulation script
        script = Template(run_script).render(lib=tb_lib,
                                             cell=tb_cell,
                                             view=view,
                                             state=state,
                                             init_file=init_file,
                                             save_dir=save_dir,
                                             precision=precision,
                                             sim_tag=sim_tag,
                                             outputs=outputs,
                                             job_opt_str=job_opt_str,
                                             )
        bag.io.write_file(script_fname, script + '\n')

        # launch ocean
        return self._launch_ocean(sim_tag, save_dir, script_fname, log_fname, block, callback)

    def load_sim_results(self, lib, cell, hist_name, outputs, precision=6,
                         block=True, callback=None):
        """Load previous simulation results.

        If block is True, returns the save directory.  Otherwise, returns a simulation
        ID that can be used to query process status.

        In blocking mode, you can press Ctrl-C to cancel the loading process.

        Parameters
        ----------
        lib : str
            testbench library name.
        cell : str
            testbench cell name.
        hist_name : str
            simulation history name.
        outputs : dict[str, str]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.
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
        # send simulation request
        init_file = self.sim_config['init_file']
        view = self.sim_config['view']

        # create temporary save directory and log/script names
        save_dir = bag.io.make_temp_dir(prefix='%s_data' % hist_name, parent_dir=self.tmp_dir)
        log_fname = os.path.join(save_dir, 'ocn_output.log')
        script_fname = os.path.join(save_dir, 'run.ocn')

        # setup ocean load script
        script = Template(load_script).render(lib=lib,
                                              cell=cell,
                                              view=view,
                                              init_file=init_file,
                                              save_dir='{{ save_dir }}',
                                              precision=precision,
                                              hist_name=hist_name,
                                              outputs=outputs,
                                              )
        bag.io.write_file(script_fname, script + '\n')

        # launch ocean
        return self._launch_ocean(hist_name, save_dir, script_fname, log_fname, block, callback)

    def format_parameter_value(self, param_config, precision):
        """Format the given parameter value as a string.

        To support both single value parameter and parameter sweeps, each parameter value is represented
        as a string instead of simple floats.  This method will cast a parameter configuration (which can
        either be a single value or a sweep) to a simulator-specific string.

        Parameters
        ----------
        param_config: dict[string, any]
            a dictionary that describes this parameter value.

            4 formats are supported.  This is best explained by example.

            single value:
            dict(type='single', value=1.0)

            sweep a given list of values:
            dict(type='list', values=[1.0, 2.0, 3.0])

            linear sweep with inclusive start, inclusive stop, and step size:
            dict(type='linstep', start=1.0, stop=3.0, step=1.0)

            logarithmic sweep with given number of points per decade:
            dict(type='decade', start=1.0, stop=10.0, num=10)

        precision : int
            the parameter value precision.

        Returns
        -------
        param_str : str
            a string representation of param_config
        """

        fmt = '%.{}e'.format(precision)
        swp_type = param_config['type']
        if swp_type == 'single':
            return fmt % param_config['value']
        elif swp_type == 'list':
            return ' '.join((fmt % val for val in param_config['values']))
        elif swp_type == 'linstep':
            syntax = '{From/To}Linear:%s:%s:%s{From/To}' % (fmt, fmt, fmt)
            return syntax % (param_config['start'], param_config['step'], param_config['stop'])
        elif swp_type == 'decade':
            syntax = '{From/To}Decade:%s:%s:%s{From/To}' % (fmt, '%d', fmt)
            return syntax % (param_config['start'], param_config['num'], param_config['stop'])
        else:
            raise Exception('Unsupported param_config: %s' % param_config)
