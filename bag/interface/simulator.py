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

"""This module handles high level simulation routines.

This module defines SimAccess, which provides methods to run simulations
and retrieve results.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import abc
from future.utils import with_metaclass

from ..io import make_temp_dir
from ..io.process import ProcessManager


class SimAccess(with_metaclass(abc.ABCMeta, object)):
    """A class that interacts with a simulator.

    Parameters
    ----------
    tmp_dir : string
        temporary file directory for SimAccess.
    sim_config : dict[string, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        self.sim_config = sim_config
        self.tmp_dir = make_temp_dir('simTmp', parent_dir=tmp_dir)

    @abc.abstractmethod
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
            and process return code when the simulation finished.

        Returns
        -------
        value : str or None
            Either the save directory path or the simulation ID.  If simulation
            is cancelled, return None.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def close(self, timeout=10.0):
        """Terminate the simulator gracefully.

        Parameters
        ----------
        timeout : float
            time to wait in seconds for each simulation process to terminate.
        """
        pass

    @abc.abstractmethod
    def cancel(self, sim_id, timeout=10.0):
        """Cancel the given simulation.

        If the process haven't started, this method prevents it from started.
        Otherwise, we first send a SIGTERM signal to kill the process.  If
        after ``timeout`` seconds the process is still alive, we will send a
        SIGKILL signal.  If after another ``timeout`` seconds the process is
        still alive, an Exception will be raised.

        Parameters
        ----------
        sim_id : string
            the simulation ID to cancel.
        timeout : float
            number of seconds to wait for cancellation.
        """
        pass

    @abc.abstractmethod
    def done(self, sim_id):
        """Returns True if the given simulation finished or is cancelled successfully.

        Parameters
        ----------
        sim_id : string
            ID of the simulation to query.

        Returns
        -------
        done : bool
            True if the simulation completed, or if simulation is cancelled.
        """
        return False

    @abc.abstractmethod
    def wait(self, sim_id, timeout=None, cancel_timeout=10.0):
        """Wait for the given simulation to finish, then return its results directory.

        If ``timeout`` is None, waits indefinitely.  Otherwise, if after
        ``timeout`` seconds the simulation is still running, a
        :class:`concurrent.futures.TimeoutError` will be raised.
        However, it is safe to catch this error and call wait again.

        If Ctrl-C is pressed before simulation finish or before timeout
        is reached, the simulation will be cancelled.

        Parameters
        ----------
        sim_id : string
            the simulation ID.
        timeout : float or None
            number of seconds to wait.  If None, waits indefinitely.
        cancel_timeout : float
            number of seconds to wait for simulation cancellation.

        Returns
        -------
        save_dir : str
            the save directory.
        retcode : int or None
            return code of the simulation, None if it is cancelled.
        """
        return None

    @abc.abstractmethod
    def format_parameter_value(self, param_config, precision):
        """Format the given parameter value as a string.

        To support both single value parameter and parameter sweeps, each parameter value is represented
        as a string instead of simple floats.  This method will cast a parameter configuration (which can
        either be a single value or a sweep) to a simulator-specific string.

        Parameters
        ----------
        param_config: dict[str, Any]
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
        return ""


# noinspection PyAbstractClass
class SimProcessManager(with_metaclass(abc.ABCMeta, SimAccess)):
    """An implementation of :class:`SimAccess` using :class:`bag.io.process.ProcessManager`.

    Parameters
    ----------
    tmp_dir : string
        temporary file directory for SimAccess.
    sim_config : dict[string, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        SimAccess.__init__(self, tmp_dir, sim_config)
        cancel_timeout = sim_config.get('cancel_timeout_ms', None)
        if cancel_timeout is not None:
            cancel_timeout /= 1e3
        self._manager = ProcessManager(max_workers=sim_config.get('max_workers', None),
                                       cancel_timeout=cancel_timeout)
        self._save_dirs = {}
        self._cancel_timeout = sim_config.get('cancel_timeout_ms', 1e4) / 1e3
        self._vproc = None
        self._run_viewer = sim_config.get('show_log_viewer', False)

    def close(self, timeout=10.0):
        """Terminate the simulator gracefully.

        Parameters
        ----------
        timeout : float
            time to wait in seconds for each simulation process to terminate.
        """
        if self._vproc is not None:
            from ..io import gui
            gui.close(self._vproc)
            self._vproc = None
        self._manager.close(timeout=timeout)

    def cancel(self, sim_id, timeout=None):
        """Cancel the given simulation.

        If the process haven't started, this method prevents it from started.
        Otherwise, we first send a SIGTERM signal to kill the process.  If
        after ``timeout`` seconds the process is still alive, we will send a
        SIGKILL signal.  If after another ``timeout`` seconds the process is
        still alive, an Exception will be raised.

        Parameters
        ----------
        sim_id : string
            the simulation ID to cancel.
        timeout : float or None
            number of seconds to wait for cancellation. If None, use default
            value.
        """
        self._manager.cancel(sim_id, timeout=timeout)

    def done(self, sim_id):
        """Returns True if the given simulation finished or is cancelled successfully.

        Parameters
        ----------
        sim_id : string
            the simulation ID.

        Returns
        -------
        done : bool
            True if the simulation is cancelled or completed.
        """
        return self._manager.done(sim_id)

    def wait(self, sim_id, timeout=None, cancel_timeout=None):
        """Wait for the given simulation to finish, then return its results directory.

        If ``timeout`` is None, waits indefinitely.  Otherwise, if after
        ``timeout`` seconds the simulation is still running, a
        :class:`concurrent.futures.TimeoutError` will be raised.
        However, it is safe to catch this error and call wait again.

        If Ctrl-C is pressed before simulation finish or before timeout
        is reached, the simulation will be cancelled.

        Parameters
        ----------
        sim_id : string
            the simulation ID.
        timeout : float or None
            number of seconds to wait.  If None, waits indefinitely.
        cancel_timeout : float or None
            number of seconds to wait for simulation cancellation.
            If None, use default settings.
        Returns
        -------
        save_dir : str
            the save directory.
        retcode : int or None
            return code of the simulation, None if it is cancelled.
        """
        retcode = self._manager.wait(sim_id, timeout=timeout, cancel_timeout=cancel_timeout)
        return self._save_dirs[sim_id], retcode

    def add_log(self, tag, fname):
        """Adds the given log file to log viewer.

        Parameters
        ----------
        tag : string
            a description of this log file.
        fname : string
            the log file name.
        """
        if self._run_viewer:
            from ..io import gui

            if self._vproc is None:
                self._vproc = gui.start_viewer()

            if not gui.add_log(self._vproc, tag, fname):
                self._vproc = None

    def remove_log(self, tag):
        """Remove the given log file from log viewer.

        Parameters
        ----------
        tag : string
            a description of this log file.
        """
        from ..io import gui

        if not gui.remove_log(self._vproc, tag):
            self._vproc = None

    def new_sim_process(self, args, save_dir, basename=None, logfile=None, append=False,
                        env=None, cwd=None, callback=None):
        """Put a new simulation process in queue.

        This is just a wrapper around :class:`~bag.io.process.ProcessManager`'s
        corresponding method.

        Parameters
        ----------
        args : string or list[string]
            the command to run as a string or list of string arguments.  See
            Python subprocess documentation.  list of string format is preferred.
        save_dir : string
            the simulation results directory.
        basename : string or None
            If given, this will be used as the basis for generating the unique
            process ID.
        logfile : string or None
            If given, stdout and stderr will be written to this file.  Otherwise,
            they will be redirected to `os.devnull`.
        append : bool
            True to append to ``logfile`` instead of overwritng it.
        env : dict[string, string] or None
            If given, environment variables of the process will be set according
            to this dictionary.
        cwd : string or None
            current working directory of the process.
        callback : callable
            If given, this function will automatically be executed when the
            process finished.  This function should take a single argument,
            which is a Future object that returns the return code of the
            process.

        Returns
        -------
        proc_id : string
            a unique string representing this process.  Can be used later
            to query process status or cancel process.
        """
        sim_id = self._manager.new_process(args, basename=basename, logfile=logfile,
                                           append=append, env=env, cwd=cwd, callback=callback)
        self._save_dirs[sim_id] = save_dir
        return sim_id
