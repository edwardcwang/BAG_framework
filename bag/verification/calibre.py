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

"""This module implements LVS/RCX using Calibre and stream out from Virtuoso.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
from typing import Optional, Union

from jinja2 import Template

from .base import Checker
from ..io import read_file, write_file, open_temp, readlines_iter
from ..io import process
from .virtuoso_export import export_lay_sch


def lvs_passed(cmd_output):
    """Returns True if LVS passed.

    Parameters
    ----------
    cmd_output : str
        output from LVS/PEX.

    Returns
    -------
    flag : bool
        True if LVS passed.
    """
    return 'LVS completed. CORRECT. See report file:' in cmd_output


class Calibre(Checker):
    """A subclass of Checker that uses calibre for verification.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    lvs_run_dir : str
        the LVS run directory.
    lvs_runset : str
        the LVS runset filename.
    rcx_run_dir : str
        the RCX run directory.
    rcx_runset : str
        the RCX runset filename.
    source_added_file : str
        the Calibre source.added file location.  Environment variable is supported.
        Default value is '$DK/Calibre/lvs/source.added'.
    rcx_mode : str
        the RC extraction mode.  Either 'pex' or 'xact'.  Defaults to 'pex'.
    xact_rules : str
        the XACT rules file name.
    """

    def __init__(self, tmp_dir, lvs_run_dir, lvs_runset, rcx_run_dir, rcx_runset,
                 source_added_file='$DK/Calibre/lvs/source.added', rcx_mode='pex',
                 xact_rules='', **kwargs):
        Checker.__init__(self, tmp_dir)

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        self.default_rcx_params = kwargs.get('rcx_params', {})
        self.default_lvs_params = kwargs.get('lvs_params', {})
        if cancel_timeout is not None:
            cancel_timeout /= 1e3
        self._manager = process.ProcessManager(max_workers=max_workers,
                                               cancel_timeout=cancel_timeout)

        self.lvs_run_dir = lvs_run_dir
        self.lvs_runset = lvs_runset
        self.rcx_run_dir = rcx_run_dir
        self.rcx_runset = rcx_runset
        self.xact_rules = xact_rules
        self.source_added_file = source_added_file
        self.rcx_mode = rcx_mode

    def modify_lvs_runset(self, lib_name, cell_name, lay_view, gds_file, netlist,
                          lvs_params):
        """Modify the given LVS runset file.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        lay_view : str
            the layout view.
        gds_file : str
            the layout gds file name.
        netlist : str
            the schematic netlist file.
        lvs_params : dict[str, any]
            override LVS parameters.

        Returns
        -------
        content : str
            the new runset content.
        """
        run_dir = os.path.abspath(self.lvs_run_dir)

        # convert runset content to dictionary
        lvs_options = {}
        for line in readlines_iter(self.lvs_runset):
            key, val = line.split(':', 1)
            key = key.strip('*')
            lvs_options[key] = val.strip()

        # override parameters
        lvs_options['lvsRunDir'] = run_dir
        lvs_options['lvsLayoutPaths'] = gds_file
        lvs_options['lvsLayoutPrimary'] = cell_name
        lvs_options['lvsLayoutLibrary'] = lib_name
        lvs_options['lvsLayoutView'] = lay_view
        lvs_options['lvsSourcePath'] = netlist
        lvs_options['lvsSourcePrimary'] = cell_name
        lvs_options['lvsSourceLibrary'] = lib_name
        lvs_options['lvsSpiceFile'] = os.path.join(run_dir, '%s.sp' % cell_name)
        lvs_options['lvsERCDatabase'] = '%s.erc.results' % cell_name
        lvs_options['lvsERCSummaryFile'] = '%s.erc.summary' % cell_name
        lvs_options['lvsReportFile'] = '%s.lvs.report' % cell_name
        lvs_options['lvsMaskDBFile'] = '%s.maskdb' % cell_name
        lvs_options['cmnFDILayoutLibrary'] = lib_name
        lvs_options['cmnFDILayoutView'] = lay_view
        lvs_options['cmnFDIDEFLayoutPath'] = '%s.def' % cell_name

        lvs_options.update(lvs_params)

        return ''.join(('*%s: %s\n' % (key, val) for key, val in lvs_options.items()))

    def modify_pex_runset(self, lib_name, cell_name, lay_view, gds_file, netlist,
                          rcx_params):
        """Modify the given RCX runset file.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        lay_view : str
            the layout view.
        gds_file : str
            the layout gds file name.
        netlist : str
            the schematic netlist file.
        rcx_params : dict[str, any]
            override RCX parameters.

        Returns
        -------
        content : str
            the new runset content.
        output_name : str
            the extracted netlist file.
        """
        run_dir = os.path.abspath(self.rcx_run_dir)

        # convert runset content to dictionary
        rcx_options = {}
        for line in readlines_iter(self.rcx_runset):
            key, val = line.split(':', 1)
            key = key.strip('*')
            rcx_options[key] = val.strip()

        output_name = '%s.pex.netlist' % cell_name

        # override parameters
        rcx_options['pexRunDir'] = run_dir
        rcx_options['pexLayoutPaths'] = gds_file
        rcx_options['pexLayoutPrimary'] = cell_name
        rcx_options['pexLayoutLibrary'] = lib_name
        rcx_options['pexLayoutView'] = lay_view
        rcx_options['pexSourcePath'] = netlist
        rcx_options['pexSourcePrimary'] = cell_name
        rcx_options['pexSourceLibrary'] = lib_name
        rcx_options['pexReportFile'] = '%s.lvs.report' % cell_name
        rcx_options['pexPexNetlistFile'] = output_name
        rcx_options['pexPexReportFile'] = '%s.pex.report' % cell_name
        rcx_options['pexMaskDBFile'] = '%s.maskdb' % cell_name
        rcx_options['cmnFDILayoutLibrary'] = lib_name
        rcx_options['cmnFDILayoutView'] = lay_view
        rcx_options['cmnFDIDEFLayoutPath'] = '%s.def' % cell_name

        rcx_options.update(rcx_params)

        content = ''.join(('*%s: %s\n' % (key, val) for key, val in rcx_options.items()))
        return content, os.path.join(run_dir, output_name)

    def modify_xact_rules(self, cell_name, gds_file, netlist, xact_params):
        """Modify the given XACT runset file.

        Parameters
        ----------
        cell_name : str
            the cell name.
        gds_file : str
            the layout gds file name.
        netlist : str
            the schematic netlist file.
        xact_params : dict[string, any]
            additional XACT parameters.

        Returns
        -------
        content : str
            the new runset content.
        output_name : str
            the extracted netlist file.
        """
        substrate_name = xact_params.get('substrate_name', 'VSS')
        power_names = xact_params.get('power_names', 'VDD')
        ground_names = xact_params.get('ground_names', 'VSS')

        run_dir = os.path.abspath(self.rcx_run_dir)

        template = read_file(self.xact_rules)
        output_name = '%s.pex.netlist' % cell_name
        content = Template(template).render(cell_name=cell_name,
                                            gds_file=gds_file,
                                            netlist=netlist,
                                            substrate_name=substrate_name,
                                            power_names=power_names,
                                            ground_names=ground_names,
                                            output_name=output_name,
                                            )

        return content, os.path.join(run_dir, output_name)

    def _lvs_task(self, proc_id, quit_dict, lib_name, cell_name, sch_view, lay_view,
                  lvs_params, log_file):

        gds_file, netlist = export_lay_sch('lvs', self.lvs_run_dir, proc_id, quit_dict, lib_name, cell_name,
                                           sch_view, lay_view, self.source_added_file, log_file)
        if proc_id in quit_dict:
            return False

        # generate new runset
        runset_content = self.modify_lvs_runset(lib_name, cell_name, lay_view, gds_file, netlist,
                                                lvs_params)

        # save runset
        with open_temp(dir=self.lvs_run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['calibre', '-gui', '-lvs', '-runset', runset_fname, '-batch']

        with open_temp(prefix='lvsOut', dir=self.tmp_dir, delete=False) as lvsf:
            lvs_file = lvsf.name

        write_file(log_file, ('**********************\n'
                              'Running LVS Comparison\n'
                              '**********************\n\n'
                              'runset file: %s\n'
                              'lvs output file: %s\n'
                              'cmd: %s\n') % (runset_fname, lvs_file, ' '.join(cmd)),
                   append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nLVS cancelled.\n', append=True)
            return False

        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=lvs_file,
                                   cwd=self.lvs_run_dir)

        if proc_id in quit_dict:
            write_file(log_file, '\nLVS cancelled.\n', append=True)
            return False

        cmd_output = read_file(lvs_file)

        write_file(log_file, ('**********************\n'
                              'Finish LVS Comparison.\n'
                              '**********************\n\n'),
                   append=True)

        # check LVS passed
        return lvs_passed(cmd_output)

    def run_lvs(self, lib_name, cell_name, sch_view, lay_view, lvs_params,
                block=True, callback=None):
        """Run LVS.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        sch_view : str
            the schematic view.
        lay_view : str
            the layout view.
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
        # update default LVS parameters.
        lvs_params_actual = self.default_lvs_params.copy()
        lvs_params_actual.update(lvs_params)

        with open_temp(prefix='lvsLog', dir=self.tmp_dir, delete=False) as logf:
            log_file = logf.name

        def lvs_task_wrapper(proc_id, quit_dict):
            success = self._lvs_task(proc_id, quit_dict, lib_name, cell_name,
                                     sch_view, lay_view, lvs_params_actual, log_file)
            if proc_id in quit_dict:
                del quit_dict[proc_id]
            return success, log_file

        if callback is not None:
            def callback_wrapper(future):
                return callback(future.result())
        else:
            callback_wrapper = None

        basename = 'lvs__%s__%s' % (lib_name, cell_name)

        pid = self._manager.new_thread(lvs_task_wrapper, basename=basename,
                                       callback=callback_wrapper)
        if block:
            result = self._manager.wait(pid)
            if result is None:
                return False, log_file
            return result
        else:
            return pid, log_file

    def _xact_task(self, proc_id, quit_dict, lib_name, cell_name, sch_view, lay_view,
                   params, log_file):

        gds_file, netlist = export_lay_sch('xact', self.rcx_run_dir, proc_id, quit_dict, lib_name, cell_name,
                                           sch_view, lay_view, self.source_added_file, log_file)
        if proc_id in quit_dict:
            return None

        num_cores = params.get('num_cores', 2)
        extract_mode = params.get('extract_mode', 'rcc')
        # generate new runset
        runset_content, result = self.modify_xact_rules(cell_name, gds_file, netlist, params)

        # save runset
        with open_temp(dir=self.rcx_run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        with open_temp(prefix='lvsOut', dir=self.tmp_dir, delete=False) as lvsf:
            lvs_file = lvsf.name

        cmd = ['calibre', '-lvs', '-hier', '-turbo', '%d' % num_cores, '-nowait', runset_fname]
        write_file(log_file, ('**********************\n'
                              'Running LVS Comparison\n'
                              '**********************\n\n'
                              'runset file: %s\n'
                              'lvs output file: %s\n'
                              '%s\n') % (runset_fname, lvs_file, ' '.join(cmd)),
                   append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nLVS cancelled.\n', append=True)
            return None

        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=lvs_file,
                                   cwd=self.rcx_run_dir)

        if proc_id in quit_dict:
            write_file(log_file, '\nLVS cancelled.\n', append=True)
            return None

        cmd_output = read_file(lvs_file)
        write_file(log_file, ('**********************\n'
                              'Finish LVS Comparison.\n'
                              '**********************\n\n'),
                   append=True)

        if not lvs_passed(cmd_output):
            write_file(log_file, '\nLVS failed.\n', append=True)
            return None

        cmd = ['calibre', '-xact', '-3d', '-%s' % extract_mode, '-turbo', '%d' % num_cores, runset_fname]
        write_file(log_file, ('****************\n'
                              'Running XACT 3D.\n'
                              '****************\n\n'
                              '%s\n') % ' '.join(cmd),
                   append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nXACT 3D cancelled.\n', append=True)
            return None

        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=log_file,
                                   cwd=self.rcx_run_dir, append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nXACT 3D cancelled.\n', append=True)
            return None

        write_file(log_file, ('****************\n'
                              'XACT 3D Finished\n'
                              '****************\n\n'
                              '%s\n'), append=True)

        return result

    def _pex_task(self, proc_id, quit_dict, lib_name, cell_name, sch_view, lay_view,
                  params, log_file):

        gds_file, netlist = export_lay_sch('pex', self.rcx_run_dir, proc_id, quit_dict, lib_name, cell_name,
                                           sch_view, lay_view, self.source_added_file, log_file)
        if proc_id in quit_dict:
            return None

        # generate new runset
        runset_content, result = self.modify_pex_runset(lib_name, cell_name, lay_view, gds_file, netlist,
                                                        params)

        # save runset
        with open_temp(dir=self.rcx_run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['calibre', '-gui', '-pex', '-runset', runset_fname, '-batch']
        write_file(log_file, ('**********************\n'
                              'Running RC Extraction.\n'
                              '**********************\n\n'
                              '%s\n') % ' '.join(cmd),
                   append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nRCX cancelled.\n', append=True)
            return None

        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=log_file,
                                   cwd=self.rcx_run_dir, append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nRCX cancelled.\n', append=True)
            return None

        write_file(log_file, ('**********************\n'
                              'Finished RC Extraction\n'
                              '**********************\n\n'
                              '%s\n'), append=True)

        return result

    def run_rcx(self, lib_name, cell_name, sch_view, lay_view, rcx_params,
                block=True, callback=None):
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If block is True this method will run RCX and return the extracted
        netlist filename.  If RCX fails, None will returned instead.

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
            and log file name when RCX finished.

        Returns
        -------
        value : string
            If blocking, the netlist file name.  If not blocking, the RCX ID.
        log_fname : str
            name of the RCX log file.
        """
        # update default RCX parameters.
        rcx_params_actual = self.default_rcx_params.copy()
        rcx_params_actual.update(rcx_params)

        log_file = None
        with open_temp(prefix='rcxLog', dir=self.tmp_dir, delete=False) as logf:
            log_file = logf.name

        if self.rcx_mode == 'pex':

            def rcx_task_wrapper(proc_id, quit_dict):
                netlist = self._pex_task(proc_id, quit_dict, lib_name, cell_name,
                                         sch_view, lay_view, rcx_params_actual, log_file)
                if proc_id in quit_dict:
                    del quit_dict[proc_id]
                return netlist, log_file
        elif self.rcx_mode == 'xact':

            def rcx_task_wrapper(proc_id, quit_dict):
                netlist = self._xact_task(proc_id, quit_dict, lib_name, cell_name,
                                          sch_view, lay_view, rcx_params_actual, log_file)
                if proc_id in quit_dict:
                    del quit_dict[proc_id]
                return netlist, log_file
        else:
            raise Exception('Unknown RCX mode: %s' % self.rcx_mode)

        if callback is not None:
            def callback_wrapper(future):
                return callback(future.result())
        else:
            callback_wrapper = None

        basename = 'rcx__%s__%s' % (lib_name, cell_name)

        pid = self._manager.new_thread(rcx_task_wrapper, basename=basename,
                                       callback=callback_wrapper)
        if block:
            result = self._manager.wait(pid)
            if result is None:
                return None, log_file
            return result
        else:
            return pid, log_file

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
        result, log_file = self._manager.wait(job_id, timeout=timeout, cancel_timeout=cancel_timeout)
        return result

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
        return self._manager.cancel(job_id, timeout=timeout)

    def done(self, proc_id):
        # type: (str) -> bool
        """Returns True if the given process finished or is cancelled successfully.

        Parameters
        ----------
        proc_id : str
            the process ID.

        Returns
        -------
        done : bool
            True if the process is cancelled or completed.
        """
        return self._manager.done(proc_id)
