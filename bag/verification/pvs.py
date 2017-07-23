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

"""This module implements LVS/RCX using PVS/QRC and stream out from Virtuoso.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import yaml
from typing import Optional, Union

from .base import Checker
from ..io import read_file, write_file, open_temp, readlines_iter, fix_string
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
    test_str = '# Run Result             : MATCH'
    return test_str in cmd_output


def rcx_passed(cmd_output):
    """Returns True if RCX passed.

    Parameters
    ----------
    cmd_output : str
        output from RCX.

    Returns
    -------
    flag : bool
        True if RCX passed.
    """
    test_str = 'INFO (LBRCXM-708): *****  Quantus QRC terminated normally  *****'
    return test_str in cmd_output


class PVS(Checker):
    """A subclass of Checker that uses PVS/QRC for verification.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    lvs_run_dir : string
        the LVS run directory.
    lvs_runset : string
        the LVS runset filename.
    lvs_rule_file : string
        the LVS rule filename.
    rcx_runset : string
        the RCX runset filename.
    source_added_file : string
        the source.added file location.  Environment variable is supported.
        Default value is '$DK/Calibre/lvs/source.added'.
    """

    def __init__(self, tmp_dir, lvs_run_dir, lvs_runset, lvs_rule_file, rcx_runset,
                 source_added_file='$DK/Calibre/lvs/source.added', **kwargs):
        Checker.__init__(self, tmp_dir)

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        if cancel_timeout is not None:
            cancel_timeout /= 1e3
        self._manager = process.ProcessManager(max_workers=max_workers,
                                               cancel_timeout=cancel_timeout)

        self.default_rcx_params = kwargs.get('rcx_params', {})
        self.default_lvs_params = kwargs.get('lvs_params', {})
        self.lvs_run_dir = lvs_run_dir
        self.lvs_runset = lvs_runset
        self.lvs_rule_file = lvs_rule_file
        self.rcx_runset = rcx_runset
        self.source_added_file = source_added_file or '$DK/Calibre/lvs/source.added'

    def modify_lvs_runset(self, cell_name, lvs_params):
        """Modify the given LVS runset file.

        Parameters
        ----------
        cell_name : str
            the cell name.
        lvs_params : dict[str, any]
            override LVS parameters.

        Returns
        -------
        content : str
            the new runset content.
        """
        run_dir = os.path.join(os.path.abspath(self.lvs_run_dir), cell_name)
        os.makedirs(run_dir, exist_ok=True)
        # convert runset content to dictionary
        lvs_options = {}
        for line in readlines_iter(self.lvs_runset):
            key, val = line.split(' ', 1)
            # remove semicolons
            val = val.strip().rstrip(';')
            if key in lvs_options:
                lvs_options[key].append(val)
            else:
                lvs_options[key] = [val]

        # get results_db file name
        results_db = os.path.join(run_dir, '%s.erc_errors.ascii' % cell_name)
        # override parameters
        lvs_options['lvs_report_file'] = ['"%s.rep"' % cell_name]
        lvs_options['report_summary'] = ['-erc "%s.sum" -replace' % cell_name]
        lvs_options['results_db'] = ['-erc "%s" -ascii' % results_db]
        lvs_options['mask_svdb_dir'] = ['"%s"' % os.path.join(run_dir, 'svdb')]

        lvs_options.update(lvs_params)
        content_list = []
        for key, val_list in lvs_options.items():
            for v in val_list:
                content_list.append('%s %s;\n' % (key, v))

        return ''.join(content_list)

    def modify_rcx_runset(self, lib_name, cell_name, lay_view, rcx_params):
        """Modify the given QRC options.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        lay_view : str
            the layout view.
        rcx_params : dict[str, any]
            override RCX parameters.

        Returns
        -------
        content : str
            the new runset content.
        output_name : str
            the extracted netlist file.
        """
        run_dir = os.path.join(os.path.abspath(self.lvs_run_dir), cell_name)
        data_dir = os.path.join(run_dir, 'svdb')
        if not os.path.isdir(data_dir):
            raise ValueError('cannot find directory %s.  Did you run PVS first?' % data_dir)

        # load default rcx options
        content = read_file(self.rcx_runset)
        rcx_options = yaml.load(content)

        # setup inputs/outputs
        rcx_options['input_db']['design_cell_name'] = '{} {} {}'.format(cell_name, lay_view, lib_name)
        rcx_options['input_db']['run_name'] = cell_name
        rcx_options['input_db']['directory_name'] = data_dir
        rcx_options['output_db']['cdl_out_map_directory'] = run_dir
        rcx_options['output_setup']['directory_name'] = data_dir
        rcx_options['output_setup']['temporary_directory_name'] = cell_name

        # override parameters
        for key, val in rcx_options.items():
            if key in rcx_params:
                val.update(rcx_params[key])

        # convert dictionary to QRC command file format.
        content_list = []
        for key, options in rcx_options.items():
            content_list.append('%s \\' % key)
            for k, v in options.items():
                v = fix_string(v)
                if isinstance(v, str):
                    # add quotes around string
                    v = '"{}"'.format(v)
                content_list.append('    -%s %s \\' % (k, v))

            # remove line continuation backslash from last option
            content_list[-1] = content_list[-1][:-2]

        return '\n'.join(content_list), ''

    def _lvs_task(self, proc_id, quit_dict, lib_name, cell_name, sch_view, lay_view,
                  lvs_params, log_file):
        run_dir = os.path.join(os.path.abspath(self.lvs_run_dir), cell_name)
        os.makedirs(run_dir, exist_ok=True)
        gds_file, netlist = export_lay_sch('lvs', run_dir, proc_id, quit_dict, lib_name, cell_name,
                                           sch_view, lay_view, self.source_added_file, log_file)
        if proc_id in quit_dict:
            return False

        # generate new runset
        runset_content = self.modify_lvs_runset(cell_name, lvs_params)

        # save runset
        with open_temp(dir=run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['pvs', '-perc', '-lvs', '-qrc_data', '-control', runset_fname,
               '-gds', gds_file, '-layout_top_cell', cell_name,
               '-source_cdl', netlist, '-source_top_cell', cell_name,
               self.lvs_rule_file,
               ]

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

        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=lvs_file, cwd=run_dir)

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

    def _rcx_task(self, proc_id, quit_dict, lib_name, cell_name, lay_view, params, log_file):
        run_dir = os.path.join(os.path.abspath(self.lvs_run_dir), cell_name)

        # generate new runset
        runset_content, result = self.modify_rcx_runset(lib_name, cell_name, lay_view, params)

        # save runset
        with open_temp(dir=run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['qrc', '-cmd', runset_fname]
        write_file(log_file, ('**********************\n'
                              'Running RC Extraction.\n'
                              '**********************\n\n'
                              'runset file: %s\n'
                              'cmd: %s\n') % (runset_fname, ' '.join(cmd)),
                   append=True)

        if proc_id in quit_dict:
            write_file(log_file, '\nRCX cancelled.\n', append=True)
            return None

        # NOTE: qrc needs to be run in the current working directory (virtuoso directory), because it needs to
        # access cds.lib
        process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=log_file, append=True)

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

        def rcx_task_wrapper(proc_id, quit_dict):
            netlist = self._rcx_task(proc_id, quit_dict, lib_name, cell_name,
                                     lay_view, rcx_params_actual, log_file)
            if proc_id in quit_dict:
                del quit_dict[proc_id]
            return netlist, log_file

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
        result = self._manager.wait(job_id, timeout=timeout, cancel_timeout=cancel_timeout)
        if result is None:
            return None
        return result[0]
