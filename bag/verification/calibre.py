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

import os
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any, Sequence

from jinja2 import Template

from .virtuoso import VirtuosoChecker
from ..io import read_file, open_temp, readlines_iter

if TYPE_CHECKING:
    from .base import FlowInfo


# noinspection PyUnusedLocal
def _all_pass(retcode, log_file):
    return True


# noinspection PyUnusedLocal
def lvs_passed(retcode, log_file):
    # type: (int, str) -> Tuple[bool, str]
    """Check if LVS passed

    Parameters
    ----------
    retcode : int
        return code of the LVS process.
    log_file : str
        log file name.

    Returns
    -------
    success : bool
        True if LVS passed.
    log_file : str
        the log file name.
    """
    if not os.path.isfile(log_file):
        return False, ''

    cmd_output = read_file(log_file)
    test_str = 'LVS completed. CORRECT. See report file:'
    return test_str in cmd_output, log_file


class Calibre(VirtuosoChecker):
    """A subclass of VirtuosoChecker that uses Calibre for verification.

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

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        if cancel_timeout is not None:
            cancel_timeout /= 1e3

        VirtuosoChecker.__init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file)

        self.default_rcx_params = kwargs.get('rcx_params', {})
        self.default_lvs_params = kwargs.get('lvs_params', {})
        self.lvs_run_dir = lvs_run_dir
        self.lvs_runset = lvs_runset
        self.rcx_run_dir = rcx_run_dir
        self.rcx_runset = rcx_runset
        self.xact_rules = xact_rules
        self.rcx_mode = rcx_mode

    def get_rcx_netlists(self, lib_name, cell_name):
        # type: (str, str) -> List[str]
        """Returns a list of generated extraction netlist file names.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name

        Returns
        -------
        netlists : List[str]
            a list of generated extraction netlist file names.  The first index is the main netlist.
        """
        # PVS generate schematic cellviews directly.
        return ['%s.pex.netlist' % cell_name,
                '%s.pex.netlist.pex' % cell_name,
                '%s.pex.netlist.%s.pxi' % (cell_name, cell_name),
                ]

    def setup_lvs_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Sequence[FlowInfo]

        run_dir = os.path.join(self.lvs_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file, sch_file = self._get_lay_sch_files(run_dir)

        # add schematic/layout export to flow
        flow_list = []
        cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))
        cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file, sch_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))

        lvs_params_actual = self.default_lvs_params.copy()
        if params is not None:
            lvs_params_actual.update(params)

        with open_temp(prefix='lvsLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        # generate new runset
        runset_content = self.modify_lvs_runset(lib_name, cell_name, lay_view, lay_file, sch_file, lvs_params_actual)

        # save runset
        with open_temp(dir=run_dir, delete=False) as runset_file:
            runset_fname = runset_file.name
            runset_file.write(runset_content)

        cmd = ['calibre', '-gui', '-lvs', '-runset', runset_fname, '-batch']

        flow_list.append((cmd, log_file, None, self.lvs_run_dir, lvs_passed))

        return flow_list

    def setup_rcx_flow(self, lib_name, cell_name, sch_view='schematic', lay_view='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Sequence[FlowInfo]

        # update default RCX parameters.
        rcx_params_actual = self.default_rcx_params.copy()
        if params is not None:
            rcx_params_actual.update(params)

        run_dir = os.path.join(self.rcx_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)
        lay_file, sch_file = self._get_lay_sch_files(run_dir)

        with open_temp(prefix='rcxLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        flow_list = []
        cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))
        cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file, sch_view, None)
        flow_list.append((cmd, log, env, cwd, _all_pass))

        if self.rcx_mode == 'pex':
            # generate new runset
            runset_content, result = self.modify_pex_runset(lib_name, cell_name, lay_view,
                                                            lay_file, sch_file, rcx_params_actual)

            # save runset
            with open_temp(dir=run_dir, delete=False) as runset_file:
                runset_fname = runset_file.name
                runset_file.write(runset_content)

            cmd = ['calibre', '-gui', '-pex', '-runset', runset_fname, '-batch']
        else:
            # generate new runset
            runset_content, result = self.modify_xact_rules(cell_name, lay_file, sch_file, rcx_params_actual)

            # save runset
            with open_temp(dir=run_dir, delete=False) as runset_file:
                runset_fname = runset_file.name
                runset_file.write(runset_content)

            with open_temp(prefix='lvsLog', dir=run_dir, delete=False) as lvsf:
                lvs_file = lvsf.name

            num_cores = rcx_params_actual.get('num_cores', 2)
            cmd = ['calibre', '-lvs', '-hier', '-turbo', '%d' % num_cores, '-nowait', runset_fname]
            flow_list.append((cmd, lvs_file, None, self.rcx_run_dir, lambda rc, lf: lvs_passed(rc, lf)[0]))

            extract_mode = rcx_params_actual.get('extract_mode', 'rcc')
            cmd = ['calibre', '-xact', '-3d', '-%s' % extract_mode, '-turbo', '%d' % num_cores, runset_fname]

        # noinspection PyUnusedLocal
        def rcx_passed(retcode, log_fname):
            if not os.path.isfile(result):
                return None, log_fname
            return result, log_fname

        flow_list.append((cmd, log_file, None, self.rcx_run_dir, rcx_passed))
        return flow_list

    @classmethod
    def _get_lay_sch_files(cls, run_dir):
        lay_file = os.path.join(run_dir, 'layout.gds')
        sch_file = os.path.join(run_dir, 'schematic.net')
        return lay_file, sch_file

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
