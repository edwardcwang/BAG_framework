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

"""This module handles exporting schematic/layout from Virtuoso.
"""

import os
import abc
from typing import TYPE_CHECKING, Optional, Dict, Any

from jinja2 import Template

from ..io import read_resource, write_file, open_temp
from .base import SubProcessChecker

lay_template = read_resource(__name__, os.path.join('templates', 'layout_export_config.pytemp'))
sch_template = read_resource(__name__, os.path.join('templates', 'si_env.pytemp'))

if TYPE_CHECKING:
    from .base import ProcInfo


class VirtuosoChecker(SubProcessChecker, metaclass=abc.ABCMeta):
    """the base Checker class for Virtuoso.

    This class implement layout/schematic export procedures.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory.
    max_workers : int
        maximum number of parallel processes.
    cancel_timeout : float
        timeout for cancelling a subprocess.
    source_added_file : str
        file to include for schematic export.
    """
    def __init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file):
        # type: (str, int, float, str) -> None
        SubProcessChecker.__init__(self, tmp_dir, max_workers, cancel_timeout)
        self._source_added_file = source_added_file

    def setup_export_layout(self, lib_name, cell_name, out_file, view_name='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'layout_export.log')

        os.makedirs(run_dir, exist_ok=True)

        # fill in stream out configuration file.
        content = Template(lay_template).render(lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                run_dir=run_dir,
                                                )

        with open_temp(prefix='stream_template', dir=run_dir, delete=False) as config_file:
            config_fname = config_file.name
            config_file.write(content)

        # run strmOut
        cmd = ['strmout', '-templateFile', config_fname]

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']

    def setup_export_schematic(self, lib_name, cell_name, out_file, view_name='schematic', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'schematic_export.log')

        # fill in stream out configuration file.
        content = Template(sch_template).render(lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                source_added_file=self._source_added_file,
                                                run_dir=run_dir,
                                                )

        # create configuration file.
        config_fname = os.path.join(run_dir, 'si.env')
        write_file(config_fname, content)

        # run command
        cmd = ['si', run_dir, '-batch', '-command', 'netlist']

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']
