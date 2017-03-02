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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import pprint

from jinja2 import Template

from ..io import read_resource, write_file, open_temp
from ..io import process

lay_template = read_resource(__name__, os.path.join('templates', 'layout_export_config.pytemp'))
sch_template = read_resource(__name__, os.path.join('templates', 'si_env.pytemp'))


def export_lay_sch(job_name, run_dir, proc_id, quit_dict, lib_name, cell_name,
                   sch_view, lay_view, source_added_file, log_file):
    """Export both layout and schematic from Virtuoso.

    Parameters
    ----------
    job_name : string
        name of the job.
    run_dir : string
        the run directory.
    proc_id : string
        the process identification string.
    quit_dict : dict[string, float]
        dictionary from process ID to timeout.
    lib_name : string
        the library name.
    cell_name : string
        the cell name.
    sch_view : string
        the schematic view.
    lay_view : string
        the layout view.
    source_added_file : string
        the source.added file location.
    log_file : string
        the log file name.

    Returns
    -------
    gds_file : string
        the gds file name.
    netlist_file : string
        the schematic netlist file name.
    """
    if proc_id in quit_dict:
        write_file(log_file, '\n%s cancelled.\n' % job_name, append=True)
        return None, None

    # export layout and schematic
    gds_file = export_layout(proc_id, quit_dict, lib_name, cell_name, lay_view, run_dir, log_file)
    if proc_id in quit_dict:
        write_file(log_file, '\n%s cancelled.\n' % job_name, append=True)
        return None, None
    netlist = export_schematic(proc_id, quit_dict, lib_name, cell_name, sch_view, source_added_file,
                               run_dir, log_file)
    if proc_id in quit_dict:
        write_file(log_file, '\n%s cancelled.\n' % job_name, append=True)
        return None, None

    return gds_file, netlist


def export_layout(proc_id, quit_dict, lib_name, cell_name, lay_view, run_dir, log_file):
    """Export the given layout cell from Virtuoso.

    Parameters
    ----------
    proc_id : string
        the process ID.
    quit_dict : dict[string, float]
        dictionary from process ID to timeout.
    lib_name : str
        the library name.
    cell_name : str
        the cell name.
    lay_view : str
        the layout view.
    run_dir : str
        the run directory.
    log_file : str
        the log file name.

    Returns
    -------
    fname : str
        the output file name.
    """
    write_file(log_file, '**********************\n'
                         'Streaming out gds file\n'
                         '**********************\n\n')

    run_dir = os.path.abspath(run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    output_name = '%s.calibre.db' % cell_name

    # fill in stream out configuration file.
    content = Template(lay_template).render(lib_name=lib_name,
                                            cell_name=cell_name,
                                            view_name=lay_view,
                                            output_name=output_name,
                                            run_dir=run_dir,
                                            )

    with open_temp(prefix='stream_template', dir=run_dir, delete=False) as config_file:
        config_fname = config_file.name
        config_file.write(content)

    write_file(log_file, 'template file: %s\n' % config_fname, append=True)

    # hack PYTHONPATH so pycell evaluates correctly
    env_copy = {}
    env_copy.update(os.environ)
    if 'PYTHONPATH_PYCELL' in env_copy:
        env_copy['PYTHONPATH'] = env_copy.pop('PYTHONPATH_PYCELL')

    # run strmOut
    cmd = ['strmout', '-templateFile', config_fname]
    write_file(log_file, 'environment:\n%s\n%s\n' % (pprint.pformat(env_copy), ' '.join(cmd)),
               append=True)

    if proc_id in quit_dict:
        write_file(log_file, 'Layout export cancelled.\n', append=True)
        return None

    process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=log_file, append=True,
                               env=env_copy)

    if proc_id in quit_dict:
        write_file(log_file, '\nLayout export cancelled.\n', append=True)
        return None

    write_file(log_file, '**********************\n'
                         'Finish gds file export\n'
                         '**********************\n\n',
               append=True)
    return os.path.join(run_dir, output_name)


def export_schematic(proc_id, quit_dict, lib_name, cell_name, sch_view, source_added_file, run_dir, log_file):
    """Export the given schematic cell from virtuoso.

    Parameters
    ----------
    proc_id : string
        the process ID.
    quit_dict : dict[string, float]
        dictionary from process ID to timeout.
    lib_name : str
        the library name.
    cell_name : str
        the cell name.
    sch_view : str
        the schematic view.
    source_added_file : str
        the source.added file location.
    run_dir : str
        the run directory.
    log_file : str
        the log file name.

    Returns
    -------
    fname : str
        the output file name.
    """
    write_file(log_file, '**********************\n'
                         'Exporting netlist file\n'
                         '**********************\n\n')

    run_dir = os.path.abspath(run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    output_name = '%s.src.net' % cell_name

    # fill in stream out configuration file.
    content = Template(sch_template).render(lib_name=lib_name,
                                            cell_name=cell_name,
                                            view_name=sch_view,
                                            output_name=output_name,
                                            source_added_file=source_added_file,
                                            run_dir=run_dir,
                                            )

    # create configuration file.
    config_fname = os.path.join(run_dir, 'si.env')
    write_file(config_fname, content)

    # run command
    cmd = ['si', run_dir, '-batch', '-command', 'netlist']
    write_file(log_file, 'si.env file: %s\n%s\n' % (config_fname, ' '.join(cmd)), append=True)

    if proc_id in quit_dict:
        write_file(log_file, 'netlist export cancelled.\n', append=True)
        return None

    process.run_proc_with_quit(proc_id, quit_dict, cmd, logfile=log_file, append=True)

    if proc_id in quit_dict:
        write_file(log_file, 'netlist export cancelled.\n', append=True)
        return None

    write_file(log_file, '**********************\n'
                         'Finish netlist export.\n'
                         '**********************\n\n',
               append=True)

    return os.path.join(run_dir, output_name)
