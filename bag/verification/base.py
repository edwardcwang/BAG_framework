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

"""This module defines Checker, an abstract base class that handles LVS/RCX."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from future.utils import with_metaclass

import abc
from typing import Optional, Union


class Checker(with_metaclass(abc.ABCMeta, object)):
    """A class that handles LVS/RCX.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    """
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    @abc.abstractmethod
    def run_lvs(self, lib_name, cell_name, sch_view, lay_view, lvs_params,
                block=True, callback=None):
        """Run LVS on the given cell.

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
        return False

    @abc.abstractmethod
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
        return None

    @abc.abstractmethod
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
        return None
