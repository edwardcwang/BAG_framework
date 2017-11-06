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

"""This module define utility classes for performing concurrent operations.
"""

import os
import asyncio
import subprocess
import multiprocessing
from concurrent.futures import CancelledError

from typing import TYPE_CHECKING, Optional, Sequence, Dict, Union

if TYPE_CHECKING:
    from asyncio.subprocess import Process


def batch_task(coro_list):
    """Execute a list of coroutines or futures concurrently.

    User may press Ctrl-C to cancel all given tasks.

    Parameters
    ----------
    coro_list :
        a list of coroutines or futures to run concurrently.

    Returns
    -------
    results :
        a list of return values or raised exceptions of given tasks.
    """
    top_future = asyncio.gather(*coro_list, return_exceptions=True)

    loop = asyncio.get_event_loop()
    try:
        print('Running tasks, Press Ctrl-C to cancel.')
        results = loop.run_until_complete(top_future)
    except KeyboardInterrupt:
        print('Ctrl-C detected, Cancelling tasks.')
        top_future.cancel()
        loop.run_forever()
        results = None

    return results


class SubProcessManager(object):
    """A class that provides convenient methods to run multiple subprocesses in parallel using asyncio.

    Parameters
    ----------
    max_workers : Optional[int]
        number of maximum allowed subprocesses.  If None, defaults to system
        CPU count.
    cancel_timeout : Optional[float]
        Number of seconds to wait for a process to terminate once SIGTERM or
        SIGKILL is issued.  Defaults to 10 seconds.
    """

    def __init__(self, max_workers=None, cancel_timeout=10.0):
        # type: (Optional[int], Optional[float]) -> None
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        if cancel_timeout is None:
            cancel_timeout = 10.0

        self._cancel_timeout = cancel_timeout
        self._semaphore = asyncio.Semaphore(max_workers)

    async def _kill_subprocess(self, proc):
        # type: (Optional[Process]) -> None
        """Helper method; send SIGTERM/SIGKILL to a subprocess.

        This method first sends SIGTERM to the subprocess.  If the process hasn't terminated
        after a given timeout, it sends SIGKILL.

        Parameter
        ---------
        proc : Optional[Process]
            the process to attempt to terminate.  If None, this method does nothing.
        """
        if proc is not None:
            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.shield(asyncio.wait_for(proc.wait(), self._cancel_timeout))
                    except CancelledError:
                        pass

                    if proc.returncode is None:
                        proc.kill()
                        try:
                            await asyncio.shield(asyncio.wait_for(proc.wait(), self._cancel_timeout))
                        except CancelledError:
                            pass
                except ProcessLookupError:
                    pass

    async def async_new_subprocess(self, args, log, append=False, env=None, cwd=None):
        # type: (Union[str, Sequence[str]], str, bool, Optional[Dict[str, str]], Optional[str]) -> Optional[int]
        """A asyncio coroutine which starts a subprocess.

        If this coroutine is cancelled, it will shut down the subprocess gracefully using SIGTERM/SIGKILL,
        then raise CancelledError.

        Parameters
        ----------
        args : Union[str, Sequence[str]]
            command to run, as string or sequence of strings.
        log : str
            the log file name.
        append : bool
            True to append to any existing log file instead of replacing it.
        env : Optional[Dict[str, str]]
            an optional dictionary of environment variables.  None to inherit from parent.
        cwd : Optional[str]
            the working directory.  None to inherit from parent.

        Returns
        -------
        retcode : Optional[int]
            the return code of the subprocess.
        """

        # get log file name, make directory if necessary
        log = os.path.abspath(log)
        if os.path.isdir(log):
            raise ValueError('log file %s is a directory.' % log)
        os.makedirs(os.path.dirname(log), exist_ok=True)

        async with self._semaphore:
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(args, stdout=log, stderr=subprocess.STDOUT,
                                                            env=env, cwd=cwd, append=append)
                retcode = await proc.wait()
                return retcode
            except CancelledError as err:
                await self._kill_subprocess(proc)
                raise err

    def batch_subprocess(self,
                         args_list,  # type: Sequence[Union[str, Sequence[str]]]
                         log_list,  # type: Sequence[str]
                         append_list=None,  # type: Optional[Sequence[bool]]
                         env_list=None,  # type: Optional[Sequence[Dict[str, str]]
                         cwd_list=None,  # type: Optional[Sequence[str]]
                         ):
        # type: (...) -> Optional[Sequence[Union[int, Exception]]]
        """Run all given subprocesses in parallel.

        Parameters
        ----------
        args_list : Sequence[Union[str, Sequence[str]]]
            list of commands to run, as string or list of string arguments.  See
            Python subprocess module documentation.
        log_list : Sequence[str]
            list of log file names.  stdout and stderr are written to these files.
        append_list : Optional[Sequence[bool]]
            list of boolean flags indicating whether to append (instead of overwrite)
            to log files.
        env_list : Optional[Sequence[Dict[str, str]]
            list of environment variable dictionaries.
        cwd_list : Optional[Sequence[str]]
            list of working directories for each process.

        Returns
        -------
        results : Optional[Sequence[Union[int, Exception]]]
            if user cancelled the subprocesses, None is returned.  Otherwise, a list of
            subprocess return codes or exceptions are returned.
        """
        num_proc = len(args_list)
        if num_proc == 0:
            return []

        # error checking
        if len(log_list) != num_proc:
            raise ValueError('log_list length != %d' % num_proc)
        if append_list is None:
            append_list = [False] * num_proc
        elif len(append_list) != num_proc:
            raise ValueError('append_list length != %d' % num_proc)
        if env_list is None:
            env_list = [None] * num_proc
        elif len(env_list) != num_proc:
            raise ValueError('env_list length != %d' % num_proc)
        if cwd_list is None:
            cwd_list = [None] * num_proc
        elif len(cwd_list) != num_proc:
            raise ValueError('cwd_list length != %d' % num_proc)

        coro_list = [self.async_new_subprocess(args, log, append, env, cwd) for args, log, append, env, cwd in
                     zip(args_list, log_list, append_list, env_list, cwd_list)]

        return batch_task(coro_list)
