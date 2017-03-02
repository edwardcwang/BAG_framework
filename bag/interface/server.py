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

"""This class defines SkillOceanServer, a server that handles skill/ocean requests.

The SkillOceanServer listens for skill/ocean requests from bag.  Skill commands will
be forwarded to Virtuoso for execution, and Ocean simulation requests will be handled
by starting an Ocean subprocess.  It also provides utility for bag to query simulation
progress and allows parallel simulation.

Client-side communication:

the client will always send a request object, which is a python dictionary.
This script processes the request and sends the appropriate commands to
Virtuoso.

Virtuoso side communication:

To ensure this process receive all the data from Virtuoso properly, Virtuoso
will print a single line of integer indicating the number of bytes to read.
Then, virtuoso will print out exactly that many bytes of data, followed by
a newline (to flush the standard input).  This script handles that protcol
and will strip the newline before sending result back to client.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import traceback

from jinja2 import Template

import bag.io
from .. import verification

calibre_tmp = bag.io.read_resource(bag.__name__, os.path.join('virtuoso_files', 'calibreview_setup.pytemp'))


def _object_to_skill_file_helper(py_obj, file_obj):
    """Recursive helper function for object_to_skill_file

    Parameters
    ----------
    py_obj : any
        the object to convert.
    file_obj : file
        the file object to write to.  Must be created with bag.io
        package so that encodings are handled correctly.
    """
    # fix potential raw bytes
    py_obj = bag.io.fix_string(py_obj)
    if isinstance(py_obj, str):
        # string
        file_obj.write(py_obj)
    elif isinstance(py_obj, float):
        # prepend type flag
        file_obj.write('#float {:f}'.format(py_obj))
    elif isinstance(py_obj, int):
        # prepend type flag
        file_obj.write('#int {:d}'.format(py_obj))
    elif isinstance(py_obj, list) or isinstance(py_obj, tuple):
        # a list of other objects.
        file_obj.write('#list\n')
        for val in py_obj:
            _object_to_skill_file_helper(val, file_obj)
            file_obj.write('\n')
        file_obj.write('#end')
    elif isinstance(py_obj, dict):
        # disembodied property lists
        file_obj.write('#prop_list\n')
        for key, val in py_obj.items():
            file_obj.write('{}\n'.format(key))
            _object_to_skill_file_helper(val, file_obj)
            file_obj.write('\n')
        file_obj.write('#end')
    else:
        raise Exception('Unsupported python data type: %s' % type(py_obj))


def object_to_skill_file(py_obj, file_obj):
    """Write the given python object to a file readable by Skill.

    Write a Python object to file that can be parsed into equivalent
    skill object by Virtuoso.  Currently only strings, lists, and dictionaries
    are supported.

    Parameters
    ----------
    py_obj : any
        the object to convert.
    file_obj : file
        the file object to write to.  Must be created with bag.io
        package so that encodings are handled correctly.
    """
    _object_to_skill_file_helper(py_obj, file_obj)
    file_obj.write('\n')


bag_proc_prompt = 'BAG_PROMPT>>> '


class SkillServer(object):
    """A server that handles skill commands.

    This server is started and ran by virtuoso.  It listens for commands from bag
    from a ZMQ socket, then pass the command to virtuoso.  It then gather the result
    and send it back to bag.

    Parameters
    ----------
    router : :class:`bag.interface.ZMQRouter`
        the :class:`~bag.interface.ZMQRouter` object used for socket communication.
    virt_in : file
        the virtuoso input file.  Must be created with bag.io
        package so that encodings are handled correctly.
    virt_out : file
        the virtuoso output file.  Must be created with bag.io
        package so that encodings are handled correctly.
    tmpdir : str or None
        if given, will save all temporary files to this folder.
    """

    def __init__(self, router, virt_in, virt_out, tmpdir=None):
        """Create a new SkillOceanServer instance.
        """
        self.handler = router
        self.virt_in = virt_in
        self.virt_out = virt_out
        self.checker = None  # type: verification.base.Checker
        self.calview_cell_map = None
        self.calview_name = None

        # create a directory for all temporary files
        self.dtmp = bag.io.make_temp_dir('skillTmp', parent_dir=tmpdir)

    def run(self):
        """Starts this server.
        """
        while not self.handler.is_closed():
            # check if socket received message
            if self.handler.poll_for_read(5):
                req = self.handler.recv_obj()
                if isinstance(req, dict) and 'type' in req:
                    if req['type'] == 'exit':
                        self.close()
                    elif req['type'] == 'skill':
                        expr, out_file = self.process_skill_request(req)
                        if expr is not None:
                            # send expression to virtuoso
                            self.send_skill(expr)
                            msg = self.recv_skill()
                            self.process_skill_result(msg, out_file)
                    elif req['type'] == 'init_checker':
                        self.process_checker_request(req)
                    elif req['type'] == 'lvs':
                        self.process_lvs_request(req)
                    elif req['type'] == 'rcx':
                        self.process_rcx_request(req)
                    else:
                        msg = '*Error* bag server error: bag request:\n%s' % str(req)
                        self.handler.send_obj(dict(type='error', data=msg))
                else:
                    msg = '*Error* bag server error: bag request:\n%s' % str(req)
                    self.handler.send_obj(dict(type='error', data=msg))

    def send_skill(self, expr):
        """Sends expr to virtuoso for evaluation.

        Parameters
        ----------
        expr : string
            the skill expression.
        """
        self.virt_in.write(expr)
        self.virt_in.flush()

    def recv_skill(self):
        """Receive response from virtuoso"""
        num_bytes = int(self.virt_out.readline())
        msg = self.virt_out.read(num_bytes)
        if msg[-1] == '\n':
            msg = msg[:-1]
        return msg

    def close(self):
        """Close this server."""
        self.handler.close()

    def process_skill_request(self, request):
        """Process the given skill request.

        Based on the given request object, returns the skill expression
        to be evaluated by Virtuoso.  This method creates temporary
        files for long input arguments and long output.

        Parameters
        ----------
        request : dict
            the request object.

        Returns
        -------
        expr : str or None
            expression to be evaluated by Virtuoso.  If None, an error occurred and
            nothing needs to be evaluated
        out_file : str or None
            if not None, the result will be written to this file.
        """
        try:
            expr = request['expr']
            input_files = request['input_files'] or {}
            out_file = request['out_file']
        except KeyError as e:
            msg = '*Error* bag server error: %s' % str(e)
            self.handler.send_obj(dict(type='error', data=msg))
            return None, None

        fname_dict = {}
        # write input parameters to files
        for key, val in input_files.items():
            with bag.io.open_temp(prefix=key, delete=False, dir=self.dtmp) as file_obj:
                fname_dict[key] = '"%s"' % file_obj.name
                try:
                    object_to_skill_file(val, file_obj)
                except Exception:
                    stack_trace = traceback.format_exc()
                    msg = '*Error* bag server error: \n%s' % stack_trace
                    self.handler.send_obj(dict(type='error', data=msg))
                    return None, None

        # generate output file
        if out_file:
            with bag.io.open_temp(prefix=out_file, delete=False, dir=self.dtmp) as file_obj:
                fname_dict[out_file] = '"%s"' % file_obj.name
                out_file = file_obj.name

        # fill in parameters to expression
        expr = expr.format(**fname_dict)
        return expr, out_file

    def process_skill_result(self, msg, out_file=None):
        """Process the given skill output, then send result to socket.

        Parameters
        ----------
        msg : str
            skill expression evaluation output.
        out_file : str or None
            if not None, read result from this file.
        """
        # read file if needed, and only if there are no errors.
        if msg.startswith('*Error*'):
            # an error occurred, forward error message directly
            self.handler.send_obj(dict(type='error', data=msg))
        elif out_file:
            # read result from file.
            try:
                msg = bag.io.read_file(out_file)
                data = dict(type='str', data=msg)
            except IOError:
                stack_trace = traceback.format_exc()
                msg = '*Error* error reading file:\n%s' % stack_trace
                data = dict(type='error', data=msg)
            self.handler.send_obj(data)
        else:
            # return output from virtuoso directly
            self.handler.send_obj(dict(type='str', data=msg))

    def process_checker_request(self, req):
        """Process the given checker request."""
        try:
            self.checker = verification.make_checker(**req['kwargs'])
            self.calview_cell_map = req['calview_cell_map']
            self.calview_name = req['calview_name']
        except Exception:
            stack_trace = traceback.format_exc()
            msg = '*Error* error creating Checker: %s' % stack_trace
            self.handler.send_obj(dict(type='error', data=msg))
            self.checker = None

        if self.checker is not None:
            self.handler.send_obj(dict(type='str', data='done'))

    def process_lvs_request(self, req):
        """Process the given LVS request."""
        if self.checker is None:
            msg = '*Error* Checker is not initialized.'
            self.handler.send_obj(dict(type='error', data=msg))
            return

        try:
            lib_name = req['lib_name']
            cell_name = req['cell_name']
            lay_view = req['lay_view']
            sch_view = req['sch_view']
            lvs_params = req['lvs_params']
        except KeyError:
            stack_trace = traceback.format_exc()
            msg = '*Error* malformed request: %s\nstack trace: \n%s' % (str(req), stack_trace)
            self.handler.send_obj(dict(type='error', data=msg))
            return

        with bag.io.open_temp(prefix='lvsLog', delete=False, dir=self.dtmp) as logfile:
            log_fname = logfile.name
        try:
            result = self.checker.run_lvs(lib_name, cell_name, lay_view, sch_view, log_fname, lvs_params)
            self.handler.send_obj(dict(type='tuple', data=(result, log_fname)))
        except Exception:
            stack_trace = traceback.format_exc()
            msg = '*Error* error running LVS:\n%s' % stack_trace
            self.handler.send_obj(dict(type='error', data=msg))

    def process_rcx_request(self, req):
        """Process the given LVS request."""
        if self.checker is None:
            msg = '*Error* Checker is not initialized.'
            self.handler.send_obj(dict(type='error', data=msg))
            return

        try:
            lib_name = req['lib_name']
            cell_name = req['cell_name']
            lay_view = req['lay_view']
            sch_view = req['sch_view']
            rcx_params = req['rcx_params']
        except KeyError:
            msg = '*Error* malformed request: %s' % str(req)
            self.handler.send_obj(dict(type='error', data=msg))
            return

        with bag.io.open_temp(prefix='rcxLog', delete=False, dir=self.dtmp) as logfile:
            log_fname = logfile.name

        try:
            netlist = self.checker.run_rcx(lib_name, cell_name, lay_view, sch_view, log_fname, rcx_params)
        except Exception:
            stack_trace = traceback.format_exc()
            msg = '*Error* error running RCX:\n%s' % stack_trace
            self.handler.send_obj(dict(type='error', data=msg))
            return

        if netlist is None:
            self.handler.send_obj(dict(type='tuple', data=(False, log_fname)))
        else:
            # delete old calibre view
            cmd = 'delete_cellview( "%s" "%s" "%s" )' % (lib_name, cell_name, self.calview_name)
            self.send_skill(cmd)
            self.recv_skill()

            # create calibre view.
            content = Template(calibre_tmp).render(netlist_file=netlist,
                                                   lib_name=lib_name,
                                                   cell_name=cell_name,
                                                   calibre_cellmap=self.calview_cell_map,
                                                   view_name=self.calview_name)

            with bag.io.open_temp(prefix='calibre', delete=False, dir=self.dtmp) as file_obj:
                setup_file = file_obj.name
                file_obj.write(content)

            cmd = 'mgc_rve_load_setup_file( "%s" )' % setup_file
            self.send_skill(cmd)
            self.recv_skill()
            self.handler.send_obj(dict(type='tuple', data=(True, log_fname)))
