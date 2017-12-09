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

"""This package provides all IO related functionalities for BAG.

Most importantly, this module sorts out all the bytes v.s. unicode differences
and simplifies writing python2/3 compatible code.
"""

from .common import fix_string, to_bytes, set_encoding, get_encoding, \
    set_error_policy, get_error_policy
from .sim_data import load_sim_results, save_sim_results, load_sim_file
from .file import read_file, read_resource, read_yaml, readlines_iter, \
    write_file, make_temp_dir, open_temp, open_file

from . import process

__all__ = ['fix_string', 'to_bytes', 'set_encoding', 'get_encoding',
           'set_error_policy', 'get_error_policy',
           'load_sim_results', 'save_sim_results', 'load_sim_file',
           'read_file', 'read_resource', 'read_yaml', 'readlines_iter',
           'write_file', 'make_temp_dir', 'open_temp', 'open_file',
           ]
