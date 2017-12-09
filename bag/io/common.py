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

"""This module contains some commonly used IO functions.

In particular, this module keeps track of BAG's system-wide encoding/decoding settings.
"""

# default BAG file encoding.
bag_encoding = 'utf-8'
# default codec error policy
bag_codec_error = 'replace'


def fix_string(obj):
    """Fix the given potential string object to ensure python 2/3 compatibility.

    If the given object is raw bytes, decode it into a string using
    current encoding and return it.  Otherwise, just return the given object.

    This method is useful for writing python 2/3 compatible code.

    Parameters
    ----------
    obj :
        any python object.

    Returns
    -------
    val :
        the given object, or a decoded string if the given object is bytes.
    """
    if isinstance(obj, bytes):
        obj = obj.decode(encoding=bag_encoding, errors=bag_codec_error)
    return obj


def to_bytes(my_str):
    """Convert the given string to raw bytes.

    Parameters
    ----------
    my_str : string
        the string to encode to bytes.

    Returns
    -------
    val : bytes
        raw bytes of the string.
    """
    return bytes(my_str.encode(encoding=bag_encoding, errors=bag_codec_error))


def set_encoding(new_encoding):
    """Sets the BAG input/output encoding.

    Parameters
    ----------
    new_encoding : string
        the new encoding name.
    """
    global bag_encoding
    if not isinstance(new_encoding, str):
        raise Exception('encoding name must be string/unicode.')
    bag_encoding = new_encoding


def get_encoding():
    """Returns the BAG input/output encoding.

    Returns
    -------
    bag_encoding : unicode
        the encoding name.
    """
    return bag_encoding


def set_error_policy(new_policy):
    """Sets the error policy on encoding/decoding errors.

    Parameters
    ----------
    new_policy : string
        the new error policy name.  See codecs package documentation
        for more information.
    """
    global bag_codec_error
    bag_codec_error = new_policy


def get_error_policy():
    """Returns the current BAG encoding/decoding error policy.

    Returns
    -------
    policy : unicode
        the current error policy name.
    """
    return bag_codec_error
