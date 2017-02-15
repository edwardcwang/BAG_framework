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


"""This module defines abstract analog mosfet template classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
from bag import float_to_si_string
from bag.layout.template import MicroTemplate


class AnalogResCore(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template for analog resistors array core.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        MicroTemplate.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def use_parity(cls):
        """Returns True if parity changes resistor core layout."""
        return False

    @classmethod
    @abc.abstractmethod
    def port_layer_id(cls):
        """Returns the bottom port layer ID.

        Bottom port layer must be horizontal.
        """
        return -1

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            x_tracks_min=1,
            y_tracks_min=1,
            is_high_speed=False,
            parity=0,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            x_tracks_min='Minimum number of horizontal tracks per block.',
            y_tracks_min='Minimum number of vertical tracks per block.',
            is_high_speed='True if this is high speed analog resistor.',
            parity='the parity of this resistor core.  Either 0 or 1.',
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        main = 'rescore_l%s_w%s_xmin%d_ymin%d' % (l_str, w_str, self.params['x_tracks_min'],
                                                  self.params['y_tracks_min'])
        if self.params['is_high_speed']:
            main += '_hs'
        if self.use_parity():
            main += '_par%d' % self.params['parity']

        return main

    def compute_unique_key(self):
        return self.get_layout_basename()
