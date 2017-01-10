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

"""This module defines template used for transistor characterization."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from .analog_core import AnalogBase


class Transistor(AnalogBase):
    """A template of a single transistor with dummies.

    This class is mainly used for transistor characterization or
    design exploration with config views.

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
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            mos_type="transistor type, either 'pch' or 'nch'.",
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            fg_dum='number of dummies on each side.',
            threshold='transistor threshold flavor.',
            track_width='horizontal track width, in meters.',
            track_space='horizontal track spacing, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            vm_layer='vertical routing metal layer name.',
            hm_layer='horizontal routing metal layer name.',
            num_track_sep='number of tracks reserved as space between ports.',
            min_ds_cap='True to minimize parasitic Cds.',
        )

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
            min_ds_cap=False,
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        mos_type = self.params['mos_type']
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        fg_dum = self.params['fg_dum']
        threshold = self.params['threshold']
        track_width = self.params['track_width']
        track_space = self.params['track_space']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        vm_layer = self.params['vm_layer']
        hm_layer = self.params['hm_layer']
        num_track_sep = self.params['num_track_sep']

        fg_tot = fg + 2 * fg_dum

        nw_list = []
        nth_list = []
        pw_list = []
        pth_list = []
        ng_tracks = []
        nds_tracks = []
        pg_tracks = []
        pds_tracks = []
        num_gate_tr = 2 + num_track_sep
        if mos_type == 'nch':
            nw_list.append(w)
            nth_list.append(threshold)
            ng_tracks.append(num_gate_tr)
            nds_tracks.append(1)
        else:
            pw_list.append(w)
            pth_list.append(threshold)
            pg_tracks.append(num_gate_tr)
            pds_tracks.append(1)

        sub_lay, bot_box_arr, top_box_arr = self.draw_base(lch, fg_tot, ptap_w, ntap_w,
                                                           nw_list, nth_list, pw_list, pth_list,
                                                           track_width, track_space, num_track_sep,
                                                           vm_layer, hm_layer,
                                                           ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                                                           pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                                                           )

        # export body
        self.add_pin('b', sub_lay, bot_box_arr, show=True)
        self.add_pin('b', sub_lay, top_box_arr, show=True)

        mos_ports = self.draw_mos_conn(0, fg_dum, fg, 0, 2, min_ds_cap=self.params['min_ds_cap'])
        tr_lay, tr_box = self.connect_to_track([mos_ports['g']], 0, 'g', num_gate_tr - 1)
        self.add_pin('g', tr_lay, tr_box, show=True)

        tr_lay, tr_box = self.connect_to_track([mos_ports['d']], 0, 'ds', 0)
        self.add_pin('d', tr_lay, tr_box, show=True)

        tr_lay, tr_box = self.connect_to_track([mos_ports['s']], 0, 'g', 0)
        self.add_pin('s', tr_lay, tr_box, show=True)

        self.fill_dummy()
