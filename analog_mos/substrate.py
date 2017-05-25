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


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Any, Set

from bag import float_to_si_string
from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB

from .core import MOSTech


class AnalogSubstrateCore(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogSubstrateCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_bound_box = None

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
            dummy_only='True if only dummy connections will be made to this substrate.',
            port_tracks='Substrate port must contain these track indices.',
            dum_tracks='Dummy port must contain these track indices.',
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        port_tracks = tuple(int(2 * v) for v in self.params['port_tracks'])
        dum_tracks = tuple(int(2 * v) for v in self.params['dum_tracks'])
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info'], port_tracks, dum_tracks,
                                     self.params['flip_parity']))

    def draw_layout(self):
        dummy_only = self.params['dummy_only']
        port_tracks = self.params['port_tracks']
        dum_tracks = self.params['dum_tracks']
        layout_info = self.params['layout_info']

        # draw substrate
        self._tech_cls.draw_mos(self, layout_info)
        # draw substrate connections
        self._tech_cls.draw_substrate_connection(self, layout_info, port_tracks, dum_tracks, dummy_only)
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()


class AnalogSubstrate(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogSubstrate, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self._layout_info = None
        self.prim_bound_box = None
        self._ext_top_info = None
        self._ext_bot_info = None
        self._sd_yc = None
        self._well_box = None

    def get_well_box(self):
        return self._well_box

    def get_ext_top_info(self):
        return self._ext_top_info

    def get_ext_bot_info(self):
        return self._ext_bot_info

    def get_sd_yc(self):
        return self._sd_yc

    def get_edge_layout_info(self):
        return self._layout_info

    @classmethod
    def get_default_param_values(cls):
        return dict(
            end_mode=1,
            port_tracks=[],
            dum_tracks=[],
            dummy_only=False,
            is_passive=False,
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
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            fg='number of fingers.',
            end_mode='An integer indicating whether top/bottom of this template is at the ends.',
            dummy_only='True if only dummy connections will be made to this substrate.',
            port_tracks='Substrate port must contain these track indices.',
            dum_tracks='Dummy port must contain these track indices.',
            is_passive='True if this substrate is used as substrate contact for passive devices.',
        )

    def get_layout_basename(self):
        fmt = '%s_l%s_w%s_%s_fg%d_end%d'
        sub_type = self.params['sub_type']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        fg = self.params['fg']
        end_mode = self.params['end_mode']
        basename = fmt % (sub_type, lstr, wstr, th, fg, end_mode)
        if self.params['dummy_only']:
            basename += '_dum'

        return basename

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        # unique key is indexed by half track id.
        port_tracks = tuple(int(2 * v) for v in self.params['port_tracks'])
        dum_tracks = tuple(int(2 * v) for v in self.params['dum_tracks'])
        return self.to_immutable_id((basename, port_tracks, dum_tracks, self.params['flip_parity']))

    def draw_layout(self):
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, w, sub_type, threshold, fg, end_mode,
                            dummy_only, port_tracks, dum_tracks, flip_parity, is_passive):
        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        info = self._tech_cls.get_substrate_info(self.grid, lch_unit, w, sub_type, threshold, fg, end_mode, is_passive)
        self._layout_info = info['layout_info']
        self._sd_yc = info['sd_yc']
        self._ext_top_info = info['ext_top_info']
        self._ext_bot_info = info['ext_bot_info']
        well_xl, well_yb, well_xr, well_yt = info['well_bounds']
        self._well_box = BBox(well_xl, well_yb, well_xr, well_yt, res, unit_mode=True)

        core_params = dict(
            dummy_only=dummy_only,
            port_tracks=port_tracks,
            dum_tracks=dum_tracks,
            layout_name=self.get_layout_basename() + '_core',
            layout_info=self._layout_info,
        )

        master = self.new_template(params=core_params, temp_cls=AnalogSubstrateCore)
        inst = self.add_instance(master, 'XCORE')
        self.array_box = master.array_box
        self.prim_bound_box = master.prim_bound_box

        for port_name in inst.port_names_iter():
            self.reexport(inst.get_port(port_name), show=False)

        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
