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
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()

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
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info']))

    def draw_layout(self):
        layout_info = self.params['layout_info']

        # draw substrate
        self._tech_cls.draw_mos(self, layout_info)


class AnalogSubstrate(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogSubstrate, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
        self._layout_info = None
        self._ext_top_info = None
        self._ext_bot_info = None
        self._sd_yc = None

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
            is_passive='True if this substrate is used as substrate contact for passive devices.',
            top_layer='The top routing layer.  Used to determine vertical pitch.',
        )

    def get_layout_basename(self):
        fmt = '%s_l%s_w%s_%s_lay%d'
        sub_type = self.params['sub_type']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        top_layer = self.params['top_layer']
        if top_layer is None:
            top_layer = 0
        basename = fmt % (sub_type, lstr, wstr, th, top_layer)
        if self.params['is_passive']:
            basename += '_passive'

        return basename

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return basename

    def draw_layout(self):
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, w, sub_type, threshold, is_passive, top_layer, **kwargs):
        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        if top_layer is not None:
            blk_pitch = self.grid.get_block_size(top_layer, unit_mode=True)[1]
        else:
            blk_pitch = 1
        fg = self._tech_cls.get_analog_unit_fg()
        info = self._tech_cls.get_substrate_info(lch_unit, w, sub_type, threshold, fg,
                                                 blk_pitch=blk_pitch, is_passive=is_passive)
        self._layout_info = info['layout_info']
        self._sd_yc = info['sd_yc']
        self._ext_top_info = info['ext_top_info']
        self._ext_bot_info = info['ext_bot_info']

        core_params = dict(
            layout_name=self.get_layout_basename() + '_core',
            layout_info=self._layout_info,
        )

        master = self.new_template(params=core_params, temp_cls=AnalogSubstrateCore)
        inst = self.add_instance(master, 'XCORE')
        self.array_box = master.array_box
        self.prim_bound_box = master.prim_bound_box

        for port_name in inst.port_names_iter():
            self.reexport(inst.get_port(port_name), show=False)
