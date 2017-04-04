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


"""This package defines various passives template classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Set, Any

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB

from ..analog_core import SubstrateContact


class MOMCap(TemplateBase):
    """differential bias resistor for differential high pass filter.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(MOMCap, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Dict[str, Any]
            dictionary of default parameter values.
        """
        return dict(
            show_pins=False,
        )

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            cap_bot_layer='MOM cap bottom layer.',
            cap_top_layer='MOM cap top layer.',
            cap_width='MOM cap width.',
            cap_height='MOM cap height.',
            sub_lch='channel length, in meters.',
            sub_w='substrate width, in meters/number of fins.',
            sub_type='substrate type.',
            threshold='substrate threshold flavor.',
            show_pins='True to show pin labels.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, cap_bot_layer, cap_top_layer, cap_width, cap_height,
                            sub_lch, sub_w, sub_type, threshold, show_pins):
        res = self.grid.resolution
        cap_width = int(round(cap_width / res))
        cap_height = int(round(cap_height / res))

        blk_w, blk_h = self.grid.get_block_size(cap_top_layer, unit_mode=True)
        nx_blk = -(-cap_width // blk_w)

        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=threshold,
            top_layer=cap_top_layer,
            blk_width=nx_blk,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        inst = self.add_instance(sub_master, inst_name='XSUB')
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(inst.get_port(port_name), show=show_pins)

        subw, subh = self.grid.get_size_dimension(sub_master.size, unit_mode=True)
        ny_blk = -(-(subh + cap_height) // blk_h)
        self.size = cap_top_layer, nx_blk, ny_blk
        self.array_box = BBox(0, 0, nx_blk * blk_w, ny_blk * blk_h, res, unit_mode=True)

        cap_xl = self.array_box.xc_unit - cap_width // 2
        cap_box = BBox(cap_xl, subh, cap_xl + cap_width, subh + cap_height, res, unit_mode=True)
        cap_ports = self.add_mom_cap(cap_box, cap_bot_layer, cap_top_layer - cap_bot_layer + 1, 2)
        cp, cn = cap_ports[cap_top_layer]
        self.add_pin('plus', cp, show=show_pins)
        self.add_pin('minus', cn, show=show_pins)
