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

import abc
from typing import Dict, Any, Set, List
from future.utils import with_metaclass

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB

from ..laygo.base import LaygoEndRow, LaygoSubstrate
from ..laygo.core import LaygoBaseInfo, LaygoIntvSet


class DigitalBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(DigitalBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._laygo_info = LaygoBaseInfo(self.grid, self.params['config'])
        self.grid = self._laygo_info.grid

        # initialize attributes
        self._num_rows = 0
        self._dig_size = None
        self._row_info = None
        self._used_list = None  # type: List[LaygoIntvSet]
        self._bot_end_master = None
        self._top_end_master = None
        self._bot_sub_master = None
        self._top_sub_master = None
        self._ybot = None
        self._ytop = None

    def initialize(self, row_info, num_rows, draw_boundaries, end_mode, guard_ring_nf=0):
        self._row_info = row_info

        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode

        tot_height = row_info['row_height'] * num_rows
        if draw_boundaries:
            lch = self._laygo_info.lch
            top_layer = self._laygo_info.top_layer
            mos_pitch = self._laygo_info.mos_pitch
            w_sub = self._laygo_info['w_sub']
            bot_end = (end_mode & 1) != 0
            top_end = (end_mode & 2) != 0

            # create end row and substrate masters
            mtype = row_info['row_types'][0]
            thres = row_info['row_thresholds'][0]
            sub_type = 'ptap' if mtype == 'nch' else 'ntap'
            params = dict(
                lch=lch,
                mos_type=mtype,
                threshold=thres,
                is_end=bot_end,
                top_layer=top_layer,
            )
            self._bot_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
            params = dict(
                lch=lch,
                w=w_sub,
                mos_type=sub_type,
                threshold=thres,
                options={},
            )
            self._bot_sub_master = self.new_template(params=params, temp_cls=LaygoSubstrate)
            bot_extw = row_info['bot_extw']
            bot_sub_extw = row_info['bot_sub_extw']

            if num_rows % 2 == 0:
                # because of mirroring, top and bottom masters are the same, except for is_end parameter.
                params = dict(
                    lch=lch,
                    mos_type=mtype,
                    threshold=thres,
                    is_end=top_end,
                    top_layer=top_layer,
                )
                self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
                self._top_sub_master = self._bot_sub_master
                top_extw = bot_extw
                top_sub_extw = bot_sub_extw
            else:
                mtype = row_info['row_types'][-1]
                thres = row_info['row_thresholds'][-1]
                sub_type = 'ptap' if mtype == 'nch' else 'ntap'
                params = dict(
                    lch=lch,
                    mos_type=mtype,
                    threshold=thres,
                    is_end=top_end,
                    top_layer=top_layer,
                )
                self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
                params = dict(
                    lch=lch,
                    w=w_sub,
                    mos_type=sub_type,
                    threshold=thres,
                    options={},
                )
                self._top_sub_master = self.new_template(params=params, temp_cls=LaygoSubstrate)
                top_extw = row_info['top_extw']
                top_sub_extw = row_info['top_sub_extw']

            y0 = self._bot_end_master.bound_box.height_unit
            y1 = y0 + self._bot_sub_master.bound_box.height_unit
            y2 = y1 + bot_sub_extw * mos_pitch
            self._ybot = (y0, y1, y2)
            y0 = y2 + tot_height - top_extw * mos_pitch
            y1 = y0 + (top_extw + top_sub_extw) * mos_pitch + self._top_sub_master.bound_box.height_unit
            y2 = y1 + self._top_end_master.bound_box.height_unit
            self._ytop = (y0, y1, y2)
        else:
            self._ybot = (0, 0, 0)
            self._ytop = (tot_height, tot_height, tot_height)

    def set_digital_size(self, num_col=None):
        if self._dig_size is None:
            if num_col is None:
                num_col = 0
                for intv in self._used_list:
                    num_col = max(num_col, intv.get_end())

            self._dig_size = num_col, self._num_rows

            top_layer = self._laygo_info.top_layer
            draw_boundaries = self._laygo_info.draw_boundaries
            col_width = self._laygo_info.col_width
            left_margin = self._laygo_info.left_margin
            right_margin = self._laygo_info.right_margin

            width = col_width * num_col
            height = self._ytop[2]
            if draw_boundaries:
                width += left_margin + right_margin

            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            self.add_cell_boundary(bound_box)

    def add_block(self):
        pass
