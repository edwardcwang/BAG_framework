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
from typing import Optional, Dict, Any, Set, Tuple
from future.utils import with_metaclass

from bag.util.interval import IntervalSet

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance

from ..analog_mos.core import MOSTech
from .base import LaygoPrimBase


class LaygoBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

        tech_params = self.grid.tech_info.tech_params
        self._tech_cls = tech_params['layout']['mos_tech_class']  # type: MOSTech

        # error checking
        for key in ('config',):
            if key not in params:
                raise ValueError('All subclasses of DigitalBase must have %s parameter.' % key)

        # update routing grid
        self._config = params['config']
        tdir = 'y'
        for lay, w, sp in zip(self._config['tr_layers'], self._config['tr_widths'], self._config['tr_spaces']):
            self.grid.add_new_layer(lay, sp, w, tdir, override=True, unit_mode=True)
            tdir = 'x' if tdir == 'y' else 'y'

        self.grid.update_block_pitch()

        # initialize attributes
        self._draw_boundaries = False
        self._num_rows = 0
        self._laygo_size = None
        self._row_types = None
        self._row_orientations = None
        self._row_infos = None
        self._ext_heights = None
        self._sd_xc = 0
        self._y_offset = 0
        self._col_width = 0
        self._used_list = None

    def set_row_types(self, row_types, row_orientations, draw_boundaries, end_mode):
        lch = self._config['lch']
        w_sub = self._config['w_sub']
        w_nominal = self._config['w_nominal']
        min_ng_tracks = self._config['min_ng_tracks']
        min_nds_tracks = self._config['min_nds_tracks']
        min_pg_tracks = self._config['min_pg_tracks']
        min_pds_tracks = self._config['min_pds_tracks']
        min_sub_tracks = self._config['min_sub_tracks']
        guard_ring_nf = self._config['guard_ring_nf']

        lch_unit = int(round(lch / self.grid.layout_unit / self.grid.resolution))
        top_layer = self._config['tr_layers'][-1]

        # get layout information for all rows
        self._draw_boundaries = draw_boundaries
        self._num_rows = len(row_types)
        self._used_list = [IntervalSet() for _ in range(self._num_rows)]
        self._row_types = []
        self._row_infos = []
        self._row_orientations = []
        self._col_width = self._tech_cls.get_sd_pitch(lch_unit) * 2

        # add bottom boundary
        bot_end = (end_mode & 1) != 0
        top_end = (end_mode & 2) != 0
        left_end = (end_mode & 4) != 0
        if draw_boundaries:
            if row_types[0] == 'nch' or row_types[0] == 'ptap':
                row_type = 'nend'
                row_info = self._tech_cls.get_laygo_end_info(self.grid, lch_unit, 'nch', 'standard', top_layer, bot_end)
            else:
                row_type = 'pend'
                row_info = self._tech_cls.get_laygo_end_info(self.grid, lch_unit, 'pch', 'standard', top_layer, bot_end)
            self._row_types.append(row_type)
            self._row_orientations.append('R0')
            self._row_infos.append(row_info)
            self._sd_xc = self._tech_cls.get_left_sd_xc(self.grid, lch_unit, guard_ring_nf, top_layer, left_end)
            self._y_offset = row_info['height']

        for (row_type, row_orient) in zip(row_types, row_orientations):
            if row_type == 'nch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_nominal, 'nch',
                                                             min_ng_tracks, min_nds_tracks)
            elif row_type == 'pch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_nominal, 'pch',
                                                             min_pg_tracks, min_pds_tracks)
            elif row_type == 'ptap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ptap',
                                                             {}, min_sub_tracks)
            elif row_type == 'ntap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ntap',
                                                             {}, min_sub_tracks)
            else:
                raise ValueError('Unknown row type: %s' % row_type)
            self._row_types.append(row_type)
            self._row_orientations.append(row_orient)
            self._row_infos.append(row_info)

        # add top boundary
        if draw_boundaries:
            if row_types[-1] == 'nch' or row_types[-1] == 'ptap':
                row_type = 'nend'
                row_info = self._tech_cls.get_laygo_end_info(self.grid, lch_unit, 'nch', 'standard', top_layer, top_end)
            else:
                row_type = 'pend'
                row_info = self._tech_cls.get_laygo_end_info(self.grid, lch_unit, 'pch', 'standard', top_layer, top_end)
            self._row_types.append(row_type)
            self._row_orientations.append('MX')
            self._row_infos.append(row_info)

        # calculate extension widths
        num_rows = len(self._row_types)
        self._ext_heights = []
        for idx in range(num_rows - 1):
            info_bot, info_top = self._row_infos[idx], self._row_infos[idx + 1]
            ori_bot, ori_top = self._row_orientations[idx], self._row_orientations[idx + 1]

            ext_h = 0
            if ori_bot == 'R0':
                ext_bot = info_bot['ext_top_info']
                ext_h += info_bot['ext_top_h']
            else:
                ext_bot = info_bot['ext_bot_info']
                ext_h += info_bot['ext_bot_h']

            if ori_top == 'R0':
                ext_top = info_top['ext_bot_info']
                ext_h += info_top['ext_bot_h']
            else:
                ext_top = info_top['ext_top_info']
                ext_h += info_top['ext_top_h']

            # check that we can draw the extension
            valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_top, ext_bot)
            if ext_h < valid_widths[-1] and ext_h not in valid_widths:
                raise ValueError('Cannot draw extension with height = %d' % ext_h)

            self._ext_heights.append(ext_h)

    def _get_row_index(self, yo):
        return yo + 1 if self._draw_boundaries else yo

    def set_laygo_size(self, num_col):
        self._laygo_size = self._num_rows, num_col

    def add_laygo_instance(self, master, inst_name=None, loc=(0, 0), orient='R0', nx=1, spx=0):
        # type: (LaygoBase, Optional[str], Tuple[int, int], str, int, int) -> Instance

        # error checking
        if loc[1] >= self._num_rows:
            raise ValueError('row_index = %d >= %d' % (loc[1], self._num_rows))

        # convert location to resolution units
        x0 = self._sd_xc + loc[0] * self._col_width
        y0 = self._y_offset
        for idx in range(loc[1] + 1):
            y0 += self._row_infos[idx]['height']

        # convert horizontal pitch to resolution units
        spx *= self._col_width

        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def add_laygo_primitive(self, blk_type, w=None, threshold=None, loc=(0, 0), nx=1, spx=0):
        # type: (str, Optional[int], Optional[str], Tuple[int, int], int, int) -> Instance

        # get mos type
        ridx = self._get_row_index(loc[1])
        mos_type = self._row_types[ridx]
        orient = self._row_orientations[ridx]
        if mos_type == 'nch' or mos_type == 'pch':
            if w is None:
                w = self._config['w_nominal']
            if threshold is None:
                threshold = self._config['thres_nominal']
        else:
            if w is None:
                w = self._config['w_sub']
            if threshold is None:
                threshold = self._config['thres_sub']

        # make master
        params = dict(
            lch=self._config['lch'],
            w=w,
            mos_type=mos_type,
            threshold=threshold,
            blk_type=blk_type,
        )
        master = self.new_template(params=params, temp_cls=LaygoPrimBase)

        inst_name = 'XR%dC%d' % (loc[1], loc[0])
        return self.add_laygo_instance(master, inst_name=inst_name, loc=loc, orient=orient,
                                       nx=nx, spx=spx)

    def finalize(self, flatten=False):
        """set laygo_size, fill empty spaces, draw extensions, then draw boundaries before finalizing layout."""

        # set laygo_size
        if self._laygo_size is None:
            ncol = 0
            for intv in self._used_list:
                ncol = max(ncol, intv.get_end())
            self.set_laygo_size(ncol)

        # fill empty spaces
        pass

        # draw extensions
        pass

        # draw boundaries
        if self._draw_boundaries:
            pass

        super(LaygoBase, self).finalize(flatten=flatten)
