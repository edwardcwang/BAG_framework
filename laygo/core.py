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
from typing import Optional, Dict, Any, Set, Tuple, List
from future.utils import with_metaclass

from bag.util.interval import IntervalSet

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance

from ..analog_mos.core import MOSTech
from ..analog_mos.mos import AnalogMOSExt
from .base import LaygoPrimitive, LaygoSubstrate, LaygoEndRow


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
        self._row_thresholds = None
        self._row_infos = None
        self._ext_params = None
        self._left_margin = 0
        self._right_margin = 0
        self._col_width = 0
        self._used_list = None  # type: List[IntervalSet]
        self._top_layer = None
        self._bot_end_master = None
        self._top_end_master = None
        self._end_mode = None
        self._add_bot_sub = False
        self._add_top_sub = False

    @property
    def laygo_size(self):
        return self._laygo_size

    def set_row_types(self, row_types, row_orientations, row_thresholds, draw_boundaries, end_mode, top_layer=None):
        lch = self._config['lch']
        w_sub = self._config['w_sub']
        w_nominal = self._config['w_nominal']
        ng_tracks = self._config['ng_tracks']
        nds_tracks = self._config['nds_tracks']
        pg_tracks = self._config['pg_tracks']
        pds_tracks = self._config['pds_tracks']
        sub_tracks = self._config['sub_tracks']
        min_sub_tracks = self._config['min_sub_tracks']
        min_n_tracks = self._config['min_n_tracks']
        min_p_tracks = self._config['min_p_tracks']

        ng_tracks = max(ng_tracks, 1)
        pg_tracks = max(pg_tracks, 1)

        guard_ring_nf = self._config['guard_ring_nf']

        lch_unit = int(round(lch / self.grid.layout_unit / self.grid.resolution))
        self._top_layer = self._config['tr_layers'][-1] if top_layer is None else top_layer

        # get layout information for all rows
        self._draw_boundaries = draw_boundaries
        self._end_mode = end_mode
        self._num_rows = len(row_types)
        self._row_types = list(row_types)
        self._row_infos = []
        self._row_orientations = list(row_orientations)
        self._row_thresholds = list(row_thresholds)
        self._col_width = self._tech_cls.get_sd_pitch(lch_unit) * 2

        bot_end = (end_mode & 1) != 0
        top_end = (end_mode & 2) != 0
        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0

        self._left_margin = self._tech_cls.get_left_sd_xc(self.grid, lch_unit, guard_ring_nf,
                                                          self._top_layer, left_end)
        self._right_margin = self._tech_cls.get_left_sd_xc(self.grid, lch_unit, guard_ring_nf,
                                                           self._top_layer, right_end)

        if draw_boundaries:
            # create boundary masters
            if self._tech_cls.get_laygo_end_with_substrate():
                # insert substrate rows if necessary
                mos_type = self._row_types[0]
                if mos_type != 'ptap' and mos_type != 'ntap':
                    self._add_bot_sub = True
                    sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
                    self._row_types.insert(0, sub_type)
                    self._row_orientations.insert(0, 'R0')
                    self._row_thresholds.insert(0, self._row_thresholds[0])
                mos_type = self._row_types[-1]
                if mos_type != 'ptap' and mos_type != 'ntap':
                    self._add_top_sub = True
                    sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
                    self._row_types.append(sub_type)
                    self._row_orientations.append('MX')
                    self._row_thresholds.append(self._row_thresholds[-1])
            # create boundary masters
            params = dict(
                lch=lch,
                mos_type=self._row_types[0],
                threshold=self._row_thresholds[0],
                is_end=bot_end,
                top_layer=self._top_layer,
            )
            self._bot_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
            params = dict(
                lch=lch,
                mos_type=self._row_types[-1],
                threshold=self._row_thresholds[-1],
                is_end=top_end,
                top_layer=self._top_layer,
            )
            self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)

        # get row types
        self._used_list = []
        for idx, (row_type, row_orient, row_thres) in enumerate(zip(self._row_types, self._row_orientations,
                                                                    self._row_thresholds)):
            if idx == 0:
                end_mode = 1 if row_orient == 'R0' else 2
            elif idx == len(self._row_types) - 1:
                end_mode = 2 if row_orient == 'R0' else 1
            else:
                end_mode = 0

            if row_type == 'nch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_nominal, 'nch',
                                                             ng_tracks, nds_tracks, min_n_tracks, end_mode)
            elif row_type == 'pch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_nominal, 'pch',
                                                             pg_tracks, pds_tracks, min_p_tracks, end_mode)
            elif row_type == 'ptap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ptap',
                                                             0, sub_tracks, min_sub_tracks, end_mode)
            elif row_type == 'ntap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ntap',
                                                             0, sub_tracks, min_sub_tracks, end_mode)
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            self._row_infos.append(row_info)
            self._used_list.append(IntervalSet())

        # calculate extension widths
        self._ext_params = []
        for idx in range(len(self._row_types) - 1):
            bot_mtype, top_mtype = self._row_types[idx], self._row_types[idx + 1]
            bot_thres, top_thres = self._row_thresholds[idx], self._row_thresholds[idx + 1]
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

            ext_params = dict(
                lch=lch,
                w=ext_h,
                bot_mtype=bot_mtype,
                top_mtype=top_mtype,
                bot_thres=bot_thres,
                top_thres=top_thres,
                top_ext_info=ext_top,
                bot_ext_info=ext_bot,
                is_laygo=True,
            )

            self._ext_params.append((ext_h, ext_params))

    def _get_row_index(self, yo):
        return yo + 1 if self._add_bot_sub else yo

    def set_laygo_size(self, num_col):
        bot_inst = top_inst = None
        if self._laygo_size is None:
            self._laygo_size = num_col, self._num_rows
            width = self._col_width * num_col
            height = sum((info['height'] for info in self._row_infos))
            if self._draw_boundaries:
                width += self._left_margin + self._right_margin
                height += self._bot_end_master.bound_box.height_unit
                height += self._top_end_master.bound_box.height_unit
            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(self._top_layer, bound_box)
            self.add_cell_boundary(bound_box)

            if self._add_bot_sub:
                bot_inst = self._add_laygo_primitive_real('sub', loc=(0, 0), nx=num_col, spx=1)
            if self._add_top_sub:
                top_inst = self._add_laygo_primitive_real('sub', loc=(0, len(self._row_types) - 1), nx=num_col, spx=1)

        return bot_inst, top_inst

    def add_laygo_instance(self, master, inst_name=None, loc=(0, 0), orient='R0', nx=1, spx=0):
        # type: (LaygoBase, Optional[str], Tuple[int, int], str, int, int) -> Instance

        if loc[1] >= self._num_rows:
            raise ValueError('row_index = %d >= %d' % (loc[1], self._num_rows))
        if nx < 1:
            raise ValueError('Must have nx >= 1.')

        col_idx, row_idx = loc
        row_idx = self._get_row_index(row_idx)

        # mark region as used
        inst_ncol, inst_nrow = master.laygo_size
        if orient == 'R0':
            row_start, row_end = row_idx, row_idx + inst_nrow
            col_start, col_end = col_idx, col_idx + inst_ncol
        elif orient == 'MX':
            row_start, row_end = row_idx - inst_nrow, row_idx
            col_start, col_end = col_idx, col_idx + inst_ncol
        elif orient == 'MY':
            row_start, row_end = row_idx, row_idx + inst_nrow
            col_start, col_end = col_idx - inst_ncol, col_idx
        elif orient == 'R180':
            row_start, row_end = row_idx - inst_nrow, row_idx
            col_start, col_end = col_idx - inst_ncol, col_idx
        else:
            raise ValueError('Unknown orientation: %s' % orient)

        if row_end > len(self._row_types) or row_start < 0:
            raise ValueError('Cannot add row interval: [%d, %d)' % (row_start, row_end))

        for row in range(row_start, row_end):
            intv = self._used_list[row]
            for inst_num in range(nx):
                inst_intv = col_start + spx * inst_num, col_end + spx * inst_num
                if not intv.add(inst_intv):
                    raise ValueError('Cannot add instance on row %d, '
                                     'column [%d, %d).' % (row, inst_intv[0], inst_intv[1]))

        # convert location to resolution units
        x0 = self._left_margin + col_idx * self._col_width
        y0 = 0 if self._bot_end_master is None else self._bot_end_master.bound_box.height_unit
        for idx in range(row_idx):
            y0 += self._row_infos[idx]['height']
        y0 += self._row_infos[row_idx]['yblk']

        # convert horizontal pitch to resolution units
        spx *= self._col_width

        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def _add_laygo_primitive_real(self, blk_type, w=None, loc=(0, 0), nx=1, spx=0):
        # type: (str, Optional[int], Tuple[int, int], int, int) -> Instance

        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= len(self._row_types):
            raise ValueError('Cannot add primitive at row %d' % row_idx)

        mos_type = self._row_types[row_idx]
        orient = self._row_orientations[row_idx]
        threshold = self._row_thresholds[row_idx]
        if mos_type == 'nch' or mos_type == 'pch':
            if w is None:
                w = self._config['w_nominal']
        else:
            if w is None:
                w = self._config['w_sub']

        # make master
        params = dict(
            lch=self._config['lch'],
            w=w,
            mos_type=mos_type,
            threshold=threshold,
        )
        if blk_type == 'sub':
            if row_idx == 0:
                end_mode = 1 if orient == 'R0' else 2
            elif row_idx == len(self._row_types) - 1:
                end_mode = 2 if orient == 'R0' else 1
            else:
                end_mode = 0
            params['end_mode'] = end_mode
            master = self.new_template(params=params, temp_cls=LaygoSubstrate)
        else:
            params['blk_type'] = blk_type
            master = self.new_template(params=params, temp_cls=LaygoPrimitive)

        intv = self._used_list[row_idx]
        for inst_num in range(nx):
            inst_intv = col_idx + spx * inst_num, col_idx + 1 + spx * inst_num
            if not intv.add(inst_intv):
                raise ValueError('Cannot add primitive on row %d, '
                                 'column [%d, %d).' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = self._left_margin + col_idx * self._col_width
        y0 = 0 if self._bot_end_master is None else self._bot_end_master.bound_box.height_unit
        for idx in range(row_idx):
            y0 += self._row_infos[idx]['height']
        if orient == 'R0':
            y0 += self._row_infos[row_idx]['yblk']
        if orient == 'MX':
            y0 += self._row_infos[row_idx]['height'] - self._row_infos[row_idx]['yblk']

        # convert horizontal pitch to resolution units
        spx *= self._col_width

        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def add_laygo_primitive(self, blk_type, w=None, loc=(0, 0), nx=1, spx=0):
        # type: (str, Optional[int], Tuple[int, int], int, int) -> Instance

        loc = (loc[0], self._get_row_index(loc[1]))
        return self._add_laygo_primitive_real(blk_type, w=w, loc=loc, nx=nx, spx=spx)

    def finalize(self, flatten=False):
        """set laygo_size, fill empty spaces, draw extensions, then draw boundaries before finalizing layout."""

        # set laygo_size
        if self._laygo_size is None:
            ncol = 0
            for intv in self._used_list:
                if intv:
                    ncol = max(ncol, intv.get_end())
            self.set_laygo_size(ncol)

        # fill empty spaces
        pass

        nx = self._laygo_size[0]
        spx = self._col_width
        # draw extensions
        yprev = 0 if self._bot_end_master is None else self._bot_end_master.bound_box.height_unit
        for idx, (bot_info, (ext_h, ext_params)) in enumerate(zip(self._row_infos, self._ext_params)):
            if ext_h > 0 or self._tech_cls.draw_zero_extension():
                ycur = yprev + bot_info['yblk'] + bot_info['blk_height']
                ext_master = self.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                self.add_instance(ext_master, inst_name='XEXT%d' % idx, loc=(self._left_margin, ycur),
                                  nx=nx, spx=spx, unit_mode=True)
            yprev += bot_info['height']

        # draw boundaries
        if self._draw_boundaries:
            # draw top and bottom end row
            self.add_instance(self._bot_end_master, inst_name='XRBOT', loc=(self._left_margin, 0),
                              nx=nx, spx=spx, unit_mode=True)
            self.add_instance(self._top_end_master, inst_name='XRBOT',
                              loc=(self._left_margin, self.bound_box.height_unit),
                              orient='MX', nx=nx, spx=spx, unit_mode=True)

        super(LaygoBase, self).finalize(flatten=flatten)
