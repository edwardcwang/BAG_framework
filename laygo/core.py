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
from ..analog_mos.edge import AnalogEdge
from .base import LaygoPrimitive, LaygoSubstrate, LaygoEndRow


class LaygoIntvSet(object):
    def __init__(self):
        super(LaygoIntvSet, self).__init__()
        self._intv = IntervalSet()
        self._end_flags = {}

    def add(self, intv, endl, endr):
        ans = self._intv.add(intv)
        if ans:
            start, stop = intv
            if start in self._end_flags:
                del self._end_flags[start]
            else:
                self._end_flags[start] = endl
            if stop in self._end_flags:
                del self._end_flags[stop]
            else:
                self._end_flags[stop] = endr
            return True
        else:
            return False

    def get_complement(self, total_intv):
        compl_intv = self._intv.get_complement(total_intv)
        intv_list = []
        end_list = []
        for intv in compl_intv:
            intv_list.append(intv)
            end_list.append((self._end_flags.get(intv[0], False), self._end_flags.get(intv[1], False)))
        return intv_list, end_list

    def get_end_flags(self, num_col):
        return self._end_flags[0], self._end_flags[num_col]

    def get_end(self):
        if not self._intv:
            return 0
        return self._intv.get_end()


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
        lch_unit = int(round(self._config['lch'] / self.grid.layout_unit / self.grid.resolution))
        vm_layer = self._tech_cls.get_dig_conn_layer()
        vm_space, vm_width = self._tech_cls.get_laygo_conn_track_info(lch_unit)
        self.grid.add_new_layer(vm_layer, vm_space, vm_width, 'y', override=True, unit_mode=True)
        tdir = 'x'
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
        self._row_y = None
        self._ext_params = None
        self._left_margin = 0
        self._right_margin = 0
        self._col_width = 0
        self._used_list = None  # type: List[LaygoIntvSet]
        self._top_layer = None
        self._bot_end_master = None
        self._top_end_master = None
        self._end_mode = None
        self._add_bot_sub = False
        self._add_top_sub = False
        self._guard_ring_nf = 0
        self._has_boundaries = False
        self._ext_edges = None

    @property
    def laygo_size(self):
        return self._laygo_size

    def set_row_types(self, row_types, row_orientations, row_thresholds, draw_boundaries, end_mode,
                      num_g_tracks, num_gb_tracks, num_ds_tracks, top_layer=None, guard_ring_nf=0):
        lch = self._config['lch']
        w_sub = self._config['w_sub']
        w_n = self._config['w_n']
        w_p = self._config['w_p']
        min_sub_tracks = self._config['min_sub_tracks']
        min_n_tracks = self._config['min_n_tracks']
        min_p_tracks = self._config['min_p_tracks']

        self._guard_ring_nf = guard_ring_nf

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
        num_g_tracks = list(num_g_tracks)
        num_gb_tracks = list(num_gb_tracks)
        num_ds_tracks = list(num_ds_tracks)
        self._row_y = []
        self._col_width = self._tech_cls.get_sd_pitch(lch_unit) * 2

        bot_end = (end_mode & 1) != 0
        top_end = (end_mode & 2) != 0
        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0

        self._left_margin = self._tech_cls.get_left_sd_xc(self.grid, lch_unit, self._guard_ring_nf,
                                                          self._top_layer, left_end)
        self._right_margin = self._tech_cls.get_left_sd_xc(self.grid, lch_unit, self._guard_ring_nf,
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
                    num_g_tracks.insert(0, 0)
                    num_gb_tracks.insert(0, 0)
                    num_ds_tracks.insert(0, 1)
                mos_type = self._row_types[-1]
                if mos_type != 'ptap' and mos_type != 'ntap':
                    self._add_top_sub = True
                    sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
                    self._row_types.append(sub_type)
                    self._row_orientations.append('MX')
                    self._row_thresholds.append(self._row_thresholds[-1])
                    num_g_tracks.append(0)
                    num_gb_tracks.append(0)
                    num_ds_tracks.append(1)
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
        y0 = 0 if self._bot_end_master is None else self._bot_end_master.bound_box.height_unit
        yrow = y0
        for idx, (row_type, row_orient, row_thres, ng, ngb, nds) in \
                enumerate(zip(self._row_types, self._row_orientations, self._row_thresholds,
                              num_g_tracks, num_gb_tracks, num_ds_tracks)):
            if idx == 0:
                end_mode = 1 if row_orient == 'R0' else 2
            elif idx == len(self._row_types) - 1:
                end_mode = 2 if row_orient == 'R0' else 1
            else:
                end_mode = 0

            if row_type == 'nch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_n, 'nch', row_thres,
                                                             ng, ngb, nds, min_n_tracks, end_mode)
            elif row_type == 'pch':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_p, 'pch', row_thres,
                                                             ng, ngb, nds, min_p_tracks, end_mode)
            elif row_type == 'ptap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ptap', row_thres,
                                                             0, 0, nds, min_sub_tracks, end_mode)
            elif row_type == 'ntap':
                row_info = self._tech_cls.get_laygo_row_info(self.grid, lch_unit, w_sub, 'ntap', row_thres,
                                                             0, 0, nds, min_sub_tracks, end_mode)
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            self._row_infos.append(row_info)
            self._used_list.append(LaygoIntvSet())
            self._row_y.append(yrow)
            yrow += row_info['height']

        # calculate extension widths
        self._ext_params = []
        yext = y0
        for idx in range(len(self._row_types) - 1):
            bot_mtype, top_mtype = self._row_types[idx], self._row_types[idx + 1]
            bot_thres, top_thres = self._row_thresholds[idx], self._row_thresholds[idx + 1]
            info_bot, info_top = self._row_infos[idx], self._row_infos[idx + 1]
            ori_bot, ori_top = self._row_orientations[idx], self._row_orientations[idx + 1]

            ycur = yext
            if ori_bot == 'R0':
                ext_bot = info_bot['ext_top_info']
                ext_bot_h = info_bot['ext_top_h']
                ycur += info_bot['yblk'] + info_bot['blk_height']
            else:
                ext_bot = info_bot['ext_bot_info']
                ext_bot_h = info_bot['ext_bot_h']
                ycur += info_bot['height'] - info_bot['yblk']

            if ori_top == 'R0':
                ext_top = info_top['ext_bot_info']
                ext_top_h = info_top['ext_bot_h']
            else:
                ext_top = info_top['ext_top_info']
                ext_top_h = info_top['ext_top_h']

            ext_h = ext_bot_h + ext_top_h
            print('ng: %d, ngb: %d, nds: %d' % (num_g_tracks[idx], num_gb_tracks[idx], num_ds_tracks[idx]))
            print('ext_bot_h: %d, ext_top_h: %d' % (ext_bot_h, ext_top_h))
            print('ext_bot_info: {}'.format(ext_bot))
            print('ext_top_info: {}'.format(ext_top))

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
            self._ext_params.append((ycur, ext_h, ext_params))
            yext += info_bot['height']

    def get_end_flags(self, row_idx):
        return self._used_list[row_idx].get_end_flags(self._laygo_size[0])

    def _get_row_index(self, yo):
        return yo + 1 if self._add_bot_sub else yo

    def set_laygo_size(self, num_col=None):
        bot_inst = top_inst = None
        if self._laygo_size is None:
            if num_col is None:
                num_col = 0
                for intv in self._used_list:
                    num_col = max(num_col, intv.get_end())

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

            # draw extensions and record edge parameters
            left_end = (self._end_mode & 4) != 0
            right_end = (self._end_mode & 8) != 0
            xr = self._left_margin + self._col_width * num_col + self._right_margin
            self._ext_edges = []
            for idx, (bot_info, (yext, ext_h, ext_params)) in enumerate(zip(self._row_infos, self._ext_params)):
                if ext_h > 0 or self._tech_cls.draw_zero_extension():
                    ext_master = self.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                    self.add_instance(ext_master, inst_name='XEXT%d' % idx, loc=(self._left_margin, yext),
                                      nx=num_col, spx=self._col_width, unit_mode=True)
                    if self._draw_boundaries:
                        for x, is_end, flip_lr in ((0, left_end, False), (xr, right_end, True)):
                            edge_params = dict(
                                top_layer=self._top_layer,
                                is_end=is_end,
                                guard_ring_nf=self._guard_ring_nf,
                                name_id=ext_master.get_layout_basename(),
                                layout_info=ext_master.get_edge_layout_info(),
                                is_laygo=True,
                            )
                            edge_orient = 'MY' if flip_lr else 'R0'
                            self._ext_edges.append((x, yext, edge_orient, edge_params))

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
            inst_row_idx = abs(row - row_idx)
            inst_endl, inst_endr = master.get_end_flags(inst_row_idx)
            for inst_num in range(nx):
                inst_intv = col_start + spx * inst_num, col_end + spx * inst_num
                if not intv.add(inst_intv, inst_endl, inst_endr):
                    raise ValueError('Cannot add instance on row %d, '
                                     'column [%d, %d).' % (row, inst_intv[0], inst_intv[1]))

        # convert location to resolution units
        x0 = self._left_margin + col_idx * self._col_width
        y0 = self._row_y[row_idx]

        # convert horizontal pitch to resolution units
        spx *= self._col_width

        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def _add_laygo_primitive_real(self, blk_type, loc=(0, 0), nx=1, spx=0, **kwargs):
        # type: (str, Tuple[int, int], int, int, **kwargs) -> Instance

        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= len(self._row_types):
            raise ValueError('Cannot add primitive at row %d' % row_idx)

        mos_type = self._row_types[row_idx]
        orient = self._row_orientations[row_idx]
        threshold = self._row_thresholds[row_idx]
        if mos_type == 'nch':
            w = self._config['w_n']
        elif mos_type == 'pch':
            w = self._config['w_p']
        else:
            w = self._config['w_sub']

        # make master
        params = dict(
            lch=self._config['lch'],
            w=w,
            mos_type=mos_type,
            threshold=threshold,
            options=kwargs,
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
        inst_endl, inst_endr = master.get_end_flags()
        for inst_num in range(nx):
            inst_intv = col_idx + spx * inst_num, col_idx + 1 + spx * inst_num
            if not intv.add(inst_intv, inst_endl, inst_endr):
                raise ValueError('Cannot add primitive on row %d, '
                                 'column [%d, %d).' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = self._left_margin + col_idx * self._col_width
        y0 = self._row_y[row_idx]
        if orient == 'R0':
            y0 += self._row_infos[row_idx]['yblk']
        else:
            y0 += self._row_infos[row_idx]['height'] - self._row_infos[row_idx]['yblk']

        # convert horizontal pitch to resolution units
        spx *= self._col_width

        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def add_laygo_primitive(self, blk_type, loc=(0, 0), nx=1, spx=0, **kwargs):
        # type: (str, Tuple[int, int], int, int, **kwargs) -> Instance

        loc = (loc[0], self._get_row_index(loc[1]))
        return self._add_laygo_primitive_real(blk_type, loc=loc, nx=nx, spx=spx, **kwargs)

    def fill_space(self):
        pass

    def draw_boundary_cells(self):
        if self._draw_boundaries and not self._has_boundaries:
            if self._laygo_size is None:
                raise ValueError('laygo_size must be set before drawing boundaries.')

            nx = self._laygo_size[0]
            spx = self._col_width

            # draw top and bottom end row
            self.add_instance(self._bot_end_master, inst_name='XRBOT', loc=(self._left_margin, 0),
                              nx=nx, spx=spx, unit_mode=True)
            yt = self.bound_box.height_unit
            self.add_instance(self._top_end_master, inst_name='XRBOT',
                              loc=(self._left_margin, yt),
                              orient='MX', nx=nx, spx=spx, unit_mode=True)
            # draw corners
            left_end = (self._end_mode & 4) != 0
            right_end = (self._end_mode & 8) != 0
            edge_inst_list = []
            xr = self._left_margin + self._col_width * nx + self._right_margin
            for orient, y, master in (('R0', 0, self._bot_end_master), ('MX', yt, self._top_end_master)):
                for x, is_end, flip_lr in ((0, left_end, False), (xr, right_end, True)):
                    edge_params = dict(
                        top_layer=self._top_layer,
                        is_end=is_end,
                        guard_ring_nf=self._guard_ring_nf,
                        name_id=master.get_layout_basename(),
                        layout_info=master.get_edge_layout_info(),
                        is_laygo=True,
                    )
                    edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)
                    if flip_lr:
                        eorient = 'MY' if orient == 'R0' else 'R180'
                    else:
                        eorient = orient
                    edge_inst_list.append(self.add_instance(edge_master, orient=eorient, loc=(x, y), unit_mode=True))

            # draw extension edges
            for x, y, orient, edge_params in self._ext_edges:
                edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)
                edge_inst_list.append(self.add_instance(edge_master, orient=orient, loc=(x, y), unit_mode=True))

            # draw row edges
            for ridx, (orient, yr, rinfo) in enumerate(zip(self._row_orientations, self._row_y, self._row_infos)):
                endl, endr = self.get_end_flags(ridx)
                if orient == 'R0':
                    y = yr + rinfo['yblk']
                else:
                    y = yr + rinfo['height'] - rinfo['yblk']
                for x, is_end, flip_lr, end_flag in ((0, left_end, False, endl), (xr, right_end, True, endr)):
                    edge_info = self._tech_cls.get_laygo_edge_info(rinfo['blk_info'], end_flag)
                    edge_params = dict(
                        top_layer=self._top_layer,
                        is_end=is_end,
                        guard_ring_nf=self._guard_ring_nf,
                        name_id=edge_info['name_id'],
                        layout_info=edge_info,
                        is_laygo=True,
                    )
                    edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)
                    if flip_lr:
                        eorient = 'MY' if orient == 'R0' else 'R180'
                    else:
                        eorient = orient
                    edge_inst_list.append(self.add_instance(edge_master, orient=eorient, loc=(x, y), unit_mode=True))

            gr_vss_warrs = []
            gr_vdd_warrs = []
            conn_layer = self._tech_cls.get_dig_conn_layer()
            for inst in edge_inst_list:
                if inst.has_port('VDD'):
                    gr_vdd_warrs.extend(inst.get_all_port_pins('VDD', layer=conn_layer))
                elif inst.has_port('VSS'):
                    gr_vss_warrs.extend(inst.get_all_port_pins('VSS', layer=conn_layer))

            # connect body guard rings together
            gr_vdd_warrs = self.connect_wires(gr_vdd_warrs)
            gr_vss_warrs = self.connect_wires(gr_vss_warrs)

            self._has_boundaries = True
            return gr_vdd_warrs, gr_vss_warrs

        return [], []
