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
from typing import Dict, Any, Set, Tuple, List
from future.utils import with_metaclass

import bisect

from bag.math import lcm
from bag.util.interval import IntervalSet

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance
from bag.layout.routing import TrackID

from ..analog_mos.mos import AnalogMOSExt
from ..analog_mos.edge import AnalogEdge
from .tech import LaygoTech
from .base import LaygoPrimitive, LaygoSubstrate, LaygoEndRow, LaygoSpace


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
        if 0 not in self._end_flags:
            start_flag = False
        else:
            start_flag = self._end_flags[0]

        if num_col not in self._end_flags:
            end_flag = False
        else:
            end_flag = self._end_flags[num_col]
        return start_flag, end_flag

    def get_end(self):
        if not self._intv:
            return 0
        return self._intv.get_end()


class LaygoBaseInfo(object):
    def __init__(self, grid, config, top_layer=None, guard_ring_nf=0, draw_boundaries=False, end_mode=0):
        # update routing grid
        self._config = config
        self.grid = grid.copy()
        self._tech_cls = self.grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self._lch_unit = int(round(self._config['lch'] / self.grid.layout_unit / self.grid.resolution))
        vm_layer = self._tech_cls.get_dig_conn_layer()
        vm_space, vm_width = self._tech_cls.get_laygo_conn_track_info(self._lch_unit)
        self.grid.add_new_layer(vm_layer, vm_space, vm_width, 'y', override=True, unit_mode=True)
        tdir = 'x'
        for lay, w, sp in zip(self._config['tr_layers'], self._config['tr_widths'], self._config['tr_spaces']):
            self.grid.add_new_layer(lay, sp, w, tdir, override=True, unit_mode=True)
            tdir = 'x' if tdir == 'y' else 'y'

        self.grid.update_block_pitch()

        # set attributes
        self.top_layer = self._config['tr_layers'][-1] if top_layer is None else top_layer
        self._col_width = self._tech_cls.get_sd_pitch(self._lch_unit) * self._tech_cls.get_laygo_unit_fg()
        self.guard_ring_nf = guard_ring_nf
        self.draw_boundaries = draw_boundaries
        self.end_mode = end_mode

    @property
    def tech_cls(self):
        return self._tech_cls

    @property
    def conn_layer(self):
        return self._tech_cls.get_dig_conn_layer()

    @property
    def fg2d_s_short(self):
        return self._tech_cls.get_laygo_fg2d_s_short()

    @property
    def lch(self):
        return self._lch_unit * self.grid.layout_unit * self.grid.resolution

    @property
    def lch_unit(self):
        return self._lch_unit

    @property
    def col_width(self):
        return self._col_width

    @property
    def tot_height_pitch(self):
        return self.grid.get_block_size(self.top_layer, unit_mode=True)[1]

    @property
    def mos_pitch(self):
        return self._tech_cls.get_mos_pitch(unit_mode=True)

    @property
    def left_margin(self):
        return self._tech_cls.get_left_sd_xc(self.grid, self._lch_unit, self.guard_ring_nf,
                                             self.top_layer, self.end_mode & 4 != 0)

    @property
    def right_margin(self):
        return self._tech_cls.get_left_sd_xc(self.grid, self._lch_unit, self.guard_ring_nf,
                                             self.top_layer, self.end_mode & 8 != 0)

    def __getitem__(self, item):
        return self._config[item]


class LaygoBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

        self._laygo_info = LaygoBaseInfo(self.grid, self.params['config'])
        self.grid = self._laygo_info.grid
        self._tech_cls = self._laygo_info.tech_cls

        # initialize attributes
        self._num_rows = 0
        self._laygo_size = None
        self._row_types = None
        self._row_orientations = None
        self._row_thresholds = None
        self._row_infos = None
        self._row_kwargs = None
        self._row_y = None
        self._ext_params = None
        self._used_list = None  # type: List[LaygoIntvSet]
        self._bot_end_master = None
        self._top_end_master = None
        self._has_boundaries = False
        self._ext_edges = None

    @property
    def laygo_size(self):
        return self._laygo_size

    @property
    def conn_layer(self):
        return self._laygo_info.conn_layer

    @property
    def fg2d_s_short(self):
        return self._laygo_info.fg2d_s_short

    def _get_track_intervals(self, hm_layer, orient, info, ycur, ybot, ytop, delta):
        if 'g_conn_y' in info:
            gyt = info['g_conn_y'][1]
        else:
            gyt = ybot

        syb = info['ds_conn_y'][0]
        dyb = info['gb_conn_y'][0]

        if orient == 'R0':
            gyt += ycur
            syb += ycur
            dyb += ycur

            gbtr = self.grid.coord_to_nearest_track(hm_layer, ybot + delta, half_track=True, mode=1, unit_mode=True)
            gttr = self.grid.coord_to_nearest_track(hm_layer, gyt - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(gttr - gbtr + 1))
            g_intv = (gttr - num_tr + 1, gttr + 1)
            sbtr = self.grid.coord_to_nearest_track(hm_layer, syb + delta, half_track=True, mode=1, unit_mode=True)
            sttr = self.grid.coord_to_nearest_track(hm_layer, ytop - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(sttr - sbtr + 1))
            s_intv = (sbtr, sbtr + num_tr)
            dbtr = self.grid.coord_to_nearest_track(hm_layer, dyb + delta, half_track=True, mode=1, unit_mode=True)
            dttr = self.grid.coord_to_nearest_track(hm_layer, ytop - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(dttr - dbtr + 1))
            d_intv = (dbtr, dbtr + num_tr)
        else:
            h = info['blk_height']
            gyb = ycur + h - gyt
            dyt = ycur + h - dyb
            syt = ycur + h - syb

            gbtr = self.grid.coord_to_nearest_track(hm_layer, gyb + delta, half_track=True, mode=1, unit_mode=True)
            gttr = self.grid.coord_to_nearest_track(hm_layer, ytop - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(gttr - gbtr + 1))
            g_intv = (gbtr, gbtr + num_tr)
            sbtr = self.grid.coord_to_nearest_track(hm_layer, ybot + delta, half_track=True, mode=1, unit_mode=True)
            sttr = self.grid.coord_to_nearest_track(hm_layer, syt - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(sttr - sbtr + 1))
            s_intv = (sttr - num_tr + 1, sttr + 1)
            dbtr = self.grid.coord_to_nearest_track(hm_layer, ybot + delta, half_track=True, mode=1, unit_mode=True)
            dttr = self.grid.coord_to_nearest_track(hm_layer, dyt - delta, half_track=True, mode=-1, unit_mode=True)
            num_tr = max(0, int(dttr - dbtr + 1))
            d_intv = (dttr - num_tr + 1, dttr + 1)

        return g_intv, s_intv, d_intv

    def set_row_types(self, row_types, row_orientations, row_thresholds, draw_boundaries, end_mode,
                      num_g_tracks, num_gb_tracks, num_ds_tracks, top_layer=None, guard_ring_nf=0,
                      row_kwargs=None):

        # error checking
        if (row_types[0] == 'ptap' or row_types[0] == 'ntap') and row_orientations[0] != 'R0':
            raise ValueError('bottom substrate orientation must be R0')
        if (row_types[-1] == 'ptap' or row_types[-1] == 'ntap') and row_orientations[-1] != 'MX':
            raise ValueError('top substrate orientation must be MX')
        if len(row_types) < 2:
            raise ValueError('Must draw at least 2 rows.')
        if row_kwargs is None:
            row_kwargs = [{}] * len(row_types)

        # update LaygoInfo information

        if not draw_boundaries:
            end_mode = 0
        if top_layer is not None:
            self._laygo_info.top_layer = top_layer
        else:
            top_layer = self._laygo_info.top_layer

        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode

        tot_height_pitch = self._laygo_info.tot_height_pitch

        # get layout information for all rows
        self._num_rows = len(row_types)
        self._row_types = row_types
        self._row_orientations = row_orientations
        self._row_thresholds = row_thresholds
        self._row_kwargs = row_kwargs
        self._used_list = [LaygoIntvSet() for _ in range(self._num_rows)]

        if draw_boundaries:
            bot_end = (end_mode & 1) != 0
            top_end = (end_mode & 2) != 0

            if row_types[0] != 'ntap' and row_types[0] != 'ptap':
                raise ValueError('Bottom row must be substrate.')
            if row_types[-1] != 'ntap' and row_types[-1] != 'ptap':
                raise ValueError('Top row must be substrate.')

            # create boundary masters
            params = dict(
                lch=self._laygo_info.lch,
                mos_type=self._row_types[0],
                threshold=self._row_thresholds[0],
                is_end=bot_end,
                top_layer=top_layer,
            )
            self._bot_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
            params = dict(
                lch=self._laygo_info.lch,
                mos_type=self._row_types[-1],
                threshold=self._row_thresholds[-1],
                is_end=top_end,
                top_layer=top_layer,
            )
            self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
            ybot = self._bot_end_master.bound_box.height_unit
        else:
            ybot = 0

        row_specs = self._get_row_specs(row_types, row_orientations, row_thresholds, row_kwargs,
                                        num_g_tracks, num_gb_tracks, num_ds_tracks)

        # compute location and information of each row
        result = self._place_rows(ybot, tot_height_pitch, row_specs, row_types, row_thresholds)
        self._row_infos, self._ext_params, self._row_y = result

    def _get_row_specs(self, row_types, row_orientations, row_thresholds, row_kwargs,
                       num_g_tracks, num_gb_tracks, num_ds_tracks):
        lch = self._laygo_info.lch
        lch_unit = int(round(lch / self.grid.layout_unit / self.grid.resolution))
        w_sub = self._laygo_info['w_sub']
        w_n = self._laygo_info['w_n']
        w_p = self._laygo_info['w_p']
        min_sub_tracks = self._laygo_info['min_sub_tracks']
        min_n_tracks = self._laygo_info['min_n_tracks']
        min_p_tracks = self._laygo_info['min_p_tracks']
        mos_pitch = self._laygo_info.mos_pitch

        row_specs = []
        for row_type, row_orient, row_thres, kwargs, ng, ngb, nds in \
                zip(row_types, row_orientations, row_thresholds, row_kwargs,
                    num_g_tracks, num_gb_tracks, num_ds_tracks):

            # get information dictionary
            if row_type == 'nch':
                mos_info = self._tech_cls.get_laygo_mos_info(lch_unit, w_n, row_type, row_thres, 'general', **kwargs)
                min_tracks = min_n_tracks
            elif row_type == 'pch':
                mos_info = self._tech_cls.get_laygo_mos_info(lch_unit, w_p, row_type, row_thres, 'general', **kwargs)
                min_tracks = min_p_tracks
            elif row_type == 'ptap':
                mos_info = self._tech_cls.get_laygo_sub_info(lch_unit, w_sub, row_type, row_thres, **kwargs)
                min_tracks = min_sub_tracks
            elif row_type == 'ntap':
                mos_info = self._tech_cls.get_laygo_sub_info(lch_unit, w_sub, row_type, row_thres, **kwargs)
                min_tracks = min_sub_tracks
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            row_pitch = min_row_height = mos_pitch
            for layer, num_tr in min_tracks:
                tr_pitch = self.grid.get_track_pitch(layer, unit_mode=True)
                min_row_height = max(min_row_height, num_tr * tr_pitch)
                row_pitch = lcm([row_pitch, tr_pitch])

            row_specs.append((row_type, row_orient, mos_info, min_row_height, row_pitch, (ng, ngb, nds)))

        return row_specs

    def _place_rows(self, ybot, tot_height_pitch, row_specs, row_types, row_thresholds):
        lch_unit = self._laygo_info.lch_unit
        ext_params_list = []
        row_infos = []
        row_y = []
        conn_layer = self._tech_cls.get_dig_conn_layer()
        hm_layer = conn_layer + 1
        via_ext = self.grid.get_via_extensions(conn_layer, 1, 1, unit_mode=True)[0]
        hm_width, hm_space = self.grid.get_track_info(hm_layer, unit_mode=True)
        mos_pitch = self._tech_cls.get_mos_pitch(unit_mode=True)
        conn_delta = hm_width // 2 + via_ext
        prev_ext_info = None
        prev_ext_h = 0
        y0 = ybot
        for idx, (row_type, row_orient, mos_info, min_row_height, row_pitch, (ng, ngb, nds)) in enumerate(row_specs):

            # get information dictionary
            is_sub = (row_type == 'ptap' or row_type == 'ntap')

            # get extension information
            if row_orient == 'R0':
                ext_bot_info = mos_info['ext_bot_info']
                ext_top_info = mos_info['ext_top_info']
            else:
                ext_top_info = mos_info['ext_bot_info']
                ext_bot_info = mos_info['ext_top_info']

            blk_height = mos_info['blk_height']
            # step 1: find Y coordinate
            if idx == 0 and is_sub:
                # bottom substrate has orientation R0, just abut to bottom.
                ycur = y0
                cur_bot_ext_h = 0
            else:
                # step A: find bottom connection Y coordinate and number of tracks
                if row_orient == 'R0':
                    # gate tracks on bottom
                    num_tr1 = 0 if is_sub else ng
                    num_tr2 = num_tr1
                    conn_yb1, conn_yt1 = mos_info.get('g_conn_y', (0, 0))
                    conn_yb2, conn_yt2 = conn_yb1, conn_yt1
                else:
                    # drain/source tracks on bottom
                    num_tr1, num_tr2 = ngb, nds
                    conn_yb1, conn_yt1 = mos_info['gb_conn_y']
                    conn_yb2, conn_yt2 = mos_info['ds_conn_y']
                    conn_yb1, conn_yt1 = blk_height - conn_yt1, blk_height - conn_yb1
                    conn_yb2, conn_yt2 = blk_height - conn_yt2, blk_height - conn_yb2

                # step B: find max Y coordinate from constraints
                ycur = y0
                tr0 = self.grid.find_next_track(hm_layer, y0 + conn_delta, half_track=True, mode=1, unit_mode=True)
                tr_ybot = self.grid.track_to_coord(hm_layer, tr0, unit_mode=True)
                for ntr, cyb, cyt in ((num_tr1, conn_yb1, conn_yt1),
                                      (num_tr2, conn_yb2, conn_yt2)):
                    if ntr > 0:
                        tr_ytop = self.grid.track_to_coord(hm_layer, tr0 + ntr - 1, unit_mode=True)
                        # make sure bottom line-end is above the bottom horizontal track
                        ycur = max(ycur, tr_ybot - cyb - conn_delta)
                        # make sure top line_end is above top horizontal track
                        ycur = max(ycur, tr_ytop - cyt + conn_delta)

                # step C: round Y coordinate to mos_pitch
                ycur = -(-ycur // mos_pitch) * mos_pitch
                cur_bot_ext_h = (ycur - y0) // mos_pitch
                # step D: make sure extension constraints is met
                if idx != 0:
                    valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_bot_info, prev_ext_info)
                    ext_h = prev_ext_h + cur_bot_ext_h
                    if ext_h not in valid_widths and ext_h < valid_widths[-1]:
                        # make sure extension height is valid
                        ext_h = valid_widths[bisect.bisect_left(valid_widths, ext_h)]
                        cur_bot_ext_h = ext_h - prev_ext_h
                else:
                    # nmos/pmos at bottom row.  Need to check we can draw mirror image row.
                    raise ValueError('Not implemented yet.')

                ycur = y0 + cur_bot_ext_h * mos_pitch

            # at this point, ycur and cur_ext_h are determined
            if idx == self._num_rows - 1 and is_sub:
                # we need to quantize row height, total height, and substrate just abut to top edge.
                ytop = ycur + blk_height
                ytop = max(ytop, y0 + min_row_height)
                ytop = -(-ytop // row_pitch) * row_pitch
                tot_height = -(-(ytop - ybot) // tot_height_pitch) * tot_height_pitch
                ytop = ybot + tot_height
                ycur = ytop - blk_height
                cur_bot_ext_h = (ycur - y0) // mos_pitch
                cur_top_ext_h = 0
            else:
                if idx != self._num_rows - 1:
                    if row_orient == 'MX':
                        # gate tracks on bottom
                        num_tr1 = 0 if is_sub else ng
                        num_tr2 = num_tr1
                        conn_yb1, conn_yt1 = mos_info['g_conn_y']
                        conn_yb1, conn_yt1 = blk_height - conn_yt1, blk_height - conn_yb1
                        conn_yb2, conn_yt2 = conn_yb1, conn_yt1
                    else:
                        # drain/source tracks on bottom
                        num_tr1, num_tr2 = ngb, nds
                        conn_yb1, conn_yt1 = mos_info['gb_conn_y']
                        conn_yb2, conn_yt2 = mos_info['ds_conn_y']

                    # compute top extension from constraints
                    ytop = max(ycur + blk_height, y0 + min_row_height)
                    for ntr, cyb, cyt in ((num_tr1, conn_yb1, conn_yt1),
                                          (num_tr2, conn_yb2, conn_yt2)):
                        if ntr > 0:
                            ybtr = ycur + cyb + conn_delta
                            tr0 = self.grid.find_next_track(hm_layer, ybtr, half_track=True, mode=1, unit_mode=True)
                            yttr = ycur + cyt - conn_delta
                            tr1 = self.grid.find_next_track(hm_layer, yttr, half_track=True, mode=-1, unit_mode=True)
                            tr1 = max(tr1, tr0 + ntr - 1)
                            ytop = max(ytop, self.grid.track_to_coord(hm_layer, tr1, unit_mode=True) + conn_delta)

                    ytop = -(-ytop // row_pitch) * row_pitch
                    cur_top_ext_h = (ytop - ycur - blk_height) // mos_pitch
                else:
                    # nmos/pmos at top row.  Compute top extension from mirror image, then move block up.
                    raise ValueError('Not implemented yet.')

            # recompute gate and drain/source track indices
            g_intv, ds_intv, gb_intv = self._get_track_intervals(hm_layer, row_orient, mos_info,
                                                                 ycur, y0, ytop, conn_delta)

            if ng > g_intv[1] - g_intv[0] or nds > ds_intv[1] - ds_intv[0] or ngb > gb_intv[1] - gb_intv[0]:
                import pdb
                pdb.set_trace()
                g_intv, ds_intv, gb_intv = self._get_track_intervals(hm_layer, row_orient, mos_info,
                                                                     ycur, y0, ytop, conn_delta)
            # record information
            mos_info['g_intv'] = g_intv
            mos_info['ds_intv'] = ds_intv
            mos_info['gb_intv'] = gb_intv
            if prev_ext_info is None:
                ext_params = None
            else:
                ext_params = dict(
                    lch=self._laygo_info.lch,
                    w=prev_ext_h + cur_bot_ext_h,
                    bot_mtype=row_types[idx - 1],
                    top_mtype=row_type,
                    bot_thres=row_thresholds[idx - 1],
                    top_thres=row_thresholds[idx],
                    top_ext_info=prev_ext_info,
                    bot_ext_info=ext_bot_info,
                    is_laygo=True,
                )
            row_y.append((y0, ycur, ycur + blk_height, ytop))
            row_infos.append(mos_info)
            ext_params_list.append(ext_params)

            y0 = ytop
            prev_ext_info = ext_top_info
            prev_ext_h = cur_top_ext_h

        return row_infos, ext_params_list, row_y

    def get_track_index(self, row_idx, tr_type, tr_idx):
        row_info = self._row_infos[row_idx]
        orient = self._row_orientations[row_idx]
        intv = row_info['%s_intv' % tr_type]
        ntr = int(intv[1] - intv[0])
        if tr_idx >= ntr:
            raise ValueError('tr_idx = %d >= %d' % (tr_idx, ntr))

        if orient == 'R0':
            return intv[0] + tr_idx
        else:
            return intv[1] - 1 - tr_idx

    def make_track_id(self, row_idx, tr_type, tr_idx, width=1, num=1, pitch=0.0):
        tid = self.get_track_index(row_idx, tr_type, tr_idx)
        hm_layer = self._tech_cls.get_dig_conn_layer() + 1
        return TrackID(hm_layer, tid, width=width, num=num, pitch=pitch)

    def get_end_flags(self, row_idx):
        return self._used_list[row_idx].get_end_flags(self._laygo_size[0])

    def set_laygo_size(self, num_col=None):
        if self._laygo_size is None:
            if num_col is None:
                num_col = 0
                for intv in self._used_list:
                    num_col = max(num_col, intv.get_end())

            self._laygo_size = num_col, self._num_rows

            top_layer = self._laygo_info.top_layer
            draw_boundaries = self._laygo_info.draw_boundaries
            end_mode = self._laygo_info.end_mode
            col_width = self._laygo_info.col_width
            left_margin = self._laygo_info.left_margin
            right_margin = self._laygo_info.right_margin
            guard_ring_nf = self._laygo_info.guard_ring_nf

            width = col_width * num_col
            height = self._row_y[-1][-1]
            if draw_boundaries:
                width += left_margin + right_margin
                height += self._top_end_master.bound_box.height_unit
            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            self.add_cell_boundary(bound_box)

            # draw extensions and record edge parameters
            left_end = (end_mode & 4) != 0
            right_end = (end_mode & 8) != 0
            xr = left_margin + col_width * num_col + right_margin
            self._ext_edges = []
            for idx, (bot_info, ext_params) in enumerate(zip(self._row_infos, self._ext_params)):
                if ext_params is not None:
                    ext_h = ext_params['w']
                    if ext_h > 0 or self._tech_cls.draw_zero_extension():
                        yext = self._row_y[idx - 1][2]
                        ext_master = self.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                        self.add_instance(ext_master, inst_name='XEXT%d' % idx, loc=(left_margin, yext),
                                          nx=num_col, spx=col_width, unit_mode=True)
                        if draw_boundaries:
                            for x, is_end, flip_lr in ((0, left_end, False), (xr, right_end, True)):
                                edge_params = dict(
                                    top_layer=top_layer,
                                    is_end=is_end,
                                    guard_ring_nf=guard_ring_nf,
                                    name_id=ext_master.get_layout_basename(),
                                    layout_info=ext_master.get_edge_layout_info(),
                                    is_laygo=True,
                                )
                                edge_orient = 'MY' if flip_lr else 'R0'
                                self._ext_edges.append((x, yext, edge_orient, edge_params))

    def add_laygo_primitive(self, blk_type, loc=(0, 0), flip=False, nx=1, spx=0, **kwargs):
        # type: (str, Tuple[int, int], bool, int, int, **kwargs) -> Instance

        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= len(self._row_types):
            raise ValueError('Cannot add primitive at row %d' % row_idx)

        lch = self._laygo_info.lch
        col_width = self._laygo_info.col_width
        left_margin = self._laygo_info.left_margin

        mos_type = self._row_types[row_idx]
        row_orient = self._row_orientations[row_idx]
        threshold = self._row_thresholds[row_idx]
        if mos_type == 'nch':
            w = self._laygo_info['w_n']
        elif mos_type == 'pch':
            w = self._laygo_info['w_p']
        else:
            w = self._laygo_info['w_sub']

        # make master
        options = self._row_kwargs[row_idx].copy()
        options.update(kwargs)
        params = dict(
            lch=lch,
            w=w,
            mos_type=mos_type,
            threshold=threshold,
            options=options,
        )
        if blk_type == 'sub':
            master = self.new_template(params=params, temp_cls=LaygoSubstrate)
        else:
            params['blk_type'] = blk_type
            master = self.new_template(params=params, temp_cls=LaygoPrimitive)

        intv = self._used_list[row_idx]
        inst_endl, inst_endr = master.get_end_flags()
        if flip:
            inst_endl, inst_endr = inst_endr, inst_endl
        for inst_num in range(nx):
            intv_offset = col_idx + spx * inst_num
            inst_intv = intv_offset, intv_offset + 1
            if not intv.add(inst_intv, inst_endl, inst_endr):
                raise ValueError('Cannot add primitive on row %d, '
                                 'column [%d, %d).' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = left_margin + col_idx * col_width
        if flip:
            x0 += col_width

        _, ycur, ytop, _ = self._row_y[row_idx]
        if row_orient == 'R0':
            y0 = self._row_y[row_idx][1]
            orient = 'MY' if flip else 'R0'
        else:
            y0 = self._row_y[row_idx][2]
            orient = 'R180' if flip else 'MX'

        # convert horizontal pitch to resolution units
        spx *= col_width

        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def fill_space(self):
        if self._laygo_size is None:
            raise ValueError('laygo_size must be set before filling spaces.')

        col_width = self._laygo_info.col_width
        left_margin = self._laygo_info.left_margin

        total_intv = (0, self._laygo_size[0])
        for row_idx, (intv, row_orient, row_info, row_y) in \
                enumerate(zip(self._used_list, self._row_orientations, self._row_infos, self._row_y)):
            for (start, end), (flag_l, flag_r) in zip(*intv.get_complement(total_intv)):
                od_flag = 0
                if flag_l:
                    od_flag |= 1
                if flag_r:
                    od_flag |= 2
                num_blk = end - start
                params = dict(
                    row_info=row_info,
                    name_id=row_info['row_name_id'],
                    num_blk=num_blk,
                    adj_od_flag=od_flag,
                )
                inst_name = 'XR%dC%d' % (row_idx, start)
                master = self.new_template(params=params, temp_cls=LaygoSpace)
                x0 = left_margin + start * col_width
                y0 = row_y[1] if row_orient == 'R0' else row_y[2]
                self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=row_orient, unit_mode=True)

    def draw_boundary_cells(self):
        draw_boundaries = self._laygo_info.draw_boundaries
        end_mode = self._laygo_info.end_mode
        top_layer = self._laygo_info.top_layer
        guard_ring_nf = self._laygo_info.guard_ring_nf
        col_width = self._laygo_info.col_width
        left_margin = self._laygo_info.left_margin
        right_margin = self._laygo_info.right_margin

        if draw_boundaries and not self._has_boundaries:
            if self._laygo_size is None:
                raise ValueError('laygo_size must be set before drawing boundaries.')

            nx = self._laygo_size[0]
            spx = col_width

            # draw top and bottom end row
            self.add_instance(self._bot_end_master, inst_name='XRBOT', loc=(left_margin, 0),
                              nx=nx, spx=spx, unit_mode=True)
            yt = self.bound_box.height_unit
            self.add_instance(self._top_end_master, inst_name='XRBOT',
                              loc=(left_margin, yt),
                              orient='MX', nx=nx, spx=spx, unit_mode=True)
            # draw corners
            left_end = (end_mode & 4) != 0
            right_end = (end_mode & 8) != 0
            edge_inst_list = []
            xr = left_margin + col_width * nx + right_margin
            for orient, y, master in (('R0', 0, self._bot_end_master), ('MX', yt, self._top_end_master)):
                for x, is_end, flip_lr in ((0, left_end, False), (xr, right_end, True)):
                    edge_params = dict(
                        top_layer=top_layer,
                        is_end=is_end,
                        guard_ring_nf=guard_ring_nf,
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
            for ridx, (orient, ytuple, rinfo) in enumerate(zip(self._row_orientations, self._row_y, self._row_infos)):
                endl, endr = self.get_end_flags(ridx)
                _, ycur, ytop, _ = ytuple
                if orient == 'R0':
                    y = ycur
                else:
                    y = ytop
                for x, is_end, flip_lr, end_flag in ((0, left_end, False, endl), (xr, right_end, True, endr)):
                    edge_info = self._tech_cls.get_laygo_edge_info(rinfo, end_flag)
                    edge_params = dict(
                        top_layer=top_layer,
                        is_end=is_end,
                        guard_ring_nf=guard_ring_nf,
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
