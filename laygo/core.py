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
from .tech import LaygoTech
from .base import LaygoPrimitive, LaygoSubstrate, LaygoEndRow, LaygoSpace


class LaygoIntvSet(object):
    def __init__(self, default_end_info):
        super(LaygoIntvSet, self).__init__()
        self._intv = IntervalSet()
        self._end_flags = {}
        self._default_end_info = default_end_info

    def add(self, intv, ext_info, endl, endr):
        ans = self._intv.add(intv, val=ext_info)
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

    def items(self):
        return self._intv.items()

    def get_complement(self, total_intv):
        compl_intv = self._intv.get_complement(total_intv)
        intv_list = []
        end_list = []
        for intv in compl_intv:
            intv_list.append(intv)
            end_list.append((self._end_flags.get(intv[0], self._default_end_info),
                             self._end_flags.get(intv[1], self._default_end_info)))
        return intv_list, end_list

    def get_end_info(self, num_col):
        if 0 not in self._end_flags:
            start_info = self._default_end_info
        else:
            start_info = self._end_flags[0]

        if num_col not in self._end_flags:
            end_info = self._default_end_info
        else:
            end_info = self._end_flags[num_col]

        return start_info, end_info

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
    def unit_fg(self):
        return self._tech_cls.get_laygo_unit_fg()

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
    def sub_columns(self):
        return self._tech_cls.get_sub_columns(self._lch_unit)

    @property
    def min_sub_space(self):
        return self._tech_cls.get_min_sub_space_columns(self._lch_unit)

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
        if self.draw_boundaries:
            return self._tech_cls.get_left_sd_xc(self.grid, self._lch_unit, self.guard_ring_nf,
                                                 self.top_layer, self.end_mode & 4 != 0)
        else:
            return 0

    @property
    def right_margin(self):
        if self.draw_boundaries:
            return self._tech_cls.get_left_sd_xc(self.grid, self._lch_unit, self.guard_ring_nf,
                                                 self.top_layer, self.end_mode & 8 != 0)
        else:
            return 0

    def __getitem__(self, item):
        return self._config[item]

    def col_to_coord(self, col_idx, ds_type, unit_mode=False):
        offset = self.left_margin + col_idx * self._col_width
        if ds_type == 's':
            ans = offset
        elif ds_type == 'd':
            ans = offset + self._col_width // 2
        else:
            raise ValueError('Unrecognized ds type: %s' % ds_type)

        if unit_mode:
            return ans
        return ans * self.grid.resolution

    def coord_to_nearest_col(self, coord, ds_type=None, mode=0, unit_mode=False):
        if not unit_mode:
            coord = int(round(coord / self.grid.resolution))

        col_width = self._col_width
        if ds_type is None or ds_type == 's':
            offset = self.left_margin
        else:
            offset = self.left_margin + col_width
        if ds_type is None:
            k = col_width // 2
        else:
            k = col_width

        coord -= offset
        if mode == 0:
            n = int(round(coord / k))
        elif mode > 0:
            if coord % k == 0 and mode == 2:
                coord += 1
            n = -(-coord // k)
        else:
            if coord % k == 0 and mode == -2:
                coord -= 1
            n = coord // k

        return self.coord_to_col(n * k + offset, unit_mode=True)

    def coord_to_col(self, coord, unit_mode=False):
        if not unit_mode:
            coord = int(round(coord / self.grid.resolution))

        k = self._col_width // 2
        offset = self.left_margin
        if (coord - offset) % k != 0:
            raise ValueError('Coordinate %d is not on pitch.' % coord)
        col_idx_half = (coord - offset) // k

        if col_idx_half % 2 == 0:
            return col_idx_half // 2, 's'
        else:
            return (col_idx_half - 1) // 2, 'd'


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
        self._row_widths = None
        self._row_orientations = None
        self._row_thresholds = None
        self._row_min_tracks = None
        self._row_infos = None
        self._row_kwargs = None
        self._row_y = None
        self._ext_params = None
        self._used_list = None  # type: List[LaygoIntvSet]
        self._bot_end_master = None
        self._top_end_master = None
        self._ext_edge_infos = None
        self._bot_sub_extw = 0
        self._top_sub_extw = 0

    @property
    def laygo_info(self):
        # type: () -> LaygoBaseInfo
        return self._laygo_info

    @property
    def laygo_size(self):
        return self._laygo_size

    @property
    def conn_layer(self):
        return self._laygo_info.conn_layer

    @property
    def fg2d_s_short(self):
        return self._laygo_info.fg2d_s_short

    @property
    def sub_columns(self):
        return self._laygo_info.sub_columns

    @property
    def min_sub_space(self):
        return self._laygo_info.min_sub_space

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

    def set_row_types(self, row_types, row_widths, row_orientations, row_thresholds, draw_boundaries, end_mode,
                      num_g_tracks, num_gb_tracks, num_ds_tracks, row_min_tracks=None, top_layer=None, guard_ring_nf=0,
                      row_kwargs=None):

        # error checking
        if (row_types[0] == 'ptap' or row_types[0] == 'ntap') and row_orientations[0] != 'R0':
            raise ValueError('bottom substrate orientation must be R0')
        if (row_types[-1] == 'ptap' or row_types[-1] == 'ntap') and row_orientations[-1] != 'MX':
            raise ValueError('top substrate orientation must be MX')
        if len(row_types) < 2:
            raise ValueError('Must draw at least 2 rows.')

        self._num_rows = len(row_types)
        if row_kwargs is None:
            row_kwargs = [{}] * self._num_rows
        if row_min_tracks is None:
            row_min_tracks = [{}] * self._num_rows

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
        self._row_types = row_types
        self._row_widths = row_widths
        self._row_orientations = row_orientations
        self._row_thresholds = row_thresholds
        self._row_kwargs = row_kwargs
        self._row_min_tracks = row_min_tracks
        default_end_info = self._tech_cls.get_default_end_info()
        self._used_list = [LaygoIntvSet(default_end_info) for _ in range(self._num_rows)]

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

        row_specs = self._get_row_specs(row_types, row_widths, row_orientations, row_thresholds, row_min_tracks,
                                        row_kwargs, num_g_tracks, num_gb_tracks, num_ds_tracks)

        # compute location and information of each row
        result = self._place_rows(ybot, tot_height_pitch, row_specs)
        self._row_infos, self._ext_params, self._row_y = result

    def get_digital_row_info(self):
        if not self.finalized:
            raise ValueError('Can only compute digital row info if this block is finalized.')
        if self._laygo_info.draw_boundaries is True:
            raise ValueError('LaygoBase with boundaries cannot be used in digital row.')

        mos_pitch = self._laygo_info.mos_pitch
        ans = dict(
            config=self.params['config'],
            row_height=self.bound_box.top_unit,
            row_types=self._row_types,
            row_thresholds=self._row_thresholds,
            bot_extw=(self._row_y[0][1] - self._row_y[0][0]) // mos_pitch,
            top_extw=(self._row_y[-1][3] - self._row_y[-1][2]) // mos_pitch,
            bot_sub_extw=self._bot_sub_extw,
            top_sub_extw=self._top_sub_extw,
            bot_ext_info=self._row_infos[0]['ext_bot_info'],
            top_ext_info=self._row_infos[-1]['ext_top_info'],
            row_edge_infos=self._get_row_edge_infos(),
            ext_edge_infos=self._ext_edge_infos,
        )
        return ans

    def _get_row_specs(self, row_types, row_widths, row_orientations, row_thresholds, row_min_tracks, row_kwargs,
                       num_g_tracks, num_gb_tracks, num_ds_tracks):
        lch = self._laygo_info.lch
        lch_unit = int(round(lch / self.grid.layout_unit / self.grid.resolution))
        mos_pitch = self._laygo_info.mos_pitch

        row_specs = []
        for row_type, row_w, row_orient, row_thres, min_tracks, kwargs, ng, ngb, nds in \
                zip(row_types, row_widths, row_orientations, row_thresholds, row_min_tracks, row_kwargs,
                    num_g_tracks, num_gb_tracks, num_ds_tracks):

            # get information dictionary
            if row_type == 'nch' or row_type == 'pch':
                mos_info = self._tech_cls.get_laygo_mos_info(lch_unit, row_w, row_type, row_thres, 'general', **kwargs)
            elif row_type == 'ptap' or row_type == 'ntap':
                mos_info = self._tech_cls.get_laygo_sub_info(lch_unit, row_w, row_type, row_thres, **kwargs)
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            row_pitch = min_row_height = mos_pitch
            for layer, num_tr in min_tracks:
                tr_pitch = self.grid.get_track_pitch(layer, unit_mode=True)
                min_row_height = max(min_row_height, num_tr * tr_pitch)
                row_pitch = lcm([row_pitch, tr_pitch])

            row_specs.append((row_type, row_orient, mos_info, min_row_height, row_pitch, (ng, ngb, nds)))

        return row_specs

    def _place_with_num_tracks(self, row_info, row_orient, y0, hm_layer, conn_delta, mos_pitch, ng, ngb, nds):
        blk_height = row_info['blk_height']
        if row_orient == 'R0':
            # gate tracks on bottom
            num_tr1 = ng
            num_tr2 = num_tr1
            conn_yb1, conn_yt1 = row_info.get('g_conn_y', (0, 0))
            conn_yb2, conn_yt2 = conn_yb1, conn_yt1
        else:
            # drain/source tracks on bottom
            num_tr1, num_tr2 = ngb, nds
            conn_yb1, conn_yt1 = row_info['gb_conn_y']
            conn_yb2, conn_yt2 = row_info['ds_conn_y']
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
        return ycur

    def _place_mirror_or_sub(self, row_type, row_thres, lch_unit, mos_pitch, ydelta, ext_info):
        # find substrate parameters
        sub_type = 'ntap' if row_type == 'pch' or row_type == 'ntap' else 'ptap'
        w_sub = self._laygo_info['w_sub']
        min_sub_tracks = self._laygo_info['min_sub_tracks']
        sub_info = self._tech_cls.get_laygo_sub_info(lch_unit, w_sub, sub_type, row_thres)
        sub_ext_info = sub_info['ext_top_info']

        # quantize substrate height to top layer pitch.
        sub_height = sub_info['blk_height']
        min_sub_height = mos_pitch
        sub_pitch = lcm([mos_pitch, self.grid.get_track_pitch(self._laygo_info.top_layer, unit_mode=True)])
        for layer, num_tr in min_sub_tracks:
            tr_pitch = self.grid.get_track_pitch(layer, unit_mode=True)
            min_sub_height = max(min_sub_height, num_tr * tr_pitch)
            sub_pitch = lcm([sub_pitch, tr_pitch])

        real_sub_height = max(sub_height, min_sub_height)
        real_sub_height = -(-real_sub_height // sub_pitch) * sub_pitch
        sub_extw = (real_sub_height - sub_height) // mos_pitch

        # repeat until we satisfy both substrate and mirror row constraint
        ext_w = -(-ydelta // mos_pitch)
        ext_w_valid = False
        while not ext_w_valid:
            ext_w_valid = True
            # check we satisfy substrate constraint
            valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, sub_ext_info, ext_info)
            ext_w_test = ext_w + sub_extw
            if ext_w_test < valid_widths[-1] and ext_w_test not in valid_widths:
                # did not pass substrate constraint, update extension width
                ext_w_valid = False
                ext_w_test = valid_widths[bisect.bisect_left(valid_widths, ext_w_test)]
                ext_w = ext_w_test - sub_extw
                continue

            # check we satisfy mirror extension constraint
            valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_info, ext_info)
            ext_w_test = ext_w * 2
            if ext_w_test < valid_widths[-1] and ext_w_test not in valid_widths:
                # did not pass extension constraint, update extension width.
                ext_w_valid = False
                ext_w_test = valid_widths[bisect.bisect_left(valid_widths, ext_w_test)]
                ext_w = -(-(ext_w_test // 2))

        return ext_w, sub_extw

    def _place_rows(self, ybot, tot_height_pitch, row_specs):
        lch_unit = self._laygo_info.lch_unit
        top_layer = self._laygo_info.top_layer
        guard_ring_nf = self._laygo_info.guard_ring_nf

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
                ng_cur = 0 if is_sub else ng
                ycur = self._place_with_num_tracks(mos_info, row_orient, y0, hm_layer, conn_delta, mos_pitch,
                                                   ng_cur, ngb, nds)
                cur_bot_ext_h = (ycur - y0) // mos_pitch
                # step D: make sure extension constraints is met
                if idx != 0:
                    valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_bot_info, prev_ext_info)
                    ext_h = prev_ext_h + cur_bot_ext_h
                    if ext_h < valid_widths[-1] and ext_h not in valid_widths:
                        # make sure extension height is valid
                        ext_h = valid_widths[bisect.bisect_left(valid_widths, ext_h)]
                        cur_bot_ext_h = ext_h - prev_ext_h
                else:
                    # nmos/pmos at bottom row.  Need to check we can draw mirror image row.
                    row_thres = self._row_thresholds[idx]
                    cur_bot_ext_h, self._bot_sub_extw = self._place_mirror_or_sub(row_type, row_thres, lch_unit,
                                                                                  mos_pitch, ycur - ybot, ext_bot_info)

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
                    # nmos/pmos at top row.
                    # step 1: compute distance of row from top edge
                    test_orient = 'R0' if row_orient == 'MX' else 'MX'
                    test_y0 = 0  # use 0 because we know the top edge is LCM of horizontal track pitches.
                    ydelta = self._place_with_num_tracks(mos_info, test_orient, test_y0, hm_layer, conn_delta,
                                                         mos_pitch, ng, ngb, nds)
                    # step 2: make sure ydelta can satisfy extension constraints.
                    row_thres = self._row_thresholds[idx]
                    cur_top_ext_h, self._top_sub_extw = self._place_mirror_or_sub(row_type, row_thres, lch_unit,
                                                                                  mos_pitch, ydelta, ext_bot_info)
                    ydelta = cur_top_ext_h * mos_pitch
                    # step 3: compute row height given ycur and ydelta, round to row_pitch
                    ytop = max(ycur + blk_height + ydelta, y0 + min_row_height)
                    ytop = -(-ytop // row_pitch) * row_pitch
                    # step 4: round to total height pitch
                    tot_height = -(-(ytop - ybot) // tot_height_pitch) * tot_height_pitch
                    ytop = ybot + tot_height
                    # step 4: update ycur
                    ycur = ytop - ydelta - blk_height

            # recompute gate and drain/source track indices
            g_intv, ds_intv, gb_intv = self._get_track_intervals(hm_layer, row_orient, mos_info,
                                                                 ycur, y0, ytop, conn_delta)

            if ng > g_intv[1] - g_intv[0] or nds > ds_intv[1] - ds_intv[0] or ngb > gb_intv[1] - gb_intv[0]:
                g_intv, ds_intv, gb_intv = self._get_track_intervals(hm_layer, row_orient, mos_info,
                                                                     ycur, y0, ytop, conn_delta)
            # record information
            mos_info['g_intv'] = g_intv
            mos_info['ds_intv'] = ds_intv
            mos_info['gb_intv'] = gb_intv
            if prev_ext_info is None:
                ext_y = 0
                edge_params = None
            else:
                nom_ext_params = dict(
                    lch=self._laygo_info.lch,
                    w=prev_ext_h + cur_bot_ext_h,
                    fg=self._laygo_info.unit_fg,
                    top_ext_info=ext_bot_info,
                    bot_ext_info=prev_ext_info,
                    is_laygo=True,
                )
                nom_ext_master = self.new_template(params=nom_ext_params, temp_cls=AnalogMOSExt)
                edge_params = dict(
                    top_layer=top_layer,
                    guard_ring_nf=guard_ring_nf,
                    name_id=nom_ext_master.get_layout_basename(),
                    layout_info=nom_ext_master.get_edge_layout_info(),
                    is_laygo=True,
                )
                ext_y = row_y[-1][2]
            row_y.append((y0, ycur, ycur + blk_height, ytop))
            row_infos.append(mos_info)
            ext_params_list.append((prev_ext_h + cur_bot_ext_h, ext_y, edge_params))

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

    def get_ext_info(self):
        return self._get_ext_info_row(self._num_rows - 1, 1), self._get_ext_info_row(0, 0)

    def _get_ext_info_row(self, row_idx, ext_idx):
        intv = self._used_list[row_idx]
        return [(end, ext_info[ext_idx]) for (_, end), ext_info in intv.items()]

    def get_end_info(self):
        endl_list, endr_list = [], []
        num_col = self._laygo_size[0]
        for intv in self._used_list:
            endl, endr = intv.get_end_info(num_col)
            endl_list.append(endl)
            endr_list.append(endr)

        return endl_list, endr_list

    def _get_end_info_row(self, row_idx):
        num_col = self._laygo_size[0]
        endl, endr = self._used_list[row_idx].get_end_info(num_col)
        return endl, endr

    def set_laygo_size(self, num_col=None):
        if self._laygo_size is None:
            if num_col is None:
                num_col = 0
                for intv in self._used_list:
                    num_col = max(num_col, intv.get_end())

            self._laygo_size = num_col, self._num_rows

            top_layer = self._laygo_info.top_layer
            draw_boundaries = self._laygo_info.draw_boundaries
            col_width = self._laygo_info.col_width
            left_margin = self._laygo_info.left_margin
            right_margin = self._laygo_info.right_margin

            width = col_width * num_col
            height = self._row_y[-1][-1]
            if draw_boundaries:
                width += left_margin + right_margin
                height += self._top_end_master.bound_box.height_unit
            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            self.add_cell_boundary(bound_box)

    def add_laygo_primitive(self, blk_type, loc=(0, 0), flip=False, nx=1, spx=0, **kwargs):
        # type: (str, Tuple[int, int], bool, int, int, **kwargs) -> Instance

        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= self._num_rows:
            raise ValueError('Cannot add primitive at row %d' % row_idx)

        lch = self._laygo_info.lch
        col_width = self._laygo_info.col_width
        left_margin = self._laygo_info.left_margin

        mos_type = self._row_types[row_idx]
        row_orient = self._row_orientations[row_idx]
        threshold = self._row_thresholds[row_idx]
        w = self._row_widths[row_idx]

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
        num_col = 1
        if mos_type == 'ntap' or mos_type == 'ptap':
            master = self.new_template(params=params, temp_cls=LaygoSubstrate)
        else:
            params['blk_type'] = blk_type
            master = self.new_template(params=params, temp_cls=LaygoPrimitive)
            if blk_type == 'sub':
                num_col = self.sub_columns

        intv = self._used_list[row_idx]
        inst_endl, inst_endr = master.get_end_info()
        ext_info = master.get_ext_info()
        if row_orient == 'MX':
            ext_info = ext_info[1], ext_info[0]
        if flip:
            inst_endl, inst_endr = inst_endr, inst_endl
        for inst_num in range(nx):
            intv_offset = col_idx + spx * inst_num
            inst_intv = intv_offset, intv_offset + num_col
            if not intv.add(inst_intv, ext_info, inst_endl, inst_endr):
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

        num_col = self._laygo_size[0]
        # add space blocks
        total_intv = (0, num_col)
        for row_idx, intv in enumerate(self._used_list):
            for (start, end), end_info in zip(*intv.get_complement(total_intv)):
                self.add_laygo_space(end_info, num_blk=end - start, loc=(start, row_idx))

        # draw extensions
        self._ext_edge_infos = []
        laygo_info = self._laygo_info
        tech_cls = laygo_info.tech_cls
        for bot_ridx in range(0, self._num_rows - 1):
            w, yext, edge_params = self._ext_params[bot_ridx + 1]
            bot_ext_list = self._get_ext_info_row(bot_ridx, 1)
            top_ext_list = self._get_ext_info_row(bot_ridx + 1, 0)
            self._ext_edge_infos.extend(tech_cls.draw_extensions(self, laygo_info, w, yext, bot_ext_list,
                                                                 top_ext_list, edge_params))

        # draw boundaries and return guard ring supplies in boundary cells
        return self._draw_boundary_cells()

    def add_laygo_space(self, adj_end_info, num_blk=1, loc=(0, 0), **kwargs):
        col_idx, row_idx = loc
        row_info = self._row_infos[row_idx]
        row_y = self._row_y[row_idx]
        row_orient = self._row_orientations[row_idx]
        intv = self._used_list[row_idx]

        params = dict(
            row_info=row_info,
            name_id=row_info['row_name_id'],
            num_blk=num_blk,
            adj_end_info=adj_end_info,
        )
        params.update(kwargs)
        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        master = self.new_template(params=params, temp_cls=LaygoSpace)

        # update used interval
        endl, endr = master.get_end_info()
        ext_info = master.get_ext_info()
        if row_orient == 'MX':
            ext_info = ext_info[1], ext_info[0]

        inst_intv = (col_idx, col_idx + num_blk)
        if not intv.add(inst_intv, ext_info, endl, endr):
            raise ValueError('Cannot add space on row %d, column [%d, %d)' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = self._laygo_info.left_margin + col_idx * self._laygo_info.col_width
        y0 = row_y[1] if row_orient == 'R0' else row_y[2]
        self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=row_orient, unit_mode=True)

    def _get_row_edge_infos(self):
        top_layer = self._laygo_info.top_layer
        guard_ring_nf = self._laygo_info.guard_ring_nf

        row_edge_infos = []
        for ridx, (orient, ytuple, rinfo) in enumerate(zip(self._row_orientations, self._row_y, self._row_infos)):
            if orient == 'R0':
                y = ytuple[1]
            else:
                y = ytuple[2]

            row_edge_params = dict(
                top_layer=top_layer,
                guard_ring_nf=guard_ring_nf,
                row_info=rinfo,
                is_laygo=True,
            )
            row_edge_infos.append((y, orient, row_edge_params))

        return row_edge_infos

    def _draw_boundary_cells(self):
        if self._laygo_info.draw_boundaries:
            if self._laygo_size is None:
                raise ValueError('laygo_size must be set before drawing boundaries.')

            end_mode = self._laygo_info.end_mode
            xr = self.bound_box.right_unit

            left_end = (end_mode & 4) != 0
            right_end = (end_mode & 8) != 0

            edge_infos = []
            # compute extension edge information
            for y, orient, edge_params in self._ext_edge_infos:
                tmp_copy = edge_params.copy()
                if orient == 'R0':
                    x = 0
                    tmp_copy['is_end'] = left_end
                else:
                    x = xr
                    tmp_copy['is_end'] = right_end
                edge_infos.append((x, y, orient, tmp_copy))

            # compute row edge information
            row_edge_infos = self._get_row_edge_infos()
            for ridx, (y, orient, re_params) in enumerate(row_edge_infos):
                endl, endr = self._get_end_info_row(ridx)
                for x, is_end, flip_lr, end_flag in ((0, left_end, False, endl), (xr, right_end, True, endr)):
                    edge_info = self._tech_cls.get_laygo_edge_info(re_params['row_info'], end_flag)
                    edge_params = re_params.copy()
                    del edge_params['row_info']
                    edge_params['is_end'] = is_end
                    edge_params['name_id'] = edge_info['name_id']
                    edge_params['layout_info'] = edge_info
                    if flip_lr:
                        eorient = 'MY' if orient == 'R0' else 'R180'
                    else:
                        eorient = orient
                    edge_infos.append((x, y, eorient, edge_params))

            yt = self.bound_box.top_unit
            vdd_warrs, vss_warrs = self._tech_cls.draw_boundaries(self, self._laygo_info, self._laygo_size[0],
                                                                  yt, self._bot_end_master, self._top_end_master,
                                                                  edge_infos)

            return vdd_warrs, vss_warrs

        return [], []
