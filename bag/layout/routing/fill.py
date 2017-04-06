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

"""This module defines classes that provides automatic fill utility on a grid.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Optional, Union, List, Tuple, Generator, Any, Dict

import numpy as np

from bag.util.interval import IntervalSet
from .base import WireArray, TrackID
from .grid import RoutingGrid


class TrackSet(object):
    """A data structure that stored tracks on the same layer.

    Parameters
    ----------
    min_length : int
        Make sure all stored track has at least min_length.
    init_tracks : Optional[Dict[int, IntervalSet]]
        Dictionary of initial tracks.
    """
    def __init__(self, min_length=0, init_tracks=None):
        # type: (float, Optional[Dict[int, IntervalSet]]) -> None
        if init_tracks is None:
            init_tracks = {}  # type: Dict[int, IntervalSet]
        else:
            pass
        self._tracks = init_tracks
        self._min_len = min_length

    def __contains__(self, item):
        # type: (int) -> bool
        return item in self._tracks

    def __getitem__(self, item):
        # type: (int) -> IntervalSet
        return self._tracks[item]

    def __len__(self):
        return len(self._tracks)

    def __iter__(self):
        return iter(self._tracks)

    def keys(self):
        # type: () -> Generator[int]
        return self._tracks.keys()

    def items(self):
        # type: () -> Generator[Tuple[int, IntervalSet]]
        return self._tracks.items()

    def subtract(self, hidx, intv):
        # type: (int, Tuple[int, int]) -> None
        """Subtract the given intervals from this TrackSet."""
        if hidx in self._tracks:
            intv_set = self._tracks[hidx]
            new_intvs = intv_set.subtract(intv)
            # delete intervals smaller than minimum length.
            for intv in new_intvs:
                if intv[1] - intv[0] < self._min_len:
                    intv_set.remove(intv)
            if not intv_set:
                del self._tracks[hidx]

    def add_track(self, hidx, intv, width, value=None):
        # type: (int, Tuple[int, int], int, Any) -> None
        """Add tracks to this data structure.

        Parameters
        ----------
        hidx : int
            the half track index.
        intv : Tuple[int, int]
            the track interval.
        width : int
            the track width.
        value : Any
            value associated with this track.
        """
        if intv[1] - intv[0] >= self._min_len:
            if hidx not in self._tracks:
                intv_set = IntervalSet()
                self._tracks[hidx] = intv_set
            else:
                intv_set = self._tracks[hidx]

            # TODO: add more robust checking?
            intv_set.add(intv, val=[width, value], merge=True)

    def transform(self, grid, layer_id, dx, dy, orient='R0'):
        # type: (RoutingGrid, int, int, int, str) -> TrackSet
        """Return a new transformed TrackSet.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        layer_id : int
            the layer ID of this TrackSet.
        dx : int
            the X shift in resolution units.
        dy : int
            the Y shift in resolution units.
        orient : str
            the new orientation.

        Returns
        -------
        result : TrackSet
            the new TrackSet.
        """
        is_x = grid.get_direction(layer_id) == 'x'
        if is_x:
            hidx_shift = int(2 * grid.coord_to_track(layer_id, dy, unit_mode=True)) + 1
            intv_shift = dx
        else:
            hidx_shift = int(2 * grid.coord_to_track(layer_id, dx, unit_mode=True)) + 1
            intv_shift = dy

        hidx_scale = intv_scale = 1
        if orient == 'R180':
            hidx_scale = -1
            intv_scale = -1
        elif orient == 'MX':
            if is_x:
                hidx_scale = -1
            else:
                intv_scale = -1
        elif orient == 'MY':
            if is_x:
                intv_scale = -1
            else:
                hidx_scale = -1

        new_tracks = {}
        for hidx, intv_set in self._tracks.items():
            new_tracks[hidx * hidx_scale + hidx_shift] = intv_set.transform(intv_scale, intv_shift)

        return TrackSet(min_length=self._min_len, init_tracks=new_tracks)

    def merge(self, track_set):
        # type: (TrackSet) -> None
        """Merge the given TrackSet to this one."""
        for hidx, new_intv_set in track_set._tracks.items():
            if hidx not in self._tracks:
                intv_set = IntervalSet()
                self._tracks[hidx] = intv_set
            else:
                intv_set = self._tracks[hidx]

            for intv, val in new_intv_set.items():
                intv_set.add(intv, val, merge=True)


class UsedTracks(object):
    """A data structure that stores used tracks on the routing grid.

    Parameters
    ----------
    resolution : float
        the layout resolution.
    init_track_sets : Optional[Dict[int, TrackSet]]
        Dictionary of initial TrackSets.
    """

    def __init__(self, resolution, init_track_sets=None):
        # type: (float, Dict[int, Optional[Dict[int, TrackSet]]]) -> None
        if init_track_sets is None:
            init_track_sets = {}  # type: Dict[int, TrackSet]
        else:
            pass
        self._track_sets = init_track_sets
        self._res = resolution

    def get_tracks_info(self, layer_id):
        # type: (int) -> TrackSet
        """Returns used tracks information on the given layer.

        Parameters
        ----------
        layer_id : int
            the layer ID.

        Returns
        -------
        tracks_info : TrackSet
            the used tracks on the given layer.
        """
        if layer_id not in self._track_sets:
            self._track_sets[layer_id] = TrackSet()
        return self._track_sets[layer_id]

    def add_wire_arrays(self, warr_list, fill_margin=0, fill_type='VSS', unit_mode=False):
        # type: (Union[WireArray, List[WireArray]], Union[float, int], str, bool) -> None
        """Adds a wire array to this data structure.

        Parameters
        ----------
        warr_list : Union[WireArray, List[WireArray]]
            the WireArrays to add.
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode : bool
            True if fill_margin is given in resolution units.
        """
        if isinstance(warr_list, WireArray):
            warr_list = [warr_list, ]
        else:
            pass

        if not unit_mode:
            fill_margin = int(round(fill_margin / self._res))

        for warr in warr_list:
            warr_tid = warr.track_id
            layer_id = warr_tid.layer_id
            width = warr_tid.width
            if layer_id not in self._track_sets:
                track_set = TrackSet()
                self._track_sets[layer_id] = track_set
            else:
                track_set = self._track_sets[layer_id]

            intv = (int(round(warr.lower / self._res)), int(round(warr.upper / self._res)))
            base_hidx = int(round(2 * warr_tid.base_index + 1))
            step = int(round(warr_tid.pitch * 2))
            for idx in range(warr_tid.num):
                hidx = base_hidx + idx * step
                track_set.add_track(hidx, intv, width, value=(fill_margin, fill_type))

    def transform(self, grid, loc=(0, 0), orient='R0', unit_mode=False):
        # type: (RoutingGrid, Tuple[Union[float, int], Union[float, int]], str, bool) -> UsedTracks
        """Return a new transformed UsedTracks.

        Parameters
        ----------
        grid : :RoutingGrid
            the RoutingGrid object.
        loc : Tuple[Union[float, int], Union[float, int]]
            the X/Y coordinate shift.
        orient : str
            the new orientation.
        unit_mode : bool
            True if loc is given in resolution units.
        """
        if not unit_mode:
            res = grid.resolution
            dx, dy = int(round(loc[0] / res)), int(round(loc[1] / res))
        else:
            dx, dy = loc

        new_track_sets = {}
        for layer_id, track_set in self._track_sets.items():
            new_track_sets[layer_id] = track_set.transform(grid, layer_id, dx, dy, orient=orient)

        return UsedTracks(self._res, init_track_sets=new_track_sets)

    def merge(self, used_tracks, layers):
        # type: (UsedTracks) -> None
        """Merge the given used tracks to this one."""
        for layer_id, new_track_set in used_tracks._track_sets.items():
            if layer_id in layers:
                if layer_id not in self._track_sets:
                    track_set = TrackSet()
                    self._track_sets[layer_id] = track_set
                else:
                    track_set = self._track_sets[layer_id]

                track_set.merge(new_track_set)


def get_available_tracks(grid,  # type: RoutingGrid
                         layer_id,  # type: int
                         tr_idx_list,  # type: List[int]
                         lower,  # type: int
                         upper,  # type: int
                         width,  # type: int
                         margin,  # type: int
                         track_set,  # type: TrackSet
                         ):
    # type: () -> List[int]
    """Fill unused tracks with supply tracks.
    """
    avail_track_set = TrackSet(min_length=upper - lower)
    for tidx in tr_idx_list:
        avail_track_set.add_track(2 * tidx + 1, (lower, upper), width, value=False)

    tech_info = grid.tech_info
    res = grid.resolution
    layer_name = tech_info.get_layer_name(layer_id)
    if isinstance(layer_name, tuple) or isinstance(layer_name, list):
        layer_name = layer_name[0]
    layer_type = tech_info.get_layer_type(layer_name)

    # subtract used tracks.
    for hidx, intv_set in track_set.items():
        for (wstart, wstop), (wwidth, (fmargin, fill_type)) in intv_set.items():
            cbeg, cend = grid.get_wire_bounds(layer_id, (hidx - 1) / 2, width=wwidth, unit_mode=True)
            min_space = tech_info.get_min_space(layer_type, (cend - cbeg) * res)
            min_space = int(round(min_space / res))
            fmargin = max(margin, fmargin, min_space)

            sub_intv = (wstart - fmargin, wstop + fmargin)
            idx0, idx1 = grid.get_overlap_tracks(layer_id, cbeg - fmargin, cend + fmargin,
                                                 half_track=True, unit_mode=True)
            hidx0 = int(round(2 * idx0 + 1)) - 2 * (width - 1)
            hidx1 = int(round(2 * idx1 + 1)) + 2 * (width - 1)

            # substract
            for sub_idx in range(hidx0, hidx1 + 1):
                avail_track_set.subtract(sub_idx, sub_intv)

    # return available tracks
    hidx_arr = np.array(sorted(avail_track_set.keys()), dtype=int)
    ans = ((hidx_arr - 1) // 2).tolist()  # type: List[int]
    return ans


def get_power_fill_tracks(grid,  # type: RoutingGrid
                          size,  # type: Tuple[int, int, int]
                          layer_id,  # type: int
                          track_set,  # type: TrackSet
                          sup_width,  # type: int
                          fill_margin,  # type: int
                          edge_margin,  # type: int
                          sup_spacing=-1,  # type: int
                          debug=False,  # type: bool
                          ):
    # type: () -> Tuple[List[WireArray], List[WireArray]]
    """Fill unused tracks with supply tracks.
    """
    # get block size and lower/upper coordinates.
    blk_width, blk_height = grid.get_size_dimension(size, unit_mode=True)
    lower = edge_margin
    if grid.get_direction(layer_id) == 'x':
        upper = blk_width - edge_margin
        cupper = blk_height
    else:
        upper = blk_height - edge_margin
        cupper = blk_width

    # find fill track indices in half tracks
    num_space = grid.get_num_space_tracks(layer_id, sup_width, half_space=False)
    # check if user specify supply spacing
    if sup_spacing >= 0:
        if sup_spacing < num_space:
            raise ValueError('Power fill spacing less then min spacing = %d' % num_space)
        num_space = sup_spacing

    start_tidx, end_tidx = grid.get_track_index_range(layer_id, 0, cupper, num_space=num_space,
                                                      edge_margin=edge_margin, half_track=False,
                                                      unit_mode=True)

    first_hidx = start_tidx * 2 + 1 + sup_width - 1
    last_hidx = end_tidx * 2 + 1 - (sup_width - 1)
    fill_hidx_list = list(range(first_hidx, last_hidx + 1, 2 * (sup_width + num_space)))

    # add all fill tracks
    min_length = grid.get_min_length(layer_id, sup_width, unit_mode=True)
    fill_track_set = TrackSet(min_length=min_length)
    fill_intv = (lower, upper)
    for hidx in fill_hidx_list:
        fill_track_set.add_track(hidx, fill_intv, sup_width, value=False)

    max_fill_hidx = fill_hidx_list[-1]
    # subtract used tracks from fill.
    sup_type = {}
    for hidx, intv_set in track_set.items():
        for (wstart, wstop), (wwidth, (fmargin, fill_type)) in intv_set.items():
            fmargin = max(fill_margin, fmargin)
            sub_intv = (wstart - fmargin, wstop + fmargin)
            cbeg, cend = grid.get_wire_bounds(layer_id, (hidx - 1) / 2, width=wwidth, unit_mode=True)
            idx0, idx1 = grid.get_overlap_tracks(layer_id, cbeg - fmargin, cend + fmargin,
                                                 half_track=True, unit_mode=True)
            hidx0 = int(round(2 * idx0 + 1)) - 2 * (sup_width - 1)
            hidx1 = int(round(2 * idx1 + 1)) + 2 * (sup_width - 1)
            if debug:
                print('Found track: hidx = %d, intv = (%d, %d), fill_type = %s' % (hidx, wstart, wstop, fill_type))
                print('deleting fill in hidx range (inclusive): (%d, %d)' % (hidx0, hidx1))

            # substract fill
            for sub_idx in range(hidx0, hidx1 + 1):
                fill_track_set.subtract(sub_idx, sub_intv)
                # TODO: more robust error/warning messages?
                if sub_idx not in sup_type and (fill_type == 'VDD' or fill_type == 'VSS'):
                    sup_type[sub_idx] = fill_type
                    if debug:
                        print('assigning hidx %d fill type %s' % (sub_idx, fill_type))
            # assign adjacent fill tracks to fill_type
            if fill_type == 'VDD' or fill_type == 'VSS':
                for sub_idx in range(hidx0 - 1, -1, -1):
                    if sub_idx in fill_track_set:
                        if sub_idx not in sup_type:
                            # TODO: more robust error/warning messages?
                            sup_type[sub_idx] = fill_type
                            if debug:
                                print('assigning hidx %d fill type %s' % (sub_idx, fill_type))
                        break
                for sub_idx in range(hidx1 + 1, max_fill_hidx + 1):
                    if sub_idx in fill_track_set:
                        if sub_idx not in sup_type:
                            # TODO: more robust error/warning messages?
                            sup_type[sub_idx] = fill_type
                            if debug:
                                print('assigning hidx %d fill type %s' % (sub_idx, fill_type))
                        break

    # count remaining fill tracks
    fill_hidx_list = sorted(fill_track_set.keys())
    tot_cnt = len(fill_hidx_list)
    vdd_cnt = 0
    vss_cnt = 0
    for hidx in fill_hidx_list:
        cur_type = sup_type.get(hidx, None)
        if cur_type == 'VDD':
            vdd_cnt += 1
        elif cur_type == 'VSS':
            vss_cnt += 1

    # assign tracks to VDD/VSS
    num_vdd = tot_cnt // 2
    num_vss = tot_cnt - num_vdd
    remaining_tot = tot_cnt - vdd_cnt - vss_cnt
    remaining_vss = max(num_vss - vss_cnt, 0)
    remaining_vdd = remaining_tot - remaining_vss
    # uniformly distribute vdd tracks among remaining tracks
    k = 0
    next_vdd = ((2 * k + 1) * remaining_tot + remaining_vdd) // (2 * remaining_vdd)
    cur_idx = 0
    res = grid.resolution
    vdd_warr_list = []
    vss_warr_list = []
    for hidx in fill_hidx_list:
        if debug:
            print('creating fill at hidx %d' % hidx)
        # get supply type
        if hidx not in sup_type:
            if cur_idx == next_vdd:
                cur_type = 'VDD'
                k += 1
                next_vdd = ((2 * k + 1) * remaining_tot + remaining_vdd) // (2 * remaining_vdd)
            else:
                cur_type = 'VSS'
            cur_idx += 1
            if debug:
                print('hidx %d, unassigned, pick fill type = %s' % (hidx, cur_type))
        else:
            cur_type = sup_type[hidx]
            if debug:
                print('hidx %d, assigned fill type = %s' % (hidx, cur_type))

        w_list = vdd_warr_list if cur_type == 'VDD' else vss_warr_list
        tid = TrackID(layer_id, (hidx - 1) / 2, width=sup_width)
        w_list.extend(WireArray(tid, intv[0] * res, intv[1] * res)
                      for intv in fill_track_set[hidx].intervals())

    return vdd_warr_list, vss_warr_list
