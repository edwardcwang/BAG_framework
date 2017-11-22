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

from typing import Optional, Union, List, Tuple, Iterable, Any, Dict

import numpy as np

from bag.util.interval import IntervalSet
from bag.util.search import BinaryIterator
from bag.layout.util import BBox
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
        # type: () -> Iterable[int]
        return self._tracks.keys()

    def items(self):
        # type: () -> Iterable[int, IntervalSet]
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
        for hidx, new_intv_set in track_set.items():
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

    def layer_track_set_iter(self):
        return self._track_sets.items()

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

    def transform(self, grid, bot_layer, top_layer, loc=(0, 0), orient='R0', unit_mode=False):
        # type: (RoutingGrid, int, int, Tuple[Union[float, int], Union[float, int]], str, bool) -> UsedTracks
        """Return a new transformed UsedTracks on the given layers.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        bot_layer : int
            the bottom layer ID, inclusive.
        top_layer : int
            the top layer ID, inclusive.
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
            if bot_layer <= layer_id <= top_layer:
                new_track_sets[layer_id] = track_set.transform(grid, layer_id, dx, dy, orient=orient)

        return UsedTracks(self._res, init_track_sets=new_track_sets)

    def merge(self, used_tracks):
        # type: (UsedTracks) -> None
        """Merge the given used tracks to this one."""
        for layer_id, new_track_set in used_tracks.layer_track_set_iter():
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
    layer_name = tech_info.get_layer_name(layer_id)
    if isinstance(layer_name, tuple) or isinstance(layer_name, list):
        layer_name = layer_name[0]
    layer_type = tech_info.get_layer_type(layer_name)

    # subtract used tracks.
    for hidx, intv_set in track_set.items():
        for (wstart, wstop), (wwidth, (fmargin, fill_type)) in intv_set.items():
            cbeg, cend = grid.get_wire_bounds(layer_id, (hidx - 1) / 2, width=wwidth, unit_mode=True)
            min_space = tech_info.get_min_space(layer_type, cend - cbeg, unit_mode=True)
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
                          bnd_box,  # type: BBox
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
    blk_width, blk_height = bnd_box.width_unit, bnd_box.height_unit
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


def fill_symmetric_const_space(area, sp_max, n_min, n_max, offset=0):
    # type: (int, int, int, int, int) -> List[Tuple[int, int]]
    """Fill the given 1-D area given maximum space spec alone.

    The method draws the minimum number of fill blocks needed to satisfy maximum spacing spec.
    The given area is filled with the following properties:

    1. all spaces are as close to the given space as possible (differ by at most 1), without exceeding it.
    2. the filled area is as uniform as possible.
    3. the filled area is symmetric about the center.
    4. fill is drawn as much as possible given the above constraints.

    fill is drawn such that space blocks abuts both area boundaries.

    Parameters
    ----------
    area : int
        the 1-D area to fill.
    sp_max : int
        the maximum space.
    n_min : int
        minimum fill length.
    n_max : int
        maximum fill length
    offset : int
        the fill area starting coordinate.

    Returns
    -------
    fill_intv : List[Tuple[int, int]]
        list of fill intervals.
    """
    if n_min > n_max:
        raise ValueError('min fill length = %d > %d = max fill length' % (n_min, n_max))

    # suppose we draw N fill blocks, then the filled area is A - (N + 1) * sp.
    # therefore, to maximize fill, with A and sp given, we need to minimize N.
    # since N = (A - sp) / (f + sp), where f is length of the fill, this tells
    # us we want to try filling with max block.
    # so we calculate the maximum number of fill blocks we'll use if we use
    # largest fill block.
    num_fill = -(-(area - sp_max) // (n_max + sp_max))
    if num_fill == 0:
        # we don't need fill; total area is less than sp_max.
        return []

    # at this point, using (num_fill - 1) max blocks is not enough, but num_fill
    # max blocks either fits perfectly or exceeds area.

    # calculate the fill block length if we use num_fill fill blocks, and sp_max
    # between blocks.
    blk_len = (area - (num_fill + 1) * sp_max) // num_fill
    if blk_len >= n_min:
        # we can draw fill using num_fill fill blocks.
        return fill_symmetric_helper(area, num_fill, sp_max, offset=offset, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)[0]

    # trying to draw num_fill fill blocks with sp_max between them results in fill blocks
    # that are too small.  This means we need to reduce the space between fill blocks.
    sp_max, remainder = divmod(area - num_fill * n_min, num_fill + 1)
    # we can achieve the new sp_max using fill with length n_min or n_min + 1.
    if n_max > n_min or remainder == 0:
        # if everything divides evenly or we can use two different fill lengths,
        # then we're done.
        return fill_symmetric_helper(area, num_fill, sp_max, offset=offset, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)[0]
    # If we're here, then we must use only one fill length
    # fill by inverting fill/space to try to get only one fill length
    sol, same_sp = fill_symmetric_helper(area, num_fill + 1, n_max, offset=offset, inc_sp=False,
                                         invert=True, fill_on_edge=True, cyclic=False)
    if same_sp:
        # we manage to fill using only one fill length
        return sol

    # If we're here, that means num_fill + 1 is even.  So using num_fill + 2 will
    # guarantee solution.
    return fill_symmetric_helper(area, num_fill + 2, n_max, offset=offset, inc_sp=False,
                                 invert=True, fill_on_edge=True, cyclic=False)[0]


def fill_symmetric_max_info(area, n_min, n_max, sp_min, cyclic=False):
    # type: (int, int, int, int, bool) -> Tuple[Any, ...]
    """Fill the given 1-D area as much as possible, given minimum space constraint.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. the area is filled as much as possible.

    fill is drawn such that fill blocks abut both area boundaries.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    n_min : int
        minimum length of the fill block.  Must be less than n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    cyclic : bool
        True if the given area actually wraps around.  This is usually the case if you are
        filling an area that is arrayed.  In cyclic fill mode, space blocks abut both area
        boundaries.
    """
    # error checking
    if n_min > n_max:
        raise ValueError('n_min = %d > %d = n_max' % (n_min, n_max))

    # step 1: find maximum block size and minimum number of blocks we can put in the given area
    blk_len_max_iter = BinaryIterator(n_min, n_max + 1)
    while blk_len_max_iter.has_next():
        blk_len_max = blk_len_max_iter.get_next()
        if cyclic:
            num_blk_min = area // (blk_len_max + sp_min)
            num_sp_min = num_blk_min
        else:
            num_blk_min = (area + sp_min) // (blk_len_max + sp_min)
            num_sp_min = num_blk_min - 1

        if num_blk_min > 0 and (not cyclic or num_sp_min > 0):
            blk_len_max_iter.save_info((num_blk_min, num_sp_min))
            blk_len_max_iter.up()
        else:
            blk_len_max_iter.down()

    blk_len_max = blk_len_max_iter.get_last_save()
    if blk_len_max is None:
        # special case: we cannot draw any fill at all
        return _fill_symmetric_info(area, 0, area, inc_sp=True, fill_on_edge=False, cyclic=False), False
    num_blk_min, num_sp_min = blk_len_max_iter.get_last_save_info()

    # step 2: compute the amount of space if we use minimum number of blocks
    min_space_with_max_blk = area - num_blk_min * blk_len_max
    # step 3: determine number of blocks to use
    # step 3A: check if we can and should use num_blk_min fill blocks.
    # first, we can use num_blk_min fill blocks if 1) we have at least 1 space
    # 2) if not, we can just fill entire area
    # second, we should use num_blk_min fill blocks if this means we have less space
    # compared to using (num_blk_min + 1) fill blocks.
    if (num_sp_min > 0 or blk_len_max == area) and min_space_with_max_blk <= (num_sp_min + 1) * sp_min:
        # If we're here, we can achieve the minimum amount of space by using all large blocks,
        # now we just need to place the blocks in a symmetric way.

        # since all blocks has width blk_len_max, we will try to distribute empty space
        # between blocks evenly symmetrically.
        inc_sp = blk_len_max < n_max
        info = _fill_symmetric_info(area, num_sp_min, blk_len_max, inc_sp=inc_sp,
                                    fill_on_edge=cyclic, cyclic=cyclic)
        same_sp = info[1][2]
        if n_min == n_max and not same_sp:
            # we get here only if number of fill blocks is odd, and
            # the middle fill block is forced to have length n_min - 1.
            # in this case, we need to remove one fill block
            info = _fill_symmetric_info(area, num_sp_min - 1, blk_len_max, inc_sp=inc_sp,
                                        fill_on_edge=cyclic, cyclic=cyclic)
        return info, True

    # If we're here, we need to use num_blk_min + 1 number of fill blocks, and we can achieve
    # a minimum space of (num_sp_min + 1) * sp.  Now we need to determine the size of each fill block
    # and place them symmetrically.
    min_blk_len = (area - (num_sp_min + 1) * sp_min) // (num_blk_min + 1)
    if min_blk_len < n_min:
        # TODO: deal with this later
        raise ValueError("No fill solution found")

    return _fill_symmetric_info(area, num_blk_min + 1, sp_min, inc_sp=True,
                                fill_on_edge=not cyclic, cyclic=cyclic), False


def fill_symmetric_max(area, n_min, n_max, sp_min, offset=0, cyclic=False):
    # type: (int, int, int, int, int, bool) -> List[Tuple[int, int]]
    """Fill the given 1-D area as much as possible, given minimum space constraint.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. the area is filled as much as possible.

    fill is drawn such that fill blocks abut both area boundaries.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    n_min : int
        minimum length of the fill block.  Must be less than n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    offset : int
        the area starting coordinate.
    cyclic : bool
        True if the given area actually wraps around.  This is usually the case if you are
        filling an area that is arrayed.  In cyclic fill mode, space blocks abut both area
        boundaries.

    Returns
    -------
    fill_interval : List[Tuple[int, int]]
        a list of [start, stop) intervals that needs to be filled.
    """
    (_, args), invert = fill_symmetric_max_info(area, n_min, n_max, sp_min, cyclic=cyclic)
    return _fill_symmetric_interval(*args, offset=offset, invert=invert)[0]


def _fill_symmetric_info(tot_area, num_blk_tot, sp, inc_sp=True, fill_on_edge=True, cyclic=False):
    # type: (int, int, int, bool, bool, bool) -> Tuple[int, Tuple[Any, ...]]
    """Helper method for all fill symmetric methods.

    This method fills an area with given number of fill blocks such that the space between
    blocks is equal to the given space.  Other fill_symmetric methods basically transpose
    the constraints into this problem, with the proper options.

    The solution has the following properties:

    1. it is symmetric about the center.
    2. it is as uniform as possible.
    3. it uses at most 2 values of fill lengths, and they differ by 1.
    4. it uses at most 2 values of space lengths, controlled by sp and inc_sp parameters.

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.
    inc_sp : bool
        when num_blk_tot is 1 or even, it may not be possible to keep the space between
        every block to be the same as given space, and the middle space will need to
        increment/decrement by 1.  If inc_sp is True, we will increase the middle space
        length by 1.  Otherwise, we will decrease it by 1.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    fill_area : int
        total filled area.
    args : Tuple[Any, ...]
        input arguments to _fill_symmetric_interval()
    """
    # error checking
    if num_blk_tot < 0:
        raise ValueError('num_blk_tot = %d < 0' % num_blk_tot)

    adj_sp_sgn = 1 if inc_sp else -1
    if num_blk_tot == 0:
        # special case, no fill at all
        if sp == tot_area:
            return 0, (tot_area, tot_area, 0, 0, 0, 0, 0, -1, tot_area, False, False)
        elif sp == tot_area - adj_sp_sgn:
            return 0, (tot_area, tot_area, 1, 0, 0, 0, 0, -1, tot_area, False, False)
        else:
            raise ValueError('Cannot draw 0 fill with given spec')

    # determine the number of space blocks
    if cyclic:
        num_sp_tot = num_blk_tot
    else:
        if fill_on_edge:
            num_sp_tot = num_blk_tot - 1
            if num_sp_tot == 0 and sp != 0:
                raise ValueError('Cannot draw 0 spaces blocks with nonzero spacing.')
        else:
            num_sp_tot = num_blk_tot + 1

    # compute total fill area
    fill_area = tot_area - num_sp_tot * sp

    # find minimum fill length
    blk_len, num_blk1 = divmod(fill_area, num_blk_tot)
    # find number of fill intervals
    if cyclic and fill_on_edge:
        # if cyclic and fill on edge, number of intervals = number of blocks + 1,
        # because the interval on the edge double counts.
        num_blk_interval = num_blk_tot + 1
    else:
        num_blk_interval = num_blk_tot

    # find space length on edge, if applicable
    num_diff_sp = 0
    sp_edge = sp
    if cyclic and not fill_on_edge and sp_edge % 2 == 1:
        # edge space must be even.  To fix, we convert space to fill
        num_diff_sp += 1
        sp_edge += adj_sp_sgn
        num_blk1 += -adj_sp_sgn
        fill_area += -adj_sp_sgn
        if num_blk1 == num_blk_tot:
            blk_len += 1
            num_blk1 = 0
        elif num_blk1 < 0:
            blk_len -= 1
            num_blk1 += num_blk_tot

    mid_blk_len = mid_sp_len = -1
    # now we have num_blk_tot blocks with length blk0.  We have num_blk1 fill units
    # remaining that we need to distribute to the fill blocks
    if num_blk_interval % 2 == 0:
        # we have even number of fill intervals, so we have a space block in the middle
        mid_sp_len = sp
        # test condition for cyclic and fill_on_edge is different than other cases
        test_val = num_blk1 + blk_len if cyclic and fill_on_edge else num_blk1
        if test_val % 2 == 1:
            # we cannot distribute remaining fill units evenly, have to convert to space
            num_diff_sp += 1
            mid_sp_len += adj_sp_sgn
            num_blk1 += -adj_sp_sgn
            fill_area += -adj_sp_sgn
            if num_blk1 == num_blk_tot:
                blk_len += 1
                num_blk1 = 0
            elif num_blk1 < 0:
                blk_len -= 1
                num_blk1 += num_blk_tot
        if num_blk1 % 2 == 1:
            # the only way we get here is if cyclic and fill_on_edge is True.
            # in this case, we need to add one to fill unit to account for edge fill double counting.
            num_blk1 += 1

        # get number of half fill intervals
        m = num_blk_interval // 2
    else:
        # we have odd number of fill intervals, so we have a fill block in the middle
        mid_blk_len = blk_len
        if cyclic and fill_on_edge:
            # special handling for this case, because edge fill block must be even
            if blk_len % 2 == 0 and num_blk1 % 2 == 1:
                # assign one fill unit to middle block
                mid_blk_len += 1
                num_blk1 -= 1
            elif blk_len % 2 == 1:
                # edge fill block is odd; we need odd number of fill units so we can
                # correct this.
                if num_blk1 % 2 == 0:
                    # we increment middle fill block to get odd number of fill units
                    mid_blk_len += 1
                    num_blk1 -= 1
                    if num_blk1 < 0:
                        # we get here only if num_blk1 == 0.  This means middle blk
                        # borrow one unit from edge block.  So we set num_blk1 to
                        # num_blk_tot - 2 to make sure rest of the blocks are one
                        # larger than edge block.
                        blk_len -= 1
                        num_blk1 = num_blk_tot - 2
                    else:
                        # Add one to account for edge fill double counting.
                        num_blk1 += 1
                else:
                    # Add one to account for edge fill double counting.
                    num_blk1 += 1
        elif num_blk1 % 2 == 1:
            # assign one fill unit to middle block
            mid_blk_len += 1
            num_blk1 -= 1

        m = (num_blk_interval - 1) // 2

    if blk_len <= 0:
        raise ValueError('Cannot fill with block less <= 0.')

    # now we need to distribute the fill units evenly.  We do so using cumulative modding
    num_large = num_blk1 // 2
    num_small = m - num_large
    if cyclic and fill_on_edge:
        # if cyclic and fill is on the edge, we need to make sure left-most block is even length
        if blk_len % 2 == 0:
            blk1, blk0 = blk_len, blk_len + 1
            k = num_small
        else:
            blk0, blk1 = blk_len, blk_len + 1
            k = num_large
    else:
        # make left-most fill interval be the most frequenct fill length
        if num_large >= num_small:
            blk0, blk1 = blk_len, blk_len + 1
            k = num_large
        else:
            blk1, blk0 = blk_len, blk_len + 1
            k = num_small

    return fill_area, (tot_area, sp, num_diff_sp, sp_edge, blk0, blk1, k, m,
                       mid_blk_len, mid_sp_len, fill_on_edge, cyclic)


def _fill_symmetric_interval(tot_area, sp, num_diff_sp, sp_edge, blk0, blk1, k, m, mid_blk_len, mid_sp_len,
                             fill_on_edge, cyclic, offset=0, invert=False):
    """Helper function, construct interval list.

    num_diff_sp = number of space blocks that has length different than sp
    sp_edge = if cyclic and not fill on edge, the edge space length.
    m = number of half fill blocks.
    blk1 = length of left-most fill block.
    blk0 = the second possible fill block length.
    k = number of half fill blocks with length = blk1.
    mid_blk_len = if > 0, length of middle fill block.  This is either blk0 or blk1.
    """
    ans = []
    if cyclic:
        if fill_on_edge:
            marker = offset - blk1 // 2
        else:
            marker = offset - sp_edge // 2
    else:
        marker = offset
    cur_sum = 0
    prev_sum = 1
    for fill_idx in range(m):
        # determine current fill length from cumulative modding result
        if cur_sum <= prev_sum:
            cur_len = blk1
        else:
            cur_len = blk0

        cur_sp = sp_edge if fill_idx == 0 else sp
        # record fill/space interval
        if invert:
            if fill_on_edge:
                ans.append((marker + cur_len, marker + cur_sp + cur_len))
            else:
                ans.append((marker, marker + cur_sp))
        else:
            if fill_on_edge:
                ans.append((marker, marker + cur_len))
            else:
                ans.append((marker + cur_sp, marker + cur_sp + cur_len))

        marker += cur_len + cur_sp
        prev_sum = cur_sum
        cur_sum = (cur_sum + k) % m

    # add middle fill or space
    if mid_blk_len >= 0:
        # fill in middle
        if invert:
            if not fill_on_edge:
                # we have one more space block before reaching middle block
                cur_sp = sp_edge if m == 0 else sp
                ans.append((marker, marker + cur_sp))
            half_len = len(ans)
        else:
            # we don't want to replicate middle fill, so get half length now
            half_len = len(ans)
            if fill_on_edge:
                ans.append((marker, marker + mid_blk_len))
            else:
                cur_sp = sp_edge if m == 0 else sp
                ans.append((marker + cur_sp, marker + cur_sp + mid_blk_len))
    else:
        # space in middle
        if invert:
            if fill_on_edge:
                # the last space we added is wrong, we need to remove
                del ans[-1]
                marker -= sp
            # we don't want to replicate middle space, so get half length now
            half_len = len(ans)
            ans.append((marker, marker + mid_sp_len))
        else:
            # don't need to do anything if we're recording blocks
            half_len = len(ans)

    # now add the second half of the list
    shift = tot_area + offset * 2
    for idx in range(half_len - 1, -1, -1):
        start, stop = ans[idx]
        ans.append((shift - stop, shift - start))

    return ans, num_diff_sp


def fill_symmetric_helper(tot_area, num_blk_tot, sp, offset=0, inc_sp=True, invert=False,
                          fill_on_edge=True, cyclic=False):
    # type: (int, int, int, int, bool, bool, bool, bool) -> Tuple[List[Tuple[int, int]], int]
    """Helper method for all fill symmetric methods.

    This method fills an area with given number of fill blocks such that the space between
    blocks is equal to the given space.  Other fill_symmetric methods basically transpose
    the constraints into this problem, with the proper options.

    The solution has the following properties:

    1. it is symmetric about the center.
    2. it is as uniform as possible.
    3. it uses at most 2 values of fill lengths, and they differ by 1.
    4. it uses at most 2 values of space lengths, controlled by sp and inc_sp parameters.

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.
    offset : int
        the starting coordinate of the area interval.
    inc_sp : bool
        when num_blk_tot is 1 or even, it may not be possible to keep the space between
        every block to be the same as given space, and the middle space will need to
        increment/decrement by 1.  If inc_sp is True, we will increase the middle space
        length by 1.  Otherwise, we will decrease it by 1.
    invert : bool
        If True, we return space intervals instead of fill intervals.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    ans : List[(int, int)]
        list of fill or space intervals.
    num_diff_sp : int
        number of space intervals with length different than sp.  If cyclic and not fill_on_edge,
        then this is at most 2 (edge space and middle space).  Otherwise, this is at most 1
        (middle space).
    """
    fill_info = _fill_symmetric_info(tot_area, num_blk_tot, sp, inc_sp=inc_sp,
                                     fill_on_edge=fill_on_edge, cyclic=cyclic)

    _, args = fill_info
    return _fill_symmetric_interval(*args, offset=offset, invert=invert)
