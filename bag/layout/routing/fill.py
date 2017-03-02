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

from typing import Union, List, Tuple, Dict

from bag.util.interval import IntervalSet
from .base import WireArray
from .grid import RoutingGrid


class UsedTracks(object):
    """A data structure that stores used tracked on the routing grid.

    Parameters
    ----------
    resolution : float
        the layout resolution.
    """

    def __init__(self, resolution):
        # type: (float) -> None
        self._tracks = {}
        self._res = resolution

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
            if layer_id not in self._tracks:
                track_table = {}
                self._tracks[layer_id] = track_table
            else:
                track_table = self._tracks[layer_id]

            intv = (int(round(warr.lower / self._res)), int(round(warr.upper / self._res)))
            for idx in warr_tid:
                hidx = int(round(2 * idx + 1))

                if hidx not in track_table:
                    intv_set = IntervalSet()
                    track_table[hidx] = intv_set
                else:
                    intv_set = track_table[hidx]

                # TODO: add more robust checking?
                intv_set.add(intv, val=(width, fill_margin, fill_type), merge=True)


def do_power_fill(grid,  # type: RoutingGrid
                  size,  # type: Tuple[int, int, int]
                  layer_id,  # type: int
                  track_table,  # type: Dict[int, IntervalSet]
                  vdd_warr_list,  # type: List[WireArray]
                  vss_warr_list,  # type: List[WireArray]
                  sup_width,  # type: int
                  fill_margin,  # type: int
                  edge_margin  # type: int
                  ):
    blk_width, blk_height = grid.get_size_dimension(size, unit_mode=True)
    lower = edge_margin
    if grid.get_direction(layer_id) == 'x':
        upper = blk_width - edge_margin
        cupper = blk_height
    else:
        upper = blk_height - edge_margin
        cupper = blk_width

    num_space = grid.get_num_space_tracks(layer_id, sup_width, half_space=False)
    start_tidx, end_tidx = grid.get_track_index_range(layer_id, 0, cupper, num_space=num_space,
                                                      edge_margin=edge_margin, half_track=False,
                                                      unit_mode=True)

    first_hidx = start_tidx * 2 + 1 + sup_width - 1
    last_hidx = end_tidx * 2 + 1 - (sup_width - 1)
    hidx_iter = range(first_hidx, last_hidx + 1, 2 * (sup_width + num_space))

    fill_table = {}  # type: Dict[int, IntervalSet]
    fill_intv_list = [(lower, upper)]
    tot_cnt = 0
    for hidx in hidx_iter:
        fill_table[hidx] = IntervalSet(intv_list=fill_intv_list)
        tot_cnt += 1

    sup_type = {}
    vdd_cnt = 0
    vss_cnt = 0
    for hidx, intv_set in track_table.items():
        for (wstart, wstop), (wwidth, fmargin, fill_type) in intv_set.items():
            fmargin = max(fill_margin, fmargin)
            sub_intv = (wstart - fmargin, wstop + fmargin)
            cbeg, cend = grid.get_wire_bounds(layer_id, (hidx - 1) / 2, width=wwidth, unit_mode=True)
            idx0, idx1 = grid.get_overlap_tracks(layer_id, cbeg - fmargin, cend + fmargin,
                                                 half_track=True, unit_mode=True)
            hidx0 = int(round(2 * idx0 + 1)) - 2 * (sup_width - 1)
            hidx1 = int(round(2 * idx1 + 1)) + 2 * (sup_width - 1)
            for sub_idx in range(hidx0, hidx1 + 1):
                if sub_idx in fill_table:
                    fill_table[sub_idx].subtract(sub_intv)
                    if sub_idx not in sup_type:
                        sup_type[sub_idx] = fill_type
                        if fill_type == 'VSS':
                            vss_cnt += 1
                        else:
                            vdd_cnt += 1

    return fill_table
