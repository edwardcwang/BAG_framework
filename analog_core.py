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

"""This module defines AmplifierBase, a base template class for Amplifier-like layout topologies."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import abc
from itertools import chain
from typing import List, Union, Optional, TypeVar, Dict, Any, Set, Tuple

from bag.math import lcm
from bag.util.interval import IntervalSet
from bag.util.search import BinaryIterator
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import TrackID, WireArray, RoutingGrid
from bag.layout.util import BBox
from bag.layout.objects import Instance, Boundary
from .analog_mos import AnalogMosBase, AnalogMosConn
from future.utils import with_metaclass

MosBase = TypeVar('MosBase', bound=AnalogMosBase)


def _flip_ud(orient):
    """Returns the new orientation after flipping the given orientation in up-down direction."""
    if orient == 'R0':
        return 'MX'
    elif orient == 'MX':
        return 'R0'
    elif orient == 'MY':
        return 'R180'
    elif orient == 'R180':
        return 'MY'
    else:
        raise ValueError('Unknown orientation: %s' % orient)


def _subtract(intv_list1, intv_list2):
    """Substrate intv_list2 from intv_list1.

    intv_list2 must be a subset of intv_list1.  Used by dummy connection calculation.

    Parameters
    ----------
    intv_list1 : list[(int, int)] or None
        first list of intervals.
    intv_list2 : list[(int, int)] or None
        second list of intervals.

    Returns
    -------
    result : list[(int, int)]
        the result of substracting the intervals in intv_list2 from intv_list1.
    """
    if not intv_list1:
        if intv_list2:
            raise ValueError('cannot substract non-empty list from empty list')
        return []
    idx1 = idx2 = 0
    result = []
    intv1 = intv_list1[idx1]
    while idx1 < len(intv_list1):
        if idx2 >= len(intv_list2):
            result.append(intv1)
            idx1 += 1
            if idx1 < len(intv_list1):
                intv1 = intv_list1[idx1]
        else:
            intv2 = intv_list2[idx2]
            if intv2[1] < intv1[0]:
                # no overlap, retire intv2
                idx2 += 1
            elif intv1[1] < intv2[0]:
                # no overlap, retire intv1
                if intv1[1] - intv1[0] > 0:
                    result.append(intv1)
                idx1 += 1
                if idx1 < len(intv_list1):
                    intv1 = intv_list1[idx1]
            else:
                # overlap, substract and update intv1
                test = intv1[0], intv2[0]
                if test[1] - test[0] > 0:
                    result.append(test)
                intv1 = (intv2[1], intv1[1])
                idx2 += 1
    return result


def _intersection(intv_list1, intv_list2):
    """Returns the intersection of two lists of intervals.

    If one of the lists is None, the other list is returned.

    Parameters
    ----------
    intv_list1 : list[(int, int)] or None
        first list of intervals.
    intv_list2 : list[(int, int)] or None
        second list of intervals.

    Returns
    -------
    result : list[(int, int)]
        the intersection of the two lists of intervals.
    """
    if intv_list1 is None:
        return list(intv_list2)
    if intv_list2 is None:
        return list(intv_list1)

    idx1 = idx2 = 0
    result = []
    while idx1 < len(intv_list1) and idx2 < len(intv_list2):
        intv1 = intv_list1[idx1]
        intv2 = intv_list2[idx2]
        test = (max(intv1[0], intv2[0]), min(intv1[1], intv2[1]))
        if test[1] - test[0] > 0:
            result.append(test)
        if intv1[1] < intv2[1]:
            idx1 += 1
        elif intv2[1] < intv1[1]:
            idx2 += 1
        else:
            idx1 += 1
            idx2 += 1

    return result


def _get_dummy_connections(intv_set_list):
    """For each row of transistors, figure out all possible substrate connection locations.

    Parameters
    ----------
    intv_set_list : list[bag.util.interval.IntervalSet]
        a list of dummy finger intervals.  index 0 is the row closest to substrate.

    Returns
    -------
    conn_list : list[list[(int, int)]]
        a list of list of intervals.  conn_list[x] contains the finger intervals where
        you can connect exactly x+1 dummies vertically to substrate.
    """
    # populate conn_list
    # conn_list[x] contains intervals where you can connect x+1 or more dummies vertically.
    conn_list = []
    prev_intv_list = None
    for intv_set in intv_set_list:
        cur_intv_list = list(intv_set.intervals())
        conn = _intersection(cur_intv_list, prev_intv_list)
        conn_list.append(conn)
        prev_intv_list = conn

    # subtract adjacent conn_list elements
    # make it so conn_list[x] contains intervals where you can connect exactly x+1 dummies vertically
    for idx in range(len(conn_list) - 1):
        cur_conn, next_conn = conn_list[idx], conn_list[idx + 1]
        conn_list[idx] = _subtract(cur_conn, next_conn)

    return conn_list


def _select_dummy_connections(conn_list, unconnected, all_conn_list):
    """Helper method for selecting dummy connections locations.

    First, look at the connections that connect the most rows of dummy.  Try to use
    as many of these connections as possible while making sure they at least connect one
    unconnected dummy.  When done, repeat on connections that connect fewer rows.

    Parameters
    ----------
    conn_list : list[list[(int, int)]]
        a list of list of intervals.  conn_list[x] contains the finger intervals where
        you can connect exactly x+1 dummies vertically to substrate.
    unconnected : list[bag.util.interval.IntervalSet]
        a list of unconnected dummy finger intervals.  index 0 is the row closest to substrate.
    all_conn_list : list[(int, int)]
        a list of dummy finger intervals where you can connect from bottom substrate to top substrate.

    Returns
    -------
    select_list : list[list[(int, int)]]
        a list of list of intervals.  select_list[x] contains the finger intervals to
        draw dummy connections for x+1 rows from substrate.
    gate_intv_set_list : list[bag.util.interval.IntervalSet]
        a list of IntervalSets.  gate_intv_set_list[x] contains the finger intervals to
        draw dummy gate connections.
    """
    if all_conn_list:
        select_list = [all_conn_list]
        gate_intv_set_list = [IntervalSet(intv_list=all_conn_list)]
    else:
        select_list = []
        gate_intv_set_list = []
    for idx in range(len(conn_list) - 1, -1, -1):
        conn_intvs = conn_list[idx]
        cur_select_list = []
        # select connections
        for intv in conn_intvs:
            select = False
            for j in range(idx + 1):
                dummy_intv_set = unconnected[j]
                if dummy_intv_set.has_overlap(intv):
                    select = True
                    break
            if select:
                cur_select_list.append(intv)
        # remove all dummies connected with selected connections
        for intv in cur_select_list:
            for j in range(idx + 1):
                unconnected[j].remove_all_overlaps(intv)

        # include in select_list
        select_list.insert(0, cur_select_list)
        # construct gate interval list.
        if not gate_intv_set_list:
            # all_conn_list must be empty.  Don't need to consider it.
            gate_intv_set_list.append(IntervalSet(intv_list=cur_select_list))
        else:
            gate_intv_set = gate_intv_set_list[0].copy()
            for intv in cur_select_list:
                if not gate_intv_set.add(intv):
                    # this should never happen.
                    raise Exception('Critical Error: report to developers.')
            gate_intv_set_list.insert(0, gate_intv_set)

    return select_list, gate_intv_set_list


class AnalogBaseInfo(object):
    """A class that calculates informations to assist in AnalogBase layout calculations.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    lch : float
        the channel length of AnalogBase, in meters.
    guard_ring_nf : int
        guard ring width in number of fingers.  0 to disable.
    min_fg_sep : int
        minimum number of separation fingers.
    """

    def __init__(self, grid, lch, guard_ring_nf, min_fg_sep=0):
        # type: (RoutingGrid, float, int, int) -> None
        tech_params = grid.tech_info.tech_params
        mos_cls = tech_params['layout']['mos_template']
        dum_cls = tech_params['layout']['mos_dummy_template']

        # get technology parameters
        self.min_fg_sep = max(min_fg_sep, tech_params['layout']['analog_base']['min_fg_sep'])
        self.mconn_diff = tech_params['layout']['analog_base']['mconn_diff_mode']
        self.float_dummy = tech_params['layout']['analog_base']['floating_dummy']

        # initialize parameters
        res = grid.resolution
        lch_unit = int(round(lch / grid.layout_unit / res))
        self._lch_unit = lch_unit
        self.num_fg_per_sd = mos_cls.get_num_fingers_per_sd(lch_unit)
        self._sd_pitch_unit = mos_cls.get_sd_pitch(lch_unit)
        self._sd_xc_unit = mos_cls.get_left_sd_xc(lch_unit, guard_ring_nf)
        self.mconn_port_layer = mos_cls.port_layer_id()
        self.dum_port_layer = dum_cls.port_layer_id()

        vm_space, vm_width = mos_cls.get_mos_conn_track_info(lch_unit)
        dum_space, dum_width = dum_cls.get_dum_conn_track_info(lch_unit)

        self.grid = grid.copy()
        self.grid.add_new_layer(self.mconn_port_layer, vm_space, vm_width, 'y', override=True, unit_mode=True)
        self.grid.add_new_layer(self.dum_port_layer, dum_space, dum_width, 'y', override=True, unit_mode=True)
        self.grid.update_block_pitch()
        self._mos_cls = mos_cls

    @property
    def sd_pitch(self):
        return self._sd_pitch_unit * self.grid.resolution

    @property
    def sd_pitch_unit(self):
        return self._sd_pitch_unit

    @property
    def sd_xc(self):
        return self._sd_xc_unit * self.grid.resolution

    @property
    def sd_xc_unit(self):
        return self._sd_xc_unit

    def get_total_width(self, fg_tot, guard_ring_nf=0):
        # type: (int, int) -> int
        """Returns the width of the AnalogMosBase in number of source/drain tracks.

        Parameters
        ----------
        fg_tot : int
            number of fingers.
        guard_ring_nf : int
            width of guard ring in number of fingers.  0 to disable guard ring.

        Returns
        -------
        mos_width : int
            the AnalogMosBase width in number of source/drain tracks.
        """
        edge_width = self._mos_cls.get_left_sd_xc(self._lch_unit, guard_ring_nf)
        tot_width = 2 * edge_width + fg_tot * self._sd_pitch_unit
        return tot_width // self._sd_pitch_unit

    def coord_to_col(self, coord, unit_mode=False, mode=0):
        """Convert the given X coordinate to transistor column index.
        
        Find the left source/drain index closest to the given coordinate.

        Parameters
        ----------
        coord : Union[float, int]
            the X coordinate.
        unit_mode : bool
            True to if coordinate is given in resolution units.
        mode : int
            rounding mode.
        Returns
        -------
        col_idx : int
            the left source/drain index closest to the given coordinate.
        """
        res = self.grid.resolution
        if not unit_mode:
            coord = int(round(coord / res))

        diff = coord - self._sd_xc_unit
        pitch = self._sd_pitch_unit
        if mode == 0:
            q = (diff + pitch // 2) // pitch
        elif mode < 0:
            q = diff // pitch
        else:
            q = -(-diff // pitch)

        return q

    def col_to_coord(self, col_idx, unit_mode=False):
        """Convert the given transistor column index to X coordinate.

        Parameters
        ----------
        col_idx : int
            the transistor index.  0 is left-most transistor.
        unit_mode : bool
            True to return coordinate in resolution units.

        Returns
        -------
        xcoord : float
            X coordinate of the left source/drain center of the given transistor.
        """
        coord = self._sd_xc_unit + col_idx * self._sd_pitch_unit
        if unit_mode:
            return coord
        return coord * self.grid.resolution

    def track_to_col_intv(self, layer_id, tr_idx, width=1):
        # type: (int, Union[float, int], int) -> Tuple[int, int]
        """Returns the smallest column interval that covers the given vertical track."""
        lower, upper = self.grid.get_wire_bounds(layer_id, tr_idx, width=width, unit_mode=True)

        lower_col_idx = (lower - self._sd_xc_unit) // self._sd_pitch_unit  # type: int
        upper_col_idx = -(-(upper - self._sd_xc_unit) // self._sd_pitch_unit)  # type: int
        return lower_col_idx, upper_col_idx

    def get_center_tracks(self, layer_id, num_tracks, col_intv, width=1, space=0):
        # type: (int, int, Tuple[int, int], int, Union[float, int]) -> float
        """Return tracks that center on the given column interval.

        Parameters
        ----------
        layer_id : int
            the vertical layer ID.
        num_tracks : int
            number of tracks
        col_intv : Tuple[int, int]
            the column interval.
        width : int
            width of each track.
        space : Union[float, int]
            space between tracks.

        Returns
        -------
        track_id : float
            leftmost track ID of the center tracks.
        """
        x0_unit = self.col_to_coord(col_intv[0], unit_mode=True)
        x1_unit = self.col_to_coord(col_intv[1], unit_mode=True)
        # find track number with coordinate strictly larger than x0
        t_start = self.grid.find_next_track(layer_id, x0_unit, half_track=True, mode=1, unit_mode=True)
        t_stop = self.grid.find_next_track(layer_id, x1_unit, half_track=True, mode=-1, unit_mode=True)
        ntracks = int(t_stop - t_start + 1)
        tot_tracks = num_tracks * width + (num_tracks - 1) * space
        if ntracks < tot_tracks:
            raise ValueError('There are only %d tracks in column interval [%d, %d)'
                             % (ntracks, col_intv[0], col_intv[1]))

        ans = t_start + (ntracks - tot_tracks + width - 1) / 2
        return ans

    def num_tracks_to_fingers(self, layer_id, num_tracks, col_idx, even=True, fg_margin=0):
        """Returns the minimum number of fingers needed to span given number of tracks.

        Returns the smallest N such that the transistor interval [col_idx, col_idx + N)
        contains num_tracks wires on routing layer layer_id.

        Parameters
        ----------
        layer_id : int
            the vertical layer ID.
        num_tracks : int
            number of tracks
        col_idx : int
            the starting column index.
        even : bool
            True to return even integers.
        fg_margin : int
            Ad this many fingers on both sides of tracks to act as margin.

        Returns
        -------
        min_fg : int
            minimum number of fingers needed to span the given number of tracks.
        """
        x0 = self.col_to_coord(col_idx, unit_mode=True)
        x1 = self.col_to_coord(col_idx + fg_margin, unit_mode=True)
        # find track number with coordinate strictly larger than x0
        t_start = self.grid.find_next_track(layer_id, x1, half_track=True, mode=1, unit_mode=True)
        # find coordinate of last track
        xlast = self.grid.track_to_coord(layer_id, t_start + num_tracks - 1, unit_mode=True)
        xlast += self.grid.get_track_width(layer_id, 1, unit_mode=True) // 2

        # divide by source/drain pitch
        q, r = divmod(xlast - x0, self._sd_pitch_unit)
        if r > 0:
            q += 1
        q += fg_margin
        if even and q % 2 == 1:
            q += 1
        return q


# noinspection PyAbstractClass
class AnalogBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    """The amplifier abstract template class

    An amplifier template consists of rows of pmos or nmos capped by substrate contacts.
    drain/source connections are mostly vertical, and gate connections are horizontal.  extension
    rows may be inserted to allow more space for gate/output connections.

    each row starts and ends with dummy transistors, and two transistors are always separated
    by separators.  Currently source sharing (e.g. diff pair) and inter-digitation are not
    supported.  All transistors have the same channel length.

    To use this class, draw_base() must be the first function called.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

        tech_params = self.grid.tech_info.tech_params
        self._mos_cls = tech_params['layout']['mos_template']
        self._sub_cls = tech_params['layout']['sub_template']
        self._ext_cls = tech_params['layout']['mos_ext_template']
        self._mconn_cls = tech_params['layout']['mos_conn_template']
        self._dum_cls = tech_params['layout']['mos_dummy_template']
        self._cap_cls = tech_params['layout'].get('mos_decap_template', None)

        # initialize parameters
        # layout information parameters
        self._lch = None
        self._w_list = None
        self._orient_list = None
        self._fg_tot = None
        self._sd_yc_list = None
        self._ds_dummy_list = None
        self._layout_info = None

        # transistor usage/automatic dummy parameters
        self._n_intvs = None  # type: List[IntervalSet]
        self._p_intvs = None  # type: List[IntervalSet]
        self._capn_intvs = None
        self._capp_intvs = None
        self._capp_wires = {-1: [], 1: []}
        self._capn_wires = {-1: [], 1: []}

        # track calculation parameters
        self._ridx_lookup = None
        self._gtr_intv = None
        self._dstr_intv = None

        # substrate parameters
        self._ntap_list = None
        self._ptap_list = None
        self._ptap_exports = None
        self._ntap_exports = None
        self._gr_vdd_warrs = None
        self._gr_vss_warrs = None

    @classmethod
    def get_mos_conn_layer(cls, tech_info):
        mos_cls = tech_info.tech_params['layout']['mos_template']
        return mos_cls.port_layer_id()

    @property
    def layout_info(self):
        # type: () -> AnalogBaseInfo
        return self._layout_info

    @property
    def min_fg_sep(self):
        """Returns the minimum number of separator fingers.
        """
        return self._layout_info.min_fg_sep

    @property
    def sd_pitch(self):
        """Returns the transistor source/drain pitch."""
        return self._layout_info.sd_pitch

    @property
    def sd_pitch_unit(self):
        """Returns the transistor source/drain pitch."""
        return self._layout_info.sd_pitch_unit

    @property
    def mos_conn_layer(self):
        """Returns the MOSFET connection layer ID."""
        return self._layout_info.mconn_port_layer

    @property
    def dum_conn_layer(self):
        """REturns the dummy connection layer ID."""
        return self._layout_info.dum_port_layer

    @property
    def min_fg_sep(self):
        """Returns the minimum number of separator fingers."""
        return self._layout_info.min_fg_sep

    @property
    def mconn_diff_mode(self):
        """Returns True if AnalogMosConn supports diffpair mode."""
        return self._layout_info.mconn_diff

    @property
    def floating_dummy(self):
        """Returns True if floating dummy connection is OK."""
        return self._layout_info.float_dummy

    def _find_row_index(self, mos_type, row_idx):
        ridx_list = self._ridx_lookup[mos_type]
        if row_idx < 0 or row_idx >= len(ridx_list):
            # error checking
            raise ValueError('%s row with index = %d not found' % (mos_type, row_idx))
        return ridx_list[row_idx]

    def get_num_tracks(self, mos_type, row_idx, tr_type):
        """Get number of tracks of the given type on the given row.

        Parameters
        ----------
        mos_type : string
            the row type, one of 'nch', 'pch', 'ntap', or 'ptap'
        row_idx : int
            the row index.  0 is the bottom-most row.
        tr_type : string
            the type of the track.  Either 'g' or 'ds'.

        Returns
        -------
        num_tracks : int
            number of tracks.
        """
        row_idx = self._find_row_index(mos_type, row_idx)
        if tr_type == 'g':
            tr_intv = self._gtr_intv[row_idx]
        else:
            tr_intv = self._dstr_intv[row_idx]

        return tr_intv[1] - tr_intv[0]

    def get_track_index(self, mos_type, row_idx, tr_type, tr_idx):
        """Convert relative track index to absolute track index.

        Parameters
        ----------
        mos_type : string
            the row type, one of 'nch', 'pch', 'ntap', or 'ptap'.
        row_idx : int
            the center row index.  0 is the bottom-most row.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        tr_idx : float
            the relative track index.

        Returns
        -------
        abs_tr_idx : float
            the absolute track index.
        """
        row_idx = self._find_row_index(mos_type, row_idx)
        if tr_type == 'g':
            tr_intv = self._gtr_intv[row_idx]
        else:
            tr_intv = self._dstr_intv[row_idx]

        # error checking
        ntr = tr_intv[1] - tr_intv[0]
        if tr_idx >= ntr:
            raise ValueError('track_index %d out of bounds: [0, %d)' % (tr_idx, ntr))

        if self._orient_list[row_idx] == 'R0':
            return tr_intv[0] + tr_idx
        else:
            return tr_intv[1] - 1 - tr_idx

    def make_track_id(self, mos_type, row_idx, tr_type, tr_idx, width=1,
                      num=1, pitch=0.0):
        """Make TrackID representing the given relative index

        Parameters
        ----------
        mos_type : string
            the row type, one of 'nch', 'pch', 'ntap', or 'ptap'.
        row_idx : int
            the center row index.  0 is the bottom-most row.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        tr_idx : float
            the relative track index.
        width : int
            track width in number of tracks.
        num : int
            number of tracks in this array.
        pitch : float
            pitch between adjacent tracks, in number of track pitches.

        Returns
        -------
        tr_id : :class:`~bag.layout.routing.TrackID`
            TrackID representing the specified track.
        """
        tid = self.get_track_index(mos_type, row_idx, tr_type, tr_idx)
        return TrackID(self.mos_conn_layer + 1, tid, width=width, num=num, pitch=pitch)

    def connect_to_substrate(self, sub_type, warr_list, inner=False, both=False):
        """Connect the given transistor wires to substrate.
        
        Parameters
        ----------
        sub_type : string
            substrate type.  Either 'ptap' or 'ntap'.
        warr_list : :class:`~bag.layout.routing.WireArray` or Iterable[:class:`~bag.layout.routing.WireArray`]
            list of WireArrays to connect to supply.
        inner : bool
            True to connect to inner substrate.
        both : bool
            True to connect to both substrates
        """
        if isinstance(warr_list, WireArray):
            warr_list = [warr_list]
        wire_yb, wire_yt = None, None
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'

        if both:
            # set inner to True if both is True
            inner = True

        # get wire upper/lower Y coordinate and record used supply tracks
        sub_port_id_list = [tid for warr in warr_list for tid in warr.track_id]
        if sub_type == 'ptap':
            if inner:
                if len(self._ptap_list) != 2:
                    raise ValueError('Inner substrate does not exist.')
                port = self._ptap_list[1].get_port(port_name)
                self._ptap_exports[1].update(sub_port_id_list)
                wire_yt = port.get_bounding_box(self.grid, self.mos_conn_layer).top
            if not inner or both:
                port = self._ptap_list[0].get_port(port_name)
                self._ptap_exports[0].update(sub_port_id_list)
                wire_yb = port.get_bounding_box(self.grid, self.mos_conn_layer).bottom
        elif sub_type == 'ntap':
            if inner:
                if len(self._ntap_list) != 2:
                    raise ValueError('Inner substrate does not exist.')
                port = self._ntap_list[0].get_port(port_name)
                self._ntap_exports[0].update(sub_port_id_list)
                wire_yb = port.get_bounding_box(self.grid, self.mos_conn_layer).bottom
            if not inner or both:
                port = self._ntap_list[-1].get_port(port_name)
                self._ntap_exports[-1].update(sub_port_id_list)
                wire_yt = port.get_bounding_box(self.grid, self.mos_conn_layer).top
        else:
            raise ValueError('Invalid substrate type: %s' % sub_type)

        self.connect_wires(warr_list, lower=wire_yb, upper=wire_yt)

    def _get_inst_flip_parity(self, xc):
        """Compute flip_parity dictionary for instance."""
        dum_layer = self.dum_conn_layer
        mconn_layer = self.mos_conn_layer
        inst_dum_par = int(self.grid.coord_to_track(dum_layer, xc, unit_mode=True)) % 2
        inst_mconn_par = int(self.grid.coord_to_track(mconn_layer, xc, unit_mode=True)) % 2

        inst_flip_parity = self.grid.get_flip_parity()
        if inst_dum_par == 1:
            inst_flip_parity[dum_layer] = not inst_flip_parity.get(dum_layer, False)
        if inst_mconn_par == 1:
            inst_flip_parity[mconn_layer] = not inst_flip_parity.get(mconn_layer, False)

        return inst_flip_parity

    def _draw_dummy_sep_conn(self, mos_type, row_idx, start, stop, gate_intv_list, sub_val_list):
        """Draw dummy/separator connection.

        Parameters
        ----------
        mos_type : string
            the row type, one of 'nch', 'pch', 'ntap', or 'ptap'.
        row_idx : int
            the center row index.  0 is the bottom-most row.
        start : int
            starting column index, inclusive.  0 is the left-most transistor.
        stop : int
            stopping column index, exclusive.
        gate_intv_list : list[(int, int)]
            sorted list of gate intervals to connect gate to M2.
            for example, if gate_intv_list = [(2, 5)], then we will draw M2 connections
            between finger number 2 (inclusive) to finger number 5 (exclusive).
        sub_val_list : List[int]
            list of substrate code that should be connected to each gate interval.
        Returns
        -------
        wires : list[:class:`~bag.layout.routing.WireArray`]
            the dummy/separator gate bus wires.
        """
        # get orientation, width, and source/drain center
        ridx = self._ridx_lookup[mos_type][row_idx]
        orient = self._orient_list[ridx]
        w = self._w_list[ridx]
        xc, yc = self._layout_info.sd_xc_unit, self._sd_yc_list[ridx]
        xc += start * self.sd_pitch_unit
        fg = stop - start

        # get edge_mode parameter
        connl, connr = False, False
        if start == 0:
            connl = True
            edge_mode = 1
            if stop == self._fg_tot:
                edge_mode = 3
                connr = True
        elif stop == self._fg_tot:
            edge_mode = 2
            connr = True
        else:
            edge_mode = 0
            # check number of fingers meet minimum finger spec
            if fg < self.min_fg_sep:
                raise ValueError('Cannot draw separator with num fingers = %d < %d' % (fg, self.min_fg_sep))

        # convert gate intervals to dummy track numbers
        dum_layer = self.dum_conn_layer
        tr_offset = self.grid.coord_to_track(dum_layer, xc, unit_mode=True) + 0.5
        dum_tr_list = []
        sub_values = []
        layout_info = self._layout_info
        for (col0, col1), sub_val in zip(gate_intv_list, sub_val_list):
            # due to the way we keep track of decaps, gate intervals may be outside of dummy boundary
            # here we throw away those cases
            col0 = max(col0, start)
            col1 = min(col1, stop)
            if col1 > col0:
                xl = layout_info.col_to_coord(col0, unit_mode=True)
                xr = layout_info.col_to_coord(col1, unit_mode=True)
                tr0 = self.grid.coord_to_track(dum_layer, xl, unit_mode=True)
                tr1 = self.grid.coord_to_track(dum_layer, xr, unit_mode=True)
                if not connl:
                    tr0 += 1
                if not connr:
                    tr1 -= 1

                # check for each interval, we can at least draw one track
                if tr1 < tr0:
                    raise ValueError('Cannot draw dummy connections in gate interval [%d, %d)' % (col0, col1))

                for htr_id in range(int(2 * tr0 + 1), int(2 * tr1 + 1) + 1, 2):
                    dum_tr_list.append((htr_id - 1) / 2 - tr_offset)
                    sub_values.append(sub_val)

        # setup parameter list
        params = dict(
            lch=self._lch,
            w=w,
            fg=fg,
            edge_mode=edge_mode,
            gate_tracks=dum_tr_list,
            flip_parity=self._get_inst_flip_parity(xc),
        )
        conn_master = self.new_template(params=params, temp_cls=self._dum_cls)
        conn_inst = self.add_instance(conn_master, loc=(xc, yc), orient=orient, unit_mode=True)

        return conn_inst.get_port().get_pins(self.dum_conn_layer), sub_values

    def mos_conn_track_used(self, tidx, margin=0):
        col_start, col_stop = self.layout_info.track_to_col_intv(self.mos_conn_layer, tidx)
        col_intv = col_start - margin, col_stop + margin
        for intv_set in chain(self._p_intvs, self._n_intvs):
            if intv_set.has_overlap(col_intv):
                return True
        return False

    def draw_mos_decap(self, mos_type, row_idx, col_idx, fg, gate_ext_mode, export_gate=False,
                       inner=False, **kwargs):
        """Draw decap connection."""
        if self._cap_cls is None:
            raise ValueError('MOS decap primitive not found in this technology.')

        # mark transistors as connected
        val = -1 if inner else 1
        if mos_type == 'pch':
            val *= -1
            intv_set = self._p_intvs[row_idx]
            cap_intv_set = self._capp_intvs[row_idx]
            wires_dict = self._capp_wires
        else:
            intv_set = self._n_intvs[row_idx]
            cap_intv_set = self._capn_intvs[row_idx]
            wires_dict = self._capn_wires

        intv = col_idx, col_idx + fg
        if not export_gate:
            # add to cap_intv_set, since we can route dummies over it
            if intv_set.has_overlap(intv) or not cap_intv_set.add(intv, val=val):
                msg = 'Cannot connect %s row %d [%d, %d); some are already connected.'
                raise ValueError(msg % (mos_type, row_idx, intv[0], intv[1]))
        else:
            # add to normal intv set.
            if cap_intv_set.has_overlap(intv) or not intv_set.add(intv, val=val):
                msg = 'Cannot connect %s row %d [%d, %d); some are already connected.'
                raise ValueError(msg % (mos_type, row_idx, intv[0], intv[1]))

        ridx = self._ridx_lookup[mos_type][row_idx]
        orient = self._orient_list[ridx]
        w = self._w_list[ridx]
        xc, yc = self._layout_info.sd_xc_unit, self._sd_yc_list[ridx]
        xc += col_idx * self.sd_pitch_unit

        conn_params = dict(
            lch=self._lch,
            w=w,
            fg=fg,
            gate_ext_mode=gate_ext_mode,
            export_gate=export_gate,
            flip_parity=self._get_inst_flip_parity(xc),
        )
        conn_params.update(kwargs)

        conn_master = self.new_template(params=conn_params, temp_cls=self._cap_cls)
        inst = self.add_instance(conn_master, loc=(xc, yc), orient=orient, unit_mode=True)
        wires_dict[val].extend(inst.get_all_port_pins('supply'))
        if export_gate:
            return {'g': inst.get_all_port_pins('g')[0]}
        else:
            return {}

    def draw_mos_conn(self, mos_type, row_idx, col_idx, fg, sdir, ddir, **kwargs):
        """Draw transistor connection.

        Parameters
        ----------
        mos_type : string
            the row type, one of 'nch', 'pch', 'ntap', or 'ptap'.
        row_idx : int
            the center row index.  0 is the bottom-most row.
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg : int
            number of fingers.
        sdir : int
            source connection direction.  0 for down, 1 for middle, 2 for up.
        ddir : int
            drain connection direction.  0 for down, 1 for middle, 2 for up.
        **kwargs :
            optional arguments for AnalogMosConn.
        Returns
        -------
        ports : dict[str, :class:`~bag.layout.routing.WireArray`]
            a dictionary of ports as WireArrays.  The keys are 'g', 'd', and 's'.
        """
        # mark transistors as connected
        if mos_type == 'pch':
            intv_set = self._p_intvs[row_idx]
            cap_intv_set = self._capp_intvs[row_idx]
        else:
            intv_set = self._n_intvs[row_idx]
            cap_intv_set = self._capn_intvs[row_idx]

        intv = col_idx, col_idx + fg
        if cap_intv_set.has_overlap(intv) or not intv_set.add(intv):
            msg = 'Cannot connect %s row %d [%d, %d); some are already connected.'
            raise ValueError(msg % (mos_type, row_idx, intv[0], intv[1]))

        sd_pitch = self.sd_pitch_unit
        ridx = self._ridx_lookup[mos_type][row_idx]
        orient = self._orient_list[ridx]
        is_ds_dummy = self._ds_dummy_list[ridx]
        w = self._w_list[ridx]
        xc, yc = self._layout_info.sd_xc_unit, self._sd_yc_list[ridx]
        xc += col_idx * sd_pitch

        if orient == 'MX':
            # flip source/drain directions
            sdir = 2 - sdir
            ddir = 2 - ddir

        conn_params = dict(
            lch=self._lch,
            w=w,
            fg=fg,
            sdir=sdir,
            ddir=ddir,
            is_ds_dummy=is_ds_dummy,
            flip_parity=self._get_inst_flip_parity(xc),
        )
        conn_params.update(kwargs)

        conn_master = self.new_template(params=conn_params, temp_cls=self._mconn_cls)  # type: AnalogMosConn
        conn_inst = self.add_instance(conn_master, loc=(xc, yc), orient=orient, unit_mode=True)

        return {key: conn_inst.get_port(key).get_pins(self.mos_conn_layer)[0]
                for key in conn_inst.port_names_iter()}

    def _make_masters(self, mos_type, lch, fg_tot, bot_sub_w, bot_sub_end, top_sub_w, top_sub_end, w_list, th_list,
                      g_tracks, ds_tracks, orientations, ds_dummy, row_offset):

        # error checking + set default values.
        num_tran = len(w_list)
        if num_tran != len(th_list):
            raise ValueError('transistor type %s width/threshold list length mismatch.' % mos_type)
        if not g_tracks:
            g_tracks = [1] * num_tran
        elif num_tran != len(g_tracks):
            raise ValueError('transistor type %s width/g_tracks list length mismatch.' % mos_type)
        if not ds_tracks:
            ds_tracks = [1] * num_tran
        elif num_tran != len(ds_tracks):
            raise ValueError('transistor type %s width/ds_tracks list length mismatch.' % mos_type)
        if not orientations:
            default_orient = 'R0' if mos_type == 'nch' else 'MX'
            orientations = [default_orient] * num_tran
        elif num_tran != len(orientations):
            raise ValueError('transistor type %s width/orientations list length mismatch.' % mos_type)
        if not ds_dummy:
            ds_dummy = [False] * num_tran
        elif num_tran != len(ds_dummy):
            raise ValueError('transistor type %s width/ds_dummy list length mismatch.' % mos_type)

        if not w_list:
            # do nothing
            return [], [], [], []

        sub_xc = self._layout_info.sd_xc_unit
        sub_flip_parity = self._get_inst_flip_parity(sub_xc)

        sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
        master_list = []
        track_spec_list = []
        w_list_final = []
        # make bottom substrate
        if bot_sub_w > 0:
            sub_params = dict(
                lch=lch,
                w=bot_sub_w,
                sub_type=sub_type,
                threshold=th_list[0],
                fg=fg_tot,
                end_mode=bot_sub_end,
                flip_parity=sub_flip_parity,
            )
            master_list.append(self.new_template(params=sub_params, temp_cls=self._sub_cls))
            track_spec_list.append(('R0', -1, -1))
            self._ridx_lookup[sub_type].append(row_offset)
            row_offset += 1
            w_list_final.append(bot_sub_w)

        # make transistors
        for w, th, gtr, dstr, orient, ds_dum in zip(w_list, th_list, g_tracks, ds_tracks, orientations, ds_dummy):
            if gtr < 0 or dstr < 0:
                raise ValueError('number of gate/drain/source tracks cannot be negative.')
            params = dict(
                lch=lch,
                w=w,
                mos_type=mos_type,
                threshold=th,
                fg=fg_tot,
                is_ds_dummy=ds_dum,
            )
            master_list.append(self.new_template(params=params, temp_cls=self._mos_cls))
            track_spec_list.append((orient, gtr, dstr))
            self._ridx_lookup[mos_type].append(row_offset)
            row_offset += 1
            w_list_final.append(w)

        # make top substrate
        if top_sub_w > 0:
            sub_params = dict(
                lch=lch,
                w=top_sub_w,
                sub_type=sub_type,
                threshold=th_list[-1],
                fg=fg_tot,
                end_mode=top_sub_end,
                flip_parity=sub_flip_parity,
            )
            master_list.append(self.new_template(params=sub_params, temp_cls=self._sub_cls))
            track_spec_list.append(('MX', -1, -1))
            self._ridx_lookup[sub_type].append(row_offset)
            w_list_final.append(top_sub_w)

        ds_dummy = [False] + ds_dummy + [False]
        return track_spec_list, master_list, ds_dummy, w_list_final

    def _place_helper(self, bot_ext_w, track_spec_list, master_list, gds_space, hm_layer, mos_pitch, tot_pitch):
        # place bottom substrate at 0
        y_cur = 0
        tr_next = 0
        y_list = []
        ext_info_list = []
        gtr_intv = []
        dtr_intv = []
        num_master = len(master_list)
        for idx in range(num_master):
            # step 1: place current master
            y_list.append(y_cur)
            cur_master = master_list[idx]
            y_top_cur = y_cur + cur_master.prim_bound_box.height_unit
            # step 2: find how many tracks current block uses
            cur_orient, cur_ng, cur_nds = track_spec_list[idx]
            if cur_ng < 0:
                # substrate.  A substrate block only use tracks within its primitive bounding box.
                tr_tmp = self.grid.find_next_track(hm_layer, y_top_cur, half_track=True, mode=1, unit_mode=True)
                dtr_intv.append((tr_next, tr_tmp))
                gtr_intv.append((tr_tmp, tr_tmp))
            else:
                # transistor.  find first unused track.
                if cur_orient == 'R0':
                    # drain/source tracks on top.  find bottom drain/source track (take gds_space into account).
                    ds_track_yc = y_cur + cur_master.get_min_ds_track_yc()
                    g_track_yc = y_cur + cur_master.get_max_g_track_yc()
                    tr_ds0 = self.grid.coord_to_nearest_track(hm_layer, ds_track_yc, half_track=True,
                                                              mode=1, unit_mode=True)
                    tr_g0 = self.grid.coord_to_nearest_track(hm_layer, g_track_yc, half_track=True,
                                                             mode=-1, unit_mode=True)
                    tr_ds0 = max(tr_ds0, tr_g0 + 1 + gds_space)
                    tr_tmp = tr_ds0 + cur_nds
                    gtr_intv.append((tr_next, tr_g0 + 1))
                    dtr_intv.append((tr_ds0, tr_tmp))
                else:
                    # gate tracks on top
                    cur_height = cur_master.array_box.height_unit
                    ds_track_yc = y_cur + cur_height - cur_master.get_min_ds_track_yc()
                    g_track_yc = y_cur + cur_height - cur_master.get_max_g_track_yc()
                    tr_dstop = self.grid.coord_to_nearest_track(hm_layer, ds_track_yc, half_track=True,
                                                                mode=-1, unit_mode=True)
                    tr_g0 = self.grid.coord_to_nearest_track(hm_layer, g_track_yc, half_track=True,
                                                             mode=1, unit_mode=True)
                    tr_dstop = min(tr_dstop, tr_g0 - 1 - gds_space)
                    tr_tmp = tr_g0 + cur_ng
                    dtr_intv.append((tr_next, tr_dstop + 1))
                    gtr_intv.append((tr_g0, tr_tmp))

            tr_next = tr_tmp

            # step 2.5: find minimum Y coordinate of next block based on track information.
            y_tr_last_top = self.grid.get_wire_bounds(hm_layer, tr_next - 1, unit_mode=True)[1]
            y_next_min = -(-y_tr_last_top // mos_pitch) * mos_pitch

            # step 3: compute extension to next master and location of next master
            if idx + 1 < num_master:
                # step 3A: figure out minimum extension width
                next_master = master_list[idx + 1]
                next_orient, next_ng, next_nds = track_spec_list[idx + 1]
                bot_ext_info = cur_master.get_ext_top_info() if cur_orient == 'R0' else cur_master.get_ext_bot_info()
                top_ext_info = next_master.get_ext_bot_info() if next_orient == 'R0' else next_master.get_ext_top_info()
                min_ext_w = self._ext_cls.get_min_width(top_ext_info, bot_ext_info)
                if idx == 0:
                    # make sure first extension width is at least bot_ext_w
                    min_ext_w = max(min_ext_w, bot_ext_w)
                min_ext_w = max(min_ext_w, (y_next_min - y_top_cur) // mos_pitch)
                # update y_next_min
                y_next_min = max(y_next_min, y_top_cur + min_ext_w * mos_pitch)
                # step 3B: figure out placement of next block
                if idx + 1 == num_master - 1:
                    # this is the last block.  Place it such that the overall height is multiples of tot_pitch.
                    next_height = next_master.prim_bound_box.height_unit
                    y_top_min = y_next_min + next_height
                    y_top = -(-y_top_min // tot_pitch) * tot_pitch
                    y_next = y_top - next_height
                else:
                    if next_ng < 0:
                        # substrate block.  place as close to current block as possible.
                        y_next = y_next_min
                    else:
                        if next_orient == 'R0':
                            # Find minimum Y coordinate to have enough gate tracks.
                            y_gtr_last_mid = self.grid.track_to_coord(hm_layer, tr_next + next_ng - 1, unit_mode=True)
                            y_next = -(-(y_gtr_last_mid - next_master.get_max_g_track_yc()) // mos_pitch) * mos_pitch
                            y_next = max(y_next, y_next_min)
                        else:
                            # find minimum Y coordinate to have enough drain/source tracks.
                            y_dtr_last_mid = self.grid.track_to_coord(hm_layer, tr_next + next_nds - 1, unit_mode=True)
                            y_coord = next_master.array_box.height_unit - next_master.get_min_ds_track_yc()
                            y_next = -(-(y_dtr_last_mid - y_coord) // mos_pitch) * mos_pitch
                            y_next = max(y_next, y_next_min)
                # step 3C: record extension information
                ext_w = (y_next - y_top_cur) // mos_pitch
                if 'mos_type' in cur_master.params:
                    ext_type = cur_master.params['mos_type']
                elif cur_master.params['sub_type'] == 'ptap':
                    ext_type = 'nch'
                else:
                    ext_type = 'pch'
                ext_params = dict(
                    lch=cur_master.params['lch'],
                    w=ext_w,
                    mos_type=ext_type,
                    threshold=cur_master.params['threshold'],
                    fg=cur_master.params['fg'],
                    top_ext_info=top_ext_info,
                    bot_ext_info=bot_ext_info,
                )
                ext_info_list.append((ext_w, ext_params))
                # step 3D: update y_cur
                y_cur = y_next

        # return placement result.
        return y_list, ext_info_list, tr_next, gtr_intv, dtr_intv

    def _place(self, track_spec_list, master_list, gds_space):
        """
        Placement strategy: make overall block match mos_pitch and horizontal track pitch, try to
        center everything between the top and bottom substrates.
        """
        # find total pitch of the analog base.
        hm_layer = self._mos_cls.port_layer_id() + 1
        mos_pitch = self._mos_cls.mos_pitch(unit_mode=True)
        hm_pitch = self.grid.get_track_pitch(hm_layer, unit_mode=True)
        tot_pitch = lcm([mos_pitch, hm_pitch])

        # first try: place everything, but blocks as close to the bottom as possible.
        y_list, ext_list, tot_ntr, gtr_intv, dtr_intv = self._place_helper(0, track_spec_list, master_list, gds_space,
                                                                           hm_layer, mos_pitch, tot_pitch)
        ext_first, ext_last = ext_list[0][0], ext_list[-1][0]
        print('ext_w0 = %d, ext_wend=%d, tot_ntr=%d' % (ext_first, ext_last, tot_ntr))
        while ext_first < ext_last - 1:
            # if the bottom extension width is smaller than the top extension width (and differ by more than 1),
            # then we can potentially get a more centered placement by increasing the minimum bottom extenison width.
            bot_ext_w = ext_first + 1
            y_next, ext_next, tot_ntr_next, gnext, dnext = self._place_helper(bot_ext_w, track_spec_list, master_list,
                                                                              gds_space, hm_layer, mos_pitch, tot_pitch)
            ext_first_next, ext_last_next = ext_next[0][0], ext_next[-1][0]
            print('ext_w0 = %d, ext_wend=%d, tot_ntr=%d' % (ext_first_next, ext_last_next, tot_ntr_next))
            if tot_ntr_next > tot_ntr or abs(ext_last - ext_first) < abs(ext_last_next - ext_first_next):
                # if either we increase the overall size of analog base, or we get a more
                # unbalanced placement, then it's not worth it anymore.
                print('abort')
                break
            else:
                # update the optimal placement strategy.
                y_list, ext_list, tot_ntr = y_next, ext_next, tot_ntr_next
                ext_last, ext_first = ext_last_next, ext_first_next
                gtr_intv, dtr_intv = gnext, dnext
                print('pick')

        # at this point we've found the optimal placement.  Place instances
        self.array_box = BBox.get_invalid_bbox()
        self._gtr_intv = gtr_intv
        self._dstr_intv = dtr_intv
        ext_list.append((0, None))
        gr_vss_warrs = []
        gr_vdd_warrs = []
        for ybot, ext_info, master, track_spec in zip(y_list, ext_list, master_list, track_spec_list):
            orient = track_spec[0]
            edge_master = master.make_edge_template()
            edgel = self.add_instance(edge_master, orient=orient)
            cur_box = edgel.translate_master_box(edge_master.prim_bound_box)
            yo = ybot - cur_box.bottom_unit
            edgel.move_by(dy=yo, unit_mode=True)
            xo = cur_box.right_unit
            inst = self.add_instance(master, loc=(xo, yo), orient=orient, unit_mode=True)
            sub_type = master.params.get('sub_type', '')
            # save substrate instance
            if sub_type == 'ptap':
                self._ptap_list.append(inst)
                self._ptap_exports.append(set())
            elif sub_type == 'ntap':
                self._ntap_list.append(inst)
                self._ntap_exports.append(set())

            sd_yc = inst.translate_master_location((0, master.get_sd_yc()), unit_mode=True)[1]
            self._sd_yc_list.append(sd_yc)
            xo = inst.array_box.right_unit
            if orient == 'R0':
                orient_r = 'MY'
            else:
                orient_r = 'R180'
            edger = self.add_instance(edge_master, loc=(xo + edge_master.prim_bound_box.width_unit, yo),
                                      orient=orient_r, unit_mode=True)
            self.array_box = self.array_box.merge(edgel.array_box).merge(edger.array_box)
            edge_inst_list = [edgel, edger]
            if ext_info[0] > 0:
                ext_master = self.new_template(params=ext_info[1], temp_cls=self._ext_cls)
                ext_edge_master = ext_master.make_edge_template()
                yo = inst.array_box.top_unit
                edgel = self.add_instance(ext_edge_master, loc=(0, yo), unit_mode=True)
                xo = ext_edge_master.prim_bound_box.width_unit
                ext_inst = self.add_instance(ext_master, loc=(xo, yo), unit_mode=True)
                xo += ext_inst.array_box.right_unit
                edger = self.add_instance(ext_edge_master, loc=(xo, yo), orient='MY', unit_mode=True)
                edge_inst_list.append(edgel)
                edge_inst_list.append(edger)

            # gather guard ring ports
            for inst in edge_inst_list:
                if inst.has_port('VDD'):
                    gr_vdd_warrs.extend(inst.get_all_port_pins('VDD'))
                elif inst.has_port('VSS'):
                    gr_vss_warrs.extend(inst.get_all_port_pins('VSS'))

        # connect body guard rings together
        self._gr_vdd_warrs = self.connect_wires(gr_vdd_warrs)
        self._gr_vss_warrs = self.connect_wires(gr_vss_warrs)

        # set array box/size/draw PR boundary
        self.set_size_from_array_box(hm_layer)
        pr_boundary = Boundary(self.grid.resolution, 'PR', self.bound_box.get_points(unit_mode=True), unit_mode=True)
        self.add_boundary(pr_boundary)

    def draw_base(self,  # type: AnalogBase
                  lch,  # type: float
                  fg_tot,  # type: int
                  ptap_w,  # type: Union[float, int]
                  ntap_w,  # type: Union[float, int]
                  nw_list,  # type: List[Union[float, int]]
                  nth_list,  # type: List[str]
                  pw_list,  # type: List[Union[float, int]]
                  pth_list,  # type: List[str]
                  gds_space=0,  # type: int
                  ng_tracks=None,  # type: Optional[List[int]]
                  nds_tracks=None,  # type: Optional[List[int]]
                  pg_tracks=None,  # type: Optional[List[int]]
                  pds_tracks=None,  # type: Optional[List[int]]
                  n_orientations=None,  # type: Optional[List[str]]
                  p_orientations=None,  # type: Optional[List[str]]
                  guard_ring_nf=0,  # type: int
                  n_ds_dummy=None,  # type: Optional[List[bool]]
                  p_ds_dummy=None,  # type: Optional[List[bool]]
                  pitch_offset=(0, 0),  # type: Tuple[int, int]
                  pgr_w=None,  # type: Optional[Union[float, int]]
                  ngr_w=None,  # type: Optional[Union[float, int]]
                  min_fg_sep=0,  # type: int
                  end_mode=3,  # type: int
                  flip_parity=None,  # type: Optional[Dict[int, bool]]
                  ):
        # type: (...) -> None
        """Draw the analog base.

        This method must be called first.

        Parameters
        ----------
        lch : float
            the transistor channel length, in meters
        fg_tot : int
            total number of fingers for each row.
        ptap_w : Union[float, int]
            pwell substrate contact width.
        ntap_w : Union[float, int]
            nwell substrate contact width.
        gds_space : int
            number of tracks to reserve as space between gate and drain/source tracks.
        nw_list : List[Union[float, int]]
            a list of nmos width for each row, from bottom to top.
        nth_list: List[str]
            a list of nmos threshold flavor for each row, from bottom to top.
        pw_list : List[Union[float, int]]
            a list of pmos width for each row, from bottom to top.
        pth_list : List[str]
            a list of pmos threshold flavor for each row, from bottom to top.
        ng_tracks : Optional[List[int]]
            number of nmos gate tracks per row, from bottom to top.  Defaults to 1.
        nds_tracks : Optional[List[int]]
            number of nmos drain/source tracks per row, from bottom to top.  Defaults to 1.
        pg_tracks : Optional[List[int]]
            number of pmos gate tracks per row, from bottom to top.  Defaults to 1.
        pds_tracks : Optional[List[int]]
            number of pmos drain/source tracks per row, from bottom to top.  Defaults to 1.
        n_orientations : Optional[List[str]]
            orientation of each nmos row. Defaults to all 'R0'.
        p_orientations : Optional[List[str]]
            orientation of each pmos row.  Defaults to all 'MX'.
        guard_ring_nf : int
            width of guard ring in number of fingers.  0 to disable guard ring.
        n_ds_dummy : Optional[List[bool]]
            is_ds_dummy flag for each nmos row.  Defaults to all False.
        p_ds_dummy : Optional[List[bool]]
            is_ds_dummy flag for each pmos row.  Defaults to all False.
        pitch_offset : Tuple[int, int]
            shift the templates right/up by this many track pitches.  This parameter is
            used to center the transistors in the grid.
        pgr_w : Optional[Union[float, int]]
            pwell guard ring substrate contact width.
        ngr_w : Optional[Union[float, int]]
            nwell guard ringsubstrate contact width.
        min_fg_sep : int
            minimum number of fingers between different transistors.
        end_mode : int
            substrate end mode flag
        flip_parity : Optional[Dict[int, bool]]
            list of whether to flip parity on each layer.
        """
        numn = len(nw_list)
        nump = len(pw_list)
        # error checking
        if numn == 0 and nump == 0:
            raise ValueError('Cannot make empty AnalogBase.')

        # make AnalogBaseInfo object.  Also update routing grid.
        self._layout_info = AnalogBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        self.grid = self._layout_info.grid
        if flip_parity is not None:
            self.grid.set_flip_parity(flip_parity)

        # initialize private attributes.
        self._lch = lch
        self._w_list = []
        self._fg_tot = fg_tot
        self._sd_yc_list = []
        self._ds_dummy_list = []

        self._n_intvs = [IntervalSet() for _ in range(numn)]
        self._p_intvs = [IntervalSet() for _ in range(nump)]
        self._capn_intvs = [IntervalSet() for _ in range(numn)]
        self._capp_intvs = [IntervalSet() for _ in range(nump)]

        self._ridx_lookup = dict(nch=[], pch=[], ntap=[], ptap=[])

        self._ntap_list = []
        self._ptap_list = []
        self._ptap_exports = []
        self._ntap_exports = []

        if pgr_w is None:
            pgr_w = ptap_w
        if ngr_w is None:
            ngr_w = ntap_w

        if guard_ring_nf == 0:
            pgr_w = ngr_w = 0

        # place transistor blocks
        master_list = []
        track_spec_list = []
        bot_sub_end = end_mode % 2
        top_sub_end = end_mode // 2
        top_nsub_end = top_sub_end if not pw_list else 0
        bot_psub_end = bot_sub_end if not nw_list else 0
        # make NMOS substrate/transistor masters.
        tr_list, m_list, n_ds_dummy, nw_list = self._make_masters('nch', self._lch, fg_tot, ptap_w, bot_sub_end, ngr_w,
                                                                  top_nsub_end, nw_list, nth_list, ng_tracks,
                                                                  nds_tracks, n_orientations, n_ds_dummy, 0)
        master_list.extend(m_list)
        track_spec_list.extend(tr_list)
        self._ds_dummy_list.extend(n_ds_dummy)
        self._w_list.extend(nw_list)
        # make PMOS substrate/transistor masters.
        tr_list, m_list, p_ds_dummy, pw_list = self._make_masters('pch', self._lch, fg_tot, pgr_w, bot_psub_end, ntap_w,
                                                                  top_sub_end, pw_list, pth_list, pg_tracks,
                                                                  pds_tracks, p_orientations, p_ds_dummy, len(m_list))
        master_list.extend(m_list)
        track_spec_list.extend(tr_list)
        self._ds_dummy_list.extend(p_ds_dummy)
        self._w_list.extend(pw_list)
        self._orient_list = [item[0] for item in track_spec_list]

        # place masters according to track specifications.  Try to center transistors
        self._place(track_spec_list, master_list, gds_space)

    def _connect_substrate(self,  # type: AnalogBase
                           sub_type,  # type: str
                           sub_list,  # type: List[Instance]
                           row_idx_list,  # type: List[int]
                           lower=None,  # type: Optional[Union[float, int]]
                           upper=None,  # type: Optional[Union[float, int]]
                           sup_wires=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                           sup_margin=0,  # type: int
                           unit_mode=False  # type: bool
                           ):
        """Connect all given substrates to horizontal tracks

        Parameters
        ----------
        sub_type : str
            substrate type.  Either 'ptap' or 'ntap'.
        sub_list : List[Instance]
            list of substrates to connect.
        row_idx_list : List[int]
            list of substrate row indices.
        lower : Optional[Union[float, int]]
            lower supply track coordinates.
        upper : Optional[Union[float, int]]
            upper supply track coordinates.
        sup_wires : Optional[Union[WireArray, List[WireArray]]]
            If given, will connect these horizontal wires to supply on mconn layer.
        sup_margin : int
            supply wires mconn layer connection horizontal margin in number of tracks.
        unit_mode : bool
            True if lower/upper is specified in resolution units.

        Returns
        -------
        track_buses : list[bag.layout.routing.WireArray]
            list of substrate tracks buses.
        """
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'

        if sup_wires is not None and isinstance(sup_wires, WireArray):
            sup_wires = [sup_wires]
        else:
            pass

        sub_warr_list = []
        hm_layer = self.mos_conn_layer + 1
        for row_idx, subinst in zip(row_idx_list, sub_list):
            # Create substrate TrackID
            sub_row_idx = self._find_row_index(sub_type, row_idx)
            dtr_intv = self._dstr_intv[sub_row_idx]
            ntr = dtr_intv[1] - dtr_intv[0]
            sub_w = self.grid.get_max_track_width(hm_layer, 1, ntr, half_end_space=False)
            track_id = TrackID(hm_layer, dtr_intv[0] + (ntr - 1) / 2, width=sub_w)

            # get all wires to connect to supply.
            warr_iter_list = [subinst.get_port(port_name).get_pins(self.mos_conn_layer)]
            if port_name == 'VDD':
                warr_iter_list.append(self._gr_vdd_warrs)
            else:
                warr_iter_list.append(self._gr_vss_warrs)

            warr_list = list(chain(*warr_iter_list))
            track_warr = self.connect_to_tracks(warr_list, track_id, track_lower=lower, track_upper=upper,
                                                unit_mode=unit_mode)
            sub_warr_list.append(track_warr)
            if sup_wires is not None:
                wlower, wupper = warr_list[0].lower, warr_list[0].upper
                for conn_warr in sup_wires:
                    if conn_warr.layer_id != hm_layer:
                        raise ValueError('vdd/vss wires must be on layer %d' % hm_layer)
                    tmin, tmax = self.grid.get_overlap_tracks(hm_layer - 1, conn_warr.lower,
                                                              conn_warr.upper, half_track=True)
                    new_warr_list = []
                    for warr in warr_list:
                        for tid in warr.track_id:
                            if tid > tmax:
                                break
                            elif tmin <= tid:
                                if not self.mos_conn_track_used(tid, margin=sup_margin):
                                    new_warr_list.append(
                                        WireArray(TrackID(hm_layer - 1, tid), lower=wlower, upper=wupper))
                    self.connect_to_tracks(new_warr_list, conn_warr.track_id)

        return sub_warr_list

    def fill_dummy(self,  # type: AnalogBase
                   lower=None,  # type: Optional[Union[float, int]]
                   upper=None,  # type: Optional[Union[float, int]]
                   vdd_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                   vss_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                   sup_margin=0,  # type: int
                   unit_mode=False  # type: bool
                   ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw dummy/separator on all unused transistors.

        This method should be called last.

        Parameters
        ----------
        lower : Optional[Union[float, int]]
            lower coordinate for the supply tracks.
        upper : Optional[Union[float, int]]
            upper coordinate for the supply tracks.
        vdd_warrs : Optional[Union[WireArray, List[WireArray]]]
            vdd wires to be connected.
        vss_warrs : Optional[Union[WireArray, List[WireArray]]]
            vss wires to be connected.
        sup_margin : int
            vdd/vss wires mos conn layer margin in number of tracks.
        unit_mode : bool
            True if lower/upper are specified in resolution units.

        Returns
        -------
        ptap_wire_arrs : List[WireArray]
            list of P-tap substrate WireArrays.
        ntap_wire_arrs : List[WireArray]
            list of N-tap substrate WireArrays.
        """
        # invert PMOS/NMOS IntervalSet to get unconnected dummies
        total_intv = (0, self._fg_tot)
        p_intvs = [intv_set.get_complement(total_intv) for intv_set in self._p_intvs]
        n_intvs = [intv_set.get_complement(total_intv) for intv_set in self._n_intvs]

        # connect NMOS dummies
        top_tracks = None
        top_sub_inst = None
        if self._ptap_list:
            bot_sub_inst = self._ptap_list[0]
            bot_tracks = self._ptap_exports[0]
            if len(self._ptap_list) > 1:
                top_sub_inst = self._ptap_list[1]
                top_tracks = self._ptap_exports[1]
            self._fill_dummy_helper('nch', n_intvs, self._capn_intvs, self._capn_wires, bot_sub_inst, top_sub_inst,
                                    bot_tracks, top_tracks, not self._ntap_list)

        # connect PMOS dummies
        bot_tracks = None
        bot_sub_inst = None
        if self._ntap_list:
            top_sub_inst = self._ntap_list[-1]
            top_tracks = self._ntap_exports[-1]
            if len(self._ntap_list) > 1:
                bot_sub_inst = self._ntap_list[0]
                bot_tracks = self._ntap_exports[0]
            self._fill_dummy_helper('pch', p_intvs, self._capp_intvs, self._capp_wires, bot_sub_inst, top_sub_inst,
                                    bot_tracks, top_tracks, not self._ptap_list)

        # connect NMOS substrates to horizontal tracks.
        if not self._ntap_list:
            # connect both substrates if NMOS only
            ptap_wire_arrs = self._connect_substrate('ptap', self._ptap_list, list(range(len(self._ptap_list))),
                                                     lower=lower, upper=upper, sup_wires=vss_warrs,
                                                     sup_margin=sup_margin, unit_mode=unit_mode)
        elif self._ptap_list:
            # NMOS exists, only connect bottom substrate to upper level metal
            ptap_wire_arrs = self._connect_substrate('ptap', self._ptap_list[:1], [0],
                                                     lower=lower, upper=upper, sup_wires=vss_warrs,
                                                     sup_margin=sup_margin, unit_mode=unit_mode)
        else:
            ptap_wire_arrs = []

        # connect PMOS substrates to horizontal tracks.
        if not self._ptap_list:
            # connect both substrates if PMOS only
            ntap_wire_arrs = self._connect_substrate('ntap', self._ntap_list, list(range(len(self._ntap_list))),
                                                     lower=lower, upper=upper, sup_wires=vdd_warrs,
                                                     sup_margin=sup_margin, unit_mode=unit_mode)
        elif self._ntap_list:
            # PMOS exists, only connect top substrate to upper level metal
            ntap_wire_arrs = self._connect_substrate('ntap', self._ntap_list[-1:], [len(self._ntap_list) - 1],
                                                     lower=lower, upper=upper, sup_wires=vdd_warrs,
                                                     sup_margin=sup_margin, unit_mode=unit_mode)
        else:
            ntap_wire_arrs = []

        return ptap_wire_arrs, ntap_wire_arrs

    def _fill_dummy_helper(self,  # type: AnalogBase
                           mos_type,  # type: str
                           intv_set_list,  # type: List[IntervalSet]
                           cap_intv_set_list,  # type: List[IntervalSet]
                           cap_wires_dict,  # type: Dict[int, List[WireArray]]
                           bot_sub_inst,  # type: Optional[Instance]
                           top_sub_inst,  # type: Optional[Instance]
                           bot_tracks,  # type: List[int]
                           top_tracks,  # type: List[int]
                           export_both  # type: bool
                           ):
        # type: (...) -> None
        num_rows = len(intv_set_list)
        bot_conn = top_conn = []

        num_sub = 0
        if bot_sub_inst is not None:
            num_sub += 1
            bot_conn = _get_dummy_connections(intv_set_list)
        if top_sub_inst is not None:
            num_sub += 1
            top_conn = _get_dummy_connections(list(reversed(intv_set_list)))

        # make list of unconnected interval sets.
        unconn_intv_set_list = []
        dum_avail_intv_set_list = []
        # subtract cap interval sets.
        for intv_set, cap_intv_set in zip(intv_set_list, cap_intv_set_list):
            unconn_intv_set_list.append(intv_set.copy())
            temp_intv = intv_set.copy()
            for intv in cap_intv_set:
                temp_intv.subtract(intv)
            dum_avail_intv_set_list.append(temp_intv)

        if num_sub == 2:
            # we have both top and bottom substrate, so we can connect all dummies together
            all_conn_list = bot_conn[-1]
            del bot_conn[-1]
            del top_conn[-1]

            # remove dummies connected by connections in all_conn_list
            for all_conn_intv in all_conn_list:
                for intv_set in unconn_intv_set_list:  # type: IntervalSet
                    intv_set.remove_all_overlaps(all_conn_intv)
        else:
            all_conn_list = []

        bot_dum_only = top_dum_only = False
        if mos_type == 'nch':
            # for NMOS, prioritize connection to bottom substrate.
            port_name = 'VSS'
            bot_select, bot_gintv = _select_dummy_connections(bot_conn, unconn_intv_set_list, all_conn_list)
            top_select, top_gintv = _select_dummy_connections(top_conn, unconn_intv_set_list[::-1], all_conn_list)
            top_dum_only = not export_both
        else:
            # for PMOS, prioritize connection to top substrate.
            port_name = 'VDD'
            top_select, top_gintv = _select_dummy_connections(top_conn, unconn_intv_set_list[::-1], all_conn_list)
            bot_select, bot_gintv = _select_dummy_connections(bot_conn, unconn_intv_set_list, all_conn_list)
            bot_dum_only = not export_both

        # make list of dummy gate connection parameters
        dummy_gate_conns = {}
        all_conn_set = IntervalSet(intv_list=all_conn_list)
        for gintvs, sign in [(bot_gintv, 1), (top_gintv, -1)]:
            if gintvs:
                for distance in range(num_rows):
                    ridx = distance if sign > 0 else num_rows - 1 - distance
                    if distance < len(gintvs):
                        gate_intv_set = gintvs[distance]
                        for dummy_intv in dum_avail_intv_set_list[ridx]:
                            key = ridx, dummy_intv[0], dummy_intv[1]
                            overlaps = list(gate_intv_set.overlap_intervals(dummy_intv))
                            val_list = [0 if ovl_intv in all_conn_set else sign for ovl_intv in overlaps]
                            if key not in dummy_gate_conns:
                                dummy_gate_conns[key] = IntervalSet(intv_list=overlaps, val_list=val_list)
                            else:
                                dummy_gate_set = dummy_gate_conns[key]  # type: IntervalSet
                                for fg_intv, ovl_val in zip(overlaps, val_list):
                                    if not dummy_gate_set.has_overlap(fg_intv):
                                        dummy_gate_set.add(fg_intv, val=ovl_val)
                                    else:
                                        # check that we don't have conflicting gate connections.
                                        for existing_intv, existing_val in dummy_gate_set.overlap_items(fg_intv):
                                            if existing_intv != fg_intv or existing_val != ovl_val:
                                                # this should never happen.
                                                raise Exception('Critical Error: report to developers.')

        wire_groups = {-1: [], 0: [], 1: []}
        for key, dummy_gate_set in dummy_gate_conns.items():
            ridx, start, stop = key
            if dummy_gate_set:
                gate_intv_list = list(dummy_gate_set.intervals())
                sub_val_list = list(dummy_gate_set.values())
                gate_buses, sub_values = self._draw_dummy_sep_conn(mos_type, ridx, start, stop, gate_intv_list,
                                                                   sub_val_list)

                for gate_warr, sub_val in zip(gate_buses, sub_values):
                    wire_groups[sub_val].append(gate_warr)
            elif not self.floating_dummy:
                raise Exception('Dummy (%d, %d) at row %d unconnected.' % (start, stop, ridx))

        for sign in (-1, 1):
            wire_groups[sign].extend(cap_wires_dict[sign])

        grid = self.grid
        sub_yb = sub_yt = None
        if bot_sub_inst is not None:
            sub_yb = bot_sub_inst.get_port(port_name).get_bounding_box(grid, self.dum_conn_layer).bottom
        if top_sub_inst is not None:
            sub_yt = top_sub_inst.get_port(port_name).get_bounding_box(grid, self.dum_conn_layer).top

        # connect dummy ports to substrates and record dummy port track numbers.
        bot_dum_tracks = set()
        top_dum_tracks = set()
        for sub_idx, wire_bus_list in wire_groups.items():
            sub_port_id_list = [tid for warr in wire_bus_list for tid in warr.track_id]
            wire_yb = wire_yt = None
            if sub_idx >= 0:
                wire_yb = sub_yb
                bot_dum_tracks.update(sub_port_id_list)
            if sub_idx <= 0:
                wire_yt = sub_yt
                top_dum_tracks.update(sub_port_id_list)
            self.connect_wires(wire_bus_list, lower=wire_yb, upper=wire_yt)

        # update substrate master to only export necessary wires
        if bot_sub_inst is not None:
            self._export_supplies(bot_dum_tracks, bot_tracks, bot_sub_inst, bot_dum_only)
        if top_sub_inst is not None:
            self._export_supplies(top_dum_tracks, top_tracks, top_sub_inst, top_dum_only)

    def _export_supplies(self, dum_tracks, port_tracks, sub_inst, dum_only):
        x0 = self._layout_info.sd_xc_unit
        dum_tr_offset = self.grid.coord_to_track(self.dum_conn_layer, x0, unit_mode=True) + 0.5
        mconn_tr_offset = self.grid.coord_to_track(self.mos_conn_layer, x0, unit_mode=True) + 0.5
        dum_tracks = [tr - dum_tr_offset for tr in dum_tracks]
        port_tracks = [tr - mconn_tr_offset for tr in port_tracks]
        sub_inst.new_master_with(dum_tracks=dum_tracks, port_tracks=port_tracks,
                                 dummy_only=dum_only)


class SubstrateContact(TemplateBase):
    """A template that draws a single substrate.

    Useful for resistor/capacitor body biasing.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(SubstrateContact, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._num_fingers = None

    @property
    def port_name(self):
        return 'VDD' if self.params['sub_type'] == 'ntap' else 'VSS'

    @property
    def num_fingers(self):
        return self._num_fingers

    @classmethod
    def default_top_layer(cls, tech_info):
        mos_cls = tech_info.tech_params['layout']['mos_template']
        mconn_layer = mos_cls.port_layer_id()
        return mconn_layer + 1

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
            lch='channel length, in meters.',
            w='substrate width, in meters/number of fins.',
            sub_type='substrate type.',
            threshold='substrate threshold flavor.',
            top_layer='the top level layer ID.',
            blk_width='Width of this template in number of blocks.',
            show_pins='True to show pin labels.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, w, sub_type, threshold, top_layer, blk_width, show_pins):
        # type: (float, Union[float, int], str, str, int, int, bool) -> None

        # get technology parameters
        res = self.grid.resolution
        tech_params = self.grid.tech_info.tech_params
        mos_cls = tech_params['layout']['mos_template']
        dum_cls = tech_params['layout']['mos_dummy_template']
        sub_cls = tech_params['layout']['sub_template']
        mconn_layer = mos_cls.port_layer_id()
        dum_layer = dum_cls.port_layer_id()
        sd_pitch = mos_cls.get_sd_pitch(lch)
        hm_layer = mconn_layer + 1

        # add transistor routing layers to grid
        vm_width = mos_cls.get_port_width(lch)
        vm_space = sd_pitch - vm_width
        dum_width = dum_cls.get_port_width(lch)
        dum_space = sd_pitch - dum_width
        hm_pitch_unit = self.grid.get_track_pitch(hm_layer, unit_mode=True)

        self.grid = self.grid.copy()
        self.grid.add_new_layer(mconn_layer, vm_space, vm_width, 'y', override=True)
        self.grid.add_new_layer(dum_layer, dum_space, dum_width, 'y', override=True)
        self.grid.update_block_pitch()

        if top_layer < hm_layer:
            raise ValueError('SubstrateContact top layer must be >= %d' % hm_layer)

        # compute template width in number of sd pitches
        wtot_unit, _ = self.grid.get_size_dimension((top_layer, blk_width, 1), unit_mode=True)
        wblk_unit, hblk_unit = self.grid.get_block_size(top_layer, unit_mode=True)
        sd_pitch_unit = int(round(sd_pitch / res))
        q, r = divmod(wtot_unit, sd_pitch_unit)
        # find maximum number of fingers we can draw
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_fg = bin_iter.get_next()
            num_sd = mos_cls.get_template_width(cur_fg)
            if num_sd == q:
                bin_iter.save()
                break
            elif num_sd < q:
                bin_iter.save()
                bin_iter.up()
            else:
                bin_iter.down()

        sub_fg_tot = bin_iter.get_last_save()
        if sub_fg_tot is None:
            raise ValueError('Cannot draw substrate that fit in width: %d' % wtot_unit)

        # compute offset
        num_sd = mos_cls.get_template_width(sub_fg_tot)
        if r != 0:
            sub_xoff = (q + 1 - num_sd) // 2
        else:
            sub_xoff = (q - num_sd) // 2

        # create substrate
        self._num_fingers = sub_fg_tot
        params = dict(
            lch=lch,
            w=w,
            sub_type=sub_type,
            threshold=threshold,
            fg=sub_fg_tot,
            end_mode=3,
        )
        master = self.new_template(params=params, temp_cls=sub_cls)
        # find substrate height and calculate block height and offset
        _, h_sub_unit = self.grid.get_size_dimension(master.size, unit_mode=True)
        blk_height = -(-h_sub_unit // hblk_unit)
        nh_tracks = blk_height * hblk_unit // hm_pitch_unit
        sub_nh_tracks = h_sub_unit // hm_pitch_unit
        sub_yoff = (nh_tracks - sub_nh_tracks) // 2

        # add instance and set size
        inst = self.add_instance(master, inst_name='XSUB', loc=(sub_xoff * sd_pitch, sub_yoff * hm_pitch_unit * res))
        self.array_box = inst.array_box
        # find the first horizontal track index inside the array box
        hm_idx0 = self.grid.coord_to_nearest_track(hm_layer, self.array_box.bottom, mode=1)
        self.size = (top_layer, blk_width, blk_height)
        # add implant layers to cover entire template
        imp_box = self.bound_box
        for imp_layer in sub_cls.get_implant_layers(sub_type, threshold):
            self.add_rect(imp_layer, imp_box)

        # connect to horizontal metal layer.
        ntr = self.array_box.height_unit // hm_pitch_unit  # type: int
        tr_width = self.grid.get_max_track_width(hm_layer, 1, ntr, half_end_space=False)
        track_id = TrackID(hm_layer, hm_idx0 + (ntr - 1) / 2, width=tr_width)
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sub_wires = self.connect_to_tracks(inst.get_port(port_name).get_pins(mconn_layer), track_id)
        self.add_pin(port_name, sub_wires, show=show_pins)
