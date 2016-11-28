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

import abc
from itertools import izip, chain, repeat
import bisect
import numpy as np

from bag.layout.template import MicroTemplate
from bag.layout.util import BBox, BBoxArray, Port
from .analog_mos import AnalogMosBase, AnalogSubstrate, AnalogMosConn, AnalogMosSep, AnalogMosDummy


class IntervalSet(object):
    """A data structure that keeps track of disjoint 1D intervals.

    Each interval has a value associated with it.  If not specified, the value defaults to None.

    Parameters
    ----------
    intv_list : list[(float, float)] or None
        the sorted initial interval list.
    val_list : list[any] or None
        the initial values list.
    res : float
        the interval coordinate resolution
    """

    def __init__(self, intv_list=None, val_list=None, res=0.001):

        if intv_list is None:
            self._start_list = []
            self._end_list = []
            self._val_list = []
        else:
            self._start_list = [v[0] for v in intv_list]
            self._end_list = [v[1] for v in intv_list]
            if val_list is None:
                self._val_list = [None] * len(self._start_list)
            else:
                self._val_list = list(val_list)

        self._res = res

    def _lt(self, a, b):
        """Return true if a < b within resolution."""
        return a - b < -self._res

    def __contains__(self, key):
        """Returns True if this IntervalSet contains the given interval.

        Parameters
        ----------
        key : (int or float, int or float)
            the interval to test.

        Returns
        -------
        contains : bool
            True if this IntervalSet contains the given interval.
        """
        idx = self._get_first_overlap_idx(key)
        return idx >= 0 and abs(self._start_list[idx] - key[0]) < self._res and abs(
            self._end_list[idx] - key[1]) < self._res

    def __iter__(self):
        """Iterates over intervals in this IntervalSet in increasing order.

        Yields
        ------
        intv : (int or float, int or float)
            the next interval.
        """
        return izip(self._start_list, self._end_list)

    def __len__(self):
        """Returns the number of intervals in this IntervalSet.

        Returns
        -------
        length : int
            number of intervals in this set.
        """
        return len(self._start_list)

    def get_start(self):
        """Returns the smallest interval lower bound.

        Returns
        -------
        start : int or float
            the smallest interval lower bound.
        """
        return self._start_list[0]

    def get_end(self):
        """Returns the largest interval upper bound.

        Returns
        -------
        end : int or float
            the largest interval upper bound.
        """
        return self._end_list[-1]

    def copy(self):
        """Create a copy of this interval set.

        Returns
        -------
        intv_set : IntervalSet
            a copy of this IntervalSet.
        """
        return IntervalSet(intv_list=list(izip(self._start_list, self._end_list)),
                           val_list=self._val_list, res=self._res)

    def _get_first_overlap_idx(self, intv):
        """Returns the index of the first interval that overlaps with the given interval.

        Parameters
        ----------
        intv : (int or float, int or float)
            the given interval.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  If no overlapping intervals are
            found, -(idx + 1) is returned, where idx is the index to insert the interval.
        """
        start, end = intv
        if not self._start_list:
            return -1
        # find the smallest start index greater than start
        idx = bisect.bisect_right(self._start_list, start)
        if idx == 0:
            # all interval's starting point is greater than start
            return 0 if self._lt(self._start_list[0], end) else -1

        # interval where start index is less than or equal to start
        test_idx = idx - 1
        if self._lt(start, self._end_list[test_idx]):
            # start is covered by the interval; overlaps.
            return test_idx
        elif idx < len(self._start_list) and self._lt(self._start_list[idx], end):
            # _start_list[idx] covered by interval.
            return idx
        else:
            # if
            # no overlap interval found
            return -(idx + 1)

    def _get_last_overlap_idx(self, intv):
        """Returns the index of the last interval that overlaps with the given interval.

        Parameters
        ----------
        intv : (int or float, int or float)
            the given interval.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  If no overlapping intervals are
            found, -(idx + 1) is returned, where idx is the index to insert the interval.
        """
        start, end = intv
        if not self._start_list:
            return -1
        # find the smallest start index greater than end
        idx = bisect.bisect_right(self._start_list, end)
        if idx == 0:
            # all interval's starting point is greater than end
            return -1

        # interval where start index is less than or equal to end
        test_idx = idx - 1
        if self._lt(self._end_list[test_idx], start):
            # end of interval less than start; no overlap
            return -(idx + 1)
        else:
            return test_idx

    def has_overlap(self, intv):
        """Returns True if the given interval overlaps at least one interval in this set.

        Parameters
        ----------
        intv : (int or float, int or float)
            the given interval.

        Returns
        -------
        has_overlap : bool
            True if there is at least one interval in this set that overlaps with the given one.
        """
        return self._get_first_overlap_idx(intv) >= 0

    def remove(self, intv):
        """Removes the given interval from this IntervalSet.

        Parameters
        ----------
        intv : (int or float, int or float)
            the interval to remove.

        Returns
        -------
        success : bool
            True if the given interval is found and removed.  False otherwise.
        """
        idx = self._get_first_overlap_idx(intv)
        if idx < 0:
            return False
        if abs(intv[0] - self._start_list[idx]) < self._res and abs(intv[1] - self._end_list[idx]) < self._res:
            del self._start_list[idx]
            del self._end_list[idx]
            del self._val_list[idx]
            return True
        return False

    def remove_all_overlaps(self, intv):
        """Remove all intervals in this set that overlaps with the given interval.

        Parameters
        ----------
        intv : (int or float, int or float)
            the given interval
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx >= 0:
            eidx = self._get_last_overlap_idx(intv) + 1
            del self._start_list[sidx:eidx]
            del self._end_list[sidx:eidx]
            del self._val_list[sidx:eidx]

    def add(self, intv, val=None):
        """Adds the given interval to this IntervalSet.

        Can only add interval that does not overlap with any existing ones.

        Parameters
        ----------
        intv : (int or float, int or float)
            the interval to add.
        val : any or None
            the value associated with the given interval.

        Returns
        -------
        success : bool
            True if the given interval is added.
        """
        idx = self._get_first_overlap_idx(intv)
        if idx >= 0:
            return False
        idx = -idx - 1
        self._start_list.insert(idx, intv[0])
        self._end_list.insert(idx, intv[1])
        self._val_list.insert(idx, val)

    def items(self):
        """Iterates over intervals and values in this IntervalSet

        The intervals are returned in increasing order.

        Yields
        ------
        intv : (int or float, int or float)
            the interval.
        val : any
            the value associated with the interval.
        """
        return izip(self.__iter__(), self._val_list)

    def intervals(self):
        """Iterates over intervals in this IntervalSet

        The intervals are returned in increasing order.

        Yields
        ------
        intv : (int or float, int or float)
            the interval.
        """
        return self.__iter__()

    def values(self):
        """Iterates over values in this IntervalSet

        The values correspond to intervals in increasing order.

        Yields
        ------
        val : any
            the value.
        """
        return self._val_list.__iter__()

    def overlap_items(self, intv):
        """Iterates over intervals and values overlapping the given interval.

        Parameters
        ----------
        intv : (int or float, int or float)
            the interval.

        Yields
        -------
        ovl_intv : (int or float, int or float)
            the overlapping interval.
        val : any
            value associated with ovl_intv.
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx < 0:
            return
        eidx = self._get_last_overlap_idx(intv) + 1
        for idx in xrange(sidx, eidx):
            yield (self._start_list[idx], self._end_list[idx]), self._val_list[idx]

    def overlap_intervals(self, intv):
        """Iterates over intervals overlapping the given interval.

        Parameters
        ----------
        intv : (int or float, int or float)
            the interval.

        Yields
        -------
        ovl_intv : (int or float, int or float)
            the overlapping interval.
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx < 0:
            return
        eidx = self._get_last_overlap_idx(intv) + 1
        for idx in xrange(sidx, eidx):
            yield self._start_list[idx], self._end_list[idx]


def _subtract_from_set(intv_set, start, end):
    """Substract the given interval from the interval set.

    Used to mark transistors as connected.  Assumes exactly one interval in intv_set overlaps with the given interval.

    Parameters
    ----------
    intv_set : IntervalSet
        the interval set.
    start : int
        the interval lower bound.
    end : int
        the interval upper bound.

    Returns
    -------
    success : bool
        True on success.
    """
    intv_list = list(intv_set.overlap_intervals((start, end)))
    if len(intv_list) != 1:
        return False

    intv = intv_list[0]
    if intv[0] > start or intv[1] < end:
        # overlap interval did not completely cover (start, end)
        return False

    intv_set.remove(intv)
    if intv[0] < start:
        intv_set.add((intv[0], start))
    if end < intv[1]:
        intv_set.add((end, intv[1]))
    return True


def _substract(intv_list1, intv_list2):
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
    idx1 = idx2 = 0
    result = []
    intv1 = intv_list1[idx1]
    while idx1 < len(intv_list1) and idx2 < len(intv_list2):
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
    intv_set_list : list[IntervalSet]
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

    # substrate adjacent conn_list elements
    # make it so conn_list[x] contains intervals where you can connect exactly x+1 dummies vertically
    for idx in xrange(len(conn_list) - 1):
        cur_conn, next_conn = conn_list[idx], conn_list[idx + 1]
        conn_list[idx] = _substract(cur_conn, next_conn)

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
    unconnected : list[IntervalSet]
        a list of unconnected dummy finger intervals.  index 0 is the row closest to substrate.
    all_conn_list : list[(int, int)]
        a list of dummy finger intervals where you can connect from bottom substrate to top substrate.

    Returns
    -------
    select_list : list[list[(int, int)]]
        a list of list of intervals.  select_list[x] contains the finger intervals to
        draw dummy connections for x+1 rows from substrate.
    gate_intv_set_list : list[IntervalSet]
        a list of IntervalSets.  gate_intv_set_list[x] contains the finger intervals to
        draw dummy gate connections.
    """
    if all_conn_list:
        select_list = [all_conn_list]
        gate_intv_set_list = [IntervalSet(intv_list=all_conn_list)]
    else:
        select_list = []
        gate_intv_set_list = []
    for idx in xrange(len(conn_list) - 1, -1, -1):
        conn_intvs = conn_list[idx]
        cur_select_list = []
        # select connections
        for intv in conn_intvs:
            select = False
            for j in xrange(idx + 1):
                dummy_intv_set = unconnected[j]
                if dummy_intv_set.has_overlap(intv):
                    select = True
                    break
            if select:
                cur_select_list.append(intv)
        # remove all dummies connected with selected connections
        for intv in cur_select_list:
            for j in xrange(idx + 1):
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


class AmplifierBase(MicroTemplate):
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
    grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.  Must have the following entries:
    used_names : set[str]
        a set of already used cell names.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid, lib_name, params, used_names):
        MicroTemplate.__init__(self, grid, lib_name, params, used_names)
        # get the concrete template classes
        self._mos_cls = grid.tech_info.get_process_param('mos_template')
        self._sub_cls = grid.tech_info.get_process_param('sub_template')
        self._mconn_cls = grid.tech_info.get_process_param('mos_conn_template')
        self._sep_cls = grid.tech_info.get_process_param('mos_sep_template')
        self._dum_cls = grid.tech_info.get_process_param('mos_dummy_template')

        # get information from template classes
        self._dummy_layer = self._dum_cls.get_port_layer()
        self._min_fg_sep = self._sep_cls.get_min_fg()

        # initialize parameters
        self._lch = None
        self._orient_list = None
        self._w_list = None
        self._sd_list = None
        self._sd_pitch = None
        self._fg_tot = None
        self._track_width = None
        self._track_space = None
        self._gds_space = None
        self._n_intvs = None
        self._p_intvs = None
        self._num_tracks = None
        self._track_offsets = None
        self._ds_tr_indices = None
        self._vm_layer = None
        self._hm_layer = None
        self._bsub_port = None  # type: Port
        self._tsub_port = None  # type: Port

    @property
    def min_fg_sep(self):
        """Returns the minimum number of separator fingers."""
        return self._min_fg_sep

    def get_mos_params(self, mos_type, thres, lch, w, fg, g_tracks, ds_tracks, gds_space):
        """Returns a dictionary of mosfet parameters.

        Override if you need to include process-specific parameters.

        Parameters
        ----------
        mos_type : str
            mosfet type.  'pch' or 'nch'.
        thres : str
            threshold flavor.
        lch : float
            channel length.
        w : int or float
            width of the transistor.
        fg : int
            number of fingers.
        g_tracks : int
            minimum number of gate tracks.
        ds_tracks : int
            minimum number of drain/source tracks.
        gds_space : int
            number of tracks to reserve as space between gate and drain/source tracks.

        Returns
        -------
        mos_params : dict[str, any]
            the mosfet parameter dictionary.
        """
        return dict(mos_type=mos_type,
                    threshold=thres,
                    lch=lch,
                    w=w,
                    fg=fg,
                    track_width=self._track_width,
                    track_space=self._track_space,
                    g_tracks=g_tracks,
                    ds_tracks=ds_tracks,
                    gds_space=gds_space,
                    )

    def get_substrate_params(self, sub_type, thres, lch, w, fg):
        """Returns a dictionary of substrate parameters.

        Override if you need to include process-specific parameters.

        Parameters
        ----------
        sub_type : str
            subtract type.  'ptap' or 'ntap'.
        thres : str
            threshold flavor.
        lch : float
            channel length.
        w : int or float
            width of the substrate.
        fg : int
            number of fingers.

        Returns
        -------
        substrate_params : dict[str, any]
            the substrate parameter dictionary.
        """
        return dict(sub_type=sub_type,
                    threshold=thres,
                    lch=lch,
                    w=w,
                    fg=fg,
                    track_width=self._track_width,
                    track_space=self._track_space,
                    )

    def get_track_yrange(self, row_idx, tr_type, tr_idx):
        """Calculate the track bottom and top coordinate.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom-most NMOS/PMOS row.  -1 is bottom substrate.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        tr_idx : int
            the track index.

        Returns
        -------
        yb : float
            the bottom coordinate.
        yt : float
            the top coordinate.
        """
        row_idx += 1
        offset = self._track_offsets[row_idx]
        if tr_type == 'g':
            row_offset = 0
            ntr = self._ds_tr_indices[row_idx] - self._gds_space
        else:
            row_offset = self._ds_tr_indices[row_idx]
            ntr = self._num_tracks[row_idx] - row_offset

        if tr_idx < 0 or tr_idx >= ntr:
            raise ValueError('track index = %d out of bounds: [0, %d)' % (tr_idx, ntr))

        if self._orient_list[row_idx] == 'R0':
            tr_idx2 = tr_idx + offset + row_offset
        else:
            tr_idx2 = offset + self._num_tracks[row_idx] - (row_offset + tr_idx) - 1

        layout_unit = self.grid.layout_unit

        tr_sp = self._track_space / layout_unit
        tr_w = self._track_width / layout_unit
        tr_yb = tr_sp / 2.0 + tr_idx2 * (tr_sp + tr_w) + self.array_box.bottom
        tr_yt = tr_yb + tr_w
        return tr_yb, tr_yt

    def connect_to_supply(self, layout, supply_idx, port_list):
        """Connect the given transistor wires to supply.
        
        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        supply_idx : int
            the supply index.  0 for the bottom substrate, 1 for the top substrate.
        port_list : list[bag.layout.util.Port]
            list of Ports to connect to supply.
        """
        wire_yb, wire_yt = None, None
        if supply_idx == 0:
            wire_yb = self._bsub_port.get_bounding_box(self._vm_layer).bottom
        else:
            wire_yt = self._tsub_port.get_bounding_box(self._vm_layer).top

        # convert port list to list of BBoxArray
        # assuming each port only has a single layer
        box_arr_list = list(chain(*(port.get_pins().__iter__() for port in port_list)))

        self._connect_vertical_wires(layout, box_arr_list, wire_yb=wire_yb, wire_yt=wire_yt)

    def connect_differential_track(self, layout, p_port_list, n_port_list, row_idx, tr_type, ptr_idx, ntr_idx):
        """Connect the given differential wires to two tracks.

        Will make sure the connects are symmetric and have identical parasitics.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        p_port_list : list[bag.layout.util.Port]
            the list of positive ports to connect.
        n_port_list : list[bag.layout.util.Port]
            the list of negative ports to connect.
        row_idx : int
            the row index.  0 is the bottom-most NMOS/PMOS row.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        ptr_idx : int
            the positive track index.
        ntr_idx : int
            the negative track index

        Returns
        -------
        tr_layer : str
            the horizontal tracks metal layer.
        p_box : bag.layout.util.BBox
            the positive horizontal track bounding box.
        n_box : bag.layout.util.BBox
            the negative horizontal track bounding box.
        """
        if not p_port_list:
            return

        res = self.grid.resolution

        tr_ybp, tr_ytp = self.get_track_yrange(row_idx, tr_type, ptr_idx)
        tr_ybn, tr_ytn = self.get_track_yrange(row_idx, tr_type, ntr_idx)

        # the ports should all be just BBoxArray on the same layer.
        test_box_arr = p_port_list[0].get_pins().as_bbox_array()
        wire_w = test_box_arr.base.width

        # make test via to get extensions
        tr_w = self._track_width / self.grid.layout_unit
        via_test = self.grid.make_via_from_bbox(BBox(0.0, tr_ybp, wire_w, tr_ytp, res),
                                                self._vm_layer, self._hm_layer, 'y')
        yext = max((via_test.bot_box.height - tr_w) / 2.0, 0.0)
        xext = max((via_test.top_box.width - wire_w) / 2.0, 0.0)

        # get track X coordinates
        tr_xl = None
        tr_xr = None
        for port in chain(p_port_list, n_port_list):
            ba = port.get_pins().as_bbox_array()
            if tr_xl is None:
                tr_xl = ba.left
                tr_xr = ba.right
            else:
                tr_xl = min(ba.left, tr_xl)
                tr_xr = max(ba.right, tr_xr)

        tr_xl -= xext
        tr_xr += xext
        wire_yb = min(tr_ybp, tr_ybn) - yext
        wire_yt = max(tr_ytp, tr_ytn) + yext

        # draw the connections
        tr_layer, p_box = self.connect_to_track(layout, p_port_list, row_idx, tr_type, ptr_idx,
                                                wire_yb=wire_yb, wire_yt=wire_yt, tr_xl=tr_xl, tr_xr=tr_xr)
        _, n_box = self.connect_to_track(layout, n_port_list, row_idx, tr_type, ntr_idx,
                                         wire_yb=wire_yb, wire_yt=wire_yt, tr_xl=tr_xl, tr_xr=tr_xr)

        return tr_layer, p_box, n_box

    def connect_to_track(self, layout, port_list, row_idx, tr_type, track_idx,
                         wire_yb=None, wire_yt=None, tr_xl=None, tr_xr=None):
        """Connect the given wires to the track on the given row.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        port_list : list[bag.layout.util.Port]
            the list of ports to connect.
        row_idx : int
            the row index.  0 is the bottom-most NMOS/PMOS row.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        track_idx : int
            the track index.
        wire_yb : float or None
            if not None, extend wires to this bottom coordinate.  Used for differential routing.
        wire_yt : float or None
            if not None, extend wires to this top coordinate.  Used for differential routing.
        tr_xl : float or None
            if not None, extend track to this left coordinate.  Used for differential routing.
        tr_xr : float or None
            if not None, extend track to this right coordinate.  Used for differential routing.

        Returns
        -------
        track_layer : str
            the horizontal track metal layer.
        track_bbox : bag.layout.util.BBox
            the horizontal track bounding box.
        """
        if not port_list:
            # do nothing
            return

        res = self.grid.resolution

        # calculate track coordinates
        tr_yb, tr_yt = self.get_track_yrange(row_idx, tr_type, track_idx)
        wire_yb = tr_yb if wire_yb is None else min(wire_yb, tr_yb)
        wire_yt = tr_yt if wire_yt is None else max(wire_yt, tr_yt)

        # convert port list to list of BBoxArray
        # assuming each port only has a single layer
        box_arr_list = list(chain(*(port.get_pins().__iter__() for port in port_list)))

        # draw vertical wires
        wire_bus_list = self._connect_vertical_wires(layout, box_arr_list, wire_yb=wire_yb, wire_yt=wire_yt)
        wire_xl = wire_bus_list[0].left
        wire_xr = wire_bus_list[-1].right
        wire_w = wire_bus_list[0].base.width
        wire_pitch = wire_bus_list[0].spx

        # determine via intervals
        via_intv_list = []
        for wire_bus in wire_bus_list:
            cur_xl = wire_bus.left
            idx_start = int(round((cur_xl - wire_xl) / wire_pitch))
            if not via_intv_list or via_intv_list[-1][1] != idx_start:
                via_intv_list.append([idx_start, idx_start + wire_bus.nx])
            else:
                via_intv_list[-1][1] = idx_start + wire_bus.nx

        # draw vias
        tr_xr = wire_xr if tr_xr is None else max(wire_xr, tr_xr)
        tr_xl = wire_xl if tr_xl is None else min(wire_xl, tr_xl)
        for via_intv in via_intv_list:
            xo = wire_xl + via_intv[0] * wire_pitch
            via = self.grid.make_via_from_bbox(BBox(xo, tr_yb, xo + wire_w, tr_yt, res),
                                               self._vm_layer, self._hm_layer, 'y')
            arr_nx = via_intv[1] - via_intv[0]
            layout.add_via_obj(via, arr_nx=arr_nx, arr_spx=wire_pitch)
            # update track X coordinates.
            tr_xl = min(tr_xl, via.top_box.left)
            tr_xr = max(tr_xr, via.top_box.right + (arr_nx - 1) * wire_pitch)

        # draw horizontal track
        track_box = BBox(tr_xl, tr_yb, tr_xr, tr_yt, res)
        layout.add_rect(self._hm_layer, track_box)

        return self._hm_layer, track_box

    def _connect_vertical_wires(self, layout, box_arr_list, wire_yb=None, wire_yt=None,
                                wire_layer=''):
        """Connect the given wires together vertically.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        box_arr_list : list[bag.layout.util.BBoxArray]
            the list of bus wires to connect.
        wire_yb : float or None
            if not None, extend wires to this bottom coordinate.  Used for differential routing.
        wire_yt : float or None
            if not None, extend wires to this top coordinate.  Used for differential routing.
        wire_layer : str
            the wire layer.  If not given, defaults to self._vm_layer
        Returns
        -------
        wire_bus_list : list[bag.layout.util.BBoxArray]
            a list of wire buses created.
        """
        if not box_arr_list:
            # do nothing
            return []

        res = self.grid.resolution

        # make sure all wires are aligned
        wire_w = box_arr_list[0].base.width
        wire_pitch = box_arr_list[0].spx
        wire_xl = box_arr_list[0].left
        for box_arr in box_arr_list:
            if abs(wire_w - box_arr.base.width) >= res or abs(wire_pitch - box_arr.spx) >= res:
                raise ValueError('Not all wires have same width and pitch')
            cur_xl = box_arr.left
            nsep = int(round((cur_xl - wire_xl) / wire_pitch))
            if abs(cur_xl - wire_xl - nsep * wire_pitch) >= res:
                raise ValueError('Wires are not aligned properly.')
            wire_xl = min(wire_xl, cur_xl)

        # calculate wire vertical coordinates
        intv_set = IntervalSet(res=res)
        for box_arr in box_arr_list:
            cur_xl = box_arr.left
            # convert wire coordinates to track number
            nstart = int(round((cur_xl - wire_xl) / wire_pitch))
            nend = nstart + box_arr.nx
            # calculate wire bus bottom and top coordinate.
            cur_range = (box_arr.bottom, box_arr.top)
            if wire_yb is not None:
                cur_range = (min(cur_range[0], wire_yb), max(cur_range[1], wire_yb))
            if wire_yt is not None:
                cur_range = (min(cur_range[0], wire_yt), max(cur_range[1], wire_yt))
            ovl_item_list = list(intv_set.overlap_items((nstart, nend)))
            # perform max/min with other wire buses.
            if not ovl_item_list:
                intv_set.add((nstart, nend), val=cur_range)
            else:
                for intv, yrang in ovl_item_list:
                    intv_set.remove(intv)
                new_item_list = []
                ovl_end = ovl_item_list[-1][0][1]
                if ovl_end < nend:
                    new_item_list.append(((ovl_end, nend), cur_range))
                prev_mark = nstart
                for (cstart, cend), yrang in ovl_item_list:
                    if prev_mark < cstart:
                        new_item_list.append(((prev_mark, cstart), cur_range))
                    elif cstart < prev_mark:
                        # can only happen for first overlap item
                        new_item_list.append(((cstart, prev_mark), yrang))
                        cstart = prev_mark
                    new_item_list.append(((cstart, cend), (min(cur_range[0], yrang[0]),
                                                           max(cur_range[1], yrang[1]))))
                    prev_mark = cend

                for intv, yrang in new_item_list:
                    intv_set.add(intv, val=yrang)
        # draw vertical wires
        wire_layer = wire_layer or self._vm_layer
        wire_bus_list = []
        for intv, val in intv_set.items():
            xl = intv[0] * wire_pitch + wire_xl
            yb, yt = val
            box = BBox(xl, yb, xl + wire_w, yt, res)
            arr_nx = intv[1] - intv[0]
            layout.add_rect(wire_layer, box, arr_nx=arr_nx, arr_spx=wire_pitch)
            wire_bus_list.append(BBoxArray(box, nx=arr_nx, spx=wire_pitch))

        return wire_bus_list

    def _draw_dummy(self, layout, temp_db, row_idx, loc, fg, gate_intv_list, conn_right):
        """Draw dummy connection.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        row_idx : int
            the row index.  0 is the bottom-most NMOS.
        loc : str
            location of the dummy.  Either 'left' or 'right'.
        fg : int
            number of fingers.
        gate_intv_list : list[(int, int)]
            sorted list of gate intervals to connect gate to M2.
            for example, if gate_intv_list = [(2, 5)], then we will draw M2 connections
            between finger number 2 (inclusive) to finger number 5 (exclusive).
        conn_right : bool
            True to connect the right-most source/drain to supply.

        Returns
        -------
        wires : bag.layout.util.BBoxCollection
            the dummy gate bus wires.
        """
        col_idx = 0 if loc == 'left' else self._fg_tot - fg

        # mark transistors as connected
        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract_from_set(intv_set, col_idx, col_idx + fg):
            msg = 'Cannot connect transistors [%d, %d) on row %d; some are already connected.'
            raise ValueError(msg % (col_idx, col_idx + fg, row_idx))

        # skip bottom substrate
        idx = row_idx + 1
        orient = self._orient_list[idx]
        if loc == 'right':
            # reverse gate location to account for flip.
            gate_intv_list = [(fg - stop, fg - start) for start, stop in gate_intv_list]
        params = dict(
            lch=self._lch,
            w=self._w_list[idx],
            fg=fg,
            gate_intv_list=gate_intv_list,
            conn_right=conn_right,
        )

        xc, yc = self._sd_list[idx]
        if loc == 'right':
            xc += self._fg_tot * self._sd_pitch
            if orient == 'R0':
                orient = 'MY'
            else:
                orient = 'R180'

        conn = temp_db.new_template(params=params, temp_cls=self._dum_cls)  # type: AnalogMosDummy
        conn_loc = (xc, yc)
        self.add_template(layout, conn, loc=conn_loc, orient=orient)

        return conn.get_port().get_pins().transform(loc=conn_loc, orient=orient)

    def _draw_mos_sep(self, layout, temp_db, row_idx, col_idx, fg, gate_intv_list):
        """Draw transistor separator connection.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        row_idx : int
            the row index.  0 is the bottom-most NMOS.
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg : int
            number of separator fingers.  If less than the minimum, the minimum will be used instead.
        gate_intv_list : list[(int, int)]
            sorted list of gate intervals to connect gate to M2.
            for example, if gate_intv_list = [(2, 5)], then we will draw M2 connections
            between finger number 2 (inclusive) to finger number 5 (exclusive).

        Returns
        -------
        wires : bag.layout.util.BBoxCollection
            the dummy gate bus wires.
        """
        fg = max(fg, self.min_fg_sep)

        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract_from_set(intv_set, col_idx, col_idx + fg):
            msg = 'Cannot connect transistors [%d, %d) on row %d; some are already connected.'
            raise ValueError(msg % (col_idx, col_idx + fg, row_idx))

        # skip bottom substrate
        idx = row_idx + 1
        orient = self._orient_list[idx]
        params = dict(
            lch=self._lch,
            w=self._w_list[idx],
            fg=fg,
            gate_intv_list=gate_intv_list
        )

        xc, yc = self._sd_list[idx]
        xc += col_idx * self._sd_pitch
        conn = temp_db.new_template(params=params, temp_cls=self._sep_cls)  # type: AnalogMosSep
        conn_loc = (xc, yc)
        self.add_template(layout, conn, loc=conn_loc, orient=orient)

        return conn.get_port().get_pins().transform(loc=conn_loc, orient=orient)

    def draw_mos_conn(self, layout, temp_db, row_idx, col_idx, fg, sdir, ddir):
        """Draw transistor connection.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        row_idx : int
            the row index.  0 is the bottom-most NMOS.
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg : int
            number of fingers.
        sdir : int
            source connection direction.  0 for down, 1 for middle, 2 for up.
        ddir : int
            drain connection direction.  0 for down, 1 for middle, 2 for up.

        Returns
        -------
        ports : dict[str, bag.layout.util.Port]
            a dictionary of ports.  The keys are 'g', 'd', and 's'.
        """
        # mark transistors as connected
        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract_from_set(intv_set, col_idx, col_idx + fg):
            msg = 'Cannot connect transistors [%d, %d) on row %d; some are already connected.'
            raise ValueError(msg % (col_idx, col_idx + fg, row_idx))

        # skip bottom substrate
        idx = row_idx + 1
        orient = self._orient_list[idx]
        conn_params = dict(
            lch=self._lch,
            w=self._w_list[idx],
            fg=fg,
            sdir=sdir,
            ddir=ddir,
        )

        xc, yc = self._sd_list[idx]
        xc += col_idx * self._sd_pitch
        conn = temp_db.new_template(params=conn_params, temp_cls=self._mconn_cls)  # type: AnalogMosConn
        loc = (xc, yc)
        self.add_template(layout, conn, loc=loc, orient=orient)

        return {key: conn.get_port(key).transform(loc=loc, orient=orient) for key in ['g', 'd', 's']}

    def draw_base(self, layout, temp_db, lch, fg_tot, ptap_w, ntap_w,
                  nw_list, nth_list, pw_list, pth_list,
                  track_width, track_space, gds_space,
                  vm_layer, hm_layer,
                  ng_tracks=None, nds_tracks=None,
                  pg_tracks=None, pds_tracks=None):
        """Draw the amplifier base.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        lch : float
            the transistor channel length, in meters
        fg_tot : int
            total number of fingers for each row.
        ptap_w : int or float
            pwell substrate contact width.
        ntap_w : int or float
            nwell substrate contact width.
        track_width : float
            the routing track width.
        track_space : float
            the routing track spacing.
        gds_space : int
            number of tracks to reserve as space between gate and drain/source tracks.
        nw_list : list[int or float]
            a list of nmos width for each row, from bottom to top.
        nth_list: list[str]
            a list of nmos threshold flavor for each row, from bottom to top.
        pw_list : list[int or float]
            a list of pmos width for each row, from bottom to top.
        pth_list : list[str]
            a list of pmos threshold flavor for each row, from bottom to top.
        vm_layer : str
            vertical metal layer name.
        hm_layer : str
            horizontal metal layer name.
        ng_tracks : list[int] or None
            number of nmos gate tracks per row, from bottom to top.  Defaults to 1.
        nds_tracks : list[int] or None
            number of nmos drain/source tracks per row, from bottom to top.  Defaults to 1.
        pg_tracks : list[int] or None
            number of pmos gate tracks per row, from bottom to top.  Defaults to 1.
        pds_tracks : list[int] or None
            number of pmos drain/source tracks per row, from bottom to top.  Defaults to 1.

        Returns
        -------
        sub_layer : str
            the substrate horizontal track layer.
        bot_box_arr : bag.layout.util.BBoxArray
            the bottom substrate tracks bounding box array.
        top_box_arr : bag.layout.util.BBoxArray
            the top substrate tracks bounding box array.
        """

        # set default values.
        ng_tracks = ng_tracks or [1] * len(pw_list)
        nds_tracks = nds_tracks or [1] * len(pw_list)
        pg_tracks = pg_tracks or [1] * len(pw_list)
        pds_tracks = pds_tracks or [1] * len(pw_list)

        # initialize private attributes.
        self._lch = lch
        self._fg_tot = fg_tot
        self._orient_list = list(chain(repeat('R0', len(nw_list) + 1), repeat('MX', len(pw_list) + 1)))
        self._w_list = list(chain([ptap_w], nw_list, pw_list, [ntap_w]))
        self._sd_list = [(None, None)]  # type: list
        self._track_space = track_space
        self._track_width = track_width
        intv_init = [(0, fg_tot)]
        self._n_intvs = [IntervalSet(intv_list=intv_init) for _ in xrange(len(nw_list))]
        self._p_intvs = [IntervalSet(intv_list=intv_init) for _ in xrange(len(pw_list))]
        self._num_tracks = []
        self._track_offsets = []
        self._ds_tr_indices = []
        self._vm_layer = vm_layer
        self._hm_layer = hm_layer
        self._gds_space = gds_space

        # draw bottom substrate
        if nw_list:
            mos_type = 'ptap'
            sub_th = nth_list[0]
            sub_w = ptap_w
        else:
            mos_type = 'ntap'
            sub_th = pth_list[0]
            sub_w = ntap_w

        bsub_params = self.get_substrate_params(mos_type, sub_th, lch, sub_w, fg_tot)
        bsub = temp_db.new_template(params=bsub_params, temp_cls=self._sub_cls)  # type: AnalogSubstrate
        bsub_arr_box = bsub.array_box
        self.add_template(layout, bsub, 'XBSUB')
        self._num_tracks.append(bsub.get_num_tracks())
        self._bsub_port = bsub.get_port()
        self._track_offsets.append(0)
        self._ds_tr_indices.append(0)
        amp_array_box = bsub_arr_box

        ycur = bsub_arr_box.top
        # draw nmos and pmos
        mos = None
        for mos_type, w_list, th_list, g_list, ds_list in izip(['nch', 'pch'],
                                                               [nw_list, pw_list],
                                                               [nth_list, pth_list],
                                                               [ng_tracks, pg_tracks],
                                                               [nds_tracks, pds_tracks]):
            if mos_type == 'nch':
                fmt = 'XMN%d'
                orient = 'R0'
            else:
                fmt = 'XMP%d'
                orient = 'MX'
            for idx, (w, thres, gntr, dntr) in enumerate(izip(w_list, th_list, g_list, ds_list)):
                mos_params = self.get_mos_params(mos_type, thres, lch, w, fg_tot, gntr, dntr, gds_space)
                mos = temp_db.new_template(params=mos_params, temp_cls=self._mos_cls)  # type: AnalogMosBase
                mos_arr_box = mos.array_box
                amp_array_box = amp_array_box.merge(mos_arr_box)
                sd_xc, sd_yc = mos.get_left_sd_center()

                # compute ybot of mosfet
                if orient == 'MX':
                    ybot = ycur + mos_arr_box.top
                else:
                    ybot = ycur - mos_arr_box.bottom

                # add mosfet
                self.add_template(layout, mos, fmt % idx, loc=(0.0, ybot), orient=orient)
                ycur += mos_arr_box.height

                # calculate source/drain center location
                sd_yc = ybot + sd_yc if orient == 'R0' else ybot - sd_yc
                self._sd_list.append((sd_xc, sd_yc))
                self._track_offsets.append(self._track_offsets[-1] + self._num_tracks[-1])
                self._num_tracks.append(mos.get_num_tracks())
                self._ds_tr_indices.append(mos.get_ds_track_index())

        # record source/drain pitch
        self._sd_pitch = mos.get_sd_pitch()

        # draw last substrate
        if pw_list:
            mos_type = 'ntap'
            sub_th = pth_list[-1]
            sub_w = ntap_w
        else:
            mos_type = 'ptap'
            sub_th = nth_list[-1]
            sub_w = ptap_w

        tsub_params = self.get_substrate_params(mos_type, sub_th, lch, sub_w, fg_tot)
        tsub = temp_db.new_template(params=tsub_params, temp_cls=self._sub_cls)  # type: AnalogSubstrate
        tsub_arr_box = tsub.array_box
        self.array_box = amp_array_box.merge(tsub_arr_box)
        self._sd_list.append((None, None))
        self._track_offsets.append(self._track_offsets[-1] + self._num_tracks[-1])
        self._ds_tr_indices.append(0)
        self._num_tracks.append(tsub.get_num_tracks())
        tsub_loc = (0.0, ycur + tsub_arr_box.top)
        tsub_orient = 'MX'
        self.add_template(layout, tsub, 'XTSUB', loc=tsub_loc, orient=tsub_orient)
        self._tsub_port = tsub.get_port().transform(loc=tsub_loc, orient=tsub_orient)

        # connect substrates to horizontal tracks.
        return self._connect_substrate(layout, bsub.get_num_tracks(), tsub.get_num_tracks(),
                                       bsub.contact_both_ds(), tsub.contact_both_ds())

    def _connect_substrate(self, layout, nbot, ntop, bot_contact, top_contact):
        """Connect substrate to horizontal tracks

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        nbot : int
            number of bottom substrate tracks.
        ntop : int
            number of top substrate tracks.
        bot_contact : bool
            True to contact both drain/source for bottom substrate.
        top_contact : bool
            True to contact both drain/source for top substrate.

        Returns
        -------
        sub_layer : str
            the substrate horizontal track layer.
        bot_box_arr : bag.layout.util.BBoxArray
            the bottom substrate tracks bounding box array.
        top_box_arr : bag.layout.util.BBoxArray
            the top substrate tracks bounding box array.
        """
        res = self.grid.resolution
        layout_unit = self.grid.layout_unit
        track_pitch = (self._track_width + self._track_space) / layout_unit
        track_pitch = round(track_pitch / res) * res
        # row index substrate by 1 to make get_track_yrange work.
        # also skip the top track to leave some margin to transistors
        iter_list = [(-1, self._bsub_port, nbot - 1, 0, bot_contact),
                     (len(self._w_list) - 2, self._tsub_port, ntop - 1, ntop - 2,
                      top_contact)]
        sub_box_arr_list = []
        for row_idx, port, ntr, tr_idx, contact in iter_list:
            yb, yt = self.get_track_yrange(row_idx, 'ds', tr_idx)
            yb2, yt2 = self.get_track_yrange(row_idx, 'ds', ntr - 1 - tr_idx)

            box_arr = port.get_pins(self._vm_layer).as_bbox_array()

            xl = box_arr.base.left
            xr = box_arr.base.right
            via = self.grid.make_via_from_bbox(BBox(xl, yb, xr, yt, res),
                                               self._vm_layer, self._hm_layer, 'y')
            yext = (via.bot_box.height - (yt - yb)) / 2.0
            xext = (via.top_box.width - (xr - xl)) / 2.0
            # add via, tracks, and wires
            wbase = box_arr.base.extend(y=yb - yext).extend(y=yt2 + yext)
            layout.add_rect(self._vm_layer, wbase, arr_nx=box_arr.nx, arr_spx=box_arr.spx)
            tr_box = BBox(xl - xext, yb, box_arr.right + xext, yt, res)
            layout.add_rect(self._hm_layer, tr_box,
                            arr_ny=ntr, arr_spy=track_pitch)
            sub_box_arr_list.append(BBoxArray(tr_box, ny=ntr, spy=track_pitch))
            if contact:
                layout.add_via_obj(via, arr_nx=box_arr.nx, arr_spx=box_arr.spx,
                                   arr_ny=ntr, arr_spy=track_pitch)
            else:
                nx = int(np.ceil(box_arr.nx / 2.0))
                num_tr = int(np.ceil(ntr / 2.0))
                layout.add_via_obj(via, arr_nx=nx, arr_spx=2 * box_arr.spx,
                                   arr_ny=num_tr, arr_spy=2 * track_pitch)
                if ntr > 1:
                    via.move_by(box_arr.spx, track_pitch)
                    layout.add_via_obj(via, arr_nx=box_arr.nx - nx, arr_spx=2 * box_arr.spx,
                                       arr_ny=ntr - num_tr, arr_spy=2 * track_pitch)

        return self._hm_layer, sub_box_arr_list[0], sub_box_arr_list[1]

    def fill_dummy(self, layout, temp_db):
        """Draw dummy/separator on all unused transistors.

        This method should be called last.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        """

        # separate bottom/top dummies
        bot_intvs = self._p_intvs if not self._n_intvs else self._n_intvs
        top_intvs = self._n_intvs if not self._p_intvs else self._p_intvs
        top_intvs = list(reversed(top_intvs))

        bot_conn = _get_dummy_connections(bot_intvs)
        top_conn = _get_dummy_connections(top_intvs)
        bot_unconnected = [intv_set.copy() for intv_set in bot_intvs]

        # check if we have NMOS only or PMOS only
        # if so get intervals where we can connect from substrate-to-substrate
        if not self._n_intvs or not self._p_intvs:
            all_conn_list = bot_conn[-1]
            top_unconnected = bot_unconnected
            del bot_conn[-1]
            del top_conn[-1]

            # remove dummies connected by connections in all_conn_list
            for all_conn_intv in all_conn_list:
                for intv_set in bot_unconnected:  # type: IntervalSet
                    intv_set.remove_all_overlaps(all_conn_intv)

        else:
            top_unconnected = [intv_set.copy() for intv_set in top_intvs]
            all_conn_list = []

        # select connections
        if not self._n_intvs:
            # PMOS only, so we should prioritize top connections
            top_select, top_gintv = _select_dummy_connections(top_conn, top_unconnected, all_conn_list)
            bot_select, bot_gintv = _select_dummy_connections(bot_conn, bot_unconnected, all_conn_list)
        else:
            bot_select, bot_gintv = _select_dummy_connections(bot_conn, bot_unconnected, all_conn_list)
            top_select, top_gintv = _select_dummy_connections(top_conn, top_unconnected, all_conn_list)

        # make list of dummy gate connection parameters
        dummy_gate_conns = {}
        num_rows = len(self._n_intvs) + len(self._p_intvs)
        all_conn_set = IntervalSet(intv_list=all_conn_list)
        for loc_intvs, gintvs, sign in [(bot_intvs, bot_gintv, 1), (top_intvs, top_gintv, -1)]:
            for distance in xrange(len(loc_intvs)):
                ridx = distance if sign > 0 else num_rows - 1 - distance
                gate_intv_set = gintvs[distance]
                for dummy_intv in loc_intvs[distance]:
                    key = ridx, dummy_intv[0], dummy_intv[1]
                    overlaps = list(gate_intv_set.overlap_intervals(dummy_intv))
                    val_list = [0 if ovl_intv in all_conn_set else sign for ovl_intv in overlaps]
                    if key not in dummy_gate_conns:
                        dummy_gate_conns[key] = IntervalSet(intv_list=overlaps, val_list=val_list)
                    else:
                        dummy_gate_set = dummy_gate_conns[key]  # type: IntervalSet
                        for fg_intv, ovl_val in izip(overlaps, val_list):
                            if not dummy_gate_set.has_overlap(fg_intv):
                                dummy_gate_set.add(fg_intv, val=ovl_val)
                            else:
                                # check that we don't have conflicting gate connections.
                                for existing_intv, existing_val in dummy_gate_set.overlap_items(fg_intv):
                                    if existing_intv != fg_intv or existing_val != ovl_val:
                                        # this should never happen.
                                        raise Exception('Critical Error: report to developers.')

        wire_groups = {-1: [], 0: [], 1: []}
        for key, dummy_gate_set in dummy_gate_conns.iteritems():
            ridx, start, stop = key
            if not dummy_gate_set:
                raise Exception('Dummy (%d, %d) at row %d unconnected.' % (start, stop, ridx))
            fg = stop - start
            gate_intv_list = [(a - start, b - start) for a, b in dummy_gate_set]
            sub_val_iter = dummy_gate_set.values()
            if start == 0:
                conn_right = (stop == self._fg_tot)
                gate_buses = self._draw_dummy(layout, temp_db, ridx, 'left', fg, gate_intv_list, conn_right)
            elif stop == self._fg_tot:
                gate_buses = self._draw_dummy(layout, temp_db, ridx, 'right', fg, gate_intv_list, False)
                sub_val_iter = reversed(list(sub_val_iter))
            else:
                if fg < self.min_fg_sep:
                    raise ValueError('Cannot draw separator with fg = %d < %d' % (fg, self.min_fg_sep))
                gate_buses = self._draw_mos_sep(layout, temp_db, ridx, start, fg, gate_intv_list)

            for box_arr, sub_val in izip(gate_buses, sub_val_iter):
                wire_groups[sub_val].append(box_arr)

        sub_yb = self._bsub_port.get_bounding_box(self._dummy_layer).bottom
        sub_yt = self._tsub_port.get_bounding_box(self._dummy_layer).top

        for sub_idx, wire_bus_list in wire_groups.iteritems():
            wire_yb = sub_yb if sub_idx >= 0 else None
            wire_yt = sub_yt if sub_idx <= 0 else None
            self._connect_vertical_wires(layout, wire_bus_list, wire_yb=wire_yb, wire_yt=wire_yt,
                                         wire_layer=self._dummy_layer)
