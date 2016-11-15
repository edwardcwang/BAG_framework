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

import abc
from itertools import izip, chain, repeat
import bisect

from bag.layout.template import MicroTemplate
from bag.layout.util import BBox
from .analog_mos import AnalogMosBase, AnalogSubstrate, AnalogMosConn


class IntervalSet(object):
    """A data structure that keeps track of disjoint 1D intervals.

    Each interval can have an optional value associated with it.

    Parameters
    ----------
    intv_list : list[(float, float)] or None
        the initial interval list.
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

    def _get_overlap_idx(self, start, end):
        """Returns the index of the first interval that overlaps with the given interval.

        Parameters
        ----------
        start : int or float
            the start index of the given interval.  Inclusive.
        end : int or float
            th end index of the given interval.  Exclusive.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  If no overlapping intervals are
            found, -(idx + 1) is returned, where idx is the index to insert the interval.
        """
        if not self._start_list:
            return -1
        # find the largest start index less than or equal to start
        idx = bisect.bisect_left(self._start_list, start)
        if idx == 0:
            # all interval's starting point is greater than start
            return 0 if self._lt(self._start_list[idx], end) else -1

        test_idx = idx - 1
        if self._lt(start, self._end_list[test_idx]):
            # start is covered by the interval; overlaps.
            return test_idx
        elif test_idx + 1 < len(self._start_list) and self._lt(self._start_list[test_idx + 1], end):
            # the interval at idx + 1 covers end; overlaps
            return test_idx + 1
        else:
            # no overlap interval found
            return -(idx + 1)

    def get_overlap(self, start, end):
        """Returns the first interval that overlaps with the given interval.

        Parameters
        ----------
        start : int or float
            the start index of the given interval.  Inclusive.
        end : int or float
            th end index of the given interval.  Exclusive.

        Returns
        -------
        intv : (float, float) or None
            the first interval.  None if no overlapping intervals are found.
        value : any
            the value associated with the given interval.
        """
        idx = self._get_overlap_idx(start, end)
        if idx < 0:
            return None, None
        return (self._start_list[idx], self._end_list[idx]), self._val_list[idx]

    def remove(self, start, end):
        """Removes the given interval from this IntervalSet.

        Parameters
        ----------
        start : int or float
            the start index of the given interval.  Inclusive.
        end : int or float
            th end index of the given interval.  Exclusive.

        Returns
        -------
        success : bool
            True if the given interval is found and removed.  False otherwise.
        """
        idx = self._get_overlap_idx(start, end)
        if idx < 0:
            return False
        if abs(start - self._start_list[idx]) < self._res and abs(end - self._end_list[idx]) < self._res:
            del self._start_list[idx]
            del self._end_list[idx]
            del self._val_list[idx]
            return True
        return False

    def add(self, start, end, val=None):
        """Adds the given interval to this IntervalSet.

        Can only add interval that does not overlap with any existing ones.

        Parameters
        ----------
        start : int or float
            the start index of the given interval.  Inclusive.
        end : int or float
            th end index of the given interval.  Exclusive.
        val : any or None
            the value associated with the given interval.

        Returns
        -------
        success : bool
            True if the given interval is added.
        """
        idx = self._get_overlap_idx(start, end)
        if idx >= 0:
            return False
        idx = -idx - 1
        self._start_list.insert(idx, start)
        self._end_list.insert(idx, end)
        self._val_list.insert(idx, val)

    def set_value(self, start, end, value):
        """Sets the value associated with the given interval.

        Parameters
        ----------
        start : int or float
            the start index of the given interval.  Inclusive.
        end : int or float
            th end index of the given interval.  Exclusive.
        value : any or None
            the value associated with the given interval.

        Raises
        ------
        ValueError :
            if the given interval is not in this IntervalSet
        """
        idx = self._get_overlap_idx(start, end)
        if idx < 0 or abs(self._start_list[idx] - start) >= self._res or abs(self._end_list[idx] - end) >= self._res:
            raise ValueError('Interval (%.4g, %.4g) not in this interval set' % (start, end))
        self._val_list[idx] = value

    def get_intervals(self):
        """Returns a list of intervals in this IntervalSet.

        Returns
        -------
        intv_list : list[(int, int)]
            a list of intervals in this IntervalSet.
        """
        return list(izip(self._start_list, self._end_list))

    def get_values(self):
        """Returns a copy of the values list.

        Returns
        -------
        val_list : list[any]
            the list of values.
        """
        return list(self._val_list)


def _subtract(intv_set, start, end):
    """Substract the given interval from the interval set.

    Used to mark transistors as connected.

    Parameters
    ----------
    intv_set : IntervalSet
        the interval set.
    start : int
        the starting index.
    end : int
        the ending index.

    Returns
    -------
    success : bool
        True on success.
    """
    intv, _ = intv_set.get_overlap(start, end)
    if intv is None:
        return False
    if intv[0] > start or intv[1] < end:
        # overlap interval did not completely cover (start, end)
        return False

    intv_set.remove(intv[0], intv[1])
    if intv[0] < start:
        intv_set.add(intv[0], start)
    if end < intv[1]:
        intv_set.add(end, intv[1])
    return True


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
    mos_cls : class
        the transistor template class.
    sub_cls : class
        the substrate template class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid, lib_name, params, used_names,
                 mos_cls, sub_cls, mconn_cls, sep_cls, dum_cls):
        MicroTemplate.__init__(self, grid, lib_name, params, used_names)
        self._mos_cls = mos_cls
        self._sub_cls = sub_cls
        self._mconn_cls = mconn_cls
        self._sep_cls = sep_cls
        self._dum_cls = dum_cls
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

        self._min_fg_sep = self._sep_cls.get_min_fg()

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

    def connect_to_track(self, layout, box_list, row_idx, tr_type, track_idx):
        """Connect the given wires to the track on the given row.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        box_list : list[bag.layout.util.BBox]
            the list of wire bounding boxes.
        row_idx : int
            the row index.  0 is the bottom-most NMOS/PMOS row.
        tr_type : str
            the type of the track.  Either 'g' or 'ds'.
        track_idx : int
            the track index.
        """
        if not box_list:
            # do nothing
            return None

        res = self.grid.get_resolution()
        # first try to connect the wires vertically
        # do so by keeping track of intervals
        intv_set = IntervalSet(res=res)
        for box in box_list:
            intv, val = intv_set.get_overlap(box.left, box.right)
            if intv is None:
                # new interval
                intv_set.add(box.left, box.right, val=(box.bottom, box.top))
            elif abs(intv[0] - box.left) >= res or abs(intv[1] - box.right) >= res:
                # overlapping interval that's not equal
                msg = 'Wires (%.4g, %.4g) and (%.4g, %.4g) misaligned.'
                raise Exception(msg % (intv[0], intv[1], box.left, box.right))
            else:
                new_val = (min(val[0], box.bottom), max(val[1], box.top))
                intv_set.set_value(box.left, box.right, new_val)

        # get new box list with all vertically aligned wires connected
        box_list = [BBox(intv[0], val[0], intv[1], val[1], res)
                    for intv, val in izip(intv_set.get_intervals(), intv_set.get_values())]

        row_idx += 1
        offset = self._track_offsets[row_idx]
        if tr_type == 'g':
            row_offset = 0
            ntr = self._ds_tr_indices[row_idx] - self._gds_space
        else:
            row_offset = self._ds_tr_indices[row_idx]
            ntr = self._num_tracks[row_idx] - row_offset

        if track_idx < 0 or track_idx >= ntr:
            raise ValueError('track index = %d out of bounds: [0, %d)' % (track_idx, ntr))

        if self._orient_list[row_idx] == 'R0':
            track_idx += offset + row_offset
        else:
            track_idx = offset + self._num_tracks[row_idx] - (row_offset + track_idx)

        layout_unit = self.grid.get_layout_unit()
        tr_sp = self._track_space / layout_unit
        tr_w = self._track_width / layout_unit
        tr_yb = tr_sp / 2.0 + track_idx * (tr_sp + tr_w)
        tr_yt = tr_yb + tr_w
        tr_xl = box_list[0].left
        tr_xr = box_list[-1].right
        layout.add_rect(self._hm_layer, BBox(tr_xl, tr_yb, tr_xr, tr_yt, res))
        for box in box_list:
            yb = min(box.bottom, tr_yb)
            yt = max(box.top, tr_yt)
            layout.add_rect(self._vm_layer, BBox(box.left, yb, box.right, yt, res))
            vbox = BBox(box.left, tr_yb, box.right, tr_yt, res)
            via = self.grid.make_via_from_bbox(vbox, self._vm_layer, self._hm_layer, 'y')
            layout.add_via_obj(via)

    def draw_dummy(self, layout, temp_db, row_idx, loc, fg):
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
        """
        col_idx = 0 if loc == 'left' else self._fg_tot - fg

        # mark transistors as connected
        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract(intv_set, col_idx, col_idx + fg):
            msg = 'Cannot connect transistors [%d, %d) on row %d; some are already connected.'
            raise ValueError(msg % (col_idx, col_idx + fg, row_idx))

        # skip bottom substrate
        idx = row_idx + 1
        orient = self._orient_list[idx]
        params = dict(
            lch=self._lch,
            w=self._w_list[idx],
            fg=fg,
        )

        xc, yc = self._sd_list[idx]
        if loc == 'right':
            xc += self._fg_tot * self._sd_pitch
            if orient == 'R0':
                orient = 'MY'
            else:
                orient = 'R180'

        conn = temp_db.new_template(params=params, temp_cls=self._dum_cls)
        self.add_template(layout, conn, loc=(xc, yc), orient=orient)

    def draw_mos_sep(self, layout, temp_db, row_idx, col_idx, fg=0):
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

        Returns
        -------
        fg_tot : int
            total number of fingers used for separator.
        """
        fg = max(fg, self.min_fg_sep)

        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract(intv_set, col_idx, col_idx + fg):
            msg = 'Cannot connect transistors [%d, %d) on row %d; some are already connected.'
            raise ValueError(msg % (col_idx, col_idx + fg, row_idx))

        # skip bottom substrate
        idx = row_idx + 1
        orient = self._orient_list[idx]
        params = dict(
            lch=self._lch,
            w=self._w_list[idx],
            fg=fg,
        )

        xc, yc = self._sd_list[idx]
        xc += col_idx * self._sd_pitch
        conn = temp_db.new_template(params=params, temp_cls=self._sep_cls)
        self.add_template(layout, conn, loc=(xc, yc), orient=orient)

        return fg

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
        ports : dict[str, list[bag.layout.util.BBox]]
            a dictionary of port bounding boxes.  The keys are 'g', 'd', and 's'.
        """
        # mark transistors as connected
        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not _subtract(intv_set, col_idx, col_idx + fg):
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

        return dict(
            g=[box.transform(loc, orient) for box in conn.get_ports('g')],
            d=[box.transform(loc, orient) for box in conn.get_ports('d')],
            s=[box.transform(loc, orient) for box in conn.get_ports('s')],
        )

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
        self._sd_list = [(None, None)]
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
        self._track_offsets.append(0)
        self._ds_tr_indices.append(0)

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
        self._sd_list.append((None, None))
        self._track_offsets.append(self._track_offsets[-1] + self._num_tracks[-1])
        self._ds_tr_indices.append(0)
        self._num_tracks.append(tsub.get_num_tracks())
        self.add_template(layout, tsub, 'XTSUB', loc=(0.0, ycur + tsub_arr_box.top), orient='MX')

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
        for ridx, intv_set in enumerate(chain(self._n_intvs, self._p_intvs)):
            for intv in intv_set.get_intervals():
                fg = intv[1] - intv[0]
                if intv[0] == 0:
                    self.draw_dummy(layout, temp_db, ridx, 'left', fg)
                elif intv[1] == self._fg_tot:
                    self.draw_dummy(layout, temp_db, ridx, 'right', fg)
                else:
                    if fg < self.min_fg_sep:
                        raise ValueError('Cannot draw separator with fg = %d < %d' % (fg, self.min_fg_sep))
                    self.draw_mos_sep(layout, temp_db, ridx, intv[0], fg=fg)
