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
from .analog_mos import AnalogMosBase


class IntervalSet(object):
    """A data structure that keeps track of disjoint 1D intervals.

    This data structure keeps track of disjoint 1D intervals, and provides API for query/modification.
    this is used to keep track of which transistors has been connected and which hasn't.  Initially,
    an IntervalSet contains a single interval (0, length).

    Parameters
    ----------
    length : int
        the length of the original interval.
    """
    def __init__(self, length):
        self._start_list = [0]
        self._end_list = [length]

    def _get_overlap(self, start, end):
        """Returns the first interval that overlaps with the given interval.

        Parameters
        ----------
        start : int
            the start index of the given interval.  Inclusive.
        end : int
            th eend index of the given interval.  Exclusive.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  -1 if not found.
        """
        # find the largest start index less than or equal to start
        idx = bisect.bisect_left(self._start_list, start)
        if idx == 0:
            # all interval's starting point is greater than start
            return 0 if self._start_list[idx] < end else -1

        idx -= 1
        if start < self._end_list[idx]:
            # start is covered by the interval; overlaps.
            return idx
        elif idx + 1 < len(self._start_list) and self._start_list[idx + 1] < end:
            # the interval at idx + 1 covers end; overlaps
            return idx + 1
        else:
            # no overlap interval found
            return -1

    def subtract(self, start, end):
        """Subtracts the given interval from this IntervalSet.

        If there is no interval completely covering the given interval, False is returned and nothing is done.

        Parameters
        ----------
        start : int
            the start index of the given interval.  Inclusive.
        end : int
            th eend index of the given interval.  Exclusive.

        Returns
        -------
        success : bool
            True on subtraction, False if subtraction cannot be done.
        """
        idx = self._get_overlap(start, end)
        if idx < 0 or self._start_list[idx] > start or self._end_list[idx] < end:
            return False

        s = self._start_list[idx]
        e = self._end_list[idx]

        # remove old interval
        del self._start_list[idx]
        del self._end_list[idx]

        # insert interval 1
        if s < start:
            self._start_list.insert(idx, s)
            self._end_list.insert(idx, start)
            idx += 1
        if end < e:
            self._start_list.insert(idx, end)
            self._end_list.insert(idx, e)

        return True

    def get_intervals(self):
        """Returns a list of intervals in this IntervalSet.

        Returns
        -------
        intv_list : list[(int, int)]
            a list of intervals in this IntervalSet.
        """
        return list(izip(self._start_list, self._end_list))


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

    def __init__(self, grid, lib_name, params, used_names, mos_cls, sub_cls, mconn_cls, sep_cls, dum_cls):
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
        self._n_intvs = None
        self._p_intvs = None

        self._min_fg_sep = self._sep_cls.get_min_fg()

    @property
    def min_fg_sep(self):
        """Returns the minimum number of separator fingers."""
        return self._min_fg_sep

    def get_mos_params(self, mos_type, thres, lch, w, fg, g_tracks, ds_tracks):
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
        if not intv_set.subtract(col_idx, col_idx + fg):
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
        if not intv_set.subtract(col_idx, col_idx + fg):
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
        """
        # mark transistors as connected
        if row_idx >= len(self._n_intvs):
            intv_set = self._p_intvs[row_idx - len(self._n_intvs)]
        else:
            intv_set = self._n_intvs[row_idx]
        if not intv_set.subtract(col_idx, col_idx + fg):
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
        conn = temp_db.new_template(params=conn_params, temp_cls=self._mconn_cls)
        self.add_template(layout, conn, loc=(xc, yc), orient=orient)

    def draw_base(self, layout, temp_db, lch, fg_tot, ptap_w, ntap_w,
                  nw_list, nth_list, pw_list, pth_list,
                  track_width, track_space,
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
        nw_list : list[int or float]
            a list of nmos width for each row, from bottom to top.
        nth_list: list[str]
            a list of nmos threshold flavor for each row, from bottom to top.
        pw_list : list[int or float]
            a list of pmos width for each row, from bottom to top.
        pth_list : list[str]
            a list of pmos threshold flavor for each row, from bottom to top.
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
        self._n_intvs = [IntervalSet(fg_tot) for _ in xrange(len(nw_list))]
        self._p_intvs = [IntervalSet(fg_tot) for _ in xrange(len(pw_list))]

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
        bsub = temp_db.new_template(params=bsub_params, temp_cls=self._sub_cls)  # type: MicroTemplate
        bsub_arr_box = bsub.array_box
        self.add_template(layout, bsub, 'XBSUB')

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
                mos_params = self.get_mos_params(mos_type, thres, lch, w, fg_tot, gntr, dntr)
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
        tsub = temp_db.new_template(params=tsub_params, temp_cls=self._sub_cls)  # type: MicroTemplate
        tsub_arr_box = tsub.array_box
        self._sd_list.append((None, None))
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
