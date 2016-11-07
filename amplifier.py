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

from bag.layout.template import MicroTemplate
from .analog_mos import AnalogMosBase


class AmplifierBase(MicroTemplate):
    """The amplifier abstract template class

    An amplifier template consists of rows of pmos or nmos capped by substrate contacts.
    drain/source connections are mostly vertical, and gate connections are horizontal.  extension
    rows may be inserted to allow more space for gate/output connections.

    each row starts and ends with dummy transistors, and two transistors are always separated
    by separators.  Currently source sharing (e.g. diff pair) and inter-digitation are not
    supported.  All transistors have the same channel length.

    Parameters
    ----------
    grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    mos_cls : class
        the transistor template class.
    sub_cls : class
        the substrate template class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid, lib_name, params, used_names, mos_cls, sub_cls, mconn_cls, sep_cls):
        MicroTemplate.__init__(self, grid, lib_name, params, used_names)
        self.mos_cls = mos_cls
        self.sub_cls = sub_cls
        self.mconn_cls = mconn_cls
        self.sep_cls = sep_cls
        self.orient_list = None
        self.w_list = None
        self.sd_list = None
        self._sd_pitch = None

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
        track_width = self.params['track_width']
        track_space = self.params['track_space']
        return dict(mos_type=mos_type,
                    threshold=thres,
                    lch=lch,
                    w=w,
                    fg=fg,
                    track_width=track_width,
                    track_space=track_space,
                    g_tracks=g_tracks,
                    ds_tracks=ds_tracks,
                    )

    def get_substrate_params(self, mos_type, thres, lch, w, fg):
        """Returns a dictionary of substrate parameters.

        Override if you need to include process-specific parameters.

        Parameters
        ----------
        mos_type : str
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
        track_width = self.params['track_width']
        track_space = self.params['track_space']
        return dict(mos_type=mos_type,
                    threshold=thres,
                    lch=lch,
                    w=w,
                    fg=fg,
                    track_width=track_width,
                    track_space=track_space,
                    )

    def draw_mos_conn(self, layout, temp_db, row_idx, po_idx, fg, sdir, ddir):
        """Draw transistor connection.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        row_idx : int
            the row index.  0 is the bottom-most NMOS.
        po_idx : int
            the poly index.  0 is the left-most poly.
        fg : int
            number of fingers.
        sdir : int
            source connection direction.  0 for down, 1 for middle, 2 for up.
        ddir : int
            drain connection direction.  0 for down, 1 for middle, 2 for up.
        """
        # skip bottom substrate
        idx = row_idx + 1
        orient = self.orient_list[idx]
        conn_params = dict(
            lch=self.params['lch'],
            w=self.w_list[idx],
            fg=fg,
            sdir=sdir,
            ddir=ddir,
        )

        xc, yc = self.sd_list[idx]
        xc += po_idx * self._sd_pitch
        conn = temp_db.new_template(params=conn_params, temp_cls=self.mconn_cls)
        self.add_template(layout, conn, loc=(xc, yc), orient=orient)

    def draw_mos_sep(self, layout, temp_db, row_idx, po_idx):
        """Draw transistor separator connection.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        row_idx : int
            the row index.  0 is the bottom-most NMOS.
        po_idx : int
            the poly index.  0 is the left-most poly.
        """
        # skip bottom substrate
        idx = row_idx + 1
        orient = self.orient_list[idx]
        params = dict(
            lch=self.params['lch'],
            w=self.w_list[idx],
        )

        xc, yc = self.sd_list[idx]
        xc += po_idx * self._sd_pitch
        conn = temp_db.new_template(params=params, temp_cls=self.sep_cls)
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
            the transistor channel length.
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
            number of nmos drain/source tracks per row, from bottom to top.  Defaults to 0.
        pg_tracks : list[int] or None
            number of pmos gate tracks per row, from bottom to top.  Defaults to 1.
        pds_tracks : list[int] or None
            number of pmos drain/source tracks per row, from bottom to top.  Defaults to 0.
        """

        # set default values.
        ng_tracks = ng_tracks or [1] * len(pw_list)
        nds_tracks = nds_tracks or [1] * len(pw_list)
        pg_tracks = pg_tracks or [1] * len(pw_list)
        pds_tracks = pds_tracks or [1] * len(pw_list)

        self.orient_list = list(chain(repeat('R0', len(nw_list) + 1), repeat('MX', len(pw_list) + 1)))
        self.w_list = list(chain([ptap_w], nw_list, pw_list, [ntap_w]))
        self.sd_list = [(None, None)]

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
        bsub = temp_db.new_template(params=bsub_params, temp_cls=self.sub_cls)  # type: MicroTemplate
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
                mos = temp_db.new_template(params=mos_params, temp_cls=self.mos_cls)  # type: AnalogMosBase
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
                self.sd_list.append((sd_xc, sd_yc))

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
        tsub = temp_db.new_template(params=tsub_params, temp_cls=self.sub_cls)  # type: MicroTemplate
        tsub_arr_box = tsub.array_box
        self.sd_list.append((None, None))
        self.add_template(layout, tsub, 'XTSUB', loc=(0.0, ycur + tsub_arr_box.top), orient='MX')
