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
from itertools import izip

from .amplifier import AmplifierBase


class SerdesRXBase(AmplifierBase):
    """Subclass of AmplifierBase that draws serdes circuits.

    To use this class, draw_rows() must be the first function called, which will call draw_base() for you with
    the right arguments.

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

    _row_names = ['tail', 'en', 'sw', 'in', 'casc', 'load']

    def __init__(self, grid, lib_name, params, used_names):
        AmplifierBase.__init__(self, grid, lib_name, params, used_names)
        self._row_idx = None

    def draw_dynamic_latch(self, layout, temp_db, col_idx, fg_list, fg_sep=0):
        """Draw dynamic latch.

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_list : list[int]
            a 6-element list of the single-sided number of fingers per row.  They are
            [fg_tail, fg_en, fg_sw, fg_in, fg_cas, fg_load].  Use fg=0 to disable.
        fg_sep : int
            number of separator fingers.  If less than the minimum, the minimum will be used instead.
        """
        fg_sep = max(fg_sep, self.min_fg_sep)
        fg_max = max(fg_list) * 2 + fg_sep

        # figure out source/drain directions and intermediate connections
        # load
        sd_dir = {'load': (0, 2)}
        conn = {'outp': [('loadp', 'd')], 'outn': [('loadn', 'd')],
                'vdd': [('loadp', 's'), ('loadn', 's')],
                'biasp': [('loadp', 'g'), ('loadn', 'g')]}
        track = {'outp': (5, 0), 'outn': (5, 2), 'biasp': (5, 0)}

        # cascode and input
        if fg_list[4] > 0:
            # if cascode, flip input source/drain
            sd_dir['casc'] = (0, 2)
            sd_dir['in'] = (2, 0)
            conn['midp'] = [('cascp', 's'), ('inp', 's')]
            conn['midn'] = [('cascn', 's'), ('inn', 's')]
            track['midp'] = (4, 0)
            track['midn'] = (4, 0)
            conn['outp'].append(('cascp', 'd'))
            conn['outn'].append(('cascn', 'd'))
            conn['tail'] = [('inp', 'd'), ('inn', 'd')]
            conn['casc'] = [('cascp', 'g'), ('cascn', 'g')]
            track['casc'] = (4, 0)
        else:
            sd_dir['in'] = (0, 2)
            conn['outp'].append(('inp', 'd'))
            conn['outn'].append(('inn', 'd'))
            conn['tail'] = [('inp', 's'), ('inn', 's')]

        conn['inp'] = [('inn', 'g')]
        conn['inn'] = [('inp', 'g')]
        track['inp'] = (3, 2)
        track['inn'] = (3, 0)

        # switch
        if fg_list[2] > 0:
            # switch follows input direction
            track['vddt'] = (2, 0)
            conn['sw'] = [('swp', 'g'), ('swn', 'g')]
            track['sw'] = (2, 0)
            if sd_dir['in'][0] == 0:
                sd_dir['sw'] = (0, 1)
                conn['vddt'] = [('swp', 'd'), ('swn', 'd')]
                conn['tail'].extend([('swp', 's'), ('swn', 's')])
            else:
                sd_dir['sw'] = (1, 0)
                conn['vddt'] = [('swp', 's'), ('swn', 's')]
                conn['tail'].extend([('swp', 'd'), ('swn', 'd')])

        # enable
        if fg_list[1] > 0:
            # enable is opposite of input direction
            track['tail'] = (1, 0)
            conn['en'] = [('enp', 'g'), ('enn', 'g')]
            track['en'] = (1, 0)
            if sd_dir['in'][0] == 0:
                sd_dir['en'] = (2, 0)
                conn['tail'].extend([('enp', 's'), ('enn', 's')])
                conn['foot'] = [('enp', 'd'), ('enn', 'd')]
            else:
                sd_dir['en'] = (0, 2)
                conn['tail'].extend([('enp', 'd'), ('enn', 'd')])
                conn['foot'] = [('enp', 's'), ('enn', 's')]

        # tail
        if 'foot' in conn:
            # enable exists.  direction opposite of enable
            key = 'foot'
            comp = 'en'
        else:
            # direction opposite of in.
            key = 'tail'
            comp = 'in'

        track['biasn'] = (0, 0)
        conn['biasn'] = [('tailp', 'g'), ('tailn', 'g')]
        track[key] = (0, 0)
        if sd_dir[comp][0] == 0:
            sd_dir['tail'] = (2, 0)
            conn[key].extend([('tailp', 's'), ('tailn', 's')])
            conn['vss'] = [('tailp', 'd'), ('tailn', 'd')]
        else:
            sd_dir['tail'] = (0, 2)
            conn[key].extend([('tailp', 'd'), ('tailn', 'd')])
            conn['vss'] = [('tailp', 's'), ('tailn', 's')]

        # create mos connections
        mos_dict = {}
        for ridx, fg, name in izip(self._row_idx, fg_list, self._row_names):
            if ridx < 0:
                # error checking
                if fg > 0:
                    raise ValueError('Row %s does not exist but fg = %d > 0' % (name, fg))
            elif fg > 0:
                fg_tot = 2 * fg + fg_sep
                col_start = col_idx + (fg_max - fg_tot) / 2
                sdir, ddir = sd_dir[name]
                mos_dict['%sp' % name] = self.draw_mos_conn(layout, temp_db, ridx, col_start, fg, sdir, ddir)
                mos_dict['%sn' % name] = self.draw_mos_conn(layout, temp_db, ridx, col_start + fg + fg_sep,
                                                            fg, sdir, ddir)

        # draw differential connections
        for diff_sig, conn_type in [('in', 'g'), ('out', 'ds')]:
            pname = '%sp' % diff_sig
            nname = '%sn' % diff_sig
            ridx, ptr_idx = track[pname]
            _, ntr_idx = track[nname]
            p_port_list = [mos_dict[mos][sd] for mos, sd in conn.pop(pname)]
            n_port_list = [mos_dict[mos][sd] for mos, sd in conn.pop(nname)]
            self.connect_differential_track(layout, p_port_list, n_port_list, ridx, conn_type, ptr_idx, ntr_idx)

        # draw intermediate connections
        for conn_name, conn_list in conn.iteritems():
            port_list = [mos_dict[mos][sd] for mos, sd in conn_list]
            if conn_name == 'vdd':
                self.connect_to_supply(layout, 1, port_list)
            elif conn_name == 'vss':
                self.connect_to_supply(layout, 0, port_list)
            else:
                conn_type = 'g' if conn_list[0][1] == 'g' else 'ds'
                ridx, tidx = track[conn_name]
                self.connect_to_track(layout, port_list, ridx, conn_type, tidx)

    def draw_rows(self, layout, temp_db, lch, fg_tot, ptap_w, ntap_w,
                  nw_list, nth_list, pw, pth, track_width, track_space, gds_space,
                  vm_layer, hm_layer, ng_tracks, nds_tracks, pg_tracks, pds_tracks):
        """Draw the transistors and substrate rows.

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
        nw_list : list[int or float]
            a 5-element list of the nmos widths.  They are
            [wtail, wen, wsw, win, wcas].  Use w=0 to disable the corresponding transistor.
        nth_list: list[str]
            a 5-element list of the nmos threshold flavors.  Use empty string
            for place holders.
        pw : int or float
            the pmos width.
        pth : str
            the pmos threshold flavor.
        track_width : float
            the routing track width.
        track_space : float
            the routing track spacing.
        gds_space : int
            number of tracks to reserve as space between gate and drain/source tracks.
        vm_layer : str
            vertical metal layer name.
        hm_layer : str
            horizontal metal layer name.
        ng_tracks : list[int]
            a 5-element list of the nmos gate tracks per row.  Use 0 for place holders.
        nds_tracks : list[int]
            a 5-element list of the nmos drain/source tracks per row.  Use 0 for place holders.
        pg_tracks : int
            number of pmos gate tracks.
        pds_tracks : int
            number of pmos drain/source tracks.
        """

        # eliminate unneeded nmos rows and build row index.
        new_nw_list = [nw_list[0]]
        new_nth_list = [nth_list[0]]
        new_ng_tracks = [ng_tracks[0]]
        new_nds_tracks = [nds_tracks[0]]

        self._row_idx = [0, -1, -1, -1, -1, -1]
        cur_row = 1

        # set enable row index
        if nw_list[1] > 0:
            self._row_idx[1] = cur_row
            cur_row += 1
            new_nw_list.append(nw_list[1])
            new_nth_list.append(nth_list[1])
            new_ng_tracks.append(ng_tracks[1])
            new_nds_tracks.append(nds_tracks[1])

        # set switch row index
        if nw_list[2] > 0:
            self._row_idx[2] = cur_row
            cur_row += 1
            new_nw_list.append(nw_list[2])
            new_nth_list.append(nth_list[2])
            new_ng_tracks.append(ng_tracks[2])
            new_nds_tracks.append(nds_tracks[2])

        # set input row index
        self._row_idx[3] = cur_row
        cur_row += 1
        new_nw_list.append(nw_list[3])
        new_nth_list.append(nth_list[3])
        new_ng_tracks.append(ng_tracks[3])
        new_nds_tracks.append(nds_tracks[3])

        # set cascode row index
        if nw_list[4] > 0:
            self._row_idx[4] = cur_row
            cur_row += 1
            new_nw_list.append(nw_list[4])
            new_nth_list.append(nth_list[4])
            new_ng_tracks.append(ng_tracks[4])
            new_nds_tracks.append(nds_tracks[4])

        # set load row index
        self._row_idx[5] = cur_row

        # draw base
        self.draw_base(layout, temp_db, lch, fg_tot, ptap_w, ntap_w,
                       new_nw_list, new_nth_list, [pw], [pth],
                       track_width, track_space, gds_space, vm_layer, hm_layer,
                       ng_tracks=new_ng_tracks, nds_tracks=new_nds_tracks,
                       pg_tracks=[pg_tracks], pds_tracks=[pds_tracks])


class DynamicLatchChain(SerdesRXBase):
    """A chain of dynamic latches.

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

    def __init__(self, grid, lib_name, params, used_names):
        SerdesRXBase.__init__(self, grid, lib_name, params, used_names)

    def draw_layout(self, layout, temp_db):
        """Draw the layout of a dynamic latch chain.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        """
        kwargs = self.params.copy()
        fg_list = kwargs.pop('fg_list')
        nstage = kwargs.pop('nstage')
        ndum = kwargs.pop('ndum')

        if nstage <= 0:
            raise ValueError('nstage = %d must be greater than 0' % nstage)

        # calculate total number of fingers.
        fg_sep = self.min_fg_sep
        fg_latch = max(fg_list) * 2 + fg_sep
        kwargs['fg_tot'] = nstage * fg_latch + (nstage - 1) * fg_sep + 2 * ndum

        self.draw_rows(layout, temp_db, **kwargs)
        for idx in xrange(nstage):
            col_idx = (fg_latch + fg_sep) * idx + ndum
            self.draw_dynamic_latch(layout, temp_db, col_idx, fg_list, fg_sep=fg_sep)

        self.fill_dummy(layout, temp_db)

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            ng_tracks=[1, 1, 1, 3, 1],
            nds_tracks=[1, 1, 1, 1, 1],
            pg_tracks=1,
            pds_tracks=3,
            gds_space=1,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            nw_list='5-element list of NMOS widths, in meters/number of fins.',
            nth_list='5-element list of NMOS threshold flavors.',
            pw='PMOS width, in meters/number of fins.',
            pth='PMOS threshold flavor.',
            track_width='horizontal track width, in meters.',
            track_space='horizontal track spacing, in meters.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            vm_layer='vertical routing metal layer name.',
            hm_layer='horizontal routing metal layer name.',
            fg_list='6-element list of single-sided transistor number of fingers, from bottom to top.',
            nstage='number of dynamic latch stages.',
            ndum='number of dummy fingers at both ends.',
            ng_tracks='5-element list of number of NMOS gate tracks per row, from bottom to top.',
            nds_tracks='5-element list of number of NMOS drain/source tracks per row, from bottom to top.',
            pg_tracks='number of PMOS gate tracks.',
            pds_tracks='number of PMOS drain/source tracks.',
        )
