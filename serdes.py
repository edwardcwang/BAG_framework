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
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """
    __metaclass__ = abc.ABCMeta

    _row_names = ['tail', 'en', 'sw', 'in', 'casc', 'load']

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AmplifierBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._row_idx = None

    def draw_dynamic_latch(self, col_idx, fg_list, fg_sep=0):
        """Draw dynamic latch.

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_list : list[int]
            a 6-element list of the single-sided number of fingers per row.  They are
            [fg_tail, fg_en, fg_sw, fg_in, fg_cas, fg_load].  Use fg=0 to disable.
        fg_sep : int
            number of separator fingers.  If less than the minimum, the minimum will be used instead.

        Returns
        -------
        port_dict : dict[str, (str, bag.layout.util.BBox)]
            a dictionary from connection name to the horizontal track associated
            with the connection.
        """
        fg_sep = max(fg_sep, self.min_fg_sep)
        fg_max = max(fg_list) * 2 + fg_sep

        # figure out source/drain directions and intermediate connections
        # load
        sd_dir = {'load': (0, 2)}
        conn = {'outp': [('loadp', 'd')], 'outn': [('loadn', 'd')],
                'VDD': [('loadp', 's'), ('loadn', 's')],
                'bias_load': [('loadp', 'g'), ('loadn', 'g')]}

        out_ntr = self.get_num_tracks(self._row_idx[5], 'ds')
        track = {'outp': (5, out_ntr - 2 - self.params['diff_space']),
                 'outn': (5, out_ntr - 1), 'bias_load': (5, 0)}

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
            conn['bias_casc'] = [('cascp', 'g'), ('cascn', 'g')]
            track['bias_casc'] = (4, 0)
        else:
            sd_dir['in'] = (0, 2)
            conn['outp'].append(('inp', 'd'))
            conn['outn'].append(('inn', 'd'))
            conn['tail'] = [('inp', 's'), ('inn', 's')]

        conn['inp'] = [('inn', 'g')]
        conn['inn'] = [('inp', 'g')]
        in_ntr = self.get_num_tracks(self._row_idx[3], 'g')
        track['inp'] = (3, in_ntr - 1)
        track['inn'] = (3, in_ntr - 2 - self.params['diff_space'])

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
            conn['enable'] = [('enp', 'g'), ('enn', 'g')]
            track['enable'] = (1, 0)
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

        track['bias_tail'] = (0, 0)
        conn['bias_tail'] = [('tailp', 'g'), ('tailn', 'g')]
        track[key] = (0, 0)
        if sd_dir[comp][0] == 0:
            sd_dir['tail'] = (2, 0)
            conn[key].extend([('tailp', 's'), ('tailn', 's')])
            conn['VSS'] = [('tailp', 'd'), ('tailn', 'd')]
        else:
            sd_dir['tail'] = (0, 2)
            conn[key].extend([('tailp', 'd'), ('tailn', 'd')])
            conn['VSS'] = [('tailp', 's'), ('tailn', 's')]

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
                mos_dict['%sp' % name] = self.draw_mos_conn(ridx, col_start, fg, sdir, ddir)
                mos_dict['%sn' % name] = self.draw_mos_conn(ridx, col_start + fg + fg_sep,
                                                            fg, sdir, ddir)

        port_dict = {}

        # draw differential connections
        for diff_sig, conn_type in [('in', 'g'), ('out', 'ds')]:
            pname = '%sp' % diff_sig
            nname = '%sn' % diff_sig
            ridx, ptr_idx = track[pname]
            _, ntr_idx = track[nname]
            ridx = self._row_idx[ridx]
            p_port_list = [mos_dict[mos][sd] for mos, sd in conn.pop(pname)]
            n_port_list = [mos_dict[mos][sd] for mos, sd in conn.pop(nname)]
            sig_layer, pbox, nbox = self.connect_differential_track(p_port_list, n_port_list, ridx,
                                                                    conn_type, ptr_idx, ntr_idx)
            port_dict[pname] = (sig_layer, pbox)
            port_dict[nname] = (sig_layer, nbox)

        # draw intermediate connections
        for conn_name, conn_list in conn.iteritems():
            port_list = [mos_dict[mos][sd] for mos, sd in conn_list]
            if conn_name == 'VDD':
                self.connect_to_supply(1, port_list)
            elif conn_name == 'VSS':
                self.connect_to_supply(0, port_list)
            else:
                conn_type = 'g' if conn_list[0][1] == 'g' else 'ds'
                ridx, tidx = track[conn_name]
                ridx = self._row_idx[ridx]
                sig_layer, sig_box = self.connect_to_track(port_list, ridx, conn_type, tidx)
                port_dict[conn_name] = sig_layer, sig_box

        return port_dict

    def draw_rows(self, lch, fg_tot, ptap_w, ntap_w,
                  nw_list, nth_list, pw, pth, track_width, track_space, gds_space,
                  vm_layer, hm_layer, ng_tracks, nds_tracks, pg_tracks, pds_tracks):
        """Draw the transistors and substrate rows.

        Parameters
        ----------
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

        Returns
        -------
        sub_layer : str
            the substrate horizontal track layer.
        bot_box_arr : bag.layout.util.BBoxArray
            the bottom substrate tracks bounding box array.
        top_box_arr : bag.layout.util.BBoxArray
            the top substrate tracks bounding box array.
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
        return self.draw_base(lch, fg_tot, ptap_w, ntap_w,
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

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        SerdesRXBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @staticmethod
    def _rename_port(pname, idx, nstage):
        """Rename the given port."""
        if nstage == 1:
            return pname
        else:
            return '%s<%d>' % (pname, idx)

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        kwargs = self.params.copy()
        fg_list = kwargs.pop('fg_list')
        nstage = kwargs.pop('nstage')
        nduml = kwargs.pop('nduml')
        ndumr = kwargs.pop('ndumr')
        show_pins = kwargs.pop('show_pins')
        rename_dict = kwargs.pop('rename_dict')

        del kwargs['diff_space']

        if nstage <= 0:
            raise ValueError('nstage = %d must be greater than 0' % nstage)

        # calculate total number of fingers.
        fg_sep = self.min_fg_sep
        fg_latch = max(fg_list) * 2 + fg_sep
        kwargs['fg_tot'] = nstage * fg_latch + (nstage - 1) * fg_sep + nduml + ndumr

        slay, barr, tarr = self.draw_rows(**kwargs)

        port_list = [('VSS', (slay, barr)), ('VDD', (slay, tarr))]
        for idx in xrange(nstage):
            col_idx = (fg_latch + fg_sep) * idx + nduml
            pdict = self.draw_dynamic_latch(col_idx, fg_list, fg_sep=fg_sep)
            for pname, port_geo in pdict.iteritems():
                pname = rename_dict.get(pname, pname)
                if pname:
                    pin_name = self._rename_port(pname, idx, nstage)
                    port_list.append((pin_name, port_geo))

        for pname, (lay, box) in port_list:
            self.add_pin(pname, lay, box, show=show_pins)

        self.fill_dummy()

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
            diff_space=1,
            show_pins=True,
            rename_dict={}
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
            diff_space='number of tracks reserved as space between differential tracks.',
            vm_layer='vertical routing metal layer name.',
            hm_layer='horizontal routing metal layer name.',
            fg_list='6-element list of single-sided transistor number of fingers, from bottom to top.',
            nstage='number of dynamic latch stages.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            ng_tracks='5-element list of number of NMOS gate tracks per row, from bottom to top.',
            nds_tracks='5-element list of number of NMOS drain/source tracks per row, from bottom to top.',
            pg_tracks='number of PMOS gate tracks.',
            pds_tracks='number of PMOS drain/source tracks.',
            show_pins='True to create pin labels.',
            rename_dict='port renaming dictionary',
        )
