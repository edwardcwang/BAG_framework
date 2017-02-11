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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc

from .analog_core import AnalogBase


# noinspection PyAbstractClass
class SerdesRXBase(with_metaclass(abc.ABCMeta, AnalogBase)):
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

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._nrow_idx = None

    def draw_gm(self, col_idx, fg_in, fg_tail,
                fg_casc=0, fg_sw=0, fg_en=0,
                fg_sep=0, cur_track_width=1):
        """Draw a differential gm stage.

        a separator is used to separate the positive half and the negative half of the gm stage.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_in : int
            number of nmos input fingers (single-sided).
        fg_tail : int
            number of nmos tail fingers (single-sided).
        fg_casc : int
            number of nmos cascode fingers (single-sided).  0 to disable.
        fg_sw : int or float
            number of nmos tail switch fingers (single-sided).  0 to disable.
        fg_en : int or float
            number of nmos enable fingers (single-sided).  0 to disable.
        fg_sep : int
            number of separator fingers.  If less than the minimum, the minimum will be used instead.
        cur_track_width : int
            width of the current-carrying horizontal track wire in number of tracks.

        Returns
        -------
        port_dict : dict[str, :class:`~bag.layout.routing.WireArray`]
            a dictionary from connection name to WireArray.  Outputs are on vertical layer,
            and rests are on the horizontal layer above that.
        """
        # error checking
        if fg_in <= 0 or fg_tail <= 0:
            raise ValueError('tail/input/load transistors num. fingers must be positive.')
        for fg, name in ((fg_casc, 'casc'), (fg_en, 'en'), (fg_sw, 'sw')):
            if fg > 0 and name not in self._nrow_idx:
                raise ValueError('nmos %s row is not drawn.' % name)

        fg_sep = max(fg_sep, self.min_fg_sep)
        fg_max = max(fg_in, fg_tail, fg_casc, fg_sw, fg_en) * 2 + fg_sep

        diff_space = self.params['diff_space']

        # figure out source/drain directions and intermediate connections
        # load always drain down.
        in_ntr = self.get_num_tracks('nch', self._nrow_idx['in'], 'g')
        sd_dir = {}
        conn = {}
        track = {}

        # cascode and input
        if fg_casc > 0:
            # if cascode, flip input source/drain
            sd_dir['casc'] = (0, 2)
            sd_dir['in'] = (2, 0)
            conn['midp'] = [('cascp', 's'), ('inp', 's')]
            conn['midn'] = [('cascn', 's'), ('inn', 's')]
            track['midp'] = ('nch', self._nrow_idx['casc'], 'ds', (cur_track_width - 1) / 2)
            track['midn'] = ('nch', self._nrow_idx['casc'], 'ds', (cur_track_width - 1) / 2)

            conn['tail'] = [('inp', 'd'), ('inn', 'd')]
            conn['bias_casc'] = [('cascp', 'g'), ('cascn', 'g')]
            track['bias_casc'] = ('nch', self._nrow_idx['casc'], 'g', 0)
        else:
            sd_dir['in'] = (0, 2)
            conn['tail'] = [('inp', 's'), ('inn', 's')]

        # switch
        if fg_sw > 0:
            # switch follows input direction
            conn['sw'] = [('swp', 'g'), ('swn', 'g')]
            if sd_dir['in'][0] == 0:
                sd_dir['sw'] = (0, 1)
                conn['vddt'] = [('swp', 'd'), ('swn', 'd')]
                conn['tail'].extend([('swp', 's'), ('swn', 's')])
            else:
                sd_dir['sw'] = (1, 0)
                conn['vddt'] = [('swp', 's'), ('swn', 's')]
                conn['tail'].extend([('swp', 'd'), ('swn', 'd')])

            track['vddt'] = ('nch', self._nrow_idx['sw'], 'ds', (cur_track_width - 1) / 2)
            track['sw'] = ('nch', self._nrow_idx['sw'], 'g', 0)

        # enable
        if fg_en > 0:
            # enable is opposite of input direction
            conn['enable'] = [('enp', 'g'), ('enn', 'g')]
            if sd_dir['in'][0] == 0:
                sd_dir['en'] = (2, 0)
                conn['tail'].extend([('enp', 's'), ('enn', 's')])
                conn['foot'] = [('enp', 'd'), ('enn', 'd')]
            else:
                sd_dir['en'] = (0, 2)
                conn['tail'].extend([('enp', 'd'), ('enn', 'd')])
                conn['foot'] = [('enp', 's'), ('enn', 's')]

            track['enable'] = ('nch', self._nrow_idx['en'], 'g', 0)
            track['tail'] = ('nch', self._nrow_idx['en'], 'ds', (cur_track_width - 1) / 2)

        # tail
        if 'foot' in conn:
            # enable exists.  direction opposite of enable
            key = 'foot'
            comp = 'en'
        else:
            # direction opposite of in.
            key = 'tail'
            comp = 'in'

        conn['bias_tail'] = [('tailp', 'g'), ('tailn', 'g')]
        if sd_dir[comp][0] == 0:
            sd_dir['tail'] = (2, 0)
            conn[key].extend([('tailp', 's'), ('tailn', 's')])
            conn['VSS'] = [('tailp', 'd'), ('tailn', 'd')]
        else:
            sd_dir['tail'] = (0, 2)
            conn[key].extend([('tailp', 'd'), ('tailn', 'd')])
            conn['VSS'] = [('tailp', 's'), ('tailn', 's')]

        track['bias_tail'] = ('nch', self._nrow_idx['tail'], 'g', 0)
        track[key] = ('nch', self._nrow_idx['tail'], 'ds', (cur_track_width - 1) / 2)

        # create mos connections
        mos_dict = {}
        for name, fg in zip(('casc', 'in', 'sw', 'en', 'tail'),
                            (fg_casc, fg_in, fg_sw, fg_en, fg_tail)):
            if fg > 0:
                fg_tot = 2 * fg + fg_sep
                col_start = col_idx + (fg_max - fg_tot) // 2
                sdir, ddir = sd_dir[name]
                ridx = self._nrow_idx[name]
                mos_dict['%sp' % name] = self.draw_mos_conn('nch', ridx, col_start, fg, sdir, ddir)
                mos_dict['%sn' % name] = self.draw_mos_conn('nch', ridx, col_start + fg + fg_sep,
                                                            fg, sdir, ddir)

        # get output WireArrays
        out_ntype = 'casc' if fg_casc > 0 else 'in'
        port_dict = dict(outp=mos_dict['%sp' % out_ntype]['d'],
                         outn=mos_dict['%sn' % out_ntype]['d'])

        # draw differential input connection
        ptr_idx = self.get_track_index('nch', self._nrow_idx['in'], 'g', in_ntr - 1)
        ntr_idx = self.get_track_index('nch', self._nrow_idx['in'], 'g', in_ntr - 2 - diff_space)
        p_tr, n_tr = self.connect_differential_tracks(mos_dict['inp']['g'], mos_dict['inn']['g'],
                                                      self.mos_conn_layer + 1, ptr_idx, ntr_idx)
        port_dict['inp'] = p_tr
        port_dict['inn'] = n_tr

        # draw intermediate connections
        for conn_name, conn_list in conn.items():
            warr_list = [mos_dict[mos][sd] for mos, sd in conn_list]
            if conn_name == 'VSS':
                self.connect_to_substrate('ptap', warr_list)
            else:
                if conn_list[0][1] == 'g':
                    tr_width = 1
                else:
                    tr_width = cur_track_width

                mos_type, ridx, tr_type, tr_idx = track[conn_name]
                tr_id = self.make_track_id(mos_type, ridx, tr_type, tr_idx, width=tr_width)
                sig_warr = self.connect_to_tracks(warr_list, tr_id)
                port_dict[conn_name] = sig_warr

        return port_dict

    def draw_diffamp(self, col_idx, fg_in, fg_tail, fg_load,
                     fg_casc=0, fg_sw=0, fg_en=0,
                     fg_sep=0, cur_track_width=1):
        """Draw a differential amplifier/dynamic latch.

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_in : int
            number of nmos input fingers (single-sided).
        fg_tail : int
            number of nmos tail fingers (single-sided).
        fg_load : int
            number of pmos load fingers (single-sided).
        fg_casc : int
            number of nmos cascode fingers (single-sided).  0 to disable.
        fg_sw : int or float
            number of nmos tail switch fingers (single-sided).  0 to disable.
        fg_en : int or float
            number of nmos enable fingers (single-sided).  0 to disable.
        fg_sep : int
            number of separator fingers.  If less than the minimum, the minimum will be used instead.
        cur_track_width : int
            width of the current-carrying horizontal track wire in number of tracks.

        Returns
        -------
        port_dict : dict[str, (str, bag.layout.routing.WireArray)]
            a dictionary from connection name to the horizontal track associated
            with the connection.
        """
        # error checking
        if fg_load <= 0:
            raise ValueError('load transistors num. fingers must be positive.')

        diff_space = self.params['diff_space']

        # compute Gm stage column index.
        fg_sep = max(fg_sep, self.min_fg_sep)
        fg_max_gm = max(fg_in, fg_tail, fg_casc, fg_sw, fg_en) * 2 + fg_sep
        fg_max_load = fg_load * 2 + fg_sep
        fg_max = max(fg_max_load, fg_max_gm)
        gm_col_idx = (fg_max - fg_max_gm) // 2 + col_idx

        # draw Gm.
        port_dict = self.draw_gm(gm_col_idx, fg_in, fg_tail, fg_casc=fg_casc,
                                 fg_sw=fg_sw, fg_en=fg_en, fg_sep=fg_sep,
                                 cur_track_width=cur_track_width)

        # draw load transistors
        load_col_idx = (fg_max - fg_max_load) // 2 + col_idx
        loadp = self.draw_mos_conn('pch', 0, load_col_idx, fg_load, 2, 0)
        loadn = self.draw_mos_conn('pch', 0, load_col_idx + fg_load + fg_sep, fg_load, 2, 0)

        # connect load gate bias
        tr_id = self.make_track_id('pch', 0, 'g', 0)
        warr = self.connect_to_tracks([loadp['g'], loadn['g']], tr_id)
        port_dict['bias_load'] = warr

        # connect differential outputs
        out_ntr = self.get_num_tracks('pch', 0, 'ds')
        ptr_idx = self.get_track_index('pch', 0, 'ds', out_ntr - 2 - diff_space)
        ntr_idx = self.get_track_index('pch', 0, 'ds', out_ntr - 1)

        p_tr, n_tr = self.connect_differential_tracks([loadp['d'], port_dict['outp']],
                                                      [loadn['d'], port_dict['outn']],
                                                      self.mos_conn_layer + 1,
                                                      ptr_idx, ntr_idx)
        port_dict['outp'] = p_tr
        port_dict['outn'] = n_tr

        # connect VDD
        self.connect_to_substrate('ntap', [loadp['s'], loadn['s']])

        return port_dict

    def draw_rows(self, lch, fg_tot, ptap_w, ntap_w,
                  w_in, w_tail, w_load,
                  w_casc=0, w_sw=0, w_en=0,
                  th_in='standard', th_tail='standard', th_load='standard',
                  th_casc='standard', th_sw='standard', th_en='standard',
                  **kwargs):
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
        w_in : int or float
            nmos input transistor row width.
        w_tail : int or float
            nmos tail transistor row width.
        w_load : int or float
            pmos load transistor row width.
        w_casc : int or float
            nmos cascode transistor row width.  0 to disable.
        w_sw : int or float
            nmos tail switch transistor row width.  0 to disable.
        w_en : int or float
            nmos enable transistor row width.  0 to disable.
        th_in : string
            nmos input transistor threshold flavor.
        th_tail : string
            nmos tail transistor threshold flavor.
        th_load : string
            pmos load transistor threshold flavor.
        th_casc : string
            nmos cascode transistor threshold flavor.
        th_sw : istring
            nmos tail switch transistor threshold flavor.
        th_en : string
            nmos enable transistor threshold flavor.
        kwargs : dict[str, any]
            any addtional parameters for AnalogBase's draw_base() method.
        """
        # error checking
        if w_tail <= 0 or w_in <= 0 or w_load <= 0:
            raise ValueError('tail/input/load transistors width must be positive.')

        # figure out row indices for each nmos row type,
        # and build nw_list/nth_list
        self._nrow_idx = {}
        nw_list = []
        nth_list = []
        cur_idx = 0
        for name, width, thres in zip(('tail', 'en', 'sw', 'in', 'casc'),
                                      (w_tail, w_en, w_sw, w_in, w_casc),
                                      (th_tail, th_en, th_sw, th_in, th_casc)):
            if width > 0:
                self._nrow_idx[name] = cur_idx
                nw_list.append(width)
                nth_list.append(thres)
                cur_idx += 1

        # draw base
        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, [w_load], [th_load], **kwargs)


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
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, ptap_w, ntap_w, w_dict, th_dict, fg_dict,
                            nstage, fg_sep, nduml, ndumr, global_gnd_layer, global_gnd_name,
                            diff_space, cur_track_width, show_pins, **kwargs):
        if nstage <= 0:
            raise ValueError('nstage = %d must be greater than 0' % nstage)

        # calculate total number of fingers.
        fg_sep = max(fg_sep, self.min_fg_sep)
        fg_latch = max(fg_dict.values()) * 2 + fg_sep
        fg_tot = nstage * fg_latch + (nstage - 1) * fg_sep + nduml + ndumr

        # figure out number of tracks
        kwargs['pg_tracks'] = [1]
        kwargs['pds_tracks'] = [2 + diff_space]
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'w_en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 + diff_space)
                else:
                    ng_tracks.append(1)
                nds_tracks.append(cur_track_width + kwargs['gds_space'])
        kwargs['ng_tracks'] = ng_tracks
        kwargs['nds_tracks'] = nds_tracks

        # draw rows with width/threshold parameters.
        for key, val in w_dict.items():
            kwargs['w_' + key] = val
        for key, val in th_dict.items():
            kwargs['th_' + key] = val
        self.draw_rows(lch, fg_tot, ptap_w, ntap_w, **kwargs)

        port_list = []

        da_kwargs = {'fg_' + key: val for key, val in fg_dict.items()}
        for idx in range(nstage):
            col_idx = (fg_latch + fg_sep) * idx + nduml
            pdict = self.draw_diffamp(col_idx, cur_track_width=cur_track_width, **da_kwargs)
            for pname, port_warr in pdict.items():
                pname = self.get_pin_name(pname)
                if pname:
                    pin_name = self._rename_port(pname, idx, nstage)
                    port_list.append((pin_name, port_warr))

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        # export supplies
        port_list.extend((('VSS', warr) for warr in ptap_wire_arrs))
        port_list.extend((('VDD', warr) for warr in ntap_wire_arrs))

        for pname, warr in port_list:
            self.add_pin(pname, warr, show=show_pins)

        # add global ground
        if global_gnd_layer is not None:
            _, global_gnd_box = next(ptap_wire_arrs[0].wire_iter(self.grid))
            self.add_pin_primitive(global_gnd_name, global_gnd_layer, global_gnd_box)

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
            th_dict={},
            gds_space=1,
            diff_space=1,
            nstage=1,
            fg_sep=0,
            nduml=4,
            ndumr=4,
            cur_track_width=1,
            show_pins=True,
            rename_dict={},
            guard_ring_nf=0,
            global_gnd_layer=None,
            global_gnd_name='gnd!',
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
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            fg_dict='NMOS/PMOS number of fingers dictionary.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            nstage='number of dynamic latch stages.',
            fg_sep='number of separator finger between stages',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            show_pins='True to create pin labels.',
            rename_dict='port renaming dictionary',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            global_gnd_layer='layer of the global ground pin.  None to disable drawing global ground.',
            global_gnd_name='name of global ground pin.',
        )
