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

from typing import Dict, Any, Set

from bag.layout.template import TemplateBase, TemplateDB

from abs_templates_ec.analog_core import AnalogBaseInfo
from abs_templates_ec.serdes import SerdesRXBase


class RXHalfTop(SerdesRXBase):
    """The top half of one data path of DDR burst mode RX core.

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
        super(RXHalfTop, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, ptap_w, ntap_w, w_dict, th_dict,
                            alat_params, intsum_params, summer_params,
                            fg_tot, global_gnd_layer, global_gnd_name,
                            show_pins, diff_space, cur_track_width, **kwargs):
        # figure out number of tracks
        kwargs['pg_tracks'] = [1]
        kwargs['pds_tracks'] = [2 + diff_space]
        ng_tracks = []
        nds_tracks = []

        # check if butterfly switches are used
        has_but = False
        gm_fg_list = intsum_params['gm_fg_list']
        for fdict in gm_fg_list:
            if fdict.get('fg_but', 0) > 0:
                has_but = True
                break
        if has_but:
            if diff_space % 2 != 1:
                raise ValueError('diff_space must be odd if butterfly switch is present.')
            # route cascode in between butterfly gates.
            gate_locs = {'bias_casc': (diff_space + 1) // 2,
                         'sgnp': diff_space + 1,
                         'sgnn': 0,
                         'inp': diff_space + 1,
                         'inn': 0}
        else:
            gate_locs = {'inp': diff_space + 1,
                         'inn': 0}

        # compute nmos gate/drain/source number of tracks
        for row_name in ['tail', 'w_en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in' or (row_name == 'casc' and has_but):
                    ng_tracks.append(2 + diff_space)
                else:
                    ng_tracks.append(1)
                nds_tracks.append(cur_track_width + kwargs['gds_space'])
        kwargs['ng_tracks'] = ng_tracks
        kwargs['nds_tracks'] = nds_tracks

        # draw rows with width/threshold parameters.
        self.draw_rows(lch, fg_tot, ptap_w, ntap_w, w_dict, th_dict, **kwargs)
        self.set_size_from_array_box(self.mos_conn_layer + 2)

        # draw blocks
        cur_col = alat_params.pop('col_idx')
        # print('rxtop alat cur_col: %d' % cur_col)
        fg_amp, alat_ports = self.draw_diffamp(cur_col, alat_params, cur_track_width=cur_track_width,
                                               diff_space=diff_space, gate_locs=gate_locs)
        cur_col = intsum_params.pop('col_idx')
        # print('rxtop intsum cur_col: %d' % cur_col)
        fg_intsum, intsum_ports = self.draw_gm_summer(cur_col, cur_track_width=cur_track_width,
                                                      diff_space=diff_space, gate_locs=gate_locs,
                                                      **intsum_params)
        cur_col = summer_params.pop('col_idx')
        # print('rxtop summer cur_col: %d' % cur_col)
        fg_summer, summer_ports = self.draw_gm_summer(cur_col, cur_track_width=cur_track_width,
                                                      diff_space=diff_space, gate_locs=gate_locs,
                                                      **summer_params)

        # add pins
        for name, warr_list in alat_ports.items():
            self.add_pin('alat_' + name, warr_list, show=show_pins)

        for (name, idx), warr_list in intsum_ports.items():
            if idx >= 0:
                self.add_pin('intsum_%s<%d>' % (name, idx), warr_list, show=show_pins)
            else:
                self.add_pin('intsum_%s' % name, warr_list, show=show_pins)

        for (name, idx), warr_list in summer_ports.items():
            if idx >= 0:
                self.add_pin('summer_%s<%d>' % (name, idx), warr_list, show=show_pins)
            else:
                self.add_pin('summer_%s' % name, warr_list, show=show_pins)

        # export supplies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        for warr in ptap_wire_arrs:
            self.add_pin('VSS', warr, show=show_pins)
        for warr in ntap_wire_arrs:
            self.add_pin('VDD', warr, show=show_pins)

        # add global ground
        if global_gnd_layer is not None:
            _, global_gnd_box = next(ptap_wire_arrs[0].wire_iter(self.grid))
            self.add_pin_primitive(global_gnd_name, global_gnd_layer, global_gnd_box)

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
            th_dict={},
            gds_space=1,
            diff_space=1,
            cur_track_width=1,
            show_pins=False,
            guard_ring_nf=0,
            global_gnd_layer=None,
            global_gnd_name='gnd!',
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
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            alat_params='Analog latch parameters',
            intsum_params='Integrator summer parameters.',
            summer_params='DFE tap-1 summer parameters.',
            fg_tot='Total number of fingers.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            global_gnd_layer='layer of the global ground pin.  None to disable drawing global ground.',
            global_gnd_name='name of global ground pin.',
        )


class RXHalfBottom(SerdesRXBase):
    """The bottom half of one data path of DDR burst mode RX core.

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
        super(RXHalfBottom, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, ptap_w, ntap_w, w_dict, th_dict,
                            integ_params, alat_params, dlat_params_list,
                            fg_tot, global_gnd_layer, global_gnd_name,
                            show_pins, diff_space, cur_track_width, **kwargs):

        # figure out number of tracks
        kwargs['pg_tracks'] = [1]
        kwargs['pds_tracks'] = [2 + diff_space]
        ng_tracks = []
        nds_tracks = []

        # compute nmos gate/drain/source number of tracks
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
        self.draw_rows(lch, fg_tot, ptap_w, ntap_w, w_dict, th_dict, **kwargs)
        self.set_size_from_array_box(self.mos_conn_layer + 2)

        gate_locs = {'inp': diff_space + 1,
                     'inn': 0}

        # draw blocks
        cur_col = integ_params.pop('col_idx')
        fg_integ, integ_ports = self.draw_diffamp(cur_col, integ_params, cur_track_width=cur_track_width,
                                                  diff_space=diff_space, gate_locs=gate_locs)
        cur_col = alat_params.pop('col_idx')
        fg_amp, alat_ports = self.draw_diffamp(cur_col, alat_params, cur_track_width=cur_track_width,
                                               diff_space=diff_space, gate_locs=gate_locs)

        # add pins
        for name, warr_list in integ_ports.items():
            self.add_pin('integ_' + name, warr_list, show=show_pins)

        for name, warr_list in alat_ports.items():
            self.add_pin('alat_' + name, warr_list, show=show_pins)

        for idx, dlat_params in enumerate(dlat_params_list):
            cur_col = dlat_params.pop('col_idx')
            fg_amp, dlat_ports = self.draw_diffamp(cur_col, dlat_params, cur_track_width=cur_track_width,
                                                   diff_space=diff_space, gate_locs=gate_locs)
            for name, warr_list in dlat_ports.items():
                self.add_pin('dlat%d_%s' % (idx, name), warr_list, show=show_pins)

        # export supplies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        for warr in ptap_wire_arrs:
            self.add_pin(self.get_pin_name('VSS'), warr, show=show_pins)
        for warr in ntap_wire_arrs:
            self.add_pin(self.get_pin_name('VDD'), warr, show=show_pins)

        # add global ground
        if global_gnd_layer is not None:
            _, global_gnd_box = next(ptap_wire_arrs[0].wire_iter(self.grid))
            self.add_pin_primitive(global_gnd_name, global_gnd_layer, global_gnd_box)

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
            th_dict={},
            gds_space=1,
            diff_space=1,
            cur_track_width=1,
            show_pins=False,
            guard_ring_nf=0,
            global_gnd_layer=None,
            global_gnd_name='gnd!',
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
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            integ_params='Integrating frontend parameters.',
            alat_params='Analog latch parameters',
            dlat_params_list='Digital latch parameters.',
            fg_tot='Total number of fingers.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            global_gnd_layer='layer of the global ground pin.  None to disable drawing global ground.',
            global_gnd_name='name of global ground pin.',
        )


class RXHalf(TemplateBase):
    """one data path of DDR burst mode RX core.

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
        super(RXHalf, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._fg_tot = 0
        self._col_idx_dict = None

    @property
    def num_fingers(self):
        # type: () -> int
        return self._fg_tot

    def get_column_index_table(self):
        return self._col_idx_dict

    def draw_layout(self):
        lch = self.params['lch']
        guard_ring_nf = self.params['guard_ring_nf']
        layout_info = AnalogBaseInfo(self.grid, lch, guard_ring_nf)

        bot_inst, top_inst, col_idx_dict = self.place(layout_info)
        self.connect(layout_info, bot_inst, top_inst, col_idx_dict)
        self._col_idx_dict = col_idx_dict

    def place(self, layout_info):
        alat_params_list = self.params['alat_params_list']
        integ_params = self.params['integ_params']
        intsum_params = self.params['intsum_params']
        summer_params = self.params['summer_params']
        dlat_params_list = self.params['dlat_params_list']
        diff_space = self.params['diff_space']
        nduml = self.params['nduml']
        ndumr = self.params['ndumr']

        # create AnalogBaseInfo object
        tech_info = self.grid.tech_info
        vm_layer_id = layout_info.mconn_port_layer + 2
        diff_track_width = 2 + diff_space

        # compute block locations
        col_idx_dict = {}
        # step 0: place integrating frontend.
        cur_col = nduml
        integ_col_idx = cur_col
        # print('integ_col: %d' % cur_col)
        # step 0A: find minimum number of fingers
        new_integ_params = integ_params.copy()
        integ_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
        new_integ_params['min'] = integ_fg_min
        integ_info = SerdesRXBase.get_diffamp_info(tech_info, new_integ_params)
        new_integ_params['col_idx'] = integ_col_idx
        integ_fg_tot = integ_info['fg_tot']
        col_idx_dict['integ'] = (cur_col, cur_col + integ_fg_tot)
        cur_col += integ_fg_tot
        # step 0B: reserve routing tracks between analog latch and intsum
        route_integ_alat_fg = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
        col_idx_dict['integ_route'] = (cur_col, cur_col + route_integ_alat_fg)
        # print('integ_route_col: %d' % cur_col)
        cur_col += route_integ_alat_fg

        # step 1: place analog latches
        alat_col_idx = cur_col
        # print('alat_col: %d' % cur_col)
        alat1_params = alat_params_list[0].copy()
        alat2_params = alat_params_list[1].copy()
        # step 1A: find minimum number of fingers
        alat_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * diff_track_width, cur_col)
        # step 1B: make both analog latches have the same width
        alat1_params['min'] = alat_fg_min
        alat1_info = SerdesRXBase.get_diffamp_info(tech_info, alat1_params)
        alat2_params['min'] = alat_fg_min
        alat2_info = SerdesRXBase.get_diffamp_info(tech_info, alat2_params)
        alat_fg_min = max(alat1_info['fg_tot'], alat2_info['fg_tot'])
        alat1_params['min'] = alat_fg_min
        alat2_params['min'] = alat_fg_min
        alat1_params['col_idx'] = alat_col_idx
        alat2_params['col_idx'] = alat_col_idx
        col_idx_dict['alat'] = (cur_col, cur_col + alat_fg_min)
        cur_col += alat_fg_min
        # step 1C: reserve routing tracks between analog latch and intsum
        route_alat_intsum_fg = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
        col_idx_dict['alat_route'] = (cur_col, cur_col + route_alat_intsum_fg)
        # print('alat_route_col: %d' % cur_col)
        cur_col += route_alat_intsum_fg

        # step 2: place intsum and most digital latches
        # assumption: we assume the load/offset fingers < total gm fingers,
        # so we can use gm_info to determine sizing.
        intsum_col_idx = cur_col
        # print('intsum_main_col: %d' % cur_col)
        col_idx_dict['intsum'] = [intsum_col_idx]
        col_idx_dict['dlat'] = [(0, 0)] * len(dlat_params_list)
        intsum_gm_fg_list = intsum_params['gm_fg_list']
        intsum_gm_sep_list = [layout_info.min_fg_sep] * (len(intsum_gm_fg_list) - 1)
        # step 2A: place main tap.  No requirements.
        intsum_main_info = SerdesRXBase.get_gm_info(tech_info, intsum_gm_fg_list[0])
        new_intsum_gm_fg_list = [intsum_gm_fg_list[0]]
        cur_col += intsum_main_info['fg_tot'] + intsum_gm_sep_list[0]
        # print('cur_col: %d' % cur_col)
        # step 2B: place precursor tap.  must fit one differential track
        col_idx_dict['intsum'].append(cur_col)
        # print('intsum_pre_col: %d' % cur_col)
        intsum_pre_fg_params = intsum_gm_fg_list[1].copy()
        intsum_pre_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
        intsum_pre_fg_params['min'] = intsum_pre_fg_min
        intsum_pre_info = SerdesRXBase.get_gm_info(tech_info, intsum_pre_fg_params)
        new_intsum_gm_fg_list.append(intsum_pre_fg_params)
        cur_col += intsum_pre_info['fg_tot'] + intsum_gm_sep_list[1]
        # print('cur_col: %d' % cur_col)
        # step 2C: place intsum DFE taps
        num_dfe = len(intsum_gm_fg_list) - 2 + 1
        new_dlat_params_list = [None] * len(dlat_params_list)
        for idx in range(2, len(intsum_gm_fg_list)):
            col_idx_dict['intsum'].append(cur_col)
            # print('intsum_idx%d_col: %d' % (idx, cur_col))
            intsum_dfe_fg_params = intsum_gm_fg_list[idx].copy()
            dfe_idx = num_dfe - (idx - 2)
            dig_latch_params = dlat_params_list[dfe_idx - 2].copy()
            in_route = False
            n_diff_tr = 1
            if dfe_idx > 2:
                # for DFE tap > 2, the intsum Gm stage must align with the corresponding
                # digital latch.
                # set digital latch column index
                if dfe_idx % 2 == 1:
                    # for odd DFE taps, we have criss-cross connections, so fit 2 differential tracks
                    n_diff_tr = 2
                    # for odd DFE taps > 3, we need to reserve additional input routing tracks
                    in_route = dfe_idx > 3

                # fit diff tracks and make diglatch and DFE tap have same width
                tot_diff_track = diff_track_width * n_diff_tr + (n_diff_tr - 1)
                intsum_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, tot_diff_track, cur_col)
                dig_latch_params['min'] = intsum_dfe_fg_min
                dlat_info = SerdesRXBase.get_diffamp_info(tech_info, dig_latch_params)
                intsum_dfe_fg_params['min'] = dlat_info['fg_tot']
                intsum_dfe_info = SerdesRXBase.get_gm_info(tech_info, intsum_dfe_fg_params)
                num_fg = intsum_dfe_info['fg_tot']
                intsum_dfe_fg_params['min'] = num_fg
                dig_latch_params['min'] = num_fg
                dig_latch_params['col_idx'] = cur_col
                col_idx_dict['dlat'][dfe_idx - 2] = (cur_col, cur_col + num_fg)
                cur_col += num_fg
                # print('cur_col: %d' % cur_col)
                if in_route:
                    # allocate input route
                    route_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
                    intsum_gm_sep_list[idx] = route_fg_min
                    col_idx_dict['dlat%d_inroute' % (dfe_idx - 2)] = (cur_col, cur_col + route_fg_min)
                    cur_col += route_fg_min
                    # print('cur_col: %d' % cur_col)
                else:
                    cur_col += intsum_gm_sep_list[idx]
                    # print('cur_col: %d' % cur_col)
            else:
                # for DFE tap 2, the Gm stage should fit 1 differential track, but we have no
                # requirements for digital latch
                intsum_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
                intsum_dfe_fg_params['min'] = intsum_dfe_fg_min
                intsum_dfe_info = SerdesRXBase.get_gm_info(tech_info, intsum_dfe_fg_params)
                # no need to add gm sep because this is the last tap.
                cur_col += intsum_dfe_info['fg_tot']
                # print('cur_col: %d' % cur_col)
            # save modified parameters
            new_dlat_params_list[dfe_idx - 2] = dig_latch_params
            new_intsum_gm_fg_list.append(intsum_dfe_fg_params)
        # add last column
        col_idx_dict['intsum'].append(cur_col)
        # print('intsum_last_col: %d' % cur_col)
        # step 2D: reserve routing tracks between intsum and summer
        route_intsum_sum_fg = layout_info.num_tracks_to_fingers(vm_layer_id, diff_track_width, cur_col)
        col_idx_dict['summer_route'] = (cur_col, cur_col + route_intsum_sum_fg)
        cur_col += route_intsum_sum_fg
        # print('summer cur_col: %d' % cur_col)

        # step 3: place DFE summer and first digital latch
        # assumption: we assume the load fingers < total gm fingers,
        # so we can use gm_info to determine sizing.
        summer_col_idx = cur_col
        col_idx_dict['summer'] = [cur_col]
        summer_gm_fg_list = summer_params['gm_fg_list']
        summer_gm_sep_list = [layout_info.min_fg_sep]
        # step 3A: place main tap.  No requirements.
        summer_main_info = SerdesRXBase.get_gm_info(tech_info, summer_gm_fg_list[0])
        new_summer_gm_fg_list = [summer_gm_fg_list[0]]
        cur_col += summer_main_info['fg_tot'] + summer_gm_sep_list[0]
        # print('cur_col: %d' % cur_col)
        # step 3B: place DFE tap.  must fit two differential tracks
        col_idx_dict['summer'].append(cur_col)
        summer_dfe_fg_params = summer_gm_fg_list[1].copy()
        summer_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * diff_track_width, cur_col)
        summer_dfe_fg_params['min'] = summer_dfe_fg_min
        summer_dfe_info = SerdesRXBase.get_gm_info(tech_info, summer_dfe_fg_params)
        new_summer_gm_fg_list.append(summer_dfe_fg_params)
        cur_col += summer_dfe_info['fg_tot']
        col_idx_dict['summer'].append(cur_col)
        # print('cur_col: %d' % cur_col)
        # step 3C: place first digital latch
        # only requirement is that the right side line up with summer.
        dig_latch_params = dlat_params_list[0].copy()
        new_dlat_params_list[0] = dig_latch_params
        dlat_info = SerdesRXBase.get_diffamp_info(tech_info, dig_latch_params)
        dig_latch_params['col_idx'] = cur_col - dlat_info['fg_tot']
        col_idx_dict['dlat'][0] = (cur_col - dlat_info['fg_tot'], cur_col)

        fg_tot = cur_col + ndumr
        self._fg_tot = fg_tot

        # make RXHalfBottom
        bot_params = {key: self.params[key] for key in RXHalfTop.get_params_info().keys()
                      if key in self.params}
        bot_params['fg_tot'] = fg_tot
        bot_params['integ_params'] = new_integ_params
        bot_params['alat_params'] = alat1_params
        bot_params['dlat_params_list'] = new_dlat_params_list
        bot_master = self.new_template(params=bot_params, temp_cls=RXHalfBottom)
        bot_inst = self.add_instance(bot_master)

        # make RXHalfTop
        top_params = {key: self.params[key] for key in RXHalfBottom.get_params_info().keys()
                      if key in self.params}
        top_params['fg_tot'] = fg_tot
        top_params['alat_params'] = alat2_params
        top_params['intsum_params'] = dict(
            col_idx=intsum_col_idx,
            fg_load=intsum_params['fg_load'],
            gm_fg_list=new_intsum_gm_fg_list,
            gm_sep_list=intsum_gm_sep_list,
            sgn_list=intsum_params['sgn_list'],
        )
        top_params['summer_params'] = dict(
            col_idx=summer_col_idx,
            fg_load=summer_params['fg_load'],
            gm_fg_list=new_summer_gm_fg_list,
            gm_sep_list=summer_gm_sep_list,
            sgn_list=summer_params['sgn_list'],
        )
        # print('top summer col: %d' % top_params['summer_params']['col_idx'])
        top_master = self.new_template(params=top_params, temp_cls=RXHalfTop)
        top_inst = self.add_instance(top_master, orient='MX')
        top_inst.move_by(dy=bot_inst.array_box.top - top_inst.array_box.bottom)
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.set_size_from_array_box(top_master.size[0])

        for port_name in bot_inst.port_names_iter():
            self.reexport(bot_inst.get_port(port_name), show=True)
        for port_name in top_inst.port_names_iter():
            self.reexport(top_inst.get_port(port_name), show=True)

        return bot_inst, top_inst, col_idx_dict

    def connect(self, layout_info, bot_inst, top_inst, col_idx_dict):
        diff_space = self.params['diff_space']
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        nintsum = len(self.params['intsum_params']['gm_fg_list'])

        # connect integ to alat1
        route_col, _ = col_idx_dict['integ_route']
        ptr_idx = self.grid.coord_to_nearest_track(vm_layer, layout_info.col_to_coord(route_col),
                                                   mode=2)
        ntr_idx = ptr_idx + diff_space + 1
        integ_outp = bot_inst.get_port('integ_outp').get_pins(hm_layer)
        integ_outn = bot_inst.get_port('integ_outn').get_pins(hm_layer)
        alat0_inp = bot_inst.get_port('alat_inp').get_pins(hm_layer)
        alat0_inn = bot_inst.get_port('alat_inn').get_pins(hm_layer)

        self.connect_differential_tracks(integ_outp + alat0_inp, integ_outn + alat0_inn,
                                         vm_layer, ptr_idx, ntr_idx)

        # connect alat1 to intsum
        route_col, _ = col_idx_dict['alat_route']
        ptr_idx = self.grid.coord_to_nearest_track(vm_layer, layout_info.col_to_coord(route_col),
                                                   mode=2)
        ntr_idx = ptr_idx + diff_space + 1
        alat1_outp = top_inst.get_port('alat_outp').get_pins(hm_layer)
        alat1_outn = top_inst.get_port('alat_outn').get_pins(hm_layer)
        intsum_inp = top_inst.get_port('intsum_inp<0>').get_pins(hm_layer)
        intsum_inn = top_inst.get_port('intsum_inn<0>').get_pins(hm_layer)

        self.connect_differential_tracks(alat1_outp + intsum_inp, alat1_outn + intsum_inn,
                                         vm_layer, ptr_idx, ntr_idx)

        # connect intsum to summer
        route_col_intv = col_idx_dict['summer_route']
        ptr_idx = layout_info.get_center_tracks(vm_layer, 2 + diff_space, route_col_intv)
        ntr_idx = ptr_idx + diff_space + 1
        intsum_outp = top_inst.get_port('intsum_outp').get_pins(hm_layer)
        intsum_outn = top_inst.get_port('intsum_outn').get_pins(hm_layer)
        summer_inp = top_inst.get_port('summer_inp<0>').get_pins(hm_layer)
        summer_inn = top_inst.get_port('summer_inn<0>').get_pins(hm_layer)

        self.connect_differential_tracks(intsum_outp + summer_inp, intsum_outn + summer_inn,
                                         vm_layer, ptr_idx, ntr_idx)

        # connect DFE tap 2
        route_col_intv = col_idx_dict['intsum'][-2], col_idx_dict['intsum'][-1]
        ptr_idx = layout_info.get_center_tracks(vm_layer, 2 + diff_space, route_col_intv)
        ntr_idx = ptr_idx + diff_space + 1
        dlat_outp = bot_inst.get_port('dlat0_outp').get_pins(hm_layer)
        dlat_outn = bot_inst.get_port('dlat0_outn').get_pins(hm_layer)
        tap_inp = top_inst.get_port('intsum_inp<%d>' % (nintsum - 1)).get_pins(hm_layer)
        tap_inn = top_inst.get_port('intsum_inn<%d>' % (nintsum - 1)).get_pins(hm_layer)
        p_list = dlat_outp + tap_inp
        n_list = dlat_outn + tap_inn
        if nintsum > 3:
            p_list.extend(bot_inst.get_port('dlat1_inp').get_pins(hm_layer))
            n_list.extend(bot_inst.get_port('dlat1_inn').get_pins(hm_layer))

        self.connect_differential_tracks(p_list, n_list, vm_layer, ptr_idx, ntr_idx)

        # connect even DFE taps
        ndfe = nintsum - 2 + 1
        for dfe_idx in range(4, ndfe + 1, 2):
            intsum_idx = nintsum - 1 - (dfe_idx - 2)
            route_col_intv = col_idx_dict['intsum'][intsum_idx], col_idx_dict['intsum'][intsum_idx + 1]
            ptr_idx = layout_info.get_center_tracks(vm_layer, 2 + diff_space, route_col_intv)
            ntr_idx = ptr_idx + diff_space + 1
            dlat_outp = bot_inst.get_port('dlat%d_outp' % (dfe_idx - 2)).get_pins(hm_layer)
            dlat_outn = bot_inst.get_port('dlat%d_outn' % (dfe_idx - 2)).get_pins(hm_layer)
            tap_inp = top_inst.get_port('intsum_inp<%d>' % intsum_idx).get_pins(hm_layer)
            tap_inn = top_inst.get_port('intsum_inn<%d>' % intsum_idx).get_pins(hm_layer)
            self.connect_differential_tracks(dlat_outp + tap_inp, dlat_outn + tap_inn,
                                             vm_layer, ptr_idx, ntr_idx)
            if dfe_idx + 1 <= ndfe:
                # connect to next digital latch
                route_col_intv = col_idx_dict['dlat%d_inroute' % (dfe_idx - 1)]
                ptr_idx = layout_info.get_center_tracks(vm_layer, 2 + diff_space, route_col_intv)
                ntr_idx = ptr_idx + diff_space + 1
                dlat_inp = bot_inst.get_port('dlat%d_inp' % (dfe_idx - 1)).get_pins(hm_layer)
                dlat_inn = bot_inst.get_port('dlat%d_inn' % (dfe_idx - 1)).get_pins(hm_layer)
                self.connect_differential_tracks(dlat_outp + dlat_inp, dlat_outn + dlat_inn,
                                                 vm_layer, ptr_idx, ntr_idx)

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
            th_dict={},
            gds_space=1,
            diff_space=1,
            fg_stage=6,
            nduml=4,
            ndumr=4,
            cur_track_width=1,
            guard_ring_nf=0,
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
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            integ_params='Integrating frontend parameters.',
            alat_params_list='Analog latch parameters',
            intsum_params='Integrator summer parameters.',
            summer_params='DFE tap-1 summer parameters.',
            dlat_params_list='Digital latch parameters.',
            fg_stage='separation between stages.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
        )


class RXCore(TemplateBase):
    """one data path of DDR burst mode RX core.

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
        super(RXCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._fg_tot = 0

    @property
    def num_fingers(self):
        # type: () -> int
        return self._fg_tot

    def draw_layout(self):
        half_params = self.params.copy()

        half_master = self.new_template(params=half_params, temp_cls=RXHalf)
        odd_inst = self.add_instance(half_master, 'X1', orient='MX')
        odd_inst.move_by(dy=half_master.bound_box.height)
        even_inst = self.add_instance(half_master, 'X0')
        even_inst.move_by(dy=odd_inst.array_box.top - even_inst.array_box.bottom)

        self.array_box = odd_inst.array_box.merge(even_inst.array_box)
        self.set_size_from_array_box(half_master.size[0])
        self._fg_tot = half_master.num_fingers

        col_idx_dict = half_master.get_column_index_table()
        self.connect([even_inst, odd_inst], col_idx_dict)

    def connect(self, inst_list, col_idx_dict):
        lch = self.params['lch']
        guard_ring_nf = self.params['guard_ring_nf']
        layout_info = AnalogBaseInfo(self.grid, lch, guard_ring_nf)
        hm_layer_id = layout_info.mconn_port_layer + 1
        vm_layer_id = hm_layer_id + 1
        diff_space = self.params['diff_space']

        # connect inputs of even and odd paths
        route_col_intv = col_idx_dict['integ']
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 2 + diff_space, route_col_intv)
        ntr_idx = ptr_idx + diff_space + 1
        p_list = []
        n_list = []
        for inst in inst_list:
            p_list += inst.get_port('integ_inp').get_pins(hm_layer_id)
            n_list += inst.get_port('integ_inn').get_pins(hm_layer_id)

        inp, inn = self.connect_differential_tracks(p_list, n_list, vm_layer_id, ptr_idx, ntr_idx)
        self.add_pin('inp', inp, show=True)
        self.add_pin('inn', inn, show=True)

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
            th_dict={},
            gds_space=1,
            diff_space=1,
            fg_stage=6,
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
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            integ_params='Integrating frontend parameters.',
            alat_params_list='Analog latch parameters',
            intsum_params='Integrator summer parameters.',
            summer_params='DFE tap-1 summer parameters.',
            dlat_params_list='Digital latch parameters.',
            fg_stage='separation between stages.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            show_pins='True to create pin labels.',
            rename_dict='port renaming dictionary',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            global_gnd_layer='layer of the global ground pin.  None to disable drawing global ground.',
            global_gnd_name='name of global ground pin.',
        )
