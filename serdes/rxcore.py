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

from typing import Dict, Any, Set, Tuple

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance
from bag.layout.routing import TrackID

from .base import SerdesRXBase, SerdesRXBaseInfo


def connect_to_xm(template, warr_p, warr_n, col_intv, layout_info, sig_widths, sig_spaces):
    """Connect differential ports to xm layer.

    Returns a list of [even_p, even_n, odd_p, odd_n] WireArrays on xm layer.
    """
    vm_width, xm_width = sig_widths
    vm_space, xm_space = sig_spaces
    hm_layer_id = layout_info.mconn_port_layer + 1
    vm_layer_id = hm_layer_id + 1
    xm_layer_id = vm_layer_id + 1

    # get vm tracks
    p_tr = layout_info.get_center_tracks(vm_layer_id, 2, col_intv, width=vm_width, space=vm_space)
    n_tr = p_tr + vm_width + vm_space
    # step 1B: connect to vm and xm layer
    vmp, vmn = template.connect_differential_tracks(warr_p, warr_n, vm_layer_id, p_tr, n_tr, width=vm_width,
                                                    fill_type='VDD')
    mid_tr = template.grid.coord_to_nearest_track(xm_layer_id, vmp.middle, half_track=True, mode=0)
    nx_tr = mid_tr - (xm_width + xm_space) / 2
    px_tr = nx_tr + xm_width + xm_space
    return template.connect_differential_tracks(vmp, vmn, xm_layer_id, px_tr, nx_tr, width=xm_width,
                                                fill_type='VDD')


def get_bias_tracks(layout_info, layer_id, col_intv, sig_width, sig_space, clk_width, sig_clk_space):
    sig_left = layout_info.get_center_tracks(layer_id, 2, col_intv, width=sig_width, space=sig_space)
    left_tr = sig_left - (sig_width + clk_width) / 2 - sig_clk_space
    right_tr = sig_left + (sig_width + sig_space) + (sig_width + clk_width) / 2 + sig_clk_space
    return left_tr, right_tr


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

    def _draw_layout_helper(self, alat_params, intsum_params, summer_params,
                            show_pins, diff_space, hm_width, hm_cur_width,
                            sig_widths, sig_spaces, clk_widths, clk_spaces,
                            sig_clk_spaces, datapath_parity, **kwargs):
        draw_params = kwargs.copy()

        result = self.place(alat_params, intsum_params, summer_params, draw_params,
                            diff_space, hm_width, hm_cur_width)
        alat_ports, intsum_ports, summer_ports, block_info = result

        ffe_inputs = self.connect_sup_io(block_info, alat_ports, intsum_ports, summer_ports,
                                         sig_widths, sig_spaces, clk_widths, clk_spaces, show_pins)

        self.connect_bias(block_info, alat_ports, intsum_ports, summer_ports, ffe_inputs,
                          sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                          show_pins, datapath_parity)

    def place(self, alat_params, intsum_params, summer_params, draw_params,
              diff_space, hm_width, hm_cur_width):
        gds_space = draw_params['gds_space']
        w_dict = draw_params['w_dict']
        alat_params = alat_params.copy()
        intsum_params = intsum_params.copy()
        summer_params = summer_params.copy()

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params['pg_tracks'] = [hm_width]
        draw_params['pds_tracks'] = [2 * hm_width + diff_space]
        ng_tracks = []
        nds_tracks = []

        # check if butterfly switches are used
        has_but = False
        gm_fg_list = intsum_params['gm_fg_list']
        for fdict in gm_fg_list:
            if fdict.get('fg_but', 0) > 0:
                has_but = True
                break

        inn_idx = (hm_width - 1) / 2
        inp_idx = inn_idx + hm_width + diff_space
        if has_but:
            if diff_space % 2 != 1:
                raise ValueError('diff_space must be odd if butterfly switch is present.')
            # route cascode in between butterfly gates.
            gate_locs = {'bias_casc': (inp_idx + inn_idx) / 2,
                         'sgnp': inp_idx,
                         'sgnn': inn_idx,
                         'inp': inp_idx,
                         'inn': inn_idx}
        else:
            gate_locs = {'inp': inp_idx,
                         'inn': inn_idx}

        # compute nmos gate/drain/source number of tracks
        for row_name in ['tail', 'w_en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in' or (row_name == 'casc' and has_but):
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)
        # set size based on 2 layer up.
        self.set_size_from_array_box(self.mos_conn_layer + 3)

        # draw blocks
        alat_col = alat_params.pop('col_idx')
        # print('rxtop alat cur_col: %d' % cur_col)
        _, alat_ports = self.draw_diffamp(alat_col, alat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                          diff_space=diff_space, gate_locs=gate_locs)
        alat_info = self.layout_info.get_diffamp_info(alat_params)

        intsum_col = intsum_params.pop('col_idx')
        # print('rxtop intsum cur_col: %d' % cur_col)
        _, intsum_ports = self.draw_gm_summer(intsum_col, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              **intsum_params)
        intsum_info = self.layout_info.get_summer_info(intsum_params['fg_load'], intsum_params['gm_fg_list'],
                                                       gm_sep_list=intsum_params.get('gm_sep_list', None))

        summer_col = summer_params.pop('col_idx')
        # print('rxtop summer cur_col: %d' % cur_col)
        _, summer_ports = self.draw_gm_summer(summer_col, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              **summer_params)
        summer_info = self.layout_info.get_summer_info(summer_params['fg_load'], summer_params['gm_fg_list'],
                                                       gm_sep_list=summer_params.get('gm_sep_list', None))

        block_info = dict(
            alat=(alat_col, alat_info),
            intsum=(intsum_col, intsum_info),
            summer=(summer_col, summer_info),
        )

        return alat_ports, intsum_ports, summer_ports, block_info

    def connect_sup_io(self, block_info, alat_ports, intsum_ports, summer_ports,
                       sig_widths, sig_spaces, clk_widths, clk_spaces, show_pins):

        intsum_col, intsum_info = block_info['intsum']

        # get vdd/cascode bias pins
        vdd_list, cascl_list = [], []
        vdd_list.extend(alat_ports['vddt'])
        vdd_list.extend(intsum_ports[('vddt', -1)])
        vdd_list.extend(summer_ports[('vddt', -1)])
        cascl_list.extend(alat_ports['bias_casc'])
        cascl_list.extend(intsum_ports[('bias_casc', 0)])

        # export alat inout pins
        inout_list = ('inp', 'inn', 'outp', 'outn')
        for name in inout_list:
            self.add_pin('alat1_%s' % name, alat_ports[name], show=show_pins)

        # export intsum inout pins
        num_intsum = len(intsum_info['amp_info_list'])
        ffe_inputs = None
        for idx in range(num_intsum):
            if idx == 1:
                # connect ffe input to xm layer
                ffe_inp = intsum_ports[('inp', 1)][0]
                ffe_inn = intsum_ports[('inn', 1)][0]
                ffe_start = intsum_col + intsum_info['gm_offsets'][1]
                ffe_stop = ffe_start + intsum_info['amp_info_list'][1]['fg_tot']
                ffe_inp, ffe_inn = connect_to_xm(self, ffe_inp, ffe_inn, (ffe_start, ffe_stop),
                                                 self.layout_info, sig_widths, sig_spaces)
                ffe_inputs = [ffe_inp, ffe_inn]
                self.add_pin('ffe_inp', ffe_inp, show=show_pins)
                self.add_pin('ffe_inn', ffe_inn, show=show_pins)
            else:
                self.add_pin('intsum_inp<%d>' % idx, intsum_ports[('inp', idx)], show=show_pins)
                self.add_pin('intsum_inn<%d>' % idx, intsum_ports[('inn', idx)], show=show_pins)

        self.add_pin('intsum_outp', intsum_ports[('outp', -1)], show=show_pins)
        self.add_pin('intsum_outn', intsum_ports[('outn', -1)], show=show_pins)

        # export summer inout pins
        for name in ('inp', 'inn'):
            self.add_pin('summer_%s<0>' % name, summer_ports[(name, 0)], show=show_pins)
            self.add_pin('summer_%s<1>' % name, summer_ports[(name, 1)], show=show_pins)
        self.add_pin('summer_outp', summer_ports[('outp', -1)], show=show_pins)
        self.add_pin('summer_outn', summer_ports[('outn', -1)], show=show_pins)

        # connect and export supplies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        sup_lower = ntap_wire_arrs[0].lower
        sup_upper = ntap_wire_arrs[0].upper
        for warr in ptap_wire_arrs:
            self.add_pin('VSS', warr, show=show_pins)

        vdd_name = self.get_pin_name('VDD')
        for warr in ntap_wire_arrs:
            self.add_pin(vdd_name, warr, show=show_pins)

        warr = self.connect_wires(vdd_list, lower=sup_lower, upper=sup_upper)
        self.add_pin(vdd_name, warr, show=show_pins)

        # connect summer cascode
        # step 1: get vm track
        layout_info = self.layout_info
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        summer_col, summer_info = block_info['summer']
        casc_sum = summer_ports[('bias_casc', 0)]
        summer_start = summer_col + summer_info['gm_offsets'][0]
        col_intv = summer_start, summer_start + summer_info['amp_info_list'][0]['fg_tot']
        clk_width_vm = clk_widths[0]
        clk_space_vm = clk_spaces[0]
        casc_tr = self.layout_info.get_center_tracks(vm_layer, 2, col_intv, width=clk_width_vm, space=clk_space_vm)
        # step 2: connect summer cascode to vdd
        casc_tr_id = TrackID(vm_layer, casc_tr, width=clk_width_vm)
        warr = self.connect_to_tracks(ntap_wire_arrs + warr + casc_sum, casc_tr_id)
        self.add_pin(vdd_name, warr, show=show_pins)

        warr = self.connect_wires(cascl_list, lower=sup_lower)
        self.add_pin(vdd_name, warr, show=show_pins)
        return ffe_inputs

    def connect_bias(self, block_info, alat_ports, intsum_ports, summer_ports, ffe_inputs,
                     sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                     show_pins, datapath_parity):
        layout_info = self.layout_info
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        sig_width_vm, sig_width_xm = sig_widths
        clk_width_vm, clk_width_xm = clk_widths
        sig_space_vm = sig_spaces[0]
        clk_space_vm, clk_space_xm = clk_spaces
        sig_clk_space_vm, sig_clk_space_xm = sig_clk_spaces

        # calculate bias track indices
        ffe_top_tr = ffe_inputs[0].track_id.base_index
        ffe_bot_tr = ffe_inputs[1].track_id.base_index
        clkp_nmos_sw_tr_xm = ffe_bot_tr - (sig_width_xm + clk_width_xm) / 2 - sig_clk_space_xm
        clkn_nmos_sw_tr_xm = clkp_nmos_sw_tr_xm - clk_width_xm - clk_space_xm
        clkn_nmos_ana_tr_xm = clkp_nmos_sw_tr_xm - clk_width_xm - clk_space_xm
        clkp_pmos_intsum_tr_xm = ffe_top_tr + (sig_width_xm + clk_width_xm) / 2 + sig_clk_space_xm
        clkn_pmos_summer_tr_xm = clkp_pmos_intsum_tr_xm
        clkp_pmos_ana_tr_xm = clkp_pmos_intsum_tr_xm + clk_width_xm + clk_space_xm

        clkn_nmos_sw_tr_id = TrackID(xm_layer, clkn_nmos_sw_tr_xm, width=clk_width_xm)
        clkn_nmos_sw_list = []

        if datapath_parity == 0:
            clkp, clkn = 'clkp', 'clkn'
        else:
            clkp, clkn = 'clkn', 'clkp'

        # connect alat biases
        alat_col, alat_info = block_info['alat']
        col_intv = alat_col, alat_col + alat_info['fg_tot']
        left_sig_vm = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm, space=sig_space_vm)
        right_tr_vm = left_sig_vm - (sig_width_vm + clk_width_vm) / 2 - sig_clk_space_vm
        left_tr_vm = right_tr_vm - clk_space_vm - clk_width_vm
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)
        if datapath_parity == 0:
            nmos_tr_id, pmos_tr_id, sw_tr_id = ltr_id, ltr_id, rtr_id
        else:
            nmos_tr_id, pmos_tr_id, sw_tr_id = rtr_id, rtr_id, ltr_id
        # nmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_tail'], nmos_tr_id, fill_type='VSS')
        xtr_id = TrackID(xm_layer, clkn_nmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VSS', min_len_mode=0)
        self.add_pin(clkn + '_nmos_analog', warr, show=show_pins)
        # pmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_load'], pmos_tr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkp_pmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkp + '_pmos_analog', warr, show=show_pins)
        # nmos_switch
        warr = self.connect_to_tracks(alat_ports['sw'], sw_tr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkp_nmos_sw_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkp + '_nmos_switch', warr, show=show_pins)

        # connect intsum main tap biases
        intsum_col, intsum_info = block_info['intsum']
        intsum_start = intsum_col + intsum_info['gm_offsets'][0]
        col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][0]['fg_tot']
        left_tr_vm = self.layout_info.get_center_tracks(vm_layer, 2, col_intv, width=clk_width_vm, space=clk_space_vm)
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        # pmos intsum
        warr = self.connect_to_tracks(intsum_ports[('bias_load', -1)], ltr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkp_pmos_intsum_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkp + '_pmos_intsum', warr, show=show_pins)
        # nmos switch
        warr = self.connect_wires(intsum_ports[('sw', 0)] + intsum_ports[('sw', 1)])
        warr = self.connect_to_tracks(warr, ltr_id, fill_type='VDD')
        clkn_nmos_sw_list.append(warr)

        # connect intsum ffe tap biases
        intsum_start = intsum_col + intsum_info['gm_offsets'][1]
        col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][1]['fg_tot']
        ltr_vm, rtr_vm = get_bias_tracks(self.layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                         clk_width_vm, sig_clk_space_vm)
        # bias_ffe
        en_tr_id = TrackID(vm_layer, rtr_vm, width=clk_width_vm)
        warr = self.connect_to_tracks(intsum_ports[('bias_casc', 1)], en_tr_id, fill_type='VDD', track_lower=0)
        self.add_pin('bias_ffe', warr, show=show_pins)
        # nmos intsum
        warr = self.connect_wires(intsum_ports[('bias_tail', 0)] + intsum_ports[('bias_tail', 1)])
        tr_id = TrackID(vm_layer, ltr_vm, width=clk_width_vm)
        warr = self.connect_to_tracks(warr, tr_id, fill_type='VSS', track_lower=0)
        self.add_pin(clkp + '_nmos_intsum', warr, show=show_pins)

        # connect intsum dfe tap biases
        num_dfe = len(intsum_info['gm_offsets']) - 2
        for fb_idx in range(num_dfe):
            dfe_idx = num_dfe + 1 - fb_idx
            intsum_start = intsum_col + intsum_info['gm_offsets'][2 + fb_idx]
            col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][2 + fb_idx]['fg_tot']
            if dfe_idx % 2 == 0:
                # no criss-cross inputs.
                bias_tr_vm, en_tr_vm = get_bias_tracks(self.layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                                       clk_width_vm, sig_clk_space_vm)
            else:
                # criss-cross inputs
                en_tr_vm = self.layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm,
                                                              space=sig_space_vm)
                if datapath_parity == 0:
                    en_tr_vm += (sig_width_vm + sig_space_vm) / 2
                    bias_tr_vm = en_tr_vm + clk_width_vm + clk_space_vm
                else:
                    en_tr_vm += (sig_width_vm + sig_space_vm) * 5 / 2
                    bias_tr_vm = en_tr_vm - clk_width_vm - clk_space_vm

            # en_dfe
            en_tr_id = TrackID(vm_layer, en_tr_vm, width=clk_width_vm)
            warr = self.connect_to_tracks(intsum_ports[('bias_casc', 2 + fb_idx)], en_tr_id,
                                          fill_type='VDD', track_lower=0)
            self.add_pin('en_dfe<%d>' % (dfe_idx - 1), warr, show=show_pins)
            # bias_dfe
            bias_tr_id = TrackID(vm_layer, bias_tr_vm, width=clk_width_vm)
            warr = self.connect_to_tracks(intsum_ports[('bias_tail', 2 + fb_idx)], bias_tr_id,
                                          fill_type='VSS', track_lower=0)
            self.add_pin('bias_dfe<%d>' % (dfe_idx - 1), warr, show=show_pins)

        # pmos intsum
        warr = self.connect_to_tracks(intsum_ports[('bias_load', -1)], ltr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkp_pmos_intsum_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkp + '_pmos_intsum', warr, show=show_pins)
        # nmos switch
        warr = self.connect_wires(intsum_ports[('sw', 0)] + intsum_ports[('sw', 1)])
        warr = self.connect_to_tracks(warr, ltr_id, fill_type='VDD')
        clkn_nmos_sw_list.append(warr)

        # connect summer main tap biases
        summer_col, summer_info = block_info['summer']
        summer_start = summer_col + summer_info['gm_offsets'][0]
        col_intv = summer_start, summer_start + summer_info['amp_info_list'][0]['fg_tot']
        left_tr_vm = self.layout_info.get_center_tracks(vm_layer, 2, col_intv, width=clk_width_vm, space=clk_space_vm)
        right_tr_vm = left_tr_vm + clk_width_vm + clk_space_vm
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)
        # pmos summer
        warr = self.connect_to_tracks(summer_ports[('bias_load', -1)], rtr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkn_pmos_summer_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkn + '_pmos_summer', warr, show=show_pins)
        # nmos summer
        warr = self.connect_to_tracks(summer_ports[('bias_tail', 0)], ltr_id, fill_type='VSS', track_lower=0)
        self.add_pin(clkp + '_nmos_summer', warr, show=show_pins)
        # nmos switch
        sw_wire = self.connect_wires(summer_ports[('sw', 0)] + summer_ports[('sw', 1)], fill_type='VDD')
        warr = self.connect_to_tracks(sw_wire, rtr_id, fill_type='VDD', min_len_mode=0)
        clkn_nmos_sw_list.append(warr)

        # connect summer feedback biases
        summer_start = summer_col + summer_info['gm_offsets'][1]
        col_intv = summer_start, summer_start + summer_info['amp_info_list'][1]['fg_tot']
        left_tr_vm = self.layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm, space=sig_space_vm)
        left_tr_vm += (sig_width_vm + sig_space_vm) / 2
        right_tr_vm = left_tr_vm + 2 * (sig_width_vm + sig_space_vm)
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)
        if datapath_parity == 0:
            en_tr_id, tap_tr_id = ltr_id, rtr_id
        else:
            en_tr_id, tap_tr_id = rtr_id, ltr_id
        # en_dfe
        warr = self.connect_to_tracks(summer_ports[('bias_casc', 1)], en_tr_id, fill_type='VDD', track_lower=0)
        self.add_pin('en_dfe<0>', warr, show=show_pins)
        warr = self.connect_to_tracks(summer_ports[('bias_tail', 1)], tap_tr_id, fill_type='VSS', track_lower=0)
        self.add_pin(clkp + '_nmos_summer_tap1', warr, show=show_pins)

        warr = self.connect_to_tracks(clkn_nmos_sw_list, clkn_nmos_sw_tr_id, fill_type='VDD')
        self.add_pin(clkn + '_nmos_switch', warr, show=show_pins)

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
            min_fg_sep=0,
            hm_width=1,
            hm_cur_width=-1,
            sig_widths=[1, 1],
            sig_spaces=[1, 1],
            clk_widths=[1, 1],
            clk_spaces=[1, 1],
            sig_clk_spaces=[1, 1],
            show_pins=False,
            guard_ring_nf=0,
            data_parity=0,
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
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            sig_widths='signal wire widths on each layer above hm layer.',
            sig_spaces='signal wire spacing on each layer above hm layer.',
            clk_widths='clk wire widths on each layer above hm layer.',
            clk_spaces='clk wire spacing on each layer above hm layer.',
            sig_clk_spaces='spacing between signal and clk on each layer above hm layer.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            datapath_parity='Parity of the DDR datapath.  Either 0 or 1.',
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

    def _draw_layout_helper(self, integ_params, alat_params, dlat_params_list, tap1_col_intv,
                            show_pins, diff_space, hm_width, hm_cur_width,
                            sig_widths, sig_spaces, clk_widths, clk_spaces,
                            sig_clk_spaces, datapath_parity, **kwargs):

        draw_params = kwargs.copy()

        result = self.place(integ_params, alat_params, dlat_params_list,
                            draw_params, hm_width, hm_cur_width, diff_space)
        integ_ports, alat_ports, block_info = result
        dlat_info_list = block_info['dlat']
        dlat_inputs = self.connect_sup_io(integ_ports, alat_ports, dlat_info_list, sig_widths, sig_spaces, show_pins)

        self.connect_bias(block_info, integ_ports, alat_ports, dlat_inputs, tap1_col_intv,
                          sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                          show_pins, datapath_parity)

    def place(self, integ_params, alat_params, dlat_params_list, draw_params,
              hm_width, hm_cur_width, diff_space):
        gds_space = draw_params['gds_space']
        w_dict = draw_params['w_dict']
        integ_params = integ_params.copy()
        alat_params = alat_params.copy()

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params['pg_tracks'] = [hm_width]
        draw_params['pds_tracks'] = [2 * hm_width + diff_space]
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'w_en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)
        # set size based on 2 layer up.
        self.set_size_from_array_box(self.mos_conn_layer + 3)

        # draw blocks
        gate_locs = {'inp': (hm_width - 1) / 2 + hm_width + diff_space,
                     'inn': (hm_width - 1) / 2}
        integ_col = integ_params.pop('col_idx')
        _, integ_ports = self.draw_diffamp(integ_col, integ_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                           diff_space=diff_space, gate_locs=gate_locs)
        integ_info = self.layout_info.get_diffamp_info(integ_params)

        alat_col = alat_params.pop('col_idx')
        _, alat_ports = self.draw_diffamp(alat_col, alat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                          diff_space=diff_space, gate_locs=gate_locs)
        alat_info = self.layout_info.get_diffamp_info(alat_params)

        dlat_info_list = []
        for idx, dlat_params in enumerate(dlat_params_list):
            dlat_params = dlat_params.copy()
            cur_col = dlat_params.pop('col_idx')
            _, dlat_ports = self.draw_diffamp(cur_col, dlat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs)
            dlat_info = self.layout_info.get_diffamp_info(dlat_params)

            dlat_info_list.append((cur_col, dlat_ports, dlat_info))

        block_info = dict(
            integ=(integ_col, integ_info),
            alat=(alat_col, alat_info),
            dlat=dlat_info_list,
        )

        return integ_ports, alat_ports, block_info

    def connect_sup_io(self, integ_ports, alat_ports, dlat_info_list, sig_widths, sig_spaces, show_pins):

        # get vdd/cascode bias pins from integ/alat
        vdd_list, casc_list = [], []
        vdd_list.extend(integ_ports['vddt'])
        vdd_list.extend(alat_ports['vddt'])
        casc_list.extend(integ_ports['bias_casc'])
        casc_list.extend(alat_ports['bias_casc'])

        # export inout pins
        inout_list = ('inp', 'inn', 'outp', 'outn')
        for name in inout_list:
            self.add_pin('integ_%s' % name, integ_ports[name], show=show_pins)
            self.add_pin('alat0_%s' % name, alat_ports[name], show=show_pins)

        dlat_inputs = None
        for idx, (cur_col, dlat_ports, dlat_info) in enumerate(dlat_info_list):
            vdd_list.extend(dlat_ports['vddt'])
            casc_list.extend(dlat_ports['bias_casc'])

            if (idx % 2 == 0) and idx > 0:
                # connect inputs to xm layer
                dlat_inp = dlat_ports['inp'][0]
                dlat_inn = dlat_ports['inn'][0]
                dlat_intv = cur_col, cur_col + dlat_info['fg_tot']
                dlat_inp, dlat_inn = connect_to_xm(self, dlat_inp, dlat_inn, dlat_intv,
                                                   self.layout_info, sig_widths, sig_spaces)
                self.add_pin('dlat%d_inp' % idx, dlat_inp, show=show_pins)
                self.add_pin('dlat%d_inn' % idx, dlat_inn, show=show_pins)
                dlat_inputs = [dlat_inp, dlat_inn]
            else:
                self.add_pin('dlat%d_inp' % idx, dlat_ports['inp'], show=show_pins)
                self.add_pin('dlat%d_inn' % idx, dlat_ports['inn'], show=show_pins)

            self.add_pin('dlat%d_outp' % idx, dlat_ports['outp'], show=show_pins)
            self.add_pin('dlat%d_outn' % idx, dlat_ports['outn'], show=show_pins)

        # connect and export supplies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        sup_lower = ntap_wire_arrs[0].lower
        sup_upper = ntap_wire_arrs[0].upper
        for warr in ptap_wire_arrs:
            self.add_pin(self.get_pin_name('VSS'), warr, show=show_pins)

        vdd_name = self.get_pin_name('VDD')
        for warr in ntap_wire_arrs:
            self.add_pin(vdd_name, warr, show=show_pins)

        warr = self.connect_wires(vdd_list, lower=sup_lower, upper=sup_upper)
        self.add_pin(vdd_name, warr, show=show_pins)
        warr = self.connect_wires(casc_list, lower=sup_lower, upper=sup_upper)
        self.add_pin(vdd_name, warr, show=show_pins)

        return dlat_inputs

    def connect_bias(self, block_info, integ_ports, alat_ports, dlat_inputs, tap1_col_intv,
                     sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                     show_pins, datapath_parity):
        layout_info = self.layout_info
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        sig_width_vm, sig_width_xm = sig_widths
        clk_width_vm, clk_width_xm = clk_widths
        sig_space_vm = sig_spaces[0]
        clk_space_vm, clk_space_xm = clk_spaces
        sig_clk_space_vm, sig_clk_space_xm = sig_clk_spaces

        # calculate bias track indices
        dlat_top_tr = dlat_inputs[0].track_id.base_index
        dlat_bot_tr = dlat_inputs[1].track_id.base_index
        clkp_nmos_ana_tr_xm = dlat_bot_tr - (sig_width_xm + clk_width_xm) / 2 - sig_clk_space_xm
        clkn_nmos_dig_tr_xm = clkp_nmos_ana_tr_xm
        clkn_nmos_sw_tr_xm = clkp_nmos_ana_tr_xm - clk_width_xm - clk_space_xm
        clkp_pmos_integ_tr_xm = dlat_top_tr + (sig_width_xm + clk_width_xm) / 2 + sig_clk_space_xm
        clkn_pmos_dig_tr_xm = clkp_pmos_integ_tr_xm
        clkn_pmos_ana_tr_xm = clkp_pmos_integ_tr_xm + clk_width_xm + clk_space_xm
        clkp_pmos_dig_tr_xm = clkn_pmos_ana_tr_xm

        # mirror to opposite side
        bot_xm_idx = self.grid.find_next_track(xm_layer, self.array_box.bottom_unit, mode=1, unit_mode=True)
        clkp_nmos_dig_tr_xm = 2 * bot_xm_idx - clkn_nmos_dig_tr_xm - 1
        clkp_nmos_sw_tr_xm = 2 * bot_xm_idx - clkn_nmos_sw_tr_xm - 1

        clkp_nmos_sw_tr_id = TrackID(xm_layer, clkp_nmos_sw_tr_xm, width=clk_width_xm)
        clkn_nmos_sw_tr_id = TrackID(xm_layer, clkn_nmos_sw_tr_xm, width=clk_width_xm)
        clkp_nmos_sw_list, clkn_nmos_sw_list = [], []
        clkp_nmos_ana_tr_id = TrackID(xm_layer, clkp_nmos_ana_tr_xm, width=clk_width_xm)
        clkp_nmos_ana_list = []
        clkp_nmos_dig_tr_id = TrackID(xm_layer, clkp_nmos_dig_tr_xm, width=clk_width_xm)
        clkn_nmos_dig_tr_id = TrackID(xm_layer, clkn_nmos_dig_tr_xm, width=clk_width_xm)
        clkp_nmos_dig_list, clkn_nmos_dig_list = [], []
        clkp_pmos_dig_tr_id = TrackID(xm_layer, clkp_pmos_dig_tr_xm, width=clk_width_xm)
        clkn_pmos_dig_tr_id = TrackID(xm_layer, clkn_pmos_dig_tr_xm, width=clk_width_xm)
        clkp_pmos_dig_list, clkn_pmos_dig_list = [], []

        if datapath_parity == 0:
            clkp, clkn = 'clkp', 'clkn'
        else:
            clkp, clkn = 'clkn', 'clkp'

        # connect integ biases
        integ_col, integ_info = block_info['integ']
        col_intv = integ_col, integ_col + integ_info['fg_tot']
        left_sig_vm = layout_info.get_center_tracks(vm_layer, 2, col_intv, width=sig_width_vm, space=sig_space_vm)
        mid_tr_vm = left_sig_vm + (sig_width_vm + sig_space_vm) / 2
        right_tr_vm = left_sig_vm - (sig_width_vm + clk_width_vm) / 2 - sig_clk_space_vm
        rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)
        mtr_id = TrackID(vm_layer, mid_tr_vm, width=clk_width_vm)
        # nmos_analog
        warr = self.connect_to_tracks(integ_ports['bias_tail'], rtr_id, fill_type='VSS')
        clkp_nmos_ana_list.append(warr)
        # pmos_integ
        warr = self.connect_to_tracks(integ_ports['bias_load'], mtr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkp_pmos_integ_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkp + '_pmos_integ', warr, show=show_pins)
        # nmos_switch
        warr = self.connect_to_tracks(integ_ports['sw'], mtr_id, fill_type='VDD')
        clkn_nmos_sw_list.append(warr)

        # connect alat biases
        alat_col, alat_info = block_info['alat']
        col_intv = alat_col, alat_col + alat_info['fg_tot']
        left_tr_vm = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm, space=sig_space_vm)
        left_tr_vm += (sig_width_vm + sig_space_vm) / 2
        right_tr_vm = left_tr_vm + 2 * (sig_width_vm + sig_space_vm)
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)
        if datapath_parity == 0:
            nmos_tr_id, pmos_tr_id, sw_tr_id = ltr_id, ltr_id, rtr_id
        else:
            nmos_tr_id, pmos_tr_id, sw_tr_id = rtr_id, rtr_id, ltr_id
        # nmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_tail'], nmos_tr_id, fill_type='VSS')
        clkp_nmos_ana_list.append(warr)
        # pmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_load'], pmos_tr_id, fill_type='VDD')
        xtr_id = TrackID(xm_layer, clkn_pmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, fill_type='VDD', min_len_mode=0)
        self.add_pin(clkn + '_pmos_analog', warr, show=show_pins)
        # nmos_switch
        warr = self.connect_to_tracks(alat_ports['sw'], sw_tr_id, fill_type='VDD')
        clkn_nmos_sw_list.append(warr)

        # connect dlat
        for dfe_idx, (dlat_col, dlat_ports, dlat_info) in enumerate(block_info['dlat']):

            if dfe_idx == 0:
                left_sig_vm = layout_info.get_center_tracks(vm_layer, 4, tap1_col_intv, width=sig_width_vm,
                                                            space=sig_space_vm)
            else:
                col_intv = dlat_col, dlat_col + dlat_info['fg_tot']
                left_sig_vm = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm,
                                                            space=sig_space_vm)
            same_tr_vm = left_sig_vm - (sig_width_vm + clk_width_vm) / 2 - sig_clk_space_vm
            left_tr_vm = left_sig_vm + (sig_width_vm + sig_space_vm) / 2
            right_tr_vm = left_tr_vm + 2 * (sig_width_vm + sig_space_vm)
            str_id = TrackID(vm_layer, same_tr_vm, width=clk_width_vm)
            ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
            rtr_id = TrackID(vm_layer, right_tr_vm, width=clk_width_vm)

            if dfe_idx % 2 == 1:
                nmos_tr_id = ltr_id if datapath_parity == 0 else rtr_id
                pmos_tr_id = nmos_tr_id
                sw_tr_id = str_id
            else:
                nmos_tr_id = str_id
                pmos_tr_id = nmos_tr_id
                sw_tr_id = ltr_id if datapath_parity == 0 else rtr_id

            nwarr = self.connect_to_tracks(dlat_ports['bias_tail'], nmos_tr_id, fill_type='VSS')
            pwarr = self.connect_to_tracks(dlat_ports['bias_load'], pmos_tr_id, fill_type='VDD')
            swarr = self.connect_to_tracks(dlat_ports['sw'], sw_tr_id, fill_type='VDD')

            if dfe_idx % 2 == 1:
                clkp_nmos_dig_list.append(nwarr)
                clkn_pmos_dig_list.append(pwarr)
                clkn_nmos_sw_list.append(swarr)
            else:
                clkn_nmos_dig_list.append(nwarr)
                clkp_pmos_dig_list.append(pwarr)
                clkp_nmos_sw_list.append(swarr)

        warr = self.connect_to_tracks(clkp_nmos_sw_list, clkp_nmos_sw_tr_id, fill_type='VDD')
        self.add_pin(clkp + '_nmos_switch', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_nmos_sw_list, clkn_nmos_sw_tr_id, fill_type='VDD')
        self.add_pin(clkn + '_nmos_switch', warr, show=show_pins)
        warr = self.connect_to_tracks(clkp_nmos_ana_list, clkp_nmos_ana_tr_id, fill_type='VSS')
        self.add_pin(clkp + '_nmos_analog', warr, show=show_pins)
        warr = self.connect_to_tracks(clkp_nmos_dig_list, clkp_nmos_dig_tr_id, fill_type='VSS')
        self.add_pin(clkp + '_nmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_nmos_dig_list, clkn_nmos_dig_tr_id, fill_type='VSS')
        self.add_pin(clkn + '_nmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkp_pmos_dig_list, clkp_pmos_dig_tr_id, fill_type='VDD')
        self.add_pin(clkp + '_pmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_pmos_dig_list, clkn_pmos_dig_tr_id, fill_type='VDD')
        self.add_pin(clkn + '_pmos_digital', warr, show=show_pins)

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
            min_fg_sep=0,
            hm_width=1,
            hm_cur_width=-1,
            sig_widths=[1, 1],
            sig_spaces=[1, 1],
            clk_widths=[1, 1],
            clk_spaces=[1, 1],
            sig_clk_spaces=[1, 1],
            show_pins=False,
            guard_ring_nf=0,
            data_parity=0,
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
            tap1_col_intv='DFE tap1 feedback gm transistor column interval.',
            fg_tot='Total number of fingers.',
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            sig_widths='signal wire widths on each layer above hm layer.',
            sig_spaces='signal wire spacing on each layer above hm layer.',
            clk_widths='clk wire widths on each layer above hm layer.',
            clk_spaces='clk wire spacing on each layer above hm layer.',
            sig_clk_spaces='spacing between signal and clk on each layer above hm layer.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            datapath_parity='Parity of the DDR datapath.  Either 0 or 1.',
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
        min_fg_sep = self.params['min_fg_sep']
        layout_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)

        bot_inst, top_inst, col_idx_dict = self.place(layout_info)
        self.connect(layout_info, bot_inst, top_inst, col_idx_dict)
        self._col_idx_dict = col_idx_dict

    def place(self, layout_info):
        # type: (SerdesRXBaseInfo) -> Tuple[Instance, Instance, Dict[str, Any]]
        alat_params_list = self.params['alat_params_list']
        integ_params = self.params['integ_params']
        intsum_params = self.params['intsum_params']
        summer_params = self.params['summer_params']
        dlat_params_list = self.params['dlat_params_list']
        nduml = self.params['nduml']
        ndumr = self.params['ndumr']
        sig_width_vm = self.params['sig_widths'][0]
        sig_space_vm = self.params['sig_spaces'][0]
        clk_width_vm = self.params['clk_widths'][0]
        sig_clk_space_vm = self.params['sig_clk_spaces'][0]
        # create AnalogBaseInfo object
        vm_layer_id = layout_info.mconn_port_layer + 2
        dtr_pitch = sig_width_vm + sig_space_vm
        diff_clk_route_tracks = 2 * sig_width_vm + 2 * clk_width_vm + sig_space_vm + 3 * sig_clk_space_vm

        # compute block locations
        col_idx_dict = {}
        # step 0: place integrating frontend.
        cur_col = nduml
        integ_col_idx = cur_col
        # print('integ_col: %d' % cur_col)
        # step 0A: find minimum number of fingers
        new_integ_params = integ_params.copy()
        integ_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_clk_route_tracks, cur_col)
        new_integ_params['min'] = integ_fg_min
        integ_info = layout_info.get_diffamp_info(new_integ_params)
        new_integ_params['col_idx'] = integ_col_idx
        integ_fg_tot = integ_info['fg_tot']
        col_idx_dict['integ'] = (cur_col, cur_col + integ_fg_tot)
        cur_col += integ_fg_tot
        # step 0B: reserve routing tracks between integrator and analog latch
        route_integ_alat_fg = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * dtr_pitch, cur_col)
        col_idx_dict['integ_route'] = (cur_col, cur_col + route_integ_alat_fg)
        # print('integ_route_col: %d' % cur_col)
        cur_col += route_integ_alat_fg

        # step 1: place analog latches
        alat_col_idx = cur_col
        # print('alat_col: %d' % cur_col)
        alat1_params = alat_params_list[0].copy()
        alat2_params = alat_params_list[1].copy()
        # step 1A: find minimum number of fingers
        alat_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 4 * dtr_pitch, cur_col)
        # step 1B: make both analog latches have the same width
        alat1_params['min'] = alat_fg_min
        alat1_info = layout_info.get_diffamp_info(alat1_params)
        alat2_params['min'] = alat_fg_min
        alat2_info = layout_info.get_diffamp_info(alat2_params)
        alat_fg_min = max(alat1_info['fg_tot'], alat2_info['fg_tot'])
        alat1_params['min'] = alat_fg_min
        alat2_params['min'] = alat_fg_min
        alat1_params['col_idx'] = alat_col_idx
        alat2_params['col_idx'] = alat_col_idx
        col_idx_dict['alat'] = (cur_col, cur_col + alat_fg_min)
        cur_col += alat_fg_min
        # step 1C: reserve routing tracks between analog latch and intsum
        route_alat_intsum_fg = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * dtr_pitch, cur_col)
        col_idx_dict['alat_route'] = (cur_col, cur_col + route_alat_intsum_fg)
        # print('alat_route_col: %d' % cur_col)
        cur_col += route_alat_intsum_fg

        # step 2: place intsum and most digital latches
        # assumption: we assume the load/offset fingers < total gm fingers,
        # so we can use gm_info to determine sizing.
        intsum_col_idx = cur_col
        # print('intsum_main_col: %d' % cur_col)
        col_idx_dict['intsum'] = []
        col_idx_dict['dlat'] = [(0, 0)] * len(dlat_params_list)
        intsum_gm_fg_list = intsum_params['gm_fg_list']
        intsum_gm_sep_list = [layout_info.min_fg_sep] * (len(intsum_gm_fg_list) - 1)
        # step 2A: place main tap.  No requirements.
        intsum_main_info = layout_info.get_gm_info(intsum_gm_fg_list[0])
        new_intsum_gm_fg_list = [intsum_gm_fg_list[0]]
        cur_col += intsum_main_info['fg_tot']
        col_idx_dict['intsum'].append((intsum_col_idx, cur_col))
        cur_col += intsum_gm_sep_list[0]
        # print('cur_col: %d' % cur_col)
        # step 2B: place precursor tap.  must fit one differential track + 2 clk rout track
        # print('intsum_pre_col: %d' % cur_col)
        intsum_pre_col_idx = cur_col
        intsum_pre_fg_params = intsum_gm_fg_list[1].copy()
        intsum_pre_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_clk_route_tracks, cur_col)
        intsum_pre_fg_params['min'] = intsum_pre_fg_min
        intsum_pre_info = layout_info.get_gm_info(intsum_pre_fg_params)
        new_intsum_gm_fg_list.append(intsum_pre_fg_params)
        cur_col += intsum_pre_info['fg_tot']
        col_idx_dict['intsum'].append((intsum_pre_col_idx, cur_col))
        cur_col += intsum_gm_sep_list[1]
        # print('cur_col: %d' % cur_col)
        # step 2C: place intsum DFE taps
        num_dfe = len(intsum_gm_fg_list) - 2 + 1
        new_dlat_params_list = [None] * len(dlat_params_list)
        for idx in range(2, len(intsum_gm_fg_list)):
            # print('intsum_idx%d_col: %d' % (idx, cur_col))
            intsum_dfe_fg_params = intsum_gm_fg_list[idx].copy()
            dfe_idx = num_dfe - (idx - 2)
            dig_latch_params = dlat_params_list[dfe_idx - 2].copy()
            in_route = False
            num_route_tracks = diff_clk_route_tracks
            if dfe_idx > 2:
                # for DFE tap > 2, the intsum Gm stage must align with the corresponding
                # digital latch.
                # set digital latch column index
                if dfe_idx % 2 == 1:
                    # for odd DFE taps, we have criss-cross connections, so fit 2 diff tracks + 2 clk tracks
                    num_route_tracks = 4 * sig_width_vm + 2 * clk_width_vm + 3 * sig_space_vm + 3 * sig_clk_space_vm
                    # for odd DFE taps > 3, we need to reserve additional input routing tracks
                    in_route = dfe_idx > 3

                # fit diff tracks and make diglatch and DFE tap have same width
                intsum_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, num_route_tracks, cur_col)
                dig_latch_params['min'] = intsum_dfe_fg_min
                dlat_info = layout_info.get_diffamp_info(dig_latch_params)
                intsum_dfe_fg_params['min'] = dlat_info['fg_tot']
                intsum_dfe_info = layout_info.get_gm_info(intsum_dfe_fg_params)
                num_fg = intsum_dfe_info['fg_tot']
                intsum_dfe_fg_params['min'] = num_fg
                dig_latch_params['min'] = num_fg
                dig_latch_params['col_idx'] = cur_col
                col_idx_dict['dlat'][dfe_idx - 2] = (cur_col, cur_col + num_fg)
                col_idx_dict['intsum'].append((cur_col, cur_col + num_fg))
                cur_col += num_fg
                # print('cur_col: %d' % cur_col)
                if in_route:
                    # allocate input route
                    route_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * dtr_pitch, cur_col)
                    intsum_gm_sep_list[idx] = route_fg_min
                    col_idx_dict['dlat%d_inroute' % (dfe_idx - 2)] = (cur_col, cur_col + route_fg_min)
                    cur_col += route_fg_min
                    # print('cur_col: %d' % cur_col)
                else:
                    cur_col += intsum_gm_sep_list[idx]
                    # print('cur_col: %d' % cur_col)
            else:
                # for DFE tap 2, we have no requirements for digital latch
                intsum_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, num_route_tracks, cur_col)
                intsum_dfe_fg_params['min'] = intsum_dfe_fg_min
                intsum_dfe_info = layout_info.get_gm_info(intsum_dfe_fg_params)
                num_fg = intsum_dfe_info['fg_tot']
                col_idx_dict['intsum'].append((cur_col, cur_col + num_fg))
                # no need to add gm sep because this is the last tap.
                cur_col += num_fg
                # print('cur_col: %d' % cur_col)
            # save modified parameters
            new_dlat_params_list[dfe_idx - 2] = dig_latch_params
            new_intsum_gm_fg_list.append(intsum_dfe_fg_params)
        # print('intsum_last_col: %d' % cur_col)
        # step 2D: reserve routing tracks between intsum and summer
        route_intsum_sum_fg = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * dtr_pitch, cur_col)
        col_idx_dict['summer_route'] = (cur_col, cur_col + route_intsum_sum_fg)
        cur_col += route_intsum_sum_fg
        # print('summer cur_col: %d' % cur_col)

        # step 3: place DFE summer and first digital latch
        # assumption: we assume the load fingers < total gm fingers,
        # so we can use gm_info to determine sizing.
        summer_col_idx = cur_col
        col_idx_dict['summer'] = []
        summer_gm_fg_list = summer_params['gm_fg_list']
        summer_gm_sep_list = [layout_info.min_fg_sep]
        # step 3A: place main tap.  No requirements.
        summer_main_info = layout_info.get_gm_info(summer_gm_fg_list[0])
        new_summer_gm_fg_list = [summer_gm_fg_list[0]]
        cur_col += summer_main_info['fg_tot']
        col_idx_dict['summer'].append((summer_col_idx, cur_col))
        cur_col += summer_gm_sep_list[0]
        # print('cur_col: %d' % cur_col)
        # step 3B: place DFE tap.  must fit two differential tracks
        summer_dfe_col_idx = cur_col
        summer_dfe_fg_params = summer_gm_fg_list[1].copy()
        summer_dfe_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 4 * dtr_pitch, cur_col)
        summer_dfe_fg_params['min'] = summer_dfe_fg_min
        summer_dfe_info = layout_info.get_gm_info(summer_dfe_fg_params)
        new_summer_gm_fg_list.append(summer_dfe_fg_params)
        cur_col += summer_dfe_info['fg_tot']
        tap1_col_intv = (summer_dfe_col_idx, cur_col)
        col_idx_dict['summer'].append(tap1_col_intv)
        # print('cur_col: %d' % cur_col)
        # step 3C: place first digital latch
        # only requirement is that the right side line up with summer.
        dig_latch_params = dlat_params_list[0].copy()
        new_dlat_params_list[0] = dig_latch_params
        dlat_info = layout_info.get_diffamp_info(dig_latch_params)
        dig_latch_params['col_idx'] = cur_col - dlat_info['fg_tot']
        col_idx_dict['dlat'][0] = (cur_col - dlat_info['fg_tot'], cur_col)

        fg_tot = cur_col + ndumr
        # add dummies until we have multiples of block pitch
        blk_w = self.grid.get_block_size(layout_info.mconn_port_layer + 3, unit_mode=True)[0]
        sd_pitch_unit = layout_info.sd_pitch_unit
        cur_width = layout_info.get_total_width(fg_tot) * sd_pitch_unit
        final_w = -(-cur_width // blk_w) * blk_w
        fg_tot = final_w // sd_pitch_unit
        self._fg_tot = fg_tot

        # make RXHalfBottom
        bot_params = {key: self.params[key] for key in RXHalfTop.get_params_info().keys()
                      if key in self.params}
        bot_params['fg_tot'] = fg_tot
        bot_params['integ_params'] = new_integ_params
        bot_params['alat_params'] = alat1_params
        bot_params['dlat_params_list'] = new_dlat_params_list
        bot_params['tap1_col_intv'] = tap1_col_intv
        bot_params['show_pins'] = False
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
        top_params['show_pins'] = False
        # print('top summer col: %d' % top_params['summer_params']['col_idx'])
        top_master = self.new_template(params=top_params, temp_cls=RXHalfTop)
        top_inst = self.add_instance(top_master, orient='MX')
        top_inst.move_by(dy=bot_inst.array_box.top - top_inst.array_box.bottom)
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.set_size_from_array_box(top_master.size[0])

        show_pins = self.params['show_pins']
        for port_name in bot_inst.port_names_iter():
            self.reexport(bot_inst.get_port(port_name), show=show_pins)
        for port_name in top_inst.port_names_iter():
            self.reexport(top_inst.get_port(port_name), show=show_pins)

        return bot_inst, top_inst, col_idx_dict

    def connect(self, layout_info, bot_inst, top_inst, col_idx_dict):
        vm_space = self.params['sig_spaces'][0]
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        vm_width = self.params['sig_widths'][0]
        nintsum = len(self.params['intsum_params']['gm_fg_list'])

        # connect integ to alat1
        self._connect_diff_io(bot_inst, col_idx_dict['integ_route'], layout_info, vm_layer,
                              vm_width, vm_space, 'integ_out{}', 'alat0_in{}')

        # connect alat1 to intsum
        self._connect_diff_io(top_inst, col_idx_dict['alat_route'], layout_info, vm_layer,
                              vm_width, vm_space, 'alat1_out{}', 'intsum_in{}<0>')

        # connect intsum to summer
        self._connect_diff_io(top_inst, col_idx_dict['summer_route'], layout_info, vm_layer,
                              vm_width, vm_space, 'intsum_out{}', 'summer_in{}<0>')

        # connect DFE tap 2
        route_col_intv = col_idx_dict['intsum'][-1]
        ptr_idx = layout_info.get_center_tracks(vm_layer, 2, route_col_intv, width=vm_width, space=vm_space)
        ntr_idx = ptr_idx + vm_space + vm_width
        p_list = [bot_inst.get_port('dlat0_outp').get_pins()[0],
                  top_inst.get_port('intsum_inp<%d>' % (nintsum - 1)).get_pins()[0], ]
        n_list = [bot_inst.get_port('dlat0_outn').get_pins()[0],
                  top_inst.get_port('intsum_inn<%d>' % (nintsum - 1)).get_pins()[0], ]
        if nintsum > 3:
            p_list.append(bot_inst.get_port('dlat1_inp').get_pins()[0])
            n_list.append(bot_inst.get_port('dlat1_inn').get_pins()[0])

        self.connect_differential_tracks(p_list, n_list, vm_layer, ptr_idx, ntr_idx, width=vm_width,
                                         fill_type='VDD')

        # connect even DFE taps
        ndfe = nintsum - 2 + 1
        for dfe_idx in range(4, ndfe + 1, 2):
            intsum_idx = nintsum - 1 - (dfe_idx - 2)
            route_col_intv = col_idx_dict['intsum'][intsum_idx]
            ptr_idx = layout_info.get_center_tracks(vm_layer, 2, route_col_intv, width=vm_width, space=vm_space)
            ntr_idx = ptr_idx + vm_space + vm_width
            p_list = [bot_inst.get_port('dlat%d_outp' % (dfe_idx - 2)).get_pins()[0],
                      top_inst.get_port('intsum_inp<%d>' % intsum_idx).get_pins()[0], ]
            n_list = [bot_inst.get_port('dlat%d_outn' % (dfe_idx - 2)).get_pins()[0],
                      top_inst.get_port('intsum_inn<%d>' % intsum_idx).get_pins()[0], ]
            self.connect_differential_tracks(p_list, n_list, vm_layer, ptr_idx, ntr_idx, width=vm_width,
                                             fill_type='VDD')
            if dfe_idx + 1 <= ndfe:
                # connect to next digital latch
                self._connect_diff_io(bot_inst, col_idx_dict['dlat%d_inroute' % (dfe_idx - 1)],
                                      layout_info, vm_layer, vm_width, vm_space,
                                      'dlat%d_out{}' % (dfe_idx - 2), 'dlat%d_in{}' % (dfe_idx - 1))

    def _connect_diff_io(self, inst, route_col_intv, layout_info, vm_layer, vm_width, vm_space,
                         out_name, in_name):
        ptr_idx = layout_info.get_center_tracks(vm_layer, 2, route_col_intv, width=vm_width, space=vm_space)
        ntr_idx = ptr_idx + vm_space + vm_width
        p_warrs = [inst.get_port(out_name.format('p')).get_pins()[0],
                   inst.get_port(in_name.format('p')).get_pins()[0], ]
        n_warrs = [inst.get_port(out_name.format('n')).get_pins()[0],
                   inst.get_port(in_name.format('n')).get_pins()[0], ]
        self.connect_differential_tracks(p_warrs, n_warrs, vm_layer, ptr_idx, ntr_idx,
                                         width=vm_width, fill_type='VDD')

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
            min_fg_sep=0,
            nduml=4,
            ndumr=4,
            hm_width=1,
            hm_cur_width=-1,
            sig_widths=[1, 1],
            sig_spaces=[1, 1],
            clk_widths=[1, 1],
            clk_spaces=[1, 1],
            sig_clk_spaces=[1, 1],
            guard_ring_nf=0,
            show_pins=False,
            datapath_parity=0,
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
            min_fg_sep='Minimum separation between transistors.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            sig_widths='signal wire widths on each layer above hm layer.',
            sig_spaces='signal wire spacing on each layer above hm layer.',
            clk_widths='clk wire widths on each layer above hm layer.',
            clk_spaces='clk wire spacing on each layer above hm layer.',
            sig_clk_spaces='spacing between signal and clk on each layer above hm layer.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            show_pins='True to draw layout pins.',
            datapath_parity='Parity of the DDR datapath.  Either 0 or 1.',
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
        half_params['datapath_parity'] = 0
        half_params['show_pins'] = False
        even_master = self.new_template(params=half_params, temp_cls=RXHalf)
        half_params['datapath_parity'] = 1
        half_params['show_pins'] = False
        odd_master = self.new_template(params=half_params, temp_cls=RXHalf)
        odd_inst = self.add_instance(odd_master, 'X1', orient='MX')
        odd_inst.move_by(dy=odd_master.bound_box.height)
        even_inst = self.add_instance(even_master, 'X0')
        even_inst.move_by(dy=odd_inst.array_box.top - even_inst.array_box.bottom)

        self.array_box = odd_inst.array_box.merge(even_inst.array_box)
        self.set_size_from_array_box(even_master.size[0])
        self._fg_tot = even_master.num_fingers

        col_idx_dict = even_master.get_column_index_table()
        inst_list = [even_inst, odd_inst]
        lch = self.params['lch']
        guard_ring_nf = self.params['guard_ring_nf']
        min_fg_sep = self.params['min_fg_sep']
        layout_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        self.connect_signal(inst_list, col_idx_dict, layout_info)

    def connect_signal(self, inst_list, col_idx_dict, layout_info):
        hm_layer_id = layout_info.mconn_port_layer + 1
        vm_layer_id = hm_layer_id + 1
        vm_space = self.params['sig_spaces'][0]
        vm_width = self.params['sig_widths'][0]
        vm_pitch = vm_width + vm_space
        show_pins = self.params['show_pins']

        # connect inputs of even and odd paths
        route_col_intv = col_idx_dict['integ']
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 2, route_col_intv, width=vm_width, space=vm_space)
        ports = ['integ_in{}']
        inp, inn = self._connect_differential(inst_list, ptr_idx, vm_layer_id, vm_width, vm_space,
                                              ports, ports)

        # export inputs/outputs
        self.add_pin('inp', inp, show=show_pins)
        self.add_pin('inn', inn, show=show_pins)
        for idx in (0, 1):
            for pname, oname in (('summer', 'summer'), ('dlat0', 'data')):
                self.reexport(inst_list[idx].get_port('%s_outp' % pname),
                              net_name='outp_%s<%d>' % (oname, idx), show=show_pins)
                self.reexport(inst_list[idx].get_port('%s_outn' % pname),
                              net_name='outn_%s<%d>' % (oname, idx), show=show_pins)

        # connect alat0 outputs
        route_col_intv = col_idx_dict['alat']
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 4, route_col_intv, width=vm_width, space=vm_space)
        tr_idx_list = [ptr_idx, ptr_idx + vm_pitch, ptr_idx + 2 * vm_pitch, ptr_idx + 3 * vm_pitch]
        warr_list_list = [[inst_list[0].get_port('alat0_outp').get_pins()[0],
                           inst_list[0].get_port('alat1_inp').get_pins()[0],
                           inst_list[1].get_port('ffe_inp').get_pins()[0],
                           ],
                          [inst_list[0].get_port('alat0_outn').get_pins()[0],
                           inst_list[0].get_port('alat1_inn').get_pins()[0],
                           inst_list[1].get_port('ffe_inn').get_pins()[0],
                           ],
                          [inst_list[1].get_port('alat0_outp').get_pins()[0],
                           inst_list[1].get_port('alat1_inp').get_pins()[0],
                           inst_list[0].get_port('ffe_inp').get_pins()[0],
                           ],
                          [inst_list[1].get_port('alat0_outn').get_pins()[0],
                           inst_list[1].get_port('alat1_inn').get_pins()[0],
                           inst_list[0].get_port('ffe_inn').get_pins()[0],
                           ], ]
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width, fill_type='VDD')
        # connect summer outputs
        route_col_intv = col_idx_dict['summer'][1]
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 4, route_col_intv, width=vm_width, space=vm_space)
        ports0 = ['summer_out{}', 'dlat0_in{}']
        ports1 = ['summer_in{}<1>']
        self._connect_differential(inst_list, ptr_idx, vm_layer_id, vm_width, vm_space,
                                   ports0, ports1)
        ptr_idx += 2 * vm_pitch
        self._connect_differential(inst_list, ptr_idx, vm_layer_id, vm_width, vm_space,
                                   ports1, ports0)

        # connect dlat1 outputs
        route_col_intv = col_idx_dict['dlat'][1]
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 4, route_col_intv, width=vm_width, space=vm_space)
        tr_idx_list = [ptr_idx, ptr_idx + vm_width + vm_space]
        warr_list_list = [[inst_list[0].get_port('dlat1_outp').get_pins()[0],
                           inst_list[0].get_port('dlat2_inp').get_pins()[0],
                           inst_list[1].get_port('intsum_inp<3>').get_pins()[0],
                           ],
                          [inst_list[0].get_port('dlat1_outn').get_pins()[0],
                           inst_list[0].get_port('dlat2_inn').get_pins()[0],
                           inst_list[1].get_port('intsum_inn<3>').get_pins()[0],
                           ], ]
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width, fill_type='VDD')
        tr_idx_list[0] += 2 * vm_pitch
        tr_idx_list[1] += 2 * vm_pitch
        warr_list_list = [[inst_list[1].get_port('dlat1_outp').get_pins()[0],
                           inst_list[1].get_port('dlat2_inp').get_pins()[0],
                           inst_list[0].get_port('intsum_inp<3>').get_pins()[0],
                           ],
                          [inst_list[1].get_port('dlat1_outn').get_pins()[0],
                           inst_list[1].get_port('dlat2_inn').get_pins()[0],
                           inst_list[0].get_port('intsum_inn<3>').get_pins()[0],
                           ], ]
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width, fill_type='VDD')

    def _connect_differential(self, inst_list, ptr_idx, vm_layer_id, vm_width, vm_space, even_ports, odd_ports):
        tr_idx_list = [ptr_idx, ptr_idx + vm_width + vm_space]
        warr_list_list = [[], []]
        for parity, warr_list in zip(('p', 'n'), warr_list_list):
            inst = inst_list[0]
            for name_fmt in even_ports:
                warr_list.append(inst.get_port(name_fmt.format(parity)).get_pins()[0])
            inst = inst_list[1]
            for name_fmt in odd_ports:
                warr_list.append(inst.get_port(name_fmt.format(parity)).get_pins()[0])

        trp, trn = self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width,
                                                fill_type='VDD')
        return trp, trn

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
            min_fg_sep=0,
            nduml=4,
            ndumr=4,
            diff_space=1,
            hm_width=1,
            hm_cur_width=-1,
            sig_widths=[1, 1],
            sig_spaces=[1, 1],
            clk_widths=[1, 1],
            clk_spaces=[1, 1],
            sig_clk_spaces=[1, 1],
            show_pins=True,
            rename_dict={},
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
            min_fg_sep='Minimum separation between transistors.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            sig_widths='signal wire widths on each layer above hm layer.',
            sig_spaces='signal wire spacing on each layer above hm layer.',
            clk_widths='clk wire widths on each layer above hm layer.',
            clk_spaces='clk wire spacing on each layer above hm layer.',
            sig_clk_spaces='spacing between signal and clk on each layer above hm layer.',
            show_pins='True to create pin labels.',
            rename_dict='port renaming dictionary',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
        )
