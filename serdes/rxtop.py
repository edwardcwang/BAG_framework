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
from bag.layout.routing import TrackID

from .rxpassive import RXClkArray, BiasBusIO, CTLE
from .rxcore import RXCore


class RXFrontend(TemplateBase):
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
        super(RXFrontend, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            show_pins=True
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
            core_params='RXCore parameters.',
            rxclk_params='RXClkArray parameters.',
            ctle_params='passive CTLE parameters.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None
        clk_inst0, clk_inst1, core_inst, ctle_inst, vdd_list, vss_list = self._place()

        self._connect_ctle(ctle_inst, core_inst)
        self._connect_clks(clk_inst0, clk_inst1, core_inst, vdd_list, vss_list)

    def _connect_ctle(self, ctle_inst, core_inst):
        show_pins = self.params['show_pins']

        # export input
        self.reexport(ctle_inst.get_port('inp'), show=show_pins)
        self.reexport(ctle_inst.get_port('inn'), show=show_pins)

        # connect signal
        for par in ('p', 'n'):
            w1 = ctle_inst.get_all_port_pins('out' + par)[0]
            w2 = core_inst.get_all_port_pins('in' + par)[0]
            self.connect_wires([w1, w2])

        ctle_vdds = ctle_inst.get_all_port_pins('VDD')
        core_vdds = core_inst.get_all_port_pins('VDD')[0]
        if ctle_vdds[0].middle > ctle_vdds[1].middle:
            ctle_vdds = (ctle_vdds[1], ctle_vdds[0])

        tid_list = []
        cvtid = core_vdds.track_id
        for tid in cvtid:
            tid_list.append(TrackID(cvtid.layer_id, tid, width=cvtid.width))
        self.connect_to_tracks(ctle_vdds[0], tid_list[0], track_upper=core_vdds.upper)
        self.connect_to_tracks(ctle_vdds[1], tid_list[1], track_upper=core_vdds.upper)

    def _connect_clks(self, clk_inst0, clk_inst1, core_inst, vdd_list, vss_list):
        rxclk_params = self.params['rxclk_params']
        clk_names = rxclk_params['clk_names']
        sup_name = 'VDD' if rxclk_params['passive_params']['sub_type'] == 'ntap' else 'VSS'
        ltr = clk_inst0.master.left_track
        rtr = clk_inst0.master.right_track
        mtr = (ltr + rtr) / 2
        ltr -= mtr
        rtr -= mtr

        sup_indices = []
        player = -1
        pwidth = 1
        clkp_warr, clkn_warr = None, None
        for name in clk_names:
            nname = 'clkn_' + name
            pname = 'clkp_' + name
            nport = clk_inst0.get_all_port_pins(nname)[0]
            pport = clk_inst1.get_all_port_pins(pname)[0]
            cur_pid = pport.track_id.base_index
            cur_nid = nport.track_id.base_index
            mid = (cur_pid + cur_nid) / 2
            if cur_pid != cur_nid:
                sup_indices.append((mid, False))

            for cur_name, port in ((nname, nport), (pname, pport)):
                self.connect_to_tracks(core_inst.get_all_port_pins(cur_name), port.track_id,
                                       track_lower=port.lower, track_upper=port.upper)
            if name == 'nmos_intsum':
                # draw clkp and clkn wires
                player = nport.layer_id
                pwidth = nport.track_id.width
                clkp_warr = self.connect_to_tracks(core_inst.get_all_port_pins('clkp'),
                                                   TrackID(player, rtr + mid, width=pwidth))
                clkn_warr = self.connect_to_tracks(core_inst.get_all_port_pins('clkn'),
                                                   TrackID(player, ltr + mid, width=pwidth))
            elif cur_pid == cur_nid:
                sup_indices.append((ltr + mid, True))
                sup_indices.append((rtr + mid, True))

        # connect supplies to M7
        vdd_list.extend(core_inst.get_all_port_pins('VDD'))
        vss_list.extend(core_inst.get_all_port_pins('VSS'))
        if sup_name == 'VDD':
            ext_vdd_list = clk_inst0.get_all_port_pins('VDD')
            ext_vdd_list.extend(clk_inst1.get_all_port_pins('VDD'))
            ext_vss_list = []
        else:
            ext_vss_list = clk_inst0.get_all_port_pins('VSS')
            ext_vss_list.extend(clk_inst1.get_all_port_pins('VSS'))
            ext_vdd_list = []

        vdd_indices = sup_indices[0::2]
        vss_indices = sup_indices[1::2]
        vdd_top_list, vss_top_list = [], []
        for idx_list, warr_list, ex_list, top_list in ((vdd_indices, vdd_list, ext_vdd_list, vdd_top_list),
                                                       (vss_indices, vss_list, ext_vss_list, vss_top_list)):
            for idx, extend in idx_list:
                if not extend:
                    top_list.append(self.connect_to_tracks(warr_list, TrackID(player, idx, width=pwidth)))
                else:
                    top_list.append(self.connect_to_tracks(warr_list + ex_list, TrackID(player, idx, width=pwidth)))


        show_pins = self.params['show_pins']
        self.add_pin('VDD', vdd_top_list, show=show_pins)
        self.add_pin('VSS', vss_top_list, show=show_pins)
        self.add_pin('clkp', clkp_warr, show=show_pins)
        self.add_pin('clkn', clkn_warr, show=show_pins)
        self.reexport(clk_inst0.get_all_port_pins('clkn'), net_name='clkno')
        self.reexport(clk_inst1.get_all_port_pins('clkp'), net_name='clkpo')

    def _place(self):
        rxclk_params = self.params['rxclk_params'].copy()
        core_params = self.params['core_params'].copy()
        ctle_params = self.params['ctle_params'].copy()

        rxclk_params['parity'] = 0
        rxclk_params['show_pins'] = False
        clk_master0 = self.new_template(params=rxclk_params, temp_cls=RXClkArray)
        rxclk_params['parity'] = 1
        clk_master1 = self.new_template(params=rxclk_params, temp_cls=RXClkArray)

        core_params['show_pins'] = False
        core_master = self.new_template(params=core_params, temp_cls=RXCore)

        in_xm_offset = core_master.in_offset
        ctle_params['cap_port_offset'] = in_xm_offset
        ctle_params['show_pins'] = False
        ctle_master = self.new_template(params=ctle_params, temp_cls=CTLE)

        clkw, clkh = self.grid.get_size_dimension(clk_master0.size, unit_mode=True)
        corew, coreh = self.grid.get_size_dimension(core_master.size, unit_mode=True)
        ctlew, ctleh = self.grid.get_size_dimension(ctle_master.size, unit_mode=True)
        maxw = max(clkw, corew)
        x_clk = ctlew + maxw - clkw
        x_core = ctlew + maxw - corew

        clk_inst0 = self.add_instance(clk_master0, 'XCLK0', loc=(x_clk, clkh), orient='MX', unit_mode=True)
        core_inst = self.add_instance(core_master, 'XCORE', loc=(x_core, clkh), unit_mode=True)
        clk_inst1 = self.add_instance(clk_master1, 'XCLK1', loc=(x_clk, clkh + coreh), unit_mode=True)
        ctle_inst = self.add_instance(ctle_master, 'XCTLE', loc=(0, 0), unit_mode=True)

        vss_names = ['bias_nmos_integ', 'bias_nmos_analog', 'bias_nmos_intsum',
                     'bias_nmos_digital', 'bias_nmos_summer', 'bias_nmos_tap1',
                     '{}_bias_dfe<1>', '{}_bias_dfe<2>', '{}_bias_dfe<3>']
        vdd_names = ['bias_pmos_analog', 'bias_pmos_digital', 'bias_pmos_summer',
                     '{}_bias_ffe', '{}_bias_offp', '{}_bias_offn', '{}_bias_dlev_outp',
                     '{}_bias_dlev_outn', ]
        en_names = ['{}_en_dfe<0>', '{}_en_dfe<1>', '{}_en_dfe<2>', '{}_en_dfe<3>']

        bus_order = [(vss_names, 'VSS'), (vdd_names, 'VDD'), (en_names, 'VDD')]
        vdd_list = []
        vss_list = []
        self._connect_bias_wires(clk_inst0, core_inst, [core_inst, clk_inst1], clkh, 'odd', bus_order,
                                 vdd_list, vss_list)
        bus_order = bus_order[::-1]
        self._connect_bias_wires(clk_inst1, core_inst, [clk_inst1], clk_inst1.location_unit[1], 'even', bus_order,
                                 vdd_list, vss_list)

        # move ctle to center of rxcore
        mid = core_inst.location_unit[1] + coreh // 2
        ctle_inst.move_by(dy=mid - ctleh // 2, unit_mode=True)

        return clk_inst0, clk_inst1, core_inst, ctle_inst, vdd_list, vss_list

    def _connect_bias_wires(self, clk_inst, core_inst, move_insts, yb, prefix, bus_order, vdd_list, vss_list):
        show_pins = self.params['show_pins']

        reserve_tracks = []
        for port_name in clk_inst.port_names_iter():
            if port_name.startswith('bias_'):
                warr = clk_inst.get_all_port_pins(port_name)[0]
                reserve_tracks.append((port_name, warr.layer_id, warr.track_id.base_index, warr.track_id.width))

        bus_layer = reserve_tracks[0][1] + 1
        bias_prefix = '%s_bias_' % prefix
        en_prefix = '%s_en_' % prefix
        for port_name in core_inst.port_names_iter():
            if port_name.startswith(bias_prefix) or port_name.startswith(en_prefix):
                warr = core_inst.get_all_port_pins(port_name)[0]
                reserve_tracks.append((port_name, warr.layer_id, warr.track_id.base_index, warr.track_id.width))

        cur_yb = yb
        delta_y = 0
        warr_dict = {}
        for bias_names, sup_name in bus_order:
            io_names = [name.format(prefix) for name in bias_names]
            bus_params = dict(
                io_names=io_names,
                sup_name=sup_name,
                reserve_tracks=reserve_tracks,
                bus_layer=bus_layer,
                bus_margin=1,
                show_pins=False,
            )
            bus_master = self.new_template(params=bus_params, temp_cls=BiasBusIO)
            bus_inst = self.add_instance(bus_master, loc=(0, cur_yb), unit_mode=True)
            bush = self.grid.get_size_dimension(bus_master.size, unit_mode=True)[1]
            cur_yb += bush
            delta_y += bush

            for name in io_names:
                self.reexport(bus_inst.get_port(name), show=show_pins)
                warr_dict[name] = bus_inst.get_all_port_pins(name + '_in')[0]

            if sup_name == 'VDD':
                vdd_list.extend(bus_inst.get_all_port_pins('VDD'))
            else:
                vss_list.extend(bus_inst.get_all_port_pins('VSS'))

        for minst in move_insts:
            minst.move_by(dy=delta_y, unit_mode=True)

        for port_name in clk_inst.port_names_iter():
            if port_name.startswith('bias_'):
                warr = clk_inst.get_all_port_pins(port_name)[0]
                self.connect_wires([warr, warr_dict[port_name]])

        for port_name in core_inst.port_names_iter():
            if port_name.startswith(bias_prefix) or port_name.startswith(en_prefix):
                warr = core_inst.get_all_port_pins(port_name)[0]
                self.connect_wires([warr, warr_dict[port_name]])
