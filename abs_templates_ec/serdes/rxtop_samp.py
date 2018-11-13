# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import TrackID
from bag.layout.util import BBox

from ..analog_core import AnalogBase
from .rxpassive import RXClkArray, BiasBusIO, CTLE, DLevCap
from .rxcore_samp import RXCore


class RXFrontendCore(TemplateBase):
    """RX frontend

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
        super(RXFrontendCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            bus_margin=1,
            show_pins=True,
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
            dlev_cap_params='dlev ac coupling cap parameters.',
            bus_margin='margin between bus wires and adjacent blocks.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None
        show_pins = self.params['show_pins']
        clk_inst0, clk_inst1, core_inst, ctle_inst, vdd_list, vss_list, x0 = self._place()

        self.reexport(core_inst.get_port('even_outp_dlat<0>'), net_name='outp_data<0>', show=show_pins)
        self.reexport(core_inst.get_port('odd_outp_dlat<0>'), net_name='outp_data<1>', show=show_pins)
        self.reexport(core_inst.get_port('even_outn_dlat<0>'), net_name='outn_data<0>', show=show_pins)
        self.reexport(core_inst.get_port('odd_outn_dlat<0>'), net_name='outn_data<1>', show=show_pins)

        self._connect_ctle(ctle_inst, core_inst, x0)
        self._connect_clks(clk_inst0, clk_inst1, core_inst, vdd_list, vss_list)

    def _connect_ctle(self, ctle_inst, core_inst, x0):
        show_pins = self.params['show_pins']

        # export input
        self.reexport(ctle_inst.get_port('inp'), show=show_pins)
        self.reexport(ctle_inst.get_port('inn'), show=show_pins)

        # export input common mode
        warrs = ctle_inst.get_all_port_pins('outcm')
        warr = self.connect_wires(warrs, lower=x0, unit_mode=True)
        self.add_pin('bias_incm', warr, show=show_pins)

        # connect signal
        for par in ('p', 'n'):
            w1 = ctle_inst.get_all_port_pins('out' + par)[0]
            w2 = core_inst.get_all_port_pins('in' + par)[0]
            self.connect_wires([w1, w2])

        sup_name = 'VSS'
        ctle_sups = ctle_inst.get_all_port_pins(sup_name)
        if ctle_sups[0].middle > ctle_sups[1].middle:
            ctle_sups = (ctle_sups[1], ctle_sups[0])

        core_sups = core_inst.get_all_port_pins(sup_name)
        tid_list = []
        for warr in core_sups:
            cvtid = warr.track_id
            for tid in cvtid:
                tid_list.append((tid, TrackID(cvtid.layer_id, tid, width=cvtid.width)))
        tid_list = sorted(tid_list)
        upper = core_sups[0].upper
        self.connect_to_tracks(ctle_sups[0], tid_list[0][1], track_upper=upper)
        self.connect_to_tracks(ctle_sups[1], tid_list[-1][1], track_upper=upper)

    def _connect_clks(self, clk_inst0, clk_inst1, core_inst, vdd_list, vss_list):
        rxclk_params = self.params['rxclk_params']
        clk_names = rxclk_params['clk_names']
        clk_master = clk_inst0.master
        port_layer = clk_master.output_layer
        track_pitch = clk_master.track_pitch
        mid_tracks = [clk_inst0.translate_master_track(port_layer, mid) for mid in clk_master.mid_tracks]
        sup_indices = []
        pwidth = 1
        for idx, (name, mid_tr) in enumerate(zip(clk_names, mid_tracks)):
            if name:
                nname = 'clkn_' + name
                pname = 'clkp_' + name
                nport = clk_inst0.get_all_port_pins(nname)[0]
                pport = clk_inst1.get_all_port_pins(pname)[0]
                pwidth = nport.track_id.width
                cur_pid = pport.track_id.base_index
                cur_nid = nport.track_id.base_index

                for cur_name, port in ((nname, nport), (pname, pport)):
                    if cur_name == 'clkp_nmos_tap1':
                        cur_name = 'even_clkp_nmos_tap1'
                    if cur_name == 'clkn_nmos_tap1':
                        cur_name = 'odd_clkn_nmos_tap1'
                    self.connect_to_tracks(core_inst.get_all_port_pins(cur_name), port.track_id,
                                           track_lower=port.lower, track_upper=port.upper)
                if cur_pid == mid_tr and cur_nid == mid_tr:
                    sup_indices.append((mid_tr - track_pitch, True))
                    sup_indices.append((mid_tr + track_pitch, True))

                if mid_tr != cur_pid and mid_tr != cur_nid:
                    sup_indices.append((mid_tr, False))

            # TODO: generalize?  This is hack for 16nm
            elif idx == 1:
                # use right track for supply
                sup_indices.append((mid_tr - track_pitch, False))
            elif idx == 2:
                sup_indices.append((mid_tr - track_pitch, False))
                sup_indices.append((mid_tr, False))
                sup_indices.append((mid_tr + track_pitch, False))

        # TODO: hard-coded hack for 16nm to connect clock wires
        clkp_warr = self.connect_to_tracks(core_inst.get_all_port_pins('clkp'),
                                           TrackID(port_layer, mid_tracks[3] - track_pitch, width=pwidth))
        clkn_warr = self.connect_to_tracks(core_inst.get_all_port_pins('clkn'),
                                           TrackID(port_layer, mid_tracks[3] + track_pitch, width=pwidth))

        # connect supplies to M7
        vdd_list.extend(core_inst.get_all_port_pins('VDD'))
        vss_list.extend(core_inst.get_all_port_pins('VSS'))
        for sup_name in ('VDD', 'VSS'):
            self.reexport(clk_inst0.get_port(sup_name), label=sup_name + ':')
            self.reexport(clk_inst1.get_port(sup_name), label=sup_name + ':')

        vdd_indices = sup_indices[0::2]
        vss_indices = sup_indices[1::2]
        vdd_top_list, vss_top_list = [], []
        for idx_list, warr_list, top_list in ((vdd_indices, vdd_list, vdd_top_list),
                                              (vss_indices, vss_list, vss_top_list)):
            for idx, _ in idx_list:
                top_list.append(self.connect_to_tracks(warr_list, TrackID(port_layer, idx, width=pwidth)))

        show_pins = self.params['show_pins']
        self.add_pin('VDD', vdd_top_list, show=show_pins, label='VDD:')
        self.add_pin('VSS', vss_top_list, show=show_pins, label='VSS:')

        self.add_pin('clkp', clkp_warr, show=show_pins, label='clkp:')
        self.add_pin('clkn', clkn_warr, show=show_pins, label='clkn:')
        self.reexport(clk_inst0.get_all_port_pins('clkn'), net_name='clkn', label='clkn:')
        self.reexport(clk_inst1.get_all_port_pins('clkp'), net_name='clkp', label='clkp:')

    def _place(self):
        rxclk_params = self.params['rxclk_params'].copy()
        core_params = self.params['core_params'].copy()
        ctle_params = self.params['ctle_params'].copy()
        dlev_cap_params = self.params['dlev_cap_params'].copy()
        bus_margin = self.params['bus_margin']
        show_pins = self.params['show_pins']

        # create template masters
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

        dlev_cap_params['show_pins'] = False
        dlev_cap_params['io_width'] = core_params['hm_cur_width']
        dlev_cap_params['io_space'] = core_params['diff_space']
        dcap_master = self.new_template(params=dlev_cap_params, temp_cls=DLevCap)

        clkw, clkh = self.grid.get_size_dimension(clk_master0.size, unit_mode=True)
        corew, coreh = self.grid.get_size_dimension(core_master.size, unit_mode=True)
        ctlew, ctleh = self.grid.get_size_dimension(ctle_master.size, unit_mode=True)

        # compute X coordinate
        # TODO: less hard coding
        vbus_layer = AnalogBase.get_mos_conn_layer(self.grid.tech_info) + 2
        ibias_width = 2
        bus_ibias_names = ['{}_ibias_nmos_intsum', '{}_ibias_dfe<1>', '{}_ibias_dfe<2>',
                           '{}_ibias_dfe<3>', '{}_ibias_offset']
        bus_vss_names = ['bias_nmos_analog', 'bias_nmos_digital', 'bias_nmos_summer', 'bias_nmos_tap1', ]
        bus_vdd_names = ['bias_pmos_analog', 'bias_pmos_digital', 'bias_pmos_summer',
                         '{}_bias_ffe', '{}_bias_dlevp', '{}_bias_dlevn', ]
        bus_en_names = ['{}_en_dfe1', '{}_offp', '{}_offn']

        num_track_tot = len(bus_ibias_names) + len(bus_vss_names) + \
                        len(bus_vdd_names) + len(bus_en_names) + (2 + bus_margin * 2) * 4
        # TODO: handle cases where vbus_layer's pitch is not multiple of all lower vertical layer's pitch
        bias_width = self.grid.get_track_pitch(vbus_layer, unit_mode=True) * num_track_tot

        maxw = max(clkw, corew)
        if maxw == corew:
            # add some room
            blk_w = self.grid.get_block_size(core_master.size[0], unit_mode=True)[0]
            maxw += -(-400 // blk_w) * blk_w
        x_clk = bias_width + ctlew + maxw - clkw
        x_core = bias_width + ctlew + maxw - corew

        clk_inst0 = self.add_instance(clk_master0, 'XCLK0', loc=(x_clk, clkh), orient='MX', unit_mode=True)
        core_inst = self.add_instance(core_master, 'XCORE', loc=(x_core, clkh), unit_mode=True)
        clk_inst1 = self.add_instance(clk_master1, 'XCLK1', loc=(x_clk, clkh + coreh), unit_mode=True)
        ctle_inst = self.add_instance(ctle_master, 'XCTLE', loc=(bias_width, 0), unit_mode=True)

        bus_order = [(bus_en_names, 'VDD', 1), (bus_vdd_names, 'VDD', 1),
                     (bus_vss_names, 'VSS', 1), (bus_ibias_names, 'VSS', ibias_width)]
        vdd_list = []
        vss_list = []
        self._connect_bias_wires(clk_inst0, core_inst, [core_inst, clk_inst1], clkh, 'odd', bus_order,
                                 vdd_list, vss_list, x_core)
        bus_order = bus_order[::-1]
        self._connect_bias_wires(clk_inst1, core_inst, [clk_inst1], clk_inst1.location_unit[1], 'even', bus_order,
                                 vdd_list, vss_list, x_core)

        # move ctle to center of rxcore
        mid = core_inst.location_unit[1] + coreh // 2
        ctle_inst.move_by(dy=mid - ctleh // 2, unit_mode=True)

        dcap_inst1 = self.add_instance(dcap_master, 'XDCAP1', loc=(x_core + corew, 0), orient='MX', unit_mode=True)
        dcap_inst0 = self.add_instance(dcap_master, 'XDCAP0', loc=(x_core + corew, 0), unit_mode=True)
        # move dcap inst to right Y location, then connect and export
        for idx, dinst in enumerate((dcap_inst0, dcap_inst1)):
            core_outp = core_inst.get_all_port_pins('outp_dlev<%d>' % idx)[0]
            dcap_outp = dinst.get_all_port_pins('outp')[0]
            hm_layer = core_outp.layer_id
            hm_pitch = self.grid.get_track_pitch(hm_layer)
            delta = core_outp.track_id.base_index - dcap_outp.track_id.base_index
            dinst.move_by(dy=hm_pitch * delta)

            for dname, cname in zip(['outp', 'outn', 'inp', 'inn'],
                                    ['outp_dlev', 'outn_dlev', 'outp_summer', 'outn_summer']):
                cname += '<%d>' % idx
                wlist = core_inst.get_all_port_pins(cname) + dinst.get_all_port_pins(dname)
                w = self.connect_wires(wlist)
                if dname.startswith('out'):
                    self.add_pin(cname, w, show=show_pins)

        # compute size
        xr = dcap_inst0.array_box.right_unit
        yt = clk_inst1.array_box.top_unit
        self.array_box = BBox(0, 0, xr, yt, self.grid.resolution, unit_mode=True)
        self.set_size_from_array_box(clk_master1.size[0])

        return clk_inst0, clk_inst1, core_inst, ctle_inst, vdd_list, vss_list, bias_width

    def _connect_bias_wires(self, clk_inst, core_inst, move_insts, yb, prefix, bus_order,
                            vdd_list, vss_list, bus_xo):
        show_pins = self.params['show_pins']

        reserve_tracks = []
        for port_name in clk_inst.port_names_iter():
            if port_name.startswith('bias_'):
                warr = clk_inst.get_all_port_pins(port_name)[0]
                cur_layer = warr.layer_id
                troff = bus_xo // self.grid.get_track_pitch(cur_layer, unit_mode=True)
                reserve_tracks.append((port_name, warr.layer_id, warr.track_id.base_index - troff, warr.track_id.width))

        bus_layer = reserve_tracks[0][1] + 1
        core_bus_names = {prefix + '_offp', prefix + '_offn', prefix + '_en_dfe1', prefix + '_bias_ffe',
                          prefix + '_bias_dlevp', prefix + '_bias_dlevn'}
        bias_prefix = '%s_ibias_' % prefix
        for port_name in core_inst.port_names_iter():
            if port_name.startswith(bias_prefix) or port_name in core_bus_names:
                warr = core_inst.get_all_port_pins(port_name)[0]
                cur_layer = warr.layer_id
                troff = bus_xo // self.grid.get_track_pitch(cur_layer, unit_mode=True)
                reserve_tracks.append((port_name, warr.layer_id, warr.track_id.base_index - troff, warr.track_id.width))

        cur_yb = yb
        delta_y = 0
        warr_dict = {}
        for bias_names, sup_name, track_width in bus_order:
            io_names = [name.format(prefix) for name in bias_names]
            bus_params = dict(
                io_names=io_names,
                sup_name=sup_name,
                reserve_tracks=reserve_tracks,
                bus_layer=bus_layer,
                bus_margin=1,
                show_pins=False,
                track_width=track_width,
            )
            bus_master = self.new_template(params=bus_params, temp_cls=BiasBusIO)
            bus_inst = self.add_instance(bus_master, loc=(bus_xo, cur_yb), unit_mode=True)
            bush = self.grid.get_size_dimension(bus_master.size, unit_mode=True)[1]
            cur_yb += bush
            delta_y += bush

            for name in io_names:
                if name == 'bias_nmos_tap1':
                    exp_name = prefix + '_bias_dfe<0>'
                    label = exp_name
                elif name.startswith(prefix):
                    exp_name = name
                    label = name
                else:
                    exp_name = name
                    label = name + ':'
                if bus_inst.has_port(name):
                    self.reexport(bus_inst.get_port(name), net_name=exp_name, label=label, show=show_pins)
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
            if not port_name.endswith('nmos_integ') and \
                    (port_name.startswith(bias_prefix) or port_name in core_bus_names):
                warr = core_inst.get_all_port_pins(port_name)[0]
                self.connect_wires([warr, warr_dict[port_name]])
