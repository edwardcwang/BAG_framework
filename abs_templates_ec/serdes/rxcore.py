# -*- coding: utf-8 -*-

from typing import Dict, Any, Set, Tuple

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance
from bag.layout.routing import TrackID

from .base import SerdesRXBase, SerdesRXBaseInfo


def connect_to_xm(template, warr_p, warr_n, col_intv, layout_info, sig_widths, sig_spaces, xm_mid_tr):
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
    vmp, vmn = template.connect_differential_tracks(warr_p, warr_n, vm_layer_id, p_tr, n_tr, width=vm_width)
    if xm_mid_tr is None:
        xm_mid_tr = template.grid.coord_to_nearest_track(xm_layer_id, vmp.middle, half_track=True, mode=0)
    nx_tr = xm_mid_tr - (xm_width + xm_space) / 2
    px_tr = nx_tr + xm_width + xm_space
    return template.connect_differential_tracks(vmp, vmn, xm_layer_id, px_tr, nx_tr, width=xm_width)


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
        self.sch_intsum_params = None
        self.sch_summer_params = None

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
            clk_widths=[1, 1, 1],
            clk_spaces=[1, 1, 1],
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
            acoff_params='AC coupling off transistor parameters.',
            buf_params='Integrator clock buffer parameters.',
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

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, alat_params, intsum_params, summer_params, acoff_params, buf_params,
                            show_pins, diff_space, hm_width, hm_cur_width,
                            sig_widths, sig_spaces, clk_widths, clk_spaces,
                            sig_clk_spaces, datapath_parity, **kwargs):
        draw_params = kwargs.copy()

        result = self.place(alat_params, intsum_params, summer_params, acoff_params, buf_params, draw_params,
                            diff_space, hm_width, hm_cur_width)
        alat_ports, intsum_ports, summer_ports, acoff_ports, buf_ports, block_info = result

        ffe_inputs = self.connect_sup_io(block_info, alat_ports, intsum_ports, summer_ports, acoff_ports,
                                         sig_widths, sig_spaces, clk_widths, clk_spaces, show_pins)

        self.connect_bias(block_info, alat_ports, intsum_ports, summer_ports, acoff_ports, buf_ports, ffe_inputs,
                          sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                          show_pins, datapath_parity)

    def place(self, alat_params, intsum_params, summer_params, acoff_params, buf_params, draw_params,
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
        draw_params['pds_tracks'] = [2 * hm_cur_width + diff_space]
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

        gate_locs['bias_load'] = (hm_width - 1) / 2

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

        # clock buffer
        buf_ports = self._draw_clock_buffer(buf_params, hm_width, hm_cur_width)

        # analog latch
        alat_col = alat_params.pop('col_idx')
        alat_flip_sd = alat_params.pop('flip_sd', False)
        # print('rxtop alat cur_col: %d' % cur_col)
        _, alat_ports = self.draw_diffamp(alat_col, alat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                          diff_space=diff_space, gate_locs=gate_locs, flip_sd=alat_flip_sd)
        alat_info = self.layout_info.get_diffamp_info(alat_params)

        # integrating summer
        intsum_col = intsum_params.pop('col_idx')
        # print('rxtop intsum cur_col: %d' % cur_col)
        _, intsum_ports = self.draw_gm_summer(intsum_col, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              **intsum_params)
        intsum_info = self.layout_info.get_summer_info(intsum_params['fg_load'], intsum_params['gm_fg_list'],
                                                       gm_sep_list=intsum_params.get('gm_sep_list', None),
                                                       flip_sd_list=intsum_params.get('flip_sd_list', None))

        # tap1 summer
        summer_col = summer_params.pop('col_idx')
        # print('rxtop summer cur_col: %d' % cur_col)
        _, summer_ports = self.draw_gm_summer(summer_col, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              **summer_params)
        summer_info = self.layout_info.get_summer_info(summer_params['fg_load'], summer_params['gm_fg_list'],
                                                       gm_sep_list=summer_params.get('gm_sep_list', None),
                                                       flip_sd_list=summer_params.get('flip_sd_list', None))

        # set schematic parameters
        self.sch_intsum_params = self._make_summer_sch_params(intsum_params, intsum_info)
        self.sch_summer_params = self._make_summer_sch_params(summer_params, summer_info)

        col_intv = acoff_params['col_intv']
        nac_off = acoff_params['nac_off']
        num_fg = col_intv[1] - col_intv[0]
        num_edge = (num_fg - 2 * nac_off) // 3
        nrow = self.get_nmos_row_index('casc')
        if nrow < 0:
            nrow = self.get_nmos_row_index('in')
        acp = self.draw_mos_conn('nch', nrow, col_intv[0] + num_edge, nac_off, 0, 2)
        acn = self.draw_mos_conn('nch', nrow, col_intv[1] - num_edge - nac_off, nac_off, 0, 2)
        gtr_idx = self.get_track_index('nch', nrow, 'g', (hm_width - 1) / 2)
        str_idx = gtr_idx - hm_width - 1
        dptr_idx = self.get_track_index('pch', 0, 'ds', (hm_cur_width - 1) / 2)
        dptr_idx += hm_cur_width + diff_space
        dntr_idx = dptr_idx - 3 * (hm_cur_width + diff_space)
        hm_layer = self.mos_conn_layer + 1

        self.connect_to_substrate('ptap', [acp['g'], acn['g']])
        sid = TrackID(hm_layer, str_idx, width=hm_width)
        sptr = self.connect_to_tracks(acp['s'], sid)
        sntr = self.connect_to_tracks(acn['s'], sid)
        dptr, dntr = self.connect_differential_tracks(acp['d'], acn['d'], hm_layer,
                                                      dptr_idx, dntr_idx, width=hm_cur_width)

        acoff_ports = dict(
            sp=sptr,
            sn=sntr,
            dp=dptr,
            dn=dntr,
        )

        block_info = dict(
            alat=(alat_col, alat_info),
            intsum=(intsum_col, intsum_info),
            summer=(summer_col, summer_info),
        )

        return alat_ports, intsum_ports, summer_ports, acoff_ports, buf_ports, block_info

    def _draw_clock_buffer(self, buf_params, hm_width, hm_cur_width):
        layout_info = self.layout_info
        col0 = buf_params['col_idx0']
        col1 = buf_params['col_idx1']
        fg0 = buf_params['fg0']
        fg1 = buf_params['fg1']
        nmos_ridx = self.get_nmos_row_index(buf_params['nmos_type'])

        hm_layer = self.mos_conn_layer + 1
        vm_layer = hm_layer + 1
        vm_width = self.params['clk_widths'][0]
        vm_space = self.params['clk_spaces'][0]

        # find track IDs
        out_tidx = self.get_num_tracks('pch', 0, 'ds') - (hm_cur_width + 1) / 2
        inp_tidx = self.get_num_tracks('pch', 0, 'g') - (hm_width + 1) / 2
        inn_tidx = self.get_num_tracks('nch', nmos_ridx, 'g') - (hm_width + 1) / 2
        out_tid = TrackID(hm_layer, self.get_track_index('pch', 0, 'ds', out_tidx), width=hm_cur_width)
        inp_tid = TrackID(hm_layer, self.get_track_index('pch', 0, 'g', inp_tidx), width=hm_width)
        inn_tid = TrackID(hm_layer, self.get_track_index('nch', nmos_ridx, 'g', inn_tidx), width=hm_width)

        # draw buffer 0
        pports = self.draw_mos_conn('pch', 0, col0, fg0, 2, 0)
        nports = self.draw_mos_conn('nch', nmos_ridx, col0, fg0, 0, 2)
        mid_warr = self.connect_to_tracks([pports['d'], nports['d']], out_tid)
        self.connect_to_substrate('ptap', nports['s'])
        self.connect_to_substrate('ntap', pports['s'])
        inp0_warr = self.connect_to_tracks(pports['g'], inp_tid)
        inn0_warr = self.connect_to_tracks(nports['g'], inn_tid)

        # draw buffer 1
        pports = self.draw_mos_conn('pch', 0, col1, fg1, 2, 0)
        nports = self.draw_mos_conn('nch', nmos_ridx, col1, fg1, 0, 2)
        out_warr = self.connect_to_tracks([pports['d'], nports['d']], out_tid)
        self.connect_to_substrate('ptap', nports['s'])
        self.connect_to_substrate('ntap', pports['s'])
        inp1_warr = self.connect_to_tracks(pports['g'], inp_tid)
        inn1_warr = self.connect_to_tracks(nports['g'], inn_tid)

        # connect buffers
        in_vm = layout_info.get_center_tracks(vm_layer, 1, (col0, col0 + fg0), width=vm_width, space=vm_space)
        in_warr = self.connect_to_tracks([inp0_warr, inn0_warr], TrackID(vm_layer, in_vm, width=vm_width))
        mid_vm = layout_info.get_center_tracks(vm_layer, 1, (col0 + fg0, col1), width=vm_width, space=vm_space)
        self.connect_to_tracks([mid_warr, inp1_warr, inn1_warr], TrackID(vm_layer, mid_vm, width=vm_width))
        out_vm = layout_info.get_center_tracks(vm_layer, 1, (col1, col1 + fg1), width=vm_width, space=vm_space)
        out_warr = self.connect_to_tracks(out_warr, TrackID(vm_layer, out_vm, width=vm_width))

        return {'in': in_warr, 'out': out_warr}

    @staticmethod
    def _make_summer_sch_params(params, info):
        sgn_list = list(params['sgn_list'])
        flip_sd_list = params.get('flip_sd_list', None)
        if flip_sd_list is not None:
            flip_sd_list = list(flip_sd_list)
        decap_list = params.get('decap_list', None)
        if decap_list is not None:
            decap_list = list(decap_list)
        load_decap_list = params.get('load_decap_list', None)
        if load_decap_list is not None:
            load_decap_list = list(load_decap_list)

        amp_fg_list = []
        amp_fg_tot_list = []
        for gm_fg_dict, load_fg, amp_info in zip(params['gm_fg_list'], info['fg_load_list'], info['amp_info_list']):
            amp_fg_dict = gm_fg_dict.copy()
            amp_fg_dict['load'] = load_fg
            amp_fg_list.append(amp_fg_dict)
            amp_fg_tot_list.append(amp_info['fg_tot'])

        return dict(
            amp_fg_list=amp_fg_list,
            amp_fg_tot_list=amp_fg_tot_list,
            sgn_list=sgn_list,
            decap_list=decap_list,
            load_decap_list=load_decap_list,
            flip_sd_list=flip_sd_list,
            fg_tot=info['fg_tot'],
        )

    def connect_sup_io(self, block_info, alat_ports, intsum_ports, summer_ports, acoff_ports,
                       sig_widths, sig_spaces, clk_widths, clk_spaces, show_pins):

        intsum_col, intsum_info = block_info['intsum']

        # get vdd/cascode bias pins
        vdd_list, cascl_list = [], []
        vdd_list.extend(alat_ports['vddt'])
        vdd_list.extend(intsum_ports[('vddt', -1)])
        vdd_list.extend(summer_ports[('vddt', -1)])
        if 'bias_casc' in alat_ports:
            cascl_list.extend(alat_ports['bias_casc'])

        # export alat inout pins
        inout_list = ('inp', 'inn', 'outp', 'outn')
        for name in inout_list:
            self.add_pin('alat1_%s' % name, alat_ports[name], show=show_pins)

        # connect ffe input to middle xm layer tracks, so we have room for vdd/vss wires.
        xm_layer_id = self.layout_info.mconn_port_layer + 3
        ffe_in_xm_mid_tr = (self.grid.get_num_tracks(self.size, xm_layer_id) - 1) / 2

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
                ffe_inp, ffe_inn = connect_to_xm(self, ffe_inp, ffe_inn, (ffe_start, ffe_stop), self.layout_info,
                                                 sig_widths, sig_spaces, ffe_in_xm_mid_tr)
                ffe_inputs = [ffe_inp, ffe_inn]
                self.add_pin('ffe_inp', ffe_inp, show=show_pins)
                self.add_pin('ffe_inn', ffe_inn, show=show_pins)
            else:
                self.add_pin('intsum_inp<%d>' % idx, intsum_ports[('inp', idx)], show=show_pins)
                self.add_pin('intsum_inn<%d>' % idx, intsum_ports[('inn', idx)], show=show_pins)

        self.add_pin('intsum_outp', intsum_ports[('outp', -1)], show=show_pins)
        self.add_pin('intsum_outn', intsum_ports[('outn', -1)], show=show_pins)

        # export dlev input pins
        self.add_pin('dlev_outp', acoff_ports['dp'], show=show_pins)
        self.add_pin('dlev_outn', acoff_ports['dn'], show=show_pins)
        wire_upper = acoff_ports['dp'].upper

        # export summer inout pins
        for name in ('inp', 'inn'):
            self.add_pin('summer_%s<0>' % name, summer_ports[(name, 0)], show=show_pins)
            self.add_pin('summer_%s<1>' % name, summer_ports[(name, 1)], show=show_pins)
        sum_outp = self.connect_wires(summer_ports[('outp', -1)], upper=wire_upper)
        sum_outn = self.connect_wires(summer_ports[('outn', -1)], upper=wire_upper)

        self.add_pin('summer_outp', sum_outp, show=show_pins)
        self.add_pin('summer_outn', sum_outn, show=show_pins)

        # connect and export supplies
        vss_name = self.get_pin_name('VSS')
        vdd_name = self.get_pin_name('VDD')
        vdd_warrs = []
        vddt = self.connect_wires(vdd_list, unit_mode=True)
        vdd_warrs.extend(vddt)
        warr = self.connect_wires(cascl_list, unit_mode=True)
        vdd_warrs.extend(warr)
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy(vdd_warrs=vdd_warrs, sup_margin=1, unit_mode=True)

        for warr in ptap_wire_arrs:
            self.add_pin(vss_name, warr, show=show_pins)

        for warr in ntap_wire_arrs:
            self.add_pin(vdd_name, warr, show=show_pins)

        # connect summer cascode
        # step 1: get vm track
        layout_info = self.layout_info
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        summer_col, summer_info = block_info['summer']
        casc_sum = summer_ports.get(('bias_casc', 0), [])
        summer_start = summer_col + summer_info['gm_offsets'][0]
        col_intv = summer_start, summer_start + summer_info['amp_info_list'][0]['fg_tot']
        clk_width_vm = clk_widths[0]
        clk_space_vm = clk_spaces[0]
        casc_tr = self.layout_info.get_center_tracks(vm_layer, 2, col_intv, width=clk_width_vm, space=clk_space_vm)
        # step 2: connect summer cascode to vdd
        casc_tr_id = TrackID(vm_layer, casc_tr, width=clk_width_vm)
        self.connect_to_tracks(ntap_wire_arrs + casc_sum, casc_tr_id)

        return ffe_inputs

    def connect_bias(self, block_info, alat_ports, intsum_ports, summer_ports, acoff_ports, buf_ports, ffe_inputs,
                     sig_widths, sig_spaces, clk_widths, clk_spaces, sig_clk_spaces,
                     show_pins, datapath_parity):
        layout_info = self.layout_info
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        sig_width_vm, sig_width_xm = sig_widths
        clk_width_vm, clk_width_xm = clk_widths[:2]
        sig_space_vm = sig_spaces[0]
        clk_space_vm, clk_space_xm = clk_spaces[:2]
        sig_clk_space_vm, sig_clk_space_xm = sig_clk_spaces

        # calculate bias track indices
        ffe_top_tr = ffe_inputs[0].track_id.base_index
        ffe_bot_tr = ffe_inputs[1].track_id.base_index
        clkp_nmos_sw_tr_xm = ffe_bot_tr - (sig_width_xm + clk_width_xm) / 2 - sig_clk_space_xm
        clkp_nmos_tap1_tr_xm = clkp_nmos_sw_tr_xm - clk_width_xm - clk_space_xm
        clkp_nmos_summer_tr_xm = clkp_nmos_sw_tr_xm + clk_width_xm + clk_space_xm
        clkn_nmos_ana_tr_xm = clkp_nmos_sw_tr_xm - clk_width_xm - clk_space_xm
        clkn_nmos_sw_tr_xm = ffe_top_tr + (sig_width_xm + clk_width_xm) / 2 + sig_clk_space_xm
        clkp_pmos_intsum_tr_xm = clkn_nmos_sw_tr_xm + clk_width_xm + clk_space_xm
        clkn_pmos_summer_tr_xm = clkp_pmos_intsum_tr_xm
        clkp_pmos_ana_tr_xm = clkn_nmos_sw_tr_xm

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
        nmos_tr_id, pmos_tr_id, sw_tr_id = rtr_id, rtr_id, ltr_id
        # nmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_tail'], nmos_tr_id)
        xtr_id = TrackID(xm_layer, clkn_nmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, min_len_mode=0)
        self.add_pin(clkn + '_nmos_analog', warr, show=show_pins)
        # pmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_load'], pmos_tr_id)
        xtr_id = TrackID(xm_layer, clkp_pmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, min_len_mode=0)
        self.add_pin(clkp + '_pmos_analog', warr, show=show_pins)
        # nmos_switch, connect to clock buffer input
        warr = self.connect_to_tracks(alat_ports['sw'], sw_tr_id)
        xtr_id = TrackID(xm_layer, clkp_nmos_sw_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks([warr, buf_ports['in']], xtr_id, min_len_mode=0)
        self.add_pin(clkp + '_nmos_switch_alat1', warr, show=show_pins)

        # connect intsum main tap biases
        intsum_col, intsum_info = block_info['intsum']
        intsum_start = intsum_col + intsum_info['gm_offsets'][0]
        col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][0]['fg_tot']
        left_tr_vm = self.layout_info.get_center_tracks(vm_layer, 2, col_intv, width=clk_width_vm, space=clk_space_vm)
        ltr_id = TrackID(vm_layer, left_tr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, left_tr_vm + clk_width_vm + clk_space_vm, width=clk_width_vm)
        # pmos intsum, M5 track 1
        pmos_intsum_list = [self.connect_to_tracks(intsum_ports[('bias_load', -1)], ltr_id)]

        # nmos switch.  Connect to vertical track later.
        nmax = len(intsum_info['gm_offsets'])
        warr_list = intsum_ports[('sw', 0)] + intsum_ports[('sw', 1)]
        intsum_sw_warr = self.connect_wires(warr_list)

        # nmos intsum
        intsum_start = intsum_col + intsum_info['gm_offsets'][1]
        col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][1]['fg_tot']
        ltr_vm, rtr_vm = get_bias_tracks(self.layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                         clk_width_vm, sig_clk_space_vm)
        warr = self.connect_wires(intsum_ports[('bias_tail', 0)] + intsum_ports[('bias_tail', 1)])
        ltr_id = TrackID(vm_layer, ltr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, rtr_vm, width=clk_width_vm)
        warr = self.connect_to_tracks(warr, ltr_id, track_lower=0)
        self.add_pin('ibias_nmos_intsum', warr, show=show_pins)
        # pmos intsum, M5 track 2, and clock buffer output
        # also connect intsum nmos switch to this vertical track.
        pmos_intsum_list.append(self.connect_to_tracks(intsum_ports[('bias_load', -1)], rtr_id))
        pmos_intsum_list.append(buf_ports['out'])
        xtr_id = TrackID(xm_layer, clkp_pmos_intsum_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(pmos_intsum_list, xtr_id, min_len_mode=0)
        self.add_pin(clkp + '_pmos_intsum', warr, show=show_pins)
        intsum_sw_warr = self.connect_to_tracks(intsum_sw_warr, rtr_id)
        clkn_nmos_sw_list.append(intsum_sw_warr)

        # connect offset biases/sign control
        intsum_start = intsum_col + intsum_info['gm_offsets'][2]
        col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][2]['fg_tot']
        ltr_vm, rtr_vm = get_bias_tracks(self.layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                         clk_width_vm, sig_clk_space_vm)
        ltr_id = TrackID(vm_layer, ltr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, rtr_vm, width=clk_width_vm)
        # connect ffe bias and load decap
        warr = self.connect_to_tracks(intsum_ports[('bias_casc', 1)] + intsum_ports[('bias_load_decap', -1)],
                                      ltr_id, track_lower=0)
        self.add_pin('bias_ffe', warr, show=show_pins)
        warr = self.connect_to_tracks(intsum_ports[('bias_tail', 2)], rtr_id, track_lower=0)
        self.add_pin('ibias_offset', warr, show=show_pins)
        p_tr = layout_info.get_center_tracks(vm_layer, 2, col_intv, width=sig_width_vm, space=sig_space_vm)
        n_tr = p_tr + sig_width_vm + sig_space_vm
        ptr_id = TrackID(vm_layer, p_tr, width=sig_width_vm)
        ntr_id = TrackID(vm_layer, n_tr, width=sig_width_vm)
        pwarr = self.connect_to_tracks(intsum_ports[('inp', 2)], ptr_id, track_lower=0)
        nwarr = self.connect_to_tracks(intsum_ports[('inn', 2)], ntr_id, track_lower=0)
        self.add_pin('offp', pwarr, show=show_pins)
        self.add_pin('offn', nwarr, show=show_pins)

        # connect intsum dfe tap biases
        num_dfe = nmax - 3
        for fb_idx in range(num_dfe):
            gm_idx = fb_idx + 3
            dfe_idx = num_dfe + 1 - fb_idx
            intsum_start = intsum_col + intsum_info['gm_offsets'][gm_idx]
            col_intv = intsum_start, intsum_start + intsum_info['amp_info_list'][gm_idx]['fg_tot']
            if dfe_idx % 2 == 0:
                # no criss-cross inputs.
                bias_tr_vm, sw_tr_vm = get_bias_tracks(self.layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                                       clk_width_vm, sig_clk_space_vm)
            else:
                # criss-cross inputs
                sw_tr_vm = self.layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm,
                                                              space=sig_space_vm)
                if datapath_parity == 0:
                    sw_tr_vm += (sig_width_vm + sig_space_vm) / 2
                    bias_tr_vm = sw_tr_vm + clk_width_vm + clk_space_vm
                else:
                    sw_tr_vm += (sig_width_vm + sig_space_vm) * 5 / 2
                    bias_tr_vm = sw_tr_vm - clk_width_vm - clk_space_vm

            # bias_dfe
            bias_tr_id = TrackID(vm_layer, bias_tr_vm, width=clk_width_vm)
            warr = self.connect_to_tracks(intsum_ports[('bias_tail', gm_idx)], bias_tr_id, track_lower=0)
            self.add_pin('ibias_dfe<%d>' % (dfe_idx - 1), warr, show=show_pins)
            # tail switch
            sw_tr_id = TrackID(vm_layer, sw_tr_vm, width=clk_width_vm)
            warr = self.connect_to_tracks(intsum_ports[('sw', gm_idx)], sw_tr_id)
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
        warr = self.connect_to_tracks(summer_ports[('bias_load', -1)], rtr_id)
        xtr_id = TrackID(xm_layer, clkn_pmos_summer_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, min_len_mode=0)
        self.add_pin(clkn + '_pmos_summer', warr, show=show_pins)
        # nmos summer
        warr = self.connect_to_tracks(summer_ports[('bias_tail', 0)], ltr_id)
        xtr_id = TrackID(xm_layer, clkp_nmos_summer_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id)
        self.add_pin(clkp + '_nmos_summer', warr, show=show_pins)
        # nmos switch
        sw_wire = self.connect_wires(summer_ports[('sw', 0)] + summer_ports[('sw', 1)])
        warr = self.connect_to_tracks(sw_wire, rtr_id, min_len_mode=0)
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
        warr = self.connect_to_tracks(summer_ports[('bias_casc', 1)], en_tr_id, track_lower=0)
        self.add_pin('en_dfe1', warr, show=show_pins)
        warr = self.connect_to_tracks(summer_ports[('bias_tail', 1)], tap_tr_id)
        xtr_id = TrackID(xm_layer, clkp_nmos_tap1_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id)
        self.add_pin(clkp + '_nmos_tap1', warr, show=show_pins)

        warr = self.connect_to_tracks(clkn_nmos_sw_list, clkn_nmos_sw_tr_id)
        self.add_pin(clkn + '_nmos_switch', warr, show=show_pins)

        # connect AC off transistor biasses
        acp = acoff_ports['sp']
        acn = acoff_ports['sn']
        ptr_vm = self.grid.coord_to_nearest_track(vm_layer, acp.middle, half_track=True)
        ntr_vm = self.grid.coord_to_nearest_track(vm_layer, acn.middle, half_track=True)
        ptr_id = TrackID(vm_layer, ptr_vm, width=clk_width_vm)
        ntr_id = TrackID(vm_layer, ntr_vm, width=clk_width_vm)

        warr = self.connect_to_tracks(acp, ptr_id, track_lower=0)
        self.add_pin('bias_dlevp', warr, show=show_pins)
        warr = self.connect_to_tracks(acn, ntr_id, track_lower=0)
        self.add_pin('bias_dlevn', warr, show=show_pins)


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
        self.in_xm_track = None

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
        draw_params['pds_tracks'] = [2 * hm_cur_width + diff_space]
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
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
        integ_pmos_vm_tid = integ_params.pop('integ_pmos_vm_tid')
        # if drawing current mirror, we must have ground on the outside of tail
        integ_flip_sd = integ_params.pop('flip_sd', False)
        _, integ_ports = self.draw_diffamp(integ_col, integ_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                           diff_space=diff_space, gate_locs=gate_locs, flip_sd=integ_flip_sd)
        integ_info = self.layout_info.get_diffamp_info(integ_params, flip_sd=integ_flip_sd)

        alat_col = alat_params.pop('col_idx')
        alat_flip_sd = alat_params.pop('flip_sd', False)
        _, alat_ports = self.draw_diffamp(alat_col, alat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                          diff_space=diff_space, gate_locs=gate_locs, flip_sd=alat_flip_sd)
        alat_info = self.layout_info.get_diffamp_info(alat_params, flip_sd=alat_flip_sd)

        dlat_info_list = []
        for idx, dlat_params in enumerate(dlat_params_list):
            dlat_params = dlat_params.copy()
            cur_col = dlat_params.pop('col_idx')
            dlat_flip_sd = dlat_params.pop('flip_sd', False)
            _, dlat_ports = self.draw_diffamp(cur_col, dlat_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs, flip_sd=dlat_flip_sd)
            dlat_info = self.layout_info.get_diffamp_info(dlat_params, flip_sd=dlat_flip_sd)

            dlat_info_list.append((cur_col, dlat_ports, dlat_info))

        block_info = dict(
            integ=(integ_col, integ_info, integ_pmos_vm_tid),
            alat=(alat_col, alat_info),
            dlat=dlat_info_list,
        )

        return integ_ports, alat_ports, block_info

    def connect_sup_io(self, integ_ports, alat_ports, dlat_info_list, sig_widths, sig_spaces, show_pins):

        # get vdd/cascode bias pins from integ/alat
        vdd_list, casc_list = [], []
        vdd_list.extend(integ_ports['vddt'])
        vdd_list.extend(alat_ports['vddt'])
        if 'bias_casc' in alat_ports:
            casc_list.extend(alat_ports['bias_casc'])

        # export inout pins
        inout_list = ('inp', 'inn', 'outp', 'outn')
        for name in inout_list:
            self.add_pin('integ_%s' % name, integ_ports[name], show=show_pins)
            self.add_pin('alat0_%s' % name, alat_ports[name], show=show_pins)

        # connect digital latch input to middle xm layer tracks, so we have room for vdd/vss wires.
        xm_layer_id = self.layout_info.mconn_port_layer + 3
        dlat_in_xm_mid_tr = (self.grid.get_num_tracks(self.size, xm_layer_id) - 1) / 2

        dlat_inputs = None
        for idx, (cur_col, dlat_ports, dlat_info) in enumerate(dlat_info_list):
            vdd_list.extend(dlat_ports['vddt'])
            if 'bias_casc' in dlat_ports:
                casc_list.extend(dlat_ports['bias_casc'])

            if (idx % 2 == 0) and idx > 0:
                # connect inputs to xm layer
                dlat_inp = dlat_ports['inp'][0]
                dlat_inn = dlat_ports['inn'][0]
                dlat_intv = cur_col, cur_col + dlat_info['fg_tot']
                dlat_inp, dlat_inn = connect_to_xm(self, dlat_inp, dlat_inn, dlat_intv, self.layout_info,
                                                   sig_widths, sig_spaces, dlat_in_xm_mid_tr)
                self.add_pin('dlat%d_inp' % idx, dlat_inp, show=show_pins)
                self.add_pin('dlat%d_inn' % idx, dlat_inn, show=show_pins)
                dlat_inputs = [dlat_inp, dlat_inn]
            else:
                self.add_pin('dlat%d_inp' % idx, dlat_ports['inp'], show=show_pins)
                self.add_pin('dlat%d_inn' % idx, dlat_ports['inn'], show=show_pins)

            self.add_pin('dlat%d_outp' % idx, dlat_ports['outp'], show=show_pins)
            self.add_pin('dlat%d_outn' % idx, dlat_ports['outn'], show=show_pins)

        # connect and export supplies
        vdd_name = self.get_pin_name('VDD')
        vdd_warrs = self.connect_wires(vdd_list, unit_mode=True)
        vdd_warrs.extend(self.connect_wires(casc_list, unit_mode=True))

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy(vdd_warrs=vdd_warrs, sup_margin=1, unit_mode=True)
        for warr in ptap_wire_arrs:
            self.add_pin(self.get_pin_name('VSS'), warr, show=show_pins)

        for warr in ntap_wire_arrs:
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
        clk_width_vm, clk_width_xm = clk_widths[:2]
        sig_space_vm = sig_spaces[0]
        clk_space_vm, clk_space_xm = clk_spaces[:2]
        sig_clk_space_vm, sig_clk_space_xm = sig_clk_spaces

        # calculate bias track indices
        dlat_top_tr = dlat_inputs[0].track_id.base_index
        dlat_bot_tr = dlat_inputs[1].track_id.base_index
        clkp_nmos_ana_tr_xm = dlat_bot_tr - (sig_width_xm + clk_width_xm) / 2 - sig_clk_space_xm
        clkp_nmos_dig_tr_xm = clkp_nmos_ana_tr_xm
        clkn_nmos_sw_tr_xm = clkp_nmos_ana_tr_xm - clk_width_xm - clk_space_xm
        clkn_pmos_dig_tr_xm = dlat_top_tr + (sig_width_xm + clk_width_xm) / 2 + sig_clk_space_xm
        clkn_pmos_ana_tr_xm = clkn_pmos_dig_tr_xm + clk_width_xm + clk_space_xm
        clkp_pmos_dig_tr_xm = clkn_pmos_ana_tr_xm
        self.in_xm_track = clkn_nmos_sw_tr_xm

        # mirror to opposite side
        bot_xm_idx = self.grid.find_next_track(xm_layer, self.array_box.bottom_unit, mode=1, unit_mode=True)
        clkn_nmos_dig_tr_xm = 2 * bot_xm_idx - clkp_nmos_dig_tr_xm - 1
        clkp_nmos_sw_tr_xm = 2 * bot_xm_idx - clkn_nmos_sw_tr_xm - 1

        clkp_nmos_sw_tr_id = TrackID(xm_layer, clkp_nmos_sw_tr_xm, width=clk_width_xm)
        clkn_nmos_sw_tr_id = TrackID(xm_layer, clkn_nmos_sw_tr_xm, width=clk_width_xm)
        clkp_nmos_sw_list, clkn_nmos_sw_list = [], []
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
        integ_col, integ_info, integ_pmos_vm_tid = block_info['integ']
        col_intv = integ_col, integ_col + integ_info['fg_tot']
        ltr_vm, rtr_vm = get_bias_tracks(layout_info, vm_layer, col_intv, sig_width_vm, sig_space_vm,
                                         clk_width_vm, sig_clk_space_vm)
        mtr_vm = (ltr_vm + rtr_vm) / 2
        mtr_id = TrackID(vm_layer, mtr_vm, width=clk_width_vm)
        # integ_nmos, route to left of differential input wires
        inr_idx = mtr_vm - (sig_clk_space_vm + (sig_width_vm + clk_width_vm) / 2)
        inl_idx = inr_idx - (sig_width_vm + sig_space_vm)
        nint_idx = inl_idx - (sig_clk_space_vm + (sig_width_vm + clk_width_vm) / 2)
        warr = self.connect_to_tracks(integ_ports['bias_tail'], TrackID(vm_layer, nint_idx, width=clk_width_vm),
                                      min_len_mode=1)
        self.add_pin('ibias_nmos_integ', warr, show=show_pins)
        # pmos_integ.  export on M5
        pmos_integ_tid = TrackID(vm_layer, integ_pmos_vm_tid, width=clk_width_vm)
        warr = self.connect_to_tracks(integ_ports['bias_load'], pmos_integ_tid)
        self.add_pin(clkp + '_pmos_integ', warr, show=show_pins)
        # nmos_switch
        warr = self.connect_to_tracks(integ_ports['sw'], mtr_id)
        clkn_nmos_sw_list.append(warr)

        # connect alat biases
        alat_col, alat_info = block_info['alat']
        col_intv = alat_col, alat_col + alat_info['fg_tot']
        right_sig_vm = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm, space=sig_space_vm)
        right_sig_vm += 3 * (sig_width_vm + sig_space_vm)
        ltr_vm = right_sig_vm + sig_clk_space_vm + (sig_width_vm + clk_width_vm) / 2
        rtr_vm = ltr_vm + clk_space_vm + clk_width_vm
        ltr_id = TrackID(vm_layer, ltr_vm, width=clk_width_vm)
        rtr_id = TrackID(vm_layer, rtr_vm, width=clk_width_vm)
        # nmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_tail'], ltr_id)
        xtr_id = TrackID(xm_layer, clkp_nmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, min_len_mode=0)
        self.add_pin(clkp + '_nmos_analog', warr, show=show_pins)

        # pmos_analog
        warr = self.connect_to_tracks(alat_ports['bias_load'], ltr_id)
        xtr_id = TrackID(xm_layer, clkn_pmos_ana_tr_xm, width=clk_width_xm)
        warr = self.connect_to_tracks(warr, xtr_id, min_len_mode=0)
        self.add_pin(clkn + '_pmos_analog', warr, show=show_pins)
        # nmos_switch
        warr = self.connect_to_tracks(alat_ports['sw'], rtr_id)
        clkn_nmos_sw_list.append(warr)

        # connect dlat
        for dfe_idx, (dlat_col, dlat_ports, dlat_info) in enumerate(block_info['dlat']):
            col_intv = dlat_col, dlat_col + dlat_info['fg_tot']

            if dfe_idx % 2 == 0 and dfe_idx > 0:
                tr_idx0 = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=clk_width_vm,
                                                        space=clk_space_vm)
                if datapath_parity == 0:
                    ntr_vm = tr_idx0 + (clk_width_vm + clk_space_vm) * 3
                    str_vm = tr_idx0 + (clk_width_vm + clk_space_vm)
                else:
                    ntr_vm = tr_idx0
                    str_vm = tr_idx0 + (clk_width_vm + clk_space_vm) * 2
                ptr_vm = tr_idx0
            elif dfe_idx == 0:
                left_sig_vm = layout_info.get_center_tracks(vm_layer, 4, tap1_col_intv, width=sig_width_vm,
                                                            space=sig_space_vm)
                tr_idx3 = left_sig_vm - (sig_width_vm + clk_width_vm) / 2 - sig_clk_space_vm
                if datapath_parity == 0:
                    ntr_vm = tr_idx3 - 1 * (clk_width_vm + clk_space_vm)
                    str_vm = tr_idx3 - 3 * (clk_width_vm + clk_space_vm)
                else:
                    ntr_vm = tr_idx3 - 2 * (clk_width_vm + clk_space_vm)
                    str_vm = tr_idx3
                ptr_vm = tr_idx3
            else:
                left_sig_vm = layout_info.get_center_tracks(vm_layer, 4, col_intv, width=sig_width_vm,
                                                            space=sig_space_vm)
                right_sig_vm = left_sig_vm + 3 * (sig_width_vm + sig_space_vm)
                ntr_vm = left_sig_vm - (sig_width_vm + clk_width_vm) / 2 - sig_clk_space_vm
                str_vm = right_sig_vm + (sig_width_vm + clk_width_vm) / 2 + sig_clk_space_vm
                ptr_vm = ntr_vm

            str_id = TrackID(vm_layer, str_vm, width=clk_width_vm)
            ntr_id = TrackID(vm_layer, ntr_vm, width=clk_width_vm)
            ptr_id = TrackID(vm_layer, ptr_vm, width=clk_width_vm)
            nwarr = self.connect_to_tracks(dlat_ports['bias_tail'], ntr_id)
            pwarr = self.connect_to_tracks(dlat_ports['bias_load'], ptr_id)
            swarr = self.connect_to_tracks(dlat_ports['sw'], str_id)

            if dfe_idx % 2 == 1:
                clkp_nmos_dig_list.append(nwarr)
                clkn_pmos_dig_list.append(pwarr)
                clkn_nmos_sw_list.append(swarr)
            else:
                clkn_nmos_dig_list.append(nwarr)
                clkp_pmos_dig_list.append(pwarr)
                clkp_nmos_sw_list.append(swarr)

        warr = self.connect_to_tracks(clkp_nmos_sw_list, clkp_nmos_sw_tr_id)
        self.add_pin(clkp + '_nmos_switch', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_nmos_sw_list, clkn_nmos_sw_tr_id)
        self.add_pin(clkn + '_nmos_switch', warr, show=show_pins)
        warr = self.connect_to_tracks(clkp_nmos_dig_list, clkp_nmos_dig_tr_id)
        self.add_pin(clkp + '_nmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_nmos_dig_list, clkn_nmos_dig_tr_id)
        self.add_pin(clkn + '_nmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkp_pmos_dig_list, clkp_pmos_dig_tr_id)
        self.add_pin(clkp + '_pmos_digital', warr, show=show_pins)
        warr = self.connect_to_tracks(clkn_pmos_dig_list, clkn_pmos_dig_tr_id)
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
            clk_widths=[1, 1, 1],
            clk_spaces=[1, 1, 1],
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
        self.in_xm_track = None
        self.sch_params = None

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
            clk_widths=[1, 1, 1],
            clk_spaces=[1, 1, 1],
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
            buf_params='integrator clock buffer parameters.',
            min_fg_sep='Minimum separation between transistors.',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            nac_off='number of off transistors for dlev AC coupling',
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
        self.draw_xm_supplies(bot_inst, top_inst)
        self._col_idx_dict = col_idx_dict

    def draw_xm_supplies(self, bot_inst, top_inst):
        show_pins = self.params['show_pins']
        # draw xm VDD wire
        bot_vdd = bot_inst.get_all_port_pins('VDD')[0]
        top_vdd = top_inst.get_all_port_pins('VDD')[0]
        hm_layer = bot_vdd.layer_id
        lower = self.grid.get_wire_bounds(hm_layer, bot_vdd.track_id.base_index, bot_vdd.track_id.width,
                                          unit_mode=True)[0]
        upper = self.grid.get_wire_bounds(hm_layer, top_vdd.track_id.base_index, top_vdd.track_id.width,
                                          unit_mode=True)[1]
        xm_layer = hm_layer + 2
        xtr_bot = self.grid.find_next_track(xm_layer, lower, half_track=False, mode=1, unit_mode=True)
        xtr_top = self.grid.find_next_track(xm_layer, upper, half_track=False, mode=-1, unit_mode=True)
        xnum_tr = xtr_top - xtr_bot + 1
        xmid_tr = (xtr_top + xtr_bot) / 2
        bot_vdd_box = bot_vdd.get_bbox_array(self.grid).base
        xm_lower = bot_vdd_box.left_unit
        xm_upper = bot_vdd_box.right_unit
        warr = self.add_wires(xm_layer, xmid_tr, lower=xm_lower, upper=xm_upper, width=xnum_tr, unit_mode=True)
        self.add_pin('VDDX', warr, show=show_pins)
        # draw xm bottom VSS wire
        bot_vss = bot_inst.get_all_port_pins('VSS')[0]
        top_vss = top_inst.get_all_port_pins('VSS')[0]
        upper = self.grid.get_wire_bounds(hm_layer, bot_vss.track_id.base_index, bot_vss.track_id.width,
                                          unit_mode=True)[1]
        xtr_top = self.grid.find_next_track(xm_layer, upper, half_track=False, mode=-1, unit_mode=True)
        xmid_tr = xtr_top - (xnum_tr - 1) / 2
        warr = self.add_wires(xm_layer, xmid_tr, lower=xm_lower, upper=xm_upper, width=xnum_tr, unit_mode=True)
        self.add_pin('VSSX', warr, show=show_pins)
        # draw xm top VSS wire
        lower = self.grid.get_wire_bounds(hm_layer, top_vss.track_id.base_index, top_vss.track_id.width,
                                          unit_mode=True)[0]
        xtr_bot = self.grid.find_next_track(xm_layer, lower, half_track=False, mode=1, unit_mode=True)
        xtr_top = self.grid.get_num_tracks(self.size, xm_layer) - 1
        xnum_tr = xtr_top - xtr_bot + 1
        xmid_tr = (xtr_top + xtr_bot) / 2
        warr = self.add_wires(xm_layer, xmid_tr, lower=xm_lower, upper=xm_upper, width=xnum_tr, unit_mode=True)
        self.add_pin('VSSX', warr, show=show_pins)

    def place(self, layout_info):
        # type: (SerdesRXBaseInfo) -> Tuple[Instance, Instance, Dict[str, Any]]
        alat_params_list = self.params['alat_params_list']
        buf_params = self.params['buf_params']
        integ_params = self.params['integ_params']
        intsum_params = self.params['intsum_params']
        summer_params = self.params['summer_params']
        dlat_params_list = self.params['dlat_params_list']
        nduml = self.params['nduml']
        ndumr = self.params['ndumr']
        nac_off = self.params['nac_off']
        sig_width_vm = self.params['sig_widths'][0]
        sig_space_vm = self.params['sig_spaces'][0]
        clk_width_vm = self.params['clk_widths'][0]
        clk_space_vm = self.params['clk_spaces'][0]
        sig_clk_space_vm = self.params['sig_clk_spaces'][0]
        # create AnalogBaseInfo object
        vm_layer_id = layout_info.mconn_port_layer + 2
        dtr_pitch = sig_width_vm + sig_space_vm
        diff_clk_route_tracks = 2 * sig_width_vm + 2 * clk_width_vm + sig_space_vm + 3 * sig_clk_space_vm

        # compute block locations
        col_idx_dict = {}
        # step -1: place clock buffers
        cur_col = nduml
        new_buf_params = buf_params.copy()
        new_buf_params['col_idx0'] = cur_col
        fg_clk_route = layout_info.num_tracks_to_fingers(vm_layer_id, clk_width_vm + clk_space_vm, cur_col)
        fg0 = max(fg_clk_route, buf_params['fg0'])
        cur_col += fg0 + fg_clk_route
        new_buf_params['col_idx1'] = cur_col
        # find minimum number of integrator frontend fingers and integrator frontend pmos vm track index.
        fg1 = max(fg_clk_route, buf_params['fg1'])
        integ_pmos_vm_tid = layout_info.get_center_tracks(vm_layer_id, 1, (cur_col, cur_col + fg1),
                                                          width=clk_width_vm, space=clk_space_vm)
        integ_fg_min = cur_col + fg1 - nduml
        # step 0: place integrating frontend.
        cur_col = nduml
        integ_col_idx = cur_col
        # print('integ_col: %d' % cur_col)
        # step 0A: find minimum number of fingers
        new_integ_params = integ_params.copy()
        new_integ_params['integ_pmos_vm_tid'] = integ_pmos_vm_tid
        integ_fg_min = max(layout_info.num_tracks_to_fingers(vm_layer_id, diff_clk_route_tracks, cur_col),
                           integ_fg_min)
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
        # step 2B: place precursor tap.  must fit one differential track + 2 clk route track
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
        # step 2B + 0.5: place offset cancellation.   Must fit 3 bias tracks
        intsum_off_col_idx = cur_col
        intsum_off_fg_params = intsum_gm_fg_list[2].copy()
        intsum_off_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, diff_clk_route_tracks, cur_col)
        intsum_off_fg_params['min'] = intsum_off_fg_min
        intsum_off_info = layout_info.get_gm_info(intsum_off_fg_params)
        new_intsum_gm_fg_list.append(intsum_off_fg_params)
        cur_col += intsum_off_info['fg_tot']
        col_idx_dict['intsum'].append((intsum_off_col_idx, cur_col))
        cur_col += intsum_gm_sep_list[2]
        # print('cur_col: %d' % cur_col)
        # step 2C: place intsum DFE taps
        num_intsum_gm = len(intsum_gm_fg_list)
        num_dfe = num_intsum_gm - 3 + 1
        new_dlat_params_list = [None] * len(dlat_params_list)
        # NOTE: here DFE index start at 1.
        for idx in range(3, num_intsum_gm):
            # print('intsum_idx%d_col: %d' % (idx, cur_col))
            intsum_dfe_fg_params = intsum_gm_fg_list[idx].copy()
            dfe_idx = num_dfe - (idx - 3)
            dlat_idx = dfe_idx - 2
            dig_latch_params = dlat_params_list[dlat_idx].copy()
            in_route = False
            num_route_tracks = diff_clk_route_tracks
            if dfe_idx > 2:
                # for DFE tap > 2, the intsum Gm stage must align with the corresponding
                # digital latch.
                # set digital latch column index
                if dfe_idx % 2 == 1:
                    # for odd DFE taps, we have criss-cross signal connections, so fit 2 diff tracks + 2 clk tracks
                    num_route_tracks = 4 * sig_width_vm + 2 * clk_width_vm + 3 * sig_space_vm + 3 * sig_clk_space_vm
                    # for odd DFE taps > 3, we need to reserve additional input routing tracks
                    in_route = dfe_idx > 3
                else:
                    # for even DFE taps, we have criss-cross bias connections, so fit 4 clk tracks
                    num_route_tracks = max(4 * clk_width_vm + 4 * clk_space_vm, num_route_tracks)

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
                col_idx_dict['dlat'][dlat_idx] = (cur_col, cur_col + num_fg)
                col_idx_dict['intsum'].append((cur_col, cur_col + num_fg))
                cur_col += num_fg
                # print('cur_col: %d' % cur_col)
                if in_route:
                    # allocate input route
                    route_fg_min = layout_info.num_tracks_to_fingers(vm_layer_id, 2 * dtr_pitch, cur_col)
                    intsum_gm_sep_list[idx] = route_fg_min
                    col_idx_dict['dlat%d_inroute' % dlat_idx] = (cur_col, cur_col + route_fg_min)
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
            new_dlat_params_list[dlat_idx] = dig_latch_params
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
        dig_latch_params['min'] = dlat_info['fg_tot']
        col_idx_dict['dlat'][0] = (cur_col - dlat_info['fg_tot'], cur_col)

        # step 4: add AC coupling off transistors
        num_fg = layout_info.num_tracks_to_fingers(vm_layer_id, clk_width_vm * 2 + clk_space_vm * 3, cur_col)
        num_fg = max(num_fg, 2 * nac_off + layout_info.min_fg_sep * 3)
        col_idx_dict['acoff'] = (cur_col, cur_col + num_fg)
        cur_col += num_fg

        fg_tot = cur_col + ndumr
        # add dummies until we have multiples of block pitch
        blk_w = self.grid.get_block_size(layout_info.mconn_port_layer + 3, unit_mode=True)[0]
        sd_pitch_unit = layout_info.sd_pitch_unit
        cur_width = layout_info.get_total_width(fg_tot)
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
        self.in_xm_track = bot_inst.master.in_xm_track

        # make RXHalfTop
        top_params = {key: self.params[key] for key in RXHalfBottom.get_params_info().keys()
                      if key in self.params}
        top_params['fg_tot'] = fg_tot
        top_params['alat_params'] = alat2_params
        top_params['buf_params'] = new_buf_params
        top_params['intsum_params'] = dict(
            col_idx=intsum_col_idx,
            fg_load=intsum_params['fg_load'],
            gm_fg_list=new_intsum_gm_fg_list,
            gm_sep_list=intsum_gm_sep_list,
            sgn_list=intsum_params['sgn_list'],
            flip_sd_list=intsum_params.get('flip_sd_list', None),
            decap_list=intsum_params.get('decap_list', None),
            load_decap_list=intsum_params.get('load_decap_list', None),
        )
        top_params['summer_params'] = dict(
            col_idx=summer_col_idx,
            fg_load=summer_params['fg_load'],
            gm_fg_list=new_summer_gm_fg_list,
            gm_sep_list=summer_gm_sep_list,
            sgn_list=summer_params['sgn_list'],
        )
        top_params['acoff_params'] = dict(
            col_intv=col_idx_dict['acoff'],
            nac_off=nac_off,
        )
        top_params['show_pins'] = False
        # print('top summer col: %d' % top_params['summer_params']['col_idx'])
        top_master = self.new_template(params=top_params, temp_cls=RXHalfTop)
        top_inst = self.add_instance(top_master, orient='MX')
        top_inst.move_by(dy=bot_inst.array_box.top - top_inst.array_box.bottom)
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.set_size_from_array_box(top_master.size[0])

        show_pins = self.params['show_pins']
        for inst in (bot_inst, top_inst):
            for port_name in inst.port_names_iter():
                if port_name.endswith('nmos_integ'):
                    # extend nmos_integ bias to edge
                    warr = inst.get_all_port_pins(port_name)[0]
                    warr = self.extend_wires(warr, upper=self.array_box.top)
                    self.add_pin(port_name, warr, show=show_pins)
                elif not (port_name.endswith('pmos_integ') or port_name.endswith('pmos_intsum')):
                    self.reexport(inst.get_port(port_name), show=show_pins)

        # record parameters for schematic
        w_dict = self.params['w_dict'].copy()
        mos_types = list(w_dict.keys())
        sch_integ_params = dict(
            fg_dict={key: new_integ_params[key] for key in mos_types if key in new_integ_params},
            fg_tot=integ_fg_tot,
            flip_sd=new_integ_params.get('flip_sd', False),
            decap=new_integ_params.get('decap', False),
        )
        if 'ref' in new_integ_params:
            sch_integ_params['fg_dict']['ref'] = new_integ_params['ref']
        sch_alat_list = [
            dict(fg_dict={key: alat1_params[key] for key in mos_types if key in alat1_params},
                 fg_tot=alat_fg_min,
                 flip_sd=alat1_params.get('flip_sd', False),
                 decap=alat1_params.get('decap', False),
                 ),
            dict(fg_dict={key: alat2_params[key] for key in mos_types if key in alat2_params},
                 fg_tot=alat_fg_min,
                 flip_sd=alat2_params.get('flip_sd', False),
                 decap=alat2_params.get('decap', False),
                 ),
        ]
        sch_dlat_list = []
        for dlat_params in new_dlat_params_list:
            sch_dlat_list.append(dict(
                fg_dict={key: dlat_params[key] for key in mos_types if key in dlat_params},
                fg_tot=dlat_params['min'],
                flip_sd=dlat_params.get('flip_sd', False),
                decap=dlat_params.get('decap', False),
            ))

        self.sch_params = dict(
            lch=self.params['lch'],
            w_dict=w_dict,
            th_dict=self.params['th_dict'].copy(),
            nac_off=self.params['nac_off'],
            integ_params=sch_integ_params,
            alat_params_list=sch_alat_list,
            intsum_params=top_master.sch_intsum_params.copy(),
            summer_params=top_master.sch_summer_params.copy(),
            dlat_params_list=sch_dlat_list,
            buf_params={key: buf_params[key] for key in ['nmos_type', 'fg0', 'fg1']},
            fg_tot=fg_tot,
        )

        return bot_inst, top_inst, col_idx_dict

    def connect(self, layout_info, bot_inst, top_inst, col_idx_dict):
        show_pins = self.params['show_pins']
        vm_space = self.params['sig_spaces'][0]
        hm_layer = layout_info.mconn_port_layer + 1
        vm_layer = hm_layer + 1
        vm_width = self.params['sig_widths'][0]
        nintsum = len(self.params['intsum_params']['gm_fg_list'])

        # connect clkp of integrators
        clk_prefix = 'clkp' if self.params['datapath_parity'] == 0 else 'clkn'
        clkpb = bot_inst.get_all_port_pins(clk_prefix + '_pmos_integ')
        clkpt = top_inst.get_all_port_pins(clk_prefix + '_pmos_intsum')
        warr = self.connect_to_tracks(clkpb, clkpt[0].track_id)
        warr = self.connect_wires([warr] + clkpt)

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
        if nintsum > 4:
            p_list.append(bot_inst.get_port('dlat1_inp').get_pins()[0])
            n_list.append(bot_inst.get_port('dlat1_inn').get_pins()[0])

        self.connect_differential_tracks(p_list, n_list, vm_layer, ptr_idx, ntr_idx, width=vm_width)

        # connect even DFE taps
        ndfe = nintsum - 3 + 1
        for dfe_idx in range(4, ndfe + 1, 2):
            dlat_idx = dfe_idx - 2
            intsum_idx = nintsum - 1 - dlat_idx
            route_col_intv = col_idx_dict['intsum'][intsum_idx]
            ptr_idx = layout_info.get_center_tracks(vm_layer, 2, route_col_intv, width=vm_width, space=vm_space)
            ntr_idx = ptr_idx + vm_space + vm_width
            p_list = [bot_inst.get_port('dlat%d_outp' % (dfe_idx - 2)).get_pins()[0],
                      top_inst.get_port('intsum_inp<%d>' % intsum_idx).get_pins()[0], ]
            n_list = [bot_inst.get_port('dlat%d_outn' % (dfe_idx - 2)).get_pins()[0],
                      top_inst.get_port('intsum_inn<%d>' % intsum_idx).get_pins()[0], ]
            self.connect_differential_tracks(p_list, n_list, vm_layer, ptr_idx, ntr_idx, width=vm_width)
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
                                         width=vm_width)


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
        self._in_xm_offset = None
        self._sch_params = None

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
            clk_widths=[1, 1, 1],
            clk_spaces=[1, 1, 1],
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
            buf_params='Integrator clock buffer parameters.',
            nac_off='Number of off transistors for dlev AC coupling',
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

    @property
    def num_fingers(self):
        # type: () -> int
        return self._fg_tot

    @property
    def in_offset(self):
        # type: () -> int
        return self._in_xm_offset

    @property
    def sch_params(self):
        # type: () -> Dict[str, Any]
        return self._sch_params

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

        self._sch_params = odd_master.sch_params

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
        self.connect_bias(inst_list)
        self.connect_supplies(inst_list)

    def connect_signal(self, inst_list, col_idx_dict, layout_info):
        hm_layer_id = layout_info.mconn_port_layer + 1
        vm_layer_id = hm_layer_id + 1
        xm_layer_id = vm_layer_id + 1
        vm_space = self.params['sig_spaces'][0]
        vm_width = self.params['sig_widths'][0]
        vm_pitch = vm_width + vm_space
        show_pins = self.params['show_pins']

        # connect inputs of even and odd paths
        route_col_intv = col_idx_dict['integ']
        ptr_idx = layout_info.get_center_tracks(vm_layer_id, 2, route_col_intv, width=vm_width, space=vm_space)
        vm_sig_clk_space = self.params['sig_clk_spaces'][0]
        vm_clk_width = self.params['clk_widths'][0]
        ptr_idx -= vm_sig_clk_space + (vm_space + vm_clk_width) / 2 + vm_width
        ports = ['integ_in{}']
        inp, inn = self._connect_differential(inst_list, ptr_idx, vm_layer_id, vm_width, vm_space,
                                              ports, ports)
        in_xm_track = inst_list[0].master.in_xm_track
        inp_track = inst_list[0].translate_master_track(xm_layer_id, in_xm_track)
        inn_track = inst_list[1].translate_master_track(xm_layer_id, in_xm_track)
        self._in_xm_offset = (inp_track - inn_track) / 2

        inp, inn = self.connect_differential_tracks(inp, inn, xm_layer_id, inp_track, inn_track,
                                                    width=self.params['sig_widths'][1])
        # export inputs/outputs
        self.add_pin('inp', inp, show=show_pins)
        self.add_pin('inn', inn, show=show_pins)
        for idx, prefix in ((0, 'even'), (1, 'odd')):
            for pname, oname in (('summer', 'summer'), ('dlev', 'dlev'), ('integ', 'intamp'), ('intsum', 'intsum')):
                self.reexport(inst_list[idx].get_port('%s_outp' % pname),
                              net_name='outp_%s<%d>' % (oname, idx), show=show_pins)
                self.reexport(inst_list[idx].get_port('%s_outn' % pname),
                              net_name='outn_%s<%d>' % (oname, idx), show=show_pins)
            for pname, num in (('alat', 2), ('dlat', 3)):
                for pidx in range(num):
                    pport = inst_list[idx].get_port('%s%d_outp' % (pname, pidx))
                    nport = inst_list[idx].get_port('%s%d_outn' % (pname, pidx))
                    self.reexport(pport, net_name='%s_outp_%s<%d>' % (prefix, pname, pidx), show=show_pins)
                    self.reexport(nport, net_name='%s_outn_%s<%d>' % (prefix, pname, pidx), show=show_pins)

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
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width)
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
                           inst_list[1].get_port('intsum_inp<4>').get_pins()[0],
                           ],
                          [inst_list[0].get_port('dlat1_outn').get_pins()[0],
                           inst_list[0].get_port('dlat2_inn').get_pins()[0],
                           inst_list[1].get_port('intsum_inn<4>').get_pins()[0],
                           ], ]
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width)
        tr_idx_list[0] += 2 * vm_pitch
        tr_idx_list[1] += 2 * vm_pitch
        warr_list_list = [[inst_list[1].get_port('dlat1_outp').get_pins()[0],
                           inst_list[1].get_port('dlat2_inp').get_pins()[0],
                           inst_list[0].get_port('intsum_inp<4>').get_pins()[0],
                           ],
                          [inst_list[1].get_port('dlat1_outn').get_pins()[0],
                           inst_list[1].get_port('dlat2_inn').get_pins()[0],
                           inst_list[0].get_port('intsum_inn<4>').get_pins()[0],
                           ], ]
        self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width)

    def connect_bias(self, inst_list):
        show_pins = self.params['show_pins']
        clk_top_list = [('clk1', 1),
                        ('nmos_analog', 2),
                        ('pmos_analog', 2),
                        ('clk2', 2),
                        ('pmos_digital', 2),
                        ('nmos_digital', 1),
                        ('pmos_summer', 2),
                        ('nmos_summer', 1),
                        ('nmos_tap1', 1),
                        ]

        clk_ports = {'clk1': ['nmos_switch_alat1'], 'clk2': ['nmos_switch']}
        clk_wires = {}
        for inst in inst_list:
            for name in inst.port_names_iter():
                if name.startswith('clk'):
                    if name not in clk_wires:
                        clk_wires[name] = []
                    clk_wires[name].extend(inst.get_all_port_pins(name))

        for base_name, _ in clk_top_list:
            if base_name.startswith('clk'):
                pwires = []
                nwires = []
                for clk_port_name in clk_ports[base_name]:
                    pwires.extend(clk_wires.pop('clkp_' + clk_port_name))
                    nwires.extend(clk_wires.pop('clkn_' + clk_port_name))
                pname = 'clkp'
                nname = 'clkn'
                labelp = 'clkp:'
                labeln = 'clkn:'
            else:
                pname = 'clkp_' + base_name
                nname = 'clkn_' + base_name
                pwires = self.connect_wires(clk_wires.pop(pname))
                nwires = self.connect_wires(clk_wires.pop(nname))
                if base_name == 'nmos_tap1':
                    pname = 'even_' + pname
                    nname = 'odd_' + nname
                labelp = pname + ':'
                labeln = nname + ':'

            self.add_pin(pname, pwires, label=labelp, show=show_pins)
            self.add_pin(nname, nwires, label=labeln, show=show_pins)

        for name, wires in clk_wires.items():
            self.add_pin(name, wires, show=show_pins)

        num_dfe = len(self.params['intsum_params']['gm_fg_list']) - 3 + 1
        for inst, prefix in zip(inst_list, ('even_', 'odd_')):
            for pname in ('bias_ffe', 'bias_dlevp', 'bias_dlevn',
                          'ibias_nmos_integ', 'ibias_nmos_intsum', 'ibias_offset',
                          'offp', 'offn'):
                if inst.has_port(pname):
                    self.reexport(inst.get_port(pname), net_name=prefix + pname, show=show_pins)
            for idx in range(num_dfe):
                if idx == 0:
                    self.reexport(inst.get_port('en_dfe1'), net_name=prefix + 'en_dfe1', show=show_pins)
                else:
                    pname = 'ibias_dfe<%d>' % idx
                    self.reexport(inst.get_port(pname), net_name=prefix + pname, show=show_pins)

    def connect_supplies(self, inst_list):
        show_pins = self.params['show_pins']

        vdd_warrs, vss_warrs = [], []
        vddx_warrs, vssx_warrs = [], []
        for inst in inst_list:
            vdd_warrs.extend(inst.get_all_port_pins('VDD'))
            vss_warrs.extend(inst.get_all_port_pins('VSS'))
            vddx_warrs.extend(inst.get_all_port_pins('VDDX'))
            vssx_warrs.extend(inst.get_all_port_pins('VSSX'))

        vddx_warrs = self.connect_wires(vddx_warrs)
        vssx_warrs = self.connect_wires(vssx_warrs)

        # connect X layer supplies to lower supplies
        hm_layer = vdd_warrs[0].layer_id
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        hw = vdd_warrs[0].track_id.width
        vidx_list = list(range(self.grid.get_num_tracks(self.size, vm_layer)))
        margin = int(round(0.5 / self.grid.resolution))
        for xwires, hwires in zip([vddx_warrs, vssx_warrs], [vdd_warrs, vss_warrs]):
            for xwarr in xwires:
                xtid = xwarr.track_id
                xw = xtid.width
                for xidx in xtid:
                    # get all hm wires to connect to this one
                    xlower, xupper = self.grid.get_wire_bounds(xm_layer, xidx, width=xw, unit_mode=True)
                    hidx_list = []
                    for hwarr in hwires:
                        for hidx in hwarr.track_id:
                            hlower, hupper = self.grid.get_wire_bounds(hm_layer, hidx, width=hw, unit_mode=True)
                            if hlower >= xlower and hupper <= xupper:
                                hidx_list.append(hidx)
                    # get all available vm wires to use as vias
                    avail_list = self.get_available_tracks(vm_layer, vidx_list, xlower, xupper,
                                                           width=1, margin=margin, unit_mode=True)
                    # connect
                    vm_warr_list = []
                    for vidx in avail_list:
                        vm_warr_list.append(self.add_wires(vm_layer, vidx, xlower, xupper, unit_mode=True))
                    self.connect_to_tracks(vm_warr_list, TrackID(xm_layer, xidx, width=xw))
                    for hidx in hidx_list:
                        self.connect_to_tracks(vm_warr_list, TrackID(hm_layer, hidx, width=hw))

        self.add_pin('VDD', vddx_warrs, label='VDD:', show=show_pins)
        self.add_pin('VSS', vssx_warrs, label='VSS:', show=show_pins)

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

        trp, trn = self.connect_matching_tracks(warr_list_list, vm_layer_id, tr_idx_list, width=vm_width)
        return trp, trn
