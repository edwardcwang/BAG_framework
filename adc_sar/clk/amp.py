# -*- coding: utf-8 -*-

from bag.layout.routing import TrackManager

from abs_templates_ec.analog_core.base import AnalogBase, AnalogBaseInfo


class InvAmp(AnalogBase):
    """A differential NMOS passgate track-and-hold circuit with clock driver.

    This template is mainly used for ADC purposes.

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
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            seg_dict='NMOS/PMOS number of segments dictionary.',
            ndum='Number of left/right dummy fingers.',
            tr_widths='signal wire width dictionary.',
            tr_spaces='signal wire space dictionary.',
            show_pins='True to create pin labels.',
            top_layer='the top level layer ID.',
        )

    @classmethod
    def get_default_param_values(cls):
        return dict(
            show_pins=False,
            top_layer=None,
        )

    def draw_layout(self):
        lch = self.params['lch']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        w_dict = self.params['w_dict']
        th_dict = self.params['th_dict']
        seg_dict = self.params['seg_dict']
        ndum = self.params['ndum']
        tr_widths = self.params['tr_widths']
        tr_spaces = self.params['tr_spaces']
        show_pins = self.params['show_pins']
        top_layer = self.params['top_layer']

        nw_list = [w_dict['n']]
        pw_list = [w_dict['p']] * 2
        nth_list = [th_dict['n']]
        pth_list = [th_dict['p']] * 2

        tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        hm_layer = self.mos_conn_layer + 1
        wtr_in = tr_manager.get_width(hm_layer, 'out')
        wtr_out = tr_manager.get_width(hm_layer, 'in')
        wtr_en = tr_manager.get_width(hm_layer, 'en')
        wtr_mid = tr_manager.get_width(hm_layer, 'mid')
        sp_io = tr_manager.get_space(hm_layer, ('in', 'out'))
        sp_mid = tr_manager.get_space(hm_layer, ('in', 'mid'))

        seg_p = seg_dict['p']
        seg_n = seg_dict['n']
        seg_single = max(seg_p, seg_n)
        seg_tot = seg_single + 2 * ndum

        if (seg_p - seg_n) % 4 != 0:
            raise ValueError('We assume seg_p and seg_n differ by multiples of 4.')

        # draw transistor rows
        ng_tracks = [wtr_in + sp_io]
        pg_tracks = [wtr_out, wtr_en]
        pds_tracks = [sp_mid + wtr_mid, 0]
        nds_tracks = [0]
        p_orient = ['R0', 'MX']
        n_orient = ['MX']
        self.draw_base(lch, seg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=n_orient, p_orientations=p_orient,
                       top_layer=top_layer,
                       )

        # draw transistors
        pidx = ndum + (seg_single - seg_p) // 2
        nidx = ndum + (seg_single - seg_n) // 2
        p_ports = self.draw_mos_conn('pch', 0, pidx, seg_p, 2, 0, s_net='', d_net='out')
        n_ports = self.draw_mos_conn('nch', 0, nidx, seg_n, 0, 2, s_net='', d_net='out')

        # connect supplies
        self.connect_to_substrate('ntap', p_ports['s'])
        self.connect_to_substrate('ptap', n_ports['s'])

        # connect input
        loc = tr_manager.align_wires(hm_layer, ['in'], ng_tracks[0], alignment=1)[0]
        tid = self.make_track_id('nch', 0, 'g', loc, width=wtr_in)
        warr = self.connect_to_tracks([p_ports['g'], n_ports['g']], tid)
        self.add_pin('in', warr, show=show_pins)
        # connect output
        loc = tr_manager.align_wires(hm_layer, ['out'], pg_tracks[0], alignment=1)[0]
        tid = self.make_track_id('pch', 0, 'g', loc, width=wtr_out)
        warr = self.connect_to_tracks([p_ports['d'], n_ports['d']], tid)
        self.add_pin('out', warr, show=show_pins)

        # draw dummies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        # export supplies
        self.add_pin('VSS', ptap_wire_arrs)
        self.add_pin('VDD', ntap_wire_arrs)

        # compute schematic parameters
        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict,
            th_dict=th_dict,
            seg_dict=seg_dict,
            dum_info=self.get_sch_dummy_info(),
        )


class NorAmp(AnalogBase):
    """A differential NMOS passgate track-and-hold circuit with clock driver.

    This template is mainly used for ADC purposes.

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
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            seg_dict='NMOS/PMOS number of segments dictionary.',
            ndum='Number of left/right dummy fingers.',
            tr_widths='signal wire width dictionary.',
            tr_spaces='signal wire space dictionary.',
            show_pins='True to create pin labels.',
            top_layer='the top level layer ID.',
        )

    @classmethod
    def get_default_param_values(cls):
        return dict(
            show_pins=False,
            top_layer=None,
        )

    def draw_layout(self):
        lch = self.params['lch']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        w_dict = self.params['w_dict']
        th_dict = self.params['th_dict']
        seg_dict = self.params['seg_dict']
        ndum = self.params['ndum']
        tr_widths = self.params['tr_widths']
        tr_spaces = self.params['tr_spaces']
        show_pins = self.params['show_pins']
        top_layer = self.params['top_layer']

        nw_list = [w_dict['n']]
        pw_list = [w_dict['p']] * 2
        nth_list = [th_dict['n']]
        pth_list = [th_dict['p']] * 2

        tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        hm_layer = self.mos_conn_layer + 1
        wtr_in = tr_manager.get_width(hm_layer, 'in')
        wtr_out = tr_manager.get_width(hm_layer, 'out')
        wtr_en = tr_manager.get_width(hm_layer, 'en')
        wtr_mid = tr_manager.get_width(hm_layer, 'mid')
        sp_io = tr_manager.get_space(hm_layer, ('in', 'out'))
        sp_mid = tr_manager.get_space(hm_layer, ('in', 'mid'))

        seg_invp = seg_dict['invp']
        seg_enp = seg_dict['enp']
        seg_invn = seg_dict['invn']
        seg_enn = seg_dict['enn']
        seg_single = max(seg_invp, seg_invn, seg_enp)

        if (seg_invp - seg_invn) % 4 != 0 or (seg_enp - seg_invp) % 4 != 0:
            raise ValueError('We assume seg_invp/seg_invn/seg_enp differ by multiples of 4.')

        # compute total number of fingers
        ana_info = AnalogBaseInfo(self.grid, lch, 0, top_layer=top_layer)
        if ana_info.abut_analog_mos:
            fg_sep = 0
        else:
            fg_sep = ana_info.min_fg_sep

        enp_idx = ndum + (seg_single - seg_enp) // 2
        invp_idx = ndum + (seg_single - seg_invp) // 2
        invn_idx = ndum + (seg_single - seg_invn) // 2
        enn_idx = invn_idx + seg_invn + fg_sep
        seg_tot = max(seg_single + 2 * ndum, enn_idx + seg_enn + ndum)

        # draw transistor rows
        ng_tracks = [wtr_out + sp_io]
        pg_tracks = [wtr_in, wtr_en]
        pds_tracks = [sp_mid + wtr_mid, 0]
        nds_tracks = [0]
        p_orient = ['R0', 'MX']
        n_orient = ['MX']
        self.draw_base(lch, seg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=n_orient, p_orientations=p_orient,
                       top_layer=top_layer,
                       )

        # draw transistors
        invn = self.draw_mos_conn('nch', 0, invn_idx, seg_invn, 0, 2, s_net='', d_net='out')
        enn = self.draw_mos_conn('nch', 0, enn_idx, seg_enn, 0, 2, s_net='', d_net='out')
        invp = self.draw_mos_conn('pch', 0, invp_idx, seg_invp, 2, 0, s_net='mid', d_net='out')
        enp = self.draw_mos_conn('pch', 1, enp_idx, seg_enp, 0, 2, s_net='mid', d_net='')

        # connect supplies
        self.connect_to_substrate('ntap', enp['d'])
        self.connect_to_substrate('ptap', invn['s'])
        self.connect_to_substrate('ptap', enn['s'])

        # connect output
        loc = tr_manager.align_wires(hm_layer, ['out'], ng_tracks[0], alignment=1)[0]
        tid = self.make_track_id('nch', 0, 'g', loc, width=wtr_out)
        warr = self.connect_to_tracks([invp['d'], invn['d'], enn['d']], tid)
        self.add_pin('out', warr, show=show_pins)
        # connect input
        loc = tr_manager.align_wires(hm_layer, ['in'], pg_tracks[0], alignment=1)[0]
        tid = self.make_track_id('pch', 0, 'g', loc, width=wtr_in)
        warr = self.connect_to_tracks([invp['g'], invn['g']], tid)
        self.add_pin('in', warr, show=show_pins)
        # connect mid
        loc = tr_manager.align_wires(hm_layer, ['mid'], pds_tracks[0], alignment=1)[0]
        tid = self.make_track_id('pch', 0, 'ds', loc, width=wtr_mid)
        self.connect_to_tracks([invp['s'], enp['s']], tid)
        # connect enable
        loc = tr_manager.align_wires(hm_layer, ['en'], pg_tracks[1], alignment=1)[0]
        tid = self.make_track_id('pch', 1, 'g', loc, width=wtr_en)
        warr = self.connect_to_tracks([enp['g'], enn['g']], tid)
        self.add_pin('en', warr, show=show_pins)

        # draw dummies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        # export supplies
        self.add_pin('VSS', ptap_wire_arrs)
        self.add_pin('VDD', ntap_wire_arrs)

        # compute schematic parameters
        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict,
            th_dict=th_dict,
            seg_dict=seg_dict,
            dum_info=self.get_sch_dummy_info(),
        )
