# -*- coding: utf-8 -*-

from bag.layout.routing import TrackManager

from abs_templates_ec.analog_core.base import AnalogBase


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
        wtr_in = tr_manager.get_width(hm_layer, 'in')
        sp_io = tr_manager.get_space(hm_layer, ('in', 'out'))
        wtr_out = tr_manager.get_width(hm_layer, 'out')
        wtr_en = tr_manager.get_width(hm_layer, 'en')

        seg_p = seg_dict['p']
        seg_n = seg_dict['n']
        seg_single = max(seg_p, seg_n)
        seg_tot = seg_single + 2 * ndum

        if (seg_p - seg_n) % 4 != 0:
            raise ValueError('Now assume seg_p and seg_n differ by multiples of 4.')

        # draw transistor rows
        ng_tracks = [wtr_in + sp_io]
        pg_tracks = [wtr_out, wtr_en]
        pds_tracks = [0, 0]
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
