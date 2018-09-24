# -*- coding: utf-8 -*-

"""This module defines Frontend sampler layout.
"""

import bag
from bag.layout.util import BBox
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase

from ..analog_core import AnalogBase, AnalogBaseInfo


class NPassGateWClkCore(AnalogBase):
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
        super(NPassGateWClkCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._fg_tot = 0

    @property
    def fg_tot(self):
        return self._fg_tot

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
            wp='pmos width, in meters/number of fins.',
            wn='nmos width, in meters/number of fins.',
            fgn='passgate nmos number of fingers.',
            fg_inbuf_list='List of input clock buffer pmos/nmos number of fingers.',
            fg_outbuf_list='List of output clock buffer pmos/nmos number of fingers.',
            nduml='number of left dummies.',
            ndumr='number of right dummies.',
            nsep='number of fingers of separator dummies.',
            threshold='transistor threshold flavor.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            pgr_w='NMOS guard-ring substrate width, in meters/number of fins.',
            ngr_w='PMOS guard-ring substrate width, in meters/number of fins.',
            num_track_sep='number of tracks reserved as space between clock and signal wires.',
            io_width='input/output track width in number of tracks.',
            rename_dict='pin renaming dictionary.',
            guard_ring_nf='Guard ring width in number of fingers.  0 for no guard ring.',
            show_pins='True to draw pins.',
        )

    def draw_layout(self):
        self._draw_layout_helper(**self.params)

    def _draw_clk_buffer(self, idx, fgp, fgn, in_ports, out_ports):
        # figure out pmos/nmos index offset
        poff = noff = 0
        if fgp > fgn:
            noff = (fgp - fgn) // 2
        elif fgn > fgp:
            poff = (fgn - fgp) // 2

        p_ports = self.draw_mos_conn('pch', 0, idx + poff, fgp, 2, 0)
        n_ports = self.draw_mos_conn('nch', 1, idx + noff, fgn, 0, 2)

        out_ports.extend((p_ports['d'], n_ports['d']))
        in_ports.extend((p_ports['g'], n_ports['g']))
        self.connect_to_substrate('ptap', n_ports['s'])
        self.connect_to_substrate('ntap', p_ports['s'])
        return [n_ports['s']]

    def _draw_pass_gates(self, idx_list, fgn, inp_tr_idx, inn_tr_idx, out_tr_idx,
                         lower_track_width, io_width, clk_ports, show_pins=False):
        # draw transistors
        p_ports = self.draw_mos_conn('nch', 1, idx_list[0], fgn, 1, 1, gate_pref_loc='s')
        n_ports = self.draw_mos_conn('nch', 1, idx_list[1], fgn, 1, 1, gate_pref_loc='s')

        # draw gate connections to match parasitics better.
        p_dum = self.draw_mos_conn('nch', 0, idx_list[0], fgn, 1, 1, gate_pref_loc='s',
                                   is_ds_dummy=False)
        n_dum = self.draw_mos_conn('nch', 0, idx_list[1], fgn, 1, 1, gate_pref_loc='s',
                                   is_ds_dummy=False)

        # connect gates
        clk_ports.extend((p_ports['g'], n_ports['g']))
        vss_warr_list = [p_dum['g'], n_dum['g']]
        # connect inputs
        inp_list = []
        inn_list = []
        pdiff_idx = self.get_track_index('nch', 1, 'ds', inp_tr_idx)
        ndiff_idx = self.get_track_index('nch', 1, 'ds', inn_tr_idx)
        pdwarr, pnwarr = self.connect_differential_tracks(p_ports['s'], n_ports['s'],
                                                          self.mos_conn_layer + 1,
                                                          pdiff_idx, ndiff_idx,
                                                          width=lower_track_width,
                                                          )
        inp_list.append(pdwarr)
        inn_list.append(pnwarr)
        # swap positive and negative input track indices
        pdiff_idx = self.get_track_index('nch', 0, 'ds', inp_tr_idx)
        ndiff_idx = self.get_track_index('nch', 0, 'ds', inn_tr_idx)
        pdwarr, pnwarr = self.connect_differential_tracks(p_dum['s'], n_dum['s'],
                                                          self.mos_conn_layer + 1,
                                                          pdiff_idx, ndiff_idx,
                                                          width=lower_track_width,
                                                          )
        inn_list.append(pdwarr)
        inp_list.append(pnwarr)

        # connect outputs to horizontal layer
        tr_id = self.make_track_id('nch', 0, 'ds', out_tr_idx, width=lower_track_width)
        outp_track = self.connect_to_tracks([p_ports['d'], p_dum['d']], tr_id)
        outn_track = self.connect_to_tracks([n_ports['d'], n_dum['d']], tr_id)

        # connect inputs/outputs to vertical layer
        io_layer = tr_id.layer_id + 1
        p_tid = self.grid.coord_to_nearest_track(io_layer, outp_track.middle, half_track=True)
        n_tid = self.grid.coord_to_nearest_track(io_layer, outn_track.middle, half_track=True)
        outp = self.connect_to_tracks(outp_track, TrackID(io_layer, p_tid, width=io_width),
                                      track_lower=0.0)
        outn = self.connect_to_tracks(outn_track, TrackID(io_layer, n_tid, width=io_width),
                                      track_lower=0.0)
        ymid_out = outp.middle
        ckout_tid = (p_tid + n_tid) / 2
        self.add_pin(self.get_pin_name('outp'), outp, show=show_pins)
        self.add_pin(self.get_pin_name('outn'), outn, show=show_pins)
        _, in_upper = self.grid.get_size_dimension(self.size)
        inp, inn = self.connect_differential_tracks(inp_list, inn_list, io_layer, p_tid, n_tid,
                                                    width=io_width, track_upper=in_upper)
        self.add_pin(self.get_pin_name('inp'), inp, show=show_pins)
        self.add_pin(self.get_pin_name('inn'), inn, show=show_pins)
        return vss_warr_list, ymid_out, ckout_tid

    # noinspection PyUnusedLocal
    def _draw_layout_helper(self, lch, wp, wn, fgn, fg_inbuf_list, fg_outbuf_list,
                            nduml, ndumr, nsep, threshold, ptap_w, ntap_w, io_width,
                            num_track_sep, rename_dict, guard_ring_nf, show_pins,
                            pgr_w, ngr_w, **kwargs):
        """Draw the layout of a transistor for characterization.
        """
        end_mode = 15

        # get AnalogBaseInfo
        mconn_layer = AnalogBase.get_mos_conn_layer(self.grid.tech_info)
        top_layer = mconn_layer + 3
        layout_info = AnalogBaseInfo(self.grid, lch, guard_ring_nf, top_layer=top_layer, end_mode=end_mode)

        lower_track_width = 1
        clk_width = 2
        nsep = max(nsep, layout_info.min_fg_sep)
        num_inbuf = len(fg_inbuf_list)
        num_outbuf = len(fg_outbuf_list)

        # calculate number of tracks and clk/signal track location.
        io_track_pitch = lower_track_width + num_track_sep
        clk_tr_idx = (clk_width - 1) / 2
        inp_tr_idx = (lower_track_width - 1) / 2
        inn_tr_idx = inp_tr_idx + io_track_pitch
        out_tr_idx = inn_tr_idx + io_track_pitch

        num_ds_tr = 3 * lower_track_width + 2 * num_track_sep

        nw_list = [wn, wn]
        nth_list = [threshold, threshold]
        pw_list = [wp]
        pth_list = [threshold]
        ng_tracks = [clk_width, clk_width]
        nds_tracks = [num_ds_tr, 2 * lower_track_width + num_track_sep + 1]
        pg_tracks = [clk_width]
        pds_tracks = [1]

        # find minimum number of fingers and clock buffers/passgate column indices
        inbuf_idx_list = []
        outbuf_idx_list = []
        fg_tot = nduml
        for buf_fgp, buf_fgn in fg_inbuf_list:
            inbuf_idx_list.append(fg_tot)
            cur_fg = max(buf_fgp, buf_fgn) + nsep
            fg_tot += cur_fg

        pg_idx_list = [fg_tot, fg_tot + fgn + nsep]
        fg_tot += 2 * (fgn + nsep)

        for buf_fgp, buf_fgn in fg_outbuf_list:
            outbuf_idx_list.append(fg_tot)
            cur_fg = max(buf_fgp, buf_fgn) + nsep
            fg_tot += cur_fg

        fg_tot += (ndumr - nsep)
        self._fg_tot = fg_tot

        n_kwargs = [dict(ds_dummy=True), dict(ds_dummy=False)]
        # draw transistor rows
        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=['MX', 'MX'], p_orientations=['R0'],
                       guard_ring_nf=guard_ring_nf,
                       n_kwargs=n_kwargs,
                       pgr_w=pgr_w, ngr_w=ngr_w,
                       top_layer=top_layer,
                       end_mode=end_mode,
                       )
        blk_right, blk_top = self.grid.get_size_dimension(self.size)
        io_layer = self.mos_conn_layer + 2
        # get dummy gate connection track ID.
        vss_id = self.make_track_id('nch', 0, 'g', clk_tr_idx, width=clk_width)
        vss_warr_list = []
        # draw clock buffers
        clk_ports_list = [[] for _ in range(num_inbuf + num_outbuf + 1)]
        for idx, (bidx, (buf_fgp, buf_fgn)) in enumerate(zip(inbuf_idx_list, fg_inbuf_list)):
            vss_warrs = self._draw_clk_buffer(bidx, buf_fgp, buf_fgn,
                                              clk_ports_list[idx], clk_ports_list[idx + 1])
            vss_warr_list.extend(vss_warrs)
        for idx, (bidx, (buf_fgp, buf_fgn)) in enumerate(zip(outbuf_idx_list, fg_outbuf_list)):
            vss_warrs = self._draw_clk_buffer(bidx, buf_fgp, buf_fgn,
                                              clk_ports_list[idx + num_inbuf],
                                              clk_ports_list[idx + num_inbuf + 1])
            vss_warr_list.extend(vss_warrs)

        # draw differential passgates
        vss_pg, ymid_out, ckout_tid = self._draw_pass_gates(pg_idx_list, fgn, inp_tr_idx, inn_tr_idx,
                                                            out_tr_idx, lower_track_width, io_width,
                                                            clk_ports_list[num_inbuf], show_pins=show_pins)
        vss_warr_list.extend(vss_pg)

        vss_exp = self.connect_to_tracks(vss_warr_list, vss_id, track_upper=blk_right, track_lower=0.0)

        pg_track = self.make_track_id('nch', 1, 'g', clk_tr_idx, width=clk_width)
        alt_track = self.make_track_id('pch', 0, 'g', clk_tr_idx, width=clk_width)
        # connect input clock wires
        cur_track = pg_track
        next_track = alt_track
        for idx in range(num_inbuf, -1, -1):
            clk_ports = clk_ports_list[idx]
            tr_warr = self.connect_to_tracks(clk_ports, cur_track)
            if idx == 0:
                # connect clock input to vertical layer
                clk_tid = self.grid.coord_to_nearest_track(io_layer, tr_warr.middle)
                tr_warr = self.connect_to_tracks(tr_warr, TrackID(io_layer, clk_tid, width=io_width),
                                                 track_upper=blk_top)
                self.add_pin(self.get_pin_name('ckin'), tr_warr, show=show_pins)
            elif idx == num_inbuf:
                self.add_pin(self.get_pin_name('ckpg'), tr_warr, show=show_pins)

            cur_track, next_track = next_track, cur_track

        # connect output clock wires
        cur_track = alt_track
        next_track = pg_track
        ckout_tr = None
        clk_reroute_tid = self.grid.coord_to_nearest_track(io_layer + 1, ymid_out, mode=1)
        clk_reroute_tid += (clk_width - 1) / 2
        ckout_ymid = self.grid.track_to_coord(io_layer + 1, clk_reroute_tid)
        for idx in range(num_inbuf + 1, num_inbuf + num_outbuf + 1):
            clk_ports = clk_ports_list[idx]
            tr_warr = self.connect_to_tracks(clk_ports, cur_track)
            if idx == num_inbuf + num_outbuf:
                # connect clock output to vertical layer
                clk_tid = self.grid.coord_to_nearest_track(io_layer, tr_warr.middle)
                ckout_tr = self.connect_to_tracks(tr_warr, TrackID(io_layer, clk_tid, width=io_width),
                                                  track_lower=ckout_ymid)

            cur_track, next_track = next_track, cur_track

        # re-route clock output to be at center of output wires
        ckout_tr = self.connect_to_tracks(ckout_tr, TrackID(io_layer + 1, clk_reroute_tid, width=clk_width))
        ckout_tr = self.connect_to_tracks(ckout_tr, TrackID(io_layer, ckout_tid, width=clk_width), track_lower=0.0)
        self.add_pin(self.get_pin_name('ckout'), ckout_tr, show=show_pins)

        # draw dummies
        vss_warrs, vdd_warrs = self.fill_dummy(lower=0.0, upper=blk_right)
        vss_warrs.append(vss_exp)

        self.add_pin('VSS', vss_warrs, show=show_pins)
        self.add_pin('VDD', vdd_warrs, show=show_pins)


class NPassGateWClk(TemplateBase):
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
        super(NPassGateWClk, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            wp='pmos width, in meters/number of fins.',
            wn='nmos width, in meters/number of fins.',
            fgn='passgate nmos number of fingers.',
            fg_inbuf_list='List of input clock buffer pmos/nmos number of fingers.',
            fg_outbuf_list='List of output clock buffer pmos/nmos number of fingers.',
            nduml='number of left dummies.',
            ndumr='number of right dummies.',
            nsep='number of fingers of separator dummies.',
            threshold='transistor threshold flavor.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            pgr_w='NMOS guard-ring substrate width, in meters/number of fins.',
            ngr_w='PMOS guard-ring substrate width, in meters/number of fins.',
            num_track_sep='number of tracks reserved as space between clock and signal wires.',
            io_width='input/output track width in number of tracks.',
            rename_dict='pin renaming dictionary.',
            guard_ring_nf='Guard ring width in number of fingers.  0 for no guard ring.',
            show_pins='True to draw pins.',
            tot_width='Total width in number of source/drain tracks.',
        )

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
            nduml=4,
            ndumr=4,
            nsep=0,
            num_track_sep=1,
            io_width=1,
            rename_dict={},
            guard_ring_nf=0,
            show_pins=False,
        )

    def draw_layout(self):
        res = self.grid.resolution
        core_params = self.params.copy()
        tot_width = core_params.pop('tot_width')
        show_pins = core_params['show_pins']

        core_params['show_pins'] = False
        core_master = self.new_template(params=core_params, temp_cls=NPassGateWClkCore)

        sd_pitch = core_master.layout_info.sd_pitch_unit
        tot_width *= sd_pitch
        cur_width = core_master.layout_info.get_total_width(core_master.fg_tot)

        if cur_width > tot_width:
            raise ValueError('Need at least width=%d, but constrained to have width=%d'
                             % (cur_width, tot_width))

        xshift = (tot_width - cur_width) // 2
        inst = self.add_instance(core_master, loc=(xshift, 0), unit_mode=True)

        vdd_warrs = inst.get_all_port_pins('VDD')
        vss_warrs = inst.get_all_port_pins('VSS')

        # set size
        vdd_lay_id = vdd_warrs[0].layer_id
        top_layer = vdd_lay_id + 2
        height = inst.bound_box.height_unit
        self.set_size_from_bound_box(top_layer, BBox(0, 0, tot_width, height, res, unit_mode=True))

        # fill and export supplies
        lay_id = vdd_lay_id + 1
        vdd_warrs, vss_warrs = self.do_power_fill(lay_id, vdd_warrs=vdd_warrs, vss_warrs=vss_warrs, unit_mode=True,
                                                  fill_width=2, fill_space=1, space=0, space_le=0)
        lay_id += 1
        vdd_warrs, vss_warrs = self.do_power_fill(lay_id, vdd_warrs=vdd_warrs, vss_warrs=vss_warrs, unit_mode=True,
                                                  space=0, space_le=0)

        self.add_pin('VSS', vss_warrs, show=show_pins)
        self.add_pin('VDD', vdd_warrs, show=show_pins)
        for port_name in inst.port_names_iter():
            if port_name != 'VSS' and port_name != 'VDD':
                self.reexport(inst.get_port(port_name), show=show_pins)
