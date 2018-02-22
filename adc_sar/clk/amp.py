# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Set, Any

import numpy as np
import math

from bag.util.search import BinaryIterator
from bag.layout.util import BBox
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase

from abs_templates_ec.analog_core.base import AnalogBase
from abs_templates_ec.resistor.core import ResArrayBase


if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class ResFeedback(ResArrayBase):
    """An template for creating termination resistors.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(ResFeedback, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=1,
            res_type='reference',
            em_specs={'idc': 0.0, 'iac_rms': 0.0, 'iac_peak': 0.0},
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
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            io_width_ntr='io track width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def _up_three_layers(self, warr, connect_to='', max_size=0):
        top_layer_width = self.params['io_width_ntr']
        # aka build your own via stack
        #connect to first layer
        lay_id = warr.layer_id
        mid_coord = self.grid.track_to_coord(lay_id, warr.track_id.base_index)
        lay_id +=1
        min_len = self.grid.get_min_length(lay_id, 1)
        mid_tr = self.grid.find_next_track(warr.layer_id + 1, warr.lower, half_track=True)
        warr = self.connect_to_tracks(warr, TrackID(lay_id, mid_tr), track_lower=mid_coord-min_len/2,
                                      track_upper=mid_coord+min_len/2)

        #connect to next layer
        mid_coord = self.grid.track_to_coord(lay_id, warr.track_id.base_index)
        lay_id += 1
        warr_tr = self.grid.coord_to_nearest_track(lay_id, warr.middle, half_track=True)
        min_len = self.grid.get_min_length(lay_id, 1)
        warr2 = self.connect_to_tracks(warr, TrackID(lay_id, warr_tr), track_lower=mid_coord-min_len/2,
                                      track_upper=mid_coord+min_len/2)

        # connect to next layer again
        mid_coord = self.grid.track_to_coord(lay_id, warr2.track_id.base_index)
        lay_id += 1
        warr_tr = self.grid.coord_to_nearest_track(lay_id, warr2.middle, half_track=True)
        min_len = self.grid.get_min_length(lay_id, 1)
        if connect_to == 'bottom':
            return self.connect_to_tracks(warr2, TrackID(lay_id, warr_tr), track_lower=0)
        if connect_to == 'top':
            return self.connect_to_tracks(warr2, TrackID(lay_id, warr_tr),
                                          track_upper=max_size)

        return self.connect_to_tracks(warr2, TrackID(lay_id, warr_tr, width=top_layer_width), track_lower=mid_coord - min_len / 2,
                                      track_upper=mid_coord + min_len / 2)


    def draw_layout(self):
        # type: () -> None

        # draw array
        nx = self.params['nx']
        if nx <= 0:
            raise ValueError('number of resistors in a row must be positive.')
        ny = self.params['ny']
        if ny % 2 != 0:
            raise ValueError('number of resistors in a column must be even')

        draw_params = self.params.copy()
        draw_params.pop('io_width_ntr')

        em_specs = draw_params.pop('em_specs')
        div_em_specs = em_specs.copy()
        for key in ('idc', 'iac_rms', 'iac_peak'):
            if key in em_specs:
                div_em_specs[key] = div_em_specs[key] / ny
        self.draw_array(em_specs=div_em_specs, **draw_params)

        # connect row resistors
        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        hl_warr_list = []
        hr_warr_list = []
        hl_snake_warr_list = []
        hr_snake_warr_list = []
        left_row = 0
        right_row = 1
        vm_width = self.w_tracks[1]
        for row_idx in range(ny):
            for col_idx in range(nx):
                ports_l = self.get_res_ports(row_idx, col_idx)
                if col_idx == nx-1:
                    ports_r = self.get_res_ports(row_idx, col_idx)
                else:
                    ports_r = self.get_res_ports(row_idx, col_idx + 1)
                con_par = (col_idx  + row_idx) % 2
                #self.connect_wires([ports_l[con_par], ports_r[con_par]])
                if col_idx != 0:
                    self.connect_wires([ports_l[con_par], port_prev[con_par]])
                port_prev = ports_l
                if col_idx == 0:
                    #column 0, left side
                    hl_warr_list.append(ports_l[1 - con_par])
                    if row_idx > 0 :
                        # not in the first row
                        ports_l_below = self.get_res_ports(row_idx - 1, col_idx)
                        if left_row == 1 and row_idx != ny/2:
                            #connect the left side of this row downward
                            #self.connect_wires(ports_l[1 - con_par], ports_l_below[con_par])
                            vl_id = self.grid.coord_to_nearest_track(vm_layer, ports_l[con_par].middle, mode = -1, half_track=True)
                            v_tid = TrackID(vm_layer, vl_id, width=vm_width)
                            #print 'should be making a grid connection'
                            cur_list_l = [ports_l[con_par], ports_l_below[1-con_par]]
                            vl_warr = self.connect_to_tracks(cur_list_l, v_tid)
                            #left_row = 0
                            #right_row= 1
                if col_idx == nx - 2:
                    hr_warr_list.append(ports_r[1 - con_par])
                    if row_idx > 0:
                        ports_r_below = self.get_res_ports(row_idx - 1, col_idx + 1)
                        if right_row == 1 and row_idx != ny/2:
                            #todo
                            #left_row = 1
                            #right_row = 0
                            vr_id = self.grid.coord_to_nearest_track(vm_layer, ports_r[con_par].middle, mode = 1, half_track=True)
                            v_tid = TrackID(vm_layer, vr_id, width=vm_width)
                            #print 'should be making a grid connection'
                            cur_list_r = [ports_r[con_par], ports_r_below[1-con_par]]
                            vr_warr = self.connect_to_tracks(cur_list_r, v_tid)
            if row_idx > 0:
                left_row = 1 - left_row
                right_row = 1 - right_row
        #connect the starting point to the edge
        print('port_r %s %s' % (ports_r[con_par], ports_r))

        min_length_vm_layer = self.grid.get_min_length(vm_layer, 1)
        min_length_xm_layer = self.grid.get_min_length(xm_layer, 1)
        min_length_ym_layer = self.grid.get_min_length(ym_layer, 1)

        #if right_row == 1:
            #ended on the left
        xm_width = self.w_tracks[2]
        x_pitch = self.num_tracks[2]
        x_base = self.get_h_track_index(0, (x_pitch - 1) / 2.0)

        x_tid = TrackID(xm_layer, x_base, width=xm_width, num=ny, pitch=x_pitch)
        bot_w = self.grid.get_track_width(xm_layer, xm_width)
        ym_width = self.grid.get_min_track_width(ym_layer, bot_w=bot_w, **em_specs)

        ports_bottom_left = self.get_res_ports(0,0)
        con_par = (0 + 0) % 2
        vl_id = self.grid.coord_to_nearest_track(ym_layer, ports_bottom_left[con_par].middle, mode = -1, half_track=True)
        code_warr = self._up_three_layers(ports_bottom_left[con_par]) #, connect_to='bottom')
        self.add_pin("in1", code_warr, show=True)
        print('track inside res')
        print(vl_id)
        myProp = self.size
        blockWidth, blockHeight = self.grid.get_size_dimension(myProp)
        ports_bottom_right = self.get_res_ports(ny / 2 - 1, nx - 1)
        con_par = int((nx-1 + (ny/2)-1)) % 2
        con_par = 1 - con_par

        code_warr2 = self._up_three_layers(ports_bottom_right[con_par]) #, connect_to='bottom')
        self.add_pin("in2", code_warr2, show=True)

        topright_tid = self.grid.coord_to_track(ym_layer, blockHeight)
        ports_top_right = self.get_res_ports(ny / 2, nx - 1)
        con_par = 1 - con_par
        topright_warr = self._up_three_layers(ports_top_right[con_par]) #, connect_to='top', max_size=self.array_box.top)
        self.add_pin("in4", topright_warr, show=True)

        ports_top_left = self.get_res_ports(ny - 1, 0)
        con_par = (ny - 1) % 2
        topleft_warr = self._up_three_layers(ports_top_left[con_par]) #, connect_to='top', max_size=self.array_box.top)
        self.add_pin("in3", topleft_warr, show=True)






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
        self.num_fingers = None

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
            n_w='nmos width, in meters/number of fins.',
            p_w='pmos width',
            nfp='pmos number of fingers.',
            nfn='nmos number of fingers.',
            ndum='number of dummies on left.',
            n_intent='nmos threshold flavor.',
            p_intent='pmos threshold flavor.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            top_layer='the top level layer ID.',
            blk_width='Width of this template in number of blocks.',
            io_width_ntr='input/output track width in number of tracks.',
            io_space_ntr='input/output track space',
            rename_dict='pin renaming dictionary.',
            show_pins='True to draw pins.',
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
            io_width_ntr=1,
            rename_dict={},
            show_pins=False,
        )

    def draw_layout(self):
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, n_w, p_w, nfp, nfn, ndum, n_intent, p_intent,
                            ptap_w, ntap_w, top_layer, blk_width, io_width_ntr,
                            rename_dict, show_pins, io_space_ntr):
        """Draw the layout of a transistor for characterization.
        """

        # compute template width in number of source/drain pitches
        res = self.grid.resolution
        tech_params = self.grid.tech_info.tech_params
        mos_cls = tech_params['layout']['mos_template']
        sd_pitch = mos_cls.get_sd_pitch(lch)
        wtot_unit, _ = self.grid.get_size_dimension((top_layer, blk_width, 1), unit_mode=True)
        self.grid.get_block_size(top_layer, unit_mode=True)
        sd_pitch_unit = int(round(sd_pitch / res))
        q, r = divmod(wtot_unit, sd_pitch_unit)

        # find maximum number of fingers we can draw
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_fg = bin_iter.get_next()
            num_sd = mos_cls.get_template_width(cur_fg)
            if num_sd == q:
                bin_iter.save()
                break
            elif num_sd < q:
                bin_iter.save()
                bin_iter.up()
            else:
                bin_iter.down()

        fg_tot = bin_iter.get_last_save()

        # compute offset
        num_sd = mos_cls.get_template_width(fg_tot)
        if r != 0:
            sub_xoff = (q + 1 - num_sd) // 2
        else:
            sub_xoff = (q - num_sd) // 2

        # draw transistor rows
        nw_list = [n_w]
        pw_list = [p_w] * 2
        nth_list = [n_intent]
        pth_list = [p_intent] * 2
        num_track_sep = 1
        ng_tracks = [io_width_ntr + 2 * io_space_ntr]
        pg_tracks = [io_width_ntr + 2 * io_space_ntr, io_width_ntr]
        pds_tracks = [1, io_width_ntr]
        nds_tracks = [1]
        p_orient = ['R0', 'MX']
        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list, num_track_sep,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=['MX'], p_orientations=p_orient,
                       pitch_offset=(sub_xoff, 0),
                       )

        # connections methods

        # draw MOS connection
        nidx_off = max((nfp - nfn) // 2, 0)
        pidx_off = max((nfn - nfp) // 2, 0)
        if (nfp - nfn) % 4 == 2:
            # if nfp and nfn differ by 2 mod 4, and we line up the center, the drain/source
            # will not match.  Therefore, we shift one of their index by 1.
            if nidx_off > pidx_off:
                nidx_off -= 1
            else:
                pidx_off -= 1


        ndum_n_leftright = (fg_tot - nfn) // 2
        ndum_p_leftright = (fg_tot - nfp) // 2
        print(ndum_p_leftright, ndum_n_leftright, io_width_ntr, nfn, nfp, num_sd, fg_tot)

        # draw PMOS with source up, drain down
        p_ports = self.draw_mos_conn('pch', 0, ndum_p_leftright + pidx_off, nfp, 2, 0)
        # draw NMOS with source down, drain up
        n_ports = self.draw_mos_conn('nch', 0, ndum_n_leftright + nidx_off, nfn, 0, 2)

        myProp = self.size
        blockWidth, blockHeight = self.grid.get_size_dimension(myProp)

        # connect to supplies
        self.connect_to_substrate('ntap', p_ports['s'])
        self.connect_to_substrate('ptap', n_ports['s'])

        # connect inputs and export
        in_id = self.make_track_id('pch', 0, 'g', (io_width_ntr - 1) / 2 + io_space_ntr, width=io_width_ntr)
        in_warr = self.connect_to_tracks([p_ports['g'], n_ports['g']], in_id)
        self.add_pin(self.get_pin_name('in'), in_warr, show=show_pins)
        #
        # connect outputs and export
        out_id = self.make_track_id('nch', 0, 'g', (io_width_ntr - 1) / 2 + io_space_ntr, width=io_width_ntr)
        out_warr = self.connect_to_tracks([p_ports['d'], n_ports['d']], out_id)
        self.add_pin(self.get_pin_name('out'), out_warr, show=show_pins)
        #
        # draw dummies
        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        # export supplies
        self.add_pin(self.get_pin_name('VSS'), ptap_wire_arrs)
        self.add_pin(self.get_pin_name('VDD'), ntap_wire_arrs)

        self.num_fingers = fg_tot


class ClkAmp(TemplateBase):
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
        super(ClkAmp, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self.inv_num_fingers = None

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
            res_params='resistor array parameters',
            amp_params='amplifier parameters.',
            cap_params='cap parameters',
        )

    def draw_layout(self):
        res_params = self.params['res_params']
        amp_params = self.params['amp_params']
        cap_params = self.params['cap_params']

        res_master = self.new_template(params=res_params, temp_cls=ResFeedback)
        top_layer, res_nxblk, res_nyblk = res_master.size
        blk_w, blk_h = self.grid.get_block_size(top_layer)

        amp_params['top_layer'] = top_layer
        amp_params['blk_width'] = res_nxblk
        invamp_master = self.new_template(params=amp_params, temp_cls=InvAmp)
        invamp_nyblk = invamp_master.size[2]

        cap_master = self.new_template(params=cap_params, temp_cls=BBCapUnitCell_basic)

        #cap_x = cap_master.size[2]
        #import pdb
        #pdb.set_trace()

        wire_room_x = 4 * blk_w
        num_blks_cap_separation = 8
        num_res_space = 4
        x_offset = 12 * blk_w

        cap_x = x_offset + cap_master.array_box.width + wire_room_x # array box width
        cap_y = cap_master.array_box.height # array box height


        print(wire_room_x, blk_h, cap_x, cap_y)
        total_height = (blk_h * (invamp_nyblk * 2 + res_nyblk + 2*num_res_space))

        cap_offset =  (total_height - (2 * cap_y)) / 2
        cap_offset = (math.floor(cap_offset / blk_h)-num_blks_cap_separation) * blk_h
        cap_bottom = self.add_instance(cap_master, 'X_CAP1', loc=(x_offset, cap_offset))
        cap_top = self.add_instance(cap_master, 'X_CAP2', loc=(x_offset,-1*cap_offset +  total_height), orient='MX')


        bot_amp = self.add_instance(invamp_master, 'X1', loc=(cap_x, 0))
        res_arr = self.add_instance(res_master, 'XR', loc=(cap_x, blk_h * (invamp_nyblk+num_res_space)))
        top_amp = self.add_instance(invamp_master, 'X2', loc=(cap_x, total_height),
                                    orient='MX')

        res_box = res_arr.array_box
        bot_vdd = bot_amp.get_all_port_pins('VDD')[0]
        top_vdd = top_amp.get_all_port_pins('VDD')[0]
        self.add_pin('VDD', bot_vdd, label='VDD:')
        self.add_pin('VDD', top_vdd, label='VDD:')

        bot_vss = bot_amp.get_all_port_pins('VSS')[0]
        top_vss = top_amp.get_all_port_pins('VSS')[0]
        self.add_pin('VSS', bot_vss, label='VSS:')
        self.add_pin('VSS', top_vss, label='VSS:')

        yb = self.grid.track_to_coord(bot_vdd.layer_id, bot_vdd.track_id.base_index)
        yt = self.grid.track_to_coord(top_vdd.layer_id, top_vdd.track_id.base_index)
        res_box = res_box.extend(y=yb).extend(y=yt)
        self.add_rect('NW', res_box)

        width = cap_x + top_amp.array_box.width
        height = total_height

        blkw, blkh = self.grid.get_block_size(top_layer, unit_mode=True)
        nxblk = -(-width // blkw)
        nyblk = -(-height // blkh)
        self.size = top_layer, width, height
        res = self.grid.resolution
        self.array_box = BBox(0, 0, width, height, res, unit_mode=True)


        res_port1 = res_arr.get_port('in1').get_pins(layer=top_layer)[0]
        inv1_in = bot_amp.get_port('in').get_pins()[0]
        inv1_out = bot_amp.get_port('out').get_pins()[0]
        inv2_in = top_amp.get_port('in').get_pins()[0]
        inv2_out = top_amp.get_port('out').get_pins()[0]
        res_port2 = res_arr.get_port('in2').get_pins(layer=top_layer)[0]
        res_port3 = res_arr.get_port('in3').get_pins(layer=top_layer)[0]
        res_port4 = res_arr.get_port('in4').get_pins(layer=top_layer)[0]
        #vl_id = self.grid.coord_to_nearest_track(ym_layer, ports_bottom_left[con_par].middle, mode=-1, half_track=True)
        #t_id = self.grid.coord_to_nearest_track(top_layer, res_port1[0].middle, mode=-1, half_track=True)
        #t_id = res_port1[0].track_id
        #warr_test2 = self.connect_to_tracks(res_port1[0], t_id)
        #print(t_id)
        #import pdb
        #pdb.set_trace()
        cap_top_port1 = cap_top.get_port('side1').get_pins()[0]
        cap_top_port2 = cap_top.get_port('side2').get_pins()[0]

        cap_bot_port1 = cap_bottom.get_port('side1').get_pins()[0]
        cap_bot_port2 = cap_bottom.get_port('side2').get_pins()[0]

        myTrackleft = res_port1.track_id
        myTrackright = res_port2.track_id

        warr_bottom_left = self.connect_to_tracks(inv1_in, myTrackleft)
        warr1 = self.connect_wires([res_port1, warr_bottom_left])

        warr_bottom_right = self.connect_to_tracks(inv1_out, myTrackright)
        warr2 = self.connect_wires([res_port2, warr_bottom_right])

        myTrackleft = res_port3.track_id
        myTrackright = res_port4.track_id

        warr_top_left = self.connect_to_tracks(inv2_in, myTrackleft)
        warr3 = self.connect_wires([res_port3, warr_top_left])[0]

        warr_top_right = self.connect_to_tracks(inv2_out, myTrackright)
        warr4 = self.connect_wires([res_port4, warr_top_right])[0]

        warr_in_top = self.connect_to_tracks(warr3, cap_top_port1.track_id)

        warr_in_top_final = self.connect_wires([warr_in_top, cap_top_port1])

        warr_in_bot = self.connect_to_tracks(warr1, cap_bot_port1.track_id)

        warr_in_bot_final = self.connect_wires([warr_in_bot, cap_bot_port1])

        #warr_in_top = self.connect_wires([warr3, cap_top_port2])
        #warr_in_bot = self.connect_wires([warr1, cap_bot_port2])

        metal_res_layer = cap_top_port2.layer_id

        track_info_phys = self._temp_db.grid.get_track_info(metal_res_layer)
        w_phys = track_info_phys[0]
        min_space_mres = self.grid.get_min_length(metal_res_layer, cap_params['w_side'])

        print(cap_top_port2.get_bbox_array(self.grid).nx)
        w_metal_res_phys = cap_params['w_side'] *  min_space_mres # the physical width of the metal resistor in um
        l_metal_res_phys = blk_w  # the physical length of the metal resistor in um
        w_metal_res = str(int(w_metal_res_phys*1e3)) + 'n'
        #l_metal_res = str(int(l_metal_res_phys*1e3)) + 'n'
        l_metal_res = '168n' # hardcoded for 16nm right now

        metal_res_params = dict(w=w_metal_res, l=l_metal_res, HardCons=False)
        #import pdb
        #pdb.set_trace()

        metal_res_bot_id = cap_bot_port2.track_id.base_index
        metal_res_bot_track = TrackID(metal_res_layer, metal_res_bot_id)
        metal_res_top_id = cap_top_port2.track_id.base_index
        metal_res_top_track = TrackID(metal_res_layer, metal_res_top_id)


        metal_res_bot_height = self.grid.track_to_coord(metal_res_layer, metal_res_bot_id)
        metal_res_top_height = self.grid.track_to_coord(metal_res_layer, metal_res_top_id)

        yb_top, yt_top = self.grid.get_wire_bounds(cap_top_port2.layer_id, cap_top_port2.track_id.base_index, cap_top_port2.track_id.width)
        metal_res_params['w'] = bag.float_to_si_string(yt_top - yb_top)

        yb_bot, yt_bot = self.grid.get_wire_bounds(cap_bot_port2.layer_id, cap_bot_port2.track_id.base_index, cap_bot_port2.track_id.width)
        self.add_instance_primitive(lib_name='tsmcN16', cell_name='rm'+str(metal_res_layer)+'w', loc=(blk_w, yb_bot), view_name = 'layout',
                                    inst_name = 'mres_bot', orient = "R0", nx = 1, ny = 1, spx = 1, spy = 0.0, params = metal_res_params)


        self.add_instance_primitive(lib_name='tsmcN16', cell_name='rm'+str(metal_res_layer)+'w', loc=(blk_w, yb_top), view_name = 'layout',
                                    inst_name = 'mres_top', orient = "R0", nx = 1, ny = 1, spx = 1, spy = 0.0, params = metal_res_params)



        self.connect_wires([cap_bot_port2, cap_top_port2], lower=0)
        tid_bot = cap_bot_port2.track_id
        wa_bot = self.add_wires(cap_bot_port2.layer_id, tid_bot.base_index, 0, blk_w, width=tid_bot.width)

        self.add_pin(self.get_pin_name('IN_BOT'), wa_bot)

        tid_top = cap_top_port2.track_id
        wa_top = self.add_wires(cap_top_port2.layer_id, tid_top.base_index, 0, blk_w, width=tid_top.width)

        self.add_pin(self.get_pin_name('IN_TOP'), wa_top)


        self.add_pin(self.get_pin_name('OUT_TOP'), warr4)
        self.add_pin(self.get_pin_name('OUT_BOT'), warr2)

        inv_num_fingers = invamp_master.num_fingers

        self.inv_num_fingers = inv_num_fingers








if __name__ == '__main__':
    impl_lib = 'AAAFOO'

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = bag.BagProject()
        temp = 70.0
        # layers = [4, 5, 6]
        # spaces = [0.08, 0.100, 0.08]
        # widths = [0.080, 0.080, 0.080]
        # bot_dir = 'x'
        #
        # routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        layers = [4, 5, 6]  # the layers for the cap
        # loading the technology min space and width from tech_config.yaml
        tech_config_yaml = bag.io.read_yaml(bprj.bag_config['tech_config_path'])
        min_space = tech_config_yaml['layout']['routing_grid_min']['space']
        min_width = tech_config_yaml['layout']['routing_grid_min']['space']
        spaces = min_space * np.ones_like(layers)
        widths = min_width * np.ones_like(layers)
        bot_dir = 'x'  # bottom layer direction ('x' for horizontal, 'y' for vertical)

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=False)
    else:
        print('loading BAG project')
