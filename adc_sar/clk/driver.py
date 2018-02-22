# -*- coding: utf-8 -*-

from typing import Dict, Set, Any

import numpy as np

import bag
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase
from bag.layout import RoutingGrid, TemplateDB

from .digital import Flop
from .amp import ClkAmp, ClkNorGate



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



class GatedClockRx(TemplateBase):
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
        super(GatedClockRx, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
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
            clkrx_params='clkrx parameters',
            nor_params='nor parameters.',
            io_width_ntr='flop io wire width.',
        )

    def draw_layout(self):
        clkrx_params = self.params['clkrx_params']
        nor_params = self.params['nor_params']
        io_width_ntr = self.params['io_width_ntr']

        clkrx_master = self.new_template(params=clkrx_params, temp_cls=ClkAmp)
        nor_master = self.new_template(params=nor_params, temp_cls=ClkNorGate)
        clkrx_inst = self.add_instance(clkrx_master, 'X0', loc=(0, 0))

        # clkrx_x = nor_nxblk
        num_space_vert = 4
        total_height_nor = 2 * nor_nyblk

        total_height = max(total_height_nor, clkrx_y)

        print(nor_master.size, total_height, blk_w, blk_h)

        flop_x = 0

        x_nor = clkrx_x + flop_x

        bot_nor = self.add_instance(nor_master, 'X1', loc=(x_nor, 0))
        top_nor = self.add_instance(nor_master, 'X2', loc=(x_nor, total_height),
                                    orient='MX')
        flops_master = self.new_template(params=dict(config_file='adc_sar_retimer_logic.yaml'), temp_cls=Flop)

        blk_w_flop, blk_h_flop = self.grid.get_block_size(flops_master.size[0])
        blk_h_flop = flops_master.array_box.height
        x_flops = round((x_nor + (nor_nxblk - 10.44) / 2) / blk_w_flop) * blk_w_flop
        y_flops = round((total_height / 2 - 1.44) / blk_h_flop) * blk_h_flop

        flop_inst = self.add_instance(flops_master, 'XFF', loc=(x_flops, y_flops))

        self.inv_num_fingers = clkrx_master.inv_num_fingers

        vss_nor_top = top_nor.get_all_port_pins('VSS')[0]
        vdd_nor_top = top_nor.get_all_port_pins('VDD')[0]

        vss_nor_bot = bot_nor.get_all_port_pins('VSS')[0]
        vdd_nor_bot = bot_nor.get_all_port_pins('VDD')[0]

        vss_clkrx = clkrx_inst.get_all_port_pins('VSS')
        vdd_clkrx = clkrx_inst.get_all_port_pins('VDD')
        clkrx_in_top = clkrx_inst.get_all_port_pins('IN_TOP')[0]
        clkrx_in_bot = clkrx_inst.get_all_port_pins('IN_BOT')[0]

        warr_vss = self.connect_wires([vss_clkrx[0], vss_clkrx[1], vss_nor_bot, vss_nor_top])

        self.add_pin('VSS', warr_vss, label='VSS:')

        clkin_nor_top = top_nor.get_all_port_pins('in')[0]
        clkout_nor_top = top_nor.get_all_port_pins('out')[0]
        en_nor_top = top_nor.get_all_port_pins('en')[0]

        clkin_nor_bot = bot_nor.get_all_port_pins('in')[0]
        clkout_nor_bot = bot_nor.get_all_port_pins('out')[0]
        en_nor_bot = bot_nor.get_all_port_pins('en')[0]

        clkrx_top_out = clkrx_inst.get_all_port_pins('OUT_TOP')[0]
        clkrx_bot_out = clkrx_inst.get_all_port_pins('OUT_BOT')[0]

        warr_clktop = self.connect_to_tracks(clkrx_top_out, clkin_nor_top.track_id)
        warr_clkbot = self.connect_to_tracks(clkrx_bot_out, clkin_nor_bot.track_id)

        warr_clk_connection_top = self.connect_wires([warr_clktop, clkin_nor_top])

        warr_clk_connection_bot = self.connect_wires([warr_clkbot, clkin_nor_bot])

        in_bot0 = flop_inst.get_all_port_pins('IBOT0')[0]
        in_bot1 = flop_inst.get_all_port_pins('IBOT1')[0]

        out_bot0 = flop_inst.get_all_port_pins('OBOT0')[0]
        out_bot1 = flop_inst.get_all_port_pins('OBOT1')[0]

        in_top0 = flop_inst.get_all_port_pins('ITOP0')[0]
        in_top1 = flop_inst.get_all_port_pins('ITOP1')[0]
        out_top0 = flop_inst.get_all_port_pins('OTOP0')[0]
        out_top1 = flop_inst.get_all_port_pins('OTOP1')[0]

        clk_top0 = flop_inst.get_all_port_pins('CLKTOP0')[0]
        clk_top1 = flop_inst.get_all_port_pins('CLKTOP1')[0]

        clk_bot0 = flop_inst.get_all_port_pins('CLKBOT0')[0]
        clk_bot1 = flop_inst.get_all_port_pins('CLKBOT1')[0]

        in_inv_top0 = flop_inst.get_all_port_pins('IINVTOP0')[0]
        out_inv_top0 = flop_inst.get_all_port_pins('OINVTOP0')[0]

        in_inv_bot0 = flop_inst.get_all_port_pins('IINVBOT0')[0]
        out_inv_bot0 = flop_inst.get_all_port_pins('OINVBOT0')[0]

        in_inv_top1 = flop_inst.get_all_port_pins('IINVTOP1')[0]
        out_inv_top1 = flop_inst.get_all_port_pins('OINVTOP1')[0]

        in_inv_bot1 = flop_inst.get_all_port_pins('IINVBOT1')[0]
        out_inv_bot1 = flop_inst.get_all_port_pins('OINVBOT1')[0]

        flop_pin_layer = in_top1.layer_id
        pin_connect_layer = flop_pin_layer + 1

        track_top_int = self.grid.coord_to_nearest_track(pin_connect_layer, in_top1.middle)
        track_top = TrackID(pin_connect_layer, track_top_int - 1)

        # hack by Eric: add 1
        track_bot_int = self.grid.coord_to_nearest_track(pin_connect_layer, in_bot1.middle) + 1
        track_bot = TrackID(pin_connect_layer, track_bot_int)

        track_flop_middle = TrackID(pin_connect_layer,
                                    self.grid.coord_to_nearest_track(pin_connect_layer, y_flops + blk_h_flop / 2))

        warr_out_top = self.connect_to_tracks(out_top1, track_flop_middle)
        warr_in_bot = self.connect_to_tracks(in_bot1, track_flop_middle)

        warr_connect_top_bot = self.connect_wires([warr_in_bot, warr_out_top])

        warr_dum_bot = self.connect_to_tracks(in_bot0, track_flop_middle)

        warr_top_int = self.connect_to_tracks([in_top1, out_top0], track_top)

        warr_top_in2 = self.connect_to_tracks([out_top1, in_inv_top0], track_top)
        warr_top_in3 = self.connect_to_tracks([out_inv_top0, in_inv_top1], track_top)

        warr_bot_int = self.connect_to_tracks([in_inv_bot0, out_bot1], track_bot)
        warr_bot_in2 = self.connect_to_tracks([in_inv_bot1, out_inv_bot0], track_bot)

        len_clk_wire = 1.6

        tid_top_clk0 = clk_top0.track_id
        warr_clk_top0 = self.add_wires(flop_pin_layer, tid_top_clk0.base_index, clk_top0.upper,
                                       clk_top0.upper + len_clk_wire)

        tid_top_clk1 = clk_top1.track_id
        warr_clk_top1 = self.add_wires(flop_pin_layer, tid_top_clk1.base_index, clk_top1.upper,
                                       clk_top1.upper + len_clk_wire)

        track_clk_top = TrackID(pin_connect_layer,
                                self.grid.coord_to_nearest_track(pin_connect_layer, warr_clk_top0.upper),
                                width=io_width_ntr)

        warr_clk_top_flop = self.connect_to_tracks([warr_clk_top0, warr_clk_top1], track_clk_top)

        tid_bot_clk0 = clk_bot0.track_id
        warr_clk_bot0 = self.add_wires(flop_pin_layer, tid_bot_clk0.base_index, clk_bot0.lower - len_clk_wire,
                                       clk_bot0.lower)

        tid_bot_clk1 = clk_bot1.track_id
        warr_clk_bot1 = self.add_wires(flop_pin_layer, tid_bot_clk1.base_index, clk_bot1.lower - len_clk_wire,
                                       clk_bot1.lower)

        track_clk_bot = TrackID(pin_connect_layer,
                                self.grid.coord_to_nearest_track(pin_connect_layer, warr_clk_bot0.lower, mode=-1),
                                width=io_width_ntr)

        warr_en_vert = self.add_wires(flop_pin_layer, in_top0.track_id.base_index, in_top0.upper,
                                      in_top0.upper + len_clk_wire + 1)

        warr_clk_bot_flop = self.connect_to_tracks([warr_clk_bot0, warr_clk_bot1], track_clk_bot)

        vdd_flop = flop_inst.get_all_port_pins('VDD')[0]
        vss_flop = flop_inst.get_all_port_pins('VSS')[0]

        vdd_flop_dummy_connection = self.add_wires(vdd_flop.layer_id, vdd_flop.track_id.base_index, vdd_flop.lower,
                                                   vdd_flop.upper, width=vdd_flop.width)
        vdd_flop_hor = self.connect_to_tracks(vdd_flop_dummy_connection, track_flop_middle)

        self.connect_wires([vdd_flop_hor, warr_dum_bot])

        # warr_vdd_bot = self.connect_to_tracks(vdd_flop, vdd_nor_bot.track_id)

        # warr_vdd_top = self.connect_to_tracks(vdd_flop, vdd_nor_top.track_id)

        # import pdb
        # pdb.set_trace()

        vdd_uplayer = pin_connect_layer + 1

        left_edge = vdd_flop.get_bbox_array(self.grid).left
        right_edge = vdd_flop.get_bbox_array(self.grid).right

        width_left = vdd_flop_dummy_connection.get_bbox_array(self.grid).right - left_edge

        center_left = left_edge + (width_left / 2)
        center_right = right_edge - (width_left / 2)

        track_uplayer = TrackID(vdd_uplayer, self.grid.coord_to_nearest_track(vdd_uplayer, center_left))
        track_uplayer_right = TrackID(vdd_uplayer, self.grid.coord_to_nearest_track(vdd_uplayer, center_right))

        lower_uplayer = vdd_flop.lower
        upper_uplayer = vdd_flop.upper

        vdd_flop_m5_left = self.add_wires(vdd_uplayer, track_uplayer.base_index, lower_uplayer, upper_uplayer,
                                          width=vdd_flop.track_id.width)

        vdd_flop_m5_right = self.add_wires(vdd_uplayer, track_uplayer_right.base_index, lower_uplayer, upper_uplayer,
                                           width=vdd_flop.track_id.width)

        # warr_vdd_bot = self.connect_to_tracks( , vdd_nor_bot.track_id)

        len_nor_wire = 2
        warr_en_nor_bot_vert = self.add_wires(en_nor_bot.layer_id, en_nor_bot.track_id.base_index, en_nor_bot.upper,
                                              en_nor_bot.upper + len_nor_wire, width=io_width_ntr)
        warr_en_nor_top_vert = self.add_wires(en_nor_top.layer_id, en_nor_top.track_id.base_index, en_nor_top.upper,
                                              en_nor_top.upper - len_nor_wire, width=io_width_ntr)

        warr_en_nor_bot_hor = self.connect_to_tracks(warr_en_nor_bot_vert, warr_clk_bot_flop.track_id)

        warr_en_nor_bot_vert2 = self.connect_to_tracks(out_inv_bot1, warr_en_nor_bot_hor.track_id)

        warr_en_nor_bot_hor2 = self.connect_wires([warr_en_nor_bot_vert2, warr_en_nor_bot_hor])

        warr_en_nor_top_hor = self.connect_to_tracks(warr_en_nor_top_vert, warr_clk_top_flop.track_id)

        warr_en_nor_top_vert2 = self.connect_to_tracks(out_inv_top1, warr_en_nor_top_hor.track_id)

        warr_en_nor_top_hor2 = self.connect_wires([warr_en_nor_top_vert2, warr_en_nor_top_hor])
        vdd_vert_layer = pin_connect_layer + 1

        track_clkrx_nor_int = TrackID(vdd_vert_layer, self.grid.coord_to_nearest_track(vdd_vert_layer, x_nor),
                                      width=vdd_nor_bot.width)
        # lower layer is on a different grid, want the wire to still be wide

        # warr_nor_bot_vdd = self.connect_to_tracks(vdd_nor_bot, track_clkrx_nor_int)

        # warr_nor_top_vdd = self.connect_to_tracks(vdd_nor_top, track_clkrx_nor_int)

        # warr_clkrx_bot = self.connect_to_tracks(vdd_clkrx[0], track_clkrx_nor_int)
        # warr_clkrx_top = self.connect_to_tracks(vdd_clkrx[1], track_clkrx_nor_int)

        warr_vdd_bot = self.connect_wires([vdd_clkrx[0], vdd_nor_bot])
        warr_vdd_top = self.connect_wires([vdd_clkrx[1], vdd_nor_top])
        self.add_pin('VDD', warr_vdd_bot + warr_vdd_top, label='VDD:')

        # import pdb
        # pdb.set_trace()

        track_clk_flop_int = TrackID(pin_connect_layer + 1,
                                     self.grid.coord_to_nearest_track(pin_connect_layer + 1, warr_clk_bot_flop.lower),
                                     width=io_width_ntr)

        warr_clk_bot_nor_vert = self.connect_to_tracks(warr_clk_connection_bot, track_clk_flop_int)

        warr_clk_bot_flop_vert = self.connect_to_tracks(warr_clk_bot_flop, track_clk_flop_int)

        warr_clk_bot_vert = self.connect_wires([warr_clk_bot_nor_vert, warr_clk_bot_flop_vert])

        warr_clk_top_nor_vert = self.connect_to_tracks(warr_clk_connection_top, track_clk_flop_int)

        warr_clk_top_flop_vert = self.connect_to_tracks(warr_clk_top_flop, track_clk_flop_int)

        warr_clk_top_vert = self.connect_wires([warr_clk_top_nor_vert, warr_clk_top_flop_vert])

        self.add_pin('VSS', vss_flop, label='VSS:')
        self.add_pin('VDD', vdd_flop, label='VDD:')
        self.add_pin('CLKP', clkout_nor_top)
        self.add_pin('CLKN', clkout_nor_bot)

        self.add_pin('CLKP_PAD', clkrx_in_top)
        self.add_pin('CLKN_PAD', clkrx_in_bot)

        self.add_pin('RST', warr_en_vert)

        self.num_transistors_row = nor_master.num_transistors_row
