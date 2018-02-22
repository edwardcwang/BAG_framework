# -*- coding: utf-8 -*-

from typing import Dict, Set, Any

import numpy as np

import bag
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase
from bag.layout import RoutingGrid, TemplateDB

from .digital import Flop
from .amp import ClkAmp, ClkNorGate


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
        parent_grid = self.grid
        self.grid = parent_grid.copy()
        self.grid.add_new_layer(3, 0.058, 0.032, 'y')
        self.grid.update_block_pitch()

        clkrx_params = self.params['clkrx_params']
        io_width_ntr = self.params['io_width_ntr']
        # res_params = self.params['res_params']
        # amp_params = self.params['amp_params']
        # cap_params = self.params['cap_params']

        # top_params = dict(
        #    clkrx_params=layout_params,
        #    nor_params=amp_params,
        # )
        nor_params = self.params['nor_params']

        clkrx_master = self.new_template(params=clkrx_params, temp_cls=ClkAmp, grid=parent_grid)
        top_layer, clkrx_nxblk, clkrx_nyblk = clkrx_master.size
        blk_w, blk_h = self.grid.get_block_size(top_layer)

        nor_master = self.new_template(params=nor_params, temp_cls=clkNorGate)
        nor_nyblk = nor_master.size[2]
        nor_nxblk = nor_master.size[1]

        clkrx_inst = self.add_instance(clkrx_master, 'X0', loc=(0, 0))

        layer_rx, clkrx_x, clkrx_y = clkrx_master.size

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
