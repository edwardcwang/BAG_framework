# -*- coding: utf-8 -*-

"""This module defines implementation of poly resistor templates in generic planar technology.
"""

from typing import Dict, Any, Tuple, List, Optional, Union, TYPE_CHECKING

import math

from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.routing.fill import fill_symmetric_const_space, fill_symmetric_max_density
from bag.layout.template import TemplateBase

from .base import ResTech


if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig
    from bag.layout.routing import RoutingGrid


class ResTechPlanarGeneric(ResTech):
    """Implementation of ResTech for generic planar technologies."""

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        ResTech.__init__(self, config, tech_info)

    def get_res_dimension(self, l, w):
        """Get PO resistor dimension in core and two edge blocks.
        """
        co_w = self.res_config['co_w']
        rpo_co_sp = self.res_config['rpo_co_sp']
        po_co_ency = self.res_config['po_co_enc'][1]
        dpo_wmin, dpo_lmin = self.res_config['dpo_dim_min']
        po_rpo_ext_exact = self.res_config.get('po_rpo_ext_exact', -1)

        if po_rpo_ext_exact >= 0:
            lres = l + 2 * po_rpo_ext_exact
        else:
            lres = l + 2 * (rpo_co_sp + co_w + po_co_ency)
        return w, lres, dpo_wmin, dpo_lmin

    def get_min_res_core_size(self, l, w, res_type, sub_type, threshold, options):
        # type: (int, int, str, str, str, Dict[str, Any]) -> Tuple[int, int]
        """Returns smallest possible resistor core dimension.

        width calculated so we can draw at least 1 dummy OD.
        height calculated so adjacent resistor is DRC clean.
        """
        od_wmin = self.res_config['od_dim_min'][0]
        po_od_sp = self.res_config['po_od_sp']
        po_sp = self.res_config['po_sp']
        imp_od_sp = self.res_config['imp_od_sp']
        imp_ency = self.res_config['imp_enc'][1]
        res_info = self.res_config['info'][res_type]
        od_in_res = res_info['od_in_res']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)
        if od_in_res:
            # OD can be drawn in RES layer
            hblk = lres + po_sp
            wblk = wres + 2 * po_od_sp + od_wmin
        else:
            # OD cannot be drawn in RES layer
            hblk = lres + 2 * (imp_ency + imp_od_sp) + od_wmin
            wblk = wres + po_sp

        return wblk, hblk

    def get_via0_info(self, xc, yc, wres, resolution):
        """Compute resistor CO parameters and metal 1 bounding box."""
        mos_layer_table = self.config['mos_layer_table']
        layer_table = self.config['layer_name']
        via_id_table = self.config['via_id']
        co_w = self.res_config['co_w']
        co_sp = self.res_config['co_sp']
        po_co_encx, po_co_ency = self.res_config['po_co_enc']
        m1_co_encx, m1_co_ency = self.res_config['m1_co_enc']
        num_co = (wres - po_co_encx * 2 + co_sp) // (co_w + co_sp)
        m1_h = co_w + 2 * m1_co_ency

        # get via parameters
        po_name = mos_layer_table['PO']
        m1_name = layer_table[1]
        via_id = via_id_table[(po_name, m1_name)]
        via_params = dict(via_type=via_id, loc=[xc, yc],
                          num_cols=num_co, sp_cols=co_sp,
                          enc1=[po_co_encx, po_co_encx, po_co_ency, po_co_ency],
                          enc2=[m1_co_encx, m1_co_encx, m1_co_ency, m1_co_ency],
                          unit_mode=True)

        varr_w = num_co * (co_w + co_sp) - co_sp
        m1_xl = xc - (varr_w // 2) - m1_co_encx
        m1_xr = m1_xl + 2 * m1_co_encx + varr_w
        m1_yb = yc - m1_h // 2
        m1_yt = m1_yb + m1_h

        m1_box = BBox(m1_xl, m1_yb, m1_xr, m1_yt, resolution, unit_mode=True)
        return via_params, m1_box

    def get_core_info(self,
                      grid,  # type: RoutingGrid
                      width,  # type: int
                      height,  # type: int
                      l,  # type: int
                      w,  # type: int
                      res_type,  # type: str
                      sub_type,  # type: str
                      threshold,  # type: str
                      track_widths,  # type: List[int]
                      track_spaces,  # type: List[Union[float, int]]
                      options,  # type: Dict[str, Any]
                      ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Compute core layout information dictionary.

        This method checks max PO and min OD density rules.
        """
        layer_table = self.config['layer_name']
        od_wmin = self.res_config['od_dim_min'][0]
        od_wmax = self.res_config['od_dim_max'][0]
        od_sp = self.res_config['od_sp']
        od_min_density = self.res_config['od_min_density']
        po_od_sp = self.res_config['po_od_sp']
        po_max_density = self.res_config['po_max_density']
        co_w = self.res_config['co_w']
        rpo_co_sp = self.res_config['rpo_co_sp']
        m1_sp_max = self.res_config['m1_sp_max']
        imp_od_sp = self.res_config['imp_od_sp']
        imp_ency = self.res_config['imp_enc'][1]
        res_info = self.res_config['info'][res_type]
        od_in_res = res_info['od_in_res']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check PO density
        max_res_area = int(width * height * po_max_density)
        if wres * lres > max_res_area:
            return None

        # compute OD fill X coordinates on the left/right edge
        if od_in_res:
            bnd_spx = (width - wres) // 2
            area = 2 * (bnd_spx - po_od_sp)
            lr_od_xloc = fill_symmetric_max_density(area, area, od_wmin, od_wmax, od_sp, offset=-bnd_spx + po_od_sp,
                                                    sp_max=None, fill_on_edge=True, cyclic=False)[0]
            lr_od_h = lres
        else:
            lr_od_xloc = []
            lr_od_h = 0
        # compute OD fill Y coordinates on the top/bottom edge
        bnd_spy = (height - lres) // 2
        if od_in_res:
            area = 2 * (bnd_spy - po_od_sp)
            tb_od_offset = -bnd_spy + po_od_sp
            tb_od_w = wres
        else:
            area = 2 * (bnd_spy - (imp_od_sp + imp_ency))
            tb_od_offset = -bnd_spy + imp_od_sp + imp_ency
            tb_od_w = width - od_sp
        dod_dx = (width - tb_od_w) // 2
        tb_od_xloc = [(dod_dx, width - dod_dx)]
        tb_od_yloc = fill_symmetric_max_density(area, area, od_wmin, od_wmax, od_sp, offset=tb_od_offset,
                                                sp_max=None, fill_on_edge=True, cyclic=False)[0]

        # check OD density
        min_od_area = int(math.ceil(width * height * od_min_density))
        # compute OD area
        od_area = 0
        for od_w, od_intv_list in ((lr_od_h, lr_od_xloc), (tb_od_w, tb_od_yloc)):
            for lower, upper in od_intv_list:
                od_area += od_w * (upper - lower)
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # split into top and bottom dummy locations
        num_dummy_half = -(-len(tb_od_yloc) // 2)
        bot_od_yloc = tb_od_yloc[-num_dummy_half:]
        top_od_yloc = [(a + height, b + height) for a, b in tb_od_yloc[:num_dummy_half]]

        # fill layout info with dummy information
        layout_info = dict(
            width=width,
            height=height,
            lr_od_xloc=lr_od_xloc,
            bot_od_yloc=bot_od_yloc,
            top_od_yloc=top_od_yloc,
            tb_od_xloc=tb_od_xloc,
        )

        # compute port information
        # first, compute M2 routing track location
        xc = width // 2
        rpdmy_yb = height // 2 - l // 2
        rpdmy_yt = rpdmy_yb + l
        bot_yc = rpdmy_yb - rpo_co_sp - co_w // 2
        top_yc = rpdmy_yt + rpo_co_sp + co_w // 2
        bot_layer = self.get_bot_layer()
        bot_pitch = grid.get_track_pitch(bot_layer, unit_mode=True)
        bot_num_tr = height // bot_pitch if height % bot_pitch == 0 else height / bot_pitch
        # first, find M2 tracks such that the top track of this block and the bottom track of
        # the top adjacent block is track_spaces[0] tracks apart.  the actual tracks cannot exceed these.
        m2_w, m2_sp = track_widths[0], track_spaces[0]
        if isinstance(m2_sp, int):
            bot_tr_min = (m2_w + m2_sp + 1) / 2 - 1
        else:
            # for half-integer spacing, we need to round up edge space so that tracks remain centered
            # in the block
            bot_tr_min = (m2_w + m2_sp + 1.5) / 2 - 1
        top_tr_max = bot_num_tr - 1 - bot_tr_min

        # find M2 tracks closest to ports
        top_tr = min(top_tr_max, grid.coord_to_nearest_track(bot_layer, top_yc, half_track=True, mode=1,
                                                             unit_mode=True))
        bot_tr = max(bot_tr_min, grid.coord_to_nearest_track(bot_layer, bot_yc, half_track=True, mode=-1,
                                                             unit_mode=True))

        # get CO/VIA1 parameters, and metal 1 bounding box
        m1_name = layer_table[1]
        m2_name = layer_table[2]
        m2_h = grid.get_track_width(bot_layer, m2_w, unit_mode=True)
        res = grid.resolution
        port_info = []
        for port_name, yc, m2_tr in (('bot', bot_yc, bot_tr), ('top', top_yc, top_tr)):
            via0_params, m1_box = self.get_via0_info(xc, yc, wres, res)
            # get via1 parameters
            m2_yc = grid.track_to_coord(bot_layer, m2_tr, unit_mode=True)
            v1_box = BBox(m1_box.left_unit, m2_yc - m2_h // 2, m1_box.right_unit, m2_yc + m2_h // 2,
                          res, unit_mode=True)
            via1_info = grid.tech_info.get_via_info(v1_box, m1_name, m2_name, 'y')
            m1_box = m1_box.merge(via1_info['bot_box'])
            m2_box = via1_info['top_box']
            via1_params = via1_info['params']
            via1_params['via_type'] = via1_params.pop('id')
            port_info.append((port_name, via0_params, via1_params, m1_box, m2_box))

        # compute fill information
        # compute fill Y coordinate in core block
        # compute fill Y coordinates between ports inside the cell
        m1_w = port_info[0][3].width_unit
        m1_h = port_info[0][3].height_unit
        m1_bot = port_info[0][3]
        m1_top = port_info[1][3]
        m1_bot_yb, m1_bot_yt = m1_bot.bottom_unit, m1_bot.top_unit
        m1_top_yb, m1_top_yt = m1_top.bottom_unit, m1_top.top_unit
        m1_core_mid_y = fill_symmetric_const_space(m1_top_yb - m1_bot_yt, m1_sp_max, m1_h, m1_h, offset=m1_bot_yt)
        # compute fill Y coordinates between ports outside the cell
        m1_core_top_y = fill_symmetric_const_space(m1_bot_yb + height - m1_top_yt, m1_sp_max,
                                                   m1_h, m1_h, offset=m1_top_yt)
        # combine fill Y coordinates together in one list
        fill_len2 = -(-len(m1_core_top_y) // 2)
        m1_core_y = [(a - height, b - height) for (a, b) in m1_core_top_y[-fill_len2:]]
        m1_core_y.append((m1_bot_yb, m1_bot_yt))
        m1_core_y.extend(m1_core_mid_y)
        m1_core_y.append((m1_top_yb, m1_top_yt))
        m1_core_y.extend(m1_core_top_y[:fill_len2])

        # compute fill X coordinate in core block
        m1_xl, m1_xr = m1_bot.left_unit, m1_bot.right_unit
        sp_xl = -width + m1_xr
        sp_xr = m1_xl
        m1_core_x = fill_symmetric_const_space(sp_xr - sp_xl, m1_sp_max, m1_w, m1_w, offset=sp_xl)
        m1_core_x.append((m1_xl, m1_xr))
        m1_core_x.extend(((a + width, b + width) for (a, b) in m1_core_x[:-1]))

        layout_info['port_info'] = port_info
        layout_info['m1_core_x'] = m1_core_x
        layout_info['m1_core_y'] = m1_core_y
        layout_info['m1_w'] = m1_w
        layout_info['m1_h'] = m1_h

        return layout_info

    def get_lr_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         wedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Returns a dictionary of LR edge layout information.

        This method checks:
        1. spacing rules.
        2. PO density rules

        if all these pass, return LR edge layout information dictionary.
        """
        edge_margin = self.res_config['edge_margin']
        imp_encx = self.res_config['imp_enc'][0]
        po_max_density = self.res_config['po_max_density']
        m1_sp_max = self.res_config['m1_sp_max']
        m1_sp_bnd = self.res_config['m1_sp_bnd']

        wcore = core_info['width']
        hcore = core_info['height']
        m1_core_x = core_info['m1_core_x']
        m1_w = core_info['m1_w']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check spacing rule
        # get space between resistor and core boundary
        spx = (wcore - wres) // 2
        if wres_lr > 0:
            # width is given by margin/enclosure/res/res-to-boundary-space
            wedge_min = edge_margin // 2 + imp_encx + wres_lr + spx
            well_xl = wedge - spx - wres_lr - imp_encx
        else:
            # width is given by margin/enclosure to core resistor
            imp_encx_edge = max(imp_encx - spx, 0)
            wedge_min = edge_margin // 2 + imp_encx_edge
            well_xl = wedge - imp_encx_edge

        if wedge < wedge_min:
            return None

        # check PO density rule
        max_res_area = int(wedge * hcore * po_max_density)
        if wres_lr * lres > max_res_area:
            return None

        # if we get here, then all density rules are met
        # compute fill X coordinate in edge block
        m1_sp = m1_core_x[0][0] * 2
        sp_xl = m1_sp_bnd + m1_w
        sp_xr = wedge + m1_core_x[0][0]
        if (sp_xr - sp_xl - m1_w) % 2 != 0:
            sp_xl -= 1
        m1_edge_x = fill_symmetric_const_space(sp_xr - sp_xl, m1_sp_max, m1_w, m1_w, offset=sp_xl)
        if sp_xr - sp_xl >= m1_sp:
            m1_edge_x.insert(0, (m1_sp_bnd, sp_xl))

        # return layout information
        return dict(
            well_xl=well_xl,
            m1_edge_x=m1_edge_x,
        )

    def get_tb_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         hedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Returns a dictionary of TB edge layout information.

        This method checks:
        1. spacing rules.
        2. PO density rules

        if all these pass, return TB edge layout information dictionary.
        """
        edge_margin = self.res_config['edge_margin']
        imp_ency = self.res_config['imp_enc'][1]
        po_max_density = self.res_config['po_max_density']
        m1_sp_max = self.res_config['m1_sp_max']
        m1_sp_bnd = self.res_config['m1_sp_bnd']

        wcore = core_info['width']
        hcore = core_info['height']
        m1_core_y = core_info['m1_core_y']
        m1_h = core_info['m1_h']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check spacing rule
        # get space between resistor and core boundary
        spy = (hcore - lres) // 2
        if lres_tb > 0:
            # width is given by NW space/enclosure/res/res-to-boundary-space
            hedge_min = edge_margin // 2 + imp_ency + lres_tb + spy
            well_yb = hedge - spy - lres_tb - imp_ency
        else:
            imp_ency_edge = max(imp_ency - spy, 0)
            hedge_min = edge_margin // 2 + imp_ency_edge
            well_yb = hedge - imp_ency_edge

        if hedge < hedge_min:
            return None

        # check PO density rule
        max_res_area = int(hedge * wcore * po_max_density)
        if wres * lres_tb > max_res_area:
            return None

        # if we get here, then all density rules are met
        # compute fill Y coordinate in edge block
        m1_sp = m1_core_y[0][0] * 2
        sp_yb = m1_sp_bnd + m1_h
        sp_yt = hedge + m1_core_y[0][0]
        if (sp_yt - sp_yb - m1_h) % 2 != 0:
            sp_yb -= 1
        m1_edge_y = fill_symmetric_const_space(sp_yt - sp_yb, m1_sp_max, m1_h, m1_h, offset=sp_yb)
        if sp_yt - sp_yb >= m1_sp:
            m1_edge_y.insert(0, (m1_sp_bnd, sp_yb))

        # return layout information
        return dict(
            well_yb=well_yb,
            m1_edge_y=m1_edge_y,
        )

    def _draw_dummies(self, template, xintv_list, yintv_list, dx=0, dy=0):
        mos_layer_table = self.config['mos_layer_table']
        od_dummy_lay = mos_layer_table['OD_dummy']
        res = template.grid.resolution

        for xl, xr in xintv_list:
            for yb, yt in yintv_list:
                box = BBox(xl, yb, xr, yt, res, unit_mode=True)
                template.add_rect(od_dummy_lay, box.move_by(dx=dx, dy=dy, unit_mode=True))

    def draw_res_core(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None

        mos_layer_table = self.config['mos_layer_table']
        res_layer_table = self.config['res_layer_table']
        metal_exclude_table = self.config['metal_exclude_table']
        layer_table = self.config['layer_name']
        res_info = self.res_config['info']
        imp_ency = self.res_config['imp_enc'][1]

        grid = template.grid
        res = grid.resolution

        w = layout_info['w']
        l = layout_info['l']
        res_type = layout_info['res_type']
        sub_type = layout_info['sub_type']
        threshold = layout_info['threshold']
        wcore = layout_info['w_core']
        hcore = layout_info['h_core']
        track_widths = layout_info['track_widths']

        core_info = layout_info['core_info']
        lr_od_xloc = core_info['lr_od_xloc']
        top_od_yloc = core_info['top_od_yloc']
        bot_od_yloc = core_info['bot_od_yloc']
        tb_od_xloc = core_info['tb_od_xloc']
        port_info = core_info['port_info']

        res_info = res_info[res_type]
        need_rpo = res_info['need_rpo']
        need_rpdmy = res_info['need_rpdmy']
        od_in_res = res_info['od_in_res']

        xc, yc = wcore // 2, hcore // 2
        wres, lres, _, _ = self.get_res_dimension(l, w)

        # set size and draw implant layers
        implant_layers = self.get_res_imp_layers(res_type, sub_type, threshold)
        arr_box = BBox(0, 0, wcore, hcore, res, unit_mode=True)
        po_yb = yc - lres // 2
        po_yt = yc + lres // 2
        if od_in_res:
            imp_box = arr_box
        else:
            imp_box = BBox(0, po_yb - imp_ency, wcore, po_yt + imp_ency, res, unit_mode=True)
        for lay in implant_layers:
            template.add_rect(lay, imp_box)
        template.array_box = arr_box
        template.prim_bound_box = arr_box
        template.add_cell_boundary(arr_box)

        # draw RPDMY
        po_xl = xc - wres // 2
        po_xr = xc + wres // 2
        rpdmy_yb = yc - l // 2
        rpdmy_yt = rpdmy_yb + l
        if need_rpdmy:
            rpdmy_name = res_layer_table['RPDMY']
            template.add_rect(rpdmy_name, BBox(po_xl, rpdmy_yb, po_xr, rpdmy_yt, res, unit_mode=True))
        if need_rpo:
            # draw RPO
            rpo_name = res_layer_table['RPO']
            template.add_rect(rpo_name, BBox(0, rpdmy_yb, wcore, rpdmy_yt, res, unit_mode=True))

        # draw PO
        po_name = mos_layer_table['PO']
        template.add_rect(po_name, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True))

        # draw vias and ports
        m1_name = layer_table[1]
        bot_layer = self.get_bot_layer()
        for port_name, v0_params, v1_params, m1_box, m2_box in port_info:
            template.add_rect(m1_name, m1_box)
            template.add_via_primitive(**v0_params)
            template.add_via_primitive(**v1_params)
            m2_tr = grid.coord_to_track(bot_layer, m2_box.yc_unit, unit_mode=True)
            template.add_pin(port_name, WireArray(TrackID(bot_layer, m2_tr, width=track_widths[0]),
                                                  m2_box.left, m2_box.right), show=False)

        # draw dummies
        po_y_list = [(po_yb, po_yt)]
        self._draw_dummies(template, lr_od_xloc, po_y_list)
        self._draw_dummies(template, lr_od_xloc, po_y_list, dx=wcore)
        self._draw_dummies(template, tb_od_xloc, bot_od_yloc)
        self._draw_dummies(template, tb_od_xloc, top_od_yloc)

        # draw M1 exclusion layer
        m1_exc_layer = metal_exclude_table[1]
        template.add_rect(m1_exc_layer, arr_box)

        # draw M1 fill
        m1_x_list = core_info['m1_core_x']
        m1_y_list = core_info['m1_core_y']
        for xl, xr in m1_x_list:
            for yb, yt in m1_y_list:
                template.add_rect(m1_name, BBox(xl, yb, xr, yt, res, unit_mode=True))

    def draw_res_boundary(self, template, boundary_type, layout_info, end_mode):
        # type: (TemplateBase, str, Dict[str, Any], bool) -> None

        mos_layer_table = self.config['mos_layer_table']
        res_layer_table = self.config['res_layer_table']
        metal_exclude_table = self.config['metal_exclude_table']
        layer_table = self.config['layer_name']
        res_info = self.res_config['info']
        imp_ency = self.res_config['imp_enc'][1]
        rpo_extx = self.res_config['rpo_extx']

        grid = template.grid
        res = grid.resolution

        w = layout_info['w']
        l = layout_info['l']
        res_type = layout_info['res_type']
        sub_type = layout_info['sub_type']
        threshold = layout_info['threshold']
        w_core = layout_info['w_core']
        h_core = layout_info['h_core']
        w_edge = layout_info['w_edge']
        h_edge = layout_info['h_edge']

        res_info = res_info[res_type]
        need_rpo = res_info['need_rpo']
        od_in_res = res_info['od_in_res']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)
        implant_layers = self.get_res_imp_layers(res_type, sub_type, threshold)
        bnd_spx = (w_core - wres) // 2
        bnd_spy = (h_core - lres) // 2

        core_info = layout_info['core_info']
        edge_lr_info = layout_info['edge_lr_info']
        edge_tb_info = layout_info['edge_tb_info']

        well_xl = edge_lr_info['well_xl']
        well_yb = edge_tb_info['well_yb']

        core_lr_od_xloc = core_info['lr_od_xloc']
        core_top_od_yloc = core_info['top_od_yloc']
        core_bot_od_yloc = core_info['bot_od_yloc']

        #  get bounding box/implant coordinates, draw RPDMY/RPO, and draw dummies.
        po_xl = po_xr = po_yb = po_yt = None
        if boundary_type == 'lr':
            # set implant Y coordinates to 0
            well_yb = 0
            # get bounding box and PO coordinates
            bnd_box = BBox(0, 0, w_edge, h_core, res, unit_mode=True)
            if wres_lr > 0:
                po_xr = w_edge - bnd_spx
                po_xl = po_xr - wres_lr
                po_yb, po_yt = bnd_spy, bnd_spy + lres
                # draw bottom/top edge dummies
                po_x_list = [(po_xl, po_xr)]
                self._draw_dummies(template, po_x_list, core_bot_od_yloc)
                self._draw_dummies(template, po_x_list, core_top_od_yloc)
            if need_rpo:
                # draw RPO in left/right edge block
                rpo_yb = h_core // 2 - l // 2
                rpo_yt = rpo_yb + l
                rpo_name = res_layer_table['RPO']
                rpo_xl = w_edge + bnd_spx - rpo_extx
                template.add_rect(rpo_name, BBox(rpo_xl, rpo_yb, w_edge, rpo_yt, res, unit_mode=True))
            m1_x_list = edge_lr_info['m1_edge_x']
            m1_y_list = core_info['m1_core_y']
        elif boundary_type == 'tb':
            # set implant X coordinates to 0
            well_xl = 0
            # get bounding box and PO coordinates
            bnd_box = BBox(0, 0, w_core, h_edge, res, unit_mode=True)
            if lres_tb > 0:
                po_yt = h_edge - bnd_spy
                po_yb = po_yt - lres_tb
                po_xl, po_xr = bnd_spx, bnd_spx + wres
                # draw bottom edge dummies
                po_y_list = [(po_yb, po_yt)]
                self._draw_dummies(template, core_lr_od_xloc, po_y_list)
                self._draw_dummies(template, core_lr_od_xloc, po_y_list, dx=w_core)
            m1_x_list = core_info['m1_core_x']
            m1_y_list = edge_tb_info['m1_edge_y']
        else:
            # get bounding box and PO coordinates
            bnd_box = BBox(0, 0, w_edge, h_edge, res, unit_mode=True)
            if wres_lr > 0 and lres_tb > 0:
                po_xr = w_edge - bnd_spx
                po_xl = po_xr - wres_lr
                po_yt = h_edge - bnd_spy
                po_yb = po_yt - lres_tb
            m1_x_list = edge_lr_info['m1_edge_x']
            m1_y_list = edge_tb_info['m1_edge_y']

        # draw DPO
        if wres_lr > 0 and lres_tb > 0:
            dpo_name = mos_layer_table['PO_dummy']
            template.add_rect(dpo_name, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True))

        # draw implant layers
        if not od_in_res:
            if boundary_type == 'lr':
                po_yb, po_yt = bnd_spy, bnd_spy + lres
                imp_box = BBox(well_xl, po_yb - imp_ency, bnd_box.right_unit, po_yt + imp_ency, res, unit_mode=True)
            else:
                imp_box = BBox.get_invalid_bbox()
        else:
            imp_box = BBox(well_xl, well_yb, bnd_box.right_unit, bnd_box.top_unit, res, unit_mode=True)

        if imp_box.is_physical():
            for lay in implant_layers:
                template.add_rect(lay, imp_box)

        # set bounding box
        template.prim_bound_box = bnd_box
        template.array_box = bnd_box
        template.add_cell_boundary(bnd_box)

        # draw M1 exclusion layer
        m1_exc_layer = metal_exclude_table[1]
        template.add_rect(m1_exc_layer, bnd_box)

        # draw M1 fill
        m1_lay = layer_table[1]
        for xl, xr in m1_x_list:
            for yb, yt in m1_y_list:
                template.add_rect(m1_lay, BBox(xl, yb, xr, yt, res, unit_mode=True))
