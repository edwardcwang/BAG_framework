# -*- coding: utf-8 -*-

"""This module defines implementation of poly resistor templates in generic planar technology.
"""

from typing import Dict, Any, Tuple, List, Optional, Union, TYPE_CHECKING

import abc
import math

from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.routing.fill import fill_symmetric_const_space, fill_symmetric_max_density
from bag.layout.template import TemplateBase

from .base import ResTech


if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig
    from bag.layout.routing import RoutingGrid
    from ..analog_mos.core import MOSTech


class ResTechFinfetBase(ResTech, metaclass=abc.ABCMeta):
    """Implementation of ResTech for generic planar technologies."""

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        ResTech.__init__(self, config, tech_info)
        self.mos_tech = tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech

    @abc.abstractmethod
    def get_port_info(self, xc, yc, wres, port_yb, port_yt, resolution):
        # type: (int, int, int, int, int, float) -> Tuple[List[Tuple[Tuple[str, str], BBox]], List[Dict[str, Any]]]
        """Calculate port geometry information.

        Parameters
        ----------
        xc : int
            resistor port center X coordinate
        yc : int
            resistor center Y coordinate.
        wres : int
            width of the resistor.
        port_yb : int
            port wire bottom Y coordinate.
        port_yb : int
            port wire top Y coordinate.
        resolution : float
            the grid resolution.

        Returns
        -------
        rect_list : List[Tuple[Tuple[str, str], BBox]]
            list of rectangles to create, in (layer, bbox) tuple form.
        via_list : List[Dict[str, Any]]
            list of vias to create.

        """
        return [], []

    def get_res_dimension(self, l, w):
        """Get PO resistor dimension in core and two edge blocks.
        """
        mp_h = self.res_config['mp_h']
        po_mp_exty = self.res_config['po_mp_exty']

        lres = l + 2 * (mp_h + po_mp_exty)
        lres_tb = min(w, lres)
        return w, lres, w, lres_tb

    def get_min_res_core_size(self, l, w, res_type, sub_type, threshold, options):
        # type: (int, int, str, str, str, Dict[str, Any]) -> Tuple[int, int]
        """Returns smallest possible resistor core dimension.

        width calculated so we can draw at least 1 dummy OD with 2 fingers.
        height calculated so adjacent resistor is DRC clean.
        """
        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        po_res_sp = self.res_config['po_res_sp']
        mp_res_sp = self.res_config['mp_res_sp']
        res_spy = self.res_config['res_spy']

        po_res_spx = max(po_res_sp, mp_res_sp - po_lch // 2)

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # at least 2 finger dummy transistor between resistors
        res_sp = po_lch + po_pitch + 2 * po_res_spx

        wcore = wres + res_sp
        hcore = lres + res_spy

        return wcore, hcore

    @classmethod
    def _compute_od_y_loc(cls, od_fin_loc, fin_pitch, fin_h, fin_offset):
        """Convert dummy OD fin location list to Y coordinate.

        fin_offset is the Y coordinate of fin 0 in the given fin number list.
        """
        new_loc = []
        fin_h2 = fin_h // 2
        for fin_start, fin_stop in od_fin_loc:
            yb = fin_offset + fin_start * fin_pitch - fin_h2
            yt = fin_offset + (fin_stop - 1) * fin_pitch + fin_h2
            new_loc.append((yb, yt))
        return new_loc

    def get_core_info(self,  # type: ResTechFinfetBase
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
        res = grid.resolution

        nfin_min, nfin_max = self.res_config['od_fill_w']
        po_od_ext = self.res_config['po_od_ext']
        po_spx = self.res_config['po_spx']
        po_spy = self.res_config['po_spy']
        po_res_sp = self.res_config['po_res_sp']
        mp_res_sp = self.res_config['mp_res_sp']
        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        mp_h = self.res_config['mp_h']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']
        m1_sp_max = self.res_config['m1_sp_max']

        po_res_spx = max(po_res_sp, mp_res_sp - po_lch // 2)

        fin_h = self.mos_tech.mos_config['fin_h']
        fin_p = self.mos_tech.mos_config['mos_pitch']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check RH_TN density
        max_res_area = int(width * height * res_max_density)
        if wres * lres > max_res_area:
            return None

        # Compute dummy Y coordinates
        # compute dummy OD Y separation in number of fins
        od_sp = po_spy + po_od_ext * 2
        # when two ODs are N fin pitches apart, the actual OD spacing is N * fin_pitch - fin_h
        od_sp = -(-(od_sp + fin_h) // fin_p)
        # compute OD Y coordinates for left/right edge.
        h_core_nfin = height // fin_p
        core_lr_od_loc = fill_symmetric_max_density(h_core_nfin, h_core_nfin, nfin_min, nfin_max, od_sp,
                                                    fill_on_edge=False, cyclic=True)[0]
        # compute OD Y coordinates for top/bottom edge.
        # compute fin offset for top edge dummies
        bnd_spy = (height - lres) // 2
        top_dummy_bnd = bnd_spy + lres + po_res_sp + po_od_ext
        # find the fin pitch index of the lower bound of the empty space.
        pitch_index = -(-(top_dummy_bnd - fin_p2 + fin_h2) // fin_p)
        tot_space = 2 * (h_core_nfin - pitch_index)
        core_tb_od_loc = fill_symmetric_max_density(tot_space, tot_space, nfin_min, nfin_max, od_sp,
                                                    fill_on_edge=True, cyclic=False)[0]

        # compute dummy number of fingers for left/right edge
        bnd_spx = (width - wres) // 2
        avail_sp = bnd_spx * 2 - po_res_spx * 2
        core_lr_fg = (avail_sp - po_lch) // po_pitch + 1
        # compute dummy number of fingers for top/bottom edge
        core_lr_dum_w = po_lch + (core_lr_fg - 1) * po_pitch
        avail_sp = width - core_lr_dum_w - 2 * po_spx
        core_tb_fg = (avail_sp - po_lch) // po_pitch + 1
        core_tb_dum_w = po_lch + (core_tb_fg - 1) * po_pitch

        # check OD density
        min_od_area = int(math.ceil(width * height * od_min_density))
        # compute OD area
        od_area = 0
        for od_w, od_fin_list in ((core_lr_dum_w, core_lr_od_loc), (core_tb_dum_w, core_tb_od_loc)):
            for fin_start, fin_stop in od_fin_list:
                od_area += od_w * ((fin_stop - fin_start - 1) * fin_p + fin_h)
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # convert dummy fin location to Y coordinates
        core_lr_od_loc = self._compute_od_y_loc(core_lr_od_loc, fin_p, fin_h, fin_p2)
        # split the top/bottom dummies into halves
        num_dummy_half = -(-len(core_tb_od_loc) // 2)
        # top dummy locations
        top_offset = pitch_index * fin_p + fin_p2
        core_top_od_loc = self._compute_od_y_loc(core_tb_od_loc[:num_dummy_half], fin_p, fin_h, top_offset)
        core_top_po_bnd = (height - bnd_spy + po_res_sp, height + bnd_spy - po_res_sp)
        # bottom dummy locations
        core_bot_od_loc = self._compute_od_y_loc(core_tb_od_loc[-num_dummy_half:], fin_p, fin_h, top_offset - height)
        core_bot_po_bnd = (-bnd_spy + po_res_sp, bnd_spy - po_res_sp)

        # fill layout info with dummy information
        layout_info = dict(
            width=width,
            height=height,
            lr_dum_w=core_lr_dum_w,
            tb_dum_w=core_tb_dum_w,
            lr_od_loc=(core_lr_od_loc, None),
            top_od_loc=(core_top_od_loc, core_top_po_bnd),
            bot_od_loc=(core_bot_od_loc, core_bot_po_bnd),
            lr_fg=core_lr_fg,
            tb_fg=core_tb_fg,
        )

        # compute port information
        # first, compute M2 routing track location
        xc = width // 2
        rpdmy_yb = height // 2 - l // 2
        rpdmy_yt = rpdmy_yb + l
        bot_yc = rpdmy_yb - mp_h // 2
        top_yc = rpdmy_yt + mp_h // 2
        bot_layer = self.get_bot_layer()
        bot_pitch = grid.get_track_pitch(bot_layer, unit_mode=True)
        bot_num_tr = height // bot_pitch
        # first, find M2 tracks such that the top track of this block and the bottom track of
        # the top adjacent block is track_spaces[0] tracks apart.  the actual tracks cannot exceed these.
        top_tr_max = bot_num_tr - (track_widths[0] + track_spaces[0] + 1) / 2
        bot_tr_min = (track_widths[0] + track_spaces[0] + 1) / 2 - 1
        # find M2 tracks closest to ports
        top_tr = min(top_tr_max, grid.coord_to_nearest_track(bot_layer, top_yc, half_track=True, mode=1,
                                                             unit_mode=True))
        bot_tr = max(bot_tr_min, grid.coord_to_nearest_track(bot_layer, bot_yc, half_track=True, mode=-1,
                                                             unit_mode=True))

        # get VIA0/VIA1 parameters, and metal 1 bounding box
        port_info = []
        for port_name, yc, m2_tr in (('bot', bot_yc, bot_tr), ('top', top_yc, top_tr)):
            port_yb, port_yt = grid.get_wire_bounds(bot_layer, m2_tr, track_widths[0], unit_mode=True)
            rect_list, via_list = self.get_port_info(xc, yc, wres, port_yb, port_yt, res)
            port_info.append((port_name, rect_list, via_list))

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
        m1_core_top_y = fill_symmetric_const_space(m1_bot_yb + height - m1_top_yt, m1_sp_max, m1_h, m1_h,
                                                   offset=m1_top_yt)
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

    def get_lr_edge_info(self,  # type: ResTechFinfetBase
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
        nw_sp = self.res_config['nw_sp']
        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        fill_fg_min = self.res_config['fill_fg_min']
        po_res_sp = self.res_config['po_res_sp']
        mp_res_sp = self.res_config['mp_res_sp']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']
        po_spx = self.res_config['po_spx']
        finfet_od_extx = self.res_config['finfet_od_extx']
        imp_od_enc = self.res_config['imp_od_enc']
        nw_od_encx = self.res_config['nw_od_encx']
        rtop_od_enc = self.res_config['rtop_od_enc']
        m1_sp_max = self.res_config['m1_sp_max']
        m1_sp_bnd = self.res_config['m1_sp_bnd']

        po_res_spx = max(po_res_sp, mp_res_sp - po_lch // 2)

        wcore = core_info['width']
        hcore = core_info['height']
        core_lr_dum_w = core_info['lr_dum_w']
        core_lr_od_loc = core_info['lr_od_loc'][0]
        core_top_od_loc = core_info['top_od_loc'][0]
        core_bot_od_loc = core_info['bot_od_loc'][0]
        m1_core_x = core_info['m1_core_x']
        m1_w = core_info['m1_w']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check spacing rule
        # get space between resistor and core boundary
        spx = (wcore - wres) // 2
        # width of dummy transistor in left/right edge
        dum_w = po_lch + po_pitch * (fill_fg_min - 1)
        # width is given by NW space/dummy/dummy-to-res-space/res/res-to-boundary-space
        wedge_min = nw_sp // 2 + nw_od_encx + dum_w + po_res_spx + wres_lr + spx

        if wedge < wedge_min:
            return None

        # check RH_TN density rule
        max_res_area = int(wedge * hcore * res_max_density)
        if wres_lr * lres > max_res_area:
            return None

        # compute dummy number of fingers for left edge of LREdge block, and also the center X coordinate
        avail_sp_xl = nw_sp // 2 + nw_od_encx
        avail_sp_xr = wedge - spx - wres_lr - po_res_spx
        avail_sp = avail_sp_xr - avail_sp_xl
        edge_lr_fg = (avail_sp - po_lch) // po_pitch + 1
        edge_lr_dum_w = po_lch + (edge_lr_fg - 1) * po_pitch
        edge_lr_xc = avail_sp_xl + edge_lr_dum_w // 2
        edge_lr_dum_xl = avail_sp_xl
        # compute dummy number of fingers for top/bottom edge of LREdge block, and also the center X coordinate
        avail_sp_xl = avail_sp_xl + edge_lr_dum_w + po_spx
        avail_sp_xr = wedge - core_lr_dum_w // 2 - po_spx
        edge_tb_xc = (avail_sp_xl + avail_sp_xr) // 2
        avail_sp = avail_sp_xr - avail_sp_xl
        edge_tb_fg = (avail_sp - po_lch) // po_pitch + 1
        edge_tb_dum_w = po_lch + (edge_tb_fg - 1) * po_pitch
        # compute total OD area
        od_area = 0
        for od_w, yloc_list in ((edge_lr_dum_w, core_lr_od_loc), (edge_tb_dum_w, core_top_od_loc),
                                (edge_tb_dum_w, core_bot_od_loc)):
            for yb, yt in yloc_list:
                # we do not add OD with bottom Y coordinates less than 0, because of arraying over-counting issues
                if yb >= 0:
                    od_area += od_w * (yt - yb)
        # check OD density rule
        min_od_area = int(math.ceil(hcore * wedge * od_min_density))
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # compute fill X coordinate in edge block
        sp_xl = m1_sp_bnd + m1_w
        sp_xr = wedge + m1_core_x[0][0]
        m1_edge_x = fill_symmetric_const_space(sp_xr - sp_xl, m1_sp_max, m1_w, m1_w, offset=sp_xl)
        m1_edge_x.insert(0, (m1_sp_bnd, sp_xl))

        # return layout information
        return dict(
            lr_fg=edge_lr_fg,
            tb_fg=edge_tb_fg,
            lr_xc=edge_lr_xc,
            tb_xc=edge_tb_xc,
            fb_xl=edge_lr_dum_xl - finfet_od_extx,
            imp_xl=edge_lr_dum_xl - imp_od_enc,
            well_xl=edge_lr_dum_xl - nw_od_encx,
            rtop_xl=edge_lr_dum_xl - rtop_od_enc,
            vt_xl=edge_lr_dum_xl - imp_od_enc,
            m1_edge_x=m1_edge_x,
        )

    def get_tb_edge_info(self,  # type: ResTechFinfetBase
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
        nfin_min, nfin_max = self.res_config['od_fill_w']
        po_od_ext = self.res_config['po_od_ext']
        nw_sp = self.res_config['nw_sp']
        po_res_sp = self.res_config['po_res_sp']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']
        po_spy = self.res_config['po_spy']
        vt_po_ext = self.res_config['vt_po_ext']
        finfet_od_exty = self.res_config['finfet_od_exty']
        imp_od_enc = self.res_config['imp_od_enc']
        nw_od_ency = self.res_config['nw_od_ency']
        rtop_od_enc = self.res_config['rtop_od_enc']
        m1_sp_max = self.res_config['m1_sp_max']
        m1_sp_bnd = self.res_config['m1_sp_bnd']

        fin_h = self.mos_tech.mos_config['fin_h']
        fin_p = self.mos_tech.mos_config['mos_pitch']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        wcore = core_info['width']
        hcore = core_info['height']
        core_lr_dum_w = core_info['lr_dum_w']
        core_tb_dum_w = core_info['tb_dum_w']
        core_lr_od_loc = core_info['lr_od_loc'][0]
        m1_core_y = core_info['m1_core_y']
        m1_h = core_info['m1_h']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # compute dummy OD Y separation in number of fins
        od_sp = po_spy + po_od_ext * 2
        # when two ODs are N fin pitches apart, the actual OD spacing is N * fin_pitch - fin_h
        od_sp = -(-(od_sp + fin_h) // fin_p)

        # check RH_TN density rule
        max_res_area = int(wcore * hedge * res_max_density)
        if wres * lres_tb > max_res_area:
            return None

        # check spacing rule, which just means we can draw dummy transistors below RH
        # get space between resistor and core boundary
        spy = (hcore - lres) // 2
        # find bottom edge OD locations
        # compute OD Y coordinate for bottom edge of TBEdge block
        bot_dummy_bnd = nw_sp // 2 + nw_od_ency
        bot_pitch_index = -(-(bot_dummy_bnd - fin_p2 + fin_h2) // fin_p)
        top_dummy_bnd = hedge - spy - lres_tb - po_res_sp - po_od_ext
        top_pitch_index = (top_dummy_bnd - fin_p2 - fin_h2) // fin_p
        tot_space = top_pitch_index - bot_pitch_index + 1
        edge_bot_od_loc = fill_symmetric_max_density(tot_space, tot_space, nfin_min, nfin_max, od_sp,
                                                     fill_on_edge=True, cyclic=False)[0]
        # compute fin 0 offset and convert fin location to Y coordinates
        fin_offset = bot_pitch_index * fin_p + fin_p2
        edge_bot_od_loc = self._compute_od_y_loc(edge_bot_od_loc, fin_p, fin_h, fin_offset)
        if not edge_bot_od_loc:
            # we cannot draw any bottom OD
            return None
        edge_tb_dum_yb = edge_bot_od_loc[0][0]

        # compute OD Y coordinate for left/right edge of TBEdge block.
        # find the fin pitch index of the lower bound of empty space.
        bot_pitch_index = top_pitch_index + od_sp
        # find the fin pitch index of the upper bound of empty space.
        adj_top_od_yb = hedge + core_lr_od_loc[0][0]
        top_pitch_index = (adj_top_od_yb - fin_p2 + fin_h2) // fin_p - od_sp
        # compute total space and fill
        tot_space = top_pitch_index - bot_pitch_index + 1
        edge_lr_od_loc = fill_symmetric_max_density(tot_space, tot_space, nfin_min, nfin_max, od_sp,
                                                    fill_on_edge=True, cyclic=False)[0]
        # compute fin 0 offset and convert fin location to Y coordinates
        fin_offset = bot_pitch_index * fin_p + fin_p2
        edge_lr_od_loc = self._compute_od_y_loc(edge_lr_od_loc, fin_p, fin_h, fin_offset)
        edge_lr_po_bnd = (edge_bot_od_loc[-1][-1] + po_od_ext + po_spy,
                          adj_top_od_yb - po_od_ext - po_spy)

        # compute total OD area
        od_area = 0
        for od_w, yloc_list in ((core_lr_dum_w + core_tb_dum_w, edge_bot_od_loc), (core_lr_dum_w, edge_lr_od_loc)):
            for yb, yt in yloc_list:
                od_area += od_w * (yt - yb)
        # check OD density rule
        min_od_area = int(math.ceil(wcore * hedge * od_min_density))
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # compute fill Y coordinate in edge block
        sp_yb = m1_sp_bnd + m1_h
        sp_yt = hedge + m1_core_y[0][0]
        m1_edge_y = fill_symmetric_const_space(sp_yt - sp_yb, m1_sp_max, m1_h, m1_h, offset=sp_yb)
        m1_edge_y.insert(0, (m1_sp_bnd, sp_yb))

        # return layout information
        return dict(
            lr_od_loc=(edge_lr_od_loc, edge_lr_po_bnd),
            bot_od_loc=(edge_bot_od_loc, None),
            fb_yb=edge_tb_dum_yb - finfet_od_exty,
            imp_yb=edge_tb_dum_yb - imp_od_enc,
            well_yb=edge_tb_dum_yb - nw_od_ency,
            rtop_yb=edge_tb_dum_yb - rtop_od_enc,
            vt_yb=edge_tb_dum_yb - po_od_ext - vt_po_ext,
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
        implant_layers = self.get_res_imp_layers(res_type, sub_type)
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
        w_core = layout_info['w_core']
        h_core = layout_info['h_core']
        w_edge = layout_info['w_edge']
        h_edge = layout_info['h_edge']

        res_info = res_info[res_type]
        need_rpo = res_info['need_rpo']
        od_in_res = res_info['od_in_res']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)
        implant_layers = self.get_res_imp_layers(res_type, sub_type)
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
