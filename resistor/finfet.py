# -*- coding: utf-8 -*-

"""This module defines implementation of poly resistor templates in generic planar technology.
"""

from typing import Dict, Any, Tuple, List, Optional, Union, TYPE_CHECKING

import abc
import math

from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.routing.fill import fill_symmetric_max_density, \
    fill_symmetric_min_density_info, fill_symmetric_interval
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
    def get_port_info(self,  # type: ResTechFinfetBase
                      xc,  # type: int
                      yc,  # type: int
                      wres,  # type: int
                      port_yb,  # type: int
                      port_yt,  # type: int
                      resolution,  # type: float
                      ):
        # type: (...) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
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
        rect_list : List[Dict[str, Any]]
            list of rectangle information dictionaries.  Must have the following fields:

            do_fill : bool
                True to draw fill geometries for this layer.
            layer : Tuple[str, str]
                the layer name.
            exc_layer : Tuple[str, str]
                the exclusion layer name.
            sp_max : int
                maximum space between fill blocks.
            sp_bnd : int
                space of fill blocks to boundary.
            bbox : BBox
                the bounding box of the rectangle.
        via_list : List[Dict[str, Any]]
            list of vias parameters.

        """
        return [], []

    def get_res_dimension(self, l, w):
        # type: (int, int) -> Tuple[int, int, int, int]
        """Get PO resistor dimension in core and two edge blocks.
        """
        mp_h = self.res_config['mp_h']
        res_mp_exty = self.res_config['res_mp_exty']

        lres = l + 2 * (mp_h + res_mp_exty)
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
        po_res_spx = self.res_config['po_res_spx']
        res_spy = self.res_config['res_spy']

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

        nfin_min, nfin_max = self.res_config['od_fill_h']
        po_od_exty = self.res_config['po_od_exty']
        po_spx = self.res_config['po_spx']
        po_spy = self.res_config['po_spy']
        po_res_spx = self.res_config['po_res_spx']
        po_res_spy = self.res_config['po_res_spy']
        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        mp_h = self.res_config['mp_h']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']

        fin_h = self.mos_tech.mos_config['fin_h']
        fin_p = self.mos_tech.mos_config['mos_pitch']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check resistor density
        max_res_area = int(width * height * res_max_density)
        if wres * lres > max_res_area:
            return None

        # Compute dummy Y coordinates
        # compute dummy OD Y separation in number of fins
        od_sp = po_spy + po_od_exty * 2
        # when two ODs are N fin pitches apart, the actual OD spacing is N * fin_pitch - fin_h
        od_sp = -(-(od_sp + fin_h) // fin_p)
        # compute OD Y coordinates for left/right edge.
        h_core_nfin = height // fin_p
        core_lr_od_loc = fill_symmetric_max_density(h_core_nfin, h_core_nfin, nfin_min, nfin_max,
                                                    od_sp, fill_on_edge=False, cyclic=True)[0]
        # compute OD Y coordinates for top/bottom edge.
        # compute fin offset for top edge dummies
        bnd_spy = (height - lres) // 2
        top_dummy_bnd = bnd_spy + lres + po_res_spy + po_od_exty
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
        core_top_od_loc = self._compute_od_y_loc(core_tb_od_loc[:num_dummy_half], fin_p, fin_h,
                                                 top_offset)
        core_top_po_bnd = (height - bnd_spy + po_res_spy, height + bnd_spy - po_res_spy)
        # bottom dummy locations
        core_bot_od_loc = self._compute_od_y_loc(core_tb_od_loc[-num_dummy_half:], fin_p, fin_h,
                                                 top_offset - height)
        core_bot_po_bnd = (-bnd_spy + po_res_spy, bnd_spy - po_res_spy)

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
        # first, compute port track location
        xc = width // 2
        rpdmy_yb = height // 2 - l // 2
        rpdmy_yt = rpdmy_yb + l
        bot_yc = rpdmy_yb - mp_h // 2
        top_yc = rpdmy_yt + mp_h // 2
        bot_layer = self.get_bot_layer()
        bot_pitch = grid.get_track_pitch(bot_layer, unit_mode=True)
        bot_num_tr = height // bot_pitch
        # first, find port tracks such that the top track of this block and the bottom track of
        # the top adjacent block is track_spaces[0] tracks apart.
        # the actual tracks cannot exceed these.
        top_tr_max = bot_num_tr - (track_widths[0] + track_spaces[0] + 1) / 2
        bot_tr_min = (track_widths[0] + track_spaces[0] + 1) / 2 - 1
        # find port tracks closest to ports
        top_tr = min(top_tr_max, grid.coord_to_nearest_track(bot_layer, top_yc, half_track=True,
                                                             mode=1, unit_mode=True))
        bot_tr = max(bot_tr_min, grid.coord_to_nearest_track(bot_layer, bot_yc, half_track=True,
                                                             mode=-1, unit_mode=True))

        # get port information
        port_info = []
        for port_name, yc, m2_tr in (('bot', bot_yc, bot_tr), ('top', top_yc, top_tr)):
            port_yb, port_yt = grid.get_wire_bounds(bot_layer, m2_tr, track_widths[0],
                                                    unit_mode=True)
            rect_list, via_list = self.get_port_info(xc, yc, wres, port_yb, port_yt, res)
            port_info.append((port_name, rect_list, via_list))

        # compute fill information
        # compute fill Y coordinate in core block
        bot_rect_list = port_info[0][1]
        top_rect_list = port_info[1][1]
        fill_info = []
        for bot_rect_info, top_rect_info in zip(bot_rect_list, top_rect_list):
            if bot_rect_info['do_fill']:
                layer = bot_rect_info['layer']
                exc_layer = bot_rect_info['exc_layer']
                density = bot_rect_info['density']
                sp_min = bot_rect_info['sp_min']
                sp_max = bot_rect_info['sp_max']
                sp_bnd = bot_rect_info['sp_bnd']
                bot_box = bot_rect_info['bbox']
                top_box = top_rect_info['bbox']
                density_1d = density ** 0.5
                w, h = bot_box.width_unit, bot_box.height_unit
                bot_yb, bot_yt = bot_box.bottom_unit, bot_box.top_unit
                top_yb, top_yt = top_box.bottom_unit, top_box.top_unit
                # compute fill Y coordinates between ports inside the cell
                area = top_yb - bot_yt
                tarea = int(math.ceil(area * density_1d))
                mid_info = fill_symmetric_min_density_info(area, tarea, h, h, sp_min, sp_max=sp_max,
                                                           fill_on_edge=False)
                core_mid_y = fill_symmetric_interval(*mid_info[0][2], offset=bot_yt,
                                                     invert=mid_info[1])[0]
                # compute fill Y coordinates between ports outside the cell
                area = bot_yb + height - top_yt
                tarea = int(math.ceil(area * density_1d))
                top_info = fill_symmetric_min_density_info(area, tarea, h, h, sp_min, sp_max=sp_max,
                                                           fill_on_edge=False)
                core_top_y = fill_symmetric_interval(*top_info[0][2], offset=top_yt,
                                                     invert=top_info[1])[0]

                # combine fill Y coordinates together in one list
                fill_len2 = -(-len(core_top_y) // 2)
                core_y = [(a - height, b - height) for (a, b) in core_top_y[-fill_len2:]]
                core_y.append((bot_yb, bot_yt))
                core_y.extend(core_mid_y)
                core_y.append((top_yb, top_yt))
                core_y.extend(core_top_y[:fill_len2])
                # compute fill X coordinate in core block
                xl, xr = bot_box.left_unit, bot_box.right_unit
                sp_xl = -width + xr
                sp_xr = xl

                area = sp_xr - sp_xl
                tarea = int(math.ceil(area * density_1d))
                x_info = fill_symmetric_min_density_info(area, tarea, w, w, sp_min,
                                                         sp_max=sp_max, fill_on_edge=False)
                core_x = fill_symmetric_interval(*x_info[0][2], offset=sp_xl, invert=x_info[1])[0]

                core_x.append((xl, xr))
                core_x.extend(((a + width, b + width) for (a, b) in core_x[:-1]))

                fill_info.append((layer, exc_layer, w, h, core_x, core_y, density, sp_min,
                                  sp_max, sp_bnd))

        layout_info['port_info'] = port_info
        layout_info['fill_info'] = fill_info
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
        well_end_mode = options.get('well_end_mode', 3)
        edge_margin = self.res_config['edge_margin']
        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        od_fg_min = self.res_config['od_fg_min']
        po_res_spx = self.res_config['po_res_spx']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']
        po_spx = self.res_config['po_spx']
        finfet_od_extx = self.res_config['finfet_od_extx']
        imp_od_encx = self.res_config['imp_od_encx']
        rtop_od_encx = self.res_config['rtop_od_encx']

        wcore = core_info['width']
        hcore = core_info['height']
        core_lr_dum_w = core_info['lr_dum_w']
        core_lr_od_loc = core_info['lr_od_loc'][0]
        core_top_od_loc = core_info['top_od_loc'][0]
        core_bot_od_loc = core_info['bot_od_loc'][0]
        core_fill_info = core_info['fill_info']
        if well_end_mode & 2 == 0:
            edge_margin = self.res_config.get('edge_margin_nowell', edge_margin)

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # check spacing rule
        # get space between resistor and core boundary
        spx = (wcore - wres) // 2
        # width of dummy transistor in left/right edge
        dum_w = po_lch + po_pitch * (od_fg_min - 1)
        # width is given by NW space/dummy/dummy-to-res-space/res/res-to-boundary-space
        wedge_min = edge_margin + imp_od_encx + dum_w + po_res_spx + wres_lr + spx

        if wedge < wedge_min:
            return None

        # check resistor density rule
        max_res_area = int(wedge * hcore * res_max_density)
        if wres_lr * lres > max_res_area:
            return None

        # number of dummy fingers for left edge of LREdge, also the center X coordinate
        avail_sp_xl = edge_margin + imp_od_encx
        avail_sp_xr = wedge - spx - wres_lr - po_res_spx
        avail_sp = avail_sp_xr - avail_sp_xl
        edge_lr_fg = (avail_sp - po_lch) // po_pitch + 1
        edge_lr_dum_w = po_lch + (edge_lr_fg - 1) * po_pitch
        edge_lr_xc = avail_sp_xl + edge_lr_dum_w // 2
        edge_lr_dum_xl = avail_sp_xl
        # number of dummy fingers for top/bottom edge of LREdge, also the center X coordinate
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
                # we do not add OD with bottom Y coordinates less than 0,
                # because of arraying over-counting issues
                if yb >= 0:
                    od_area += od_w * (yt - yb)
        # check OD density rule
        min_od_area = int(math.ceil(hcore * wedge * od_min_density))
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # compute fill X coordinate in edge block
        fill_edge_x_list = []
        for _, _, w, h, core_x, core_y, density, sp_min, sp_max, sp_bnd in core_fill_info:
            sp_xl = sp_bnd + w
            sp_xr = wedge + core_x[0][0]

            area = sp_xr - sp_xl
            tarea = int(math.ceil(area * density ** 0.5))
            c_info = fill_symmetric_min_density_info(area, tarea, w, w, sp_min, sp_max=sp_max,
                                                     fill_on_edge=False)
            edge_x = fill_symmetric_interval(*c_info[0][2], offset=sp_xl, invert=c_info[1])[0]
            edge_x.insert(0, (sp_bnd, sp_xl))
            fill_edge_x_list.append(edge_x)

        # return layout information
        well_xl = edge_lr_dum_xl - imp_od_encx
        return dict(
            lr_fg=edge_lr_fg,
            tb_fg=edge_tb_fg,
            lr_xc=edge_lr_xc,
            tb_xc=edge_tb_xc,
            fb_xl=edge_lr_dum_xl - finfet_od_extx,
            imp_xl=well_xl,
            well_xl=well_xl,
            rtop_xl=edge_lr_dum_xl - rtop_od_encx,
            fill_edge_x_list=fill_edge_x_list,
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
        well_end_mode = options.get('well_end_mode', 3)

        nfin_min, nfin_max = self.res_config['od_fill_h']
        po_od_exty = self.res_config['po_od_exty']
        edge_margin = self.res_config['edge_margin']
        po_res_spy = self.res_config['po_res_spy']
        res_max_density = self.res_config['res_max_density']
        od_min_density = self.res_config['od_min_density']
        po_spy = self.res_config['po_spy']
        finfet_od_exty = self.res_config['finfet_od_exty']
        imp_po_ency = self.res_config['imp_po_ency']
        rtop_od_ency = self.res_config['rtop_od_ency']

        fin_h = self.mos_tech.mos_config['fin_h']
        fin_p = self.mos_tech.mos_config['mos_pitch']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        wcore = core_info['width']
        hcore = core_info['height']
        core_lr_dum_w = core_info['lr_dum_w']
        core_tb_dum_w = core_info['tb_dum_w']
        core_lr_od_loc = core_info['lr_od_loc'][0]
        core_fill_info = core_info['fill_info']
        if well_end_mode & 1 == 0:
            edge_margin = self.res_config.get('edge_margin_nowell', edge_margin)

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)

        # compute dummy OD Y separation in number of fins
        od_sp = po_spy + po_od_exty * 2
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
        bot_dummy_bnd = edge_margin + imp_po_ency + po_od_exty
        bot_pitch_index = -(-(bot_dummy_bnd - fin_p2 + fin_h2) // fin_p)
        top_dummy_bnd = hedge - spy - lres_tb - po_res_spy - po_od_exty
        top_pitch_index = (top_dummy_bnd - fin_p2 - fin_h2) // fin_p
        tot_space = top_pitch_index - bot_pitch_index + 1
        edge_bot_od_loc = fill_symmetric_max_density(tot_space, tot_space, nfin_min, nfin_max,
                                                     od_sp, fill_on_edge=True, cyclic=False)[0]
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
        edge_lr_po_bnd = (edge_bot_od_loc[-1][-1] + po_od_exty + po_spy,
                          adj_top_od_yb - po_od_exty - po_spy)

        # compute total OD area
        od_area = 0
        for od_w, yloc_list in ((core_lr_dum_w + core_tb_dum_w, edge_bot_od_loc),
                                (core_lr_dum_w, edge_lr_od_loc)):
            for yb, yt in yloc_list:
                od_area += od_w * (yt - yb)
        # check OD density rule
        min_od_area = int(math.ceil(wcore * hedge * od_min_density))
        if od_area < min_od_area:
            return None

        # if we get here, then all density rules are met
        # compute fill Y coordinate in edge block
        fill_edge_y_list = []
        for _, _, w, h, core_x, core_y, density, sp_min, sp_max, sp_bnd in core_fill_info:
            sp_yb = sp_bnd + h
            sp_yt = hedge + core_y[0][0]

            area = sp_yt - sp_yb
            tarea = int(math.ceil(area * density ** 0.5))
            c_info = fill_symmetric_min_density_info(area, tarea, h, h, sp_min, sp_max=sp_max,
                                                     fill_on_edge=False)
            edge_y = fill_symmetric_interval(*c_info[0][2], offset=sp_yb, invert=c_info[1])[0]
            edge_y.insert(0, (sp_bnd, sp_yb))
            fill_edge_y_list.append(edge_y)

        # return layout information
        return dict(
            lr_od_loc=(edge_lr_od_loc, edge_lr_po_bnd),
            bot_od_loc=(edge_bot_od_loc, None),
            fb_yb=edge_tb_dum_yb - finfet_od_exty,
            rtop_yb=edge_tb_dum_yb - rtop_od_ency,
            imp_yb=edge_tb_dum_yb - po_od_exty - imp_po_ency,
            fill_edge_y_list=fill_edge_y_list,
        )

    def _draw_dummies(self, template, xc, fg, od_po_loc):
        mos_lay_table = self.config['mos_layer_table']

        po_lch = self.res_config['po_lch']
        po_pitch = self.res_config['po_pitch']
        po_od_exty = self.res_config['po_od_exty']
        mp_spy_dum = self.res_config['mp_spy_dum']
        mp_h_dum = self.res_config['mp_h_dum']
        od_mp_ency_dum = self.res_config['od_mp_ency_dum']
        po_h_min = self.res_config['po_h_min']

        mp_pitch = mp_spy_dum + mp_h_dum

        res = template.grid.resolution

        od_w = po_lch + (fg - 1) * po_pitch
        od_xl = xc - od_w // 2
        od_xr = od_xl + od_w
        po_xr = od_xl + po_lch
        od_loc, po_loc = od_po_loc
        od_dum_lay = mos_lay_table['OD_dummy']
        po_dum_lay = mos_lay_table['PO_dummy']
        mp_dum_lay = mos_lay_table['MP_dummy']
        if od_loc:
            # draw dummy transistors
            for od_yb, od_yt in od_loc:
                po_yb = od_yb - po_od_exty
                po_yt = od_yt + po_od_exty
                # draw OD
                template.add_rect(od_dum_lay, BBox(od_xl, od_yb, od_xr, od_yt, res, unit_mode=True))
                # draw PO
                template.add_rect(po_dum_lay, BBox(od_xl, po_yb, po_xr, po_yt, res, unit_mode=True),
                                  nx=fg, spx=po_pitch * res)
                # draw M0PO
                # compute number of M0PO
                od_h = od_yt - od_yb
                avail_sp = od_h - 2 * od_mp_ency_dum
                num_mp = (avail_sp - mp_h_dum) // mp_pitch + 1
                if num_mp > 0:
                    mp_harr = mp_h_dum + (num_mp - 1) * mp_pitch
                    mp_xl = od_xl + po_lch // 2
                    mp_xr = od_xr - po_lch // 2
                    mp_yb = od_yb + (od_h - mp_harr) // 2
                    mp_yt = mp_yb + mp_h_dum
                    template.add_rect(mp_dum_lay, BBox(mp_xl, mp_yb, mp_xr, mp_yt, res,
                                                       unit_mode=True),
                                      ny=num_mp, spy=mp_pitch * res)
        elif po_loc is not None:
            # draw dummy PO only
            po_yb, po_yt = po_loc
            if po_yt - po_yb >= po_h_min:
                template.add_rect(po_dum_lay, BBox(od_xl, po_yb, po_xr, po_yt, res, unit_mode=True),
                                  nx=fg, spx=po_pitch * res)

    def draw_res_core(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None

        mos_lay_table = self.config['mos_layer_table']
        res_lay_table = self.config['res_layer_table']

        fin_h2 = self.mos_tech.mos_config['fin_h'] // 2
        fin_p2 = self.mos_tech.mos_config['mos_pitch'] // 2

        grid = template.grid
        res = grid.resolution

        finfet_lay = mos_lay_table['FB']
        rpdmy_lay = res_lay_table['RPDMY']
        res_lay = res_lay_table['RES']

        w = layout_info['w']
        l = layout_info['l']
        threshold = layout_info['threshold']
        res_type = layout_info['res_type']
        sub_type = layout_info['sub_type']
        wcore = layout_info['w_core']
        hcore = layout_info['h_core']
        track_widths = layout_info['track_widths']

        core_info = layout_info['core_info']
        lr_od_loc = core_info['lr_od_loc']
        top_od_loc = core_info['top_od_loc']
        bot_od_loc = core_info['bot_od_loc']
        lr_fg = core_info['lr_fg']
        tb_fg = core_info['tb_fg']
        port_info = core_info['port_info']
        fill_info = core_info['fill_info']

        xc, yc = wcore // 2, hcore // 2
        wres, lres, _, _ = self.get_res_dimension(l, w)

        # set size and draw implant layers
        implant_layers = self.get_res_imp_layers(res_type, sub_type, threshold)
        arr_box = BBox(0, 0, wcore, hcore, res, unit_mode=True)
        # mos layer need to be snap to fin edges.
        fin_box = BBox(0, -fin_p2 - fin_h2, wcore, hcore + fin_p2 + fin_h2, res, unit_mode=True)
        for lay in implant_layers:
            if lay == finfet_lay:
                template.add_rect(lay, fin_box)
            else:
                template.add_rect(lay, arr_box)
        template.array_box = arr_box
        template.prim_bound_box = arr_box
        template.add_cell_boundary(arr_box)

        # draw RPDMY
        rpdmy_yb = yc - l // 2
        rpdmy_yt = rpdmy_yb + l
        template.add_rect(rpdmy_lay, BBox(0, rpdmy_yb, wcore, rpdmy_yt, res, unit_mode=True))

        # draw resistor
        rh_yb = yc - lres // 2
        rh_xl = xc - wres // 2
        template.add_rect(res_lay, BBox(rh_xl, rh_yb, rh_xl + wres, rh_yb + lres, res,
                                        unit_mode=True))

        # draw vias and ports
        bot_layer = self.get_bot_layer()
        for port_name, rect_list, via_list in port_info:
            for rect_info in rect_list:
                template.add_rect(rect_info['layer'], rect_info['bbox'])
            for via_params in via_list:
                template.add_via_primitive(**via_params)

            tr_bbox = rect_list[-1]['bbox']
            port_tr = grid.coord_to_track(bot_layer, tr_bbox.yc_unit, unit_mode=True)
            pin_warr = WireArray(TrackID(bot_layer, port_tr, width=track_widths[0]),
                                 tr_bbox.left_unit, tr_bbox.right_unit, res=res, unit_mode=True)
            template.add_pin(port_name, pin_warr, show=False)

        # draw dummies
        self._draw_dummies(template, 0, lr_fg, lr_od_loc)
        self._draw_dummies(template, wcore, lr_fg, lr_od_loc)
        self._draw_dummies(template, xc, tb_fg, top_od_loc)
        self._draw_dummies(template, xc, tb_fg, bot_od_loc)

        # draw metal fill
        for layer, exc_layer, _, _, core_x, core_y, _, _, _, _ in fill_info:
            template.add_rect(exc_layer, arr_box)
            for xl, xr in core_x:
                for yb, yt in core_y:
                    template.add_rect(layer, BBox(xl, yb, xr, yt, res, unit_mode=True))

    def draw_res_boundary(self, template, boundary_type, layout_info, end_mode):
        # type: (TemplateBase, str, Dict[str, Any], bool) -> None

        mos_lay_table = self.config['mos_layer_table']
        res_lay_table = self.config['res_layer_table']

        fin_h2 = self.mos_tech.mos_config['fin_h'] // 2
        fin_p2 = self.mos_tech.mos_config['mos_pitch'] // 2

        grid = template.grid
        res = grid.resolution

        finfet_lay = mos_lay_table['FB']
        rpdmy_lay = res_lay_table['RPDMY']
        rtop_lay = res_lay_table['RTOP']
        res_dum_lay = res_lay_table['RES_dummy']

        w = layout_info['w']
        l = layout_info['l']
        threshold = layout_info['threshold']
        res_type = layout_info['res_type']
        sub_type = layout_info['sub_type']
        w_core = layout_info['w_core']
        h_core = layout_info['h_core']
        w_edge = layout_info['w_edge']
        h_edge = layout_info['h_edge']

        wres, lres, wres_lr, lres_tb = self.get_res_dimension(l, w)
        implant_layers = self.get_res_imp_layers(res_type, sub_type, threshold)
        bnd_spx = (w_core - wres) // 2
        bnd_spy = (h_core - lres) // 2

        core_info = layout_info['core_info']
        edge_lr_info = layout_info['edge_lr_info']
        edge_tb_info = layout_info['edge_tb_info']

        fb_xl = edge_lr_info['fb_xl']
        imp_xl = edge_lr_info['imp_xl']
        rtop_xl = edge_lr_info['rtop_xl']
        fb_yb = edge_tb_info['fb_yb']
        imp_yb = edge_tb_info['imp_yb']
        rtop_yb = edge_tb_info['rtop_yb']

        core_fill_info = core_info['fill_info']
        core_lr_od_loc = core_info['lr_od_loc']
        core_top_od_loc = core_info['top_od_loc']
        core_bot_od_loc = core_info['bot_od_loc']
        edge_bot_od_loc = edge_tb_info['bot_od_loc']
        edge_lr_od_loc = edge_tb_info['lr_od_loc']
        core_lr_fg = core_info['lr_fg']
        core_tb_fg = core_info['tb_fg']
        edge_lr_fg = edge_lr_info['lr_fg']
        edge_tb_fg = edge_lr_info['tb_fg']
        edge_tb_xc = edge_lr_info['tb_xc']
        edge_lr_xc = edge_lr_info['lr_xc']

        #  get bounding box/implant coordinates, draw RPDMY, and draw dummies.
        if boundary_type == 'lr':
            # get bounding box and RH coordinates
            bnd_box = BBox(0, 0, w_edge, h_core, res, unit_mode=True)
            rh_xr = w_edge - bnd_spx
            rh_xl = rh_xr - wres_lr
            rh_yb, rh_yt = bnd_spy, bnd_spy + lres
            # draw RPDMY in left/right edge block
            rpdmy_yb = h_core // 2 - l // 2
            rpdmy_yt = rpdmy_yb + l
            template.add_rect(rpdmy_lay, BBox(rh_xr, rpdmy_yb, w_edge, rpdmy_yt, res,
                                              unit_mode=True))
            # set implant Y coordinates to 0
            imp_yb = rtop_yb = 0
            fb_yb = -fin_p2 - fin_h2
            # draw left edge dummies
            self._draw_dummies(template, edge_lr_xc, edge_lr_fg, core_lr_od_loc)
            # draw bottom/top edge dummies
            self._draw_dummies(template, edge_tb_xc, edge_tb_fg, core_bot_od_loc)
            self._draw_dummies(template, edge_tb_xc, edge_tb_fg, core_top_od_loc)
            fill_x_list = edge_lr_info['fill_edge_x_list']
            fill_y_list = [info[5] for info in core_fill_info]
        elif boundary_type == 'tb':
            # get bounding box and RH coordinates
            bnd_box = BBox(0, 0, w_core, h_edge, res, unit_mode=True)
            rh_xl, rh_xr = bnd_spx, bnd_spx + wres
            rh_yt = h_edge - bnd_spy
            rh_yb = rh_yt - lres_tb
            # set implant X coordinates to 0
            fb_xl = imp_xl = rtop_xl = 0
            # draw bottom edge dummies
            self._draw_dummies(template, 0, core_lr_fg, edge_bot_od_loc)
            self._draw_dummies(template, w_core // 2, core_tb_fg, edge_bot_od_loc)
            self._draw_dummies(template, w_core, core_lr_fg, edge_bot_od_loc)
            # draw left/right edge dummies
            self._draw_dummies(template, 0, core_lr_fg, edge_lr_od_loc)
            self._draw_dummies(template, w_core, core_lr_fg, edge_lr_od_loc)
            fill_x_list = [info[4] for info in core_fill_info]
            fill_y_list = edge_tb_info['fill_edge_y_list']
        else:
            # get bounding box and RH coordinates
            bnd_box = BBox(0, 0, w_edge, h_edge, res, unit_mode=True)
            rh_xr = w_edge - bnd_spx
            rh_xl = rh_xr - wres_lr
            rh_yt = h_edge - bnd_spy
            rh_yb = rh_yt - lres_tb
            # draw bottom edge dummies
            self._draw_dummies(template, edge_lr_xc, edge_lr_fg, edge_bot_od_loc)
            self._draw_dummies(template, edge_tb_xc, edge_tb_fg, edge_bot_od_loc)
            # draw left edge dummies
            self._draw_dummies(template, edge_lr_xc, edge_lr_fg, edge_lr_od_loc)
            fill_x_list = edge_lr_info['fill_edge_x_list']
            fill_y_list = edge_tb_info['fill_edge_y_list']

        # draw resistor dummy
        template.add_rect(res_dum_lay, BBox(rh_xl, rh_yb, rh_xr, rh_yt, res, unit_mode=True))

        # draw implant layers
        for lay in implant_layers:
            yt = bnd_box.top_unit
            if lay == finfet_lay:
                xl, yb = fb_xl, fb_yb
                yt += fin_p2 + fin_h2
            elif lay == rtop_lay:
                xl, yb = rtop_xl, rtop_yb
            else:
                xl, yb = imp_xl, imp_yb
            template.add_rect(lay, BBox(xl, yb, bnd_box.right_unit, yt, res, unit_mode=True))

        # set bounding box
        template.prim_bound_box = bnd_box
        template.array_box = bnd_box
        template.add_cell_boundary(bnd_box)

        # draw metal fill
        for (layer, exc_layer, _, _, _, _, _, _, _, _), fill_x, fill_y in zip(core_fill_info,
                                                                              fill_x_list,
                                                                              fill_y_list):
            template.add_rect(exc_layer, bnd_box)
            for xl, xr in fill_x:
                for yb, yt in fill_y:
                    template.add_rect(layer, BBox(xl, yb, xr, yt, res, unit_mode=True))
