# -*- coding: utf-8 -*-

"""This module defines bias routing related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any, Iterable, List, Union, Tuple, Generator, Optional

from collections import namedtuple
from itertools import chain, repeat, islice

from bag.math import lcm
from bag.util.interval import IntervalSet
from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing.base import TrackManager, TrackID, WireArray

if TYPE_CHECKING:
    from bag.layout.routing.grid import RoutingGrid
    from bag.layout.template import TemplateDB

BiasInfo = namedtuple('BiasInfo', ['tracks', 'supplies', 'p0', 'p1', 'shields'])


def _add_inst_r180(template, master, x, y):
    box = master.array_box
    return template.add_instance(master, loc=(x + box.width_unit, y + box.height_unit),
                                 orient='R180', unit_mode=True)


def compute_vroute_width(template, vm_layer, blk_w, num_vdd, num_vss, bias_config):
    grid = template.grid

    hm_layer = vm_layer + 1
    if num_vdd == 0:
        vdd_xl = vdd_xr = 0
    else:
        vdd_xl = BiasShieldEnd.get_dimension(grid, hm_layer, bias_config, num_vdd)
        vdd_xr = vdd_xl + BiasShield.get_block_size(grid, vm_layer, bias_config, num_vdd)[0]
    if num_vss == 0:
        vss_xr = vss_xl = vdd_xr
    else:
        vss_xl = vdd_xr + BiasShieldEnd.get_dimension(grid, hm_layer, bias_config, num_vss)
        vss_xr = vss_xl + BiasShield.get_block_size(grid, vm_layer, bias_config, num_vss)[0]

    route_w = -(-vss_xr // blk_w) * blk_w
    return route_w, (vdd_xl, vdd_xr), (vss_xl, vss_xr)


def join_bias_vroutes(template, vm_layer, vdd_dx, vss_dx, xr, num_vdd_tot, num_vss_tot,
                      hm_bias_info_list, bias_config, vdd_pins, vss_pins,
                      xl=0, yt=None, vss_warrs=None, vdd_warrs=None):
    grid = template.grid

    vss_intvs = IntervalSet()
    vdd_intvs = IntervalSet()
    if vss_warrs is not None:
        for w in WireArray.single_warr_iter(vss_warrs):
            vss_intvs.add(w.track_id.get_bounds(grid, unit_mode=True), val=w)
    if vdd_warrs is not None:
        for w in WireArray.single_warr_iter(vdd_warrs):
            vdd_intvs.add(w.track_id.get_bounds(grid, unit_mode=True), val=w)

    vdd_xl, vdd_xr = vdd_dx
    vss_xl, vss_xr = vss_dx
    vdd_xl += xl
    vdd_xr += xl
    vss_xl += xl
    vss_xr += xl
    vss_params = dict(
        nwire=num_vss_tot,
        width=1,
        space_sig=0,
    )
    vdd_params = dict(
        nwire=num_vdd_tot,
        width=1,
        space_sig=0,
    )
    hm_layer = vm_layer + 1
    vss_hm_list = []
    vdd_hm_list = []
    inst_info_list = []
    vss_y_prev = vdd_y_prev = None
    params = dict(
        bot_layer=vm_layer,
        bias_config=bias_config,
    )
    for idx, (code, num, y0) in enumerate(hm_bias_info_list):
        if code == 0:
            params['bot_params'] = vss_params
            if vss_y_prev is not None and y0 > vss_y_prev:
                sup_warrs = list(vss_intvs.overlap_values((vss_y_prev, y0)))
                tmp = BiasShield.draw_bias_shields(template, vm_layer, bias_config, num_vss_tot,
                                                   vss_xl, vss_y_prev, y0, sup_warrs=sup_warrs)
                vss_hm_list.extend(tmp[0])
        else:
            params['bot_params'] = vdd_params
            if vdd_y_prev is not None and y0 > vdd_y_prev:
                sup_warrs = list(vdd_intvs.overlap_values((vdd_y_prev, y0)))
                tmp = BiasShield.draw_bias_shields(template, vm_layer, bias_config, num_vdd_tot,
                                                   vdd_xl, vdd_y_prev, y0, sup_warrs=sup_warrs)
                vdd_hm_list.extend(tmp[0])
            if vss_y_prev is not None and y0 > vss_y_prev:
                sup_warrs = list(vss_intvs.overlap_values((vss_y_prev, y0)))
                tmp = BiasShield.draw_bias_shields(template, vm_layer, bias_config, num_vss_tot,
                                                   vss_xl, vss_y_prev, y0, sup_warrs=sup_warrs)
                vss_hm_list.extend(tmp[0])

        params['top_params'] = dict(
            nwire=num,
            width=1,
            space_sig=0,
        )
        if idx == 0:
            master = template.new_template(params=params, temp_cls=BiasShieldJoin)
            if code == 1:
                inst_info_list.append((1, master, vdd_xl, y0))
                BiasShield.draw_bias_shields(template, hm_layer, bias_config, num, y0,
                                             vdd_xr, xr)
                vdd_y_prev = y0 + master.array_box.top_unit
            else:
                inst_info_list.append((0, master, vss_xl, y0))
                BiasShield.draw_bias_shields(template, hm_layer, bias_config, num, y0,
                                             vss_xr, xr)
                vss_y_prev = y0 + master.array_box.top_unit
        elif code == 1:
            params['bot_open'] = True
            master = template.new_template(params=params, temp_cls=BiasShieldJoin)
            inst_info_list.append((1, master, vdd_xl, y0))
            BiasShield.draw_bias_shields(template, hm_layer, bias_config, num, y0,
                                         vdd_xr, vss_xl, tb_mode=1)
            vdd_y_prev = y0 + master.array_box.top_unit
            params['draw_top'] = False
            params['bot_params'] = vss_params
            master = template.new_template(params=params, temp_cls=BiasShieldCrossing)
            inst_info_list.append((0, master, vss_xl, y0))
            BiasShield.draw_bias_shields(template, hm_layer, bias_config, num, y0,
                                         vss_xr, xr)
            vss_y_prev = y0 + master.array_box.top_unit
        else:
            params['bot_open'] = True
            master = template.new_template(params=params, temp_cls=BiasShieldJoin)
            inst_info_list.append((0, master, vss_xl, y0))
            BiasShield.draw_bias_shields(template, hm_layer, bias_config, num, y0,
                                         vss_xr, xr)
            vss_y_prev = y0 + master.array_box.top_unit

    if yt is None:
        yt = max(vss_y_prev, vdd_y_prev)
    if vss_y_prev < yt:
        sup_warrs = list(vss_intvs.overlap_values((vss_y_prev, yt)))
        tmp = BiasShield.draw_bias_shields(template, vm_layer, bias_config, num_vss_tot,
                                           vss_xl, vss_y_prev, yt, sup_warrs=sup_warrs)
        vss_hm_list.extend(tmp[0])
    if vdd_y_prev < yt:
        sup_warrs = list(vdd_intvs.overlap_values((vdd_y_prev, yt)))
        tmp = BiasShield.draw_bias_shields(template, vm_layer, bias_config, num_vdd_tot,
                                           vdd_xl, vdd_y_prev, yt, sup_warrs=sup_warrs)
        vdd_hm_list.extend(tmp[0])

    sup_layer = hm_layer + 1
    vss_list = []
    vdd_list = []
    for code, master, x0, y0 in inst_info_list:
        inst = _add_inst_r180(template, master, x0, y0)
        if code == 1:
            vdd_list.extend(inst.port_pins_iter('sup', layer=sup_layer))
        else:
            vss_list.extend(inst.port_pins_iter('sup', layer=sup_layer))

    vdd_list = template.connect_wires(vdd_list, upper=yt, unit_mode=True)
    vss_list = template.connect_wires(vss_list, upper=yt, unit_mode=True)
    template.draw_vias_on_intersections(vdd_hm_list, vdd_list)
    template.draw_vias_on_intersections(vss_hm_list, vss_list)

    # connect pins
    vss_tidx_list = BiasShield.get_route_tids(grid, vm_layer, vss_xl, bias_config, num_vss_tot)
    vdd_tidx_list = BiasShield.get_route_tids(grid, vm_layer, vdd_xl, bias_config, num_vdd_tot)
    pin_map = {}
    vss_tr_warrs = []
    vdd_tr_warrs = []
    for pins, tidx_list, result_list in ((vss_pins, vss_tidx_list, vss_tr_warrs),
                                         (vdd_pins, vdd_tidx_list, vdd_tr_warrs)):
        next_idx = 0
        for name, warr in pins:
            if name in pin_map:
                cur_idx = pin_map[name]
                add_pin = False
            else:
                cur_idx = pin_map[name] = next_idx
                next_idx += 1
                add_pin = True
            tidx, tr_w = tidx_list[cur_idx + 1]
            tid = TrackID(vm_layer, tidx, width=tr_w)
            warr = template.connect_to_tracks(warr, tid, track_upper=yt, unit_mode=True)
            if add_pin:
                result_list.append((name, warr))

        pin_map.clear()

    return vdd_tr_warrs, vss_tr_warrs, vdd_list, vss_list


class BiasShield(TemplateBase):
    """Unit cell template for shield around bias wires.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._route_tids = None
        self._sup_intv = None

    @property
    def route_tids(self):
        return self._route_tids

    @property
    def sup_intv(self):
        return self._sup_intv

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layer='the routing layer.',
            nwire='number of routing wires.',
            bias_config='The bias configuration dictionary.',
            top='True to draw top shield.',
            width='route wire width.',
            space_sig='route wire spacing.',
            is_end='True if this is used in BiasShieldEnd.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            top=True,
            width=1,
            space_sig=0,
            is_end=False,
        )

    def get_layout_basename(self):
        layer = self.params['layer']
        nwire = self.params['nwire']
        desc = 'top' if self.params['top'] else 'bot'
        return 'bias_shield_%s_lay%d_n%d' % (desc, layer, nwire)

    def draw_layout(self):
        # type: () -> None
        route_layer = self.params['layer']
        bias_config = self.params['bias_config']
        nwire_real = self.params['nwire']
        width = self.params['width']
        space_sig = self.params['space_sig']
        top = self.params['top']
        is_end = self.params['is_end']

        grid = self.grid
        res = grid.resolution

        bot_layer = route_layer - 1
        top_layer = route_layer + 1
        route_dir = grid.get_direction(route_layer)
        is_horiz = (route_dir == 'x')

        # compute size
        tot_w, tot_h = self.get_block_size(grid, route_layer, bias_config, nwire_real,
                                           width=width, space_sig=space_sig)

        dim_par = tot_w if is_horiz else tot_h
        bbox = BBox(0, 0, tot_w, tot_h, res, unit_mode=True)
        self.prim_top_layer = top_layer
        self.prim_bound_box = bbox
        self.array_box = bbox
        self.add_cell_boundary(bbox)

        # compute wire location
        self._route_tids = locs = self.get_route_tids(self.grid, route_layer, 0, bias_config,
                                                      nwire_real, width=width, space_sig=space_sig)

        pitch = locs[nwire_real + 1][0] - locs[0][0]
        tr_upper = bbox.width_unit if is_horiz else bbox.height_unit
        sh_warr = self.add_wires(route_layer, locs[0][0], 0, tr_upper, width=1, num=2, pitch=pitch,
                                 unit_mode=True)

        sup_layer = top_layer if top else bot_layer
        pitch2 = grid.get_track_pitch(sup_layer, unit_mode=True) // 2
        sup_np2, sup_w, sup_spe2, sup_p2 = bias_config[sup_layer]
        sup_unit = sup_np2 * pitch2
        num = dim_par // sup_unit
        if num == 1:
            num = 1 if sup_p2 == 0 else 1 + (sup_np2 - 2 * sup_spe2) // sup_p2
            sup_tid = TrackID(sup_layer, (sup_spe2 - 1) / 2, width=sup_w, num=num, pitch=sup_p2 / 2)
        else:
            if sup_p2 == 0:
                sup_tid = TrackID(sup_layer, (sup_spe2 - 1) / 2, width=sup_w, num=num,
                                  pitch=sup_np2 / 2)
            else:
                # TODO: come back to fix this
                raise ValueError('umm...see Eric')

        if is_end:
            sup_n = sup_tid.num
            if sup_n > 1:
                sup_pitch = sup_tid.pitch
                new_tid = TrackID(sup_layer, sup_tid.base_index, width=sup_w, num=sup_n - 1,
                                  pitch=sup_pitch)
                warr = self.connect_to_tracks(sh_warr, new_tid)
                self.add_pin('sup', warr, show=False)
                sup_tid = TrackID(sup_layer, sup_tid.base_index + (sup_n - 1) * sup_pitch,
                                  width=sup_w)
            shl = grid.get_wire_bounds(route_layer, locs[0][0], width=1, unit_mode=True)[0]
            shr = grid.get_wire_bounds(route_layer, locs[nwire_real + 1][0], width=1,
                                       unit_mode=True)[1]
            sh_singles = sh_warr.to_warr_list()
            if top:
                test_warr = self.connect_to_tracks(sh_singles[0], sup_tid)
                vext = shl - test_warr.lower_unit
            else:
                test_warr = self.connect_to_tracks(sh_singles[1], sup_tid)
                vext = test_warr.upper_unit - shr
            warr = self.add_wires(sup_layer, sup_tid.base_index, shl - vext, shr + vext,
                                  width=sup_w, unit_mode=True)
        else:
            warr = self.connect_to_tracks(sh_warr, sup_tid)

        self.add_pin('shield', sh_warr, show=False)
        self.add_pin('sup', warr, show=False)
        sup_box = warr.get_bbox_array(self.grid).get_overall_bbox()
        self._sup_intv = sup_box.get_interval(route_dir, unit_mode=True)

    @classmethod
    def get_block_size(cls, grid, route_layer, bias_config, nwire, width=1, space_sig=0):
        # type: (RoutingGrid, int, Dict[int, Tuple[int, ...]], int, int, int) -> Tuple[int, int]

        bot_layer = route_layer - 1
        top_layer = route_layer + 1
        route_dir = grid.get_direction(route_layer)
        is_horiz = route_dir == 'x'

        # calculate dimension
        bot_pitch2 = grid.get_track_pitch(bot_layer, unit_mode=True) // 2
        route_pitch2 = grid.get_track_pitch(route_layer, unit_mode=True) // 2
        top_pitch2 = grid.get_track_pitch(top_layer, unit_mode=True) // 2
        bot_np2, bot_w, bot_spe2, _ = bias_config[bot_layer]
        route_np2, route_w, route_spe2, _ = bias_config[route_layer]
        top_np2, top_w, top_spe2, _ = bias_config[top_layer]
        bot_unit = bot_np2 * bot_pitch2
        route_unit = route_np2 * route_pitch2
        top_unit = top_np2 * top_pitch2
        dim_par = lcm([bot_unit, top_unit])

        tr_manager = TrackManager(grid, {'sig': {route_layer: width}},
                                  {('sig', ''): {route_layer: space_sig}}, half_space=True)

        tmp = [route_w]
        nwire = -(-nwire // 2) * 2
        route_list = list(chain(tmp, repeat('sig', nwire), tmp))
        _, locs = tr_manager.place_wires(route_layer, route_list)
        num_htr = int(2 * (locs[nwire + 1] - locs[0])) + 2 * route_spe2
        dim_perp = -(-(num_htr * route_pitch2) // route_unit) * route_unit
        if is_horiz:
            return dim_par, dim_perp
        else:
            return dim_perp, dim_par

    @classmethod
    def get_route_tids(cls,  # type: BiasShield
                       grid,  # type: RoutingGrid
                       layer,  # type: int
                       offset,  # type: int
                       bias_config,  # type: Dict[int, Tuple[int, ...]]
                       nwire,  # type: int
                       width=1,  # type: int
                       space_sig=0,  # type: int
                       ):
        # type: (...) -> List[Tuple[int, int]]

        tot_w, tot_h = cls.get_block_size(grid, layer, bias_config, nwire,
                                          width=width, space_sig=space_sig)
        nwire_alloc = -(-nwire // 2) * 2

        route_dir = grid.get_direction(layer)
        is_horiz = (route_dir == 'x')
        dim_perp = tot_h if is_horiz else tot_w

        tr_manager = TrackManager(grid, {'sig': {layer: width}},
                                  {('sig', ''): {layer: space_sig}}, half_space=True)
        tmp = [1]
        route_list = list(chain(tmp, repeat('sig', nwire_alloc), tmp))
        route_pitch = grid.get_track_pitch(layer, unit_mode=True)
        route_ntr = dim_perp / route_pitch
        locs = tr_manager.align_wires(layer, route_list, route_ntr, alignment=0, start_idx=0)

        htr0 = int(round(grid.coord_to_track(layer, offset, unit_mode=True) * 2 + 1))
        tr0 = htr0 // 2 if htr0 % 2 == 0 else htr0 / 2
        return list(chain([(locs[0] + tr0, 1)],
                          ((locs[idx] + tr0, width) for idx in range(1, nwire + 1)),
                          [(locs[nwire_alloc + 1] + tr0, 1)]))

    @classmethod
    def draw_bias_shields(cls,  # type: BiasShield
                          template,  # type: TemplateBase
                          layer,  # type: int
                          bias_config,  # type: Dict[int, Tuple[int, ...]]
                          nwire,  # type: int
                          offset,  # type: int
                          lower,  # type: int
                          upper,  # type: int
                          width=1,  # type: int
                          space_sig=0,  # type: int
                          sup_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                          check_blockage=True,  # type: bool
                          tb_mode=3,  # type: int
                          ):
        if lower >= upper:
            return [], None

        grid = template.grid
        res = grid.resolution

        params = dict(
            layer=layer,
            nwire=nwire,
            bias_config=bias_config,
            top=False,
            width=width,
            space_sig=space_sig,
        )
        bot_master = template.new_template(params=params, temp_cls=BiasShield)
        sh_box = bot_master.bound_box
        params['top'] = True
        top_master = template.new_template(params=params, temp_cls=BiasShield)
        blk_w = sh_box.width_unit
        blk_h = sh_box.height_unit

        route_tids = bot_master.route_tids
        tr_dir = grid.get_direction(layer)
        is_horiz = tr_dir == 'x'
        qdim = blk_w if is_horiz else blk_h
        tr0 = grid.coord_to_track(layer, offset, unit_mode=True) + 0.5
        bot_intvs = IntervalSet()
        top_intvs = IntervalSet()

        nstart, tr_offset = divmod(lower, qdim)
        nstop, tr_off_test = divmod(upper, qdim)
        if tr_offset != tr_off_test:
            raise ValueError('upper - lower not divisible by %d' % qdim)

        # draw shields
        tr0_ref = route_tids[0][0]
        sh0 = tr0_ref + tr0
        shp = route_tids[nwire + 1][0] - tr0_ref
        shields = template.add_wires(layer, sh0, lower, upper, num=2,
                                     pitch=shp, unit_mode=True)

        # connect supply wires if necessary
        sup_layer = layer + 1
        if sup_warrs is not None:
            new_sup_warrs = []
            _, wires = template.connect_to_tracks(sup_warrs, shields.track_id, return_wires=True)
            for warr in wires:
                if warr.layer_id == sup_layer:
                    new_sup_warrs.append(warr)
            sup_warrs = new_sup_warrs
        else:
            sup_warrs = []

        # get blockages
        if check_blockage:
            if is_horiz:
                test_box = BBox(lower, offset, upper, offset + blk_h, res, unit_mode=True)
            else:
                test_box = BBox(offset, lower, offset + blk_w, upper, res, unit_mode=True)
            for lay_id, intv in ((layer - 1, bot_intvs), (layer + 1, top_intvs)):
                w_ntr = bias_config[lay_id][1]
                if is_horiz:
                    spx = grid.get_space(lay_id, w_ntr, unit_mode=True)
                    spy = grid.get_line_end_space(lay_id, w_ntr, unit_mode=True)
                else:
                    spy = grid.get_space(lay_id, w_ntr, unit_mode=True)
                    spx = grid.get_line_end_space(lay_id, w_ntr, unit_mode=True)
                for box in template.blockage_iter(lay_id, test_box, spx=spx, spy=spy):
                    blkl, blku = box.get_interval(tr_dir, unit_mode=True)
                    blkl = max(blkl, lower)
                    blku = min(blku, upper)
                    if blku > blkl:
                        intv.add((blkl, blku), merge=True, abut=True)

        master_intv_list = []
        if tb_mode & 1 != 0:
            master_intv_list.append((bot_master, bot_intvs))
        if tb_mode & 2 != 0:
            master_intv_list.append((top_master, top_intvs))
        # draw blocks
        for master, intvs in master_intv_list:
            sl, su = master.sup_intv
            ncur = nstart
            for nend, nnext in cls._get_blk_idx_iter(intvs, sl + tr_offset, su + tr_offset,
                                                     qdim, nstart, nstop):
                nblk = nend - ncur
                if nblk > 0:
                    if is_horiz:
                        loc = (ncur * qdim + tr_offset, offset)
                        nx = nblk
                        ny = 1
                    else:
                        loc = (offset, ncur * qdim + tr_offset)
                        nx = 1
                        ny = nblk
                    inst = template.add_instance(master, loc=loc, nx=nx, ny=ny,
                                                 spx=qdim, spy=qdim, unit_mode=True)
                    sup_warrs.extend(inst.port_pins_iter('sup', layer=sup_layer))
                ncur = nnext

        return sup_warrs, shields

    @classmethod
    def connect_bias_shields(cls,  # type: BiasShield
                             template,  # type: TemplateBase
                             layer,  # type: int
                             bias_config,  # type: Dict[int, Tuple[int, ...]]
                             warr_list2,  # type: List[Union[WireArray, Iterable[WireArray]]]
                             offset,  # type: int
                             width=1,  # type: int
                             space_sig=0,  # type: int
                             tr_lower=None,  # type: Optional[int]
                             tr_upper=None,  # type: Optional[int]
                             lu_end_mode=0,  # type: int
                             sup_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                             add_end=True,  # type: bool
                             extend_tracks=True,  # type: bool
                             ):
        # type: (...) -> BiasInfo
        grid = template.grid

        nwire = len(warr_list2)

        blk_w, blk_h = cls.get_block_size(grid, layer, bias_config, nwire,
                                          width=width, space_sig=space_sig)
        route_tids = cls.get_route_tids(grid, layer, offset, bias_config, nwire,
                                        width=width, space_sig=space_sig)
        tr_dir = grid.get_direction(layer)
        is_horiz = tr_dir == 'x'
        qdim = blk_w if is_horiz else blk_h
        if lu_end_mode == 0:
            min_len_mode = 0
        elif lu_end_mode & 2 != 0:
            min_len_mode = 1
        else:
            min_len_mode = -1
        tr_warr_list = []

        if tr_lower is not None:
            tr_offset = tr_lower % qdim
        elif tr_upper is not None:
            tr_offset = tr_upper % qdim
        else:
            tr_offset = 0

        for warr_list, (tidx, tr_width) in zip(warr_list2, islice(route_tids, 1, nwire + 1)):
            if isinstance(warr_list, WireArray):
                warr_list = [warr_list]

            cur_tid = TrackID(layer, tidx, width=width)
            tr_warr = template.connect_to_tracks(warr_list, cur_tid, min_len_mode=min_len_mode)
            tr_warr_list.append(tr_warr)
            if tr_warr is not None:
                if tr_lower is None:
                    tr_lower = tr_warr.lower_unit
                else:
                    tr_lower = min(tr_lower, tr_warr.lower_unit)
                if tr_upper is None:
                    tr_upper = tr_warr.upper_unit
                else:
                    tr_upper = max(tr_upper, tr_warr.upper_unit)

        if tr_lower is None or tr_upper is None:
            raise ValueError('Cannot determine bias shield location.')

        # round lower/upper to nearest coordinates, then draw shields.
        nstart = (tr_lower - tr_offset) // qdim
        nstop = -(-(tr_upper - tr_offset) // qdim)
        tr_lower = nstart * qdim + tr_offset
        tr_upper = nstop * qdim + tr_offset

        tmp = cls.draw_bias_shields(template, layer, bias_config, nwire, offset, tr_lower, tr_upper,
                                    width=width, space_sig=space_sig, sup_warrs=sup_warrs)
        sup_warrs, shields = tmp

        # get bias information, extend tracks, and draw ends
        if is_horiz:
            p0 = (tr_lower, offset)
            p1 = (tr_upper, offset + blk_h)
        else:
            p0 = (offset, tr_lower)
            p1 = (offset + blk_w, tr_upper)
        if lu_end_mode == 0:
            if extend_tracks:
                tr_warr_list = template.extend_wires(tr_warr_list, lower=tr_lower, upper=tr_upper,
                                                     unit_mode=True)
        else:
            if lu_end_mode & 2 != 0:
                if extend_tracks:
                    tr_warr_list = template.extend_wires(tr_warr_list, upper=tr_upper,
                                                         unit_mode=True)
                eorient = 'MY' if is_horiz else 'MX'
                loc = p0
            else:
                if extend_tracks:
                    tr_warr_list = template.extend_wires(tr_warr_list, lower=tr_lower,
                                                         unit_mode=True)
                eorient = 'R0'
                loc = (tr_upper, offset) if is_horiz else (offset, tr_upper)
            if add_end:
                params = dict(
                    layer=layer,
                    nwire=nwire,
                    bias_config=bias_config,
                    width=width,
                    space_sig=space_sig,
                )
                end_master = template.new_template(params=params, temp_cls=BiasShieldEnd)
                inst = template.add_instance(end_master, loc=loc, orient=eorient, unit_mode=True)
                sup_warrs.extend(inst.get_all_port_pins('sup', layer=layer + 1))

        return BiasInfo(tracks=tr_warr_list, supplies=sup_warrs, p0=p0, p1=p1, shields=shields)

    @classmethod
    def _get_blk_idx_iter(cls,  # type: BiasShield
                          intvs,  # type: IntervalSet
                          sl,  # type: int
                          su,  # type: int
                          qdim,  # type: int
                          nstart,  # type: int
                          nstop,  # type: int
                          ):
        # type: (...) -> Generator[Tuple[int, int], None, None]
        for lower, upper in intvs:
            nend = 1 + (lower - su) // qdim
            if nend < nstart:
                raise ValueError('wire interval (%d, %d) is out of bounds.' % (lower, upper))
            ncur = -(-(upper - sl) // qdim)
            yield nend, ncur
        yield nstop, nstop


class BiasShieldEnd(TemplateBase):
    """end cap of bias shield wires.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._nblk = None

    @property
    def num_blk(self):
        return self._nblk

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layer='the routing layer.',
            nwire='number of routing wires.',
            bias_config='The bias configuration dictionary.',
            width='route wire width.',
            space_sig='route wire spacing.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            width=1,
            space_sig=0,
        )

    def get_layout_basename(self):
        layer = self.params['layer']
        nwire = self.params['nwire']
        return 'bias_shield_end_lay%d_n%d' % (layer, nwire)

    def draw_layout(self):
        # type: () -> None
        route_layer = self.params['layer']
        nwire_real = self.params['nwire']
        bias_config = self.params['bias_config']
        width = self.params['width']
        space_sig = self.params['space_sig']

        grid = self.grid

        top_layer = route_layer + 1
        route_dir = grid.get_direction(route_layer)
        is_horiz = (route_dir == 'x')

        params = self.params.copy()
        params['top'] = False
        params['is_end'] = True
        bot_master = self.new_template(params=params, temp_cls=BiasShield)
        params['top'] = True
        top_master = self.new_template(params=params, temp_cls=BiasShield)
        bot_box = bot_master.bound_box
        blk_w = bot_box.width_unit
        blk_h = bot_box.height_unit
        min_len = grid.get_min_length(route_layer, width, unit_mode=True)

        orig = (0, 0)
        nx = ny = 1
        cr = self.get_dimension(grid, route_layer, bias_config, nwire_real, width=width,
                                space_sig=space_sig)
        if is_horiz:
            nx = nblk = cr // blk_w
        else:
            ny = nblk = cr // blk_h
        self._nblk = nblk

        bot_inst = self.add_instance(bot_master, 'XBOT', loc=orig, nx=nx, ny=ny,
                                     spx=blk_w, spy=blk_h, unit_mode=True)
        top_inst = self.add_instance(top_master, 'XTOP', loc=orig, nx=nx, ny=ny,
                                     spx=blk_w, spy=blk_h, unit_mode=True)

        nwire2 = -(-nwire_real // 2)
        (tidx0, tr_w), (tidx1, _) = bot_master.route_tids[1:3]
        pitch2 = 2 * (tidx1 - tidx0)
        cl = cr - min_len
        warr0 = self.add_wires(route_layer, tidx0, cl, cr, width=tr_w, num=nwire2, pitch=pitch2,
                               unit_mode=True)
        warr1 = self.add_wires(route_layer, tidx1, cl, cr, width=tr_w, num=nwire2, pitch=pitch2,
                               unit_mode=True)
        bot_port = bot_inst.get_port('sup', row=ny - 1, col=nx - 1)
        top_port = top_inst.get_port('sup', row=ny - 1, col=nx - 1)
        self.connect_to_track_wires(warr0, bot_port.get_pins()[-1])
        self.connect_to_track_wires(warr1, top_port.get_pins()[-1])

        bnd_box = bot_inst.bound_box
        self.prim_top_layer = top_layer
        self.prim_bound_box = self.array_box = bnd_box
        self.add_pin('sup', top_inst.get_all_port_pins('sup'), show=False)

    @classmethod
    def get_dimension(cls, grid, route_layer, bias_config, nwire, width=1, space_sig=0):
        # type: (RoutingGrid, int, Dict[int, Tuple[int, ...]], int, int, int) -> int

        blk_w, blk_h = BiasShield.get_block_size(grid, route_layer, bias_config, nwire,
                                                 width=width, space_sig=space_sig)

        route_dir = grid.get_direction(route_layer)
        is_horiz = (route_dir == 'x')

        sp_le = grid.get_line_end_space(route_layer, width, unit_mode=True)  # type: int
        min_len = grid.get_min_length(route_layer, width, unit_mode=True)  # type: int

        if is_horiz:
            min_len = max(min_len, blk_w)
            return -(-(sp_le + min_len) // blk_w) * blk_w
        else:
            min_len = max(min_len, blk_h)
            return -(-(sp_le + min_len) // blk_h) * blk_h


class BiasShieldCrossing(TemplateBase):
    """This template allows orthogonal bias bus to cross each other.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._hm_route_tids = None
        self._vm_route_tids = None

    @property
    def hm_route_tids(self):
        # type: () -> List[Tuple[int, int]]
        return self._hm_route_tids

    @property
    def vm_route_tids(self):
        # type: () -> List[Tuple[int, int]]
        return self._vm_route_tids

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            bot_layer='The bottom routing layer ID.',
            bias_config='The bias configuration dictionary.',
            bot_params='the bottom layer routing parameters.',
            top_params='the top layer routing parameters.',
            connect_shield='True to connect shield wires together.',
            draw_top='True to draw top shield.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            connect_shield=False,
            draw_top=True,
        )

    def draw_layout(self):
        # type: () -> None
        bot_layer = self.params['bot_layer']
        bias_config = self.params['bias_config']
        bot_params = self.params['bot_params']
        top_params = self.params['top_params']
        connect_shield = self.params['connect_shield']
        draw_top = self.params['draw_top']

        grid = self.grid

        top_layer = bot_layer + 1
        bot_dir = grid.get_direction(bot_layer)
        bot_horiz = (bot_dir == 'x')

        bot_w, bot_h = BiasShield.get_block_size(grid, bot_layer, bias_config, **bot_params)
        top_w, top_h = BiasShield.get_block_size(grid, top_layer, bias_config, **top_params)

        nx_bot = ny_bot = nx_top = ny_top = 1
        if bot_horiz:
            nx_bot = -(-top_w // bot_w)
            ny_top = -(-bot_h // top_h)

        else:
            ny_bot = -(-top_h // bot_h)
            nx_top = -(-bot_w // top_w)

        bot_params = bot_params.copy()
        bot_params['layer'] = bot_layer
        bot_params['bias_config'] = bias_config
        bot_params['top'] = False
        bb_master = self.new_template(params=bot_params, temp_cls=BiasShield)
        top_params = top_params.copy()
        top_params['layer'] = top_layer
        top_params['bias_config'] = bias_config
        top_params['top'] = True
        tt_master = self.new_template(params=top_params, temp_cls=BiasShield)

        if bot_horiz:
            self._hm_route_tids = bb_master.route_tids
            self._vm_route_tids = tt_master.route_tids
        else:
            self._hm_route_tids = tt_master.route_tids
            self._vm_route_tids = bb_master.route_tids

        bb_inst = self.add_instance(bb_master, 'XBB', loc=(0, 0), nx=nx_bot,
                                    ny=ny_bot, spx=bot_w, spy=bot_h, unit_mode=True)

        self.prim_top_layer = top_layer + 1
        self.prim_bound_box = self.array_box = bb_inst.bound_box

        bb_shields = self.connect_wires(bb_inst.get_all_port_pins('shield'))
        if draw_top:
            tt_inst = self.add_instance(tt_master, 'XTT', loc=(0, 0), nx=nx_top,
                                        ny=ny_top, spx=top_w, spy=top_h, unit_mode=True)
            if connect_shield:
                tt_shields = self.connect_wires(tt_inst.get_all_port_pins('shield'))
                self.connect_to_track_wires(bb_shields, tt_shields)

            self.add_pin('sup', tt_inst.get_all_port_pins('sup'), show=False)
        else:
            # still draw shield wires
            sh_warr = tt_master.get_port('shield').get_pins()
            upper = self.prim_bound_box.top_unit if bot_horiz else self.prim_bound_box.right_unit
            for warr in sh_warr:
                tid = warr.track_id
                self.add_wires(tid.layer_id, tid.base_index, 0, upper,
                               width=tid.width, num=tid.num, pitch=tid.pitch, unit_mode=True)
            if connect_shield:
                self.connect_to_track_wires(bb_shields, sh_warr)


class BiasShieldJoin(TemplateBase):
    """This template joins orthogonal bias routes together.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._hm_route_tids = None
        self._vm_route_tids = None

    @property
    def hm_route_tids(self):
        # type: () -> List[Tuple[int, int]]
        return self._hm_route_tids

    @property
    def vm_route_tids(self):
        # type: () -> List[Tuple[int, int]]
        return self._vm_route_tids

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            bot_layer='The bottom routing layer ID.',
            bias_config='The bias configuration dictionary.',
            bot_params='the bottom layer routing parameters.',
            top_params='the top layer routing parameters.',
            bot_open='True if bottom layer routing is open on both ends.',
            top_open='True if top layer routing is open on both ends.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            bot_open=False,
            top_open=False,
        )

    def draw_layout(self):
        # type: () -> None
        bot_layer = self.params['bot_layer']
        bias_config = self.params['bias_config']
        bot_open = self.params['bot_open']
        top_open = self.params['top_open']

        cross_params = self.params.copy()
        cross_params['connect_shield'] = True
        master = self.new_template(params=cross_params, temp_cls=BiasShieldCrossing)
        inst = self.add_instance(master, 'XCROSS', loc=(0, 0), unit_mode=True)
        self.array_box = bnd_box = master.bound_box

        sup_list = inst.get_all_port_pins('sup')
        self.add_pin('sup_core', sup_list, show=False)
        bot_horiz = self.grid.get_direction(bot_layer) == 'x'
        if not top_open:
            top_params = self.params['top_params'].copy()
            top_params['layer'] = bot_layer + 1
            top_params['bias_config'] = bias_config
            end_master = self.new_template(params=top_params, temp_cls=BiasShieldEnd)
            if bot_horiz:
                inst = self.add_instance(end_master, 'XENDT', loc=(0, bnd_box.top_unit),
                                         unit_mode=True)
            else:
                inst = self.add_instance(end_master, 'XENDT', loc=(bnd_box.right_unit, 0),
                                         unit_mode=True)
            sup_list.extend(inst.port_pins_iter('sup'))
            bnd_box = bnd_box.merge(inst.bound_box)
        if not bot_open:
            bot_params = self.params['bot_params'].copy()
            bot_params['layer'] = bot_layer
            bot_params['bias_config'] = bias_config
            end_master = self.new_template(params=bot_params, temp_cls=BiasShieldEnd)
            if bot_horiz:
                inst = self.add_instance(end_master, 'XENDB', loc=(bnd_box.right_unit, 0),
                                         unit_mode=True)
            else:
                inst = self.add_instance(end_master, 'XENDB', loc=(0, bnd_box.top_unit),
                                         unit_mode=True)
            sup_list.extend(inst.port_pins_iter('sup'))
            bnd_box = bnd_box.merge(inst.bound_box)

        self.add_pin('sup', sup_list, show=False)

        self.prim_top_layer = master.top_layer
        self.prim_bound_box = bnd_box
        self.add_cell_boundary(bnd_box)

        self._hm_route_tids = master.hm_route_tids
        self._vm_route_tids = master.vm_route_tids
