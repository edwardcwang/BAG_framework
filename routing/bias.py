# -*- coding: utf-8 -*-

"""This module defines bias routing related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any, Iterable, List, Union, Tuple, Generator

import numbers
from itertools import chain, repeat

from bag.math import lcm
from bag.util.interval import IntervalSet
from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing.base import TrackManager, TrackID, WireArray

if TYPE_CHECKING:
    from bag.layout.routing.grid import RoutingGrid
    from bag.layout.template import TemplateDB


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
        self._bias_tids = None
        self._sup_intv = None

    @property
    def bias_tids(self):
        return self._bias_tids

    @property
    def sup_intv(self):
        return self._sup_intv

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layer='the routing layer.',
            nwire='number of routing wires.',
            top='True to draw top shield.',
            width='route wire width.',
            space_sig='route wire spacing.',
            space_sup='supply wire spacing.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            top=True,
            width=1,
            space_sig=0,
            space_sup=1,
        )

    def get_layout_basename(self):
        layer = self.params['layer']
        nwire = self.params['nwire']
        desc = 'top' if self.params['top'] else 'bot'
        return 'bias_shield_%s_lay%d_n%d' % (desc, layer, nwire)

    @classmethod
    def get_block_size(cls, grid, route_layer, nwire, width=1, space_sig=0, space_sup=1):
        # type: (RoutingGrid, int, int, int, int, Union[int, Tuple[int, int]]) -> Tuple[int, int]
        if isinstance(space_sup, numbers.Integral):
            space_sup = (space_sup, space_sup)

        bot_layer = route_layer - 1
        top_layer = route_layer + 1
        route_dir = grid.get_direction(route_layer)
        is_horiz = route_dir == 'x'
        if is_horiz:
            half_blk_x = False
            half_blk_y = True
        else:
            half_blk_x = True
            half_blk_y = False
        blk_w, blk_h = grid.get_block_size(top_layer, unit_mode=True, half_blk_x=half_blk_x,
                                           half_blk_y=half_blk_y)

        bot_pitch = grid.get_track_pitch(bot_layer, unit_mode=True)
        top_pitch = grid.get_track_pitch(top_layer, unit_mode=True)
        route_pitch = grid.get_track_pitch(route_layer, unit_mode=True)

        tr_manager = TrackManager(grid, {'sig': {route_layer: width}},
                                  {('sig', ''): {route_layer: space_sig}}, half_space=True)

        tmp = [1]
        route_list = list(chain(tmp, repeat('sig', nwire), tmp))
        ntr, _ = tr_manager.place_wires(route_layer, route_list)
        par_dim = lcm([bot_pitch * (1 + space_sup[0]), top_pitch * (1 + space_sup[1])])
        perp_dim = int(round(ntr * route_pitch))
        if is_horiz:
            par_dim = -(-par_dim // blk_w) * blk_w
            return par_dim, perp_dim
        else:
            par_dim = -(-par_dim // blk_h) * blk_h
            return perp_dim, par_dim

    @classmethod
    def add_bias_shields(cls,
                         template,  # type: TemplateBase
                         layer,  # type: int
                         warr_list2,  # type: List[Union[WireArray, Iterable[WireArray]]]
                         tr0,  # type: int
                         mode=1,  # type: int
                         width=1,  # type: int
                         space_sig=0,  # type: int
                         space_sup=1,  # type: Union[int, Tuple[int, int]]
                         lu_end_mode=0,  # type: int
                         ):
        # type: (...) -> List[WireArray]
        grid = template.grid

        nwire = len(warr_list2)
        params = dict(
            layer=layer,
            nwire=nwire,
            top=False,
            width=width,
            space_sig=space_sig,
            space_sup=space_sup,
        )
        bot_master = template.new_template(params=params, temp_cls=BiasShield)
        sh_box = bot_master.bound_box
        params['top'] = True
        top_master = template.new_template(params=params, temp_cls=BiasShield)

        bias_tids = bot_master.bias_tids
        tr_dir = grid.get_direction(layer)
        is_horiz = tr_dir == 'x'
        if is_horiz:
            qdim = sh_box.width_unit
            blk_off = grid.track_to_coord(layer, tr0 - mode * 0.5, unit_mode=True)
            orient = 'R0' if mode > 0 else 'MX'
        else:
            qdim = sh_box.height_unit
            blk_off = grid.track_to_coord(layer, tr0 - mode * 0.5, unit_mode=True)
            orient = 'R0' if mode > 0 else 'MY'

        bot_warrs = []
        top_warrs = []
        tr_warr_list = []
        bot_intvs = IntervalSet()
        top_intvs = IntervalSet()
        for warr_list, (tidx, tr_width) in zip(warr_list2, bias_tids):
            if isinstance(warr_list, WireArray):
                warr_list = [warr_list]

            cur_tid = TrackID(layer, mode * tidx + tr0, width=width)
            tr_warr_list.append(template.connect_to_tracks(warr_list, cur_tid))

            for warr in warr_list:
                cur_layer = warr.layer_id
                if cur_layer == layer - 1:
                    bot_warrs.append(warr)
                    cur_intvs = bot_intvs
                elif cur_layer == layer + 1:
                    top_warrs.append(warr)
                    cur_intvs = top_intvs
                else:
                    raise ValueError('Cannot connect to wire %s' % warr)

                cur_width = warr.width
                sp = grid.get_space(cur_layer, cur_width, unit_mode=True)
                box_arr = warr.get_bbox_array(grid)
                for box in box_arr:
                    wl, wu = box.get_interval(tr_dir, unit_mode=True)
                    cur_intvs.add((wl - sp, wu + sp), merge=True, abut=True)

        # find nstart/nstop
        nstart = nstop = None
        iter_list = [(bot_master, bot_intvs), (top_master, top_intvs)]
        for master, intvs in iter_list:
            sl, su = master.sup_intv
            nend = 1 + (intvs.get_interval(0)[0] - su) // qdim
            nnext = -(-(intvs.get_interval(-1)[-1] - sl) // qdim)
            if nstart is None:
                nstart = nend
            else:
                nstart = min(nstart, nend)
            if nstop is None:
                nstop = nnext
            else:
                nstop = max(nstop, nnext)

        # draw blocks
        for master, intvs in iter_list:
            sl, su = master.sup_intv
            ncur = nstart
            for nend, nnext in cls._get_blk_idx_iter(intvs, sl, su, qdim, nstart, nstop):
                nblk = nend - ncur
                if nblk > 0:
                    if is_horiz:
                        loc = (ncur * qdim, blk_off)
                        nx = nblk
                        ny = 1
                    else:
                        loc = (blk_off, ncur * qdim)
                        nx = 1
                        ny = nblk
                    template.add_instance(master, loc=loc, orient=orient, nx=nx, ny=ny,
                                          spx=qdim, spy=qdim, unit_mode=True)
                ncur = nnext

        if lu_end_mode != 0:
            end_master = template.new_template(params=params, temp_cls=BiasShieldEnd)
            if lu_end_mode & 2 != 0:
                eorient = 'R180' if mode < 0 else ('MY' if is_horiz else 'MX')
                nend = nstart
            else:
                eorient = orient
                nend = nstop
            if is_horiz:
                loc = (nend * qdim, blk_off)
            else:
                loc = (blk_off, nend * qdim)
            template.add_instance(end_master, loc=loc, orient=eorient, unit_mode=True)

        return tr_warr_list

    @classmethod
    def _get_blk_idx_iter(cls, intvs, sl, su, qdim, nstart, nstop):
        # type: (IntervalSet, int, int, int, int, int) -> Generator[Tuple[int, int], None, None]
        for lower, upper in intvs:
            nend = 1 + (lower - su) // qdim
            if nend < nstart:
                raise ValueError('wire interval (%d, %d) is out of bounds.' % (lower, upper))
            ncur = -(-(upper - sl) // qdim)
            yield nend, ncur
        yield nstop, nstop

    def draw_layout(self):
        # type: () -> None
        route_layer = self.params['layer']
        nwire = self.params['nwire']
        width = self.params['width']
        space_sig = self.params['space_sig']
        top = self.params['top']
        space_sup = self.params['space_sup']

        if isinstance(space_sup, numbers.Integral):
            space_sup = (space_sup, space_sup)

        grid = self.grid
        res = grid.resolution

        bot_layer = route_layer - 1
        top_layer = route_layer + 1
        bot_pitch = grid.get_track_pitch(bot_layer, unit_mode=True)
        top_pitch = grid.get_track_pitch(top_layer, unit_mode=True)
        route_dir = grid.get_direction(route_layer)

        tot_w, tot_h = self.get_block_size(grid, route_layer, nwire, width=width,
                                           space_sig=space_sig, space_sup=space_sup)
        bbox = BBox(0, 0, tot_w, tot_h, res, unit_mode=True)
        self.prim_top_layer = top_layer
        self.prim_bound_box = bbox
        self.array_box = bbox

        tr_manager = TrackManager(grid, {'sig': {route_layer: width}},
                                  {('sig', ''): {route_layer: space_sig}}, half_space=True)

        tmp = [1]
        route_list = list(chain(tmp, repeat('sig', nwire), tmp))
        ntr, locs = tr_manager.place_wires(route_layer, route_list)

        self._bias_tids = [(locs[idx], width) for idx in range(1, nwire + 1)]

        pitch = locs[nwire + 1] - locs[0]
        tr_upper = bbox.width_unit if route_dir == 'x' else bbox.height_unit
        sh_warr = self.add_wires(route_layer, locs[0], 0, tr_upper, num=2, pitch=pitch,
                                 unit_mode=True)
        if top:
            sup_pitch = space_sup[1] + 1
            sup_tid = TrackID(top_layer, space_sup[1] / 2,
                              num=(tr_upper // (sup_pitch * top_pitch)), pitch=sup_pitch)
            warr = self.connect_to_tracks(sh_warr, sup_tid)
        else:
            sup_pitch = space_sup[0] + 1
            sup_tid = TrackID(bot_layer, space_sup[0] / 2,
                              num=(tr_upper // (sup_pitch * bot_pitch)), pitch=sup_pitch)
            warr = self.connect_to_tracks(sh_warr, sup_tid)
        self.add_pin('sup', warr, show=False)

        sup_box = warr.get_bbox_array(self.grid).get_overall_bbox()
        self._sup_intv = sup_box.get_interval(route_dir, unit_mode=True)


class BiasShieldEnd(TemplateBase):
    """end cap of biasl shield wires.

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
            width='route wire width.',
            space_sig='route wire spacing.',
            space_sup='supply wire spacing.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            width=1,
            space_sig=0,
            space_sup=1,
        )

    def get_layout_basename(self):
        layer = self.params['layer']
        nwire = self.params['nwire']
        return 'bias_shield_end_lay%d_n%d' % (layer, nwire)

    def draw_layout(self):
        # type: () -> None
        route_layer = self.params['layer']
        nwire = self.params['nwire']
        width = self.params['width']

        grid = self.grid

        top_layer = route_layer + 1
        route_dir = grid.get_direction(route_layer)
        is_horiz = (route_dir == 'x')

        params = self.params.copy()
        params['top'] = False
        bot_master = self.new_template(params=params, temp_cls=BiasShield)
        params['top'] = True
        top_master = self.new_template(params=params, temp_cls=BiasShield)
        bot_box = bot_master.bound_box
        blk_w = bot_box.width_unit
        blk_h = bot_box.height_unit
        sp_le = grid.get_line_end_space(route_layer, width, unit_mode=True)
        min_len = grid.get_min_length(route_layer, width, unit_mode=True)

        orig = (0, 0)
        nx = ny = 1
        if is_horiz:
            min_len = max(min_len, blk_w)
            nx = nblk = -(-(sp_le + min_len) // blk_w)  # type: int
            cr = nblk * blk_w
        else:
            min_len = max(min_len, blk_h)
            ny = nblk = -(-(sp_le + min_len) // blk_h)  # type: int
            cr = nblk * blk_h
        self._nblk = nblk

        bot_inst = self.add_instance(bot_master, 'XBOT', loc=orig, nx=nx, ny=ny,
                                     spx=blk_w, spy=blk_h, unit_mode=True)
        top_inst = self.add_instance(top_master, 'XTOP', loc=orig, nx=nx, ny=ny,
                                     spx=blk_w, spy=blk_h, unit_mode=True)

        (tidx0, tr_w), (tidx1, _) = bot_master.bias_tids[:2]
        warr = self.add_wires(route_layer, tidx0, cr - min_len, cr, width=tr_w, num=nwire,
                              pitch=tidx1 - tidx0, unit_mode=True)
        bot_port = bot_inst.get_port('sup', row=ny - 1, col=nx - 1)
        top_port = top_inst.get_port('sup', row=ny - 1, col=nx - 1)
        self.connect_to_track_wires(warr, bot_port.get_pins())
        self.connect_to_track_wires(warr, top_port.get_pins())

        bnd_box = bot_inst.bound_box
        self.prim_top_layer = top_layer
        self.prim_bound_box = bnd_box
        self.array_box = bnd_box
        self.add_pin('sup', top_inst.get_all_port_pins('sup'), show=False)
