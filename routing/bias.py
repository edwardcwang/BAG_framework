# -*- coding: utf-8 -*-

"""This module defines bias routing related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any, Iterable, List, Union, Tuple

import numbers
from itertools import chain, repeat

from bag.math import lcm
from bag.util.interval import IntervalSet
from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing.base import TrackManager, TrackID, WireArray

if TYPE_CHECKING:
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
            space_sup='supplu wire spacing.',
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
    def get_shield_size(cls, template, layer, nwire, width=1, space_sig=0, space_sup=1):
        # type: (TemplateBase, int, int, int, int, Union[int, Tuple[int, int]]) -> Tuple[int, int]
        params = dict(
            layer=layer,
            nwire=nwire,
            top=False,
            width=width,
            space_sig=space_sig,
            space_sup=space_sup,
        )
        bot_master = template.new_template(params=params, temp_cls=BiasShield)
        box = bot_master.bound_box
        return box.width_unit, box.height_unit

    @classmethod
    def add_bias_shields(cls,
                         template,  # type: TemplateBase
                         layer,  # type: int
                         warr_list2,  # type: List[Union[WireArray, Iterable[WireArray]]]
                         x0,  # type: int,
                         y0,  # type: int
                         tr_upper,  # type: int
                         mode=1,  # type: int
                         width=1,  # type: int
                         space_sig=0,  # type: int
                         space_sup=1,  # type: Union[int, Tuple[int, int]]
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
            tr0 = grid.find_next_track(layer, y0, half_track=True,
                                       mode=mode, unit_mode=True)

            blk_off = grid.track_to_coord(layer, tr0 - mode * 0.5, unit_mode=True)
            tr_lower = x0
            orient = 'R0' if mode > 0 else 'MX'
        else:
            qdim = sh_box.height_unit
            tr0 = grid.find_next_track(layer, x0, half_track=True,
                                       mode=mode, unit_mode=True)
            blk_off = grid.track_to_coord(layer, tr0 - mode * 0.5, unit_mode=True)
            tr_lower = y0
            orient = 'R0' if mode > 0 else 'MY'

        tr_intv = (tr_lower, tr_upper)
        if tr_lower % qdim != 0 or tr_upper % qdim != 0:
            raise ValueError('track lower/upper = %s not divisible by %d' % (tr_intv, qdim))

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

        for master, intvs in zip((bot_master, top_master), (bot_intvs, top_intvs)):
            sl, su = master.sup_intv
            sl += tr_lower
            su += tr_lower
            for lower, upper in intvs.complement_iter(tr_intv):
                n0 = -(-(lower - sl) // qdim)
                n1 = (upper - su) // qdim
                nblk = n1 - n0 + 1
                if nblk > 0:
                    if is_horiz:
                        loc = (tr_lower + n0 * qdim, blk_off)
                        nx = nblk
                        ny = 1
                    else:
                        loc = (blk_off, tr_lower + n0 * qdim)
                        nx = 1
                        ny = nblk
                    template.add_instance(master, loc=loc, orient=orient, nx=nx, ny=ny,
                                          spx=qdim, spy=qdim, unit_mode=True)

        return tr_warr_list

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

        res = self.grid.resolution

        bot_layer = route_layer - 1
        top_layer = route_layer + 1
        route_dir = self.grid.get_direction(route_layer)
        is_horiz = route_dir == 'x'
        if is_horiz:
            half_blk_x = False
            half_blk_y = True
        else:
            half_blk_x = True
            half_blk_y = False
        blk_w, blk_h = self.grid.get_block_size(top_layer, unit_mode=True, half_blk_x=half_blk_x,
                                                half_blk_y=half_blk_y)
        bot_pitch = self.grid.get_track_pitch(bot_layer, unit_mode=True)
        top_pitch = self.grid.get_track_pitch(top_layer, unit_mode=True)
        route_pitch = self.grid.get_track_pitch(route_layer, unit_mode=True)

        tr_manager = TrackManager(self.grid, {'sig': {route_layer: width}},
                                  {('sig', ''): {route_layer: space_sig}}, half_space=True)

        tmp = [1]
        route_list = list(chain(tmp, repeat('sig', nwire), tmp))
        ntr, locs = tr_manager.place_wires(route_layer, route_list)
        par_dim = lcm([bot_pitch * (1 + space_sup[0]), top_pitch * (1 + space_sup[1])])
        perp_dim = ntr * route_pitch
        if is_horiz:
            tr_upper = -(-par_dim // blk_w) * blk_w
            tot_h = -(-perp_dim // blk_h) * blk_h
            bbox = BBox(0, 0, tr_upper, tot_h, res, unit_mode=True)

        else:
            tr_upper = -(-par_dim // blk_h) * blk_h
            tot_w = -(-perp_dim // blk_w) * blk_w
            bbox = BBox(0, 0, tot_w, tr_upper, res, unit_mode=True)

        self.set_size_from_bound_box(top_layer, bbox)
        self.array_box = bbox

        self._bias_tids = [(locs[idx], width) for idx in range(1, nwire + 1)]

        pitch = locs[nwire + 1] - locs[0]
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
