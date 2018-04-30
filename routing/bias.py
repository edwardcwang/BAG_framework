# -*- coding: utf-8 -*-

"""This module defines bias routing related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any

from itertools import chain, repeat

from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing.base import TrackManager, TrackID

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class BiasBusUnit(TemplateBase):
    """Unit cell template for bias bus.

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

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            route_layer='the routing layer.',
            nwire='number of routing wires.',
            width='route wire width.',
            space='route wire spacing.',
            show_pins='True to show pins.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            width=1,
            space=0,
            show_pins=True,
        )

    def draw_layout(self):
        # type: () -> None
        route_layer = self.params['route_layer']
        nwire = self.params['nwire']
        width = self.params['width']
        space = self.params['space']
        show_pins = self.params['show_pins']

        res = self.grid.resolution

        top_layer = route_layer + 1
        is_horiz = self.grid.get_direction(route_layer) == 'x'
        blk_w, blk_h = self.grid.get_block_size(top_layer, unit_mode=True)
        route_pitch = self.grid.get_track_pitch(route_layer, unit_mode=True)

        tr_manager = TrackManager(self.grid, {'sig': {route_layer: width}},
                                  {('sig', ''): {route_layer: space}}, half_space=True)

        tmp = [1]
        route_list = list(chain(tmp, repeat('sig', nwire), tmp))
        ntr, locs = tr_manager.place_wires(route_layer, route_list)
        if is_horiz:
            tr_upper = blk_w
            tot_h = -(-(ntr * route_pitch) // blk_h) * blk_h
            bbox = BBox(0, 0, tr_upper, tot_h, res, unit_mode=True)

        else:
            tr_upper = blk_h
            tot_w = -(-(ntr * route_pitch) // blk_w) * blk_w
            bbox = BBox(0, 0, tot_w, tr_upper, res, unit_mode=True)

        self.set_size_from_bound_box(top_layer, bbox)
        self.array_box = bbox

        for idx in range(1, len(route_list) - 1):
            warr = self.add_wires(route_layer, locs[idx], 0, tr_upper, width=width, unit_mode=True)
            self.add_pin('sig%d' % (idx - 1), warr, show=show_pins)

        tidx = locs[0]
        sh_warr = self.add_wires(route_layer, tidx, 0, tr_upper, num=2, pitch=locs[-1] - tidx)
        for lay in (route_layer - 1, top_layer):
            cur_pitch = self.grid.get_track_pitch(lay, unit_mode=True)
            ntr = tr_upper // cur_pitch
            exp = self.connect_to_tracks(sh_warr, TrackID(lay, 0, num=ntr, pitch=1))

        self.add_pin('sup', exp, show=show_pins)
