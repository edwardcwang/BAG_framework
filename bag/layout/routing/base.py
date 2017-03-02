# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""This module provides basic routing classes.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Tuple, Union

from ..util import BBox, BBoxArray
from .grid import RoutingGrid


class TrackID(object):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : float
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : float
        pitch between adjacent tracks, in number of track pitches.
    """

    def __init__(self, layer_id, track_idx, width=1, num=1, pitch=0.0):
        self._layer_id = layer_id
        self._idx = track_idx
        self._w = width
        self._n = num
        self._pitch = pitch

    @property
    def layer_id(self):
        # type: () -> int
        return self._layer_id

    @property
    def width(self):
        return self._w

    @property
    def base_index(self):
        return self._idx

    @property
    def num(self):
        return self._n

    @property
    def pitch(self):
        return self._pitch

    def get_bounds(self, grid, unit_mode=False):
        # type: (RoutingGrid, bool) -> Tuple[Union[float, int], Union[float, int]]
        """Calculate the track bounds coordinate.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        unit_mode : bool
            True to return coordinates in resolution units.

        Returns
        -------
        lower : Union[float, int]
            the lower bound coordinate perpendicular to track direction.
        upper : Union[float, int]
            the upper bound coordinate perpendicular to track direction.
        """
        lower, upper = grid.get_wire_bounds(self.layer_id, self.base_index,
                                            width=self.width, unit_mode=True)
        upper += (self.num - 1) * self.pitch * grid.get_track_pitch(self.layer_id, unit_mode=True)
        if unit_mode:
            return lower, int(upper)
        else:
            res = grid.resolution
            return lower * res, upper * res

    def __iter__(self):
        """Iterate over all middle track indices in this TrackID."""
        for idx in range(self._n):
            yield self._idx + idx * self._pitch

    def sub_tracks_iter(self, grid):
        """Iterate through sub-TrackIDs where every track in sub-TrackID has the same layer name.

        This method is used to deal with double patterning layer.  If this TrackID is not
        on a double patterning layer, it simply yields itself.

        Yields
        ------
        sub_id : TrackID
            a TrackID where all tracks has the same layer name.
        """
        layer_id = self.layer_id
        pitch = self.pitch
        layer_names = grid.tech_info.get_layer_name(layer_id)
        nlayer = len(layer_names)
        if isinstance(layer_names, tuple) and pitch % nlayer != 0:
            # double patterning layer
            num = self.num
            base_idx = self.base_index
            nlayer = len(layer_names)
            q, r = divmod(num, nlayer)
            tr_pitch = pitch * nlayer
            for idx in range(min(nlayer, num)):
                cur_idx = base_idx + idx * pitch
                cur_num = q + 1 if idx < r else q
                yield TrackID(layer_id, cur_idx, self.width, num=cur_num, pitch=tr_pitch)
            pass
        else:
            yield self


class WireArray(object):
    """An array of wires on the routing grid.

    Parameters
    ----------
    track_id : :class:`bag.layout.routing.TrackID
        TrackArray representing the track locations of this wire array.
    lower : float
        the lower coordinate along the track direction.
    upper : float
        the upper coordinate along the track direction.
    """

    def __init__(self, track_id, lower, upper):
        self._track_id = track_id
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def middle(self):
        return (self._lower + self._upper) / 2

    @property
    def track_id(self):
        # type: () -> TrackID
        """Returns the TrackID of this WireArray."""
        return self._track_id

    @property
    def layer_id(self):
        # type: () -> int
        """Returns the layer ID of this WireArray."""
        return self.track_id.layer_id

    @property
    def width(self):
        return self.track_id.width

    def get_bbox_array(self, grid):
        # type: ('RoutingGrid') -> BBoxArray
        """Returns the BBoxArray representing this WireArray.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this WireArray.

        Returns
        -------
        bbox_arr : BBoxArray
            the BBoxArray of the wires.
        """
        track_id = self.track_id
        tr_w = track_id.width
        layer_id = track_id.layer_id
        base_idx = track_id.base_index
        num = track_id.num
        pitch = track_id.pitch

        base_box = grid.get_bbox(layer_id, base_idx, self._lower, self._upper, width=tr_w)
        if grid.get_direction(layer_id) == 'x':
            return BBoxArray(base_box, ny=num, spy=pitch * grid.get_track_pitch(layer_id))
        else:
            return BBoxArray(base_box, nx=num, spx=pitch * grid.get_track_pitch(layer_id))

    def wire_iter(self, grid):
        """Iterate over all wires in this WireArray as layer/BBox pair.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : string
            the wire layer name.
        bbox : :class:`bag.layout.util.BBox`
            the wire bounding box.
        """
        tr_w = self.track_id.width
        layer_id = self.layer_id
        for tr_idx in self.track_id:
            layer_name = grid.get_layer_name(layer_id, tr_idx)
            bbox = grid.get_bbox(layer_id, tr_idx, self._lower, self._upper, width=tr_w)
            yield layer_name, bbox

    def wire_arr_iter(self, grid):
        """Iterate over all wires in this WireArray as layer/BBoxArray pair.

        This method group all rectangles in the same layer together.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : string
            the wire layer name.
        bbox : :class:`bag.layout.util.BBoxArray`
            the wire bounding boxes.
        """
        tid = self.track_id
        layer_id = tid.layer_id
        num = tid.num
        pitch = tid.pitch
        base_idx = tid.base_index
        tr_width = tid.width
        direction = grid.get_direction(layer_id)
        wire_pitch = pitch * grid.get_track_pitch(layer_id)

        layer_names = grid.tech_info.get_layer_name(layer_id)
        nlayer = len(layer_names)
        if isinstance(layer_names, tuple) and pitch % nlayer != 0:
            # double patterning layer and layer name will change
            q, r = divmod(num, nlayer)
            wire_pitch *= nlayer
            for idx in range(min(nlayer, num)):
                cur_idx = base_idx + idx * pitch
                cur_layer = grid.get_layer_name(layer_id, cur_idx)
                bbox = grid.get_bbox(layer_id, cur_idx, self._lower, self._upper, width=tr_width)
                cur_num = q + 1 if idx < r else q
                if direction == 'x':
                    yield cur_layer, BBoxArray(bbox, ny=cur_num, spy=wire_pitch)
                else:
                    yield cur_layer, BBoxArray(bbox, nx=cur_num, spx=wire_pitch)
        else:
            cur_layer = grid.get_layer_name(layer_id, base_idx)
            bbox = grid.get_bbox(layer_id, base_idx, self._lower, self._upper, width=tr_width)
            if direction == 'x':
                yield cur_layer, BBoxArray(bbox, ny=num, spy=wire_pitch)
            else:
                yield cur_layer, BBoxArray(bbox, nx=num, spx=wire_pitch)

    def transform(self, grid, loc=(0, 0), orient='R0'):
        """Return a new transformed WireArray.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.
        loc : tuple(float, float)
            the X/Y coordinate shift.
        orient : string
            the new orientation.
        """
        track_id = self.track_id
        tr_w = track_id.width
        layer_id = track_id.layer_id
        num = track_id.num
        pitch = track_id.pitch

        box_arr = self.get_bbox_array(grid)
        new_box_arr = box_arr.transform(loc=loc, orient=orient)
        new_box_base = new_box_arr.base
        if grid.get_direction(layer_id) == 'x':
            new_tr_idx = grid.coord_to_track(layer_id, new_box_base.yc)
            lower, upper = new_box_base.left, new_box_base.right
        else:
            new_tr_idx = grid.coord_to_track(layer_id, new_box_base.xc)
            lower, upper = new_box_base.bottom, new_box_base.top

        new_tr_id = TrackID(layer_id, new_tr_idx, tr_w, num=num, pitch=pitch)
        return WireArray(new_tr_id, lower, upper)


class Port(object):
    """A layout port.

    a port is a group of pins that represent the same net.
    The pins can be on different layers.

    Parameters
    ----------
    term_name : str
        the terminal name of the port.
    pin_dict : dict[int, list[bag.layout.routing.WireArray]]
        a dictionary from layer ID to pin geometries on that layer.
    """

    def __init__(self, term_name, pin_dict):
        self._term_name = term_name
        self._pin_dict = pin_dict

    def __iter__(self):
        """Iterate through all pin geometries in this port.

        the iteration order is not guaranteed.
        """
        for wire_arr_list in self._pin_dict.values():
            for wire_arr in wire_arr_list:
                yield wire_arr

    def _get_layer(self, layer):
        """Get the layer number."""
        if layer < 0:
            if len(self._pin_dict) > 1:
                raise ValueError('This port has more than one layer.')
            layer = next(iter(self._pin_dict))
        return layer

    @property
    def net_name(self):
        """Returns the net name of this port."""
        return self._term_name

    def get_pins(self, layer=-1):
        """Returns the pin geometries on the given layer.

        Parameters
        ----------
        layer : int
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        track_bus_list : list[:class:`~bag.layout.routing.WireArray`]
            pins on the given layer representing as WireArrays.
        """
        return self._pin_dict[self._get_layer(layer)]

    def get_bounding_box(self, grid, layer=-1):
        """Calculate the overall bounding box of this port on the given layer.

        Parameters
        ----------
        grid : :class:`~bag.layout.routing.RoutingGrid`
            the RoutingGrid of this Port.
        layer : int
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        bbox : :class:`~bag.layout.util.BBox`
            the bounding box.
        """
        layer = self._get_layer(layer)
        box = BBox.get_invalid_bbox()
        for warr in self._pin_dict[layer]:
            box = box.merge(warr.get_bbox_array(grid).get_overall_bbox())
        return box

    def transform(self, grid, loc=(0, 0), orient='R0'):
        """Return a new transformed Port.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this Port.
        loc : tuple(float, float)
            the X/Y coordinate shift.
        orient : string
            the new orientation.
        """
        new_pin_dict = {lay: [wa.transform(grid, loc=loc, orient=orient) for wa in wa_list]
                        for lay, wa_list in self._pin_dict.items()}
        return Port(self.net_name, new_pin_dict)
