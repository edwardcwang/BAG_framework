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

from typing import Tuple, Union, Generator

from ..util import BBox, BBoxArray
from .grid import RoutingGrid


class TrackID(object):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : Union[float, int]
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : Union[float, int]
        pitch between adjacent tracks, in number of track pitches.
    """

    def __init__(self, layer_id, track_idx, width=1, num=1, pitch=0.0):
        # type: (int, Union[float, int], int, int, Union[float, int]) -> None
        self._layer_id = layer_id
        self._hidx = int(round(2 * track_idx)) + 1
        self._w = width
        self._n = num
        self._hpitch = int(pitch * 2)

    @property
    def layer_id(self):
        # type: () -> int
        return self._layer_id

    @property
    def width(self):
        # type: () -> int
        return self._w

    @property
    def base_index(self):
        # type: () -> Union[float, int]
        if self._hidx % 2 == 1:
            return (self._hidx - 1) // 2
        return (self._hidx - 1) / 2

    @property
    def num(self):
        # type: () -> int
        return self._n

    @property
    def pitch(self):
        # type: () -> Union[float, int]
        if self._hpitch % 2 == 0:
            return self._hpitch // 2
        return self._hpitch / 2

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
        upper += (self.num - 1) * self.pitch * grid.get_track_pitch(self._layer_id, unit_mode=True)
        if unit_mode:
            return lower, int(upper)
        else:
            res = grid.resolution
            return lower * res, upper * res

    def __iter__(self):
        # type: () -> Generator[Union[float, int]]
        """Iterate over all middle track indices in this TrackID."""
        for idx in range(self._n):
            num = self._hidx + idx * self._hpitch
            if num % 2 == 1:
                yield (num - 1) // 2
            else:
                yield (num - 1) / 2

    def sub_tracks_iter(self, grid):
        # type: (RoutingGrid) -> Generator[TrackID]
        """Iterate through sub-TrackIDs where every track in sub-TrackID has the same layer name.

        This method is used to deal with double patterning layer.  If this TrackID is not
        on a double patterning layer, it simply yields itself.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.

        Yields
        ------
        sub_id : TrackID
            a TrackID where all tracks has the same layer name.
        """
        layer_id = self._layer_id
        layer_names = grid.tech_info.get_layer_name(layer_id)
        if isinstance(layer_names, tuple):
            den = 2 * len(layer_names)
            if self._hpitch % den == 0:
                # layer name will never change
                yield self
            else:
                # TODO: have more robust solution than just yielding tracks one by one?
                for tr_idx in self:
                    yield TrackID(layer_id, tr_idx, width=self.width)
        else:
            yield self

    def transform(self, grid, loc=(0, 0), orient="R0", unit_mode=False):
        # type: (RoutingGrid, Tuple[Union[float, int], Union[float, int]], str, bool) -> TrackID
        """returns a transformation of this TrackID."""
        layer_id = self._layer_id
        is_x = grid.get_direction(layer_id) == 'x'
        if orient == 'R180' or (is_x and orient == 'MX') or (not is_x and orient == 'MY'):
            base_hidx = -self._hidx - (self._n - 1) * self._hpitch
        else:
            base_hidx = self._hidx

        delta = loc[1] if is_x else loc[0]
        delta = grid.coord_to_track(layer_id, delta, unit_mode=unit_mode) + 0.5
        return TrackID(layer_id, (base_hidx - 1) / 2 + delta, width=self._w, num=self._n, pitch=self.pitch)


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
        tr_width = tid.width
        track_pitch = grid.get_track_pitch(layer_id, unit_mode=True)
        res = grid.resolution
        lower_unit = int(round(self._lower / res))
        upper_unit = int(round(self._upper / res))
        is_x = grid.get_direction(layer_id) == 'x'
        for track_idx in tid.sub_tracks_iter(grid):
            base_idx = track_idx.base_index
            cur_layer = grid.get_layer_name(layer_id, base_idx)
            cur_num = track_idx.num
            wire_pitch = track_idx.pitch * track_pitch
            tl, tu = grid.get_wire_bounds(layer_id, base_idx, width=tr_width, unit_mode=True)
            if is_x:
                base_box = BBox(lower_unit, tl, upper_unit, tu, res, unit_mode=True)
                box_arr = BBoxArray(base_box, ny=cur_num, spy=wire_pitch, unit_mode=True)
            else:
                base_box = BBox(tl, lower_unit, tu, upper_unit, res, unit_mode=True)
                box_arr = BBoxArray(base_box, nx=cur_num, spx=wire_pitch, unit_mode=True)

            yield cur_layer, box_arr

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
        layer_id = self.layer_id
        is_x = grid.get_direction(layer_id) == 'x'
        if orient == 'R180' or (is_x and orient == 'MY') or (not is_x and orient == 'MX'):
            lower, upper = -self._upper, -self._lower
        else:
            lower, upper = self._lower, self._upper

        delta = loc[0] if is_x else loc[1]
        return WireArray(self.track_id.transform(grid, loc=loc, orient=orient), lower + delta, upper + delta)


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
        layer_id = self._get_layer(layer)
        return self._pin_dict.get(layer_id, [])

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
