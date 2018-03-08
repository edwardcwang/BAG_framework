# -*- coding: utf-8 -*-

"""This module provides basic routing classes.
"""

from typing import Tuple, Union, Generator, Dict, List, Sequence

from ...util.search import BinaryIterator
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
        if num < 1:
            raise ValueError('TrackID must have 1 or more tracks.')

        self._layer_id = layer_id
        self._hidx = int(round(2 * track_idx)) + 1
        self._w = width
        self._n = num
        self._hpitch = 0 if num == 1 else int(pitch * 2)

    def __repr__(self):
        arg_list = ['layer=%d' % self._layer_id]
        if self._hidx % 2 == 1:
            arg_list.append('track=%d' % ((self._hidx - 1) // 2))
        else:
            arg_list.append('track=%.1f' % ((self._hidx - 1) / 2))
        if self._w != 1:
            arg_list.append('width=%d' % self._w)
        if self._n != 1:
            arg_list.append('num=%d' % self._n)
            if self._hpitch % 2 == 0:
                arg_list.append('pitch=%d' % (self._hpitch // 2))
            else:
                arg_list.append('pitch=%.1f' % (self._hpitch / 2))

        return '%s(%s)' % (self.__class__.__name__, ', '.join(arg_list))

    def __str__(self):
        return repr(self)

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
        if orient == 'R0':
            base_hidx = self._hidx
        elif orient == 'MX':
            if is_x:
                base_hidx = -self._hidx - (self._n - 1) * self._hpitch
            else:
                base_hidx = self._hidx
        elif orient == 'MY':
            if is_x:
                base_hidx = self._hidx
            else:
                base_hidx = -self._hidx - (self._n - 1) * self._hpitch
        elif orient == 'R180':
            base_hidx = -self._hidx - (self._n - 1) * self._hpitch
        else:
            raise ValueError('Unsupported orientation: %s' % orient)

        delta = loc[1] if is_x else loc[0]
        delta = grid.coord_to_track(layer_id, delta, unit_mode=unit_mode) + 0.5
        return TrackID(layer_id, (base_hidx - 1) / 2 + delta, width=self._w,
                       num=self._n, pitch=self.pitch)


class WireArray(object):
    """An array of wires on the routing grid.

    Parameters
    ----------
    track_id : :class:`bag.layout.routing.TrackID`
        TrackArray representing the track locations of this wire array.
    lower : float
        the lower coordinate along the track direction.
    upper : float
        the upper coordinate along the track direction.
    """

    def __init__(self, track_id, lower, upper):
        # type: (TrackID, float, float) -> None
        self._track_id = track_id
        self._lower = lower
        self._upper = upper

    def __repr__(self):
        return '%s(%s, %.4g, %.4g)' % (self.__class__.__name__, self._track_id,
                                       self._lower, self._upper)

    def __str__(self):
        return repr(self)

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

    @classmethod
    def list_to_warr(cls, warr_list):
        # type: (List[WireArray]) -> WireArray
        """Convert a list of WireArrays to a single WireArray.

        this method assumes all WireArrays have the same layer and lower and upper coordinates.
        Overlapping WireArrays will be compacted.
        """
        if len(warr_list) == 1:
            return warr_list[0]

        layer = warr_list[0].layer_id
        lower, upper = warr_list[0].lower, warr_list[0].upper
        tid_list = sorted(set((int(idx * 2) for warr in warr_list for idx in warr.track_id)))
        base_idx2 = tid_list[0]
        diff = tid_list[1] - tid_list[0]
        for idx in range(1, len(tid_list) - 1):
            if tid_list[idx + 1] - tid_list[idx] != diff:
                raise ValueError('pitch mismatch.')
        base_idx = base_idx2 // 2 if base_idx2 % 2 == 0 else base_idx2 / 2
        pitch = diff // 2 if diff % 2 == 0 else diff / 2

        return WireArray(TrackID(layer, base_idx, num=len(tid_list), pitch=pitch), lower, upper)

    def to_warr_list(self):
        tid = self._track_id
        layer = tid.layer_id
        width = tid.width
        return [WireArray(TrackID(layer, tr, width=width), self._lower, self._upper) for tr in tid]

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
        if orient == 'R0':
            lower, upper = self._lower, self._upper
        elif orient == 'MX':
            if is_x:
                lower, upper = self._lower, self._upper
            else:
                lower, upper = -self._upper, -self._lower
        elif orient == 'MY':
            if is_x:
                lower, upper = -self._upper, -self._lower
            else:
                lower, upper = self._lower, self._upper
        elif orient == 'R180':
            lower, upper = -self._upper, -self._lower
        else:
            raise ValueError('Unsupported orientation: %s' % orient)

        delta = loc[0] if is_x else loc[1]
        return WireArray(self.track_id.transform(grid, loc=loc, orient=orient),
                         lower + delta, upper + delta)


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


class TrackManager(object):
    """A class that makes it easy to compute track locations.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    tr_widths : Dict[str, Dict[int, int]]
        dictionary from wire types to its width on each layer.
    tr_spaces : Dict[Union[str, Tuple[str, str]], Dict[int, Union[float, int]]]
        dictionary from wire types to its spaces on each layer.
    **kwargs :
        additional options.
    """
    def __init__(self,
                 grid,  # type: RoutingGrid
                 tr_widths,  # type: Dict[str, Dict[int, int]]
                 tr_spaces,  # type: Dict[Union[str, Tuple[str, str]], Dict[int, Union[float, int]]]
                 **kwargs,
                 ):
        # type: (...) -> None
        half_space = kwargs.get('half_space', False)

        self._grid = grid
        self._tr_widths = tr_widths
        self._tr_spaces = tr_spaces
        self._half_space = half_space

    @property
    def grid(self):
        # type: () -> RoutingGrid
        return self._grid

    @property
    def half_space(self):
        # type: () -> bool
        return self._half_space

    def get_width(self, layer_id, wire_name):
        # type: (int, str) -> int
        """Returns the track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        wire_name : str
            the track type.
        """
        if wire_name not in self._tr_widths:
            return 1
        return self._tr_widths[wire_name].get(layer_id, 1)

    def get_space(self, layer_id, name_tuple, **kwargs):
        # type: (int, Union[str, Tuple[str, str]], **kwargs) -> Union[int, float]
        """Returns the track spacing.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        name_tuple : Union[str, Tuple[str, str]]
            If a single string is given, will return the minimum spacing needed around that track
            type.  If a tuple of two strings are given, will return the specific spacing between
            those two track types if specified.  Otherwise, returns the maximum of all the
            valid spacing.
        **kwargs:
            optional parameters.
        """
        half_space = kwargs.get('half_space', self._half_space)
        sp_override = kwargs.get('sp_override', None)

        if isinstance(name_tuple, tuple):
            # if two specific wires are given, first check if any specific rules exist
            ans = self._get_space_from_tuple(layer_id, name_tuple, sp_override)
            if ans is not None:
                return ans
            ans = self._get_space_from_tuple(layer_id, name_tuple, self._tr_spaces)
            if ans is not None:
                return ans
            # no specific rules, so return max of wire spacings.
            ans = 0
            for name in name_tuple:
                cur_space = self._get_space_from_str(layer_id, name_tuple, sp_override)
                cur_width = self.get_width(layer_id, name)
                ans = max(ans, cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                          half_space=half_space))
            return ans
        else:
            cur_space = self._get_space_from_str(layer_id, name_tuple, sp_override)
            cur_width = self.get_width(layer_id, name_tuple)
            return max(cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                  half_space=half_space))

    @classmethod
    def _get_space_from_tuple(cls, layer_id, ntup, sp_dict):
        if sp_dict is not None and ntup in sp_dict:
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
            ntup = (ntup[1], ntup[0])
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
        return None

    def _get_space_from_str(self, layer_id, name, sp_override):
        if sp_override is not None and name in sp_override:
            test = sp_override[name]
            if layer_id in test:
                return test[layer_id]
        if name in self._tr_spaces:
            return self._tr_spaces[name].get(layer_id, 0)
        return 0

    def place_wires(self,
                    layer_id,  # type: int
                    name_list,  # type: Sequence[str]
                    start_idx=0,  # type: Union[float, int]
                    **kwargs):
        # type: (...) -> Tuple[Union[float, int], List[Union[float, int]]]
        """Place the given wires next to each other.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        name_list : Sequence[str]
            list of wire types.
        start_idx : Union[float, int]
            the starting track index.
        **kwargs:
            optional parameters for get_num_space_tracks() method of RoutingGrid.

        Returns
        -------
        num_tracks : Union[float, int]
            number of tracks used.
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        num_wires = len(name_list)
        marker_htr = int(start_idx * 2 + 1)
        answer = []
        num_tracks = 0
        for idx, name in enumerate(name_list):
            cur_width = self.get_width(layer_id, name, **kwargs)
            num_tracks += cur_width
            cur_center_htr = marker_htr + cur_width - 1
            cur_center = (cur_center_htr - 1) // 2 if cur_center_htr % 2 == 1 \
                else (cur_center_htr - 1) / 2
            answer.append(cur_center)
            if idx != num_wires - 1:
                next_name = name_list[idx + 1]
                # figure out the current spacing
                cur_space = self.get_space(layer_id, (name, next_name), **kwargs)

                num_tracks += cur_space
                # advance marker
                cur_space_htr = int(cur_space * 2 + 1)
                marker_htr += 2 * cur_width - 1 + cur_space_htr

        # make sure num_tracks is integer type if we use integer number of tracks.
        num_tracks_half = int(2 * num_tracks)
        if num_tracks_half % 2 == 0:
            num_tracks = num_tracks_half // 2

        return num_tracks, answer

    @classmethod
    def _get_align_delta(cls, tot_ntr, num_used, alignment):
        if alignment == -1 or num_used == tot_ntr:
            # we already aligned to left
            return 0
        elif alignment == 0:
            # center tracks
            delta_htr = int((tot_ntr - num_used) * 2) // 2
            return delta_htr / 2 if delta_htr % 2 == 1 else delta_htr // 2
        elif alignment == 1:
            # align to right
            return tot_ntr - num_used
        else:
            raise ValueError('Unknown alignment code: %d' % alignment)

    def align_wires(self,  # type: TrackManager
                    layer_id,  # type: int
                    name_list,  # type: Sequence[str]
                    tot_ntr,  # type: Union[float, int]
                    alignment=0,  # type: int
                    start_idx=0,  # type: int
                    **kwargs):
        # type: (...) -> List[Union[float, int]]
        """Place the given wires in the given space with the specified alignment.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        name_list : Sequence[str]
            list of wire types.
        tot_ntr : Union[float, int]
            total available space in number of tracks.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : Union[float, int]
            the starting track index.
        **kwargs:
            optional parameters for place_wires().

        Returns
        -------
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        num_used, idx_list = self.place_wires(layer_id, name_list, start_idx=start_idx, **kwargs)
        if num_used > tot_ntr:
            raise ValueError('Given tracks occupy more space than given.')

        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]

    def spread_wires(self,  # type: TrackManager
                     layer_id,  # type: int
                     name_list,  # type: Sequence[str]
                     tot_ntr,  # type: Union[float, int]
                     sp_name_tuple,  # type: Union[str, Tuple[str, str]]
                     alignment=0,  # type: int
                     start_idx=0,  # type: int
                     max_sp=10000,  # type: int
                     ):
        # type: (...) -> List[Union[float, int]]
        """Spread out the given wires in the given space.

        This method tries to spread out wires by increasing the space around the given
        wire/combination of wires.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        name_list : Sequence[str]
            list of wire types.
        tot_ntr : Union[float, int]
            total available space in number of tracks.
        sp_name_tuple : Union[str, Tuple[str, str]]
            The space to increase.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : Union[float, int]
            the starting track index.
        max_sp : int
            maximum space.

        Returns
        -------
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        sp_override = {sp_name_tuple: {layer_id: 0}}
        cur_sp = int(round(2 * self.get_space(layer_id, sp_name_tuple)))
        bin_iter = BinaryIterator(cur_sp, None)
        while bin_iter.has_next():
            new_sp = bin_iter.get_next()
            if new_sp > 2 * max_sp:
                break
            sp_override[sp_name_tuple][layer_id] = new_sp / 2
            tmp = self.place_wires(layer_id, name_list, start_idx=start_idx)
            if tmp[0] > tot_ntr:
                bin_iter.down()
            else:
                bin_iter.save_info(tmp)
                bin_iter.up()

        num_used, idx_list = bin_iter.get_last_save_info()
        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]
