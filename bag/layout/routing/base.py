# -*- coding: utf-8 -*-

"""This module provides basic routing classes.
"""

from typing import (
    TYPE_CHECKING, Tuple, Union, Iterable, Iterator, Dict, List, Sequence, Any, Optional
)

from math import trunc, ceil, floor
from numbers import Integral, Real

from pybag.enum import Orientation
from pybag.util.geometry import BBox, BBoxArray

from ...util.search import BinaryIterator

if TYPE_CHECKING:
    from .grid import RoutingGrid
    from bag.typing import TrackType


class HalfInt(Integral):
    """A class that represents a half integer."""

    def __init__(self, dbl_val):
        if isinstance(dbl_val, Integral):
            self._val = int(dbl_val)
        else:
            raise ValueError('HafInt internal value must be an integer.')

    @classmethod
    def convert(cls, val):
        # type: (Any) -> HalfInt
        if isinstance(val, HalfInt):
            return val
        elif isinstance(val, Integral):
            return HalfInt(2 * int(val))
        elif isinstance(val, Real):
            tmp = float(2 * val)
            if tmp.is_integer():
                return HalfInt(int(tmp))
        raise ValueError('Cannot convert {} type {} to HalfInt.'.format(val, type(val)))

    @property
    def value(self):
        # type: () -> Union[int, float]
        q, r = divmod(self._val, 2)
        return q if r == 0 else q + 0.5

    def is_integer(self):
        # type: () -> bool
        return self._val % 2 == 0

    def div2(self, round_up=False):
        # type: (bool) -> HalfInt
        q, r = divmod(self._val, 2)
        if r or not round_up:
            self._val = q
        else:
            self._val = q + 1
        return self

    def val_str(self):
        # type: () -> str
        q, r = divmod(self._val, 2)
        if r == 0:
            return '{:d}'.format(q)
        return '{:d}.5'.format(q)

    def up(self):
        # type: () -> None
        self._val += 1

    def down(self):
        # type: () -> None
        self._val -= 1

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'HalfInt({})'.format(self._val / 2)

    def __hash__(self):
        return hash(self._val)

    def __eq__(self, other):
        if isinstance(other, HalfInt):
            return self._val == other._val
        return self._val == 2 * other

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        if isinstance(other, HalfInt):
            return self._val <= other._val
        return self._val <= 2 * other

    def __lt__(self, other):
        if isinstance(other, HalfInt):
            return self._val < other._val
        return self._val < 2 * other

    def __ge__(self, other):
        return not (self < other)

    def __gt__(self, other):
        return not (self <= other)

    def __add__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val + other._val)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val * other._val, 2)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __truediv__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(2 * self._val, other._val)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __floordiv__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(2 * (self._val // other._val))

    def __mod__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val % other._val)

    def __divmod__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val, other._val)
        return HalfInt(2 * q), HalfInt(r)

    def __pow__(self, other, modulus=None):
        other = HalfInt.convert(other)
        if self.is_integer() and other.is_integer():
            return HalfInt(2 * (self._val // 2)**(other._val // 2))
        raise ValueError('result is not a HalfInt.')

    def __lshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __and__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __xor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __or__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return HalfInt.convert(other) / self

    def __rfloordiv__(self, other):
        return HalfInt.convert(other) // self

    def __rmod__(self, other):
        return HalfInt.convert(other) % self

    def __rdivmod__(self, other):
        return HalfInt.convert(other).__divmod__(self)

    def __rpow__(self, other):
        return HalfInt.convert(other)**self

    def __rlshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rrshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __rand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __rxor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ror__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    def __imod__(self, other):
        return self % other

    def __ipow__(self, other):
        return self ** other

    def __ilshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __irshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __iand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __ixor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ior__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __neg__(self):
        return HalfInt(-self._val)

    def __pos__(self):
        return HalfInt(self._val)

    def __abs__(self):
        return HalfInt(abs(self._val))

    def __invert__(self):
        return -self

    def __complex__(self):
        raise TypeError('Cannot cast to complex')

    def __int__(self):
        if self._val % 2 == 1:
            raise ValueError('Not an integer.')
        return self._val // 2

    def __float__(self):
        return self._val / 2

    def __index__(self):
        return int(self)

    def __round__(self, ndigits=0):
        if self.is_integer():
            return HalfInt(self._val)
        else:
            return HalfInt(round(self._val / 2) * 2)

    def __trunc__(self):
        if self.is_integer():
            return HalfInt(self._val)
        else:
            return HalfInt(trunc(self._val / 2) * 2)

    def __floor__(self):
        if self.is_integer():
            return HalfInt(self._val)
        else:
            return HalfInt(floor(self._val / 2) * 2)

    def __ceil__(self):
        if self.is_integer():
            return HalfInt(self._val)
        else:
            return HalfInt(ceil(self._val / 2) * 2)


class TrackID(object):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : TrackType
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : TrackType
        pitch between adjacent tracks, in number of track pitches.
    """

    def __init__(self, layer_id, track_idx, width=1, num=1, pitch=0):
        # type: (int, TrackType, int, int, TrackType) -> None
        if num < 1:
            raise ValueError('TrackID must have 1 or more tracks.')

        self._layer_id = layer_id
        self._idx = HalfInt.convert(track_idx)
        self._w = width
        self._n = num
        self._pitch = HalfInt(0) if num == 1 else HalfInt.convert(pitch)

    def __repr__(self):
        # type: () -> str
        arg_list = ['layer={}'.format(self._layer_id), 'track={}'.format(self._idx.val_str())]
        if self._w != 1:
            arg_list.append('width={}'.format(self._w))
        if self._n != 1:
            arg_list.append('num={}'.format(self._n))
        arg_list.append('pitch={}'.format(self._pitch.val_str()))

        return '{}({})'.format(self.__class__.__name__, ', '.join(arg_list))

    def __str__(self):
        # type: () -> str
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
        # type: () -> HalfInt
        return self._idx

    @property
    def num(self):
        # type: () -> int
        return self._n

    @property
    def pitch(self):
        # type: () -> HalfInt
        return self._pitch

    def get_immutable_key(self):
        # type: () -> Tuple[str, int, HalfInt, int, int, HalfInt]
        return self.__class__.__name__, self._layer_id, self._idx, self._w, self._n, self._pitch

    def get_bounds(self, grid, unit_mode=True):
        # type: (RoutingGrid, bool) -> Tuple[int, int]
        """Calculate the track bounds coordinate.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        lower : int
            the lower bound coordinate perpendicular to track direction.
        upper : int
            the upper bound coordinate perpendicular to track direction.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        lower, upper = grid.get_wire_bounds(self.layer_id, self._idx, width=self.width)
        pitch_dim = int(self._pitch * grid.get_track_pitch(self._layer_id))
        upper += (self.num - 1) * pitch_dim
        return lower, upper

    def __iter__(self):
        # type: () -> Iterator[HalfInt]
        """Iterate over all middle track indices in this TrackID."""
        return (self._idx + idx * self._pitch for idx in range(self._n))

    def sub_tracks_iter(self, grid):
        # type: (RoutingGrid) -> Iterable[TrackID]
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
            if self._pitch.is_integer() and int(self._pitch) % len(layer_names) == 0:
                # layer name will never change
                yield self
            else:
                # TODO: have more robust solution than just yielding tracks one by one?
                for tr_idx in self:
                    yield TrackID(layer_id, tr_idx, width=self.width)
        else:
            yield self

    def transform(self, grid, loc=(0, 0), orient="R0", unit_mode=True):
        # type: (RoutingGrid, Tuple[int, int], str, bool) -> TrackID
        """returns a transformation of this TrackID."""
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        layer_id = self._layer_id
        is_x = grid.is_horizontal(layer_id)
        oenum = Orientation[orient]
        if oenum is Orientation.R0:
            base_idx = self._idx
        elif oenum is Orientation.MX:
            if is_x:
                base_idx = -self._idx - (self._n - 1) * self._pitch - 1
            else:
                base_idx = self._idx
        elif oenum is Orientation.MY:
            if is_x:
                base_idx = self._idx
            else:
                base_idx = -self._idx - (self._n - 1) * self._pitch - 1
        elif oenum is Orientation.R180:
            base_idx = -self._idx - (self._n - 1) * self._pitch - 1
        else:
            raise ValueError('Unsupported orientation: {}'.format(oenum))

        delta = grid.coord_to_track(layer_id, loc[1] if is_x else loc[0]) + 0.5
        return TrackID(layer_id, base_idx + delta, width=self._w,
                       num=self._n, pitch=self._pitch)


class WireArray(object):
    """An array of wires on the routing grid.

    Parameters
    ----------
    track_id : :class:`bag.layout.routing.TrackID`
        TrackArray representing the track locations of this wire array.
    lower : int
        the lower coordinate along the track direction.
    upper : int
        the upper coordinate along the track direction.
    res : Optional[float]
        deprecated parameter.
    unit_mode : bool
        deprecated parameter.
    """

    # noinspection PyUnusedLocal
    def __init__(self, track_id, lower, upper, res=None, unit_mode=True):
        # type: (TrackID, int, int, Optional[float], bool) -> None
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        self._track_id = track_id
        self._lower_unit = lower
        self._upper_unit = upper

    def __repr__(self):
        return '{}({}, {:d}, {:d})'.format(self.__class__.__name__, self._track_id,
                                           self._lower_unit, self._upper_unit)

    def __str__(self):
        return repr(self)

    @property
    def lower_unit(self):
        # type: () -> int
        return self._lower_unit

    @property
    def upper_unit(self):
        # type: () -> int
        return self._upper_unit

    @property
    def middle_unit(self):
        # type: () -> int
        return (self._lower_unit + self._upper_unit) // 2

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
        # type: () -> int
        return self.track_id.width

    @classmethod
    def list_to_warr(cls, warr_list):
        # type: (Sequence[WireArray]) -> WireArray
        """Convert a list of WireArrays to a single WireArray.

        this method assumes all WireArrays have the same layer, width, and lower/upper coordinates.
        Overlapping WireArrays will be compacted.
        """
        if len(warr_list) == 1:
            return warr_list[0]

        tid0 = warr_list[0].track_id
        layer = tid0.layer_id
        width = tid0.width
        lower, upper = warr_list[0].lower_unit, warr_list[0].upper_unit
        tid_list = sorted(set((idx for warr in warr_list for idx in warr.track_id)))
        base_idx = tid_list[0]
        if len(tid_list) < 2:
            return WireArray(TrackID(layer, base_idx, width=width), lower, upper)
        diff = tid_list[1] - tid_list[0]
        for idx in range(1, len(tid_list) - 1):
            if tid_list[idx + 1] - tid_list[idx] != diff:
                raise ValueError('pitch mismatch.')

        return WireArray(TrackID(layer, base_idx, width=width, num=len(tid_list), pitch=diff),
                         lower, upper)

    @classmethod
    def single_warr_iter(cls, warr):
        # type: (Union[WireArray, Sequence[WireArray]]) -> Iterable[WireArray]
        if isinstance(warr, WireArray):
            yield from warr.warr_iter()
        else:
            for w in warr:
                yield from w.warr_iter()

    def get_immutable_key(self):
        # type: () -> Any
        return (self.__class__.__name__, self._track_id.get_immutable_key(), self._lower_unit,
                self._upper_unit)

    def to_warr_list(self):
        # type: () -> List[WireArray]
        return list(self.warr_iter())

    def warr_iter(self):
        # type: () -> Iterable[WireArray]
        tid = self._track_id
        layer = tid.layer_id
        width = tid.width
        for tr in tid:
            yield WireArray(TrackID(layer, tr, width=width), self._lower_unit,
                            self._upper_unit)

    def get_bbox_array(self, grid):
        # type: (RoutingGrid) -> BBoxArray
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

        base_box = grid.get_bbox(layer_id, base_idx, self._lower_unit, self._upper_unit,
                                 width=tr_w)
        tot_pitch = int(track_id.pitch * grid.get_track_pitch(layer_id))
        if grid.get_direction(layer_id) == 'x':
            return BBoxArray(base_box, ny=num, spy=tot_pitch)
        else:
            return BBoxArray(base_box, nx=num, spx=tot_pitch)

    def wire_iter(self, grid):
        # type: (RoutingGrid) -> Iterable[Tuple[str, BBox]]
        """Iterate over all wires in this WireArray as layer/BBox pair.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : str
            the wire layer name.
        bbox : BBox
            the wire bounding box.
        """
        tr_w = self.track_id.width
        layer_id = self.layer_id
        for tr_idx in self.track_id:
            layer_name = grid.get_layer_name(layer_id, tr_idx)
            bbox = grid.get_bbox(layer_id, tr_idx, self._lower_unit, self._upper_unit,
                                 width=tr_w)
            yield layer_name, bbox

    def wire_arr_iter(self, grid):
        # type: (RoutingGrid) -> Iterable[Tuple[str, BBoxArray]]
        """Iterate over all wires in this WireArray as layer/BBoxArray pair.

        This method group all rectangles in the same layer together.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : str
            the wire layer name.
        bbox : BBoxArray
            the wire bounding boxes.
        """
        tid = self.track_id
        layer_id = tid.layer_id
        tr_width = tid.width
        track_pitch = grid.get_track_pitch(layer_id, unit_mode=True)
        is_x = grid.is_horizontal(layer_id)
        for track_idx in tid.sub_tracks_iter(grid):
            base_idx = track_idx.base_index
            cur_layer = grid.get_layer_name(layer_id, base_idx)
            cur_num = track_idx.num
            wire_pitch = int(track_idx.pitch * track_pitch)
            tl, tu = grid.get_wire_bounds(layer_id, base_idx, width=tr_width)
            if is_x:
                base_box = BBox(self._lower_unit, tl, self._upper_unit, tu)
                box_arr = BBoxArray(base_box, ny=cur_num, spy=wire_pitch)
            else:
                base_box = BBox(tl, self._lower_unit, tu, self._upper_unit)
                box_arr = BBoxArray(base_box, nx=cur_num, spx=wire_pitch)

            yield cur_layer, box_arr

    def transform(self, grid, loc=(0, 0), orient='R0', unit_mode=True):
        # type: (RoutingGrid, Tuple[int, int], str, bool) -> WireArray
        """Return a new transformed WireArray.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this WireArray.
        loc : Tuple[int, int]
            the X/Y coordinate shift.
        orient : str
            the new orientation.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        warr : WireArray
            the new WireArray object.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        layer_id = self.layer_id
        is_x = grid.is_horizontal(layer_id)
        oenum = Orientation[orient]
        if oenum is Orientation.R0:
            lower, upper = self._lower_unit, self._upper_unit
        elif oenum is Orientation.MX:
            if is_x:
                lower, upper = self._lower_unit, self._upper_unit
            else:
                lower, upper = -self._upper_unit, -self._lower_unit
        elif oenum is Orientation.MY:
            if is_x:
                lower, upper = -self._upper_unit, -self._lower_unit
            else:
                lower, upper = self._lower_unit, self._upper_unit
        elif oenum is Orientation.R180:
            lower, upper = -self._upper_unit, -self._lower_unit
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
    pin_dict : Dict[int, List[WireArray]]
        a dictionary from layer ID to pin geometries on that layer.
    label : str
        the label of this port.
    """

    def __init__(self, term_name, pin_dict, label):
        # type: (str, Dict[int, Union[List[WireArray], List[BBox]]], str) -> None
        self._term_name = term_name
        self._pin_dict = pin_dict
        self._label = label

    def __iter__(self):
        # type: () -> WireArray
        """Iterate through all pin geometries in this port.

        the iteration order is not guaranteed.
        """
        for geo_list in self._pin_dict.values():
            yield from geo_list

    def get_single_layer(self):
        # type: () -> Union[int, str]
        """Returns the layer of this port if it only has a single layer."""
        if len(self._pin_dict) > 1:
            raise ValueError('This port has more than one layer.')
        return next(iter(self._pin_dict))

    def _get_layer(self, layer):
        # type: (Union[int, str]) -> Union[int, str]
        """Get the layer ID or name."""
        if isinstance(layer, Integral):
            return self.get_single_layer() if layer < 0 else layer
        else:
            return self.get_single_layer() if not layer else layer

    @property
    def net_name(self):
        # type: () -> str
        """Returns the net name of this port."""
        return self._term_name

    @property
    def label(self):
        # type: () -> str
        """Returns the label of this port."""
        return self._label

    def get_pins(self, layer=-1):
        # type: (Union[int, str]) -> Union[List[WireArray], List[BBox]]
        """Returns the pin geometries on the given layer.

        Parameters
        ----------
        layer : Union[int, str]
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        track_bus_list : Union[List[WireArray], List[BBox]]
            pins on the given layer representing as WireArrays.
        """
        layer = self._get_layer(layer)
        return self._pin_dict.get(layer, [])

    def get_bounding_box(self, grid, layer=-1):
        # type: (RoutingGrid, Union[int, str]) -> BBox
        """Calculate the overall bounding box of this port on the given layer.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this Port.
        layer : Union[int, str
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        bbox : BBox
            the bounding box.
        """
        layer = self._get_layer(layer)
        box = BBox.get_invalid_bbox()
        for geo in self._pin_dict[layer]:
            if isinstance(geo, BBox):
                box = box.merge(geo)
            else:
                box = box.merge(geo.get_bbox_array(grid).get_overall_bbox())
        return box

    def transform(self, grid, loc=(0, 0), orient='R0', unit_mode=True):
        # type: (RoutingGrid, Tuple[int, int], str, bool) -> Port
        """Return a new transformed Port.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this Port.
        loc : Tuple[int, int]
            the X/Y coordinate shift.
        orient : str
            the new orientation.
        unit_mode: bool
            deprecated parameter.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        new_pin_dict = {}
        for lay, geo_list in self._pin_dict.items():
            new_geo_list = []
            for geo in geo_list:
                if isinstance(geo, BBox):
                    new_geo_list.append(geo.transform(loc=loc, orient=orient))
                else:
                    new_geo_list.append(geo.transform(grid, loc=loc, orient=orient))
            new_pin_dict[lay] = new_geo_list

        return Port(self._term_name, new_pin_dict, self._label)


class TrackManager(object):
    """A class that makes it easy to compute track locations.

    This class provides many helper methods for computing track locations and spacing when
    each track could have variable width.  All methods in this class accepts a "track_type",
    which is either a string in the track dictionary or an integer representing the track
    width.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    tr_widths : Dict[str, Dict[int, int]]
        dictionary from wire types to its width on each layer.
    tr_spaces : Dict[Union[str, Tuple[str, str]], Dict[int, TrackType]]
        dictionary from wire types to its spaces on each layer.
    **kwargs :
        additional options.
    """

    def __init__(self,
                 grid,  # type: RoutingGrid
                 tr_widths,  # type: Dict[str, Dict[int, int]]
<<<<<<< HEAD
                 tr_spaces,  # type: Dict[Union[str, Tuple[str, str]], Dict[int, TrackType]]
                 **kwargs,  # type: Any
=======
                 tr_spaces,  # type: Dict[Union[str, Tuple[str, str]], Dict[int, Union[float, int]]]
                 **kwargs
>>>>>>> master
                 ):
        # type: (...) -> None
        half_space = kwargs.get('half_space', True)

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

    def get_width(self, layer_id, track_type):
        # type: (int, Union[str, int]) -> int
        """Returns the track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        track_type : Union[str, int]
            the track type.
        """
        if isinstance(track_type, int):
            return track_type
        if track_type not in self._tr_widths:
            return 1
        return self._tr_widths[track_type].get(layer_id, 1)

    def get_space(self,  # type: TrackManager
                  layer_id,  # type: int
                  type_tuple,  # type: Union[str, int, Tuple[Union[str, int], Union[str, int]]]
                  **kwargs,  # type: Any
                  ):
        # type: (...) -> TrackType
        """Returns the track spacing.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        type_tuple : Union[str, int, Tuple[Union[str, int], Union[str, int]]]
            If a single track type is given, will return the minimum spacing needed around that
            track type.  If a tuple of two types are given, will return the specific spacing
            between those two track types if specified.  Otherwise, returns the maximum of all the
            valid spacing.
        **kwargs:
            optional parameters.
        """
        half_space = kwargs.get('half_space', self._half_space)
        sp_override = kwargs.get('sp_override', None)

        if isinstance(type_tuple, tuple):
            # if two specific wires are given, first check if any specific rules exist
            ans = self._get_space_from_tuple(layer_id, type_tuple, sp_override)
            if ans is not None:
                return ans
            ans = self._get_space_from_tuple(layer_id, type_tuple, self._tr_spaces)
            if ans is not None:
                return ans
            # no specific rules, so return max of wire spacings.
            ans = 0
            for wtype in type_tuple:
                cur_space = self._get_space_from_type(layer_id, wtype, sp_override)
                if cur_space is None:
                    cur_space = self._get_space_from_type(layer_id, wtype, self._tr_spaces)
                if cur_space is None:
                    cur_space = 0
                cur_width = self.get_width(layer_id, wtype)
                ans = max(ans, cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                          half_space=half_space))
            return ans
        else:
            cur_space = self._get_space_from_type(layer_id, type_tuple, sp_override)
            if cur_space is None:
                cur_space = self._get_space_from_type(layer_id, type_tuple, self._tr_spaces)
            if cur_space is None:
                cur_space = 0
            cur_width = self.get_width(layer_id, type_tuple)
            return max(cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                  half_space=half_space))

    @classmethod
    def _get_space_from_tuple(cls, layer_id, ntup, sp_dict):
        if sp_dict is not None:
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
            ntup = (ntup[1], ntup[0])
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
        return None

    @classmethod
    def _get_space_from_type(cls, layer_id, wtype, sp_dict):
        if sp_dict is None:
            return None
        if wtype in sp_dict:
            test = sp_dict[wtype]
        else:
            key = (wtype, '')
            if key in sp_dict:
                test = sp_dict[key]
            else:
                key = ('', wtype)
                if key in sp_dict:
                    test = sp_dict[key]
                else:
                    test = None

        if test is None:
            return None
        return test.get(layer_id, None)

    def get_next_track(self,  # type: TrackManager
                       layer_id,  # type: int
                       cur_idx,  # type: TrackType
                       cur_type,  # type: Union[str, int]
                       next_type,  # type: Union[str, int]
                       up=True,  # type: bool
                       **kwargs,  # type: Any
                       ):
        # type: (...) -> HalfInt
        """Compute the track location of a wire next to a given one.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        cur_idx : TrackType
            the current wire track index.
        cur_type : Union[str, int]
            the current wire type.
        next_type : Union[str, int]
            the next wire type.
        up : bool
            True to return the next track index that is larger than cur_idx.
        **kwargs :
            optional parameters.

        Returns
        -------
        next_int : HalfInt
            the next track index.
        """
        cur_width = self.get_width(layer_id, cur_type)
        next_width = self.get_width(layer_id, next_type)
        space = self.get_space(layer_id, (cur_type, next_type), **kwargs)
        cur_idx = HalfInt.convert(cur_idx)
        wsum = HalfInt(cur_width + next_width)
        if up:
            return wsum + space + cur_idx
        else:
            return -wsum - space + cur_idx

    def place_wires(self,  # type: TrackManager
                    layer_id,  # type: int
                    type_list,  # type: Sequence[Union[str, int]]
                    start_idx=0,  # type: TrackType
                    **kwargs,  # type: Any
                    ):
        # type: (...) -> Tuple[HalfInt, List[HalfInt]]
        """Place the given wires next to each other.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        start_idx : TrackType
            the starting track index.
        **kwargs : Any
            optional parameters for get_num_space_tracks() method of RoutingGrid.

        Returns
        -------
        num_tracks : HalfInt
            number of tracks used.
        locations : List[HalfInt]
            the center track index of each wire.
        """
        if not type_list:
            return HalfInt(0), []

        prev_type = type_list[0]
        w0 = self.get_width(layer_id, prev_type)

        mid_idx = HalfInt.convert(start_idx) + (w0 - 1) / 2
        ans = [mid_idx]
        for idx in range(1, len(type_list)):
            ans.append(self.get_next_track(layer_id, ans[-1], type_list[idx - 1],
                                           type_list[idx], up=True, **kwargs))

        w1 = self.get_width(layer_id, type_list[-1])
        ntr = (ans[-1] - ans[0]) + (w0 + w1) / 2
        return ntr, ans

    @classmethod
    def _get_align_delta(cls, tot_ntr, num_used, alignment):
        # type: (TrackType, TrackType, int) -> HalfInt
        if alignment == -1 or num_used == tot_ntr:
            # we already aligned to left
            return HalfInt(0)
        elif alignment == 0:
            # center tracks
            return HalfInt.convert(tot_ntr - num_used).div2()
        elif alignment == 1:
            # align to right
            return HalfInt.convert(tot_ntr - num_used)
        else:
            raise ValueError('Unknown alignment code: %d' % alignment)

    def align_wires(self,  # type: TrackManager
                    layer_id,  # type: int
                    type_list,  # type: Sequence[Union[str, int]]
                    tot_ntr,  # type: TrackType
                    alignment=0,  # type: int
                    start_idx=0,  # type: TrackType
                    **kwargs,  # type: Any
                    ):
        # type: (...) -> List[HalfInt]
        """Place the given wires in the given space with the specified alignment.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        tot_ntr : TrackType
            total available space in number of tracks.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : TrackType
            the starting track index.
        **kwargs:
            optional parameters for place_wires().

        Returns
        -------
        locations : List[HalfInt]
            the center track index of each wire.
        """
        num_used, idx_list = self.place_wires(layer_id, type_list, start_idx=start_idx, **kwargs)
        if num_used > tot_ntr:
            raise ValueError('Given tracks occupy more space than given.')

        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]

    def spread_wires(self,  # type: TrackManager
                     layer_id,  # type: int
                     type_list,  # type: Sequence[Union[str, int]]
                     tot_ntr,  # type: TrackType
                     sp_type,  # type: Union[str, int, Tuple[Union[str, int], Union[str, int]]]
                     alignment=0,  # type: int
                     start_idx=0,  # type: TrackType
                     max_sp=10000,  # type: int
                     sp_override=None,
                     ):
        # type: (...) -> List[HalfInt]
        """Spread out the given wires in the given space.

        This method tries to spread out wires by increasing the space around the given
        wire/combination of wires.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        tot_ntr : TrackType
            total available space in number of tracks.
        sp_type : Union[str, int, Tuple[Union[str, int], Union[str, int]]]
            The space to increase.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : TrackType
            the starting track index.
        max_sp : int
            maximum space.
        sp_override :
            tracking spacing override dictionary.

        Returns
        -------
        locations : List[HalfInt]
            the center track index of each wire.
        """
        if not sp_override:
            sp_override = {sp_type: {layer_id: HalfInt(0)}}
        else:
            sp_override = sp_override.copy()
            sp_override[sp_type] = {layer_id: HalfInt(0)}
        cur_sp = round(2 * self.get_space(layer_id, sp_type))
        bin_iter = BinaryIterator(cur_sp, None)
        while bin_iter.has_next():
            new_sp = bin_iter.get_next()
            if new_sp > 2 * max_sp:
                break
            sp_override[sp_type][layer_id] = HalfInt(new_sp)
            tmp = self.place_wires(layer_id, type_list, start_idx=start_idx,
                                   sp_override=sp_override)
            if tmp[0] > tot_ntr:
                bin_iter.down()
            else:
                bin_iter.save_info(tmp)
                bin_iter.up()

        if bin_iter.get_last_save_info() is None:
            raise ValueError('No solution found.')

        num_used, idx_list = bin_iter.get_last_save_info()
        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]
