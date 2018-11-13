# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Union, Tuple

CoordType = int
LayerType = Union[str, Tuple[str, str]]
PointType = Tuple[CoordType, CoordType]

if TYPE_CHECKING:
    from bag.layout.routing.base import HalfInt
    TrackType = Union[float, int, HalfInt]
    SizeType = Tuple[int, HalfInt, HalfInt]
