# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from bag.layout.routing.base import HalfInt

CoordType = int
LayerType = Union[str, Tuple[str, str]]
PointType = Tuple[CoordType, CoordType]
SizeType = Tuple[int, int, int]
TrackType = Union[float, int, HalfInt]
