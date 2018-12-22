# -*- coding: utf-8 -*-

from typing import Union, Tuple

from bag.layout.routing.base import HalfInt

CoordType = int
PointType = Tuple[CoordType, CoordType]

TrackType = Union[float, HalfInt]
SizeType = Tuple[int, HalfInt, HalfInt]
