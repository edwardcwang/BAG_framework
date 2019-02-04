# -*- coding: utf-8 -*-

from typing import Union

import pytest

from pybag.enum import RoundMode

from bag.util.math import HalfInt
from bag.layout.routing import RoutingGrid


@pytest.mark.parametrize("lay, coord, w_ntr, mode, half_track, expect", [
    (4, 720, 1, RoundMode.LESS_EQ, True, HalfInt(10)),
])
def test_find_next_track(routing_grid: RoutingGrid, lay: int, coord: int, w_ntr: int, mode: Union[RoundMode, int],
                         half_track: bool, expect: HalfInt) -> None:
    """Check that find_next_htr() works properly."""
    ans = routing_grid.find_next_track(lay, coord, tr_width=w_ntr, half_track=half_track, mode=mode)
    assert ans == expect
    assert isinstance(ans, HalfInt)
