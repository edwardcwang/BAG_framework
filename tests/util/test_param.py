# -*- coding: utf-8 -*-

import os

import pytest

from bag.util.cache import Param


@pytest.mark.parametrize("val", [
    2,
    3.5,
    'hi',
    (1, 2, 'bye'),
    [1, 2, 'foo'],
    {1: 'fi', 3.5: 'bar'},
    (1, [1, 2], 'lol'),
])
def test_get_hash(val):
    """Check that get_hash() works properly. on supported datatypes"""
    ans = Param.get_hash(val)
    assert isinstance(ans, int)
