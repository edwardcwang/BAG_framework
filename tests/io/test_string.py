# -*- coding: utf-8 -*-

import pytest

from bag.io.string import to_yaml_str


@pytest.mark.parametrize("arg, expect", [
    ([1, 2], '[1, 2]\n'),
    ({'a': 3, 'b': 'hi'}, '{a: 3, b: hi}\n'),
])
def test_to_yaml_str(arg, expect):
    """Check that to_yaml_str() converts Python objects to YAML string correctly."""
    assert to_yaml_str(arg) == expect
