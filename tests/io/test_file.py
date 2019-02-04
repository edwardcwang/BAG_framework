# -*- coding: utf-8 -*-

import os

from bag.io.file import make_temp_dir


def test_make_temp_dir():
    """Check that make_temp_dir creates a temporary directory."""
    # check prefix is correct
    prefix = 'foobar'
    dirname = make_temp_dir(prefix)
    assert os.path.basename(dirname).startswith(prefix)
    assert os.path.isdir(dirname)

    # check make_temp_dir will create parent directory if it's not there
    tmp_p1 = make_temp_dir('tmp_parent')
    parent = os.path.join(tmp_p1, 'parent')
    assert not os.path.isdir(parent)
    dirname = make_temp_dir(prefix, parent_dir=parent)
    assert os.path.isdir(parent)
    assert os.path.basename(dirname).startswith(prefix)
    assert os.path.isdir(dirname)
