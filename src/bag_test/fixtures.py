# -*- coding: utf-8 -*-

import os
import pathlib

import pytest

from bag.core import create_tech_info
from bag.design.database import ModuleDB


def get_test_cells(metafunc, conftest_path):
    if 'test_cell' in metafunc.fixturenames:
        data_dir = pathlib.Path(os.path.realpath(conftest_path)).parent / 'data'
        test_cells = [(p, p.stem) for p in data_dir.iterdir() if p.is_file()]
        metafunc.parametrize('test_cell', test_cells, indirect=True)


@pytest.fixture(scope='session')
def tech_info():
    return create_tech_info()


@pytest.fixture
def module_db(tech_info):
    return ModuleDB(tech_info, 'PYTEST')


@pytest.fixture
def test_cell(request):
    return request.param
