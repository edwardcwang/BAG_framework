# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib
import importlib

import pytest
import yaml

from bag.core import create_tech_info


def pytest_addoption(parser):
    parser.addoption(
        '--data_root', action='store', default='', help='test data root directory',
    )
    parser.addoption(
        '--package', action='store', default='', help='generator package to test',
    )


def get_test_data_id(data: Dict[str, Any]) -> str:
    return data['test_id']


def setup_test_data(metafunc, data_name: str, data_type: str) -> None:
    pkg_name = metafunc.config.getoption('package')
    root_dir = pathlib.Path(metafunc.config.getoption('--data_root'))

    # get list of packages
    if pkg_name:
        # check package is importable
        try:
            importlib.import_module(pkg_name)
        except ImportError:
            raise ImportError("Cannot find python package {}, "
                              "make sure it's on your PYTHONPATH".format(pkg_name))

        # check data directory exists
        tmp = root_dir / pkg_name
        if not tmp.is_dir():
            raise ValueError('package data directory {} is not a directory'.format(tmp))
        pkg_iter = [pkg_name]
    else:
        pkg_iter = (d.name for d in root_dir.iterdir() if d.is_dir())

    data = []

    for pkg in pkg_iter:
        cur_dir = root_dir / pkg / data_type
        if not cur_dir.is_dir():
            raise ValueError('Data directory {} is not a directory'.format(cur_dir))

        for p in cur_dir.iterdir():
            if p.is_file():
                # noinspection PyTypeChecker
                with open(p, 'r') as f:
                    content = yaml.load(f)
                # inject fields
                test_id = p.stem  # type: str
                content['test_id'] = test_id
                content['lib_name'] = pkg
                content['cell_name'] = test_id.rsplit('_', maxsplit=1)[0]
                data.append(content)

    metafunc.parametrize(data_name, data, indirect=True, ids=get_test_data_id)


def pytest_generate_tests(metafunc):
    for name, dtype in [('sch_design_params', 'schematic'),
                        ('lay_design_params', 'layout')]:
        if name in metafunc.fixturenames:
            setup_test_data(metafunc, name, dtype)
            break


@pytest.fixture(scope='session')
def tech_info():
    return create_tech_info()


@pytest.fixture
def sch_design_params(request):
    return request.param


@pytest.fixture
def lay_design_params(request):
    return request.param
