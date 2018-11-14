# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib
import importlib

import pytest
import yaml

from bag.core import create_tech_info


def pytest_addoption(parser):
    parser.addoption(
        '--package', action='store', default='', help='generator package to test',
    )


def get_test_data_id(data: Dict[str, Any]) -> str:
    return data['test_id']


def setup_test_data(metafunc, data_name: str, data_type: str) -> None:
    pkg_name = metafunc.config.getoption("--package")
    try:
        package = importlib.import_module(pkg_name)
    except ImportError:
        raise ImportError("Cannot find python package {}, "
                          "make sure it's on your PYTHONPATH".format(pkg_name))
    if not hasattr(package, '__file__'):
        raise ImportError(
            '{} is not a normal python package (no __file__ attribute).'.format(pkg_name))

    pkg_path = pathlib.Path(package.__file__).parent.parent
    if pkg_path.stem != 'src':
        raise ValueError('Non-standard directory structure: '
                         'package {} not in a "src" directory'.format(pkg_name))

    repo_path = pkg_path.parent
    data_dir = repo_path / 'tests_data' / pkg_name / data_type
    if not data_dir.is_dir():
        raise ValueError('Non-standard directory structure: '
                         '{} is not a directory (or does not exist)'.format(data_dir))

    data = []
    for p in data_dir.iterdir():
        if p.is_file():
            with open(p, 'r') as f:
                content = yaml.load(f)
            # inject fields
            content['lib_name'] = pkg_name
            content['test_id'] = p.stem
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
