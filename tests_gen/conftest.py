# -*- coding: utf-8 -*-

from typing import Dict, Any, List

import pathlib
import importlib

import pytest

from bag.core import create_tech_info
from bag.io.file import read_yaml


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, dict) and isinstance(right, dict) and op == '==':
        return get_dict_diff_msg(left, right)


def pytest_addoption(parser):
    parser.addoption(
        '--data_root', action='store', default='', help='test data root directory',
    )
    parser.addoption(
        '--package', action='store', default='', help='generator package to test',
    )
    parser.addoption(
        '--gen_output', action='store_true', default=False,
        help='True to generate expected outputs',
    )


def get_dict_diff_msg(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    ans = ['Comparing (Nested) Dictionaries:']  # type: List[str]
    get_dict_diff_msg_helper(left, right, ans, [])
    return ans


def get_dict_diff_msg_helper(left: Dict[str, Any], right: Dict[str, Any], msgs: List[str],
                             prefix: List[str]) -> None:
    keys1 = sorted(left.keys())
    keys2 = sorted(right.keys())

    idx1 = 0
    idx2 = 0
    n1 = len(keys1)
    n2 = len(keys2)
    prefix_str = ','.join(prefix)
    while idx1 < n1 and idx2 < n2:
        k1 = keys1[idx1]
        k2 = keys2[idx2]
        v1 = left[k1]
        v2 = right[k2]
        if k1 == k2:
            if v1 != v2:
                if isinstance(v1, dict) and isinstance(v2, dict):
                    next_prefix = prefix.copy()
                    next_prefix.append(k1)
                    get_dict_diff_msg_helper(v1, v2, msgs, next_prefix)
                else:
                    msgs.append('L[{}]:'.format(prefix_str))
                    msgs.append('{}'.format(v1))
                    msgs.append('R[{}]:'.format(prefix_str))
                    msgs.append('{}'.format(v2))
            idx1 += 1
            idx2 += 1
        elif k1 < k2:
            msgs.append('R[{}] missing key:'.format(prefix_str))
            msgs.append('{}'.format(k1))
            idx1 += 1
        else:
            msgs.append('L[{}] missing key:'.format(prefix_str))
            msgs.append('{}'.format(k2))
            idx2 += 1
    while idx1 < n1:
        msgs.append('R[{}] missing key:'.format(prefix_str))
        msgs.append('{}'.format(keys1[idx1]))
        idx1 += 1
    while idx2 < n2:
        msgs.append('L[{}] missing key:'.format(prefix_str))
        msgs.append('{}'.format(keys2[idx2]))
        idx2 += 1


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
            continue

        for p in cur_dir.iterdir():
            if p.is_dir():
                test_id = p.name  # type: str
                # noinspection PyTypeChecker
                content = read_yaml(p / 'params.yaml')
                # inject fields
                content['test_id'] = pkg + '__' + test_id
                content['test_output_dir'] = str(pathlib.Path(pkg) / data_type / test_id)
                content['lib_name'] = pkg
                content['cell_name'] = test_id.rsplit('_', maxsplit=1)[0]
                for fpath in p.iterdir():
                    if fpath.stem.startswith('out'):
                        content['{}_{}'.format(fpath.stem, fpath.suffix[1:])] = str(
                            fpath.absolute())
                data.append(content)
    if data:
        metafunc.parametrize(data_name, data, indirect=True, ids=get_test_data_id)


def pytest_generate_tests(metafunc):
    for name, dtype in [('sch_design_params', 'schematic'),
                        ('lay_design_params', 'layout')]:
        if name in metafunc.fixturenames:
            setup_test_data(metafunc, name, dtype)
            break


@pytest.fixture(scope='session')
def gen_output(request):
    return request.config.getoption("--gen_output")


@pytest.fixture(scope='session')
def tech_info():
    return create_tech_info()


@pytest.fixture
def sch_design_params(request):
    return request.param if hasattr(request, 'param') else None


@pytest.fixture
def lay_design_params(request):
    return request.param if hasattr(request, 'param') else None
