# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib
import importlib
from shutil import copyfile

import pytest

from pybag.core import read_gds
from pybag.enum import DesignOutput, get_extension

from bag.layout.template import TemplateDB, TemplateBase
from bag.env import get_gds_layer_map, get_gds_object_map


def get_master(temp_db: TemplateDB, lay_design_params: Dict[str, Any]) -> TemplateBase:
    module_name = lay_design_params['module']
    cls_name = lay_design_params['class']
    params = lay_design_params['params']
    try:
        lay_module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError('Cannot find Python module {} for layout generator.  '
                          'Is it on your PYTHONPATH?'.format(module_name))

    if not hasattr(lay_module, cls_name):
        raise ImportError('Cannot find layout generator class {} '
                          'in module {}'.format(cls_name, module_name))

    gen_cls = getattr(lay_module, cls_name)
    ans = temp_db.new_master(gen_cls, params=params)  # type: TemplateBase
    return ans


@pytest.mark.parametrize("output_type", [
    DesignOutput.GDS,
])
def test_layout(tmpdir,
                temp_db: TemplateDB,
                lay_design_params: Dict[str, Any],
                output_type: DesignOutput,
                gen_output: bool,
                ) -> None:
    """Test design() method of each schematic generator."""
    if lay_design_params is None:
        # no layout tests
        return

    extension = get_extension(output_type)

    base = 'out'
    expect_fname = lay_design_params.get('{}_{}'.format(base, extension), '')
    if not expect_fname and not gen_output:
        pytest.skip('Cannot find expected output file.')

    dsn = get_master(temp_db, lay_design_params)
    assert dsn is not None

    out_base_name = '{}.{}'.format(base, extension)
    path = tmpdir.join(out_base_name)
    temp_db.instantiate_layout(dsn, 'PYTEST_TOP', output=output_type, fname=str(path))
    assert path.check(file=1)

    actual_fname = str(path)
    if gen_output:
        dir_name = pathlib.Path('pytest_output') / lay_design_params['test_output_dir']
        expect_fname = dir_name / out_base_name
        dir_name.mkdir(parents=True, exist_ok=True)
        copyfile(actual_fname, expect_fname)
    else:
        # currently only works for GDS
        lay_map = get_gds_layer_map()
        obj_map = get_gds_object_map()
        grid = temp_db.grid
        expect_cv_list = read_gds(expect_fname, lay_map, obj_map, grid)
        actual_cv_list = read_gds(actual_fname, lay_map, obj_map, grid)

        assert expect_cv_list == actual_cv_list
