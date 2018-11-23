# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib

import yaml
import pytest

from pybag.enum import DesignOutput

from bag.design.database import ModuleDB
from bag.design.module import Module


def get_sch_master(module_db: ModuleDB, sch_design_params: Dict[str, Any]) -> Module:
    lib_name = sch_design_params['lib_name']
    cell_name = sch_design_params['cell_name']
    params = sch_design_params['params']

    gen_cls = module_db.get_schematic_class(lib_name, cell_name)
    ans = module_db.new_master(gen_cls, params=params)  # type: Module
    return ans


def get_extension(output_type: DesignOutput) -> str:
    if output_type is DesignOutput.YAML:
        return 'yaml'
    elif output_type is DesignOutput.CDL:
        return 'cdl'
    elif output_type is DesignOutput.VERILOG:
        return 'v'
    else:
        raise ValueError('Unsupported design output type: {}'.format(output_type.name))


@pytest.mark.parametrize("output_type, kwargs", [
    (DesignOutput.YAML, {}),
    (DesignOutput.CDL, {'prim_fname': '', 'flat': True, 'shell': False, 'rmin': 2000}),
])
def test_design_yaml(tmpdir,
                     module_db: ModuleDB,
                     sch_design_params: Dict[str, Any],
                     output_type: DesignOutput,
                     kwargs: Dict[str, Any],
                     gen_output: bool,
                     ) -> None:
    """Test design() method of each schematic generator."""
    extension = get_extension(output_type)
    expect_fname = sch_design_params.get('out_' + extension, '')
    if not expect_fname and not gen_output:
        pytest.skip('Cannot find expected output file.')

    dsn = get_sch_master(module_db, sch_design_params)

    path = tmpdir.join('out.' + extension)
    module_db.instantiate_master(output_type, dsn, top_cell_name='PYTEST', fname=str(path),
                                 **kwargs)

    assert path.check(file=1)

    with path.open('r') as f:
        actual = f.read()

    if gen_output:
        dir_name = pathlib.Path('pytest_output')
        out_fname = dir_name / (sch_design_params['test_id'] + '.' + extension)
        dir_name.mkdir(exist_ok=True)
        with out_fname.open('w') as f:
            f.write(actual)
        expect = actual
    else:
        with open(expect_fname, 'r') as f:
            expect = f.read()

    if output_type is DesignOutput.YAML:
        actual_dict = yaml.load(actual)
        expect_dict = yaml.load(expect)
        assert actual_dict == expect_dict
    else:
        assert actual == expect
