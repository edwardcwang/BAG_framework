# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib

import yaml
import pytest

from pybag.enum import DesignOutput, get_extension, is_model_type

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.core import get_netlist_setup_file


def get_sch_master(module_db: ModuleDB, sch_design_params: Dict[str, Any]) -> Module:
    lib_name = sch_design_params['lib_name']
    cell_name = sch_design_params['cell_name']
    params = sch_design_params['params']

    gen_cls = module_db.get_schematic_class(lib_name, cell_name)
    ans = module_db.new_master(gen_cls, params=params)  # type: Module
    return ans


@pytest.mark.parametrize("output_type, options", [
    (DesignOutput.YAML, {}),
    (DesignOutput.CDL, {'flat': True, 'shell': False, 'rmin': 2000}),
    (DesignOutput.VERILOG, {'flat': True, 'shell': False, 'rmin': 2000}),
    (DesignOutput.VERILOG, {'flat': True, 'shell': True, 'rmin': 2000}),
    (DesignOutput.SYSVERILOG, {'flat': True, 'shell': False, 'rmin': 2000}),
])
def test_design(tmpdir,
                module_db: ModuleDB,
                sch_design_params: Dict[str, Any],
                output_type: DesignOutput,
                options: Dict[str, Any],
                gen_output: bool,
                ) -> None:
    """Test design() method of each schematic generator."""
    extension = get_extension(output_type)

    if is_model_type(output_type):
        model_params = sch_design_params.get('model_params', None)
        if model_params is None:
            pytest.skip('Cannot find model parameters.')
    else:
        model_params = None

    if output_type is DesignOutput.YAML:
        base = 'out'
    else:
        base = 'out_{}'.format(int(options['shell']))

    expect_fname = sch_design_params.get('{}_{}'.format(base, extension), '')
    if not expect_fname and not gen_output:
        pytest.skip('Cannot find expected output file.')

    dsn = get_sch_master(module_db, sch_design_params)

    path = tmpdir.join('{}.{}'.format(base, extension))
    if output_type is not DesignOutput.YAML:
        try:
            prim_fname = get_netlist_setup_file()
        except ValueError:
            prim_fname = ''
            pytest.skip('BAG technology directory not defined.')

        # noinspection PyTypeChecker
        options['prim_fname'] = prim_fname

    if is_model_type(output_type):
        module_db.batch_model([(dsn, 'PYTEST', model_params)], output=output_type,
                              fname=str(path), **options)
    else:
        module_db.instantiate_master(output_type, dsn, top_cell_name='PYTEST', fname=str(path),
                                     **options)

    assert path.check(file=1)

    with path.open('r') as f:
        actual = f.read()

    if gen_output:
        dir_name = pathlib.Path('pytest_output') / sch_design_params['test_id']
        out_fname = dir_name / (base + '.' + extension)
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
