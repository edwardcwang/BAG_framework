# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib

import yaml

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


def test_design_yaml(tmpdir,
                     module_db: ModuleDB,
                     sch_design_params: Dict[str, Any],
                     gen_output: bool,
                     ) -> None:
    """Test design() method of each schematic generator."""
    expect_fname = sch_design_params.get('out_yaml', '')

    dsn = get_sch_master(module_db, sch_design_params)

    path = tmpdir.join('out.yaml')
    module_db.instantiate_master(DesignOutput.YAML, dsn, top_cell_name='PYTEST', fname=str(path))

    with path.open('r') as f:
        actual = yaml.load(f)

    if gen_output or expect_fname:
        dir_name = pathlib.Path('pytest_output')
        out_fname = dir_name / (sch_design_params['test_id'] + '.yaml')
        dir_name.mkdir(exist_ok=True)
        with out_fname.open('w') as f:
            yaml.dump(actual, f)
        expect = actual
    else:
        with open(expect_fname, 'r') as f:
            expect = yaml.load(f)

    assert path.check(file=1)
    assert expect == actual
