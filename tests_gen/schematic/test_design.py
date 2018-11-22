# -*- coding: utf-8 -*-

from typing import Dict, Any

import pathlib

import yaml

from pybag.enum import DesignOutput

from bag.design.database import ModuleDB


def test_design(tmpdir,
                module_db: ModuleDB,
                sch_design_params: Dict[str, Any],
                gen_output: bool,
                ) -> None:
    """Test design() method of each schematic generator."""
    config = sch_design_params
    lib_name = config['lib_name']
    cell_name = config['cell_name']
    params = config['params']
    expect = config['expect']

    path = tmpdir.join('output.yaml')

    gen_cls = module_db.get_schematic_class(lib_name, cell_name)
    dsn = module_db.new_master(gen_cls, params=params)

    module_db.instantiate_master(DesignOutput.YAML, dsn, top_cell_name='PYTEST', fname=str(path))

    with path.open('r') as f:
        actual = yaml.load(f)

    if gen_output:
        dir_name = pathlib.Path('pytest_output')
        out_fname = dir_name / (config['test_id'] + '.yaml')
        dir_name.mkdir(exist_ok=True)
        with out_fname.open('w') as f:
            yaml.dump(actual, f)
        expect = actual

    assert path.check(file=1)
    assert expect == actual
