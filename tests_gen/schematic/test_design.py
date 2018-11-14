# -*- coding: utf-8 -*-

from typing import Dict, Any

from bag.design.database import ModuleDB
from bag.util.cache import DesignOutput


def test_design(tmpdir, module_db: ModuleDB, sch_design_params: Dict[str, Any]) -> None:
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

    assert path.check(file=1)
    assert expect == path.read_text()
