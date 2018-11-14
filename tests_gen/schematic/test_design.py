# -*- coding: utf-8 -*-


def test_design(sch_design_params):
    """Test design() method of each schematic generator."""
    cell_name, config = sch_design_params
    assert cell_name == ''
