# -*- coding: utf-8 -*-

import abs_templates_ec.test.geo_inst as geo_inst


def test_master_key(temp_db):
    """Check that instance master keys are set properly."""
    master = temp_db.new_template(geo_inst.TestPolyInst00, {})
    child_keys = list(master.children())
    assert len(child_keys) == 1
