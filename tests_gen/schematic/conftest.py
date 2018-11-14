# -*- coding: utf-8 -*-


import pytest

from bag.design.database import ModuleDB


@pytest.fixture
def module_db(tech_info):
    return ModuleDB(tech_info, 'PYTEST')
