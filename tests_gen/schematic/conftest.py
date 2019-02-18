# -*- coding: utf-8 -*-


import pytest

from bag.layout.tech import TechInfo
from bag.design.database import ModuleDB


@pytest.fixture
def module_db(tech_info: TechInfo) -> ModuleDB:
    return ModuleDB(tech_info, 'PYTEST')
