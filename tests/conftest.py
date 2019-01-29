# -*- coding: utf-8 -*-

import pytest

from bag.env import create_tech_info


@pytest.fixture(scope='session')
def tech_info():
    return create_tech_info()
