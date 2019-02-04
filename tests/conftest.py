# -*- coding: utf-8 -*-

import pytest

from bag.env import create_tech_info, create_routing_grid


@pytest.fixture(scope='session')
def tech_info():
    return create_tech_info()


@pytest.fixture(scope='session')
def routing_grid(tech_info):
    return create_routing_grid(tech_info=tech_info)
