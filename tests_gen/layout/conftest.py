# -*- coding: utf-8 -*-


import pytest

from bag.env import create_routing_grid
from bag.layout.template import TemplateDB


@pytest.fixture(scope='session')
def routing_grid():
    return create_routing_grid()


@pytest.fixture
def temp_db(routing_grid):
    return TemplateDB(routing_grid, 'PYTEST')
