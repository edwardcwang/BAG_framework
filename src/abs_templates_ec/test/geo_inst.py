# -*- coding: utf-8 -*-

from typing import Any

from pybag.enum import GeometryMode, PathStyle, Orientation, Orient2D, BlockageType, BoundaryType
from pybag.core import BBox, Transform

from bag.util.cache import Param
from bag.layout.template import TemplateBase, TemplateDB


class TestRectPath00(TemplateBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        self.set_geometry_mode(GeometryMode.POLY_45)

        # simple rectangle
        self.add_rect(('M1', ''), BBox(100, 60, 180, 80))

        # a path
        width = 20
        points = [(0, 0), (2000, 0), (3000, 1000), (3000, 3000)]
        self.add_path(('M2', ''), width, points, PathStyle.truncate, join_style=PathStyle.round,
                      stop_style=PathStyle.round)

        # set top layer and bounding box so parent can query those
        self.prim_top_layer = 3
        self.prim_bound_box = BBox(0, 0, 400, 400)


class TestPolyInst00(TemplateBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        self.set_geometry_mode(GeometryMode.POLY)

        # instantiate Test1
        master = self.template_db.new_template(params={}, temp_cls=TestRectPath00)
        self.add_instance(master, inst_name='X0', xform=Transform(-100, -100, Orientation.MX))

        # add via, using BAG's technology DRC calculator
        self.add_via(BBox(0, 0, 100, 100), ('M1', ''), ('M2', ''), Orient2D.x)

        # add a primitive pin
        self.add_pin_primitive('mypin', 'M1', BBox(-100, 0, 0, 20))

        # add a polygon
        points = [(0, 0), (300, 200), (100, 400)]
        self.add_polygon(('M3', ''), points)

        # add a blockage
        points = [(-1000, -1000), (-1000, 1000), (1000, 1000), (1000, -1000)]
        self.add_blockage('', BlockageType.placement, points)

        # add a boundary
        points = [(-500, -500), (-500, 500), (500, 500), (500, -500)]
        self.add_boundary(BoundaryType.PR, points)

        # add a parallel path bus
        widths = [100, 50, 100]
        spaces = [80, 80]
        points = [(0, -3000), (-3000, -3000), (-4000, -2000), (-4000, 0)]
        self.add_path45_bus(('M2', ''), points, widths, spaces, start_style=PathStyle.truncate,
                            join_style=PathStyle.round)

        self.prim_top_layer = 3
        self.prim_bound_box = BBox(-10000, -10000, 10000, 10000)
