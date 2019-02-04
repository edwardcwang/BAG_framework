# -*- coding: utf-8 -*-

"""This module defines analog mosfet primitive template classes.
"""

from typing import Dict, Any, Tuple, Optional

from bag.math import float_to_si_string
from bag.util.cache import Param
from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB


class AnalogMOSBase(TemplateBase):
    """A primitive template of a transistor row.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        self._layout_info = None
        self._ext_top_info = None
        self._ext_bot_info = None
        self._left_edge_info = None
        self._right_edge_info = None

        self._g_conn_y = None
        self._d_conn_y = None
        self._od_y = None
        self._po_y = None
        self._sd_yc = None

    def get_g_conn_y(self):
        # type: () -> Tuple[int, int]
        return self._g_conn_y

    def get_d_conn_y(self):
        # type: () -> Tuple[int, int]
        return self._d_conn_y

    def get_od_y(self):
        # type: () -> Tuple[int, int]
        return self._od_y

    def get_po_y(self):
        # type: () -> Tuple[int, int]
        return self._po_y

    def get_ext_top_info(self):
        # type: () -> Any
        return self._ext_top_info

    def get_ext_bot_info(self):
        # type: () -> Any
        return self._ext_bot_info

    def get_left_edge_info(self):
        # type: () -> Any
        return self._left_edge_info

    def get_right_edge_info(self):
        # type: () -> Any
        return self._right_edge_info

    def get_sd_yc(self):
        # type: () -> int
        return self._sd_yc

    def get_edge_layout_info(self):
        # type: () -> Dict[str, Any]
        return self._layout_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            fg='number of fingers.',
            w='transistor width, in meters/number of fins.',
            mos_type="transistor type, either 'pch' or 'nch'.",
            threshold='transistor threshold flavor.',
            options='a dictionary of transistor options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        fmt = '%s_l%s_w%s_%s_%d'
        mos_type = self.params['mos_type']
        fg = self.params['fg']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        return fmt % (mos_type, lstr, wstr, th, fg)

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        mos_info = tech_cls.get_mos_info(lch_unit, w, mos_type, threshold, fg, **options)
        self._layout_info = mos_info['layout_info']
        # set parameters
        self._ext_top_info = mos_info['ext_top_info']
        self._ext_bot_info = mos_info['ext_bot_info']
        self._left_edge_info = mos_info['left_edge_info']
        self._right_edge_info = mos_info['right_edge_info']
        self._sd_yc = mos_info['sd_yc']
        self._g_conn_y = mos_info['g_conn_y']
        self._d_conn_y = mos_info['d_conn_y']
        self._od_y = mos_info['od_y']
        self._po_y = mos_info['po_y']

        # draw transistor
        tech_cls.draw_mos(self, self._layout_info)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()


class AnalogMOSExt(TemplateBase):
    """A primitive template of the geometry between transistor/substrate rows.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        self._layout_info = None
        self._left_edge_info = None
        self._right_edge_info = None
        self._sub_ysep = None

    @property
    def sub_ysep(self):
        # type: () -> Tuple[Optional[int], Optional[int]]
        return self._sub_ysep

    def get_edge_layout_info(self):
        # type: () -> Dict[str, Any]
        return self._layout_info

    def get_left_edge_info(self):
        # type: () -> Any
        return self._left_edge_info

    def get_right_edge_info(self):
        # type: () -> Any
        return self._right_edge_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            w='extension width, in resolution units/number of fins.',
            fg='number of fingers.',
            top_ext_info='top extension info.',
            bot_ext_info='bottom extension info.',
            is_laygo='True if this extension is used in LaygoBase.',
            options='Additional options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            is_laygo=False,
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        fmt = 'ext_l%s_w%s_fg%d'
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        fg = self.params['fg']
        ans = fmt % (lstr, wstr, fg)
        if self.params['is_laygo']:
            ans = 'laygo_' + ans
        return ans

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        top_ext_info = self.params['top_ext_info']
        bot_ext_info = self.params['bot_ext_info']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        ext_info = tech_cls.get_ext_info(lch_unit, w, fg, top_ext_info, bot_ext_info, **options)
        self._layout_info = ext_info['layout_info']
        self._left_edge_info = ext_info['left_edge_info']
        self._right_edge_info = ext_info['right_edge_info']
        self._sub_ysep = ext_info.get('sub_ysep', (None, None))
        tech_cls.draw_mos(self, self._layout_info)
        if self.params['is_laygo']:
            self.prim_top_layer = tech_cls.get_dig_conn_layer()
        else:
            self.prim_top_layer = tech_cls.get_mos_conn_layer()


class SubRingExt(TemplateBase):
    """A primitive template of the geometry inside a substrate ring.

    This template is just empty, but it contains the information needed to draw the left/right
    edges of a substrate ring.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        self._layout_info = None
        self._left_edge_info = None
        self._right_edge_info = None

    def get_edge_layout_info(self):
        # type: () -> Dict[str, Any]
        return self._layout_info

    def get_left_edge_info(self):
        # type: () -> Any
        return self._left_edge_info

    def get_right_edge_info(self):
        # type: () -> Any
        return self._right_edge_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            sub_type='substrate type.  Either ptap or ntap.',
            height='extension width, in resolution units.',
            fg='number of fingers.',
            end_ext_info='substrate ring inner end row extension info.',
            options='Optional layout parameters.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        fmt = 'subringext_%s_h%d_fg%d'
        sub_type = self.params['sub_type']
        h = self.params['height']
        fg = self.params['fg']
        ans = fmt % (sub_type, h, fg)
        return ans

    def draw_layout(self):
        sub_type = self.params['sub_type']
        h = self.params['height']
        fg = self.params['fg']
        end_ext_info = self.params['end_ext_info']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        ext_info = tech_cls.get_sub_ring_ext_info(sub_type, h, fg, end_ext_info, **options)
        self._layout_info = ext_info['layout_info']
        self._left_edge_info = ext_info['left_edge_info']
        self._right_edge_info = ext_info['right_edge_info']
        tech_cls.draw_mos(self, self._layout_info)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()


class DummyFillActive(TemplateBase):
    """A template that fills an area with active devices.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_min_fill_dim(cls, tech_info, mos_type, threshold):
        tech_cls = tech_info.tech_params['layout']['mos_tech_class']
        return tech_cls.get_min_fill_dim(mos_type, threshold)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            mos_type='transistor type.',
            threshold='transistor threshold.',
            width='The width of the fill area, in resolution units.',
            height='The height of the fill area, in resolution units.',
        )

    def get_layout_basename(self):
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        w = self.params['width']
        h = self.params['height']
        return '%s_%s_dummy_fill_w%d_h%d' % (mos_type, threshold, w, h)

    def draw_layout(self):
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        w = self.params['width']
        h = self.params['height']

        # draw fill
        tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        tech_cls.draw_active_fill(self, mos_type, threshold, w, h)

        # set size
        box = BBox(0, 0, w, h)
        self.prim_top_layer = 1
        self.array_box = self.prim_bound_box = box

        self.grid.tech_info.draw_device_blockage(self)
