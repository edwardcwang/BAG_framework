# -*- coding: utf-8 -*-

"""This module defines analog mosfet primitive template classes.
"""

from typing import TYPE_CHECKING, Dict, Any, Set, Tuple, Optional

from bag import float_to_si_string
from bag.layout.template import TemplateBase

from .core import MOSTech

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class AnalogMOSBase(TemplateBase):
    """A primitive template of a transistor row.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()

        self._layout_info = None
        self._ext_top_info = None
        self._ext_bot_info = None
        self._left_edge_info = None
        self._right_edge_info = None

        self._g_conn_y = None
        self._d_conn_y = None
        self._sd_yc = None

    def get_g_conn_y(self):
        # type: () -> Tuple[int, int]
        return self._g_conn_y

    def get_d_conn_y(self):
        # type: () -> Tuple[int, int]
        return self._d_conn_y

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
        )

    def get_layout_basename(self):
        fmt = '%s_l%s_w%s_%s_%d'
        mos_type = self.params['mos_type']
        fg = self.params['fg']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        return fmt % (mos_type, lstr, wstr, th, fg)

    def compute_unique_key(self):
        options = self.params['options']
        return self.to_immutable_id((self.get_layout_basename(), options))

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        options = self.params['options']

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        mos_info = self._tech_cls.get_mos_info(lch_unit, w, mos_type, threshold, fg, **options)
        self._layout_info = mos_info['layout_info']
        # set parameters
        self._ext_top_info = mos_info['ext_top_info']
        self._ext_bot_info = mos_info['ext_bot_info']
        self._left_edge_info = mos_info['left_edge_info']
        self._right_edge_info = mos_info['right_edge_info']
        self._sd_yc = mos_info['sd_yc']
        self._g_conn_y = mos_info['g_conn_y']
        self._d_conn_y = mos_info['d_conn_y']

        # draw transistor
        self._tech_cls.draw_mos(self, self._layout_info)


class AnalogMOSExt(TemplateBase):
    """A primitive template of the geometry between transistor/substrate rows.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self._layout_info = None
        if self.params['is_laygo']:
            self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        else:
            self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
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
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(is_laygo=False)

    def get_layout_basename(self):
        fmt = 'ext_l%s_w%s_fg%d'
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        fg = self.params['fg']
        ans = fmt % (lstr, wstr, fg)
        if self.params['is_laygo']:
            ans = 'laygo_' + ans
        return ans

    def compute_unique_key(self):
        key = self.get_layout_basename(), self.params['top_ext_info'], self.params['bot_ext_info']
        return self.to_immutable_id(key)

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        top_ext_info = self.params['top_ext_info']
        bot_ext_info = self.params['bot_ext_info']

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        ext_info = self._tech_cls.get_ext_info(lch_unit, w, fg, top_ext_info, bot_ext_info)
        self._layout_info = ext_info['layout_info']
        self._left_edge_info = ext_info['left_edge_info']
        self._right_edge_info = ext_info['right_edge_info']
        self._sub_ysep = ext_info.get('sub_ysep', (None, None))
        self._tech_cls.draw_mos(self, self._layout_info)


class SubRingExt(TemplateBase):
    """A primitive template of the geometry inside a substrate ring.

    This template is just empty, but it contains the information needed to draw the left/right
    edges of a substrate ring.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        super(SubRingExt, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
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
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(options={}, )

    def get_layout_basename(self):
        fmt = 'subringext_%s_h%d_fg%d'
        sub_type = self.params['sub_type']
        h = self.params['height']
        fg = self.params['fg']
        ans = fmt % (sub_type, h, fg)
        return ans

    def compute_unique_key(self):
        key = self.get_layout_basename(), self.params['end_ext_info'], self.params['options']
        return self.to_immutable_id(key)

    def draw_layout(self):
        sub_type = self.params['sub_type']
        h = self.params['height']
        fg = self.params['fg']
        end_ext_info = self.params['end_ext_info']
        options = self.params['options']

        ext_info = self._tech_cls.get_sub_ring_ext_info(sub_type, h, fg, end_ext_info, **options)
        self._layout_info = ext_info['layout_info']
        self._left_edge_info = ext_info['left_edge_info']
        self._right_edge_info = ext_info['right_edge_info']
        self._tech_cls.draw_mos(self, self._layout_info)
