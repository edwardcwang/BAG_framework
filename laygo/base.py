# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

from bag import float_to_si_string
from bag.layout.template import TemplateBase, TemplateDB

from .tech import LaygoTech


class LaygoPrimitive(TemplateBase):
    """A Laygo primitive block.

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
    kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        tech_info = self.grid.tech_info
        self._tech_cls = tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._num_col = 0
        self._lr_edge_info = None
        self._tb_ext_info = None
        self._layout_info = None

    @property
    def num_col(self):
        return self._num_col

    @property
    def lr_edge_info(self):
        return self._lr_edge_info

    @property
    def tb_ext_info(self):
        return self._tb_ext_info

    @property
    def layout_info(self):
        return self._layout_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            blk_type="digital block type.",
            w='transistor width, in meters/number of fins.',
            row_info='laygo row information dictionary.',
            options="layout options.",
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(options=None)

    def get_layout_basename(self):
        fmt = 'laygo_%s_l%s_w%s_%s_%s'
        blk_type = self.params['blk_type']
        wstr = float_to_si_string(self.params['w'])
        row_info = self.params['row_info']
        row_type = row_info['row_type']
        th = row_info['threshold']
        lch_unit = row_info['lch_unit']
        lstr = float_to_si_string(lch_unit * self.grid.layout_unit * self.grid.resolution)
        return fmt % (row_type, lstr, wstr, th, blk_type)

    def draw_layout(self):
        blk_type = self.params['blk_type']
        w = self.params['w']
        row_info = self.params['row_info']
        options = self.params['options']

        if options is None:
            options = {}

        blk_info = self._tech_cls.get_laygo_blk_info(blk_type, w, row_info, **options)
        self._lr_edge_info = blk_info['left_edge_info'], blk_info['right_edge_info']
        self._tb_ext_info = blk_info['ext_top_info'], blk_info['ext_bot_info']
        self._layout_info = blk_info['layout_info']
        self._num_col = self._layout_info['fg']
        # draw transistor
        self._tech_cls.draw_mos(self, self._layout_info)
        # draw connection
        self._tech_cls.draw_laygo_connection(self, blk_info, blk_type, options)


class LaygoSubstrate(TemplateBase):
    """A laygo substrate block.

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
    kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        tech_info = self.grid.tech_info
        self._tech_cls = tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._num_col = 1
        self._lr_edge_info = None
        self._tb_ext_info = None
        self._layout_info = None

    @property
    def num_col(self):
        return self._num_col

    @property
    def lr_edge_info(self):
        return self._lr_edge_info

    @property
    def tb_ext_info(self):
        return self._tb_ext_info

    @property
    def layout_info(self):
        return self._layout_info

    @property
    def row_info(self):
        return self.params['row_info']

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            row_info='laygo row information dictionary.',
            options="additional substrate options.",
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(options=None)

    def get_layout_basename(self):
        fmt = 'laygo_%s_l%s_w%s_%s'
        row_info = self.params['row_info']
        sub_type = row_info['sub_type']
        th = row_info['threshold']
        wstr = float_to_si_string(row_info['w_sub'])
        lch_unit = row_info['lch_unit']
        lstr = float_to_si_string(lch_unit * self.grid.layout_unit * self.grid.resolution)
        return fmt % (sub_type, lstr, wstr, th)

    def draw_layout(self):
        row_info = self.params['row_info']
        options = self.params['options']

        if options is None:
            options = {}

        w_sub = row_info['w_sub']
        blk_info = self._tech_cls.get_laygo_blk_info('sub', w_sub, row_info, **options)
        self._lr_edge_info = blk_info['left_edge_info'], blk_info['right_edge_info']
        self._tb_ext_info = blk_info['ext_top_info'], blk_info['ext_bot_info']
        self._layout_info = blk_info['layout_info']
        self._num_col = self._layout_info['fg']
        # draw transistor
        self._tech_cls.draw_mos(self, self._layout_info)
        # draw connection
        self._tech_cls.draw_laygo_connection(self, blk_info, 'sub', options)


class LaygoEndRow(TemplateBase):
    """A laygo end row block.

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
    kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        tech_info = self.grid.tech_info
        self._tech_cls = tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._end_info = None

    def get_edge_layout_info(self):
        return self._end_info['layout_info']

    def get_left_edge_info(self):
        return self._end_info['left_edge_info']

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            mos_type="thetransistor type, one of 'pch', 'nch', 'ptap', or 'ntap'.",
            threshold='transistor threshold flavor.',
            is_end='True if there are no blocks abutting the bottom.',
            top_layer='The top routing layer.  Used to determine height quantization.',
        )

    def get_layout_basename(self):
        lstr = float_to_si_string(self.params['lch'])
        mos_type = self.params['mos_type']
        thres = self.params['threshold']
        top_layer = self.params['top_layer']
        is_end = self.params['is_end']

        fmt = 'laygo_%s_end_l%s_%s_lay%d'
        basename = fmt % (mos_type, lstr, thres, top_layer)
        if is_end:
            basename += '_end'

        return basename

    def draw_layout(self):
        lch_unit = int(round(self.params['lch'] / self.grid.layout_unit / self.grid.resolution))
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        is_end = self.params['is_end']
        top_layer = self.params['top_layer']

        blk_pitch = self.grid.get_block_size(top_layer, unit_mode=True)[1]
        self._end_info = self._tech_cls.get_laygo_end_info(lch_unit, mos_type, threshold, 1,
                                                           is_end, blk_pitch)
        self._tech_cls.draw_mos(self, self._end_info['layout_info'])


class LaygoSpace(TemplateBase):
    """A laygo space block.

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
    kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        tech_info = self.grid.tech_info
        self._tech_cls = tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._num_col = 0
        self._lr_edge_info = None
        self._tb_ext_info = None
        self._layout_info = None

    @property
    def num_col(self):
        return self._num_col

    @property
    def lr_edge_info(self):
        return self._lr_edge_info

    @property
    def tb_ext_info(self):
        return self._tb_ext_info

    @property
    def layout_info(self):
        return self._layout_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            row_info='the Laygo row information dictionary.',
            num_blk='number of space blocks.',
            left_blk_info='left block layout information.',
            right_blk_info='right block layout information.',
        )

    def get_layout_basename(self):
        fmt = '%s_space%d'
        name_id = self.params['row_info']['row_name_id']
        num_blk = self.params['num_blk']
        return fmt % (name_id, num_blk)

    def draw_layout(self):
        row_info = self.params['row_info']
        num_blk = self.params['num_blk']
        left_blk_info = self.params['left_blk_info']
        right_blk_info = self.params['right_blk_info']

        blk_info = self._tech_cls.get_laygo_space_info(row_info, num_blk, left_blk_info,
                                                       right_blk_info)
        self._lr_edge_info = blk_info['left_edge_info'], blk_info['right_edge_info']
        self._tb_ext_info = blk_info['ext_top_info'], blk_info['ext_bot_info']
        self._layout_info = blk_info['layout_info']
        self._num_col = num_blk

        # draw transistor
        self._tech_cls.draw_mos(self, self._layout_info)
        self._tech_cls.draw_laygo_space_connection(self, blk_info, left_blk_info,
                                                   right_blk_info)
