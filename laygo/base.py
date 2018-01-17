# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

from bag import float_to_si_string
from bag.layout.template import TemplateBase, TemplateDB

from .tech import LaygoTech


class LaygoPrimitive(TemplateBase):
    """A Laygo primitive block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoPrimitive, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._num_col = 1
        self._blk_info = None

    def get_left_edge_info(self):
        return self._blk_info['left_edge_info']

    def get_right_edge_info(self):
        return self._blk_info['right_edge_info']

    def get_ext_bot_info(self):
        return self._blk_info['ext_bot_info']

    def get_ext_top_info(self):
        return self._blk_info['ext_top_info']

    @property
    def laygo_size(self):
        return self._num_col, 1

    @classmethod
    def get_default_param_values(cls):
        return dict(options=None)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            blk_type="digital block type.",
            w='transistor width, in meters/number of fins.',
            row_info='laygo row information dictionary.',
            options="layout options.",
        )

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

        self._blk_info = self._tech_cls.get_laygo_blk_info(blk_type, w, row_info, **options)
        layout_info = self._blk_info['layout_info']
        self._num_col = layout_info['fg']
        # draw transistor
        self._tech_cls.draw_mos(self, layout_info)
        # draw connection
        if options is None:
            options = {}
        self._tech_cls.draw_laygo_connection(self, self._blk_info, blk_type, options)


class LaygoSubstrate(TemplateBase):
    """A laygo substrate block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoSubstrate, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._blk_info = None
        self._num_col = 1

    def get_left_edge_info(self):
        return self._blk_info['left_edge_info']

    def get_right_edge_info(self):
        return self._blk_info['right_edge_info']

    def get_ext_bot_info(self):
        return self._blk_info['ext_bot_info']

    def get_ext_top_info(self):
        return self._blk_info['ext_top_info']

    @property
    def row_info(self):
        return self.params['row_info']

    @property
    def laygo_size(self):
        return self._num_col, 1

    @classmethod
    def get_default_param_values(cls):
        return dict(options=None)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            row_info='laygo row information dictionary.',
            options="additional substrate options.",
        )

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

        w_sub = row_info['w_sub']
        self._blk_info = self._tech_cls.get_laygo_blk_info('sub', w_sub, row_info, **options)

        layout_info = self._blk_info['layout_info']
        self._num_col = layout_info['fg']
        # draw transistor
        self._tech_cls.draw_mos(self, layout_info)
        # draw connection
        if options is None:
            options = {}
        self._tech_cls.draw_laygo_connection(self, self._blk_info, 'sub', options)


class LaygoEndRow(TemplateBase):
    """A laygo end row block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoEndRow, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._end_info = None

    def get_edge_layout_info(self):
        return self._end_info['layout_info']

    def get_left_edge_info(self):
        return self._end_info['left_edge_info']

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
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
        self._end_info = self._tech_cls.get_laygo_end_info(lch_unit, mos_type, threshold, 1, is_end, blk_pitch)
        self._tech_cls.draw_mos(self, self._end_info['layout_info'])


class LaygoSpace(TemplateBase):
    """A laygo space block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(LaygoSpace, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech
        self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        self._num_blk = self.params['num_blk']
        self._blk_info = None

    def get_left_edge_info(self):
        return self._blk_info['left_edge_info']

    def get_right_edge_info(self):
        return self._blk_info['right_edge_info']

    def get_ext_bot_info(self):
        return self._blk_info['ext_bot_info']

    def get_ext_top_info(self):
        return self._blk_info['ext_top_info']

    @property
    def laygo_size(self):
        return self._num_blk, 1

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
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

        self._blk_info = self._tech_cls.get_laygo_space_info(row_info, num_blk, left_blk_info, right_blk_info)
        # draw transistor
        self._tech_cls.draw_mos(self, self._blk_info['layout_info'])
        self._tech_cls.draw_laygo_space_connection(self, self._blk_info, left_blk_info, right_blk_info)
