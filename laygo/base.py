# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Any, Set

from bag import float_to_si_string
from bag.layout.template import TemplateBase, TemplateDB

from .tech import LaygoTech


class LaygoPrimitive(TemplateBase):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

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
        self._row_info = None

    def get_left_edge_info(self):
        return self._row_info['left_edge_info']

    def get_right_edge_info(self):
        return self._row_info['right_edge_info']

    def get_ext_bot_info(self):
        return self._row_info['ext_bot_info']

    def get_ext_top_info(self):
        return self._row_info['ext_top_info']

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
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            mos_type="transistor type, one of 'pch', 'nch', 'ntap', or 'ptap'.",
            threshold='transistor threshold flavor.',
            blk_type="digital block type.",
            options="digital block options.",
            bot_row_type='gate side laygo row type.',
            top_row_type='drain/source side laygo row type.',
        )

    def get_layout_basename(self):
        fmt = 'laygo_%s_l%s_w%s_%s_%s'
        mos_type = self.params['mos_type']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        blk_type = self.params['blk_type']
        return fmt % (mos_type, lstr, wstr, th, blk_type)

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return self.to_immutable_id((basename, self.params['options'], self.grid.get_flip_parity()))

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        blk_type = self.params['blk_type']
        options = self.params['options']
        bot_row_type = self.params['bot_row_type']
        top_row_type = self.params['top_row_type']

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        if blk_type == 'sub':
            self._num_col = self._tech_cls.get_sub_columns(lch_unit)
        self._row_info = self._tech_cls.get_laygo_mos_info(lch_unit, w, mos_type, threshold, blk_type,
                                                           bot_row_type, top_row_type, **options)
        # draw transistor
        self._tech_cls.draw_mos(self, self._row_info['layout_info'])
        # draw connection
        if options is None:
            options = {}
        self._tech_cls.draw_laygo_connection(self, self._row_info, blk_type, options)


class LaygoSubstrate(TemplateBase):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

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
        self._row_info = None

    def get_left_edge_info(self):
        return self._row_info['left_edge_info']

    def get_right_edge_info(self):
        return self._row_info['right_edge_info']

    def get_ext_bot_info(self):
        return self._row_info['ext_bot_info']

    def get_ext_top_info(self):
        return self._row_info['ext_top_info']

    @property
    def row_info(self):
        return self._row_info

    @property
    def laygo_size(self):
        return 1, 1

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
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            mos_type="transistor type, one of 'pch', 'nch', 'ntap', or 'ptap'.",
            threshold='transistor threshold flavor.',
            options="additional substrate options.",
        )

    def get_layout_basename(self):
        fmt = 'laygo_%s_l%s_w%s_%s'
        mos_type = self.params['mos_type']
        lstr = float_to_si_string(self.params['lch'])
        wstr = float_to_si_string(self.params['w'])
        th = self.params['threshold']
        return fmt % (mos_type, lstr, wstr, th)

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return self.to_immutable_id((basename, self.params['options'], self.grid.get_flip_parity()))

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        options = self.params['options']

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))

        self._row_info = self._tech_cls.get_laygo_sub_info(lch_unit, w, mos_type, threshold, **options)
        # draw transistor
        self._tech_cls.draw_mos(self, self._row_info['layout_info'])
        # draw connection
        if options is None:
            options = {}
        self._tech_cls.draw_laygo_connection(self, self._row_info, 'sub', options)


class LaygoEndRow(TemplateBase):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

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
        self._fg = self._tech_cls.get_laygo_unit_fg()
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

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self):
        lch_unit = int(round(self.params['lch'] / self.grid.layout_unit / self.grid.resolution))
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        is_end = self.params['is_end']
        top_layer = self.params['top_layer']

        blk_pitch = self.grid.get_block_size(top_layer, unit_mode=True)[1]
        self._end_info = self._tech_cls.get_laygo_end_info(lch_unit, mos_type, threshold, self._fg, is_end, blk_pitch)
        self._tech_cls.draw_mos(self, self._end_info['layout_info'])

class LaygoSpace(TemplateBase):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

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
        self._edge_info = self._tech_cls.get_default_end_info()
        self._row_info = None
        self._ext_bot_info = None
        self._ext_top_info = None

    def get_left_edge_info(self):
        return self._edge_info

    def get_right_edge_info(self):
        return self._edge_info

    def get_ext_bot_info(self):
        return self._ext_bot_info

    def get_ext_top_info(self):
        return self._ext_top_info

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
            name_id='the layout name ID.',
            num_blk='number of space blocks.',
            left_blk_info='left block layout information.',
            right_blk_info='right block layout information.',
        )

    def get_layout_basename(self):
        fmt = '%s_space%d'
        name_id = self.params['name_id']
        num_blk = self.params['num_blk']
        return fmt % (name_id, num_blk)

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return self.to_immutable_id((basename, self.params['row_info'],
                                     self.params['left_blk_info'], self.params['right_blk_info']))

    def draw_layout(self):
        row_info = self.params['row_info']
        num_blk = self.params['num_blk']
        left_blk_info = self.params['left_blk_info']
        right_blk_info = self.params['right_blk_info']

        self._row_info = self._tech_cls.get_laygo_space_info(row_info, num_blk, left_blk_info, right_blk_info)
        # draw transistor
        self._tech_cls.draw_mos(self, self._row_info['layout_info'])
        self._ext_bot_info, self._ext_top_info = self._tech_cls.draw_laygo_space_connection(self, self._row_info,
                                                                                            left_blk_info,
                                                                                            right_blk_info)
