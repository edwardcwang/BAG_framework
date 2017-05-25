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

from bag.layout.template import TemplateBase, TemplateDB

from .core import MOSTech
from .substrate import AnalogSubstrateCore


class AnalogOuterEdge(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogOuterEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_bound_box = None

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
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_mos(self, self.params['layout_info'])
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()


class AnalogGuardRingSep(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogGuardRingSep, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_bound_box = None

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
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_mos(self, self.params['layout_info'])
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()


class AnalogEdge(TemplateBase):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(AnalogEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
        self.prim_bound_box = None

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
            top_layer='The top layer used to calculate width quantization.',
            is_end='True if this edge is at the end.',
            guard_ring_nf='number of guard ring fingers.',
            name_id='cell name ID.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        base = 'edge_%s_gr%d_lay%d' % (self.params['name_id'], self.params['guard_ring_nf'], self.params['top_layer'])
        if self.params['is_end']:
            base += '_end'
        return base

    def compute_unique_key(self):
        base_name = self.get_layout_basename()
        return self.to_immutable_id((base_name, self.params['layout_info'], self.params['flip_parity']))

    def draw_layout(self):
        top_layer = self.params['top_layer']
        is_end = self.params['is_end']
        guard_ring_nf = self.params['guard_ring_nf']
        layout_info = self.params['layout_info']
        basename = self.get_layout_basename()

        out_info = self._tech_cls.get_outer_edge_info(self.grid, guard_ring_nf, layout_info, top_layer, is_end)
        # add outer edge
        out_params = dict(
            layout_name='%s_outer' % basename,
            layout_info=out_info,
        )
        master = self.new_template(params=out_params, temp_cls=AnalogOuterEdge)
        self.add_instance(master, 'XOUTER')

        self.array_box = master.array_box
        self.prim_bound_box = master.prim_bound_box

        if guard_ring_nf > 0:
            # draw guard ring and guard ring separator
            x0 = self.array_box.right_unit
            sub_info = self._tech_cls.get_gr_sub_info(guard_ring_nf, layout_info)
            loc = x0, 0
            sub_params = dict(
                dummy_only=False,
                port_tracks=[],
                dum_tracks=[],
                layout_name='%s_sub' % basename,
                layout_info=sub_info,
            )
            master = self.new_template(params=sub_params, temp_cls=AnalogSubstrateCore)
            inst = self.add_instance(master, 'XSUB', loc=loc, unit_mode=True)

            for port_name in inst.port_names_iter():
                self.reexport(inst.get_port(port_name), show=False)

            x0 = inst.array_box.right_unit
            sep_info = self._tech_cls.get_gr_sep_info(layout_info)
            sep_params = dict(
                layout_name='%s_sep' % basename,
                layout_info=sep_info,
            )
            master = self.new_template(params=sep_params, temp_cls=AnalogGuardRingSep)
            inst = self.add_instance(master, 'XSEP', loc=(x0, 0), unit_mode=True)
            self.array_box = self.array_box.merge(inst.array_box)
            self.prim_bound_box = self.prim_bound_box.merge(inst.translate_master_box(master.prim_bound_box))
