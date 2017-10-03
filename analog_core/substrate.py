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

"""This module defines various substrate related classes."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Any, Set

from bag.util.search import BinaryIterator
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import TrackID
from bag.layout.util import BBox

from ..analog_mos.substrate import AnalogSubstrate
from ..analog_mos.edge import AnalogEdge, AnalogEndRow
from ..analog_mos.conn import AnalogSubstrateConn

from .base import AnalogBaseInfo


class SubstrateContact(TemplateBase):
    """A template that draws a single substrate.

    Useful for resistor/capacitor body biasing.

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
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(SubstrateContact, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @property
    def port_name(self):
        return 'VDD' if self.params['sub_type'] == 'ntap' else 'VSS'

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Dict[str, Any]
            dictionary of default parameter values.
        """
        return dict(
            well_end_mode=0,
            show_pins=False,
            is_passive=False,
        )

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            top_layer='the top layer of the template.',
            lch='channel length, in meters.',
            w='substrate width, in meters/number of fins.',
            sub_type='substrate type.',
            threshold='substrate threshold flavor.',
            well_width='Width of the well in layout units.',
            show_pins='True to show pin labels.',
            is_passive='True if this substrate is used as substrate contact for passive devices.',
        )

    def draw_layout(self):
        # type: () -> None

        top_layer = self.params['top_layer']
        lch = self.params['lch']
        w = self.params['w']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        well_width = self.params['well_width']
        show_pins = self.params['show_pins']
        is_passive = self.params['is_passive']

        sub_end_mode = 15
        res = self.grid.resolution
        well_width = int(round(well_width / res))

        # get layout info, also set RoutingGrid to substrate grid.
        layout_info = AnalogBaseInfo(self.grid, lch, 0, top_layer=top_layer, end_mode=sub_end_mode)

        # compute template width in number of sd pitches
        # find maximum number of fingers we can draw
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_fg = bin_iter.get_next()
            cur_core_width = layout_info.get_core_width(cur_fg)
            if cur_core_width == well_width:
                bin_iter.save()
                break
            elif cur_core_width < well_width:
                bin_iter.save()
                bin_iter.up()
            else:
                bin_iter.down()

        sub_fg_tot = bin_iter.get_last_save()
        if sub_fg_tot is None:
            raise ValueError('Cannot draw substrate that fit in width: %d' % well_width)

        layout_info.fg_tot = sub_fg_tot
        self.grid = layout_info.grid

        place_info = layout_info.get_placement_info(sub_fg_tot)
        edgel_x0 = place_info.edge_margins[0]
        arr_box_x = place_info.arr_box_x
        tot_width = place_info.tot_width

        # create substrate
        params = dict(
            lch=lch,
            w=w,
            sub_type=sub_type,
            threshold=threshold,
            fg=sub_fg_tot,
            is_passive=is_passive,
            top_layer=top_layer,
        )
        sub_master = self.new_template(params=params, temp_cls=AnalogSubstrate)
        edge_layout_info = sub_master.get_edge_layout_info()
        edge_params = dict(
            is_end=True,
            guard_ring_nf=0,
            name_id=sub_master.get_layout_basename(),
            layout_info=edge_layout_info,
            adj_blk_info=sub_master.get_left_edge_info(),
        )
        edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)

        end_row_params = dict(
            lch=lch,
            fg=sub_fg_tot,
            sub_type=sub_type,
            threshold=threshold,
            is_end=True,
            top_layer=top_layer,
        )
        end_row_master = self.new_template(params=end_row_params, temp_cls=AnalogEndRow)
        end_edge_params = dict(
            is_end=True,
            guard_ring_nf=0,
            name_id=end_row_master.get_layout_basename(),
            layout_info=end_row_master.get_edge_layout_info(),
            adj_blk_info=end_row_master.get_left_edge_info(),
        )
        end_edge_master = self.new_template(params=end_edge_params, temp_cls=AnalogEdge)
        conn_params = dict(
            layout_info=edge_layout_info,
            layout_name=sub_master.get_layout_basename() + '_subconn',
            is_laygo=False,
        )
        conn_master = self.new_template(params=conn_params, temp_cls=AnalogSubstrateConn)

        # find substrate height and set size
        hsub = sub_master.prim_bound_box.height_unit
        hend = end_row_master.prim_bound_box.height_unit
        htot = hsub + 2 * hend
        # add substrate and edge at the right locations
        x1 = edgel_x0 + edge_master.prim_bound_box.width_unit
        x2 = x1 + sub_master.prim_bound_box.width_unit + edge_master.prim_bound_box.width_unit
        y1 = end_edge_master.prim_bound_box.height_unit
        y2 = y1 + edge_master.prim_bound_box.height_unit + end_edge_master.prim_bound_box.height_unit
        instlb = self.add_instance(end_edge_master, inst_name='XLBE', loc=(edgel_x0, 0), unit_mode=True)
        self.add_instance(edge_master, inst_name='XLE', loc=(edgel_x0, y1), unit_mode=True)
        self.add_instance(end_edge_master, inst_name='XLTE', orient='MX', loc=(edgel_x0, y2), unit_mode=True)
        self.add_instance(end_row_master, inst_name='XB', loc=(x1, 0), unit_mode=True)
        self.add_instance(sub_master, inst_name='XSUB', loc=(x1, y1), unit_mode=True)
        self.add_instance(end_row_master, inst_name='XT', orient='MX', loc=(x1, y2), unit_mode=True)
        self.add_instance(end_edge_master, inst_name='XRBE', orient='MY', loc=(x2, 0), unit_mode=True)
        self.add_instance(edge_master, inst_name='XRE', orient='MY', loc=(x2, y1), unit_mode=True)
        sub_conn = self.add_instance(conn_master, inst_name='XSUBCONN', loc=(x1, y1), unit_mode=True)
        instrt = self.add_instance(end_edge_master, inst_name='XRTE', orient='R180', loc=(x2, y2), unit_mode=True)

        # set array box and size
        arr_box = instlb.array_box.merge(instrt.array_box)
        self.array_box = BBox(arr_box_x[0], arr_box.bottom_unit, arr_box_x[1], arr_box.top_unit, res, unit_mode=True)
        self.set_size_from_bound_box(top_layer, BBox(0, 0, tot_width, htot, res, unit_mode=True))
        self.add_cell_boundary(self.bound_box)

        # find center track index
        hm_layer = layout_info.mconn_port_layer + 1
        hm_mid = self.grid.coord_to_nearest_track(hm_layer, self.array_box.yc_unit, mode=0,
                                                  half_track=True, unit_mode=True)
        # connect to horizontal metal layer.
        hm_pitch = self.grid.get_track_pitch(hm_layer, unit_mode=True)
        ntr = self.array_box.height_unit // hm_pitch  # type: int
        tr_width = self.grid.get_max_track_width(hm_layer, 1, ntr, half_end_space=False)
        track_id = TrackID(hm_layer, hm_mid, width=tr_width)
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sub_wires = self.connect_to_tracks(sub_conn.get_port(port_name).get_pins(hm_layer - 1), track_id)
        self.add_pin(port_name, sub_wires, show=show_pins)
