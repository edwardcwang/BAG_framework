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


"""This package defines various passives template classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Set, Any, List, Tuple

from bag.layout.util import BBox
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase, TemplateDB

from ..resistor.core import ResArrayBase
from ..analog_core import SubstrateContact


class BiasResistorDiff(ResArrayBase):
    """differential bias resistor for differential high pass filter.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(BiasResistorDiff, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
            em_specs={},
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
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            p_idx='positive output track index',
            n_idx='negative output track index',
            output_width='output track width',
            num_seg='number of segments in each resistor.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def connect_series_resistor(self, col_start, col_stop):
        last_port = None
        port_out = None
        if col_stop < col_start:
            col_iter = range(col_start, col_stop, -1)
        else:
            col_iter = range(col_start, col_stop)

        for idx in col_iter:
            bot_warr, top_warr = self.get_res_ports(0, idx)
            if last_port is None:
                last_port = top_warr, 0
                port_out = bot_warr
            else:
                if last_port[1] == 0:
                    self.connect_wires([last_port[0], bot_warr])
                    last_port = top_warr, 1
                else:
                    self.connect_wires([last_port[0], top_warr])
                    last_port = bot_warr, 0

        return port_out, last_port[0]

    def draw_layout(self):
        # type: () -> None

        kwargs = self.params.copy()
        num_seg = kwargs.pop('num_seg')
        p_idx = kwargs.pop('p_idx')
        n_idx = kwargs.pop('n_idx')
        output_width = kwargs.pop('output_width')

        # draw array
        self.draw_array(nx=2 * num_seg, ny=1, edge_space=True, **kwargs)

        # connect resistor in series
        p_port_out, p_port_bias = self.connect_series_resistor(num_seg - 1, -1)
        n_port_out, n_port_bias = self.connect_series_resistor(num_seg, 2 * num_seg)

        # connect to vertical layer
        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        vm_width = self.w_tracks[1]
        vports = []
        for warr in [p_port_out, n_port_out, p_port_bias, n_port_bias]:
            vtid = self.grid.coord_to_nearest_track(vm_layer, warr.middle, half_track=True)
            vtid = TrackID(vm_layer, vtid, width=vm_width)
            vports.append(self.connect_to_tracks(warr, vtid))

        # connect to horizontal layer
        num_xm_tracks = self.array_box.height_unit // self.grid.get_track_pitch(xm_layer, unit_mode=True)
        tid = TrackID(xm_layer, num_xm_tracks - 1)
        for warr, name in zip(vports[2:], ('biasp', 'biasn')):
            warr = self.connect_to_tracks(warr, tid, min_len_mode=0)
            self.add_pin(name, warr, show=False)

        outp, outn = self.connect_differential_tracks(vports[0], vports[1], xm_layer, p_idx, n_idx, width=output_width)
        self.add_pin('outp', outp, show=False)
        self.add_pin('outn', outn, show=False)


class HighPassFilterDiff(TemplateBase):
    """An template for creating high pass filter.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(HighPassFilterDiff, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
            em_specs={},
            show_pins=True,
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
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            cap_edge_margin='margin between cap to block edge',
            cap_diff_margin='margin between differential cap',
            num_seg='number of segments in each resistor.',
            num_cap_layer='number of layers to use for AC coupling cap.',
            tr_idx_list='input track indices',
            io_width='input/output track width',
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None

        edge_margin = self.params['cap_edge_margin'] / self.grid.layout_unit
        diff_margin = self.params['cap_diff_margin'] / self.grid.layout_unit
        num_cap_layer = self.params['num_cap_layer']

        # place instances
        io_layer, io_idx_list, io_width, bias_tr_id = self.place()

        # draw AC coupling caps bounding boxes
        # step 1: figure out cap bottom Y coordinate
        top_io_idx = max(io_idx_list)
        num_space = self.grid.get_num_space_tracks(io_layer, io_width)
        bot_avail_track = top_io_idx + num_space + (io_width + 1) / 2
        yb = self.grid.get_wire_bounds(io_layer, bot_avail_track, unit_mode=True)[0]
        # step 2: figure out cap top Y coordinate
        bias_idx = bias_tr_id.base_index
        num_space = self.grid.get_num_space_tracks(io_layer, bias_tr_id.width)
        top_avail_track = bias_idx - num_space - (bias_tr_id.width + 1) / 2
        yt = self.grid.get_wire_bounds(io_layer, top_avail_track, unit_mode=True)[1]
        # step 3: figure out cap left/right X coordinate
        res = self.grid.resolution
        blk_w = self.grid.get_size_dimension(self.size, unit_mode=False)[0]
        xpl = int(round(edge_margin / res))
        xpr = int(round((blk_w - diff_margin) / (2 * res)))
        xnl = int(round((blk_w + diff_margin) / (2 * res)))
        xnr = int(round((blk_w - edge_margin) / res))

        pwarr, nwarr = self.draw_cap(xpl, yb, xpr, yt, io_layer, io_idx_list[:2], io_width, num_cap_layer)
        pwarr, nwarr = self.draw_cap(xnl, yb, xnr, yt, io_layer, io_idx_list[2:], io_width, num_cap_layer)

    def draw_cap(self, xl, yb, xr, yt, io_layer, io_idx_list, io_width, num_cap_layer):
        cap_box = BBox(xl, yb, xr, yt, self.grid.resolution, unit_mode=True)
        top_layer = io_layer + num_cap_layer - 1
        self.add_rect(self.grid.tech_info.get_layer_name(top_layer), cap_box)
        return None, None

    def place(self):
        # type: () -> Tuple[int, List[float], int, TrackID]
        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        inp_idx, inn_idx = res_params.pop('tr_idx_list')
        io_width = res_params.pop('io_width')

        # compute resistor output track indices and create resistor
        io_layer = BiasResistorDiff.get_port_layer_id(self.grid.tech_info) + 2
        num_space = self.grid.get_num_space_tracks(io_layer, io_width)
        if inp_idx > inn_idx:
            outp_idx = inp_idx + io_width + num_space
            outn_idx = inn_idx - io_width - num_space
        else:
            outp_idx = inp_idx - io_width - num_space
            outn_idx = inn_idx + io_width + num_space
        res_params['p_idx'] = outp_idx
        res_params['n_idx'] = outn_idx
        res_params['output_width'] = io_width

        res_master = self.new_template(params=res_params, temp_cls=BiasResistorDiff)

        # draw contact and move array up
        sub_type = self.params['sub_type']
        show_pins = self.params['show_pins']
        top_layer, nx_arr, ny_arr = res_master.size
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            top_layer=top_layer,
            blk_width=nx_arr,
            show_pins=False,
        )
        _, blk_h = self.grid.get_block_size(top_layer)
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        ny_shift = sub_master.size[2]
        res_inst = self.add_instance(res_master, inst_name='XRES')
        top_yo = (ny_arr + ny_shift) * blk_h
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(0.0, top_yo), orient='MX')

        # export supply and recompute array_box/size
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(top_inst.get_port(port_name), show=show_pins)
        self.size = top_layer, nx_arr, ny_arr + ny_shift
        self.array_box = top_inst.array_box.extend(y=0, unit_mode=True)

        # export bias ports
        for port_name in ('biasp', 'biasn'):
            self.reexport(res_inst.get_port(port_name), show=show_pins)
        bias_tr_id = res_inst.get_all_port_pins('biasp')[0].track_id

        return io_layer, [inp_idx, outp_idx, inn_idx, outn_idx], io_width, bias_tr_id
