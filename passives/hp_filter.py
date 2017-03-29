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

from typing import Dict, Set, Any

from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase, TemplateDB

from ..resistor.core import ResArrayBase
from ..analog_core import SubstrateContact


class BiasResistor(ResArrayBase):
    """the bias resistor template.

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
        super(BiasResistor, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def draw_layout(self):
        # type: () -> None

        # draw array
        self.draw_array(nx=1, ny=1, edge_space=True, **self.params)

        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1

        # connect to vertical layer
        bot_warr, top_warr = self.get_res_ports(0, 0)
        vm_width = self.w_tracks[1]
        vtid = self.grid.coord_to_nearest_track(vm_layer, bot_warr.middle, half_track=True)
        vtid = TrackID(vm_layer, vtid, width=vm_width)
        bot_warr = self.connect_to_tracks(bot_warr, vtid)
        top_warr = self.connect_to_tracks(top_warr, vtid)

        # connect to horizontal layer
        tid = TrackID(xm_layer, 0)
        warr = self.connect_to_tracks(bot_warr, tid, min_len_mode=0)
        self.add_pin('bot', warr, show=False)
        num_xm_tracks = self.array_box.height_unit // self.grid.get_track_pitch(xm_layer, unit_mode=True)
        tid = TrackID(xm_layer, num_xm_tracks - 1)
        warr = self.connect_to_tracks(top_warr, tid, min_len_mode=0)
        self.add_pin('top', warr, show=False)


class HighPassFilter(TemplateBase):
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
        super(HighPassFilter, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']

        res_master = self.new_template(params=res_params, temp_cls=BiasResistor)

        # draw contact and move array up
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

        # export supplies and recompute array_box/size
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(top_inst.get_port(port_name))
        self.size = top_layer, nx_arr, ny_arr + ny_shift
        self.array_box = top_inst.array_box.extend(y=0, unit_mode=True)

        for port_name in res_inst.port_names_iter():
            self.reexport(res_inst.get_port(port_name))
