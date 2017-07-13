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


"""This module defines ResArrayBase, an abstract class that draws resistor arrays.

This module also define some simple subclasses of ResArrayBase.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
from typing import Dict, Set, Tuple, Union, Any, Optional
from itertools import chain

from bag.layout.routing import TrackID, WireArray, Port
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.core import TechInfo

from .base import ResTech, AnalogResCore, AnalogResBoundary
from ..analog_core import SubstrateContact


# noinspection PyAbstractClass
class ResArrayBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template that draws analog resistors array and connections.

    This template assumes that the ressistor array uses 4 routing layers, with
    directions x/y/x/y.  The lower 2 routing layers is used to connect between
    adjacent resistors, and pin will be drawn on the upper 2 routing layers.

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
        super(ResArrayBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        tech_params = self.grid.tech_info.tech_params
        self._tech_cls = tech_params['layout']['res_tech_class']  # type: ResTech
        self._bot_port = None  # type: Port
        self._top_port = None  # type: Port
        self._core_offset = None  # type: Tuple[int, int]
        self._core_pitch = None  # type: Tuple[int, int]
        self._num_tracks = None  # type: Tuple[int, ...]
        self._num_corner_tracks = None  # type: Tuple[int, ...]
        self._w_tracks = None  # type: Tuple[int, ...]
        self._hm_layer = self._tech_cls.get_bot_layer()
        self._well_width = None

    @classmethod
    def get_port_layer_id(cls, tech_info):
        # type: (TechInfo) -> int
        return tech_info.tech_params['layout']['res_tech_class'].get_bot_layer()

    @property
    def num_tracks(self):
        # type: () -> Tuple[int, ...]
        """Returns the number of tracks per resistor block on each routing layer."""
        return self._num_tracks

    @property
    def bot_layer_id(self):
        # type: () -> int
        """Returns the bottom resistor routing layer ID."""
        return self._hm_layer

    @property
    def w_tracks(self):
        # type: () -> Tuple[int, ...]
        """Returns the track width on each routing layer, in number of tracks."""
        return self._w_tracks

    @property
    def bot_port_idx(self):
        # type: () -> int
        """Returns the relative track index of the bottom resistor port."""
        return self._bot_port.get_pins()[0].track_id.base_index

    @property
    def top_port_idx(self):
        # type: () -> int
        """Returns the relative track index of the top resistor port."""
        return self._top_port.get_pins()[0].track_id.base_index

    @property
    def res_unit_size(self):
        # type: () -> Tuple[int, int, int]
        """Returns the size of a unit resistor block."""
        top_layer = self.bot_layer_id + len(self._num_tracks) - 1
        return self.grid.get_size_tuple(top_layer, self._core_pitch[0], self._core_pitch[1], unit_mode=True)

    def get_well_width(self, unit_mode=False):
        # type: (bool) -> Union[float, int]
        """Returns the NW/PW width in this block."""
        if unit_mode:
            return self._well_width
        return self._well_width * self.grid.resolution

    def get_res_ports(self, row_idx, col_idx):
        # type: (int, int) -> Tuple[WireArray, WireArray]
        """Returns the port of the given resistor.

        Parameters
        ----------
        row_idx : int
            the resistor row index.  0 is the bottom row.
        col_idx : int
            the resistor column index.  0 is the left-most column.

        Returns
        -------
        bot_warr : WireArray
            the bottom port as WireArray.
        top_warr : WireArray
            the top port as WireArray.
        """
        res = self.grid.resolution
        dx = self._core_offset[0] + self._core_pitch[0] * col_idx
        dy = self._core_offset[1] + self._core_pitch[1] * row_idx
        loc = dx * res, dy * res
        bot_port = self._bot_port.transform(self.grid, loc=loc)
        top_port = self._top_port.transform(self.grid, loc=loc)
        return bot_port.get_pins()[0], top_port.get_pins()[0]

    def get_track_offsets(self, row_idx, col_idx):
        # type: (int, int) -> Tuple[float, ...]
        """Compute track offsets on each routing layer for the given resistor block.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom row.
        col_idx : int
            the column index.  0 is the left-most column.

        Returns
        -------
        offsets : Tuple[float, ...]
            track index offset on each routing layer.
        """
        dx = self._core_offset[0] + self._core_pitch[0] * col_idx
        dy = self._core_offset[1] + self._core_pitch[1] * row_idx
        bot_layer = self.bot_layer_id
        offsets = []
        for lay in range(bot_layer, bot_layer + len(self._num_tracks)):
            dim = dx if self.grid.get_direction(lay) == 'y' else dy
            pitch = self.grid.get_track_pitch(lay, unit_mode=True)
            offsets.append(dim / pitch)

        return tuple(offsets)

    def get_h_track_index(self, row_idx, tr_idx):
        # type: (int, Union[float, int]) -> float
        """Compute absolute track index from relative track index.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom row.
        tr_idx : Union[int, float]
            the track index within the given row.

        Returns
        -------
        abs_idx : float
            the absolute track index in this template.
        """
        delta = self._core_offset[1] + self._core_pitch[1] * row_idx
        targ_layer = self.bot_layer_id + len(self._num_tracks) - 1
        if self.grid.get_direction(targ_layer) == 'y':
            targ_layer -= 1

        return delta / self.grid.get_track_pitch(targ_layer, unit_mode=True) + tr_idx

    def draw_array(self,  # type: ResArrayBase
                   l,  # type: float
                   w,  # type: float
                   sub_type,  # type: str
                   threshold,  # type: str
                   nx=1,  # type: int
                   ny=1,  # type: int
                   min_tracks=(1, 1, 1, 1),  # type: Tuple[int, ...]
                   res_type='reference',  # type: str
                   em_specs=None,  # type: Optional[Dict[str, Any]]
                   grid_type='standard',  # type: str
                   ext_dir='',  # type: str
                   **kwargs
                   ):
        # type: (...) -> None
        """Draws the resistor array.

        This method updates the RoutingGrid with resistor routing layers, then add
        resistor instances to this template.

        Parameters
        ----------
        l : float
            unit resistor length, in meters.
        w : float
            unit resistor width, in meters.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold flavor.
        nx : int
            number of resistors in a row.
        ny : int
            number of resistors in a column.
        min_tracks : Tuple[int, ...]
            minimum number of tracks per layer in the resistor unit cell.
        res_type : str
            the resistor type.
        em_specs : Union[None, Dict[str, Any]]
            resistor EM specifications dictionary.
        grid_type : str
            the lower resistor routing grid name.
        ext_dir : str
            resistor core extension direction.
        """
        if em_specs is None:
            em_specs = {}
        # modify resistor layer routing grid.
        grid_layers = self.grid.tech_info.tech_params['layout']['analog_res'][grid_type]
        for lay_id, tr_w, tr_sp, tr_dir, necessary in grid_layers:
            if necessary or lay_id not in self.grid:
                self.grid.add_new_layer(lay_id, tr_sp, tr_w, tr_dir, override=True)

        self.grid.update_block_pitch()

        # find location of the lower-left resistor core
        res = self.grid.resolution
        lay_unit = self.grid.layout_unit
        l_unit = int(round(l / lay_unit / res))
        w_unit = int(round(w / lay_unit / res))
        res_info = self._tech_cls.get_res_info(self.grid, l_unit, w_unit, res_type, sub_type, threshold,
                                               min_tracks, em_specs, ext_dir)
        w_edge, h_edge = res_info['w_edge'], res_info['h_edge']
        w_core, h_core = res_info['w_core'], res_info['h_core']
        self._core_offset = w_edge, h_edge
        self._core_pitch = w_core, h_core
        self._num_tracks = tuple(res_info['num_tracks'])
        self._num_corner_tracks = tuple(res_info['num_corner_tracks'])
        self._w_tracks = tuple(res_info['track_widths'])

        # make template masters
        core_params = dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
            min_tracks=min_tracks,
            em_specs=em_specs,
            ext_dir=ext_dir,
        )
        core_master = self.new_template(params=core_params, temp_cls=AnalogResCore)
        lr_params = core_master.get_boundary_params('lr')
        lr_master = self.new_template(params=lr_params, temp_cls=AnalogResBoundary)
        top_params = core_master.get_boundary_params('tb')
        top_master = self.new_template(params=top_params, temp_cls=AnalogResBoundary)
        bot_params = core_master.get_boundary_params('tb')
        bot_master = self.new_template(params=bot_params, temp_cls=AnalogResBoundary)
        tcorner_params = core_master.get_boundary_params('corner')
        tcorner_master = self.new_template(params=tcorner_params, temp_cls=AnalogResBoundary)
        bcorner_params = core_master.get_boundary_params('corner')
        bcorner_master = self.new_template(params=bcorner_params, temp_cls=AnalogResBoundary)

        # place core
        for row in range(ny):
            for col in range(nx):
                cur_name = 'XCORE%d' % (col + nx * row)
                cur_loc = (w_edge + col * w_core, h_edge + row * h_core)
                cur_master = self.new_template(params=core_params, temp_cls=AnalogResCore)
                self.add_instance(cur_master, inst_name=cur_name, loc=cur_loc, unit_mode=True)
                if row == 0 and col == 0:
                    self._bot_port = cur_master.get_port('bot')
                    self._top_port = cur_master.get_port('top')

        # place boundaries
        # bottom-left corner
        inst_bl = self.add_instance(bcorner_master, inst_name='XBL')
        # bottom edge
        self.add_instance(bot_master, inst_name='XB', loc=(w_edge, 0), nx=nx, spx=w_core, unit_mode=True)
        # bottom-right corner
        loc = (2 * w_edge + nx * w_core, 0)
        self.add_instance(bcorner_master, inst_name='XBR', loc=loc, orient='MY', unit_mode=True)
        # left edge
        loc = (0, h_edge)
        well_xl = lr_master.get_well_left(unit_mode=True)
        self.add_instance(lr_master, inst_name='XL', loc=loc, ny=ny, spy=h_core, unit_mode=True)
        # right edge
        loc = (2 * w_edge + nx * w_core, h_edge)
        well_xr = loc[0] - well_xl
        self._well_width = well_xr - well_xl
        self.add_instance(lr_master, inst_name='XR', loc=loc, orient='MY', ny=ny, spy=h_core, unit_mode=True)
        # top-left corner
        loc = (0, 2 * h_edge + ny * h_core)
        self.add_instance(tcorner_master, inst_name='XTL', loc=loc, orient='MX', unit_mode=True)
        # top edge
        loc = (w_edge, 2 * h_edge + ny * h_core)
        self.add_instance(top_master, inst_name='XT', loc=loc, orient='MX', nx=nx, spx=w_core, unit_mode=True)
        # top-right corner
        loc = (2 * w_edge + nx * w_core, 2 * h_edge + ny * h_core)
        inst_tr = self.add_instance(tcorner_master, inst_name='XTR', loc=loc, orient='R180', unit_mode=True)

        # set array box and size
        self.array_box = inst_bl.array_box.merge(inst_tr.array_box)
        top_layer = self._hm_layer + len(min_tracks) - 1
        self.set_size_from_array_box(top_layer)
        self.add_cell_boundary(self.bound_box)

        # draw device blockages
        self.grid.tech_info.draw_device_blockage(self)


class TerminationCore(ResArrayBase):
    """An template for creating termination resistors.

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
        super(TerminationCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=1,
            res_type='reference',
            em_specs={},
            ext_dir='',
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
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
        )

    def draw_layout(self):
        # type: () -> None

        # draw array
        nx = self.params['nx']
        ny = self.params['ny']
        em_specs = self.params.pop('em_specs')

        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')

        div_em_specs = em_specs.copy()
        for key in ('idc', 'iac_rms', 'iac_peak'):
            if key in div_em_specs:
                div_em_specs[key] = div_em_specs[key] / ny
            else:
                div_em_specs[key] = 0.0

        min_tracks = (1, 1, 1, 1)
        self.draw_array(min_tracks=min_tracks, em_specs=div_em_specs, grid_type='low_res', **self.params)

        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # connect row resistors
        hc_warr_list = []
        hl_warr_list = []
        hr_warr_list = []
        for row_idx in range(ny):
            for col_idx in range(nx - 1):
                ports_l = self.get_res_ports(row_idx, col_idx)
                ports_r = self.get_res_ports(row_idx, col_idx + 1)
                con_par = (col_idx + row_idx) % 2
                mid_wire = self.connect_wires([ports_l[con_par], ports_r[con_par]])
                if col_idx == 0:
                    hl_warr_list.append(ports_l[1 - con_par])
                if col_idx == nx - 2:
                    hr_warr_list.append(ports_r[1 - con_par])
                if col_idx == (nx // 2) - 1:
                    hc_warr_list.append(mid_wire[0])

        # connect to v layer
        vm_width = self.w_tracks[1]
        v_pitch = self.num_tracks[1]
        vc_id = self.grid.coord_to_nearest_track(vm_layer, hc_warr_list[0].middle, half_track=True)
        v_tid = TrackID(vm_layer, vc_id, width=vm_width)
        vc_warr = self.connect_to_tracks(hc_warr_list, v_tid)
        v_tid = TrackID(vm_layer, vc_id - v_pitch, width=vm_width)
        vl_warr = self.connect_to_tracks(hl_warr_list, v_tid)
        v_tid = TrackID(vm_layer, vc_id + v_pitch, width=vm_width)
        vr_warr = self.connect_to_tracks(hr_warr_list, v_tid)

        # connect to x layer
        xm_width = self.w_tracks[2]
        x_pitch = self.num_tracks[2]
        x_base = self.get_h_track_index(0, (x_pitch - 1) / 2.0)
        x_tid = TrackID(xm_layer, x_base, width=xm_width, num=ny, pitch=x_pitch)
        xc_warr = self.connect_to_tracks(vc_warr, x_tid, min_len_mode=0)
        xl_warr = self.connect_to_tracks(vl_warr, x_tid, min_len_mode=0)
        xr_warr = self.connect_to_tracks(vr_warr, x_tid, min_len_mode=0)

        # connect to y layer
        bot_w = self.grid.get_track_width(xm_layer, xm_width)
        ym_width = self.grid.get_min_track_width(ym_layer, bot_w=bot_w, **em_specs)
        y_pitch = self.num_tracks[3]
        yc_id = self.grid.coord_to_nearest_track(ym_layer, xc_warr.middle, half_track=True)
        y_tid = TrackID(ym_layer, yc_id, width=ym_width)
        yc_warr = self.connect_to_tracks(xc_warr, y_tid)
        y_tid = TrackID(ym_layer, yc_id - y_pitch, width=ym_width)
        yl_warr = self.connect_to_tracks(xl_warr, y_tid)
        y_tid = TrackID(ym_layer, yc_id + y_pitch, width=ym_width)
        yr_warr = self.connect_to_tracks(xr_warr, y_tid)

        self.add_pin('inp', yl_warr, show=False)
        self.add_pin('inn', yr_warr, show=False)
        self.add_pin('incm', yc_warr, show=False)


class Termination(TemplateBase):
    """An template for creating termination resistors.

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
        super(Termination, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=1,
            ext_dir='',
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
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']

        res_master = self.new_template(params=res_params, temp_cls=TerminationCore)

        # draw contact and move array up
        top_layer, nx_arr, ny_arr = res_master.size
        w_pitch, h_pitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            well_width=res_master.get_well_width(),
            show_pins=False,
            is_passive=True,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        sub_box = sub_master.bound_box
        ny_shift = -(-sub_box.height_unit // h_pitch)

        # compute substrate X coordinate so substrate is on its own private horizontal pitch
        sub_x_pitch, _ = sub_master.grid.get_size_pitch(sub_master.size[0], unit_mode=True)
        sub_x = ((w_pitch * nx_arr - sub_box.width_unit) // 2 // sub_x_pitch) * sub_x_pitch

        bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(sub_x, 0), unit_mode=True)
        res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_shift * h_pitch), unit_mode=True)
        top_yo = (ny_arr + 2 * ny_shift) * h_pitch
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(sub_x, top_yo), orient='MX', unit_mode=True)

        # connect implant layers of resistor array and substrate contact together
        for lay in self.grid.tech_info.get_implant_layers(sub_type):
            self.add_rect(lay, self.get_rect_bbox(lay))

        # export supplies and recompute array_box/size
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(bot_inst.get_port(port_name))
        self.reexport(top_inst.get_port(port_name))
        self.size = top_layer, nx_arr, ny_arr + 2 * ny_shift
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.add_cell_boundary(self.bound_box)

        for port_name in res_inst.port_names_iter():
            self.reexport(res_inst.get_port(port_name))


class ResLadderCore(ResArrayBase):
    """An template for creating a resistor ladder from VDD to VSS.

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
        super(ResLadderCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=2,
            ndum=1,
            res_type='reference',
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
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            ndum='number of dummy resistors.',
            res_type='the resistor type.',
        )

    def draw_layout(self):
        # type: () -> None
        l = self.params['l']
        w = self.params['w']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        nx = self.params['nx']
        ny = self.params['ny']
        ndum = self.params['ndum']
        res_type = self.params['res_type']

        hcon_space = 0
        vcon_space = 0

        # error checking
        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')
        if ny % 2 != 0 or ny <= 0:
            raise ValueError('number of resistors in a column must be even and positive.')

        # copy routing grid before calling draw_array so substrate contact can have its own grid
        min_tracks = (4 + 2 * hcon_space, 7 + vcon_space, nx, 1)
        self.draw_array(l, w, sub_type, threshold, nx=nx + 2 * ndum, ny=ny + 2 * ndum,
                        min_tracks=min_tracks, res_type=res_type, grid_type='low_res', ext_dir='x')

        # export supplies and recompute array_box/size
        hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup = self._draw_metal_tracks(nx, ny, ndum, hcon_space)

        self._connect_ladder(nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)

    def _connect_ladder(self, nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        # connect main ladder
        for row_idx in range(ndum, ny + ndum):
            rmod = row_idx - ndum
            for col_idx in range(ndum, nx + ndum):
                if (col_idx == ndum and rmod % 2 == 1) or (col_idx == nx - 1 + ndum and rmod % 2 == 0):
                    mode = 1 if row_idx == ny + ndum - 1 else 0
                    self._connect_tb(row_idx, col_idx, ndum, tp_idx, hcon_idx_list,
                                     vcon_idx_list, xm_bot_idx, mode=mode)
                if col_idx != nx - 1 + ndum:
                    self._connect_lr(row_idx, col_idx, nx, ndum, tp_idx, bp_idx, hcon_idx_list,
                                     vcon_idx_list, xm_bot_idx)

        # connect to ground
        self._connect_tb(ndum - 1, ndum, ndum, tp_idx, hcon_idx_list,
                         vcon_idx_list, xm_bot_idx, mode=-1)
        # connect to supplies
        self._connect_ground(nx, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)
        self._connect_power(ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)

        # connect horizontal dummies
        for row_idx in range(ny + 2 * ndum):
            if row_idx < ndum or row_idx >= ny + ndum:
                col_iter = range(nx + 2 * ndum)
            else:
                col_iter = chain(range(ndum), range(nx + ndum, nx + 2 * ndum))
            for col_idx in col_iter:
                conn_tb = col_idx < ndum or col_idx >= nx + ndum
                self._connect_dummy(row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list, vcon_idx_list)

    def _connect_power(self, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(ny + ndum, ndum)
        _, vm_prev, _, _ = self.get_track_offsets(ndum, ndum - 1)
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        hconn = hcon_idx_list[0]
        vm_idx_list = [vm_off + vcon_idx_list[2], vm_prev + vcon_idx_list[-3],
                       vm_prev + vcon_idx_list[-2]]
        xm_idx_list = [xm_off + xm_bot_idx + idx for idx in range(num_xm_sup)]
        for vm_idx in vm_idx_list:
            # connect supply to vm layer
            self.add_via_on_grid(hm_layer, hm_off + hconn, vm_idx)
            # connect supply to xm layer
            for xm_idx in xm_idx_list:
                self.add_via_on_grid(vm_layer, vm_idx, xm_idx)

    def _connect_ground(self, nx, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        xm_prev = self.get_track_offsets(ndum - 1, ndum)[2]
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(ndum, ndum)
        _, vm_prev, _, _ = self.get_track_offsets(ndum, ndum - 1)
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        hconn = hcon_idx_list[0]

        # connect all dummies to ground
        self.add_via_on_grid(hm_layer, hm_off + hconn, vm_prev + vcon_idx_list[-4])

        vm_idx_list = [vm_off + vcon_idx_list[1], vm_off + vcon_idx_list[2],
                       vm_prev + vcon_idx_list[-3], vm_prev + vcon_idx_list[-2]]
        xm_idx_list = [xm_prev + xm_bot_idx + idx for idx in range(nx - num_xm_sup, nx)]
        xm_idx_list.append(xm_off + xm_bot_idx)
        for vm_idx in vm_idx_list:
            # connect supply to vm layer
            self.add_via_on_grid(hm_layer, hm_off + hconn, vm_idx)
            # connect supply to xm layer
            for xm_idx in xm_idx_list:
                self.add_via_on_grid(vm_layer, vm_idx, xm_idx)

    def _connect_dummy(self, row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list, vcon_idx_list):
        hm_off, vm_off, _, _ = self.get_track_offsets(row_idx, col_idx)
        hm_layer = self.bot_layer_id
        self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + hcon_idx_list[1], vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + hcon_idx_list[1], vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[-4])
        if conn_tb:
            self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[1])
            self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[1])

    def _connect_lr(self, row_idx, col_idx, nx, ndum, tp_idx, bp_idx, hcon_idx_list,
                    vcon_idx_list, xm_bot_idx):
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(row_idx, col_idx)
        _, vm_next, _, _ = self.get_track_offsets(row_idx, col_idx + 1)
        hm_layer = self.bot_layer_id
        col_real = col_idx - ndum
        row_real = row_idx - ndum
        if col_real % 2 == 0:
            port = bp_idx
            conn = hcon_idx_list[1]
        else:
            port = tp_idx
            conn = hcon_idx_list[0]
        self.add_via_on_grid(hm_layer, hm_off + port, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_off + vcon_idx_list[-1])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_next + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + port, vm_next + vcon_idx_list[3])

        # connect to output port
        vm_layer = hm_layer + 1
        if row_real % 2 == 0:
            xm_idx = xm_bot_idx + col_real + 1
        else:
            xm_idx = xm_bot_idx + (nx - 1 - col_real)
        self.add_via_on_grid(vm_layer, vm_off + vcon_idx_list[-1], xm_off + xm_idx)

    def _connect_tb(self, row_idx, col_idx, ndum, tp_idx, hcon_idx_list,
                    vcon_idx_list, xm_bot_idx, mode=0):
        # mode = 0 is normal connection, mode = 1 is vdd connection, mode = -1 is vss connection
        hm_off, vm_off, _, _ = self.get_track_offsets(row_idx, col_idx)
        hm_next, _, xm_next, _ = self.get_track_offsets(row_idx + 1, col_idx)
        hm_layer = self.bot_layer_id
        if col_idx == ndum:
            conn1 = vcon_idx_list[1]
            tap = vcon_idx_list[2]
            conn2 = vcon_idx_list[3]
        else:
            conn1 = vcon_idx_list[-2]
            tap = vcon_idx_list[-3]
            conn2 = vcon_idx_list[-4]
        if mode >= 0:
            self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + conn1)
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + conn1)
        if mode == 0:
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + tap)

            # connect to output port
            vm_layer = hm_layer + 1
            self.add_via_on_grid(vm_layer, vm_off + tap, xm_next + xm_bot_idx)
        if mode <= 0:
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + conn2)
            self.add_via_on_grid(hm_layer, hm_next + tp_idx, vm_off + conn2)

    def _draw_metal_tracks(self, nx, ny, ndum, hcon_space):
        num_h_tracks, num_v_tracks, num_x_tracks = self.num_tracks[0:3]
        xm_bot_idx = (num_x_tracks - nx) / 2

        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        hm_dtr = hcon_space + 1
        if tp_idx + hm_dtr >= num_h_tracks or bp_idx - hm_dtr < 0:
            # use inner hm tracks instead.
            hm_dtr *= -1
        bcon_idx = bp_idx - hm_dtr
        tcon_idx = tp_idx + hm_dtr

        # get via extensions
        grid = self.grid
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        hm_ext, vm_ext = grid.get_via_extensions(hm_layer, 1, 1, unit_mode=True)
        vmx_ext, _ = grid.get_via_extensions(vm_layer, 1, 1, unit_mode=True)

        vm_tidx = [-0.5, 0.5, 1.5, 2.5, num_v_tracks - 3.5, num_v_tracks - 2.5,
                   num_v_tracks - 1.5, num_v_tracks - 0.5]

        # get unit block size
        unit_size = self.res_unit_size
        blk_w, blk_h = self.grid.get_size_dimension(unit_size, unit_mode=True)

        # find top X layer track index that can be connected to supply.
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(0, 0)
        vm_y1 = max(grid.get_wire_bounds(hm_layer, hm_off + max(bp_idx, bcon_idx),
                                         unit_mode=True)[1] + vm_ext,
                    grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                         unit_mode=True)[1] + vmx_ext)
        xm_vdd_top_idx = grid.find_next_track(xm_layer, vm_y1 - vmx_ext, half_track=True,
                                              mode=-1, unit_mode=True)
        num_xm_sup = int(xm_vdd_top_idx - xm_bot_idx - xm_off + 1)

        # get lower/upper bounds of output ports.
        xm_lower = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[0], unit_mode=True)[0]
        vm_off = self.get_track_offsets(0, nx + 2 * ndum - 1)[1]
        xm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-1], unit_mode=True)[1]

        # expand range by +/- 1 to draw metal pattern on dummies too
        for row_idx in range(ny + 2 * ndum):
            for col_idx in range(nx + 2 * ndum):
                hm_off, vm_off, xm_off, _ = self.get_track_offsets(row_idx, col_idx)

                # extend port tracks on hm layer
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[1], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-2], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bp_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=tp_idx - bp_idx, unit_mode=True)

                # draw hm layer bridge
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[0], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[3], unit_mode=True)
                pitch = tcon_idx - bcon_idx
                self.add_wires(hm_layer, hm_off + bcon_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-4], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-1], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bcon_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)

                # draw vm layer bridges
                vm_lower = min(grid.get_wire_bounds(hm_layer, hm_off + min(bp_idx, bcon_idx),
                                                    unit_mode=True)[0] - vm_ext,
                               grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                                    unit_mode=True)[0] - vmx_ext)
                vm_upper = max(grid.get_wire_bounds(hm_layer, hm_off + max(tp_idx, tcon_idx),
                                                    unit_mode=True)[1] + vm_ext,
                               grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx + nx - 1,
                                                    unit_mode=True)[1] + vmx_ext)
                self.add_wires(vm_layer, vm_off + vm_tidx[0], vm_lower, vm_upper,
                               num=2, pitch=3, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-4], vm_lower, vm_upper,
                               num=2, pitch=3, unit_mode=True)

                vm_y1 = max(grid.get_wire_bounds(hm_layer, hm_off + max(bp_idx, bcon_idx),
                                                 unit_mode=True)[1] + vm_ext,
                            grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                                 unit_mode=True)[1] + vmx_ext)
                vm_y2 = min(grid.get_wire_bounds(hm_layer, hm_off + min(tp_idx, tcon_idx),
                                                 unit_mode=True)[0] - vm_ext,
                            grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx + nx - 1,
                                                 unit_mode=True)[0] - vmx_ext)
                self.add_wires(vm_layer, vm_off + vm_tidx[1], vm_y2 - blk_h, vm_y1,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[1], vm_y2, vm_y1 + blk_h,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-3], vm_y2 - blk_h, vm_y1,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-3], vm_y2, vm_y1 + blk_h,
                               num=2, pitch=1, unit_mode=True)

        # draw and export output ports
        for row in range(ny + 2 * ndum):
            tr_off = self.get_track_offsets(row, 0)[2]
            for tidx in range(nx):
                warr = self.add_wires(xm_layer, tr_off + xm_bot_idx + tidx, lower=xm_lower, upper=xm_upper,
                                      fill_type='VSS', unit_mode=True)
                if row < ndum or (row == ndum and tidx == 0):
                    net_name = 'VSS'
                elif row >= ny + ndum:
                    net_name = 'VDD'
                else:
                    net_name = 'out<%d>' % (tidx + (row - ndum) * nx)
                self.add_pin(net_name, warr, show=False)

        return [bcon_idx, tcon_idx], vm_tidx, xm_bot_idx, num_xm_sup


class ResLadder(TemplateBase):
    """An template for creating termination resistors.

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
        super(ResLadder, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            ndum=1,
            res_type='reference',
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
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            ndum='number of dummy resistors.',
            res_type='the resistor type.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']

        res_master = self.new_template(params=res_params, temp_cls=ResLadderCore)

        # draw contact and move array up
        top_layer, nx_arr, ny_arr = res_master.size
        w_pitch, h_pitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            well_width=res_master.get_well_width(),
            show_pins=False,
            is_passive=True,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        sub_box = sub_master.bound_box
        ny_shift = -(-sub_box.height_unit // h_pitch)

        # compute substrate X coordinate so substrate is on its own private horizontal pitch
        sub_x_pitch, _ = sub_master.grid.get_size_pitch(sub_master.size[0], unit_mode=True)
        sub_x = ((w_pitch * nx_arr - sub_box.width_unit) // 2 // sub_x_pitch) * sub_x_pitch

        bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(sub_x, 0), unit_mode=True)
        res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_shift * h_pitch), unit_mode=True)
        top_yo = (ny_arr + 2 * ny_shift) * h_pitch
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(sub_x, top_yo), orient='MX', unit_mode=True)

        # connect implant layers of resistor array and substrate contact together
        for lay in self.grid.tech_info.get_implant_layers(sub_type):
            self.add_rect(lay, self.get_rect_bbox(lay))

        # recompute array_box/size
        self.size = top_layer, nx_arr, ny_arr + 2 * ny_shift
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.add_cell_boundary(self.bound_box)

        # gather supply and re-export outputs
        sup_table = {'VDD': [], 'VSS': []}
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sup_table[port_name].extend(bot_inst.get_all_port_pins(port_name))
        sup_table[port_name].extend(top_inst.get_all_port_pins(port_name))

        for port_name in res_inst.port_names_iter():
            if port_name in sup_table:
                sup_table[port_name].extend(res_inst.get_all_port_pins(port_name))
            else:
                self.reexport(res_inst.get_port(port_name), show=False)

        vdd_warrs, vss_warrs = sup_table['VDD'], sup_table['VSS']
        sup_layer = vdd_warrs[0].layer_id + 1
        # get power fill width and spacing
        sup_width = 1
        sup_spacing = self.grid.get_num_space_tracks(sup_layer, sup_width)
        num_sup_tracks = res_master.num_tracks[-1]
        # make sure every resistor sees the same power fill
        if sup_width + sup_spacing > num_sup_tracks:
            raise ValueError('Cannot draw power fill with width = %d' % sup_width)
        while num_sup_tracks % (sup_width + sup_spacing) != 0:
            sup_spacing += 1

        vdd_warrs, vss_warrs = self.do_power_fill(sup_layer, vdd_warrs, vss_warrs, sup_width=sup_width,
                                                  fill_margin=0.5, edge_margin=0.2, sup_spacing=sup_spacing)
        self.add_pin('VDD', vdd_warrs, show=False)
        self.add_pin('VSS', vss_warrs, show=False)
