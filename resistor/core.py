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

from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.template import TemplateBase, TemplateDB

from .base import AnalogResCore
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
        self._core_cls = tech_params['layout']['res_core_template']
        self._edgelr_cls = tech_params['layout']['res_edgelr_template']
        self._edgetb_cls = tech_params['layout']['res_edgetb_template']
        self._corner_cls = tech_params['layout']['res_corner_template']
        self._use_parity = self._core_cls.use_parity()
        self._bot_port = None
        self._top_port = None
        self._core_offset = None
        self._core_pitch = None
        self._num_tracks = None
        self._num_corner_tracks = None
        self._w_tracks = None
        self._hm_layer = self._core_cls.port_layer_id()

    @property
    def num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
        """Returns the number of tracks per resistor block on each routing layer."""
        return self._num_tracks

    @property
    def bot_layer_id(self):
        # type: () -> int
        """Returns the bottom resistor routing layer ID."""
        return self._hm_layer

    @property
    def w_tracks(self):
        # type: () -> Tuple[int, int, int, int]
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
        top_layer = self.bot_layer_id + 3
        blk_w, blk_h = self.grid.get_block_size(top_layer)
        return top_layer, int(round(self._core_pitch[0] / blk_w)), int(round(self._core_pitch[1] / blk_h))

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
        dx = self._core_offset[0] + self._core_pitch[0] * col_idx
        dy = self._core_offset[1] + self._core_pitch[1] * row_idx
        loc = dx, dy
        bot_port = self._bot_port.transform(self.grid, loc=loc)
        top_port = self._top_port.transform(self.grid, loc=loc)
        return bot_port.get_pins()[0], top_port.get_pins()[0]

    def get_track_offsets(self, row_idx, col_idx):
        # type: (int, int) -> Tuple[int, int, int, int]
        """Compute track offsets on each routing layer for the given resistor block.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom row.
        col_idx : int
            the column index.  0 is the left-most column.

        Returns
        -------
        offsets : Tuple[int, int, int, int]
            track index offset on each routing layer.
        """
        res = self.grid.resolution
        dy_unit = int(round((self._core_offset[1] + self._core_pitch[1] * row_idx) / res))
        dx_unit = int(round((self._core_offset[0] + self._core_pitch[0] * col_idx) / res))
        hm_layer = self.bot_layer_id
        hm_pitch = self.grid.get_track_pitch(hm_layer, unit_mode=True)
        vm_pitch = self.grid.get_track_pitch(hm_layer + 1, unit_mode=True)
        xm_pitch = self.grid.get_track_pitch(hm_layer + 2, unit_mode=True)
        ym_pitch = self.grid.get_track_pitch(hm_layer + 3, unit_mode=True)

        return dy_unit // hm_pitch, dx_unit // vm_pitch, dy_unit // xm_pitch, dx_unit // ym_pitch

    def get_h_track_index(self, row_idx, tr_idx):
        # type: (int, Union[float, int]) -> Union[float, int]
        """Compute absolute track index from relative track index.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom row.
        tr_idx : Union[int, float]
            the track index within the given row.

        Returns
        -------
        abs_idx : Union[int, float]
            the absolute track index in this template.
        """
        delta = self._core_offset[1] + self._core_pitch[1] * row_idx
        delta_unit = int(round(delta / self.grid.resolution))
        xm_pitch_unit = self.grid.get_track_pitch(self.bot_layer_id + 2, unit_mode=True)
        return (delta_unit // xm_pitch_unit) + tr_idx

    def move_array(self, nx_blk=0, ny_blk=0):
        """Move the whole array by the given number of block pitches.

        The block pitch is calculated on the top resistor routing layer.

        Note: this method does not update size or array_box.  They must be updated manually.

        Parameters
        ----------
        nx_blk : int
            number of horizontal block pitches.
        ny_blk : int
            number of vertical block pitches.
        """
        blk_w, blk_h = self.grid.get_block_size(self.bot_layer_id + 3)
        dx = nx_blk * blk_w
        dy = ny_blk * blk_h
        self.move_all_by(dx=dx, dy=dy)
        self._core_offset = self._core_offset[0] + dx, self._core_offset[1] + dy

    def draw_array(self,  # type: ResArrayBase
                   l,  # type: float
                   w,  # type: float
                   sub_type,  # type: str
                   threshold,  # type: str
                   nx=1,  # type: int
                   ny=1,  # type: int
                   min_tracks=(1, 1, 1, 1),  # type: Tuple[int, int, int, int]
                   res_type='reference',  # type: str
                   em_specs=None,  # type: Optional[Dict[str, Any]]
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
        min_tracks : Tuple[int, int, int, int]
            minimum number of tracks per layer in the resistor unit cell.
        res_type : str
            the resistor type.
        em_specs : Union[None, Dict[str, Any]]
            resistor EM specifications dictionary.
        """
        # add resistor array layers to RoutingGrid
        self.grid = self.grid.copy()
        grid_layers = self.grid.tech_info.tech_params['layout']['analog_res']['grid_layers']
        for lay_id, tr_w, tr_sp, tr_dir in grid_layers:
            self.grid.add_new_layer(lay_id, tr_sp, tr_w, tr_dir)
        self.grid.update_block_pitch()

        layout_params = dict(
            l=l,
            w=w,
            min_tracks=min_tracks,
            sub_type=sub_type,
            threshold=threshold,
            res_type=res_type,
            parity=0,
            em_specs=em_specs or {},
        )
        # create BL corner
        master = self.new_template(params=layout_params, temp_cls=self._corner_cls)
        bl_corner = self.add_instance(master)
        w_edge_lr = master.array_box.width
        h_edge_tb = master.array_box.height

        # create bottom edge
        master = self._add_blk(self._edgetb_cls, layout_params, (w_edge_lr, 0.0),
                               'R0', nx, 1, 1)
        w_core = master.array_box.width

        # create BR corner
        layout_params['parity'] = (nx + 1) % 2
        master = self.new_template(params=layout_params, temp_cls=self._corner_cls)
        inst = self.add_instance(master, orient='MY')
        inst.move_by(dx=w_edge_lr + nx * w_core - inst.array_box.left)

        # create left edge
        master = self._add_blk(self._edgelr_cls, layout_params, (0.0, h_edge_tb),
                               'R0', 1, ny, 1)
        h_core = master.array_box.height

        # create TL corner
        layout_params['parity'] = (ny + 1) % 2
        master = self.new_template(params=layout_params, temp_cls=self._corner_cls)
        inst = self.add_instance(master, orient='MX')
        inst.move_by(dy=h_edge_tb + ny * h_core - inst.array_box.bottom)

        # create core
        self._core_offset = (w_edge_lr, h_edge_tb)
        self._core_pitch = (w_core, h_core)
        self._add_blk(self._core_cls, layout_params, self._core_offset,
                      'R0', nx, ny, 0)

        # create top edge
        loc = (w_edge_lr, 2 * h_edge_tb + ny * h_core)
        self._add_blk(self._edgetb_cls, layout_params, loc, 'MX', nx, 1, ny % 2)

        # create right edge
        loc = (2 * w_edge_lr + nx * w_core, h_edge_tb)
        self._add_blk(self._edgelr_cls, layout_params, loc, 'MY', 1, ny, nx % 2)

        # create TR corner
        self.array_box = BBox(0.0, 0.0, 2 * w_edge_lr + nx * w_core, 2 * h_edge_tb + ny * h_core,
                              self.grid.resolution)
        layout_params['parity'] = (nx + ny) % 2
        master = self.new_template(params=layout_params, temp_cls=self._corner_cls)
        tr_corner = self.add_instance(master, loc=(self.array_box.right, self.array_box.top),
                                      orient='R180')

        # set array box and size
        self.array_box = bl_corner.array_box.merge(tr_corner.array_box)
        self.set_size_from_array_box(self._hm_layer + 3)

    def _add_blk(self, temp_cls, params, loc, orient, nx, ny, par0):
        params['parity'] = par0
        master0 = self.new_template(params=params, temp_cls=temp_cls)
        if isinstance(master0, AnalogResCore):
            self._bot_port = master0.get_port('bot')
            self._top_port = master0.get_port('top')
            self._num_tracks = master0.get_num_tracks()
            self._num_corner_tracks = master0.get_num_corner_tracks()
            self._w_tracks = master0.get_track_widths()

        spx = master0.array_box.width
        spy = master0.array_box.height
        if not self._use_parity:
            self.add_instance(master0, loc=loc, nx=nx, ny=ny, spx=spx, spy=spy,
                              orient=orient)
        else:
            # add current parity
            nx0 = (nx + 1) // 2
            ny0 = (ny + 1) // 2
            self.add_instance(master0, loc=loc, nx=nx0, ny=ny0, spx=spx * 2, spy=spy * 2,
                              orient=orient)
            nx0 = nx // 2
            ny0 = ny // 2
            if nx0 > 0 and ny0 > 0:
                self.add_instance(master0, loc=(loc[0] + spx, loc[1] + spy),
                                  nx=nx0, ny=ny0, spx=spx * 2, spy=spy * 2, orient=orient)

            # add opposite parity
            params['parity'] = 1 - par0
            master1 = self.new_template(params=params, temp_cls=temp_cls)
            nx1 = nx // 2
            ny1 = (ny + 1) // 2
            if nx1 > 0 and ny1 > 0:
                self.add_instance(master1, loc=(loc[0] + spx, loc[1]),
                                  nx=nx1, ny=ny1, spx=spx * 2, spy=spy * 2, orient=orient)
            nx1 = (nx + 1) // 2
            ny1 = ny // 2
            if nx1 > 0 and ny1 > 0:
                self.add_instance(master1, loc=(loc[0], loc[1] + spy),
                                  nx=nx1, ny=ny1, spx=spx * 2, spy=spy * 2, orient=orient)

        return master0


class Termination(ResArrayBase):
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
        )

    def draw_layout(self):
        # type: () -> None

        # copy routing grid before calling draw_array so substrate contact can have its own grid
        sub_grid = self.grid.copy()

        # draw array
        nx = self.params['nx']
        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')
        ny = self.params['ny']
        em_specs = self.params.pop('em_specs')
        sub_lch = self.params.pop('sub_lch')
        sub_w = self.params.pop('sub_w')
        sub_type = self.params['sub_type']
        div_em_specs = em_specs.copy()
        for key in ('idc', 'iac_rms', 'iac_peak'):
            if key in div_em_specs:
                div_em_specs[key] = div_em_specs[key] / ny
            else:
                div_em_specs[key] = 0.0
        self.draw_array(em_specs=div_em_specs, **self.params)

        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # draw contact and move array up
        nx_arr, ny_arr = self.size[1], self.size[2]
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            top_layer=ym_layer,
            blk_width=nx_arr,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact, grid=sub_grid)
        ny_shift = sub_master.size[2]
        self.move_array(ny_blk=ny_shift)
        bot_inst = self.add_instance(sub_master, inst_name='XBSUB')
        top_yo = (ny_arr + 2 * ny_shift) * self.grid.get_block_pitch(xm_layer)
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(0.0, top_yo), orient='MX')

        # export supplies and recompute array_box/size
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(bot_inst.get_port(port_name))
        self.reexport(top_inst.get_port(port_name))
        self.size = ym_layer, nx_arr, ny_arr + 2 * ny_shift
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)

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
        xc_warr = self.connect_to_tracks(vc_warr, x_tid)
        xl_warr = self.connect_to_tracks(vl_warr, x_tid)
        xr_warr = self.connect_to_tracks(vr_warr, x_tid)

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

        self.add_pin('inp', yl_warr, show=True)
        self.add_pin('inn', yr_warr, show=True)
        self.add_pin('incm', yc_warr, show=True)


class ResLadder(ResArrayBase):
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
            nx=2,
            ny=2,
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
            res_type='the resistor type.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: ResLadder
                            l,  # type: float
                            w,  # type: float
                            sub_lch,  # type: float
                            sub_w,  # type: Union[float, int]
                            sub_type,  # type: str
                            threshold,  # type: str
                            nx,  # type: int
                            ny,  # type: int
                            res_type  # type: str
                            ):
        # type: (...) -> None

        # error checking
        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')
        if ny % 2 != 0 or ny <= 0:
            raise ValueError('number of resistors in a column must be even and positive.')

        # copy routing grid before calling draw_array so substrate contact can have its own grid
        sub_grid = self.grid.copy()
        min_tracks = (4, 7, nx, 1)
        self.draw_array(l, w, sub_type, threshold, nx=nx, ny=ny,
                        min_tracks=min_tracks, res_type=res_type)

        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # draw contact and move array up
        nx_arr, ny_arr = self.size[1], self.size[2]
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=threshold,
            top_layer=ym_layer,
            blk_width=nx_arr,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact, grid=sub_grid)
        ny_shift = sub_master.size[2]

        # shift whole array to the right by 1 block to reserve room for metals on dummy resistors
        _, unit_nxblk, _ = self.res_unit_size
        nx_blk_shift = -(-unit_nxblk // 2)  # ceiling division
        print(unit_nxblk, nx_blk_shift)
        self.move_array(nx_blk=nx_blk_shift, ny_blk=ny_shift)
        dx, _ = self.grid.get_block_size(self.bot_layer_id + 3)
        dx *= nx_blk_shift
        bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(dx, 0))
        top_yo = (ny_arr + 2 * ny_shift) * self.grid.get_block_pitch(xm_layer)
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(dx, top_yo), orient='MX')

        # export supplies and recompute array_box/size
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(bot_inst.get_port(port_name))
        self.reexport(top_inst.get_port(port_name))
        self.size = ym_layer, nx_arr + 2 * nx_blk_shift, ny_arr + 2 * ny_shift
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)

        self._draw_metal_tracks(nx, ny)

    def _draw_metal_tracks(self, nx, ny):
        num_h_tracks, num_v_tracks = self.num_tracks[0:2]

        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        hm_dtr = 1
        if tp_idx + hm_dtr >= num_h_tracks or bp_idx - hm_dtr < 0:
            # use inner hm tracks instead.
            hm_dtr = -1

        # get via extensions
        grid = self.grid
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        hm_ext, vm_ext = grid.get_via_extensions(hm_layer, 1, 1, unit_mode=True)

        vm_tidx = [-0.5, 0.5, 1.5, 2.5, num_v_tracks - 3.5, num_v_tracks - 2.5,
                   num_v_tracks - 1.5, num_v_tracks - 0.5]

        # expand range by +/- 1 to draw metal pattern on dummies too
        for row_idx in range(-1, ny + 1):
            for col_idx in range(-1, nx + 1):
                hm_off, vm_off, _, _ = self.get_track_offsets(row_idx, col_idx)

                # extend port tracks on hm layer
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[1], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-2], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bp_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=tp_idx - bp_idx, unit_mode=True)

                # draw hm layer bridge
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[0], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[3], unit_mode=True)
                pitch = tp_idx - bp_idx + 2 * hm_dtr
                self.add_wires(hm_layer, hm_off + bp_idx - hm_dtr, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-4], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-1], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bp_idx - hm_dtr, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)
