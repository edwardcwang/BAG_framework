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


"""This module defines abstract analog mosfet template classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc

from typing import Dict, Set, Tuple, Union

from bag import float_to_si_string
from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.template import MicroTemplate


class AnalogResCore(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template for analog resistors array core.

    Parameters
    ----------
    temp_db : :class:`~bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogResCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def use_parity(cls):
        """Returns True if parity changes resistor core layout."""
        return False

    @classmethod
    @abc.abstractmethod
    def port_layer_id(cls):
        """Returns the resistor port layer ID.

        Bottom port layer must be horizontal.
        """
        return -1

    @classmethod
    def get_default_param_values(cls):
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
            parity=0,
            sub_type='ntap',
            em_specs={},
            min_tracks=(1, 1, 1, 1),
        )

    @classmethod
    def get_params_info(cls):
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            res_type='the resistor type.',
            em_specs='resistor EM spec specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        """Returns a list of the number of tracks on each routing layer in this template.

        Note: this method must work before draw_layout() is called.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return 1, 1, 1, 1

    @abc.abstractmethod
    def get_num_corner_tracks(self):
        """Returns a list of number of tracks on each routing layer in corner templates.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in corner templates on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return 1, 1, 1, 1

    @abc.abstractmethod
    def get_track_widths(self):
        """Returns a list of track widths on each routing layer.

        Returns
        -------
        width_list : Tuple[int, int, int, int]
            a list of track widths in number of tracks on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            port_layer_id().
        """
        return 1, 1, 1, 1

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        ntrx, ntry = self.get_num_tracks()[-2:]
        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        main = 'rescore_%s_%s_l%s_w%s_xtr%d_ytr%d' % (self.params['res_type'],
                                                      self.params['sub_type'],
                                                      l_str, w_str, ntrx, ntry)
        if self.use_parity():
            main += '_par%d' % self.params['parity']

        return main


# noinspection PyAbstractClass
class AnalogResLREdge(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template for analog resistors array left/right edge.

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
        super(AnalogResLREdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
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
            min_tracks=(1, 1, 1, 1),
            parity=0,
            sub_type='ntap',
            res_type='reference',
            em_specs={},
        )

    @classmethod
    def get_params_info(cls):
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            res_type='the resistor type.',
            em_specs='resistor EM spec specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        """Returns a list of the number of tracks on each routing layer in this template.

        Note: this method must work before draw_layout() is called.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return 1, 1, 1, 1

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        tech_params = self.grid.tech_info.tech_params
        res_cls = tech_params['layout']['res_core_template']

        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        ntrx, ntry = self.get_num_tracks()[-2:]
        main = 'resedgelr_%s_%s_l%s_w%s_xtr%d_ytr%d' % (self.params['res_type'],
                                                        self.params['sub_type'],
                                                        l_str, w_str, ntrx, ntry)
        if res_cls.use_parity():
            main += '_par%d' % self.params['parity']

        return main


# noinspection PyAbstractClass
class AnalogResTBEdge(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template for analog resistors array left/right edge.

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
        super(AnalogResTBEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
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
            min_tracks=(1, 1, 1, 1),
            parity=0,
            sub_type='ntap',
            res_type='reference',
            em_specs={},
        )

    @classmethod
    def get_params_info(cls):
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            res_type='the resistor type.',
            em_specs='resistor EM specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        """Returns a list of the number of tracks on each routing layer in this template.

        Note: this method must work before draw_layout() is called.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return 1, 1, 1, 1

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        tech_params = self.grid.tech_info.tech_params
        res_cls = tech_params['layout']['res_core_template']

        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        ntrx, ntry = self.get_num_tracks()[-2:]
        main = 'resedgetb_%s_%s_l%s_w%s_xtr%d_ytr%d' % (self.params['res_type'],
                                                        self.params['sub_type'],
                                                        l_str, w_str, ntrx, ntry)
        if res_cls.use_parity():
            main += '_par%d' % self.params['parity']

        return main


# noinspection PyAbstractClass
class AnalogResCorner(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template for analog resistors array left/right edge.

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
        super(AnalogResCorner, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
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
            min_tracks=(1, 1, 1, 1),
            parity=0,
            sub_type='ntap',
            res_type='reference',
            em_specs={},
        )

    @classmethod
    def get_params_info(cls):
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            res_type='the resistor type.',
            em_specs='resistor EM specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        """Returns a list of the number of tracks on each routing layer in this template.

        Note: this method must work before draw_layout() is called.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return 1, 1, 1, 1

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        tech_params = self.grid.tech_info.tech_params
        res_cls = tech_params['layout']['res_core_template']

        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        ntrx, ntry = self.get_num_tracks()[-2:]
        main = 'rescorner_%s_%s_l%s_w%s_xtr%d_ytr%d' % (self.params['res_type'],
                                                        self.params['sub_type'],
                                                        l_str, w_str, ntrx, ntry)
        if res_cls.use_parity():
            main += '_par%d' % self.params['parity']

        return main


# noinspection PyAbstractClass
class ResArrayBase(with_metaclass(abc.ABCMeta, MicroTemplate)):
    """An abstract template that draws analog resistors array and connections.

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
        """Returns the number of tracks on each resistor routing layer."""
        return self._num_tracks

    @property
    def bot_layer_id(self):
        """Returns the bottom resistor routing layer ID."""
        return self._hm_layer

    @property
    def w_tracks(self):
        """Returns the track width on each resistor routing layer, in number of tracks."""
        return self._w_tracks

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

    def draw_array(self, l, w, nx=1, ny=1, min_tracks=(1, 1, 1, 1), sub_type='ntap',
                   res_type='reference', em_specs=None):
        """Draws the resistor array.

        Parameters
        ----------
        l : float
            unit resistor length, in meters.
        w : float
            unit resistor width, in meters.
        nx : int
            number of resistors in a row.
        ny : int
            number of resistors in a column.
        min_tracks : Tuple[int, int, int, int]
            minimum number of tracks per layer in the resistor unit cell.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
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
            res_type=res_type,
            parity=0,
            em_specs=em_specs or {},
        )
        # create BL corner
        master = self.new_template(params=layout_params, temp_cls=self._corner_cls)  # type: MicroTemplate
        self.add_instance(master)
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
        self.add_instance(master, loc=(self.array_box.right, self.array_box.top),
                          orient='R180')

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
        super(Termination, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
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
            sub_type='ntap',
            res_type='reference',
            em_specs={},
        )

    @classmethod
    def get_params_info(cls):
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
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            sub_type='the substrate type.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def draw_layout(self):
        nx = self.params['nx']
        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')
        ny = self.params['ny']
        em_specs = self.params.pop('em_specs')
        div_em_specs = em_specs.copy()
        for key in ('idc', 'iac_rms', 'iac_peak'):
            div_em_specs[key] = div_em_specs[key] / ny
        self.draw_array(em_specs=div_em_specs, **self.params)

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
        vm_layer = self.bot_layer_id + 1
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
        xm_layer = vm_layer + 1
        xm_width = self.w_tracks[2]
        x_pitch = self.num_tracks[2]
        x_base = self._num_corner_tracks[2] + (x_pitch - 1) / 2.0
        x_tid = TrackID(xm_layer, x_base, width=xm_width, num=ny, pitch=x_pitch)
        xc_warr = self.connect_to_tracks(vc_warr, x_tid)
        xl_warr = self.connect_to_tracks(vl_warr, x_tid)
        xr_warr = self.connect_to_tracks(vr_warr, x_tid)

        # connect to y layer
        ym_layer = xm_layer + 1
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
