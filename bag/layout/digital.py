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


"""This module defines layout template classes for digital standard cells.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
from typing import Dict, Any, Set, Tuple, List

import yaml

from .util import BBox
from .template import TemplateDB, TemplateBase
from .objects import Instance
from .routing import TrackID, WireArray


# noinspection PyAbstractClass
class StdCellBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    """The base class of all micro templates.

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
        with open(params['config_file'], 'r') as f:
            self._config = yaml.load(f)
        self._tech_params = self._config['tech_params']
        self._cells = self._config['cells']
        self._spaces = self._config['spaces']
        self._bound_params = self._config['boundaries']
        super(StdCellBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._std_size = None
        self._std_size_bare = None
        self._draw_boundaries = False

    @property
    def min_space_width(self):
        # type: () -> int
        return self._spaces[-1]['num_col']

    @property
    def std_col_width(self):
        # type: () -> float
        return self._tech_params['col_pitch']

    @property
    def std_row_height(self):
        # type: () -> float
        return self._tech_params['height']

    @property
    def std_size(self):
        # type: () -> Tuple[int, int]
        """Returns the number of columns/rows this standard cell occupy."""
        return self._std_size

    @property
    def std_routing_layers(self):
        # type: () -> List[int]
        return self._tech_params['layers']

    def set_draw_boundaries(self, draw_boundaries):
        # type: (bool) -> None
        self._draw_boundaries = draw_boundaries

    def get_space_blocks(self):
        # type: () -> List[Dict[str, Any]]
        return self._spaces

    def get_cell_params(self, cell_name):
        # type: (str) -> Dict[str, Any]
        for key, val in self._cells.items():
            if key == cell_name:
                return val
        raise ValueError('Cannot find standard cell with name %s' % cell_name)

    def set_std_size(self, std_size):
        # type: (Tuple[int, int]) -> None
        num_col, num_row = std_size
        self._std_size_bare = std_size
        if self._draw_boundaries:
            dx = self._bound_params['lr_width'] * self.std_col_width
            dy = self._bound_params['tb_height'] * self.std_row_height
            self._std_size = (int(std_size[0] + 2 * self._bound_params['lr_width']),
                              int(std_size[1] + 2 * self._bound_params['tb_height']))
        else:
            self._std_size = std_size
            dx, dy = 0, 0
        self.array_box = BBox(0.0, 0.0, num_col * self.std_col_width + 2 * dx,
                              num_row * self.std_row_height + 2 * dy, self.grid.resolution)
        self.set_size_from_array_box(self.std_routing_layers[-1])

    def update_routing_grid(self):
        # type: () -> None
        layers = self._tech_params['layers']
        widths = self._tech_params['widths']
        spaces = self._tech_params['spaces']
        directions = self._tech_params['directions']

        self.grid = self.grid.copy()
        for lay_id, w, sp, tdir in zip(layers, widths, spaces, directions):
            self.grid.add_new_layer(lay_id, sp, w, tdir, override=True)

    def get_num_tracks(self, layer_id):
        # type: (int) -> int
        """Get number of tracks in this cell."""
        ncol, nrow = self.std_size

        tdir = self.grid.get_direction(layer_id)
        pitch = self.grid.get_track_pitch(layer_id, unit_mode=True)
        if tdir == 'x':
            tot_dim = nrow * int(round(self.std_row_height / self.grid.resolution))
        else:
            tot_dim = ncol * int(round(self.std_col_width / self.grid.resolution))
        return tot_dim // pitch

    def add_std_instance(self, master, inst_name=None, loc=(0, 0), nx=1, ny=1,
                         spx=0, spy=0, flip_lr=False):
        # type: (StdCellBase, str, Tuple[int, int], int, int, int, int, bool) -> Instance
        """Add a standard cell instance.

        """
        col_pitch = self.std_col_width
        row_pitch = self.std_row_height
        if loc[1] % 2 == 0:
            orient = 'R0'
            dy = loc[1] * row_pitch
        else:
            orient = 'MX'
            dy = (loc[1] + 1) * row_pitch

        dx = loc[0] * col_pitch
        if flip_lr:
            dx += master.std_size[0] * col_pitch
            if orient == 'R0':
                orient = 'MY'
            else:
                orient = 'R180'

        if spy % 2 != 0:
            raise ValueError('row pitch must be even')
        spx *= col_pitch
        spy *= row_pitch
        if self._draw_boundaries:
            dx += self._bound_params['lr_width'] * self.std_col_width
            dy += self._bound_params['tb_height'] * self.std_row_height

        return self.add_instance(master, inst_name=inst_name, loc=(dx, dy),
                                 orient=orient, nx=nx, ny=ny, spx=spx, spy=spy)

    def draw_boundaries(self):
        lib_name = self._bound_params['lib_name']
        num_col, num_row = self._std_size_bare
        num_row_even = (num_row + 1) // 2
        num_row_odd = num_row - num_row_even
        wcol, hrow = self.std_col_width, self.std_row_height
        dx = self._bound_params['lr_width'] * wcol
        dy = self._bound_params['tb_height'] * hrow

        # add bottom-left
        self.add_instance_primitive(lib_name, 'boundary_bottomleft', (0, 0))

        # add left
        self.add_instance_primitive(lib_name, 'boundary_left', (0, dy), ny=num_row_even, spy=hrow * 2)
        if num_row_odd > 0:
            self.add_instance_primitive(lib_name, 'boundary_left', (0, dy + 2 * hrow),
                                        orient='MX', ny=num_row_odd, spy=-hrow * 2)

        # add top-left
        if num_row % 2 == 1:
            yc = dy + num_row * hrow
            self.add_instance_primitive(lib_name, 'boundary_topleft', (0, yc))
        else:
            yc = 2 * dy + num_row * hrow
            self.add_instance_primitive(lib_name, 'boundary_bottomleft', (0, yc), orient='MX')

        # add bottom
        self.add_instance_primitive(lib_name, 'boundary_bottom', (dx, 0), nx=num_col, spx=wcol)

        # add top
        if num_row % 2 == 1:
            self.add_instance_primitive(lib_name, 'boundary_top', (dx, yc), nx=num_col, spx=wcol)
        else:
            self.add_instance_primitive(lib_name, 'boundary_bottom', (dx, yc), orient='MX',
                                        nx=num_col, spx=wcol)

        # add bottom right
        xc = dx + num_col * wcol
        self.add_instance_primitive(lib_name, 'boundary_bottomright', (xc, 0))

        # add right
        self.add_instance_primitive(lib_name, 'boundary_right', (xc, dy), ny=num_row_even, spy=hrow * 2)
        if num_row_odd > 0:
            self.add_instance_primitive(lib_name, 'boundary_right', (xc, dy + 2 * hrow),
                                        orient='MX', ny=num_row_odd, spy=-hrow * 2)

        # add top right
        if num_row % 2 == 1:
            self.add_instance_primitive(lib_name, 'boundary_topright', (xc, yc))
        else:
            self.add_instance_primitive(lib_name, 'boundary_bottom', (xc, yc), orient='MX')


class StdCellTemplate(StdCellBase):
    """A template wrapper around a standard cell block.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(StdCellTemplate, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            cell_name='standard cell cell name.',
            config_file='standard cell configuration file name.',
        )

    def get_layout_basename(self):
        return 'stdcell_%s' % self.params['cell_name']

    def compute_unique_key(self):
        cell_params = self.get_cell_params(self.params['cell_name'])
        return 'stdcell_%s_%s' % (cell_params['lib_name'], cell_params['cell_name'])

    def draw_layout(self):
        # type: () -> None

        cell_params = self.get_cell_params(self.params['cell_name'])
        lib_name = cell_params['lib_name']
        cell_name = cell_params['cell_name']
        size = cell_params['size']
        ports = cell_params['ports']

        # update routing grid
        self.update_routing_grid()
        # add instance
        self.add_instance_primitive(lib_name, cell_name, (0, 0))
        # compute size
        self.set_std_size(size)

        # add pins
        for port_name, pin_list in ports.items():
            for pin in pin_list:
                port_lay_id = pin['layer']
                bbox = pin['bbox']
                layer_dir = self.grid.get_direction(port_lay_id)
                if layer_dir == 'x':
                    intv = bbox[1], bbox[3]
                    lower, upper = bbox[0], bbox[2]
                else:
                    intv = bbox[0], bbox[2]
                    lower, upper = bbox[1], bbox[3]
                tr_idx, tr_w = self.grid.interval_to_track(port_lay_id, intv)
                warr = WireArray(TrackID(port_lay_id, tr_idx, width=tr_w), lower, upper)
                self.add_pin(port_name, warr, show=False)


class StdCellSpace(StdCellBase):
    """An template for creating termination resistors.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(StdCellSpace, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_col='number of space columns.',
            config_file='standard cell configuration file name.',
        )

    def get_layout_basename(self):
        return 'stdcell_space_%dx' % self.params['num_col']

    def compute_unique_key(self):
        return self.get_layout_basename() + '_' + self.params['config_file']

    def draw_layout(self):
        # type: () -> None

        num_col = self.params['num_col']

        # update routing grid
        self.update_routing_grid()
        # compute size
        self.set_std_size((num_col, 1))

        # instantiate space blocks
        col_pitch = self.std_col_width
        xcur = 0.0
        for blk_params in self.get_space_blocks():
            lib_name = blk_params['lib_name']
            cell_name = blk_params['cell_name']
            blk_col = blk_params['num_col']
            num_blk, num_col = divmod(num_col, blk_col)
            blk_width = blk_col * col_pitch
            if num_blk > 0:
                self.add_instance_primitive(lib_name, cell_name, (xcur, 0.0),
                                            nx=num_blk, spx=blk_width)
                xcur += num_blk * blk_width

        if num_col > 0:
            raise ValueError('has %d columns remaining' % num_col)
