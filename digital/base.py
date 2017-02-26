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


"""This module defines classes for creating custom digital circuits.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, Set, List, Any

import yaml

from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.template import TemplateBase, TemplateDB


class StandardCell(TemplateBase):
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
        super(StandardCell, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._std_size = None

    @property
    def std_size(self):
        return self._std_size

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
            tech_params='standard cell technology parameters.',
            lib_name='standard cell library name.',
            cell_name='standard cell cell name.',
            size='size in [row, columns] format.',
            ports='standard cell ports dictionary.',
        )

    def get_layout_basename(self):
        return 'stdcell_%s' % self.params['cell_name']

    def compute_unique_key(self):
        return 'stdcell_%s_%s' % (self.params['lib_name'], self.params['cell_name'])

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lib_name, cell_name, size, ports, tech_params):
        # type: (str, str, List[int], Dict[str, Dict[str, Any]], Dict[str, Any]) -> None
        res = self.grid.resolution

        col_pitch = tech_params['col_pitch']
        height = tech_params['height']
        layers = tech_params['layers']
        widths = tech_params['widths']
        spaces = tech_params['spaces']
        directions = tech_params['directions']

        # update routing grid
        self.grid = self.grid.copy()
        for lay_id, w, sp, tdir in zip(layers, widths, spaces, directions):
            self.grid.add_new_layer(lay_id, sp, w, tdir, override=True)

        # add instance
        self.add_instance_primitive(lib_name, cell_name, (0.0, 0.0))

        # compute size
        self._std_size = size
        num_col, num_row = size
        self.array_box = BBox(0.0, 0.0, num_col * col_pitch, num_row * height, res)
        self.set_size_from_array_box(layers[-1])

        # add pins
        for port_name, port_params in ports.items():
            port_lay_id = port_params['layer']
            bbox = port_params['bbox']
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


class SpaceBlock(TemplateBase):
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
        super(SpaceBlock, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            space_blocks='list of space blocks information.',
            tech_params='standard cell technology parameters.',
        )

    def get_layout_basename(self):
        return 'stdcell_space_%dx' % self.params['num_col']

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self):
        # type: () -> None

        res = self.grid.resolution

        num_col = self.params['num_col']
        space_blocks = self.params['space_blocks']
        tech_params = self.params['tech_params']

        col_pitch = tech_params['col_pitch']
        height = tech_params['height']
        layers = tech_params['layers']
        widths = tech_params['widths']
        spaces = tech_params['spaces']
        directions = tech_params['directions']

        # update routing grid
        self.grid = self.grid.copy()
        for lay_id, w, sp, tdir in zip(layers, widths, spaces, directions):
            self.grid.add_new_layer(lay_id, sp, w, tdir, override=True)

        # compute size
        self.array_box = BBox(0.0, 0.0, num_col * col_pitch, height, res)
        self.set_size_from_array_box(layers[-1])

        # instantiate space blocks
        xcur = 0.0
        for blk_params in space_blocks:
            lib_name = blk_params['lib_name']
            cell_name = blk_params['cell_name']
            blk_col = blk_params['num_col']
            num_blk, num_col = divmod(num_col, blk_col)
            blk_width = blk_col * col_pitch
            self.add_instance_primitive(lib_name, cell_name, (xcur, 0.0),
                                        nx=num_blk, spx=blk_width)
            xcur += num_blk * blk_width

        if num_col > 0:
            raise ValueError('has %d columns remaining' % num_col)


class StdCellLibrary(object):
    def __init__(self, temp_db, config_yaml):
        # type: (TemplateDB, str) -> None
        self._temp_db = temp_db
        self._config = None
        with open(config_yaml, 'r') as f:
            self._config = yaml.load(f)
        self._tech_params = self._config['tech_params']

    def new_stdcell(self, cell_name):
        for key, val in self._config['cells'].items():
            if key == cell_name:
                temp_params = dict(tech_params=self._tech_params)
                temp_params.update(val)
                return self._temp_db.new_template(params=temp_params, temp_cls=StandardCell)
        raise ValueError('Cannot find standard cell with name %s' % cell_name)

    def new_space_block(self, num_col):
        temp_params = dict(
            tech_params=self._tech_params,
            space_blocks=self._config['spaces'],
            num_col=num_col,
        )
        return self._temp_db.new_template(params=temp_params, temp_cls=SpaceBlock)
