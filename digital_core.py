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

import abc
import yaml
from itertools import chain
from typing import List, Union, Optional, Dict, Any, Set, Tuple
from future.utils import with_metaclass

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.objects import Instance
from bag.layout.routing import TrackID, WireArray, RoutingGrid

from .analog_mos.core import MOSTech
from .analog_mos.mos import AnalogMOSBase, AnalogMOSExt
from .analog_mos.edge import AnalogEdge


class DigitalBase(with_metaclass(abc.ABCMeta, TemplateBase)):

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(DigitalBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

        tech_params = self.grid.tech_info.tech_params
        self._tech_cls = tech_params['layout']['mos_tech_class']  # type: MOSTech

        # error checking
        for key in ('config_file', 'row_parity', 'draw_boundaries'):
            if key not in params:
                raise ValueError('All subclasses of DigitalBase must have %s parameter.' % key)

        # load configuration file
        with open(params['config_file'], 'r') as f:
            self._config = yaml.load(f)

        # update routing grid
        tdir = 'y'
        for lay, w, sp in zip(self._config['layers'], self._config['widths'], self._config['spaces']):
            self.grid.add_new_layer(lay, sp, w, tdir, override=True, unit_mode=True)
            tdir = 'x' if tdir == 'y' else 'y'

        self.grid.update_block_pitch()

        # set properties
        self._row_parity = self.params['row_parity']

    @property
    def row_parity(self):
        # type: () -> int
        return self._row_parity

    def new_dig_template(self, temp_cls, params):
        # type: (Any, Dict[str, Any]) -> DigitalBase
        new_params = params.copy()
        new_params['config_file'] = self.params['config_file']
        return self.new_template(params=new_params, temp_cls=temp_cls)

    def add_dig_instance(self, master, inst_name=None, loc=(0, 0), nx=1, ny=1, spx=0, spy=0, orient='R0'):
        # type: (DigitalBase, Optional[str], Tuple[int, int], int, int, int, int, orient) -> Instance

        # error checking
        # check spy is multiples of 4
        if spy % 4 != 0:
            raise ValueError('spy = %d is not multiples of 4.' % spy)

        # check if row parity matches
        row_idx = loc[1]
        flip_ud = (orient == 'MX' or orient == 'R180')
        inst_parity = master.row_parity

        if flip_ud:
            inst_parity = 3 - inst_parity
        my_parity = (self._row_parity + row_idx) % 4

        if my_parity != inst_parity:
            raise ValueError('Cannot add instance at location {0}; '
                             'inst_parity = {1} != {2}'.format(loc, inst_parity, my_parity))


