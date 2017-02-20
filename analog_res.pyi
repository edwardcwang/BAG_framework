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
from builtins import *
from future.utils import with_metaclass

import abc

from typing import Dict, Any, Set, Tuple, Union, Type, TypeVar

from bag.layout.routing import Port, WireArray
from bag.layout.template import MicroTemplate, TemplateDB


class AnalogResCore(with_metaclass(abc.ABCMeta, MicroTemplate)):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        super(AnalogResCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def use_parity(cls) -> bool: ...

    @classmethod
    def port_layer_id(cls) -> int: ...

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]: ...

    @classmethod
    def get_params_info(cls) -> Dict[str, str]: ...

    def get_num_tracks(self) -> Tuple[int, int, int, int] : ...

    def get_num_corner_tracks(self) -> Tuple[int, int, int, int] : ...

    def get_track_widths(self) -> Tuple[int, int, int, int] : ...

    def get_layout_basename(self) -> str: ...


class AnalogResLREdge(with_metaclass(abc.ABCMeta, MicroTemplate)):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        super(AnalogResLREdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]: ...

    @classmethod
    def get_params_info(cls) -> Dict[str, str]: ...

    def get_num_tracks(self) -> Tuple[int, int, int, int] : ...

    def get_layout_basename(self) -> str: ...


class AnalogResTBEdge(with_metaclass(abc.ABCMeta, MicroTemplate)):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        super(AnalogResTBEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]: ...

    @classmethod
    def get_params_info(cls) -> Dict[str, str]: ...

    def get_num_tracks(self) -> Tuple[int, int, int, int] : ...

    def get_layout_basename(self) -> str: ...


class AnalogResCorner(with_metaclass(abc.ABCMeta, MicroTemplate)):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        super(AnalogResCorner, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]: ...

    @classmethod
    def get_params_info(cls) -> Dict[str, str]: ...

    def get_num_tracks(self) -> Tuple[int, int, int, int] : ...

    def get_layout_basename(self) -> str: ...


A = TypeVar('A', bound=AnalogResCore)
B = TypeVar('B', bound=AnalogResLREdge)
C = TypeVar('C', bound=AnalogResTBEdge)
D = TypeVar('D', bound=AnalogResCorner)

class ResArrayBase(with_metaclass(abc.ABCMeta, MicroTemplate)):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        super(ResArrayBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._core_cls = ...  # type: Type[A]
        self._edgelr_cls = ...  # type: Type[B]
        self._edgetb_cls = ...  # type: Type[C]
        self._corner_cls = ...  # type: Type[D]
        self._use_parity = ... # type: bool
        self._bot_port = ...  # type: Port
        self._top_port = ...  # type: Port
        self._core_offset = ...  # type: Tuple[float, float]
        self._core_pitch = ...  # type: Tuple[float, float]
        self._num_tracks = ...  # type: Tuple[int, int, int, int]
        self._num_corner_tracks = ...  # type: Tuple[int, int, int, int]
        self._w_tracks = ...  # type: Tuple[int, int, int, int]
        self._hm_layer = ... # type: int

    @property
    def num_tracks(self) -> Tuple[int, int, int, int]:
        return self._num_tracks

    @property
    def bot_layer_id(self) -> int:
        return self._hm_layer

    @property
    def w_tracks(self) -> Tuple[int, int, int, int]:
        return self._w_tracks

    def get_res_ports(self, row_idx: int, col_idx: int) -> Tuple[WireArray, WireArray]: ...

    def draw_array(self, l: float, w: float, nx: int, ny: int, min_tracks: Tuple[int, int, int, int],
                   sub_type: str, res_type: str, em_specs: Union[None, Dict[str, Any]]) -> None: ...

    def _add_blk(self, temp_cls: Type[MicroTemplate], params: Dict[str, Any], loc: Tuple[float, float],
                 orient: str, nx: int, ny: int, par0: int) -> MicroTemplate: ...


class Termination(ResArrayBase):
    def __init__(self, temp_db: TemplateDB, lib_name: str, params: Dict[str, Any],
                 used_names: Set[str], **kwargs):
        MicroTemplate.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]: ...

    @classmethod
    def get_params_info(cls) -> Dict[str, str]: ...

    def draw_layout(self) -> None: ...
