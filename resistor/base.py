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


"""This module defines abstract analog resistor array component classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
from typing import Dict, Set, Tuple, Any

from bag import float_to_si_string
from bag.layout.template import TemplateBase, TemplateDB


class AnalogResCore(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog resistors array core.

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
        super(AnalogResCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def use_parity(cls):
        # type: () -> bool
        """Returns True if parity changes resistor core layout."""
        return False

    @classmethod
    @abc.abstractmethod
    def port_layer_id(cls):
        # type: () -> int
        """Returns the resistor port layer ID.

        Bottom port layer must be horizontal.
        """
        return -1

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
            parity=0,
            em_specs={},
            min_tracks=(1, 1, 1, 1),
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            threshold='substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='resistor EM spec specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> str
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
class AnalogResLREdge(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog resistors array left/right edge.

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
        super(AnalogResLREdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            min_tracks=(1, 1, 1, 1),
            parity=0,
            res_type='reference',
            em_specs={},
            edge_space=False,
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            threshold='substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='resistor EM spec specifications.',
            edge_space='True to reserve space to adjacent transistor blocks.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> str
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
        if self.params['edge_space']:
            main += '_edge'

        return main


# noinspection PyAbstractClass
class AnalogResTBEdge(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog resistors array top/bottom edge.

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
        super(AnalogResTBEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            min_tracks=(1, 1, 1, 1),
            parity=0,
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            threshold='substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='resistor EM specifications.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> str
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
class AnalogResCorner(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog resistors array lower-left corner.

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
        super(AnalogResCorner, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            min_tracks=(1, 1, 1, 1),
            parity=0,
            res_type='reference',
            em_specs={},
            edge_space=False,
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
            min_tracks='Minimum number of tracks on each layer per block.',
            parity='the parity of this resistor core.  Either 0 or 1.',
            sub_type='the substrate type.',
            threshold='substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='resistor EM specifications.',
            edge_space='True to reserve space to adjacent transistor blocks.',
        )

    @abc.abstractmethod
    def get_num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
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
        # type: () -> str
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
        if self.params['edge_space']:
            main += '_edge'
        return main
