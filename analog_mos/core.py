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

from bag import float_to_si_string
from bag.layout.template import TemplateBase


class AnalogMosBase(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogMosBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def get_num_fingers_per_sd(cls, lch):
        # type: (float) -> int
        """Returns number of transistor fingers per source/drain pitch.

        Parameters
        ----------
        lch : float
            channel length, in meters.

        Returns
        -------
        num_fg : int
            number of transistor fingers per source/drain pitch.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_template_width(cls, fg_tot, guard_ring_nf=0):
        # type: (int, int) -> int
        """Returns the width of the AnalogMosBase in number of vertical tracks.

        Parameters
        ----------
        fg_tot : int
            number of fingers.
        guard_ring_nf : int
            width of guard ring in number of fingers.  0 to disable guard ring.

        Returns
        -------
        mos_width : int
            the AnalogMosBase width in number of vertical tracks.
        """
        return 0

    @classmethod
    @abc.abstractmethod
    def get_sd_pitch(cls, lch):
        """Returns the source/drain pitch given channel length.

        Parameters
        ----------
        lch : float
            channel length, in meters.

        Returns
        -------
        sd_pitch : float
            the source/drain pitch
        """
        return 0.0

    @classmethod
    @abc.abstractmethod
    def port_layer_id(cls):
        """Returns the mosfet connection layer ID.

        Returns
        -------
        port_layer_id : int
            dummy connection layer ID.
        """
        return -1

    @classmethod
    @abc.abstractmethod
    def get_port_width(cls, lch):
        """Returns the width of the AnalogMosConn port.

        Parameters
        ----------
        lch : float
            channel length, in meters.

        Returns
        -------
        port_width : float
            th port width in layout units.
        """
        return 0.0

    @classmethod
    @abc.abstractmethod
    def get_left_sd_xc(cls, lch, guard_ring_nf):
        """Returns the center X coordinate of the leftmost source/drain connection.

        Parameters
        ----------
        lch : float
            channel length, in meters.
        guard_ring_nf : int
            guard ring width in number of fingers.  0 to disable.

        Returns
        -------
        xc : float
            center X coordinate of leftmost source/drain
        """
        return 0.0

    @abc.abstractmethod
    def get_left_sd_center(self):
        """Returns the center coordinate of the leftmost source/drain connection.

        Returns
        -------
        sd_loc : (float, float)
            center coordinate of leftmost source/drain
        """
        return 0.0, 0.0

    @abc.abstractmethod
    def get_ds_track_index(self):
        """Returns the bottom drain/source track index.

        Returns
        -------
        tr_idx : int
            the bottom drain/source track index.
        """
        return 2

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            g_tracks=1,
            ds_tracks=2,
            gds_space=1,
            guard_ring_nf=0,
            end_mode=0,
            is_ds_dummy=False,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            mos_type="transistor type, either 'pch' or 'nch'.",
            threshold='transistor threshold flavor.',
            fg='number of fingers.',
            g_tracks='number of gate tracks.',
            ds_tracks='number of drain/source tracks.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  Use 0 for no guard ring.',
            is_ds_dummy='True if this template is only used to create drain/source dummy metals.',
            end_mode='An integer indicating whether top/bottom of this template is at the ends.',
        )

    def get_num_tracks(self):
        """Returns the number of horizontal tracks in this template.

        AnalogMosBase should always have at least one track, and the bottom-most track is always
        for gate connection.

        Returns
        -------
        num_track : int
            number of tracks in this template.
        """
        h_layer_id = self.port_layer_id() + 1
        tr_pitch = self.grid.get_track_pitch(h_layer_id, unit_mode=True)
        h = self.array_box.height_unit
        q, r = divmod(h, tr_pitch)
        if r != 0:
            raise Exception('array box height = %.4g not integer number of track pitch = %.4g' % (h, tr_pitch))
        return q

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        g_ntr = self.params['g_tracks']
        ds_ntr = self.params['ds_tracks']
        tr_sp = self.params['gds_space']
        gr_nf = self.params['guard_ring_nf']
        main = '%s_%s_l%s_w%s_fg%d_ng%d_nds%d_sp%d' % (self.params['mos_type'],
                                                       self.params['threshold'],
                                                       lch_str, w_str,
                                                       self.params['fg'],
                                                       g_ntr, ds_ntr, tr_sp)
        name = 'base_end%d' % self.params['end_mode']
        if self.params['is_ds_dummy']:
            name += '_dsdummy'
        if gr_nf > 0:
            return '%s_gr%d_%s' % (main, gr_nf, name)
        else:
            return main + '_' + name

    def compute_unique_key(self):
        return self.get_layout_basename()


# noinspection PyAbstractClass
class AnalogSubstrate(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for substrate connection.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogSubstrate, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def get_implant_layers(cls, sub_type, threshold):
        """Returns a list of all layers needed to define substrate.

        Parameters
        ----------
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold flavor.

        Returns
        -------
        imp_list : List[Tuple[str, str]]
            list of implantation layer/purpose pairs.
        """
        return []

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            guard_ring_nf=0,
            end_mode=1,
            port_tracks=[],
            dum_tracks=[],
            dummy_only=False,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='substrate width, in meters/number of fins.',
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            fg='number of fingers.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  Use 0 for no guard ring.',
            end_mode='An integer indicating whether top/bottom of this template is at the ends.',
            dummy_only='True if only dummy connections will be made to this substrate.',
            port_tracks='Substrate port must contain these track indices.',
            dum_tracks='Dummy port must contain these track indices.',
        )

    def get_num_tracks(self):
        """Returns the number of horizontal tracks in this template.

        AnalogMosBase should always have at least one track, and the bottom-most track is always
        for gate connection.

        Returns
        -------
        num_track : int
            number of tracks in this template.
        """
        tech_params = self.grid.tech_info.tech_params
        mos_cls = tech_params['layout']['mos_template']
        h_layer_id = mos_cls.port_layer_id() + 1
        tr_pitch = self.grid.get_track_pitch(h_layer_id, unit_mode=True)
        h = self.array_box.height_unit
        q, r = divmod(h, tr_pitch)
        if r != 0:
            raise Exception('array box height = %.4g not integer number of track pitch = %.4g' % (h, tr_pitch))
        return q

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        gr_nf = self.params['guard_ring_nf']
        dum_only = self.params['dummy_only']
        main = '%s_%s_l%s_w%s_fg%d' % (self.params['sub_type'],
                                       self.params['threshold'],
                                       lch_str, w_str,
                                       self.params['fg'])
        name = 'base_end%d' % self.params['end_mode']

        if dum_only:
            name += '_dumonly'

        if gr_nf > 0:
            return '%s_gr%d_%s' % (main, gr_nf, name)
        else:
            return main + '_' + name

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        port_tracks = self.params['port_tracks']
        dum_tracks = self.params['dum_tracks']
        return '%s_%s_%s' % (basename, repr(port_tracks), repr(dum_tracks))


# noinspection PyAbstractClass
class AnalogMosConn(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog mosfet connections.

    Connects drain, gate, and source to a high level vertical metal layer.
    Assumes the center of the left-most source/drain junction is at (0, 0).

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogMosConn, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def support_diff_mode(cls):
        """Returns True if diff pair mode is supported."""
        return True

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            sdir='source connection direction.  0 for down, 1 for middle, 2 for up.',
            ddir='drain connection direction.  0 for down, 1 for middle, 2 for up.',
            min_ds_cap='True to minimize parasitic Cds.',
            gate_pref_loc="Preferred gate vertical track location.  Either 's' or 'd'.",
            is_ds_dummy='True if this is only a drain/source dummy metal connection.',
            is_diff='True to draw a differential pair connection instead (shared source).',
            diode_conn='True to short drain/gate',
            gate_ext_mode='connect gate using lower level metal to adjacent transistors.',
        )

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            min_ds_cap=False,
            gate_pref_loc='d',
            is_ds_dummy=False,
            is_diff=False,
            diode_conn=False,
            gate_ext_mode=0,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        prefix = 'mconn'
        if self.params['is_diff']:
            prefix += '_diff'

        basename = '%s_l%s_w%s_fg%d_s%d_d%d' % (prefix, lch_str, w_str,
                                                self.params['fg'],
                                                self.params['sdir'],
                                                self.params['ddir'],
                                                )

        if self.params['min_ds_cap']:
            basename += '_minds'
        if self.params['is_ds_dummy']:
            basename += '_dsdummy'
        if self.params['diode_conn']:
            basename += '_diode'
        gext = self.params['gate_ext_mode']
        if gext > 0:
            basename += '_gext%d' % gext
        return basename

    def compute_unique_key(self):
        return self.get_layout_basename()


# noinspection PyAbstractClass
class AnalogMosSep(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog mosfet separator.

    A separator is a group of dummy transistors that separates the drain/source
    junction of one transistor from another.

    To subclass this class, make sure to implement the get_min_fg() class method.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogMosSep, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_min_fg(cls):
        """Returns the minimum number of fingers.

        Subclasses must override this method to return the correct value.

        Returns
        -------
        min_fg : int
            minimum number of fingers.
        """
        return 2

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            gate_intv_list='list of gate intervals to draw substrate connections.',
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        return 'msep_l%s_w%s_fg%d' % (lch_str, w_str,
                                      self.params['fg'])

    def compute_unique_key(self):
        base_name = self.get_layout_basename()
        return '%s_%s' % (base_name, repr(self.params['gate_intv_list']))


# noinspection PyAbstractClass
class AnalogMosDummy(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog mosfet dummy.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogMosDummy, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    @abc.abstractmethod
    def port_layer_id(cls):
        """Returns the dummy connection layer ID.

        Returns
        -------
        port_layer_id : int
            dummy connection layer ID.
        """
        return -1

    @classmethod
    @abc.abstractmethod
    def get_port_width(cls, lch):
        """Returns the width of the AnalogMosConn port.

        Parameters
        ----------
        lch : float
            channel length, in meters.

        Returns
        -------
        port_width : float
            th port width in layout units.
        """
        return 0.0

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            gate_intv_list='list of gate intervals to draw substrate connections.',
            conn_right='True to connect the right-most source to substrate.',
        )

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            conn_right=False,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        basename = 'mdummy_l%s_w%s_fg%d' % (lch_str, w_str,
                                            self.params['fg'],)
        if self.params['conn_right']:
            return basename + '_full'
        return basename

    def compute_unique_key(self):
        base_name = self.get_layout_basename()
        return '%s_%s' % (base_name, repr(self.params['gate_intv_list']))


# noinspection PyAbstractClass
class AnalogMosDecap(with_metaclass(abc.ABCMeta, TemplateBase)):
    """An abstract template for analog mosfet dummy.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogMosDecap, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            gate_ext_mode='connect gate using lower level metal to adjacent transistors.',
        )

    @classmethod
    def get_default_param_values(cls):
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            gate_ext_mode=0,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        gext = self.params['gate_ext_mode']
        basename = 'mdecap_l%s_w%s_fg%d_gext%d' % (lch_str, w_str, self.params['fg'], gext)

        return basename

    def compute_unique_key(self):
        return self.get_layout_basename()
