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

from typing import Dict, Any, Union, Tuple, List

from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateBase

import abc


class MOSTech(with_metaclass(abc.ABCMeta, object)):
    """An abstract static class for drawing transistor related layout.
    
    This class defines various static methods use to draw layouts used by AnalogBase.
    """

    @classmethod
    @abc.abstractmethod
    def get_mos_tech_constants(cls, lch_unit):
        # type: (int) -> Dict[str, Any]
        """Returns a dictionary of technology constants given transistor channel length.
        
        Must have the following entries:
        
        sd_pitch : the source/drain pitch of the transistor in resolution units.
        mos_conn_w : the transistor connection track width in resolution units.
        dum_conn_w : the dummy connection track width in resolution units.
        num_sd_per_track : number of transistor source/drain junction per vertical track.
        
        
        Parameters
        ----------
        lch_unit : int
            the channel length, in resolution units.
        
        Returns
        -------
        tech_dict : Dict[str, Any]
            a technology constants dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_dum_conn_layer(cls):
        # type: () -> int
        """Returns the dummy connection layer ID.  Must be vertical.
        
        Returns
        -------
        dum_layer : int
            the dummy connection layer ID.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_mos_conn_layer(cls):
        # type: () -> int
        """Returns the transistor connection layer ID.  Must be vertical.
        
        Returns
        -------
        mos_layer : int
            the transistor connection layer ID.
        """
        return 3

    @classmethod
    @abc.abstractmethod
    def get_tech_constant(cls, name):
        # type: (str) -> Any
        """Returns the value of the given technology constant.
        
        Parameters
        ----------
        name : str
            constant name.
            
        Returns
        -------
        val : Any
            constant value.
        """
        return 0

    @classmethod
    @abc.abstractmethod
    def get_mos_pitch(cls, unit_mode=False):
        # type: (bool) -> Union[float, int]
        """Returns the transistor vertical placement quantization pitch.
        
        This is usually the fin pitch for finfet process.
        
        Parameters
        ----------
        unit_mode : bool
            True to return the pitch in resolution units.
            
        Returns
        -------
        mos_pitch : Union[float, int]
            the transistor vertical placement quantization pitch.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_edge_info(cls, lch_unit, guard_ring_nf):
        # type: (int, int) -> Dict[str, Any]
        """Returns a dictionary containing transistor edge layout information.
        
        The returned dictionary must have an entry 'edge_width', which is the width
        of the edge block in resolution units.
        
        Parameters
        ----------
        lch_unit : int
            the channel length, in resolution units.
        guard_ring_nf : int
            guard ring width in number of fingers.
        
        Returns
        -------
        edge_info : Dict[str, Any]
            edge layout information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_mos_info(cls, grid, lch_unit, w, mos_type, threshold, fg):
        # type: (RoutingGrid, int, int, str, str, int) -> Dict[str, Any]
        """Returns the transistor information dictionary.
        
        The returned dictionary must have the following entries:
        
        layout_info: the layout information dictionary.
        ext_top_info: a tuple of values used to compute extension layout above the transistor.
        ext_bot_info : a tuple of values used to compute extension layout below the transistor.
        sd_yc : the Y coordinate of the center of source/drain junction.
        top_gtr_yc : maximum Y coordinate of the center of top gate track.
        bot_dstr_yc : minimum Y coordinate of the center of bottom drain/source track.
        max_bot_tr_yc : maximum Y coordinate of the center of bottom block tracks.
        min_top_tr_yc : minimum Y coordinate of the center of top block tracks.
        
        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor w in number of fins/resolution units.
        mos_type : str
            the transistor type.  Either 'pch' or 'nch'.
        threshold : str
            the transistor threshold flavor.
        fg : int
            number of fingers in this transistor row.
            
        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_valid_extension_widths(cls, lch_unit, top_ext_info, bot_ext_info):
        # type: (int, Tuple[Any, ...], Tuple[Any, ...]) -> List[int]
        """Returns a list of valid extension widths in mos_pitch units.
        
        the list should be sorted in increasing order, and any extension widths greater than
        or equal to the last element should be valid.  For example, if the returned list
        is [0, 2, 5], then extension widths 0, 2, 5, 6, 7, ... are valid, while extension
        widths 1, 3, 4 are not valid.
        
        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        top_ext_info : Tuple[Any, ...]
            a tuple containing layout information about the top block.
        bot_ext_info : Tuple[Any, ...]
            a tuple containing layout information about the bottom block.
        """
        return [0]

    @classmethod
    @abc.abstractmethod
    def get_ext_info(cls, lch_unit, w, bot_mtype, top_mtype, bot_thres, top_thres, fg,
                     top_ext_info, bot_ext_info):
        # type: (int, int, str, str, str, str, int, Tuple[Any, ...], Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the extension layout information dictionary.
        
        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        bot_mtype : str
            the bottom block type.
        top_mtype : str
            the top block type.
        bot_thres : str
            the bottom threshold flavor.
        top_thres : str
            the top threshold flavor.
        fg : int
            total number of fingers.
        top_ext_info : Tuple[Any, ...]
            a tuple containing layout information about the top block.
        bot_ext_info : Tuple[Any, ...]
            a tuple containing layout information about the bottom block.
            
        Returns
        -------
        ext_info : Dict[str, Any]
            the extension information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_substrate_info(cls, lch_unit, w, sub_type, threshold, fg, end_mode, is_passive):
        # type: (int, int, str, str, int, int, bool) -> Dict[str, Any]
        """Returns the substrate layout information dictionary.
        
        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        end_mode : int
            the substrate end_mode flag.  This is a two bit integer, the LSB is 1 if
            there are no blocks abutting the substrate bottom, and the MSB is 1 if there
            are no bloks abutting the substrate top.
        is_passive : bool
            True if this is the substrate connection for passive devices.
            
        Returns
        -------
        sub_info : Dict[str, Any]
            the substrate information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_outer_edge_info(cls, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]
        """Returns the outer edge layout information dictionary.
        
        Parameters
        ----------
        guard_ring_nf : int
            guard ring width in number of fingers.  0 if there is no guard ring.
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.
            
        Returns
        -------
        outer_edge_info : Dict[str, Any]
            the outer edge layout information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_gr_sub_info(cls, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]
        """Returns the guard ring substrate layout information dictionary.

        Parameters
        ----------
        guard_ring_nf : int
            guard ring width in number of fingers.  0 if there is no guard ring.
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.

        Returns
        -------
        gr_sub_info : Dict[str, Any]
            the guard ring substrate layout information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_gr_sep_info(cls, layout_info):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """Returns the guard ring separator layout information dictionary.

        Parameters
        ----------
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.

        Returns
        -------
        gr_sub_info : Dict[str, Any]
            the guard ring separator layout information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def draw_mos(cls, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor layout structure in the given template.
        
        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        layout_info : Dict[str, Any]
            layout information dictionary for the transistor/substrate/extension/edge blocks.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_substrate_connection(cls, template, layout_info, port_tracks, dum_tracks, dummy_only):
        # type: (TemplateBase, Dict[str, Any], List[int], List[int], bool) -> None
        """Draw substrate connection layout in the given template.
        
        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        layout_info : Dict[str, Any]
            the substrate layout information dictionary.
        port_tracks : List[int]
            list of port track indices that must be drawn on transistor connection layer.
        dum_tracks : List[int]
            list of dummy port track indices that must be drawn on dummy connection layer.
        dummy_only : bool
            True to only draw connections up to dummy connection layer.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_mos_connection(cls, template, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                            min_ds_cap, is_ds_dummy, is_diff, diode_conn):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, bool) -> None
        """Draw transistor connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        sdir : int
            source direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        ddir : int
            drain direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        gate_pref_loc : str
            preferred gate location flag, either 's' or 'd'.  This is only used if both source
            and drain did not go down.
        gate_ext_mode : int
            gate extension flag.  This is a 2 bit integer, the LSB is 1 if we should connect
            gate to the left adjacent block on lower metal layers, the MSB is 1 if we should
            connect gate to the right adjacent block on lower metal layers.
        min_ds_cap : bool
            True to minimize drain-to-source parasitic capacitance.
        is_ds_dummy : bool
            True if this connection is drawn on a dummy row used only for drain/source capacitance
            neutralization.
        is_diff : bool
            True if this is a differential pair connection.
        diode_conn : bool
            True to short gate and drain together.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_dum_connection(cls, template, mos_info, edge_mode, gate_tracks):
        # type: (TemplateBase, Dict[str, Any], int, List[int]) -> None
        """Draw dummy connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        edge_mode : int
            the dummy edge mode flag.  This is a 2-bit integer, the LSB is 1 if there are no
            transistors on the left side of the dummy, the MSB is 1 if there are no transistors
            on the right side of the dummy.
        gate_tracks : List[int]
            list of dummy connection track indices.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_decap_connection(cls, template, mos_info, sdir, ddir, gate_ext_mode, export_gate):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool) -> None
        """Draw decoupling cap connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        sdir : int
            source direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        ddir : int
            drain direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        gate_ext_mode : int
            gate extension flag.  This is a 2 bit integer, the LSB is 1 if we should connect
            gate to the left adjacent block on lower metal layers, the MSB is 1 if we should
            connect gate to the right adjacent block on lower metal layers.
        export_gate : bool
            True to draw gate connections up to transistor connection layer.
        """
        pass

    @classmethod
    def get_dum_conn_track_info(cls, lch_unit):
        # type: (int) -> Tuple[int, int]
        """Returns dummy connection layer space and width.
        
        Parameters
        ----------
        lch_unit : int
            channel length in resolution units.
            
        Returns
        -------
        dum_sp : int
            space between dummy tracks in resolution units.
        dum_w : int
            width of dummy tracks in resolution units.
        """
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        dum_conn_w = mos_constants['dum_conn_w']
        num_sd_per_track = mos_constants['num_sd_per_track']
        return sd_pitch * num_sd_per_track - dum_conn_w, dum_conn_w

    @classmethod
    def get_mos_conn_track_info(cls, lch_unit):
        # type: (int) -> Tuple[int, int]
        """Returns transistor connection layer space and width.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units.

        Returns
        -------
        tr_sp : int
            space between transistor connection tracks in resolution units.
        tr_w : int
            width of transistor connection tracks in resolution units.
        """
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        mos_conn_w = mos_constants['mos_conn_w']
        num_sd_per_track = mos_constants['num_sd_per_track']

        return sd_pitch * num_sd_per_track - mos_conn_w, mos_conn_w

    @classmethod
    def get_num_fingers_per_sd(cls, lch_unit):
        # type: (int) -> int
        """Returns the number of transistor source/drain junction per vertical track.
        
        Parameters
        ----------
        lch_unit : int
            channel length in resolution units
            
        Returns
        -------
        num_sd_per_track : number of source/drain junction per vertical track.
        """
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        return mos_constants['num_sd_per_track']

    @classmethod
    def get_sd_pitch(cls, lch_unit):
        # type: (int) -> int
        """Returns the source/drain pitch in resolution units.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units

        Returns
        -------
        sd_pitch : the source/drain pitch in resolution units.
        """
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        return mos_constants['sd_pitch']

    @classmethod
    def get_left_sd_xc(cls, lch_unit, guard_ring_nf):
        # type: (int, int) -> int
        """Returns the X coordinate of the center of the left-most source/drain junction in a transistor row.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units
        guard_ring_nf : int
            guard ring width in number of fingers.

        Returns
        -------
        sd_xc : X coordinate of the center of the left-most source/drain junction.
        """
        edge_info = cls.get_edge_info(lch_unit, guard_ring_nf)
        return edge_info['edge_width']
