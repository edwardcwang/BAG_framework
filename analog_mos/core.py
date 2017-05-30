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

from bag.math import lcm
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
    def get_analog_unit_fg(cls):
        # type: () -> int
        """Returns the number of fingers in an AnalogBase row unit.

        Returns
        -------
        num_fg : int
            number of fingers in an AnalogBase row unit.
        """

    @classmethod
    @abc.abstractmethod
    def draw_zero_extension(cls):
        # type: () -> bool
        """Returns True if we should draw 0 width extension.

        Returns
        -------
        draw_ext : bool
            True to draw 0 width extension.
        """
        return False

    @classmethod
    @abc.abstractmethod
    def get_dum_conn_pitch(cls):
        # type: () -> int
        """Returns the minimum track pitch of dummy connections in number of tracks.

        Some technology can only draw dummy connections on every other track.  In that case,
        this method should return 2.

        Returns
        -------
        dum_conn_pitch : pitch between adjacent dummy connection.
        """
        return 1

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
    def get_dig_conn_layer(cls):
        # type: () -> int
        """Returns the digital connection layer ID.  Must be vertical.

        Returns
        -------
        dig_layer : int
            the transistor connection layer ID.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_min_fg_decap(cls, lch_unit):
        # type: (int) -> int
        """Returns the minimum number of fingers for decap connections.
        
        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        
        Returns
        -------
        num_fg : int
            minimum number of decap fingers.
        """
        return 2

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
    def get_edge_info(cls, grid, lch_unit, guard_ring_nf, top_layer, is_end):
        # type: (RoutingGrid, int, int, int, bool) -> Dict[str, Any]
        """Returns a dictionary containing transistor edge layout information.
        
        The returned dictionary must have an entry 'edge_width', which is the width
        of the edge block in resolution units.
        
        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            the channel length, in resolution units.
        guard_ring_nf : int
            guard ring width in number of fingers.
        top_layer : int
            the top routing layer ID.  Used to determine width quantization.
        is_end : bool
            True if there are no blocks abutting the left edge.
        
        Returns
        -------
        edge_info : Dict[str, Any]
            edge layout information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_mos_info(cls, lch_unit, w, mos_type, threshold, fg):
        # type: (int, int, str, str, int) -> Dict[str, Any]
        """Returns the transistor information dictionary.
        
        The returned dictionary must have the following entries:
        
        layout_info: the layout information dictionary.
        ext_top_info: a tuple of values used to compute extension layout above the transistor.
        ext_bot_info : a tuple of values used to compute extension layout below the transistor.
        sd_yc : the Y coordinate of the center of source/drain junction.
        g_conn_y : a Tuple of bottom/top Y coordinates of gate wire on mos_conn_layer.  Used to determine
                   gate tracks location.
        d_conn_y : a Tuple of bottom/top Y coordinates of drain/source wire on mos_conn_layer.  Used to
                   determine drain/source tracks location.

        Parameters
        ----------
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
    def get_substrate_info(cls, lch_unit, w, sub_type, threshold, fg, end_mode, blk_pitch=1, **kwargs):
        # type: (int, int, str, str, int, int, int, **kwargs) -> Dict[str, Any]
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
            the end mode flag.  This is a 2-bit integer.  The LSB is 1 if there are no blocks abutting
            the bottom.  The MSB is 1 if there are no blocks abutting the top.
        blk_pitch : int
            substrate height quantization pitch.  Defaults to 1 (no quantization).
        **kwargs :
            additional arguments.

        Returns
        -------
        sub_info : Dict[str, Any]
            the substrate information dictionary.
        """
        return {}

    @classmethod
    def get_analog_end_info(cls, lch_unit, sub_type, threshold, fg, is_end, blk_pitch):
        # type: (int, str, str, int, bool, int) -> Dict[str, Any]
        """Returns the AnalogBase end row layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        is_end : bool
            True if there are no block abutting the bottom.
        blk_pitch : int
            substrate height quantization pitch.  Defaults to 1 (no quantization).

        Returns
        -------
        sub_info : Dict[str, Any]
            the substrate information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_outer_edge_info(cls, grid, guard_ring_nf, layout_info, top_layer, is_end):
        # type: (RoutingGrid, int, Dict[str, Any], int, bool) -> Dict[str, Any]
        """Returns the outer edge layout information dictionary.
        
        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        guard_ring_nf : int
            guard ring width in number of fingers.  0 if there is no guard ring.
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.
        top_layer : int
            the top routing layer ID.  Used to determine width quantization.
        is_end : bool
            True if there are no blocks abutting the left edge.
            
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
                            min_ds_cap, is_diff, diode_conn, options):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, Dict[str, Any]) -> None
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
        is_diff : bool
            True if this is a differential pair connection.
        diode_conn : bool
            True to short gate and drain together.
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_dum_connection(cls, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[int], Dict[str, Any]) -> None
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
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_decap_connection(cls, template, mos_info, sdir, ddir, gate_ext_mode, export_gate, options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
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
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_laygo_info(cls, grid, lch_unit, w, mos_type, threshold, blk_type):
        # type: (RoutingGrid, int, int, str, str, str) -> Dict[str, Any]
        """Returns the transistor information dictionary for laygo blocks.

        The returned dictionary must have the following entries:

        layout_info: the layout information dictionary.
        ext_top_info: a tuple of values used to compute extension layout above the transistor.
        ext_bot_info : a tuple of values used to compute extension layout below the transistor.
        sd_yc : the Y coordinate of the center of source/drain junction.
        top_gtr_yc : maximum Y coordinate of the center of top gate track.
        bot_dstr_yc : minimum Y coordinate of the center of bottom drain/source track.
        max_bot_tr_yc : maximum Y coordinate of the center of bottom block tracks.
        min_top_tr_yc : minimum Y coordinate of the center of top block tracks.
        blk_height : the block height in mos pitches.

        The 4 track Y coordinates (top_gtr_yc, bot_dstr_yc, max_bot_tr_yc, min_top_tr_yc)
        should be independent of the transistor width.  In this way, you can use different
        width transistors in the same row.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        blk_type : str
            the digital block type.

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_laygo_end_info(cls, grid, lch_unit, mos_type, threshold, top_layer, is_end):
        # type: (RoutingGrid, int, str, str, int, bool) -> Dict[str, Any]
        """Returns the digital end block layout information dictionary.

        The returned dictionary must have the following entries:

        height : the height of the end row.
        yblk : Y coordinate of the end block.
        blk_height : the height of the end block excluding top extension.
        mos_type : the end row transistor type.  One of 'nch', 'pch', 'ptap', or 'ntap'.  Used to determine
                   implant layers.
        ext_top_info : extension information Tuple for adjacent block.
        ext_top_h : minimum top extension height.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            the channel length in resolution units.
        mos_type : str
            the adjacent row transistor type.  Either 'pch' or 'nch'.
        threshold : str
            the threshold type.
        top_layer : int
            the top level routing layer.  Used to determine vertical pitch.
        is_end : bool
            True if no blocks are abutting the bottom.

        Returns
        -------
        end_info : Dict[str, Any]
            the laygo end row information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def draw_laygo_connection(cls, template, mos_info, blk_type):
        # type: (TemplateBase, Dict[str, Any], str) -> None
        """Draw digital transistor connection in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        blk_type : str
            the digital block type.
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
    def get_left_sd_xc(cls, grid, lch_unit, guard_ring_nf, top_layer, is_end):
        # type: (RoutingGrid, int, int, int, bool) -> int
        """Returns the X coordinate of the center of the left-most source/drain junction in a transistor row.

        Parameters
        ----------
        grid: RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            channel length in resolution units
        guard_ring_nf : int
            guard ring width in number of fingers.
        top_layer : int
            the top routing layer ID.  Used to determine width quantization.
        is_end : bool
            True if there are no blocks abutting the left edge.

        Returns
        -------
        sd_xc : X coordinate of the center of the left-most source/drain junction.
        """
        edge_info = cls.get_edge_info(grid, lch_unit, guard_ring_nf, top_layer, is_end)
        return edge_info['edge_width']

    @classmethod
    def get_laygo_row_info(cls, grid, lch_unit, w, mos_type, min_g_tracks, min_ds_tracks):
        # type: (RoutingGrid, int, int, str, Dict[int, int], Dict[int, int]) -> Dict[str, Any]
        """Calculate the height of a PMOS/NMOS row in digital block.

        Parameters
        ----------
        grid: RoutingGrid
            the RoutingGrid object.
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins or resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        min_g_tracks : Dict[int, int]
            a dictionary from layer ID to minimum number of gate tracks on that layer.
            For substrates, this should be an empty dictionary.
        min_ds_tracks : Dict[int, int]
            a dictionary from layer ID to minimum number of drain/source tracks on that layer.

        Returns
        -------
        dig_row_info : Dict[str, Any]
            a dictionary containing information about this digital row.
        """
        mos_info = cls.get_laygo_info(grid, lch_unit, w, mos_type, 'standard', 'fg2d')

        blk_height = mos_info['blk_height']
        ext_bot_info = mos_info['ext_bot_info']
        ext_top_info = mos_info['ext_top_info']
        d_conn_yb, d_conn_yt = mos_info['d_conn_y']
        conn_spy = mos_info['conn_spy']

        conn_layer = cls.get_dig_conn_layer()
        hm_layer = conn_layer + 1

        # step 1: get minimum height and blk_pitch
        min_height = 0
        mos_pitch = cls.get_mos_pitch(unit_mode=True)
        blk_pitch = mos_pitch
        for lay, dstr in min_ds_tracks.items():
            gtr = min_g_tracks.get(lay, 0)
            track_pitch = grid.get_track_pitch(lay, unit_mode=True)
            blk_pitch = lcm([blk_pitch, track_pitch])
            min_height = max(min_height, track_pitch * (gtr + dstr))

        # step 2: based on line-end spacing, find the number of horizontal tracks
        # needed between routing tracks of adjacent blocks.
        hm_width, hm_space = grid.get_track_info(hm_layer, unit_mode=True)
        hm_pitch = hm_width + hm_space
        conn_ext, _ = grid.get_via_extensions(conn_layer, 1, 1, unit_mode=True)
        margin = 2 * conn_ext + conn_spy
        hm_sep = max(-(-(margin - hm_space) // hm_pitch), 0)

        # step 3: find Y coordinate of mos block
        if hm_layer in min_g_tracks:
            g_conn_yb, g_conn_yt = mos_info['g_conn_y']
            # step A: find bottom horizontal track index
            gtr_idx0 = hm_sep // 2
            # step B: find minimum Y coordinate such that the vertical track bottom line-end
            # does not extend below via extension of bottom horizontal track.
            vm_yb = grid.track_to_coord(hm_layer, gtr_idx0, unit_mode=True) - hm_width // 2 - conn_ext
            ymin0 = vm_yb - g_conn_yb
            # step C: find top horizontal track index
            gtr_idx1 = gtr_idx0 + min_g_tracks[hm_layer] - 1
            # step D: find minimum Y coordinate such that vertical track top line-end
            # is higher than the via extension of top horizontal track.
            vm_yt = grid.track_to_coord(hm_layer, gtr_idx1, unit_mode=True) + hm_width // 2 + conn_ext
            ymin1 = vm_yt - g_conn_yt
            # step E: find Y coordinate, round to pitch
            y0 = max(ymin0, ymin1)
            ext_bot_h = -(-y0 // mos_pitch)
            y0 = ext_bot_h * mos_pitch
        else:
            # this is substrate, we just set Y coordinate to 0.
            y0 = 0
            ext_bot_h = 0

        # step 4: find block top boundary Y coordinate
        # step A: find bottom drain track index so via extension line-end is above d_conn_yb
        dtr_y0 = y0 + d_conn_yb + conn_ext + hm_width // 2
        dtr_idx0 = grid.coord_to_nearest_track(hm_layer, dtr_y0, half_track=True, mode=1, unit_mode=True)
        # step B1: find top drain track index based on min_ds_tracks
        dtr_idx1 = dtr_idx0 + min_ds_tracks[hm_layer] - 1
        # step B2: find top drain track index based on d_conn_yt
        dtr_y1 = y0 + d_conn_yt - conn_ext - hm_width // 2
        dtr_idx2 = grid.coord_to_nearest_track(hm_layer, dtr_y1, half_track=True, mode=1, unit_mode=True)
        dtr_idx1 = max(dtr_idx1, dtr_idx2)
        # step C: find block top boundary Y coordinate based on hm_sep
        y1 = grid.track_to_coord(hm_layer, dtr_idx1, unit_mode=True)
        y1 += hm_pitch // 2 * (hm_sep + 1)

        y1 = max(min_height, y1)
        y1 = -(-y1 // blk_pitch) * blk_pitch
        ext_top_h = (y1 - y0 - blk_height) // mos_pitch

        return dict(
            height=y1,
            yblk=y0,
            blk_height=blk_height,
            ext_bot_info=ext_bot_info,
            ext_top_info=ext_top_info,
            ext_bot_h=ext_bot_h,
            ext_top_h=ext_top_h,
        )
