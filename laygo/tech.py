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

from typing import Dict, Any, Tuple, List

from bag.layout.template import TemplateBase
from bag.layout.routing import WireArray

import abc

from ..analog_mos.core import MOSTech
from ..analog_mos.mos import AnalogMOSExt
from ..analog_mos.edge import AnalogEdge


class LaygoTech(with_metaclass(abc.ABCMeta, MOSTech)):
    """An abstract static class for drawing transistor related layout.

    This class defines various static methods use to draw layouts used by AnalogBase.
    """

    @classmethod
    @abc.abstractmethod
    def get_default_end_info(cls):
        # type: () -> Any
        """Returns the default end_info object."""
        return 0

    @classmethod
    @abc.abstractmethod
    def get_laygo_fg2d_s_short(cls):
        # type: () -> bool
        """Returns True if the two source wires of fg2d is shorted together in the primitive.

        Returns
        -------
        s_short : bool
            True if the two source wires of fg2d is shorted together
        """
        return False

    @classmethod
    @abc.abstractmethod
    def get_laygo_unit_fg(cls):
        # type: () -> int
        """Returns the number of fingers in a LaygoBase unit cell.

        Returns
        -------
        num_fg : int
            number of fingers in a LaygoBase unit cell.
        """
        return 2

    @classmethod
    @abc.abstractmethod
    def get_sub_columns(cls, lch_unit):
        # type: (int) -> int
        """Returns the number of columns per substrate block.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        num_cols : int
            number of columns per substrate block.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_min_sub_space_columns(cls, lch_unit):
        # type: (int) -> int
        """Returns the minimum number of space columns needed around substrate blocks.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        num_cols : int
            minimum number of space columns.
        """
        return 1

    @classmethod
    @abc.abstractmethod
    def get_laygo_sub_info(cls, lch_unit, w, mos_type, threshold, **kwargs):
        # type: (int, int, str, str, **kwargs) -> Dict[str, Any]
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
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        **kwargs
            optional keyword arguments

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_laygo_mos_info(cls, lch_unit, w, mos_type, threshold, blk_type, **kwargs):
        # type: (int, int, str, str, str, **kwargs) -> Dict[str, Any]
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
        **kwargs
            optional keyword arguments.

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_laygo_end_info(cls, lch_unit, mos_type, threshold, fg, is_end, blk_pitch):
        # type: (int, str, str, int, bool, int) -> Dict[str, Any]
        """Returns the LaygoBase end row layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        mos_type : str
            the transistor type, one of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        is_end : bool
            True if there are no block abutting the bottom.
        blk_pitch : int
            height quantization pitch, in resolution units.

        Returns
        -------
        end_info : Dict[str, Any]
            the laygo end row information dictionary.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_laygo_edge_info(cls, blk_info, end_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]
        """Returns a new layout information dictionary for drawing LaygoBase edge blocks.

        Parameters
        ----------
        blk_info : Dict[str, Any]
            the layout information dictionary.
        end_info : Any
            the end information of the block adjacent to this edge.

        Returns
        -------
        edge_info : Dict[str, Any]
            the edge layout information dictionary.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_laygo_space_info(cls, row_info, num_blk, adj_end_info):
        # type: (Dict[str, Any], int, Any) -> Dict[str, Any]
        """Returns a new layout information dictionary for drawing LaygoBase space blocks.

        Parameters
        ----------
        row_info : Dict[str, Any]
            the Laygo row information dictionary.
        num_blk : int
            number of space blocks.
        adj_end_info : int
            end information ofthe blocks adjacent to this space.

        Returns
        -------
        space_info : Dict[str, Any]
            the space layout information dictionary.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_laygo_connection(cls, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]
        """Draw digital transistor connection in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        blk_type : str
            the digital block type.
        options : Dict[str, Any]
            any additional connection options.

        Returns
        -------
        ext_info : Tuple[Any, ANy]
            a tuple of extension information on top and bottom.
        end_info : Tuple[Any, Any]
            a tuple of the end information on left and right.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_laygo_space_connection(cls, template, space_info, adj_end_info):
        # type: (TemplateBase, Dict[str, Any], Tuple[Any, Any]) -> Tuple[Any, Any]
        """Draw digital transistor connection in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        space_info : Dict[str, Any]
            the layout information dictionary.
        adj_end_info : Tuple[Any, Any]
            tuple of left and right end information object.

        Returns
        -------
        ext_info : Tuple[Any, ANy]
            a tuple of extension information on top and bottom.
        """
        return None, None

    @classmethod
    def get_laygo_conn_track_info(cls, lch_unit):
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
        laygo_conn_w = mos_constants['laygo_conn_w']
        laygo_num_sd_per_track = mos_constants['laygo_num_sd_per_track']
        return sd_pitch * laygo_num_sd_per_track - laygo_conn_w, laygo_conn_w

    @classmethod
    def draw_extensions(cls,  # type: LaygoTech
                        template,  # type: TemplateBase
                        laygo_info,  # type: 'LaygoBaseInfo'
                        w,  # type: int
                        yext,  # type: int
                        bot_ext_list,  # type: List[Tuple[int, Any]]
                        top_ext_list,  # type: List[Tuple[int, Any]]
                        edge_params,  # type: Dict[str, Any]
                        ):
        # type: (...) -> List[Tuple[int, int, str, Dict[str, Any]]]
        """Draw extension rows in the given LaygoBase/DigitalBase template.

        Parameters
        ----------
        template : TemplateBase
            the LaygoBase/DigitalBase object to draw layout in.
        laygo_info : LaygoBaseInfo
            the LaygoBaseInfo object.
        w : int
            extension width in number of mos pitches.
        yext : int
            Y coordinate of the extension block.
        bot_ext_list : List[Tuple[int, Any]]
            list of tuples of end finger index and bottom extension information
        top_ext_list : List[Tuple[int, Any]]
            list of tuples of end finger index and top extension information
        edge_params : Dict[str, Any]
            edge parameters dictionary.

        Returns
        -------
        ext_edges : List[Tuple[int, int, str, Dict[str, Any]]]
            a list of X/Y coordinate, orientation, and parameters for extension edge blocks.
            empty list if draw_boundaries is False.
        """
        lch = laygo_info.lch
        left_margin = laygo_info.left_margin

        cur_col = 0
        bot_idx = top_idx = 0
        bot_len = len(bot_ext_list)
        top_len = len(top_ext_list)
        ext_groups = []
        while bot_idx < bot_len and top_idx < top_len:
            bot_stop, bot_info = bot_ext_list[bot_idx]
            top_stop, top_info = top_ext_list[top_idx]
            if bot_stop < top_stop:
                ext_groups.append((bot_stop - cur_col, bot_info, top_info))
                cur_col = bot_stop
                bot_idx += 1
            else:
                ext_groups.append((top_stop - cur_col, bot_info, top_info))
                cur_col = top_stop
                top_idx += 1
                if bot_stop == top_stop:
                    bot_idx += 1

        curx = left_margin
        for num_col, bot_info, top_info in ext_groups:
            if w > 0 or cls.draw_zero_extension():
                ext_params = dict(
                    lch=lch,
                    w=w,
                    fg=num_col * cls.get_laygo_unit_fg(),
                    top_ext_info=top_info,
                    bot_ext_info=bot_info,
                    is_laygo=True,
                )
                ext_master = template.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                template.add_instance(ext_master, loc=(curx, yext), unit_mode=True)
                curx += ext_master.prim_bound_box.width_unit

        return [(left_margin, yext, 'R0', edge_params),
                (curx + left_margin, yext, 'MY', edge_params)]

    @classmethod
    def draw_boundaries(cls,  # type: LaygoTech
                        template,  # type: TemplateBase
                        laygo_info,  # type: 'LaygoBaseInfo'
                        num_col,  # type: int
                        yt,  # type: int
                        bot_end_master,  # type: 'LaygoEndRow'
                        top_end_master,  # type: 'LaygoEndRow'
                        edge_infos,  # type: List[Tuple[int, int, str, Dict[str, Any]]]
                        ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw boundaries for LaygoBase/DigitalBase.

        Parameters
        ----------
        template : TemplateBase
            the LaygoBase/DigitalBase object to draw layout in.
        laygo_info : LaygoBaseInfo
            the LaygoBaseInfo object.
        num_col : int
            number of primitive columns in the template.
        yt : int
            the top Y coordinate of the template.  Used to determine top end row placement.
        bot_end_master: LaygoEndRow
            the bottom LaygoEndRow master.
        top_end_master : LaygoEndRow
            the top LaygoEndRow master.
        edge_infos:  List[Tuple[int, int, str, Dict[str, Any]]]
            a list of X/Y coordinate, orientation, and parameters for all edge blocks.

        Returns
        -------
        vdd_warrs : List[WireArray]
            any VDD wires in the edge block due to guard ring.
        vss_warrs : List[WireArray]
            any VSS wires in the edge block due to guard ring.
        """
        end_mode = laygo_info.end_mode
        top_layer = laygo_info.top_layer
        guard_ring_nf = laygo_info.guard_ring_nf
        col_width = laygo_info.col_width
        left_margin = laygo_info.left_margin
        right_margin = laygo_info.right_margin

        nx = num_col
        spx = col_width

        # draw top and bottom end row
        template.add_instance(bot_end_master, inst_name='XRBOT', loc=(left_margin, 0),
                              nx=nx, spx=spx, unit_mode=True)
        template.add_instance(top_end_master, inst_name='XRBOT', loc=(left_margin, yt),
                              orient='MX', nx=nx, spx=spx, unit_mode=True)
        # draw corners
        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0
        edge_inst_list = []
        xr = left_margin + col_width * nx + right_margin
        for orient, y, master in (('R0', 0, bot_end_master), ('MX', yt, top_end_master)):
            for x, is_end, flip_lr in ((0, left_end, False), (xr, right_end, True)):
                edge_params = dict(
                    top_layer=top_layer,
                    is_end=is_end,
                    guard_ring_nf=guard_ring_nf,
                    name_id=master.get_layout_basename(),
                    layout_info=master.get_edge_layout_info(),
                    is_laygo=True,
                )
                edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
                if flip_lr:
                    eorient = 'MY' if orient == 'R0' else 'R180'
                else:
                    eorient = orient
                edge_inst_list.append(template.add_instance(edge_master, orient=eorient, loc=(x, y), unit_mode=True))

        # draw edge blocks
        for x, y, orient, edge_params in edge_infos:
            edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
            edge_inst_list.append(template.add_instance(edge_master, orient=orient, loc=(x, y), unit_mode=True))

        gr_vss_warrs = []
        gr_vdd_warrs = []
        conn_layer = cls.get_dig_conn_layer()
        for inst in edge_inst_list:
            if inst.has_port('VDD'):
                gr_vdd_warrs.extend(inst.get_all_port_pins('VDD', layer=conn_layer))
            elif inst.has_port('VSS'):
                gr_vss_warrs.extend(inst.get_all_port_pins('VSS', layer=conn_layer))

        # connect body guard rings together
        gr_vdd_warrs = template.connect_wires(gr_vdd_warrs)
        gr_vss_warrs = template.connect_wires(gr_vss_warrs)

        return gr_vdd_warrs, gr_vss_warrs
