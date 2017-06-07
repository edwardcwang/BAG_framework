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

from typing import Dict, Any, Tuple

from bag.layout.template import TemplateBase

import abc

from ..analog_mos.core import MOSTech


class LaygoTech(with_metaclass(abc.ABCMeta, MOSTech)):
    """An abstract static class for drawing transistor related layout.

    This class defines various static methods use to draw layouts used by AnalogBase.
    """

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
    def get_laygo_edge_info(cls, blk_info, endr):
        # type: (Dict[str, Any], bool) -> Dict[str, Any]
        """Returns a new layout information dictionary for drawing LaygoBase edge blocks.

        Parameters
        ----------
        blk_info : Dict[str, Any]
            the layout information dictionary.
        endr : bool
            True if the edge block abuts OD on the right.

        Returns
        -------
        edge_info : Dict[str, Any]
            the edge layout information dictionary.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_laygo_space_info(cls, row_info, num_blk):
        # type: (Dict[str, Any], int) -> Dict[str, Any]
        """Returns a new layout information dictionary for drawing LaygoBase space blocks.

        Parameters
        ----------
        row_info : Dict[str, Any]
            the Laygo row information dictionary.
        num_blk : int
            number of space blocks.

        Returns
        -------
        space_info : Dict[str, Any]
            the space layout information dictionary.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_laygo_connection(cls, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None
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
            any extra connection options.
        """
        pass

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
