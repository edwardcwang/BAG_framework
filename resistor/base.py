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
import math
from typing import Dict, Set, Tuple, Any, List, Optional

from bag import float_to_si_string
from bag.math import lcm
from bag.util.search import BinaryIterator
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import RoutingGrid


class ResTech(with_metaclass(abc.ABCMeta, object)):
    @classmethod
    @abc.abstractmethod
    def get_bot_layer(cls):
        # type: () -> int
        """Returns the layer ID of the bottom horizontal routing layer.
        
        Returns
        -------
        layer_id : int
            the bottom horizontal routing layer ID.
        """
        return 2

    @classmethod
    @abc.abstractmethod
    def get_res_density(cls):
        # type: () -> float
        """Returns the maximum resistor density, as a floating point number between 0 and 1.
        
        Returns
        -------
        density : float
            the maximum resistor density.
        """
        return 0.5

    @classmethod
    @abc.abstractmethod
    def get_block_pitch(cls):
        # type: () -> Tuple[int, int]
        """Returns the horizontal/vertical block pitch of the resistor core in resolution units.  
        
        The vertical block pitch is usually the fin pitch.
        
        Returns
        -------
        x_pitch : int
            the horizontal block pitch, in resolution units.
        y_pitch : int
            the vertical block pitch, in resolution units.
        """
        return 1, 1

    @classmethod
    @abc.abstractmethod
    def get_min_res_core_size(cls, l, w):
        # type: (int, int) -> Tuple[int, int, Tuple[int, int, int]]
        """Calculate the minimum size of a resistor core based on DRC rules.

        Parameters
        ----------
        l : int
            length of resistor in resolution units.
        w : int
            width of resistor in resolution units.

        Returns
        -------
        wres : int
            minimum width of resistor core from DRC rules.
        hres : int
            minimum height of resistor core from DRC rules.
        ares : Tuple[int, int, int]
            resistor area in core/lr_edge/tb_edge blocks, in resolution units squared.  Used for density
            DRC rule.
        """
        return 1, 1, (1, 1, 1)

    @classmethod
    @abc.abstractmethod
    def get_edge_size(cls, wblk, hblk, nxcore, nycore, nxlr, nytb, l, w):
        # type: (int, int, int, int, int, int, int, int) -> Tuple[int, int]
        """Calculate the width and height of boundary cells.

        Parameters
        ----------
        wblk : int
            width quantization, in resolution units.
        hblk : int
            height quantization, in resolution units.
        nxcore : int
            number of horizontal blocks in resistor core.
        nycore : int
            number of vertical blocks in resistor core.
        nxlr : int
            minimum number of horizontal blocks in left/right edge.
        nytb : int
            minimum number of vertical blocks in top/bottom edge.
        l : int
            length of resistor in resolution units.
        w : int
            width of resistor in resolution units.

        Returns
        -------
        wedge : int
            width of left/right edge, in resolution units.
        hedge : int
            width of top/bottom edge, in resolution units.
        """
        return 1, 1

    @classmethod
    @abc.abstractmethod
    def update_layout_info(cls, layout_info):
        # type: (Dict[str, Any]) -> None
        """Given the layout information dictionary, compute and add extra entries if needed.
        
        Parameters
        ----------
        layout_info : Dict[str, Any]
            dictionary containing block dimensions.  Add new entries to this dictionary
            if they are needed by draw_res_core() or draw_res_boundary().
        """

    @classmethod
    @abc.abstractmethod
    def draw_res_core(cls, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw the resistor core in the given template.
        
        Parameters
        ----------
        template : TemplateBase
            the template to draw the resistor core in.
        layout_info : Dict[str, Any]
            the resistor layout information dictionary.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def draw_res_boundary(cls, template, boundary_type, layout_info):
        # type: (TemplateBase, str, Dict[str, Any]) -> None
        """Draw the resistor left/right edge in the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the resistor edge in.
        boundary_type : str
            the resistor boundary type.  One of 'lr', 'tb', or 'corner'.
        layout_info : Dict[str, Any]
            the resistor layout information dictionary.
        """
        pass

    @classmethod
    def fill_symmetric(cls, tot_space, n_min, n_max, sp):
        """Compute 1-D fill pattern that maximizes the filled region and symmetric about the center.
        
        Given an empty space and fill parameters, determine location and size of each fill block
        such that we maximize filled region but keep the pattern symmetric about the center.  This method
        is useful for computing fill pattern inside the resistor core.
        
        Parameters
        ----------
        tot_space : int
            total number of space we need to fill.
        n_min : int
            minimum length of the fill block.
        n_max : int
            maximum length of the fill block.
        sp : int
            minimum space between each fill block.
            
        Returns
        -------
        num_filled : int
            total number of units filled
        fill_interval : List[Tuple[int, int]]
            a list of [start, end) intervals that needs to be filled.
        """
        if tot_space <= 0:
            return 0, []

        # dynamic programming
        soln_vec = [(0, [])] * n_min + [(i, [(0, i)]) for i in range(n_min, n_max + 1)]
        for i in range(n_max + 1, tot_space + 1):
            # try using one block
            opt_n = n_max if (i - n_max) % 2 == 0 else n_max - 1
            start = (i - opt_n) // 2
            opt_combo = [(start, start + opt_n)]
            # try using two blocks
            sp_sep = sp if (i - sp) % 2 == 0 else sp + 1
            n2 = min((i - sp_sep) // 2, n_max)
            if n2 >= n_min and n2 * 2 > opt_n:
                # using two blocks is better than using one block, update maximum
                opt_n = n2 * 2
                opt_combo = [(0, n2), (i - n2, i)]
            # try using three blocks
            for n_end in range(n_min, n_max + 1):
                remainder = i - 2 * (n_end + sp)
                if remainder >= n_min:
                    recur_n, recur_combo = soln_vec[remainder]
                    if recur_n + 2 * n_end > opt_n:
                        # found new optimum.  update
                        opt_n = recur_n + 2 * n_end
                        opt_combo = [(0, n_end)]
                        delta = n_end + sp
                        for sidx, eidx in recur_combo:
                            opt_combo.append((sidx + delta, eidx + delta))
                        opt_combo.append((i - n_end, i))
                else:
                    break

            # record best solution.
            soln_vec.append((opt_n, opt_combo))

        # return best combination
        return soln_vec[tot_space]

    @classmethod
    def get_core_track_info(cls,  # type: ResTech
                            grid,  # type: RoutingGrid
                            min_tracks,  # type: Tuple[int, ...]
                            em_specs  # type: Dict[str, Any]
                            ):
        # type: (...) -> Tuple[List[int], List[int], Tuple[int, int], Tuple[int, int]]
        """Calculate resistor core size/track information based on given specs.

        This method calculate the track width/spacing on each routing layer from EM specifications,
        then compute minimum width/height and block pitch of resistor blocks from given
        constraints.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        min_tracks : List[int]
            minimum number of tracks on each layer.
        em_specs : Dict[str, Any]
            EM specification dictionary.

        Returns
        -------
        track_widths : List[int]
            the track width on each layer that satisfies EM specs.
        track_spaces : List[int]
            the track space on each layer.
        min_size : Tuple[int, int]
            a tuple of minimum width and height of the core in resolution units.
        blk_pitch : Tuple[int, int]
            a tuple of width and height pitch of the core in resolution units.
        """
        track_widths = []
        track_spaces = []
        prev_width = -1
        min_w = min_h = 0
        cur_layer = cls.get_bot_layer()
        for min_num_tr in min_tracks:
            tr_w, tr_sp = grid.get_track_info(cur_layer, unit_mode=True)
            cur_width = grid.get_min_track_width(cur_layer, bot_w=prev_width, unit_mode=True, **em_specs)
            cur_space = grid.get_num_space_tracks(cur_layer, cur_width)
            track_widths.append(cur_width)
            track_spaces.append(cur_space)
            cur_width_lay = cur_width * tr_w + (cur_width - 1) * tr_sp
            cur_space_lay = (cur_space + 1) * tr_sp + cur_space * tr_w
            min_dim = min_num_tr * (cur_width_lay + cur_space_lay)
            if grid.get_direction(cur_layer) == 'x':
                min_h = max(min_h, min_dim)
            else:
                min_w = max(min_w, min_dim)

            cur_layer += 1
            prev_width = cur_width_lay

        cur_layer -= 1
        wblk, hblk = grid.get_block_size(cur_layer, unit_mode=True)
        rwblk, rhblk = cls.get_block_pitch()
        wblk = lcm([wblk, rwblk])
        hblk = lcm([hblk, rhblk])
        min_w = -(-min_w // wblk) * wblk
        min_h = -(-min_h // hblk) * hblk

        return track_widths, track_spaces, (min_w, min_h), (wblk, hblk)

    @classmethod
    def check_density_rule_core(cls, wres, hres, wblk, hblk, ares, ext_dir):
        # type: (int, int, int, int, int, Optional[str]) -> Tuple[int, int]
        """Compute resistor core size that meets resistor density DRC rule.
        
        Given current resistor block size and the block pitch, increase the resistor block
        size if necessary to meet density DRC rule.
        
        Parameters
        ----------
        wres : int
            resistor core width, in resolution units.
        hres : int
            resistor core height, in resolution units.
        wblk : int
            the horizontal block pitch, in resolution units.
        hblk : int
            the vertical block pitch, in resolution units.
        ares : int
            the resistor area inside the resistor core that should be used for density rule
            calculation.  In resolution units squared.
        ext_dir : Optional[str]
            if equal to 'x', then we will only stretch the resistor core horizontally.  If equal
            to 'y', we will only stretch the resistor core vertically.  Otherwise, we will find
            the resistor core with the minimum area that meets the density spec.
        
        Returns
        -------
        nxblk : int
            width of the resistor core, in units of wblk.
        nyblk : int
            height of the resistor core, in units of hblk.
        """
        nxblk = wres // wblk
        nyblk = hres // hblk
        density = cls.get_res_density()

        # convert to float so we're doing floating point comparison
        ares = float(ares)
        if ares < wres * hres * density:
            return nxblk, nyblk

        # find maximum horizontal/vertical extension that meets DRC rule.
        nxblk_max = int(math.ceil(ares / density / hres / wblk))
        nyblk_max = int(math.ceil(ares / density / wres / hblk))
        # if we're only extending in one direction, just return
        if ext_dir == 'x':
            return nxblk_max, nyblk
        elif ext_dir == 'y':
            return nxblk, nyblk_max

        # otherwise, find extension that minimizes area of the core.
        # this is a simple exhaustive search.
        opt = nxblk, nyblk_max
        opt_area = nxblk * nyblk_max * wblk * hblk
        for nxblk_cur in range(nxblk + 1, nxblk_max + 1):
            wcur = nxblk_cur * wblk
            for nyblk_cur in range(nyblk, nyblk_max + 1):
                acur = wcur * nyblk_cur * hblk
                if acur >= opt_area:
                    break
                if ares < acur * density:
                    opt_area = acur
                    opt = nxblk_cur, nyblk_cur

        return opt

    @classmethod
    def check_density_rule_edge(cls, n0, s0, s1, area):
        # type: (int, int, int, int) -> int
        """Compute edge block dimension from density spec.

        Given edge width or height (as dimension 0), find the missing dimension (dimension 1)
        such that density rule is met.

        Parameters
        ----------
        n0 : int
            edge length in dimension 0 as number of blocks.
        s0 : int
            dimension 0 block length in resolution units.
        s1 : int
            dimension 1 block length in resolution units.
        area : int
            the resistor area in the edge block that should be used for density spec.
            In resolution units squared.

        Returns
        -------
        n1 : int
            edge length in dimension 1 as number of blocks.
        """
        density = cls.get_res_density()
        # convert to float so we're doing floating point comparison
        area = float(area)

        bin_iter = BinaryIterator(1, None)
        a0 = n0 * s0 * s1
        while bin_iter.has_next():
            n1 = bin_iter.get_next()
            if area <= a0 * n1 * density:
                bin_iter.save()
                bin_iter.down()
            else:
                bin_iter.up()

        return bin_iter.get_last_save()

    @classmethod
    def get_res_info(cls, grid, l, w, res_type, sub_type, threshold, min_tracks, em_specs, ext_dir):
        # type: (RoutingGrid, int, int, str, str, str, Tuple[int, ...], Dict[str, Any], Optional[str]) -> Dict[str, Any]
        """Compute the resistor layout information dictionary.
        
        This method compute the width/height of each resistor primitive block and also the
        track width and space on each routing layer, then return the result as a dictionary.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        l : int
            length of the resistor, in resolution units.
        w : int
            width of the resistor, in resolution units.
        res_type : str
            the resistor type.
        sub_type : str
            the resistor substrate type.
        threshold : str
            the substrate threshold flavor.
        min_tracks : Tuple[int, ...]
            list of minimum number of tracks in each routing layer.
        em_specs : Dict[str, Any]
            the EM specification dictionary.
        ext_dir : Optional[str]
            if equal to 'x', then we will only stretch the resistor core horizontally.  If equal
            to 'y', we will only stretch the resistor core vertically.  Otherwise, we will find
            the resistor core with the minimum area that meets the density spec.

        Returns
        -------
        res_info : Dict[str, Any]
            the resistor layout information dictionary.
        
        """
        # step 1: get track/size parameters
        track_widths, track_spaces, min_size, blk_pitch = cls.get_core_track_info(grid, min_tracks, em_specs)
        # step 2: get minimum DRC size, then update and round to block size.
        wres, hres, ares = cls.get_min_res_core_size(l, w)
        wres = max(wres, min_size[0])
        hres = max(hres, min_size[1])
        wblk, hblk = blk_pitch
        wres = -(-wres // wblk) * wblk
        hres = -(-hres // hblk) * hblk
        # step 3: extend core until density rule is satisfied.
        nxblk, nyblk = cls.check_density_rule_core(wres, hres, wblk, hblk, ares[0], ext_dir)
        # step 4: calculate edge size that satisfies density rule.
        nxblk_lr = cls.check_density_rule_edge(nyblk, hblk, wblk, ares[1])
        nyblk_tb = cls.check_density_rule_edge(nxblk, wblk, hblk, ares[2])

        # step 5: get final edge sizes
        wedge, hedge = cls.get_edge_size(wblk, hblk, nxblk, nyblk, nxblk_lr, nyblk_tb, l, w)
        wcore, hcore = nxblk * wblk, nyblk * hblk

        # step 6: calculate geometry information of each primitive block.
        bot_layer = cls.get_bot_layer()
        num_tracks = []
        num_corner_tracks = []
        for lay in range(bot_layer, bot_layer + len(min_tracks)):
            if grid.get_direction(lay) == 'y':
                dim = wcore
                dim_corner = wedge
            else:
                dim = hcore
                dim_corner = hedge

            pitch = grid.get_track_pitch(lay, unit_mode=True)
            num_tracks.append(dim // pitch)
            num_corner_tracks.append(dim_corner // pitch)

        res_info = dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
            w_core=wcore,
            h_core=hcore,
            w_edge=wedge,
            h_edge=hedge,
            track_widths=track_widths,
            track_spaces=track_spaces,
            num_tracks=num_tracks,
            num_corner_tracks=num_corner_tracks,
        )

        cls.update_layout_info(res_info)
        return res_info


class AnalogResCore(TemplateBase):
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
        self._tech_cls = self.grid.tech_info.tech_params['layout']['res_tech_class']  # type: ResTech
        self._num_tracks = None
        self._track_widths = None
        self._layout_info = None

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
            em_specs={},
            ext_dir='',
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
            res_type='the resistor type.',
            sub_type='the substrate type.',
            threshold='substrate threshold flavor.',
            min_tracks='Minimum number of tracks on each layer per block.',
            em_specs='resistor EM spec specifications.',
            ext_dir='resistor core extension direction.',
            flip_parity='the flip track parity dictionary.',
        )

    def get_num_tracks(self):
        # type: () -> Tuple[int, int, int, int]
        """Returns a list of the number of tracks on each routing layer in this template.

        Returns
        -------
        ntr_list : Tuple[int, int, int, int]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return self._num_tracks

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
        return self._track_widths

    def get_boundary_params(self, boundary_type):
        # type: (str) -> Dict[str, Any]
        """Returns boundary parameters dictioanry."""
        return dict(
            boundary_type=boundary_type,
            layout_id=self.get_name_id(),
            layout_info=self._layout_info,
        )

    def get_name_id(self):
        # type: () -> str
        """Returns a string identifier representing this resistor core."""
        l_str = float_to_si_string(self.params['l'])
        w_str = float_to_si_string(self.params['w'])
        res_type = self.params['res_type']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        ext_dir = self.params['ext_dir']
        main = '%s_%s_%s_l%s_w%s' % (res_type, sub_type, threshold, l_str, w_str)
        if ext_dir:
            main += '_ext%s' % ext_dir
        return main

    def get_layout_basename(self):
        # type: () -> str
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        return 'rescore_' + self.get_name_id()

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        min_tracks = self.params['min_tracks']
        flip_parity = self.params['flip_parity']
        em_specs = self.params['em_specs']
        return self.to_immutable_id((basename, min_tracks, em_specs, flip_parity))

    def draw_layout(self):
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, l, w, res_type, sub_type, threshold, min_tracks, em_specs, ext_dir, flip_parity):
        self.grid = self.grid.copy()
        self.grid.set_flip_parity(flip_parity)

        res = self.grid.resolution
        lay_unit = self.grid.layout_unit
        l_unit = int(round(l / lay_unit / res))
        w_unit = int(round(w / lay_unit / res))

        self._layout_info = self._tech_cls.get_res_info(self.grid, l_unit, w_unit, res_type, sub_type, threshold,
                                                        min_tracks, em_specs, ext_dir)
        self._num_tracks = self._layout_info['num_tracks']
        self._track_widths = self._layout_info['track_widths']
        self._tech_cls.draw_res_core(self, self._layout_info)


class AnalogResBoundary(TemplateBase):
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
        super(AnalogResBoundary, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['res_tech_class']  # type: ResTech

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
            boundary_type='resistor boundary type.',
            layout_id='the layout ID',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        # type: () -> str
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        bound_type = self.params['boundary_type']
        if bound_type == 'lr':
            prefix = 'resedgelr'
        elif bound_type == 'tb':
            prefix = 'resedgetb'
        else:
            prefix = 'rescorner'
        return '%s_%s' % (prefix, self.params['layout_id'])

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return self.to_immutable_id((basename, self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_res_boundary(self, self.params['boundary_type'], self.params['layout_info'])
