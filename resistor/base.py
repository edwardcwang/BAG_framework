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
    def get_min_res_core_size(cls, l, w, res_type, sub_type, threshold):
        # type: (int, int, str, str, str) -> Tuple[int, int]
        """Calculate the minimum size of a resistor core based on DRC rules.

        This function usually calculates the minimum size based on spacing rules and not density rules.
        density rule calculations are usually handled in get_core_info().

        Parameters
        ----------
        l : int
            resistor length in resolution units.
        w : int
            resistor width in resolution units
        res_type : str
            resistor type.
        sub_type : str
            substrate type.
        threshold : str
            threshold type.

        Returns
        -------
        wcore : int
            minimum width of resistor core from DRC rules.
        hcore : int
            minimum height of resistor core from DRC rules.
        """
        return 1, 1

    @classmethod
    @abc.abstractmethod
    def get_core_info(cls, width, height, l, w, res_type, sub_type, threshold):
        # type: (int, int, int, int, str, str, str) -> Optional[Dict[str, Any]]
        """Returns a dictionary of core layout information.

        If the given core size does not meet DRC rules, return None.

        Parameters
        ----------
        width : int
            the width of core block in resolution units.
        height : int
            the height tof core block in resolution units.
        l : int
            resistor length in resolution units.
        w : int
            resistor width in resolution units
        res_type : str
            resistor type.
        sub_type : str
            substrate type.
        threshold : str
            threshold type.

        Returns
        -------
        layout_info : Optional[Dict[str, Any]]
            the core layout information dictionary.
        """
        return None

    @classmethod
    @abc.abstractmethod
    def get_edge_info(cls, width, height, is_lr_edge, l, w, res_type, sub_type, threshold):
        # type: (int, int, bool, int, int, str, str, str) -> Optional[Dict[str, Any]]
        """Returns a dictionary of edge layout information.

        If the given edge size does not meet DRC rules, return None.

        Parameters
        ----------
        width : int
            the width of edge block in resolution units.
        height : int
            the height tof edge block in resolution units.
        is_lr_edge : bool
            True if this is left/right edge, False if this is top/bottom edge.
        l : int
            resistor length in resolution units.
        w : int
            resistor width in resolution units
        res_type : str
            resistor type.
        sub_type : str
            substrate type.
        threshold : str
            threshold type.

        Returns
        -------
        layout_info : Optional[Dict[str, Any]]
            the edge layout information dictionary.
        """
        return None

    @classmethod
    @abc.abstractmethod
    def update_layout_info(cls, grid, layout_info):
        # type: (RoutingGrid, Dict[str, Any]) -> None
        """Given the layout information dictionary, compute and add extra entries if needed.
        
        Parameters
        ----------
        grid : RoutingGrid
            the routing grid object.
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
    def draw_res_boundary(cls, template, boundary_type, layout_info, end_mode):
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
        end_mode : bool
            True to extend well layers to bottom.
        """
        pass

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
    def find_core_size(cls, params, wres, hres, wblk, hblk, ext_dir, max_blk_dim):
        # type: (Dict[str, Any], int, int, int, int, str, int) -> Tuple[int, int, Dict[str, Any]]
        """Compute resistor core size that meets DRC rules.
        
        Given current resistor block size and the block pitch, increase the resistor block
        size if necessary to meet DRC rules.
        
        Parameters
        ----------
        params : Dict[str, Any]
            the resistor parameters dictionary.
        wres : int
            resistor core width, in resolution units.
        hres : int
            resistor core height, in resolution units.
        wblk : int
            the horizontal block pitch, in resolution units.
        hblk : int
            the vertical block pitch, in resolution units.
        ext_dir : Optional[str]
            if equal to 'x', then we will only stretch the resistor core horizontally.  If equal
            to 'y', we will only stretch the resistor core vertically.  Otherwise, we will find
            the resistor core with the minimum area that meets the density spec.
        max_blk_dim : int
            number of block pitches we can extend the resistor core size by.  If we cannot
            find a valid core size by extending this many block pitches, we declare failure.
        
        Returns
        -------
        nxblk : int
            width of the resistor core, in units of wblk.
        nyblk : int
            height of the resistor core, in units of hblk.
        layout_info : Dict[str, Any]
            the core layout information dictionary.
        """
        nxblk = wres // wblk
        nyblk = hres // hblk

        ans = None
        x_only = (ext_dir == 'x')
        if x_only or (ext_dir == 'y'):
            # only extend X or Y direction
            if x_only:
                bin_iter = BinaryIterator(nxblk, nxblk + max_blk_dim + 1)
            else:
                bin_iter = BinaryIterator(nyblk, nyblk + max_blk_dim + 1)
            while bin_iter.has_next():
                ncur = bin_iter.get_next()
                if x_only:
                    wcur, hcur = ncur * wblk, hres
                else:
                    wcur, hcur = wres, ncur * hblk
                tmp = cls.get_core_info(wcur, hcur, **params)
                if tmp is None:
                    bin_iter.up()
                else:
                    ans = tmp
                    bin_iter.save()
                    bin_iter.down()

            if ans is None:
                raise ValueError('failed to find DRC clean core with maximum %d '
                                 'additional block pitches.' % max_blk_dim)
            if x_only:
                nxblk = bin_iter.get_last_save()
            else:
                nyblk = bin_iter.get_last_save()
            return nxblk, nyblk, ans
        else:
            # extend in both direction
            opt_area = (nxblk + max_blk_dim + 1) * (nyblk + max_blk_dim + 1)
            # linear search in height, binary search in width
            # in this way, for same area, use height as tie breaker
            nxopt, nyopt = nxblk, nyblk
            for nycur in range(nyblk, nyblk + max_blk_dim + 1):
                # check if we should terminate linear search
                if nycur * nxblk >= opt_area:
                    break
                bin_iter = BinaryIterator(nxblk, nxblk + max_blk_dim + 1)
                hcur = nycur * hblk
                while bin_iter.has_next():
                    nxcur = bin_iter.get_next()
                    if nxcur * nycur >= opt_area:
                        # this point can't beat current optimum
                        bin_iter.down()
                    else:
                        tmp = cls.get_core_info(nxcur * wblk, hcur, **params)
                        if tmp is None:
                            bin_iter.up()
                        else:
                            # found new optimum
                            ans, nxopt, nyopt = tmp, nxcur, nycur
                            opt_area = nxcur * nycur
                            bin_iter.down()

            if ans is None:
                raise ValueError('failed to find DRC clean core with maximum %d '
                                 'additional block pitches.' % max_blk_dim)
            return nxopt, nyopt, ans

    @classmethod
    def find_edge_size(cls, is_lr_edge, params, dim0, blk1, max_blk_dim):
        # type: (bool, Dict[str, Any], int, int, int) -> Tuple[int, Dict[str, Any]]
        """Compute resistor edge size that meets DRC rules.

        Given edge width or height (as dimension 0), extend the other dimension (dimension 1)
        until we meet DRC rules

        Parameters
        ----------
        is_lr_edge : bool
            True if this is left/right edge, False if this is top/bottom edge.
        params : Dict[str, Any]
            the resistor parameters dictionary.
        dim0 : int
            length of dimension 0 in resolution units.
        blk1 : int
            dimension1 block size in resolution units.
        max_blk_dim : int
            maximum number of blocks we can extend by.

        Returns
        -------
        n1 : int
            edge length in dimension 1 as number of blocks.
        layout_info : Dict[str, Any]
            the edge layout information dictionary.
        """

        bin_iter = BinaryIterator(1, max_blk_dim + 2)
        ans = None
        while bin_iter.has_next():
            n1 = bin_iter.get_next()
            if is_lr_edge:
                wcur, hcur = n1 * blk1, dim0
            else:
                wcur, hcur = dim0, n1 * blk1

            tmp = cls.get_edge_info(wcur, hcur, is_lr_edge, **params)
            if tmp is None:
                bin_iter.up()
            else:
                ans = tmp
                bin_iter.save()
                bin_iter.down()

        if ans is None:
            raise ValueError('failed to find DRC clean core with maximum %d '
                             'additional block pitches.' % max_blk_dim)

        return bin_iter.get_last_save(), ans

    @classmethod
    def get_res_info(cls,
                     grid,  # type: RoutingGrid
                     l,  # type: int
                     w,  # type: int
                     res_type,  # type: str
                     sub_type,  # type: str
                     threshold,  # type: str
                     min_tracks,  # type: Tuple[int, ...]
                     em_specs,  # type: Dict[str, Any]
                     ext_dir,  # type: Optional[str]
                     max_blk_dim=100,  # type: int
                     ):
        # type: (...) -> Dict[str, Any]
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
        max_blk_dim : int
            number of block pitches we can extend the resistor core/edge size by.  If we cannot
            find a valid core size by extending this many block pitches, we declare failure.

        Returns
        -------
        res_info : Dict[str, Any]
            the resistor layout information dictionary.
        
        """
        params = dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
        )
        # step 1: get track/size parameters
        track_widths, track_spaces, min_size, blk_pitch = cls.get_core_track_info(grid, min_tracks, em_specs)
        # step 2: get minimum DRC core size, then update with minimum size and round to block size.
        wres, hres = cls.get_min_res_core_size(**params)
        wres = max(wres, min_size[0])
        hres = max(hres, min_size[1])
        wblk, hblk = blk_pitch
        wres = -(-wres // wblk) * wblk
        hres = -(-hres // hblk) * hblk
        # step 3: extend core until density rule is satisfied.
        nxblk, nyblk, core_info = cls.find_core_size(params, wres, hres, wblk, hblk, ext_dir, max_blk_dim)
        wcore, hcore = nxblk * wblk, nyblk * hblk
        # step 4: calculate edge size that satisfies density rule.
        nxblk_lr, edge_lr_info = cls.find_edge_size(True, params, hcore, wblk, max_blk_dim)
        nyblk_tb, edge_tb_info = cls.find_edge_size(False, params, wcore, hblk, max_blk_dim)
        wedge, hedge = nxblk_lr * wblk, nyblk_tb * hblk

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

        cls.update_layout_info(grid, res_info)
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

    def get_boundary_params(self, boundary_type, end_mode=False):
        # type: (str, bool) -> Dict[str, Any]
        """Returns boundary parameters dictioanry."""
        return dict(
            boundary_type=boundary_type,
            layout_id=self.get_name_id(),
            layout_info=self._layout_info,
            end_mode=end_mode,
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
        flip_parity = self.grid.get_flip_parity()
        em_specs = self.params['em_specs']
        return self.to_immutable_id((basename, min_tracks, em_specs, flip_parity))

    def draw_layout(self):
        l = self.params['l']
        w = self.params['w']
        res_type = self.params['res_type']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        min_tracks = self.params['min_tracks']
        em_specs = self.params['em_specs']
        ext_dir = self.params['ext_dir']

        res = self.grid.resolution
        lay_unit = self.grid.layout_unit
        l_unit = int(round(l / lay_unit / res))
        w_unit = int(round(w / lay_unit / res))

        self._layout_info = self._tech_cls.get_res_info(self.grid, l_unit, w_unit, res_type, sub_type, threshold,
                                                        min_tracks, em_specs, ext_dir)
        self._num_tracks = self._layout_info['num_tracks']
        self._track_widths = self._layout_info['track_widths']
        self._tech_cls.draw_res_core(self, self._layout_info)
        self.prim_top_layer = self._tech_cls.get_bot_layer()


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
        self._well_xl = self.params['layout_info']['well_xl']

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
            end_mode='integer flag indicating whether to extend well layers to bottom.',
        )

    def get_well_left(self, unit_mode=False):
        if unit_mode:
            return self._well_xl
        return self._well_xl * self.grid.resolution

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
        base = '%s_%s' % (prefix, self.params['layout_id'])
        if self.params['end_mode']:
            base += '_end'
        return base

    def compute_unique_key(self):
        basename = self.get_layout_basename()
        return self.to_immutable_id((basename, self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_res_boundary(self, self.params['boundary_type'], self.params['layout_info'],
                                         self.params['end_mode'])
        self.prim_top_layer = self._tech_cls.get_bot_layer()
