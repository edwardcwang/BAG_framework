# -*- coding: utf-8 -*-

"""This module defines abstract analog resistor array component classes.
"""

from typing import TYPE_CHECKING, Dict, Set, Tuple, Any, List, Optional, Union

import abc

from bag import float_to_si_string
from bag.math import lcm
from bag.util.search import BinaryIterator
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import RoutingGrid

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig


class ResTech(object, metaclass=abc.ABCMeta):
    """An abstract class for drawing resistor related layout.

    This class defines various methods use to draw layouts used by ResArrayBase.

    Parameters
    ----------
    config : Dict[str, Any]
        the technology configuration dictionary.
    tech_info : TechInfo
        the TechInfo object.
    """

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        self.config = config
        self.res_config = self.config['resistor']
        self.res = self.config['resolution']
        self.tech_info = tech_info

    @abc.abstractmethod
    def get_min_res_core_size(self, l, w, res_type, sub_type, threshold, options):
        # type: (int, int, str, str, str, Dict[str, Any]) -> Tuple[int, int]
        """Calculate the minimum size of a resistor core based on DRC rules.

        This function usually calculates the minimum size based on spacing rules and not density
        rules. density rule calculations are usually handled in get_core_info().

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
        options : Dict[str, Any]
            optional parameter values.

        Returns
        -------
        wcore : int
            minimum width of resistor core from DRC rules.
        hcore : int
            minimum height of resistor core from DRC rules.
        """
        return 1, 1

    @abc.abstractmethod
    def get_core_info(self,
                      grid,  # type: RoutingGrid
                      width,  # type: int
                      height,  # type: int
                      l,  # type: int
                      w,  # type: int
                      res_type,  # type: str
                      sub_type,  # type: str
                      threshold,  # type: str
                      track_widths,  # type: List[int]
                      track_spaces,  # type: List[Union[float, int]]
                      options,  # type: Dict[str, Any]
                      ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Returns a dictionary of core layout information.

        If the given core size does not meet DRC rules, return None.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
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
        track_widths : List[int]
            the track widths on each layer to meet EM specs.
        track_spaces : List[Union[float, int]]
            the track spaces on each layer.
        options : Dict[str, Any]
            optional parameter values.

        Returns
        -------
        layout_info : Optional[Dict[str, Any]]
            the core layout information dictionary.
        """
        return None

    @abc.abstractmethod
    def get_lr_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         wedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Returns a dictionary of LR edge layout information.

        If the given edge size does not meet DRC rules, return None.

        Parameters
        ----------
        grid: RoutingGrid
            the RoutingGrid object.
        core_info : Dict[str, Any]
            core layout information dictionary.
        wedge : int
            LR edge width in resolution units.
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
        track_widths : List[int]
            the track widths on each layer to meet EM specs.
        track_spaces : List[Union[float, int]]
            the track spaces on each layer.
        options : Dict[str, Any]
            optional parameter values.

        Returns
        -------
        layout_info : Optional[Dict[str, Any]]
            the edge layout information dictionary.
        """
        return None

    @abc.abstractmethod
    def get_tb_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         hedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        """Returns a dictionary of TB edge layout information.

        If the given edge size does not meet DRC rules, return None.

        Parameters
        ----------
        grid: RoutingGrid
            the RoutingGrid object.
        core_info : Dict[str, Any]
            core layout information dictionary.
        hedge : int
            TB edge height in resolution units.
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
        track_widths : List[int]
            the track widths on each layer to meet EM specs.
        track_spaces : List[Union[float, int]]
            the track spaces on each layer.
        options : Dict[str, Any]
            optional parameter values.

        Returns
        -------
        layout_info : Optional[Dict[str, Any]]
            the edge layout information dictionary.
        """
        return None

    @abc.abstractmethod
    def draw_res_core(self, template, layout_info):
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

    @abc.abstractmethod
    def draw_res_boundary(self, template, boundary_type, layout_info, end_mode):
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

    def get_res_imp_layers(self, res_type, sub_type, threshold):
        # type: (str, str, str) -> List[Tuple[str, str]]
        """Returns a list of resistor implant layers.

        Parameters
        ----------
        res_type : str
            the resistor type.
        sub_type : str
            the resistor substrate type.
        threshold : str
            the threshold flavor.

        Returns
        -------
        imp_list : List[Tuple[str, str]]
            a list of implant layers.
        """
        imp_layers = self.tech_info.get_implant_layers(sub_type, res_type=res_type)
        imp_layers.extend(self.res_config['res_layers'][res_type].keys())
        imp_layers.extend(self.res_config['thres_layers'][sub_type][threshold].keys())
        return imp_layers

    def get_bot_layer(self):
        # type: () -> int
        """Returns the layer ID of the bottom horizontal routing layer.

        Returns
        -------
        layer_id : int
            the bottom horizontal routing layer ID.
        """
        return self.res_config['bot_layer']

    def get_block_pitch(self):
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
        return self.res_config['block_pitch']

    def get_core_track_info(self,  # type: ResTech
                            grid,  # type: RoutingGrid
                            min_tracks,  # type: Tuple[int, ...]
                            em_specs,  # type: Dict[str, Any]
                            connect_up=False,  # type: bool
                            ):
        # type: (...) -> Tuple[List[int], List[Union[int, float]], Tuple[int, int], Tuple[int, int]]
        """Calculate resistor core size/track information based on given specs.

        This method calculate the track width/spacing on each routing layer from EM specifications,
        then compute minimum width/height and block pitch of resistor blocks from given
        constraints.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        min_tracks : Tuple[int, ...]
            minimum number of tracks on each layer.
        em_specs : Dict[str, Any]
            EM specification dictionary.
        connect_up : bool
            True if the last used layer needs to be able to connect to the layer above.
            This options will make sure that the width of the last track is wide enough to support
            the inter-layer via.

        Returns
        -------
        track_widths : List[int]
            the track width on each layer that satisfies EM specs.
        track_spaces : List[Union[int, float]]
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
        cur_layer = self.get_bot_layer()
        for idx, min_num_tr in enumerate(min_tracks):
            # make sure that current layer can connect to next layer
            if idx < len(min_tracks) - 1 or connect_up:
                top_tr_w = grid.get_min_track_width(cur_layer + 1, unit_mode=True, **em_specs)
                top_w = grid.get_track_width(cur_layer + 1, top_tr_w, unit_mode=True)
            else:
                top_w = -1

            tr_p = grid.get_track_pitch(cur_layer, unit_mode=True)
            cur_width = grid.get_min_track_width(cur_layer, bot_w=prev_width, top_w=top_w,
                                                 unit_mode=True, **em_specs)
            cur_space = grid.get_num_space_tracks(cur_layer, cur_width, half_space=True)
            track_widths.append(cur_width)
            track_spaces.append(cur_space)
            cur_ntr = min_num_tr * (cur_width + cur_space)
            if isinstance(cur_space, float):
                cur_ntr += 0.5
            min_dim = int(round(tr_p * cur_ntr))

            if grid.get_direction(cur_layer) == 'x':
                min_h = max(min_h, min_dim)
            else:
                min_w = max(min_w, min_dim)

            prev_width = grid.get_track_width(cur_layer, cur_width, unit_mode=True)
            cur_layer += 1

        # get block size
        wblk, hblk = grid.get_block_size(cur_layer - 1, unit_mode=True, include_private=True)
        wblk_drc, hblk_drc = self.get_block_pitch()
        wblk = lcm([wblk, wblk_drc])
        hblk = lcm([hblk, hblk_drc])
        min_w = -(-min_w // wblk) * wblk
        min_h = -(-min_h // hblk) * hblk

        return track_widths, track_spaces, (min_w, min_h), (wblk, hblk)

    def find_core_size(self,  # type: ResTech
                       grid,  # type: RoutingGrid
                       params,  # type: Dict[str, Any]
                       wres,  # type: int
                       hres,  # type: int
                       wblk,  # type: int
                       hblk,  # type: int
                       ext_dir,  # type: str
                       max_blk_ext,  # type: int
                       ):
        # type: (...) -> Tuple[int, int, Dict[str, Any]]
        """Compute resistor core size that meets DRC rules.
        
        Given current resistor block size and the block pitch, increase the resistor block
        size if necessary to meet DRC rules.
        
        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
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
        max_blk_ext : int
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
                bin_iter = BinaryIterator(nxblk, nxblk + max_blk_ext + 1)
            else:
                bin_iter = BinaryIterator(nyblk, nyblk + max_blk_ext + 1)
            while bin_iter.has_next():
                ncur = bin_iter.get_next()
                if x_only:
                    wcur, hcur = ncur * wblk, hres
                else:
                    wcur, hcur = wres, ncur * hblk
                tmp = self.get_core_info(grid, wcur, hcur, **params)
                if tmp is None:
                    bin_iter.up()
                else:
                    ans = tmp
                    bin_iter.save()
                    bin_iter.down()

            if ans is None:
                raise ValueError('failed to find DRC clean core with maximum %d '
                                 'additional block pitches.' % max_blk_ext)
            if x_only:
                nxblk = bin_iter.get_last_save()
            else:
                nyblk = bin_iter.get_last_save()
            return nxblk, nyblk, ans
        else:
            # extend in both direction
            opt_area = (nxblk + max_blk_ext + 1) * (nyblk + max_blk_ext + 1)
            # linear search in height, binary search in width
            # in this way, for same area, use height as tie breaker
            nxopt, nyopt = nxblk, nyblk
            for nycur in range(nyblk, nyblk + max_blk_ext + 1):
                # check if we should terminate linear search
                if nycur * nxblk >= opt_area:
                    break
                bin_iter = BinaryIterator(nxblk, nxblk + max_blk_ext + 1)
                hcur = nycur * hblk
                while bin_iter.has_next():
                    nxcur = bin_iter.get_next()
                    if nxcur * nycur >= opt_area:
                        # this point can't beat current optimum
                        bin_iter.down()
                    else:
                        tmp = self.get_core_info(grid, nxcur * wblk, hcur, **params)
                        if tmp is None:
                            bin_iter.up()
                        else:
                            # found new optimum
                            ans, nxopt, nyopt = tmp, nxcur, nycur
                            opt_area = nxcur * nycur
                            bin_iter.down()

            if ans is None:
                raise ValueError('failed to find DRC clean core with maximum %d '
                                 'additional block pitches.' % max_blk_ext)
            return nxopt, nyopt, ans

    def find_edge_size(self,  # type: ResTech
                       grid,  # type: RoutingGrid
                       core_info,  # type: Dict[str, Any]
                       is_lr_edge,  # type: bool
                       params,  # type: Dict[str, Any]
                       blk1,  # type: int
                       max_blk_ext,  # type: int
                       ):
        # type: (...) -> Tuple[int, Dict[str, Any]]
        """Compute resistor edge size that meets DRC rules.

        Calculate edge dimension (width for LR edge, height for TB edge) that meets DRC rules

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        core_info : Dict[str, Any]
            core layout information dictionary.
        is_lr_edge : bool
            True if this is left/right edge, False if this is top/bottom edge.
        params : Dict[str, Any]
            the resistor parameters dictionary.
        blk1 : int
            dimension1 block size in resolution units.
        max_blk_ext : int
            maximum number of blocks we can extend by.

        Returns
        -------
        n1 : int
            edge length in dimension 1 as number of blocks.
        layout_info : Dict[str, Any]
            the edge layout information dictionary.
        """

        bin_iter = BinaryIterator(1, max_blk_ext + 2)
        ans = None
        while bin_iter.has_next():
            n1 = bin_iter.get_next()

            if is_lr_edge:
                tmp = self.get_lr_edge_info(grid, core_info, n1 * blk1, **params)
            else:
                tmp = self.get_tb_edge_info(grid, core_info, n1 * blk1, **params)

            if tmp is None:
                bin_iter.up()
            else:
                ans = tmp
                bin_iter.save()
                bin_iter.down()

        if ans is None:
            raise ValueError('failed to find DRC clean core with maximum %d '
                             'additional block pitches.' % max_blk_ext)

        return bin_iter.get_last_save(), ans

    def get_res_info(self,
                     grid,  # type: RoutingGrid
                     l,  # type: int
                     w,  # type: int
                     res_type,  # type: str
                     sub_type,  # type: str
                     threshold,  # type: str
                     min_tracks,  # type: Tuple[int, ...]
                     em_specs,  # type: Dict[str, Any]
                     ext_dir,  # type: Optional[str]
                     max_blk_ext=100,  # type: int
                     connect_up=False,  # type: bool
                     options=None,  # type: Optional[Dict[str, Any]]
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
        max_blk_ext : int
            number of block pitches we can extend the resistor core/edge size by.  If we cannot
            find a valid core size by extending this many block pitches, we declare failure.
        connect_up : bool
            True if the last used layer needs to be able to connect to the layer above.
            This options will make sure that the width of the last track is wide enough to support
            the inter-layer via.
        options : Optional[Dict[str, Any]]
            dictionary of optional parameters.

        Returns
        -------
        res_info : Dict[str, Any]
            the resistor layout information dictionary.
        
        """
        if options is None:
            options = {}
        else:
            pass

        # step 1: get track/size parameters
        tmp = self.get_core_track_info(grid, min_tracks, em_specs, connect_up=connect_up)
        track_widths, track_spaces, min_size, blk_pitch = tmp
        params = dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
            track_widths=track_widths,
            track_spaces=track_spaces,
            options=options,
        )
        # step 2: get minimum DRC core size, then update with minimum size and round to block size.
        wres, hres = self.get_min_res_core_size(l, w, res_type, sub_type, threshold, options)
        wres = max(wres, min_size[0])
        hres = max(hres, min_size[1])
        wblk, hblk = blk_pitch
        wres = -(-wres // wblk) * wblk
        hres = -(-hres // hblk) * hblk
        # step 3: extend core until density rule is satisfied.
        nxblk, nyblk, core_info = self.find_core_size(grid, params, wres, hres, wblk, hblk, ext_dir,
                                                      max_blk_ext)
        wcore, hcore = nxblk * wblk, nyblk * hblk
        # step 4: calculate edge size that satisfies density rule.
        nxblk_lr, edge_lr_info = self.find_edge_size(grid, core_info, True, params, wblk,
                                                     max_blk_ext)
        nyblk_tb, edge_tb_info = self.find_edge_size(grid, core_info, False, params, hblk,
                                                     max_blk_ext)
        wedge, hedge = nxblk_lr * wblk, nyblk_tb * hblk

        # step 6: calculate geometry information of each primitive block.
        bot_layer = self.get_bot_layer()
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
            if dim % pitch == 0:
                num_tracks.append(dim // pitch)
            else:
                num_tracks.append(dim / pitch)
            if dim_corner % pitch == 0:
                num_corner_tracks.append(dim_corner // pitch)
            else:
                num_corner_tracks.append(dim_corner / pitch)

        res_info = dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
            options=options,
            w_core=wcore,
            h_core=hcore,
            w_edge=wedge,
            h_edge=hedge,
            track_widths=track_widths,
            track_spaces=track_spaces,
            num_tracks=num_tracks,
            num_corner_tracks=num_corner_tracks,
            core_info=core_info,
            edge_lr_info=edge_lr_info,
            edge_tb_info=edge_tb_info,
            well_xl=edge_lr_info['well_xl'],
        )

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
        self._layout_info = params['res_info']
        self._num_tracks = self._layout_info['num_tracks']
        self._track_widths = self._layout_info['track_widths']
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['res_tech_class']

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
            res_info='resistor layout information dictionary.',
        )

    def get_num_tracks(self):
        # type: () -> Tuple[Union[float, int], ...]
        """Returns a list of the number of tracks on each routing layer in this template.

        Returns
        -------
        ntr_list : Tuple[Union[float, int], ...]
            a list of number of tracks in this template on each layer.
            index 0 is the bottom-most routing layer, and corresponds to
            AnalogResCore.port_layer_id().
        """
        return self._num_tracks

    def get_track_widths(self):
        # type: () -> Tuple[int, ...]
        """Returns a list of track widths on each routing layer.

        Returns
        -------
        width_list : Tuple[int, ...]
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
        l_str = float_to_si_string(self._layout_info['l'])
        w_str = float_to_si_string(self._layout_info['w'])
        res_type = self._layout_info['res_type']
        sub_type = self._layout_info['sub_type']
        threshold = self._layout_info['threshold']
        main = '%s_%s_%s_l%s_w%s' % (res_type, sub_type, threshold, l_str, w_str)
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
        flip_parity = self.grid.get_flip_parity()
        return self.to_immutable_id((self.get_layout_basename(), self._layout_info, flip_parity))

    def draw_layout(self):
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
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['res_tech_class']
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
        self._tech_cls.draw_res_boundary(self, self.params['boundary_type'],
                                         self.params['layout_info'], self.params['end_mode'])
        self.prim_top_layer = self._tech_cls.get_bot_layer()
