# -*- coding: utf-8 -*-

"""This module defines the RoutingGrid class.
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Any, Union

from warnings import warn

from pybag.core import Transform, PyRoutingGrid, coord_to_custom_htr
from pybag.enum import Orient2D, Direction, RoundMode

from bag.util.search import BinaryIterator
from bag.math import lcm
from bag.layout.tech import TechInfo

from .base import TrackID
from ...util.math import HalfInt
from ...typing import TrackType

SizeType = Tuple[int, HalfInt, HalfInt]
FillConfigType = Dict[int, Tuple[int, int, int, int]]
OptHalfIntType = Optional[HalfInt]


class RoutingGrid(PyRoutingGrid):
    """A class that represents the routing grid.

    This class provides various methods to convert between Cartesian coordinates and
    routing tracks.  This class assumes the lower-left coordinate is (0, 0)

    the track numbers are at half-track pitch.  That is, even track numbers corresponds
    to physical tracks, and odd track numbers corresponds to middle between two tracks.
    This convention is chosen so it is easy to locate a via for 2-track wide wires, for
    example.

    Assumptions:

    1. the pitch of all layers evenly divides the largest pitch.

    Parameters
    ----------
    tech_info : TechInfo
        the TechInfo instance used to create metals and vias.
    config_fname : str
        the routing grid configuration file.
    grid : Optional[RoutingGrid]
        If not None, create a copy of grid instead.
    """

    def __init__(self, tech_info: TechInfo, config_fname: str,
                 grid: Optional[RoutingGrid] = None) -> None:
        PyRoutingGrid.__init__(self, tech_info, config_fname, grid)
        self._tech_info = tech_info

    @classmethod
    def get_middle_track(cls, tr1: TrackType, tr2: TrackType, round_up: bool = False) -> HalfInt:
        """Get the track between the two given tracks."""
        tmp = HalfInt.convert(tr1)
        tmp.increment(tr2)
        return tmp.div2(round_up=round_up)

    @property
    def tech_info(self) -> TechInfo:
        """TechInfo: The TechInfo technology object."""
        return self._tech_info

    def get_bot_common_layer(self, inst_grid: RoutingGrid, inst_top_layer: int) -> int:
        """Given an instance's RoutingGrid, return the bottom common layer ID.

        Parameters
        ----------
        inst_grid : RoutingGrid
            the instance's RoutingGrid object.
        inst_top_layer : int
            the instance top layer ID.

        Returns
        -------
        bot_layer : int
            the bottom common layer ID.
        """
        last_layer = self.bot_layer
        for lay in range(inst_top_layer, last_layer - 1, -1):
            if not self.get_track_info(lay).compatible(inst_grid.get_track_info(lay)):
                return lay + 1

        return last_layer

    def is_horizontal(self, layer_id: int) -> bool:
        """Returns true if the given layer is horizontal."""
        return self.get_direction(layer_id) is Orient2D.x

    def get_num_tracks(self, size: SizeType, layer_id: int) -> HalfInt:
        """Returns the number of tracks on the given layer for a block with the given size.

        Parameters
        ----------
        size : SizeType
            the block size tuple.
        layer_id : int
            the layer ID.

        Returns
        -------
        num_tracks : HalfInt
            number of tracks on that given layer.
        """
        blk_dim = self.get_size_dimension(size)[self.get_direction(layer_id).value]
        tr_half_pitch = self.get_track_pitch(layer_id) // 2
        return HalfInt(blk_dim // tr_half_pitch)

    def get_num_space_tracks(self, layer_id: int, width_ntr: int,
                             half_space: bool = True, same_color: bool = False,
                             even: bool = False) -> HalfInt:
        """Returns the number of tracks needed for space around a track of the given width.

        In advance technologies, metal spacing is often a function of the metal width, so for a
        a wide track we may need to reserve empty tracks next to this.  This method computes the
        minimum number of empty tracks needed.

        Parameters
        ----------
        layer_id : int
            the track layer ID
        width_ntr : int
            the track width in number of tracks.
        half_space : bool
            True to allow half-integer spacing.
        same_color : bool
            True to use same-color spacing.
        even : bool
            True to enforce even resolution unit spacing.

        Returns
        -------
        num_sp_tracks : HalfInt
            minimum space needed around the given track in number of tracks.
        """
        htr = self.get_min_space_htr(layer_id, width_ntr, same_color=same_color, even=even)
        return HalfInt(htr + (htr & half_space))

    def get_line_end_space_tracks(self, layer_dir: Direction, wire_layer: int, width_ntr: int,
                                  half_space: bool = True) -> HalfInt:
        """Returns the minimum line end spacing in number of space tracks.

        Parameters
        ----------
        layer_dir : Direction
            the direction of the specified layer.  LOWER if the layer is the
            bottom layer, UPPER if the layer is the top layer.
        wire_layer : int
            line-end wire layer ID.
        width_ntr : int
            wire width, in number of tracks.
        half_space : bool
            True to allow half-track spacing.

        Returns
        -------
        space_ntr : HalfInt
            number of tracks needed to reserve as space.
        """
        htr = self.get_line_end_space_htr(layer_dir.value, wire_layer, width_ntr)
        return HalfInt(htr + (htr & half_space))

    def get_max_track_width(self, layer_id: int, num_tracks: int, tot_space: int,
                            half_end_space: bool = False) -> int:
        """Compute maximum track width and space that satisfies DRC rule.

        Given available number of tracks and numbers of tracks needed, returns
        the maximum possible track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        num_tracks : int
            number of tracks to draw.
        tot_space : int
            available number of tracks.
        half_end_space : bool
            True if end spaces can be half of minimum spacing.  This is true if you're
            these tracks will be repeated, or there are no adjacent tracks.

        Returns
        -------
        tr_w : int
            track width.
        """
        bin_iter = BinaryIterator(1, None)
        num_space = num_tracks if half_end_space else num_tracks + 1
        while bin_iter.has_next():
            tr_w = bin_iter.get_next()
            tr_sp = self.get_num_space_tracks(layer_id, tr_w, half_space=False)
            used_tracks = tr_w * num_tracks + tr_sp * num_space
            if used_tracks > tot_space:
                bin_iter.down()
            else:
                bin_iter.save()
                bin_iter.up()

        opt_w = bin_iter.get_last_save()
        return opt_w

    @staticmethod
    def get_evenly_spaced_tracks(num_tracks: int, tot_space: int, track_width: int,
                                 half_end_space: bool = False) -> List[HalfInt]:
        """Evenly space given number of tracks in the available space.

        Currently this method may return half-integer tracks.

        Parameters
        ----------
        num_tracks : int
            number of tracks to draw.
        tot_space : int
            avilable number of tracks.
        track_width : int
            track width in number of tracks.
        half_end_space : bool
            True if end spaces can be half of minimum spacing.  This is true if you're
            these tracks will be repeated, or there are no adjacent tracks.

        Returns
        -------
        idx_list : List[HalfInt]
            list of track indices.  0 is the left-most track.
        """
        if half_end_space:
            tot_space_htr = 2 * tot_space
            scale = 2 * tot_space_htr
            offset = tot_space_htr + num_tracks
            den = 2 * num_tracks
        else:
            tot_space_htr = 2 * tot_space
            width_htr = 2 * track_width - 2
            # magic math.  You can work it out
            scale = 2 * (tot_space_htr + width_htr)
            offset = 2 * tot_space_htr - width_htr * (num_tracks - 1) + (num_tracks + 1)
            den = 2 * (num_tracks + 1)

        return [HalfInt((scale * idx + offset) // den - 1) for idx in range(num_tracks)]

    def get_fill_size(self, top_layer: int, fill_config: FillConfigType, *,
                      include_private: bool = False, half_blk_x: bool = True,
                      half_blk_y: bool = True) -> Tuple[int, int]:
        """Returns unit block size given the top routing layer and power fill configuration.

        Parameters
        ----------
        top_layer : int
            the top layer ID.
        fill_config : Dict[int, Tuple[int, int, int, int]]
            the fill configuration dictionary.
        include_private : bool
            True to include private layers in block size calculation.
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.

        Returns
        -------
        block_width : int
            the block width in resolution units.
        block_height : int
            the block height in resolution units.
        """
        blk_w, blk_h = self.get_block_size(top_layer, include_private=include_private,
                                           half_blk_x=half_blk_x, half_blk_y=half_blk_y)

        dim_list = [[blk_w], [blk_h]]
        for lay, (tr_w, tr_sp, _, _) in fill_config.items():
            if lay <= top_layer:
                cur_pitch = self.get_track_pitch(lay)
                cur_dim = (tr_w + tr_sp) * cur_pitch * 2
                dim_list[1 - self.get_direction(lay).value].append(cur_dim)

        blk_w = lcm(dim_list[0])
        blk_h = lcm(dim_list[1])
        return blk_w, blk_h

    def get_size_tuple(self, layer_id: int, width: int, height: int, *, round_up: bool = False,
                       half_blk_x: bool = False, half_blk_y: bool = False) -> SizeType:
        """Compute the size tuple corresponding to the given width and height from block pitch.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        width : int
            width of the block, in resolution units.
        height : int
            height of the block, in resolution units.
        round_up : bool
            True to round up instead of raising an error if the given width and height
            are not on pitch.
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.

        Returns
        -------
        size : SizeType
            the size tuple.  the first element is the top layer ID, second element is the width in
            number of vertical tracks, and third element is the height in number of
            horizontal tracks.
        """
        w_pitch, h_pitch = self.get_size_pitch(layer_id)

        wblk, hblk = self.get_block_size(layer_id, half_blk_x=half_blk_x, half_blk_y=half_blk_y)
        if width % wblk != 0:
            if round_up:
                width = -(-width // wblk) * wblk
            else:
                raise ValueError('width = %d not on block pitch (%d)' % (width, wblk))
        if height % hblk != 0:
            if round_up:
                height = -(-height // hblk) * hblk
            else:
                raise ValueError('height = %d not on block pitch (%d)' % (height, hblk))

        return layer_id, HalfInt(2 * width // w_pitch), HalfInt(2 * height // h_pitch)

    def get_size_dimension(self, size: SizeType) -> Tuple[int, int]:
        """Compute width and height from given size.

        Parameters
        ----------
        size : SizeType
            size of a block.

        Returns
        -------
        width : int
            the width in resolution units.
        height : int
            the height in resolution units.
        """
        w_pitch, h_pitch = self.get_size_pitch(size[0])
        return int(size[1] * w_pitch), int(size[2] * h_pitch)

    def convert_size(self, size: SizeType, new_top_layer: int) -> SizeType:
        """Convert the given size to a new top layer.

        Parameters
        ----------
        size : SizeType
            size of a block.
        new_top_layer : int
            the new top level layer ID.

        Returns
        -------
        new_size : SizeType
            the new size tuple.
        """
        wblk, hblk = self.get_size_dimension(size)
        return self.get_size_tuple(new_top_layer, wblk, hblk)

    def get_track_parity(self, layer_id: int, tr_idx: TrackType) -> int:
        """Returns the parity of the given track.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the track index.

        Returns
        -------
        parity : int
            the track parity.
        """
        return self.get_htr_parity(layer_id, int(round(2 * tr_idx)))

    def get_layer_purpose(self, layer_id: int, tr_idx: TrackType) -> Tuple[str, str]:
        """Get the layer/purpose pair of the given track.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the track index.

        Returns
        -------
        layer_name : str
            the layer name.
        purpose_name : str
            the purpose name.
        """
        return self.get_htr_layer_purpose(layer_id, int(round(2 * tr_idx)))

    def get_wire_bounds(self, layer_id: int, tr_idx: TrackType, width: int = 1) -> Tuple[int, int]:
        """Calculate the wire bounds coordinate.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the center track index.
        width : int
            width of wire in number of tracks.

        Returns
        -------
        lower : int
            the lower bound coordinate perpendicular to wire direction.
        upper : int
            the upper bound coordinate perpendicular to wire direction.
        """
        return self.get_wire_bounds_htr(layer_id, int(round(2 * tr_idx)), width)

    def coord_to_track(self, layer_id: int, coord: int, mode: RoundMode = RoundMode.NONE,
                       even: bool = False) -> HalfInt:
        """Convert given coordinate to track number.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        mode : RoundMode
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

            If mode == NONE, raise error if coordinate is not on track.

        even : bool
            True to round coordinate to integer tracks.

        Returns
        -------
        track : HalfInt
            the track number
        """
        return HalfInt(self.coord_to_htr(layer_id, coord, mode, even))

    def coord_to_fill_track(self, layer_id: int, coord: int, fill_config: Dict[int, Any],
                            mode: RoundMode = RoundMode.NEAREST) -> HalfInt:
        """Returns the fill track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        fill_config : Dict[int, Any]
            the fill configuration dictionary.
        mode : RoundMode
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

            If mode == NONE, raise error if coordinate is not on track.

        Returns
        -------
        track : HalfInt
            the track number
        """
        ntr_w, ntr_sp, _, _ = fill_config[layer_id]

        num_htr = round(2 * (ntr_w + ntr_sp))
        fill_pitch = num_htr * self.get_track_pitch(layer_id) // 2
        return HalfInt(coord_to_custom_htr(coord, fill_pitch, fill_pitch // 2, mode, False))

    def coord_to_nearest_track(self, layer_id: int, coord: int, *,
                               half_track: bool = True,
                               mode: Union[RoundMode, int] = RoundMode.NEAREST) -> HalfInt:
        """Returns the track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        half_track : bool
            if True, allow half integer track numbers.
        mode : Union[RoundMode, int]
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

        Returns
        -------
        track : HalfInt
            the track number
        """
        warn('coord_to_nearest_track is deprecated, use coord_to_track with optional flags instead',
             DeprecationWarning)
        return HalfInt(self.coord_to_htr(layer_id, coord, mode, not half_track))

    def find_next_track(self, layer_id: int, coord: int, *, tr_width: int = 1,
                        half_track: bool = True,
                        mode: Union[RoundMode, int] = RoundMode.GREATER_EQ) -> HalfInt:
        """Find the track such that its edges are on the same side w.r.t. the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        tr_width : int
            the track width, in number of tracks.
        half_track : bool
            True to allow half integer track center numbers.
        mode : Union[RoundMode, int]
            the rounding mode.  NEAREST and NONE are not supported.

            If mode == LESS_EQ, return the track with both edges less
            than or equal to coord.

            If mode == LESS, return the nearest track with both edges less
            than coord.

            If mode == GREATER, return the nearest track with both edges greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with both edges greater
            than coord.

        Returns
        -------
        tr_idx : HalfInt
            the center track index.
        """
        return HalfInt(self.find_next_htr(layer_id, coord, tr_width, mode, not half_track))

    def get_track_index_range(self, layer_id: int, lower: int, upper: int, *,
                              num_space: TrackType = 0, edge_margin: int = 0,
                              half_track: bool = True) -> Tuple[OptHalfIntType, OptHalfIntType]:
        """ Returns the first and last track index strictly in the given range.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        lower : int
            the lower coordinate.
        upper : int
            the upper coordinate.
        num_space : TrackType
            number of space tracks to the tracks right outside of the given range.
        edge_margin : int
            minimum space from outer tracks to given range.
        half_track : bool
            True to allow half-integer tracks.

        Returns
        -------
        start_track : OptHalfIntType
            the first track index.  None if no solution.
        end_track : OptHalfIntType
            the last track index.  None if no solution.
        """
        even = not half_track
        # get start track half index
        lower_bnd = self.find_next_track(layer_id, lower, mode=RoundMode.LESS_EQ)
        start_track = self.find_next_track(layer_id, lower + edge_margin, mode=RoundMode.GREATER_EQ)
        start_track = max(start_track, lower_bnd + num_space)
        start_track.up_even(even)

        # get end track half index
        upper_bnd = self.find_next_track(layer_id, upper, mode=RoundMode.GREATER_EQ)
        end_track = self.find_next_track(layer_id, upper - edge_margin, mode=RoundMode.LESS_EQ)
        end_track = min(end_track, upper_bnd - num_space)
        end_track.down_even(even)

        if end_track < start_track:
            # no solution
            return None, None
        return start_track, end_track

    def get_overlap_tracks(self, layer_id: int, lower: int, upper: int,
                           half_track: bool = True) -> Tuple[OptHalfIntType, OptHalfIntType]:
        """ Returns the first and last track index that overlaps with the given range.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        lower : int
            the lower coordinate.
        upper : int
            the upper coordinate.
        half_track : bool
            True to allow half-integer tracks.

        Returns
        -------
        start_track : OptHalfIntType
            the first track index.  None if no solution.
        end_track : OptHalfIntType
            the last track index.  None if no solution.
        """
        even = not half_track
        lower_tr = self.find_next_track(layer_id, lower, mode=RoundMode.LESS_EQ)
        lower_tr.up().up_even(even)
        upper_tr = self.find_next_track(layer_id, upper, mode=RoundMode.GREATER_EQ)
        upper_tr.down().down_even(even)

        if upper_tr < lower_tr:
            return None, None
        return lower_tr, upper_tr

    def transform_track(self, layer_id: int, track_idx: TrackType, xform: Transform) -> HalfInt:
        """Transform the given track index.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        track_idx : TrackType
            the track index.
        xform : Transform
            the transformation object.

        Returns
        -------
        new_track_idx : HalfInt
            the transformed track index.
        """
        return TrackID(layer_id, track_idx).transform(xform, self).base_index

    def track_to_coord(self, layer_id: int, track_idx: TrackType) -> int:
        """Convert given track number to coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        track_idx : TrackType
            the track number.

        Returns
        -------
        coord : int
            the coordinate perpendicular to track direction.
        """
        return self.htr_to_coord(layer_id, int(round(2 * track_idx)))

    def interval_to_track(self, layer_id: int, intv: Tuple[int, int]) -> Tuple[HalfInt, int]:
        """Convert given coordinates to track number and width.

        Parameters
        ----------
        layer_id : int
            the layer number.
        intv : Tuple[int, int]
            lower and upper coordinates perpendicular to the track direction.

        Returns
        -------
        track : HalfInt
            the track number
        width : int
            the track width, in number of tracks.
        """
        start, stop = intv
        htr = self.coord_to_htr(layer_id, (start + stop) // 2, RoundMode.NONE, False)
        width = stop - start

        # binary search to take width override into account
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_ntr = bin_iter.get_next()
            wire_bounds = self.get_wire_bounds_htr(layer_id, htr, cur_ntr)
            wire_width = wire_bounds[1] - wire_bounds[0]
            if wire_width == width:
                return HalfInt(htr), cur_ntr
            elif wire_width > width:
                bin_iter.down()
            else:
                bin_iter.up()

        # never found solution; width is not quantized.
        raise ValueError('Interval {} on layer {} width not quantized'.format(intv, layer_id))

    def copy(self) -> RoutingGrid:
        """Returns a copy of this RoutingGrid."""
        return RoutingGrid(self._tech_info, '', grid=self)
