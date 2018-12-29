# -*- coding: utf-8 -*-

"""This module defines the RoutingGrid class.
"""

from typing import TYPE_CHECKING, Sequence, Union, Tuple, List, Optional, Dict, Any

from pybag.core import BBox
from pybag.enum import Orientation, Orient2D

from bag.util.search import BinaryIterator
from bag.math import lcm

from .base import HalfInt

if TYPE_CHECKING:
    from bag.layout.core import TechInfo
    from bag.typing import TrackType


class RoutingGrid(object):
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
    layers : Sequence[int]
        list of available routing layers.  Must be in increasing order.
    spaces : Sequence[int]
        list of track spacings for each layer.
    widths : Sequence[int]
        list of minimum track widths for each layer.
    bot_dir : str
        the direction of the bottom-most layer.  Either 'x' for horizontal tracks or 'y' for
        vertical tracks.
    max_num_tr : Union[int, Sequence[int]]
        maximum track width in number of tracks.  Can be given as an integer (which applies to
        all layers), our a list to specify maximum width per layer.
    width_override : Dict[int, Dict[int, int]]
        the width override dictionary.
    """

    def __init__(self,  # type: RoutingGrid
                 tech_info,  # type: TechInfo
                 layers,  # type: Sequence[int]
                 spaces,  # type: Sequence[int]
                 widths,  # type: Sequence[int]
                 bot_dir,  # type: str
                 max_num_tr=1000,  # type: Union[int, Sequence[int]]
                 width_override=None,  # type: Dict[int, Dict[int, int]]
                 ):
        # type: (...) -> None
        # error checking
        num_layer = len(layers)
        if len(spaces) != num_layer:
            raise ValueError('spaces length = %d != %d' % (len(spaces), num_layer))
        if len(widths) != num_layer:
            raise ValueError('spaces length = %d != %d' % (len(widths), num_layer))
        if isinstance(max_num_tr, int):
            max_num_tr = [max_num_tr] * num_layer
        elif len(max_num_tr) != num_layer:
            raise ValueError('max_num_tr length = %d != %d' % (len(max_num_tr), num_layer))

        self._tech_info = tech_info
        self._resolution = tech_info.resolution
        self._layout_unit = tech_info.layout_unit
        self._flip_parity = {}
        self._ignore_layers = set()
        self.layers = []  # type: List[int]
        self.sp_tracks = {}
        self.w_tracks = {}
        self.offset_tracks = {}
        self.dir_tracks = {}  # type: Dict[int, Orient2D]
        self.max_num_tr_tracks = {}
        self.block_pitch = {}
        self.w_override = {}
        self.private_layers = []

        cur_dir = bot_dir
        for lay, sp, w, max_num in zip(layers, spaces, widths, max_num_tr):
            self.add_new_layer(lay, sp, w, cur_dir, max_num_tr=max_num, is_private=False)
            # alternate track direction
            cur_dir = 'y' if cur_dir == 'x' else 'x'

        self.update_block_pitch()

        # add width overrides
        if width_override is not None:
            for layer_id, w_info in width_override.items():
                for width_ntr, tr_w in w_info.items():
                    self.add_width_override(layer_id, width_ntr, tr_w)

    def __contains__(self, layer):
        # type: (int) -> bool
        """Returns True if this RoutingGrid contains the given layer. """
        return layer in self.sp_tracks

    @classmethod
    def get_middle_track(cls, tr1, tr2, round_up=False):
        # type: (TrackType, TrackType, bool) -> HalfInt
        tmp = HalfInt.convert(tr1)
        tmp += tr2
        return tmp.div2(round_up=round_up)

    def _get_track_offset(self, layer_id):
        # type: (int) -> int
        """Returns the track offset in resolution units on the given layer."""
        track_pitch = self.get_track_pitch(layer_id)
        return self.offset_tracks.get(layer_id, track_pitch // 2)

    def get_flip_parity(self):
        # type: () -> Dict[int, Tuple[int, int]]
        """Returns a copy of the flip parity dictionary."""
        return self._flip_parity.copy()

    def get_bot_common_layer(self, inst_grid, inst_top_layer):
        # type: (RoutingGrid, int) -> int
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
        my_bot_layer = self.layers[0]
        for bot_layer in range(inst_top_layer, my_bot_layer - 1, -1):
            has_bot = (bot_layer in self.layers)
            inst_has_bot = (bot_layer in inst_grid.layers)
            if has_bot and inst_has_bot:
                w_par, sp_par = self.get_track_info(bot_layer)
                w_inst, sp_inst = inst_grid.get_track_info(bot_layer)
                if w_par != w_inst or sp_par != sp_inst or \
                        self.dir_tracks[bot_layer] is not inst_grid.dir_tracks[bot_layer]:
                    return bot_layer + 1
            elif has_bot != inst_has_bot:
                return bot_layer + 1

        return my_bot_layer

    def get_flip_parity_at(self,  # type: RoutingGrid
                           bot_layer,  # type: int
                           top_layer,  # type: int
                           loc,  # type: Tuple[int, int]
                           orient,  # type: str
                           unit_mode=True,  # type: bool
                           ):
        # type: (...) -> Dict[int, Tuple[int, int]]
        """Compute the flip parity dictionary for an instance placed at the given location.

        Parameters
        ----------
        bot_layer : int
            the bottom layer ID, inclusive.
        top_layer : int
            the top layer ID, inclusive.
        loc : Tuple[int, int]
            the instance origin location.
        orient : str
            the instance orientation.
        unit_mode : bool
            Deprecated parameter.

        Returns
        -------
        flip_parity : Dict[int, Tuple[int, int]]
            the flip_parity dictionary.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported')

        xo, yo = loc

        oenum = Orientation[orient]
        if oenum is Orientation.R0:
            xscale, yscale = 1, 1
        elif oenum is Orientation.MX:
            xscale, yscale = -1, 1
        elif oenum is Orientation.MY:
            xscale, yscale = 1, -1
        elif oenum is Orientation.R180:
            xscale, yscale = -1, -1
        else:
            raise ValueError('Unknown orientation: %s' % orient)

        flip_par = {}
        for lay in range(bot_layer, top_layer + 1):
            if lay in self.layers:
                # find the track in top level that corresponds to the track at instance origin
                if self.is_horizontal(lay):
                    coord, scale = yo, xscale
                else:
                    coord, scale = xo, yscale

                tr_idx = self.coord_to_track(lay, coord)
                offset_htr = int(tr_idx * 2) + 1

                cur_scale, cur_offset = self._flip_parity.get(lay, (1, 0))
                new_scale = cur_scale * scale
                new_offset = (cur_scale * offset_htr + cur_offset) % 4
                flip_par[lay] = (new_scale, new_offset)

        return flip_par

    def set_flip_parity(self, fp):
        # type: (Dict[int, Tuple[int, int]]) -> None
        """set the flip track parity dictionary."""
        for lay in fp:
            self._flip_parity[lay] = fp[lay]

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """The TechInfo technology object."""
        return self._tech_info

    @property
    def resolution(self):
        # type: () -> float
        """Returns the grid resolution."""
        return self._resolution

    @property
    def layout_unit(self):
        # type: () -> float
        """Returns the layout unit length, in meters."""
        return self._layout_unit

    @property
    def top_private_layer(self):
        # type: () -> int
        """Returns the top private layer ID."""
        return -99 if not self.private_layers else self.private_layers[-1]

    def update_block_pitch(self):
        # type: () -> None
        """Update block pitch."""
        self.block_pitch.clear()
        top_private_layer = self.top_private_layer

        # update private block pitches
        lay_list = [lay for lay in self.layers
                    if lay <= top_private_layer and lay not in self._ignore_layers]
        self._update_block_pitch_helper(lay_list)

        # update public block pitches
        lay_list = [lay for lay in self.layers
                    if lay > top_private_layer and lay not in self._ignore_layers]
        self._update_block_pitch_helper(lay_list)

    def _update_block_pitch_helper(self, lay_list):
        # type: (Sequence[int]) -> None
        """helper method for updating block pitch."""
        pitch_list = []
        for lay in lay_list:
            cur_bp = self.get_track_pitch(lay)
            cur_bp2 = cur_bp // 2
            cur_dir = self.dir_tracks[lay]
            if pitch_list:
                # the pitch of each layer = LCM of all layers below with same direction
                for play, (bp, bp2) in zip(lay_list, pitch_list):
                    if self.dir_tracks[play] is cur_dir:
                        cur_bp = lcm([cur_bp, bp])
                        cur_bp2 = lcm([cur_bp2, bp2])
            result = (cur_bp, cur_bp2)
            pitch_list.append(result)
            self.block_pitch[lay] = result

    def is_horizontal(self, layer_id):
        # type: (int) -> bool
        """Returns true if the given layer is horizontal."""
        return self.dir_tracks[layer_id] is Orient2D.x

    def get_direction(self, layer_id):
        # type: (int) -> str
        """Returns the track direction of the given layer.

        Parameters
        ----------
        layer_id : int
            the layer ID.

        Returns
        -------
        tdir : str
            'x' for horizontal tracks, 'y' for vertical tracks.
        """
        return self.dir_tracks[layer_id].name

    def get_track_pitch(self, layer_id, unit_mode=True):
        # type: (int, bool) -> int
        """Returns the routing track pitch on the given layer.

        Parameters
        ----------
        layer_id : int
            the routing layer ID.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track_pitch : int
            the track pitch.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        return self.w_tracks[layer_id] + self.sp_tracks[layer_id]

    def get_track_width(self, layer_id, width_ntr, unit_mode=True):
        # type: (int, int, bool) -> int
        """Calculate track width in layout units from number of tracks.

        Parameters
        ----------
        layer_id : int
            the track layer ID
        width_ntr : int
            the track width in number of tracks.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        width : int
            the track width.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        w = self.w_tracks[layer_id]
        sp = self.sp_tracks[layer_id]
        w_unit = width_ntr * (w + sp) - sp
        return self.w_override[layer_id].get(width_ntr, w_unit)

    def get_track_width_inverse(self, layer_id, width, mode=-1, unit_mode=True):
        # type: (int, int, int, bool) -> int
        """Given track width in resolution units, compute equivalent number of tracks.

        This is the inverse function of get_track_width().

        Parameters
        ----------
        layer_id : int
            the track layer ID
        width : int
            the track width in resolution units.
        mode : int
            If negative, the result wire will have width less than or equal to the given width.
            If positive, the result wire will have width greater than or equal to the given width.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        width_ntr : int
            number of tracks needed to achieve the given width.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        # use binary search to find the minimum track width
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            ntr = bin_iter.get_next()
            w_test = self.get_track_width(layer_id, ntr, unit_mode=True)
            if w_test == width:
                return ntr
            elif w_test < width:
                if mode < 0:
                    bin_iter.save()
                bin_iter.up()
            else:
                if mode > 0:
                    bin_iter.save()
                bin_iter.down()

        ans = bin_iter.get_last_save()
        if ans is None:
            return 0
        return ans

    def get_num_tracks(self, size, layer_id):
        # type: (Tuple[int, TrackType, TrackType], int) -> HalfInt
        """Returns the number of tracks on the given layer for a block with the given size.

        Parameters
        ----------
        size : Tuple[int, HalfInt, HalfInt]
            the block size tuple.
        layer_id : int
            the layer ID.

        Returns
        -------
        num_tracks : HalfInt
            number of tracks on that given layer.
        """
        is_x = self.is_horizontal(layer_id)
        blk_w, blk_h = self.get_size_dimension(size)
        tr_half_pitch = self.get_track_pitch(layer_id) // 2
        return HalfInt((blk_h if is_x else blk_w) // tr_half_pitch)

    def get_min_length(self, layer_id, width_ntr, unit_mode=True):
        # type: (int, int, bool) -> int
        """Returns the minimum length for the given track.

        Parameters
        ----------
        layer_id : int
            the track layer ID
        width_ntr : int
            the track width in number of tracks.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        min_length : int
            the minimum length.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        layer_name = self.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]
        layer_type = self.tech_info.get_layer_type(layer_name)

        width = self.get_track_width(layer_id, width_ntr)
        return self.tech_info.get_min_length(layer_type, width)

    def get_space(self, layer_id, width_ntr, same_color=False, unit_mode=True):
        # type: (int, int, bool, bool) -> int
        """Returns the space needed around a track, in resolution units.

        Parameters
        ----------
        layer_id : int
            the track layer ID
        width_ntr : int
            the track width in number of tracks.
        same_color : bool
            True to use same-color spacing.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        sp : int
            minimum space needed around the given track in resolution units.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        layer_name = self.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]
        layer_type = self.tech_info.get_layer_type(layer_name)

        width = self.get_track_width(layer_id, width_ntr)
        return self.tech_info.get_min_space(layer_type, width, same_color=same_color)

    def get_num_space_tracks(self, layer_id, width_ntr, half_space=True, same_color=False):
        # type: (int, int, bool, bool) -> HalfInt
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

        Returns
        -------
        num_sp_tracks : HalfInt
            minimum space needed around the given track in number of tracks.
        """
        width = self.get_track_width(layer_id, width_ntr)
        sp_min_unit = self.get_space(layer_id, width_ntr, same_color=same_color)
        w_unit = self.w_tracks[layer_id]
        sp_unit = self.sp_tracks[layer_id]
        # if this width is overridden, we may have extra space
        width_normal = (w_unit + sp_unit) * width_ntr - sp_unit
        extra_space = (width_normal - width) // 2
        half_pitch = (w_unit + sp_unit) // 2
        num_half_pitch = -(-(sp_min_unit - sp_unit - extra_space) // half_pitch)
        if num_half_pitch % 2 == 0 or half_space:
            return HalfInt(num_half_pitch)
        else:
            return HalfInt(num_half_pitch + 1)

    def get_line_end_space(self, layer_id, width_ntr, unit_mode=True):
        # type: (int, int, bool) -> int
        """Returns the minimum line end spacing for the given wire.

        Parameters
        ----------
        layer_id : int
            wire layer ID.
        width_ntr : int
            wire width, in number of tracks.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        space : int
            the line-end spacing.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        layer_name = self.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]
        layer_type = self.tech_info.get_layer_type(layer_name)
        width = self.get_track_width(layer_id, width_ntr)
        return self.tech_info.get_min_line_end_space(layer_type, width)

    def get_line_end_space_tracks(self, wire_layer, space_layer, width_ntr, half_space=True):
        # type: (int, int, int, bool) -> HalfInt
        """Returns the minimum line end spacing in number of space tracks.

        Parameters
        ----------
        wire_layer : int
            line-end wire layer ID.
        space_layer : int
            the layer used to measure line-end space.  Must be adjacent to wire_layer, and its
            direction must be orthogonal to the wire layer.
        width_ntr : int
            wire width, in number of tracks.
        half_space : bool
            True to allow half-track spacing.

        Returns
        -------
        space_ntr : HalfInt
            number of tracks needed to reserve as space.
        """
        if space_layer == wire_layer - 1:
            _, conn_ext = self.get_via_extensions(space_layer, 1, width_ntr)
        elif space_layer == wire_layer + 1:
            conn_ext, _ = self.get_via_extensions(wire_layer, width_ntr, 1)
        else:
            raise ValueError('space_layer must be adjacent to wire_layer')

        if self.get_direction(space_layer) == self.get_direction(wire_layer):
            raise ValueError('space_layer must be orthogonal to wire_layer.')

        wire_sp = self.get_line_end_space(wire_layer, width_ntr)
        margin = 2 * conn_ext + wire_sp
        w, sp = self.get_track_info(space_layer)
        half_pitch = (w + sp) // 2
        space_ntr = max(-(-(margin - sp) // half_pitch), 0)
        if space_ntr % 2 == 0 or half_space:
            return HalfInt(space_ntr)
        else:
            return HalfInt(space_ntr + 1)

    def get_max_track_width(self, layer_id, num_tracks, tot_space, half_end_space=False):
        # type: (int, int, int, bool) -> int
        """Compute maximum track width and space that satisfies DRC rule.

        Given available number of tracks and numbers of tracks needed, returns
        the maximum possible track width and spacing.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        num_tracks : int
            number of tracks to draw.
        tot_space : int
            avilable number of tracks.
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
    def get_evenly_spaced_tracks(num_tracks, tot_space, track_width, half_end_space=False):
        # type: (int, int, int, bool) -> List[HalfInt]
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

    def get_block_size(self, layer_id, unit_mode=True, include_private=False,
                       half_blk_x=True, half_blk_y=True):
        # type: (int, bool, bool, bool, bool) -> Tuple[int, int]
        """Returns unit block size given the top routing layer.

        Parameters
        ----------
        layer_id : int
            the routing layer ID.
        unit_mode : bool
            deprecated parameter.
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
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        top_private_layer = self.top_private_layer
        top_dir = self.dir_tracks[layer_id]

        # get bottom layer that has different direction
        bot_layer = layer_id - 1
        while bot_layer in self.block_pitch and self.dir_tracks[bot_layer] is top_dir:
            bot_layer -= 1

        if bot_layer not in self.block_pitch:
            bot_pitch = (2, 1)
        else:
            bot_pitch = self.block_pitch[bot_layer]

        top_pitch = self.block_pitch[layer_id]

        if layer_id > top_private_layer >= bot_layer and not include_private:
            # if top layer not private but bottom layer is, then bottom is not quantized.
            bot_pitch = (2, 1)

        if top_dir is Orient2D.y:
            w_pitch, h_pitch = top_pitch, bot_pitch
        else:
            w_pitch, h_pitch = bot_pitch, top_pitch

        w_pitch = w_pitch[1] if half_blk_x else w_pitch[0]
        h_pitch = h_pitch[1] if half_blk_y else h_pitch[0]
        return w_pitch, h_pitch

    def get_fill_size(self,  # type: RoutingGrid
                      top_layer,  # type: int
                      fill_config,  # type: Dict[int, Tuple[int, int, int, int]]
                      unit_mode=True,  # type: bool
                      include_private=False,  # type: bool
                      half_blk_x=True,  # type: bool
                      half_blk_y=True,  # type: bool
                      ):
        # type: (...) -> Tuple[int, int]
        """Returns unit block size given the top routing layer and power fill configuration.

        Parameters
        ----------
        top_layer : int
            the top layer ID.
        fill_config : Dict[int, Tuple[int, int, int, int]]
            the fill configuration dictionary.
        unit_mode : bool
            deprecated parameter.
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
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        blk_w, blk_h = self.get_block_size(top_layer, include_private=include_private,
                                           half_blk_x=half_blk_x, half_blk_y=half_blk_y)

        w_list = [blk_w]
        h_list = [blk_h]
        for lay, (tr_w, tr_sp, _, _) in fill_config.items():
            if lay <= top_layer:
                cur_pitch = self.get_track_pitch(lay)
                cur_dim = (tr_w + tr_sp) * cur_pitch * 2
                if self.is_horizontal(lay):
                    h_list.append(cur_dim)
                else:
                    w_list.append(cur_dim)

        blk_w = lcm(w_list)
        blk_h = lcm(h_list)
        return blk_w, blk_h

    def size_defined(self, layer_id):
        # type: (int) -> bool
        """Returns True if size is defined on the given layer."""
        return layer_id >= self.top_private_layer + 2

    def get_size_pitch(self, layer_id, unit_mode=True):
        # type: (int, bool) -> Tuple[int, int]
        """Returns the horizontal/vertical pitch that defines template size.

        Parameters
        ----------
        layer_id : int
            the size layer.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        w_pitch : int
            the width pitch.
        h_pitch : int
            the height pitch.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        if not self.size_defined(layer_id):
            raise ValueError('Size tuple is undefined for layer = %d' % layer_id)

        top_dir = self.dir_tracks[layer_id]
        bot_layer = layer_id - 1
        while bot_layer in self.dir_tracks and self.dir_tracks[bot_layer] is top_dir:
            bot_layer -= 1

        h_pitch = self.get_track_pitch(layer_id)
        w_pitch = self.get_track_pitch(bot_layer)
        if top_dir is Orient2D.y:
            return h_pitch, w_pitch
        return w_pitch, h_pitch

    def get_size_tuple(self,  # type: RoutingGrid
                       layer_id,  # type: int
                       width,  # type: int
                       height,  # type: int
                       round_up=False,  # type: bool
                       unit_mode=True,  # type: bool
                       half_blk_x=True,  # type: bool
                       half_blk_y=True,  # type: bool
                       ):
        # type: (...) -> Tuple[int, HalfInt, HalfInt]
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
        unit_mode : bool
            deprecated parameter.
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.

        Returns
        -------
        size : Tuple[int, HalfInt, HalfInt]
            the size tuple.  the first element is the top layer ID, second element is the width in
            number of vertical tracks, and third element is the height in number of
            horizontal tracks.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

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

    def get_size_dimension(self,  # type: RoutingGrid
                           size,  # type: Tuple[int, HalfInt, HalfInt]
                           unit_mode=True,  # type: bool
                           ):
        # type: (...) -> Tuple[int, int]
        """Compute width and height from given size.

        Parameters
        ----------
        size : Tuple[int, HalfInt, HalfInt]
            size of a block.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        width : int
            the width in resolution units.
        height : int
            the height in resolution units.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        w_pitch, h_pitch = self.get_size_pitch(size[0])
        return int(size[1] * w_pitch), int(size[2] * h_pitch)

    def convert_size(self, size, new_top_layer):
        # type: (Tuple[int, HalfInt, HalfInt], int) -> Tuple[int, HalfInt, HalfInt]
        """Convert the given size to a new top layer.

        Parameters
        ----------
        size : Tuple[int, HalfInt, HalfInt]
            size of a block.
        new_top_layer : int
            the new top level layer ID.

        Returns
        -------
        new_size : Tuple[int, HalfInt, HalfInt]
            the new size tuple.
        """
        wblk, hblk = self.get_size_dimension(size)
        return self.get_size_tuple(new_top_layer, wblk, hblk)

    def get_track_info(self, layer_id, unit_mode=True):
        # type: (int, bool) -> Tuple[int, int]
        """Returns the routing track width and spacing on the given layer.

        Parameters
        ----------
        layer_id : int
            the routing layer ID.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track_width : int
            the track width in resolution units.
        track_spacing : int
            the track spacing in resolution units
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        return self.w_tracks[layer_id], self.sp_tracks[layer_id]

    def get_track_parity(self, layer_id, tr_idx):
        # type: (int, TrackType) -> int
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
            the track parity, either 0 or 1.
        """
        # multiply then divide by 2 makes sure negative tracks are colored correctly.
        htr = round(tr_idx * 2 + 1)
        scale, offset = self._flip_parity[layer_id]
        par_htr = scale * htr + offset
        if par_htr % 4 < 2:
            return 0
        return 1

    def get_layer_name(self, layer_id, tr_idx):
        # type: (int, TrackType) -> str
        """Returns the layer name of the given track.

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
        """
        layer_name = self.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, tuple):
            # round down half integer track
            tr_parity = self.get_track_parity(layer_id, tr_idx)
            return layer_name[tr_parity]
        else:
            return layer_name

    def get_wire_bounds(self, layer_id, tr_idx, width=1, unit_mode=True):
        # type: (int, TrackType, int, bool) -> Tuple[int, int]
        """Calculate the wire bounds coordinate.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the center track index.
        width : int
            width of wire in number of tracks.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        lower : int
            the lower bound coordinate perpendicular to wire direction.
        upper : int
            the upper bound coordinate perpendicular to wire direction.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        width_unit = self.get_track_width(layer_id, width)
        center = self.track_to_coord(layer_id, tr_idx)
        return center - width_unit // 2, center + width_unit // 2

    def get_bbox(self, layer_id, tr_idx, lower, upper, width=1, unit_mode=True):
        # type: (int, TrackType, int, int, int, bool) -> BBox
        """Compute bounding box for the given wire.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the center track index.
        lower : int
            the lower coordinate along track direction.
        upper : int
            the upper coordinate along track direction.
        width : int
            width of wire in number of tracks.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        bbox : BBox
            the bounding box.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        cl, cu = self.get_wire_bounds(layer_id, tr_idx, width=width)
        if self.is_horizontal(layer_id):
            return BBox(lower, cl, upper, cu)
        return BBox(cl, lower, cu, upper)

    def get_min_track_width(self, layer_id, idc=0, iac_rms=0, iac_peak=0, l=-1,
                            bot_w=-1, top_w=-1, unit_mode=True, **kwargs):
        # type: (int, float, float, float, int, int, int, bool, Any) -> int
        """Returns the minimum track width required for the given EM specs.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        idc : float
            the DC current spec.
        iac_rms : float
            the AC RMS current spec.
        iac_peak : float
            the AC peak current spec.
        l : int
            the length of the wire in resolution units.  Use negative length
            to disable length enhancement factor.
        bot_w : int
            the bottom layer track width in resolution units.  If given, will make sure
            that the via between the two tracks meet EM specs too.
        top_w : int
            the top layer track width in resolution units.  If given, will make sure
            that the via between the two tracks meet EM specs too.
        unit_mode : bool
            deprecated parameter.
        **kwargs :
            override default EM spec parameters.

        Returns
        -------
        track_width : int
            the minimum track width in number of tracks.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        # if double patterning layer, just use any name.
        layer_name = self.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]
        if bot_w > 0:
            bot_layer_name = self.tech_info.get_layer_name(layer_id - 1)
            if isinstance(bot_layer_name, tuple):
                bot_layer_name = bot_layer_name[0]
        else:
            bot_layer_name = None
        if top_w > 0:
            top_layer_name = self.tech_info.get_layer_name(layer_id + 1)
            if isinstance(top_layer_name, tuple):
                top_layer_name = top_layer_name[0]
        else:
            top_layer_name = None

        # use binary search to find the minimum track width
        bin_iter = BinaryIterator(1, None)
        tr_dir = self.dir_tracks[layer_id]
        alt_dir = Orient2D.x if tr_dir is Orient2D.y else Orient2D.y
        bot_dir = self.dir_tracks.get(layer_id - 1, alt_dir)
        top_dir = self.dir_tracks.get(layer_id + 1, alt_dir)
        while bin_iter.has_next():
            ntr = bin_iter.get_next()
            width = self.get_track_width(layer_id, ntr)
            idc_max, irms_max, ipeak_max = self.tech_info.get_metal_em_specs(layer_name,
                                                                             width,
                                                                             l=l, **kwargs)
            if idc > idc_max or iac_rms > irms_max or iac_peak > ipeak_max:
                # check metal satisfies EM spec
                bin_iter.up()
                continue
            if bot_w > 0 and bot_dir is not tr_dir:
                if tr_dir is Orient2D.x:
                    bbox = BBox(0, 0, bot_w, width)
                else:
                    bbox = BBox(0, 0, width, bot_w)
                vinfo = self.tech_info.get_via_info(bbox, bot_layer_name, layer_name,
                                                    bot_dir.name, **kwargs)
                if (vinfo is None or idc > vinfo['idc'] or iac_rms > vinfo['iac_rms'] or
                        iac_peak > vinfo['iac_peak']):
                    bin_iter.up()
                    continue
            if top_w > 0 and top_dir is not tr_dir:
                if tr_dir is Orient2D.x:
                    bbox = BBox(0, 0, top_w, width)
                else:
                    bbox = BBox(0, 0, width, top_w)
                vinfo = self.tech_info.get_via_info(bbox, layer_name, top_layer_name,
                                                    tr_dir.name, **kwargs)
                if (vinfo is None or idc > vinfo['idc'] or iac_rms > vinfo['iac_rms'] or
                        iac_peak > vinfo['iac_peak']):
                    bin_iter.up()
                    continue

            # we got here, so all EM specs passed
            bin_iter.save()
            bin_iter.down()

        return bin_iter.get_last_save()

    def get_min_track_width_for_via(self,
                                    bot_layer: int,
                                    next_ntr: int = 1,
                                    **kwargs: Any,
                                    ) -> int:
        """Returns the minimum track width required to fit a via to the next layer.

        Parameters
        ----------
        bot_layer : int
            the layer ID.
        next_ntr : int
            the width of the track on the next layer, in track widths.
        **kwargs : Any
            Override the default EM specs and pass additional arguments that are accepted by get_min_track_width

        Returns
        -------
        track_width : int
            the minimum track width in number of tracks
        """
        next_layer_min_width_unit = self.get_track_width(layer_id=bot_layer + 1, width_ntr=next_ntr, unit_mode=True)
        return self.get_min_track_width(layer_id=bot_layer, top_w=next_layer_min_width_unit, unit_mode=True, **kwargs)

    def get_track_index_range(self,  # type: RoutingGrid
                              layer_id,  # type: int
                              lower,  # type: int
                              upper,  # type: int
                              num_space=0,  # type: TrackType
                              edge_margin=0,  # type: int
                              half_track=True,  # type: bool
                              unit_mode=True  # type: bool
                              ):
        # type: (...) -> Tuple[Optional[HalfInt], Optional[HalfInt]]
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
        unit_mode : bool
            True if lower/upper/edge_margin are given in resolution units.

        Returns
        -------
        start_track : Optional[HalfInt]
            the first track index.  None if no solution.
        end_track : Optional[HalfInt]
            the last track index.  None if no solution.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        # get start track half index
        lower_bnd = self.coord_to_nearest_track(layer_id, lower, half_track=True, mode=-1)
        start_track = self.find_next_track(layer_id, lower + edge_margin, half_track=True, mode=1)
        start_track = max(start_track, lower_bnd + num_space)
        # check if half track is allowed
        if not half_track and not start_track.is_integer():
            start_track.up()

        # get end track half index
        upper_bnd = self.coord_to_nearest_track(layer_id, upper, half_track=True, mode=1)
        end_track = self.find_next_track(layer_id, upper - edge_margin, half_track=True, mode=-1)
        end_track = min(end_track, upper_bnd - num_space)
        # check if half track is allowed
        if not half_track and not end_track.is_integer():
            end_track.down()

        if end_track < start_track:
            # no solution
            return None, None
        return start_track, end_track

    def get_overlap_tracks(self,  # type: RoutingGrid
                           layer_id,  # type: int
                           lower,  # type: int
                           upper,  # type: int
                           half_track=True,  # type: bool
                           unit_mode=True  # type: bool
                           ):
        # type: (...) -> Tuple[Optional[HalfInt], Optional[HalfInt]]
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
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        start_track : Optional[HalfInt]
            the first track index.  None if no solution.
        end_track : Optional[HalfInt]
            the last track index.  None if no solution.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        wtr = self.w_tracks[layer_id]
        lower_tr = self.find_next_track(layer_id, lower - wtr, half_track=half_track, mode=1)
        upper_tr = self.find_next_track(layer_id, upper + wtr, half_track=half_track, mode=-1)

        if upper_tr < lower_tr:
            return None, None
        return lower_tr, upper_tr

    def get_via_extensions_dim(self,  # type: RoutingGrid
                               bot_layer_id,  # type: int
                               bot_dim,  # type: int
                               top_dim,  # type: int
                               unit_mode=True,  # type: bool
                               ):
        # type: (...) -> Tuple[int, int]
        """Returns the via extension.

        Parameters
        ----------
        bot_layer_id : int
            the via bottom layer ID.
        bot_dim : int
            the bottom track width in resolution units.
        top_dim : int
            the top track width in resolution units.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        bot_ext : int
            via extension on the bottom layer.
        top_ext : int
            via extension on the top layer.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        bot_lay_name = self.get_layer_name(bot_layer_id, 0)
        top_lay_name = self.get_layer_name(bot_layer_id + 1, 0)
        bot_dir = self.dir_tracks[bot_layer_id]
        top_dir = self.dir_tracks[bot_layer_id + 1]
        if top_dir is bot_dir:
            raise ValueError('This method only works if top and bottom layers are orthogonal.')

        if bot_dir is Orient2D.x:
            vbox = BBox(0, 0, top_dim, bot_dim)
            vinfo = self._tech_info.get_via_info(vbox, bot_lay_name, top_lay_name, bot_dir.name)
            if vinfo is None:
                raise ValueError('Cannot create via')
            bot_ext = (vinfo['bot_box'].width_unit - top_dim) // 2
            top_ext = (vinfo['top_box'].height_unit - bot_dim) // 2
        else:
            vbox = BBox(0, 0, bot_dim, top_dim)
            vinfo = self._tech_info.get_via_info(vbox, bot_lay_name, top_lay_name, bot_dir.name)
            if vinfo is None:
                raise ValueError('Cannot create via')
            bot_ext = (vinfo['bot_box'].height_unit - top_dim) // 2
            top_ext = (vinfo['top_box'].width_unit - bot_dim) // 2

        return bot_ext, top_ext

    def get_via_extensions(self, bot_layer_id, bot_width, top_width, unit_mode=True):
        # type: (int, int, int, bool) -> Tuple[int, int]
        """Returns the via extension.

        Parameters
        ----------
        bot_layer_id : int
            the via bottom layer ID.
        bot_width : int
            the bottom track width in number of tracks.
        top_width : int
            the top track width in number of tracks.
        unit_mode : bool
            deprecated paramteer.

        Returns
        -------
        bot_ext : int
            via extension on the bottom layer.
        top_ext : int
            via extension on the top layer.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        bot_dim = self.get_track_width(bot_layer_id, bot_width)
        top_dim = self.get_track_width(bot_layer_id + 1, top_width)
        return self.get_via_extensions_dim(bot_layer_id, bot_dim, top_dim)

    def coord_to_track(self, layer_id, coord, unit_mode=True):
        # type: (int, int, bool) -> HalfInt
        """Convert given coordinate to track number.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track : HalfInt
            the track number
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        pitch = self.get_track_pitch(layer_id)
        q, r = divmod(coord - self._get_track_offset(layer_id), pitch)

        if r == 0:
            return HalfInt(2 * q)
        elif r == (pitch // 2):
            return HalfInt(2 * q + 1)
        else:
            raise ValueError('coordinate %.4g is not on track.' % coord)

    def find_next_track(self, layer_id, coord, tr_width=1, half_track=True,
                        mode=1, unit_mode=True):
        # type: (int, int, int, bool, int, bool) -> HalfInt
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
        mode : int
            1 to find track with both edge coordinates larger than or equal to the given one,
            -1 to find track with both edge coordinates less than or equal to the given one.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        tr_idx : HalfInt
            the center track index.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        tr_w = self.get_track_width(layer_id, tr_width)
        if mode > 0:
            return self.coord_to_nearest_track(layer_id, coord + tr_w // 2, half_track=half_track,
                                               mode=mode)
        else:
            return self.coord_to_nearest_track(layer_id, coord - tr_w // 2, half_track=half_track,
                                               mode=mode)

    def coord_to_nearest_track(self, layer_id, coord, half_track=True, mode=0,
                               unit_mode=True):
        # type: (int, int, bool, int, bool) -> HalfInt
        """Returns the track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        half_track : bool
            if True, allow half integer track numbers.
        mode : int
            the "rounding" mode.

            If mode == 0, return the nearest track (default).

            If mode == -1, return the nearest track with coordinate less
            than or equal to coord.

            If mode == -2, return the nearest track with coordinate less
            than coord.

            If mode == 1, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == 2, return the nearest track with coordinate greater
            than coord.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track : HalfInt
            the track number
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        pitch = self.get_track_pitch(layer_id)
        if half_track:
            pitch //= 2

        q, r = divmod(coord - self._get_track_offset(layer_id), pitch)

        if r == 0:
            # exactly on track
            if mode == -2:
                # move to lower track
                q -= 1
            elif mode == 2:
                # move to upper track
                q += 1
        else:
            # not on track
            if mode > 0 or (mode == 0 and r >= pitch / 2):
                # round up
                q += 1

        if not half_track:
            return HalfInt(2 * q)
        return HalfInt(q)

    def coord_to_nearest_fill_track(self, layer_id, coord, fill_config, mode=0,
                                    unit_mode=True):
        # type: (int, int, Dict[int, Any], int, bool) -> HalfInt
        """Returns the fill track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        fill_config : Dict[int, Any]
            the fill configuration dictionary.
        mode : int
            the "rounding" mode.

            If mode == 0, return the nearest track (default).

            If mode == -1, return the nearest track with coordinate less
            than or equal to coord.

            If mode == -2, return the nearest track with coordinate less
            than coord.

            If mode == 1, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == 2, return the nearest track with coordinate greater
            than coord.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track : HalfInt
            the track number
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        tr_w, tr_sp, _, _ = fill_config[layer_id]

        num_htr = round(2 * (tr_w + tr_sp))
        fill_pitch = num_htr * self.get_track_pitch(layer_id) // 2
        fill_pitch2 = fill_pitch // 2
        fill_q, fill_r = divmod(coord - fill_pitch2, fill_pitch)

        if fill_r == 0:
            # exactly on track
            if mode == -2:
                # move to lower track
                fill_q -= 1
            elif mode == 2:
                # move to upper track
                fill_q += 1
        else:
            # not on track
            if mode > 0 or (mode == 0 and fill_r >= fill_pitch2):
                # round up
                fill_q += 1

        return self.coord_to_track(layer_id, fill_q * fill_pitch + fill_pitch2)

    def transform_track(self,  # type: RoutingGrid
                        layer_id,  # type: int
                        track_idx,  # type: TrackType
                        dx=0,  # type: int
                        dy=0,  # type: int
                        orient='R0',  # type: str
                        unit_mode=True,  # type: bool
                        ):
        # type: (...) -> HalfInt
        """Transform the given track index.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        track_idx : TrackType
            the track index.
        dx : int
            X shift.
        dy : int
            Y shift.
        orient : str
            orientation.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        new_track_idx : HalfInt
            the transformed track index.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        is_x = self.is_horizontal(layer_id)
        if is_x:
            hidx_shift = int(2 * self.coord_to_track(layer_id, dy)) + 1
        else:
            hidx_shift = int(2 * self.coord_to_track(layer_id, dx)) + 1

        if orient == 'R0':
            hidx_scale = 1
        elif orient == 'R180':
            hidx_scale = -1
        elif orient == 'MX':
            hidx_scale = -1 if is_x else 1
        elif orient == 'MY':
            hidx_scale = 1 if is_x else -1
        else:
            raise ValueError('Unsupported orientation: %s' % orient)

        old_hidx = round(track_idx * 2 + 1)
        return HalfInt(old_hidx * hidx_scale + hidx_shift - 1)

    def track_to_coord(self, layer_id, track_idx, unit_mode=True):
        # type: (int, TrackType, bool) -> int
        """Convert given track number to coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        track_idx : TrackType
            the track number.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        coord : int
            the coordinate perpendicular to track direction.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        pitch = self.get_track_pitch(layer_id)
        return round(pitch * track_idx) + self._get_track_offset(layer_id)

    def interval_to_track(self,  # type: RoutingGrid
                          layer_id,  # type: int
                          intv,  # type: Tuple[int, int]
                          unit_mode=True,  # type: bool
                          ):
        # type: (...) -> Tuple[HalfInt, int]
        """Convert given coordinates to track number and width.

        Parameters
        ----------
        layer_id : int
            the layer number.
        intv : Tuple[int, int]
            lower and upper coordinates perpendicular to the track direction.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        track : HalfInt
            the track number
        width : int
            the track width, in number of tracks.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        start, stop = intv
        track = self.coord_to_track(layer_id, (start + stop) // 2)
        width = stop - start

        # binary search to take width override into account
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_ntr = bin_iter.get_next()
            cur_w = self.get_track_width(layer_id, cur_ntr)
            if cur_w == width:
                return track, cur_ntr
            elif cur_w > width:
                bin_iter.down()
            else:
                bin_iter.up()

        # never found solution; width is not quantized.
        raise ValueError('Interval {} on layer {} width not quantized'.format(intv, layer_id))

    def copy(self):
        # type: () -> RoutingGrid
        """Returns a deep copy of this RoutingGrid."""
        cls = self.__class__
        result = cls.__new__(cls)
        attrs = result.__dict__
        attrs['_tech_info'] = self._tech_info
        attrs['_resolution'] = self._resolution
        attrs['_layout_unit'] = self._layout_unit
        attrs['_flip_parity'] = self._flip_parity.copy()
        attrs['_ignore_layers'] = self._ignore_layers.copy()
        attrs['layers'] = list(self.layers)
        attrs['sp_tracks'] = self.sp_tracks.copy()
        attrs['dir_tracks'] = self.dir_tracks.copy()
        attrs['offset_tracks'] = {}
        attrs['w_tracks'] = self.w_tracks.copy()
        attrs['max_num_tr_tracks'] = self.max_num_tr_tracks.copy()
        attrs['block_pitch'] = self.block_pitch.copy()
        attrs['w_override'] = self.w_override.copy()
        attrs['private_layers'] = list(self.private_layers)
        for lay in self.layers:
            attrs['w_override'][lay] = self.w_override[lay].copy()

        return result

    def ignore_layers_under(self, layer_id):
        # type: (int) -> None
        """Ignore all layers under the given layer (inclusive) when calculating block pitches.

        Parameters
        ----------
        layer_id : int
            ignore this layer and below.
        """
        for lay in self.layers:
            if lay > layer_id:
                break
            self._ignore_layers.add(lay)

    def add_new_layer(self, layer_id, tr_space, tr_width, direction,
                      max_num_tr=100, override=False, unit_mode=True, is_private=True):
        # type: (int, int, int, str, int, bool, bool, bool) -> None
        """Add a new private layer to this RoutingGrid.

        This method is used to add customized routing grid per template on lower level layers.
        The new layers doesn't necessarily need to follow alternating track direction, however,
        if you do this you cannot connect to adjacent level metals.

        Note: do not use this method to add/modify top level layers, as it does not calculate
        block pitch.

        Parameters
        ----------
        layer_id : int
            the new layer ID.
        tr_space : float
            the track spacing, in layout units.
        tr_width : float
            the track width, in layout units.
        direction : str
            track direction.  'x' for horizontal, 'y' for vertical.
        max_num_tr : int
            maximum track width in number of tracks.
        override : bool
            True to override existing layers if they already exist.
        unit_mode : bool
            deprecated parameter.
        is_private : bool
            True if this is a private layer.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported')

        self._ignore_layers.discard(layer_id)

        sp_unit = -(-tr_space // 2) * 2
        w_unit = -(-tr_width // 2) * 2
        dir_enum = Orient2D[direction]

        if layer_id in self.sp_tracks:
            # double check to see if we actually need to modify layer
            w_cur = self.w_tracks[layer_id]
            sp_cur = self.sp_tracks[layer_id]
            dir_cur = self.dir_tracks[layer_id]

            if w_cur == w_unit and sp_cur == sp_unit and dir_cur is dir_enum:
                # everything is the same, just return
                return

            if not override:
                raise ValueError('Layer %d already on routing grid.' % layer_id)
        else:
            self.layers.append(layer_id)
            self.layers.sort()

        if is_private and layer_id not in self.private_layers:
            self.private_layers.append(layer_id)
            self.private_layers.sort()

        self.sp_tracks[layer_id] = sp_unit
        self.w_tracks[layer_id] = w_unit
        self.dir_tracks[layer_id] = dir_enum
        self.w_override[layer_id] = {}
        self.max_num_tr_tracks[layer_id] = max_num_tr
        if layer_id not in self._flip_parity:
            self._flip_parity[layer_id] = (1, 0)

    def set_track_offset(self, layer_id, offset, unit_mode=True):
        # type: (int, int, bool) -> None
        """Set track offset for this RoutingGrid.

        Parameters
        ----------
        layer_id : int
            the routing layer ID.
        offset : int
            the track offset.
        unit_mode : bool
            deprecated parameter.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        self.offset_tracks[layer_id] = offset

    def add_width_override(self, layer_id, width_ntr, tr_width, unit_mode=True):
        # type: (int, int, int, bool) -> None
        """Add width override.

        NOTE: call this method only directly after you construct the RoutingGrid.  Do not
        use this to modify an existing grid.

        Parameters
        ----------
        layer_id : int
            the new layer ID.
        width_ntr : int
            the width in number of tracks.
        tr_width : int
            the actual width.
        unit_mode : bool
            deprecated parameter.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        if width_ntr == 1:
            raise ValueError('Cannot override width_ntr=1.')

        if layer_id not in self.w_override:
            self.w_override[layer_id] = {width_ntr: tr_width}
        else:
            self.w_override[layer_id][width_ntr] = tr_width
