# -*- coding: utf-8 -*-

"""This module defines ResArrayBase, an abstract class that draws resistor arrays.

This module also define some simple subclasses of ResArrayBase.
"""

from typing import TYPE_CHECKING, Dict, Set, Tuple, Union, Any

import abc
from itertools import chain

from bag.math import lcm
from bag.layout.util import BBox
from bag.layout.routing import TrackID, WireArray
from bag.layout.template import TemplateBase

from .base import ResTech, AnalogResCore, AnalogResBoundary
from ..analog_core.substrate import SubstrateContact

if TYPE_CHECKING:
    from bag.layout.routing import RoutingGrid, Port
    from bag.layout.template import TemplateDB
    from bag.layout.core import TechInfo


class ResArrayBaseInfo(object):
    """A class that provides information to assist in AnalogBase layout calculations.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    sub_type : str
        the substrate type.  Either 'ptap' or 'ntap'.
    threshold : str
        the substrate threshold flavor.
    **kwargs :
        optional arguments.  Right now support:

        min_tracks : Optional[Tuple[int, ...]]
            minimum number of tracks per layer in the unit cell. If None, Defaults to all 1's.
            This parameter also represents the number of layers that will be used.
        em_specs : Optional[Dict[str, Any]]
            resistor EM specifications dictionary.
        grid_type : str
            the resistor private routing grid name.
        top_layer : Optional[int]
            The top metal layer this block is quantized by.  Defaults to the last layer if it is on
            the global routing grid, or the layer above that if otherwise.
            If the top metal layer is only one layer above the private routing grid, then this will
            be a primitive template; self.size will be None, and only one dimension is quantized.
            Otherwise, this will be a standard template, and both width and height will be quantized
            according to the block size.
        res_type : str
            the resistor type.
        ext_dir : str
            resistor core extension direction.
        max_blk_ext : int
            maximum number of block pitches we can extend for primitives.
        options : Optional[Dict[str, Any]]
            custom options for resistor primitives.
        connect_up : bool
            True if the last used layer needs to be able to connect to the layer above.
            This options will make sure that the width of the last track is wide enough to support
            the inter-layer via.
        half_blk_x : bool
            True to allow half-block width.  Defaults to True.
        half_blk_y : bool
            True to allow half-block height.  Defaults to True.
    """

    def __init__(self, grid, sub_type, threshold, **kwargs):
        # type: (RoutingGrid, str, str, **kwargs) -> None
        min_tracks = kwargs.get('min_tracks', None)
        em_specs = kwargs.get('em_specs', None)
        grid_type = kwargs.get('grid_type', 'standard')
        top_layer = kwargs.get('top_layer', None)
        self.res_type = kwargs.get('res_type', 'standard')
        self.ext_dir = kwargs.get('ext_dir', '')
        self.max_blk_ext = kwargs.get('max_blk_ext', 100)
        self.options = kwargs.get('options', None)
        self.connect_up = kwargs.get('connect_up', False)
        self.half_blk_x = kwargs.get('half_blk_x', True)
        self.half_blk_y = kwargs.get('half_blk_y', True)
        self.sub_type = sub_type
        self.threshold = threshold

        if em_specs is None:
            self.em_specs = {}
        else:
            self.em_specs = em_specs

        tech_params = grid.tech_info.tech_params
        self._tech_cls = tech_params['layout']['res_tech_class']  # type: ResTech
        self.bot_layer = self._tech_cls.get_bot_layer()
        if grid_type is None:
            w_tr, sp_tr = grid.get_track_info(self.bot_layer, unit_mode=False)
            self._grid_layers = [[self.bot_layer, w_tr, sp_tr, 'x']]
        else:
            self._grid_layers = tech_params['layout']['analog_res'][grid_type]

        # modify resistor layer routing grid.
        self.grid = grid.copy()
        min_tracks_default = []
        for lay_id, tr_w, tr_sp, tr_dir in self._grid_layers:
            self.grid.add_new_layer(lay_id, tr_sp, tr_w, tr_dir, override=True)
            min_tracks_default.append(1)
        self.grid.update_block_pitch()

        if min_tracks is None:
            self.min_tracks = tuple(min_tracks_default)
        else:
            self.min_tracks = min_tracks

        min_top_layer = max(self.grid.top_private_layer + 1,
                            self.bot_layer + len(self.min_tracks) - 1)

        if top_layer is None:
            self.top_layer = min_top_layer
        elif top_layer < min_top_layer:
            raise ValueError('Cannot set top_layer below %d' % min_top_layer)
        else:
            self.top_layer = top_layer

    def get_res_info(self, l_unit, w_unit, **kwargs):
        # type: (int, int, **kwargs) -> Dict[str, Any]
        res_type = kwargs.get('res_type', self.res_type)
        sub_type = kwargs.get('sub_type', self.sub_type)
        threshold = kwargs.get('threshold', self.threshold)
        min_tracks = kwargs.get('min_tracks', self.min_tracks)
        em_specs = kwargs.get('em_specs', self.em_specs)
        ext_dir = kwargs.get('ext_dir', self.ext_dir)
        max_blk_ext = kwargs.get('max_blk_ext', self.max_blk_ext)
        connect_up = kwargs.get('connect_up', self.connect_up)
        options = kwargs.get('options', self.options)
        return self._tech_cls.get_res_info(self.grid, l_unit, w_unit, res_type, sub_type,
                                           threshold, min_tracks, em_specs, ext_dir,
                                           max_blk_ext=max_blk_ext, connect_up=connect_up,
                                           options=options)

    def get_place_info(self,
                       l_unit,  # type: int
                       w_unit,  # type: int
                       nx,  # type: int
                       ny,  # type: int
                       min_width=0,  # type: int
                       min_height=0,  # type: int
                       update_grid=False,  # type: bool
                       **kwargs):
        # type: (...) -> Tuple[int, int, int, int, Dict[str, Any]]
        top_layer = kwargs.get('top_layer', self.top_layer)
        half_blk_x = kwargs.get('half_blk_x', self.half_blk_x)
        half_blk_y = kwargs.get('half_blk_y', self.half_blk_y)

        res_info = self.get_res_info(l_unit, w_unit, **kwargs)
        w_edge, h_edge = res_info['w_edge'], res_info['h_edge']
        w_core, h_core = res_info['w_core'], res_info['h_core']

        wblk, hblk = self.grid.get_block_size(top_layer, unit_mode=True,
                                              half_blk_x=half_blk_x,
                                              half_blk_y=half_blk_y)
        wblk_res, hblk_res = self._tech_cls.get_block_pitch()
        wblk = lcm([wblk, wblk_res])
        hblk = lcm([hblk, hblk_res])
        warr = w_edge * 2 + w_core * nx
        harr = h_edge * 2 + h_core * ny
        wtot = -(-max(min_width, warr) // wblk) * wblk
        htot = -(-max(min_height, harr) // hblk) * hblk
        dx = (wtot - warr) // 2
        dy = (htot - harr) // 2
        # if wtot - warr is odd number of resistor block pitch, we could be misaligned.
        # fix by adding extra block
        if dx % wblk_res != 0:
            if (wblk // wblk_res) % 2 == 0:
                raise ValueError('Cannot center resistor array primitives.  See developer.')
            wtot += wblk
            dx = (wtot - warr) // 2
        if dy % hblk_res != 0:
            if (hblk // hblk_res) % 2 == 0:
                raise ValueError('Cannot center resistor array primitives.  See developer.')
            htot += hblk
            dy = (htot - harr) // 2

        if update_grid:
            for lay_id, tr_w, tr_sp, tr_dir in self._grid_layers:
                offset = dy if self.grid.get_direction(lay_id) == 'x' else dx
                self.grid.set_track_offset(lay_id, offset, unit_mode=True)

        return dx, dy, wtot, htot, res_info

    def get_res_length_bounds(self, **kwargs):
        # type: (**kwargs) -> Tuple[int, int]
        res_type = kwargs.get('res_type', self.res_type)

        lmin, lmax = self.grid.tech_info.get_res_length_bounds(res_type)
        res = self.grid.resolution
        return int(round(lmin / res)), int(round(lmax / res))


class ResArrayBase(TemplateBase, metaclass=abc.ABCMeta):
    """An abstract template that draws analog resistors array and connections.

    This template assumes that the ressistor array uses 4 routing layers, with
    directions x/y/x/y.  The lower 2 routing layers is used to connect between
    adjacent resistors, and pin will be drawn on the upper 2 routing layers.

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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._layout_info = None  # type: ResArrayBaseInfo
        self._bot_port = None  # type: Port
        self._top_port = None  # type: Port
        self._core_offset = None  # type: Tuple[int, int]
        self._core_pitch = None  # type: Tuple[int, int]
        self._num_tracks = None  # type: Tuple[int, ...]
        self._num_corner_tracks = None  # type: Tuple[int, ...]
        self._w_tracks = None  # type: Tuple[int, ...]
        self._well_width = None

    @classmethod
    def get_port_layer_id(cls, tech_info):
        # type: (TechInfo) -> int
        return tech_info.tech_params['layout']['res_tech_class'].get_bot_layer()

    @classmethod
    def get_top_layer(cls, tech_info, grid_type='standard'):
        # type: (TechInfo, str) -> int
        """Returns the top layer ID for the given resistor routing grid setting."""
        grid_layers = tech_info.tech_params['layout']['analog_res'][grid_type]
        return grid_layers[-1][0] + 1

    @property
    def num_tracks(self):
        # type: () -> Tuple[int, ...]
        """Returns the number of tracks per resistor block on each routing layer."""
        return self._num_tracks

    @property
    def bot_layer_id(self):
        # type: () -> int
        """Returns the bottom resistor routing layer ID."""
        return self.get_port_layer_id(self.grid.tech_info)

    @property
    def w_tracks(self):
        # type: () -> Tuple[int, ...]
        """Returns the track width on each routing layer, in number of tracks."""
        return self._w_tracks

    @property
    def bot_port_idx(self):
        # type: () -> int
        """Returns the relative track index of the bottom resistor port."""
        return self._bot_port.get_pins()[0].track_id.base_index

    @property
    def top_port_idx(self):
        # type: () -> int
        """Returns the relative track index of the top resistor port."""
        return self._top_port.get_pins()[0].track_id.base_index

    @property
    def res_unit_size(self):
        # type: () -> Tuple[int, int]
        """Returns the size of a unit resistor block in resolution units"""
        return self._core_pitch

    @property
    def core_offset(self):
        # type: () -> Tuple[int, int]
        """Returns the core resistor block offset."""
        return self._core_offset

    def get_well_width(self, unit_mode=False):
        # type: (bool) -> Union[float, int]
        """Returns the NW/PW width in this block."""
        if unit_mode:
            return self._well_width
        return self._well_width * self.grid.resolution

    def get_res_ports(self, row_idx, col_idx):
        # type: (int, int) -> Tuple[WireArray, WireArray]
        """Returns the port of the given resistor.

        Parameters
        ----------
        row_idx : int
            the resistor row index.  0 is the bottom row.
        col_idx : int
            the resistor column index.  0 is the left-most column.

        Returns
        -------
        bot_warr : WireArray
            the bottom port as WireArray.
        top_warr : WireArray
            the top port as WireArray.
        """
        loc = (self._core_offset[0] + self._core_pitch[0] * col_idx,
               self._core_offset[1] + self._core_pitch[1] * row_idx)
        bot_port = self._bot_port.transform(self.grid, loc=loc, unit_mode=True)
        top_port = self._top_port.transform(self.grid, loc=loc, unit_mode=True)
        return bot_port.get_pins()[0], top_port.get_pins()[0]

    def get_track_offsets(self, row_idx, col_idx):
        # type: (int, int) -> Tuple[float, ...]
        """Compute track offsets on each routing layer for the given resistor block.

        Parameters
        ----------
        row_idx : int
            the row index.  0 is the bottom row.
        col_idx : int
            the column index.  0 is the left-most column.

        Returns
        -------
        offsets : Tuple[float, ...]
            track index offset on each routing layer.
        """
        dx = self._core_offset[0] + self._core_pitch[0] * col_idx
        dy = self._core_offset[1] + self._core_pitch[1] * row_idx
        bot_layer = self.bot_layer_id
        offsets = []
        for lay in range(bot_layer, bot_layer + len(self._num_tracks)):
            dim = dx if self.grid.get_direction(lay) == 'y' else dy
            pitch2 = self.grid.get_track_pitch(lay, unit_mode=True) // 2
            offsets.append(self.grid.coord_to_track(lay, dim + pitch2, unit_mode=True))

        return tuple(offsets)

    def get_abs_track_index(self, layer_id, cell_idx, tr_idx, mode=0):
        # type: (int, int, Union[float, int], int) -> float
        """Compute absolute track index from relative track index.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        cell_idx : int
            the row or column index.  0 is the bottom row/left-most column.
        tr_idx : Union[int, float]
            the track index within the given cell.
        mode : int
            the rounding mode

        Returns
        -------
        abs_idx : float
            the absolute track index in this template.
        """
        dim_idx = 1 if self.grid.get_direction(layer_id) == 'x' else 0
        delta = self._core_offset[dim_idx] + self._core_pitch[dim_idx] * cell_idx
        half_pitch = self.grid.get_track_pitch(layer_id, unit_mode=True) // 2
        coord = delta + (int(round(2 * tr_idx)) + 1) * half_pitch
        return self.grid.coord_to_nearest_track(layer_id, coord, half_track=True,
                                                mode=mode, unit_mode=True)

    def draw_array(self, l, w, sub_type, threshold, nx=1, ny=1, **kwargs):
        # type: (float, float, str, str, int, int, **kwargs) -> None
        """Draws the resistor array.

        This method updates the RoutingGrid with resistor routing layers, then add
        resistor instances to this template.

        Parameters
        ----------
        l : float
            unit resistor length, in meters.
        w : float
            unit resistor width, in meters.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold flavor.
        nx : int
            number of resistors in a row.
        ny : int
            number of resistors in a column.
        **kwargs :
            optional arguments.  Right now support:

            min_tracks : Optional[Tuple[int, ...]]
                minimum number of tracks per layer in the unit cell. If None, Defaults to all 1's.
                This parameter also represents the number of layers that will be used.
            em_specs : Optional[Dict[str, Any]]
                resistor EM specifications dictionary.
            grid_type : str
                the resistor private routing grid name.
            top_layer : Optional[int]
                The top metal layer this block is quantized by.  Defaults to the last layer if it
                is on the global routing grid, or the layer above that if otherwise.
                If the top metal layer is only one layer above the private routing grid, then this
                will be a primitive template; self.size will be None, and only one dimension is
                quantized.
                Otherwise, this will be a standard template, and both width and height will be
                quantized according to the block size.
            res_type : str
                the resistor type.
            ext_dir : str
                resistor core extension direction.
            max_blk_ext : int
                maximum number of block pitches we can extend for primitives.
            options : Optional[Dict[str, Any]]
                custom options for resistor primitives.
            connect_up : bool
                True if the last used layer needs to be able to connect to the layer above.
                This options will make sure that the width of the last track is wide enough to
                support the inter-layer via.
            half_blk_x : bool
                True to allow half-block width.  Defaults to True.
            half_blk_y : bool
                True to allow half-block height.  Defaults to True.
            min_width : int
               Minimum ResArrayBase width, in resolution units.
            min_height : int
                Minimum ResArraybase height, in resolution units.
        """
        min_width = kwargs.pop('min_width', 0)
        min_height = kwargs.pop('min_height', 0)

        # create ResArrayBaseInfo object, and update RoutingGrid
        res = self.grid.resolution
        lay_unit = self.grid.layout_unit
        l_unit = int(round(l / lay_unit / res))
        w_unit = int(round(w / lay_unit / res))
        self._layout_info = ResArrayBaseInfo(self.grid, sub_type, threshold, **kwargs)
        self.grid = self._layout_info.grid

        # compute template quantization and coordinates
        tmp = self._layout_info.get_place_info(l_unit, w_unit, nx, ny, min_width=min_width,
                                               min_height=min_height, update_grid=True)
        dx, dy, wtot, htot, res_info = tmp
        w_edge, h_edge = res_info['w_edge'], res_info['h_edge']
        w_core, h_core = res_info['w_core'], res_info['h_core']

        self._core_offset = dx + w_edge, dy + h_edge
        self._core_pitch = w_core, h_core
        self._num_tracks = tuple(res_info['num_tracks'])
        self._num_corner_tracks = tuple(res_info['num_corner_tracks'])
        self._w_tracks = tuple(res_info['track_widths'])

        # make template masters
        core_params = dict(res_info=res_info)
        core_master = self.new_template(params=core_params, temp_cls=AnalogResCore)
        lr_params = core_master.get_boundary_params('lr')
        lr_master = self.new_template(params=lr_params, temp_cls=AnalogResBoundary)
        tb_params = core_master.get_boundary_params('tb')
        tb_master = self.new_template(params=tb_params, temp_cls=AnalogResBoundary)
        corner_params = core_master.get_boundary_params('corner')
        corner_master = self.new_template(params=corner_params, temp_cls=AnalogResBoundary)

        # place core
        for row in range(ny):
            for col in range(nx):
                cur_name = 'XCORE%d' % (col + nx * row)
                cur_loc = (dx + w_edge + col * w_core, dy + h_edge + row * h_core)
                self.add_instance(core_master, inst_name=cur_name, loc=cur_loc, unit_mode=True)
                if row == 0 and col == 0:
                    self._bot_port = core_master.get_port('bot')
                    self._top_port = core_master.get_port('top')

        # place boundaries
        # bottom-left corner
        inst_bl = self.add_instance(corner_master, inst_name='XBL', loc=(dx, dy), unit_mode=True)
        # bottom edge
        self.add_instance(tb_master, inst_name='XB', loc=(dx + w_edge, dy), nx=nx, spx=w_core,
                          unit_mode=True)
        # bottom-right corner
        loc = (dx + 2 * w_edge + nx * w_core, dy)
        self.add_instance(corner_master, inst_name='XBR', loc=loc, orient='MY', unit_mode=True)
        # left edge
        loc = (dx, dy + h_edge)
        well_xl = lr_master.get_well_left(unit_mode=True)
        self.add_instance(lr_master, inst_name='XL', loc=loc, ny=ny, spy=h_core, unit_mode=True)
        # right edge
        loc = (dx + 2 * w_edge + nx * w_core, dy + h_edge)
        well_xr = loc[0] - well_xl
        self._well_width = well_xr - well_xl
        self.add_instance(lr_master, inst_name='XR', loc=loc, orient='MY', ny=ny, spy=h_core,
                          unit_mode=True)
        # top-left corner
        loc = (dx, dy + 2 * h_edge + ny * h_core)
        self.add_instance(corner_master, inst_name='XTL', loc=loc, orient='MX', unit_mode=True)
        # top edge
        loc = (dx + w_edge, dy + 2 * h_edge + ny * h_core)
        self.add_instance(tb_master, inst_name='XT', loc=loc, orient='MX', nx=nx, spx=w_core,
                          unit_mode=True)
        # top-right corner
        loc = (dx + 2 * w_edge + nx * w_core, dy + 2 * h_edge + ny * h_core)
        inst_tr = self.add_instance(corner_master, inst_name='XTR', loc=loc, orient='R180',
                                    unit_mode=True)

        # set array box and size
        top_layer = self._layout_info.top_layer
        self.array_box = inst_bl.array_box.merge(inst_tr.array_box)
        bnd_box = BBox(0, 0, wtot, htot, res, unit_mode=True)
        if self.grid.size_defined(top_layer):
            self.set_size_from_bound_box(top_layer, bnd_box)
        else:
            self.prim_top_layer = top_layer
            self.prim_bound_box = bnd_box

        self.add_cell_boundary(self.bound_box)

        # draw device blockages
        self.grid.tech_info.draw_device_blockage(self)


class TerminationCore(ResArrayBase):
    """An template for creating termination resistors.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        ResArrayBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=1,
            res_type='standard',
            grid_type='standard',
            em_specs={},
            ext_dir='',
            show_pins=True,
            top_layer=None,
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
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            grid_type='the resistor routing grid type.',
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
            show_pins='True to show pins.',
            top_layer='The top level metal layer.  None for primitive template.',
        )

    def draw_layout(self):
        # type: () -> None

        # draw array
        nx = self.params['nx']
        ny = self.params['ny']
        em_specs = self.params.pop('em_specs')
        show_pins = self.params.pop('show_pins')

        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')

        div_em_specs = em_specs.copy()
        for key in ('idc', 'iac_rms', 'iac_peak'):
            if key in div_em_specs:
                div_em_specs[key] = div_em_specs[key] / ny
            else:
                div_em_specs[key] = 0.0

        self.draw_array(em_specs=div_em_specs, **self.params)

        # connect row resistors
        port_wires = [[], [], []]
        for row_idx in range(ny):
            for col_idx in range(nx - 1):
                ports_l = self.get_res_ports(row_idx, col_idx)
                ports_r = self.get_res_ports(row_idx, col_idx + 1)
                con_par = (col_idx + row_idx) % 2
                mid_wire = self.connect_wires([ports_l[con_par], ports_r[con_par]])
                if col_idx == 0:
                    port_wires[0].append(ports_l[1 - con_par])
                if col_idx == nx - 2:
                    port_wires[2].append(ports_r[1 - con_par])
                if col_idx == (nx // 2) - 1:
                    port_wires[1].append(mid_wire[0])

        lay_offset = self.bot_layer_id
        last_dir = 'x'
        for lay_idx in range(1, len(self.w_tracks)):
            cur_lay = lay_idx + lay_offset
            cur_w = self.w_tracks[lay_idx]
            cur_dir = self.grid.get_direction(cur_lay)
            if cur_dir != last_dir:
                # layer direction is orthogonal
                if cur_dir == 'y':
                    # connect all horizontal wires in last layer to one vertical wire
                    for warrs_idx in range(3):
                        cur_warrs = port_wires[warrs_idx]
                        tidx = self.grid.coord_to_nearest_track(cur_lay, cur_warrs[0].middle,
                                                                half_track=True)
                        tid = TrackID(cur_lay, tidx, width=cur_w)
                        port_wires[warrs_idx] = [self.connect_to_tracks(cur_warrs, tid)]
                else:
                    # draw one horizontal wire in middle of each row, then connect last vertical
                    # wire to it.  this way we distribute currents evenly.
                    cur_p = self.num_tracks[lay_idx]
                    # relative base index.  Round down if we have half-integer number of tracks
                    base_idx_rel = (int(round(cur_p * 2)) // 2 - 1) / 2
                    base_idx = self.get_abs_track_index(cur_lay, 0, base_idx_rel)
                    tid = TrackID(cur_lay, base_idx, width=cur_w, num=ny, pitch=cur_p)
                    for warrs_idx in range(3):
                        port_wires[warrs_idx] = self.connect_to_tracks(port_wires[warrs_idx], tid,
                                                                       min_len_mode=0)
            else:
                # layer direction is the same.  Strap wires to current layer.
                for warrs_idx in range(3):
                    cur_warrs = port_wires[warrs_idx]
                    new_warrs = [self.strap_wires(warr, cur_lay, tr_w_list=[cur_w],
                                                  min_len_mode_list=[0])
                                 for warr in cur_warrs]
                    port_wires[warrs_idx] = new_warrs

            last_dir = cur_dir

        self.add_pin('inp', port_wires[0], show=show_pins)
        self.add_pin('inn', port_wires[2], show=show_pins)
        self.add_pin('incm', port_wires[1], show=show_pins)


class Termination(TemplateBase):
    """An template for creating termination resistors.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=1,
            ext_dir='',
            res_type='standard',
            grid_type='standard',
            em_specs={},
            show_pins=True,
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
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width. Set to 0 to disable drawing substrate contact.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            grid_type='the resistor routing grid type.',
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
            show_pins='True to show pins.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        res_type = res_params['res_type']
        grid_type = self.params['grid_type']
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']
        show_pins = self.params['show_pins']

        # force TerminationCore to be quantized
        top_layer = ResArrayBase.get_top_layer(self.grid.tech_info, grid_type=grid_type) + 1
        res_params['top_layer'] = top_layer

        res_master = self.new_template(params=res_params, temp_cls=TerminationCore)
        if sub_w == 0:
            # do not draw substrate contact.
            inst = self.add_instance(res_master, inst_name='XRES', loc=(0, 0), unit_mode=True)
            for port_name in inst.port_names_iter():
                self.reexport(inst.get_port(port_name), show=show_pins)
            self.array_box = inst.array_box
            self.set_size_from_bound_box(res_master.top_layer, res_master.bound_box)
        else:
            # draw contact and move array up
            top_layer, nx_arr, ny_arr = res_master.size
            w_pitch, h_pitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
            sub_params = dict(
                top_layer=top_layer,
                lch=sub_lch,
                w=sub_w,
                sub_type=sub_type,
                threshold=self.params['threshold'],
                well_width=res_master.get_well_width(),
                show_pins=False,
                is_passive=True,
                tot_width_parity=nx_arr % 2,
            )
            sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
            sub_box = sub_master.bound_box
            ny_shift = -(-sub_box.height_unit // h_pitch)

            # compute substrate X coordinate so substrate is on its own private horizontal pitch
            sub_x_pitch, _ = sub_master.grid.get_size_pitch(sub_master.size[0], unit_mode=True)
            sub_x = ((w_pitch * nx_arr - sub_box.width_unit) // 2 // sub_x_pitch) * sub_x_pitch

            bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(sub_x, 0),
                                         unit_mode=True)
            res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_shift * h_pitch),
                                         unit_mode=True)
            top_yo = (ny_arr + 2 * ny_shift) * h_pitch
            top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(sub_x, top_yo),
                                         orient='MX', unit_mode=True)

            # connect implant layers of resistor array and substrate contact together
            for lay in self.grid.tech_info.get_implant_layers(sub_type, res_type=res_type):
                self.add_rect(lay, self.get_rect_bbox(lay))

            # export supplies and recompute array_box/size
            port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
            self.reexport(bot_inst.get_port(port_name), show=show_pins)
            self.reexport(top_inst.get_port(port_name), show=show_pins)
            self.size = top_layer, nx_arr, ny_arr + 2 * ny_shift
            self.array_box = bot_inst.array_box.merge(top_inst.array_box)
            self.add_cell_boundary(self.bound_box)

            for port_name in res_inst.port_names_iter():
                self.reexport(res_inst.get_port(port_name), show=show_pins)


class ResLadderCore(ResArrayBase):
    """An template for creating a resistor ladder from VDD to VSS.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        ResArrayBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            nx=2,
            ny=2,
            ndum=1,
            res_type='standard',
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
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            ndum='number of dummy resistors.',
            res_type='the resistor type.',
        )

    def draw_layout(self):
        # type: () -> None
        l = self.params['l']
        w = self.params['w']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        nx = self.params['nx']
        ny = self.params['ny']
        ndum = self.params['ndum']
        res_type = self.params['res_type']

        hcon_space = 0
        vcon_space = 0

        # error checking
        if nx % 2 != 0 or nx <= 0:
            raise ValueError('number of resistors in a row must be even and positive.')
        if ny % 2 != 0 or ny <= 0:
            raise ValueError('number of resistors in a column must be even and positive.')

        # copy routing grid before calling draw_array so substrate contact can have its own grid
        min_tracks = (4 + 2 * hcon_space, 7 + vcon_space, nx, 1)
        self.draw_array(l, w, sub_type, threshold, nx=nx + 2 * ndum, ny=ny + 2 * ndum,
                        min_tracks=min_tracks, res_type=res_type, grid_type='low_res', ext_dir='x')

        # export supplies and recompute array_box/size
        tmp = self._draw_metal_tracks(nx, ny, ndum, hcon_space)
        hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup = tmp
        self._connect_ladder(nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)

    def _connect_ladder(self, nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        # connect main ladder
        for row_idx in range(ndum, ny + ndum):
            rmod = row_idx - ndum
            for col_idx in range(ndum, nx + ndum):
                if (col_idx == ndum and rmod % 2 == 1) or \
                        (col_idx == nx - 1 + ndum and rmod % 2 == 0):
                    mode = 1 if row_idx == ny + ndum - 1 else 0
                    self._connect_tb(row_idx, col_idx, ndum, tp_idx, hcon_idx_list,
                                     vcon_idx_list, xm_bot_idx, mode=mode)
                if col_idx != nx - 1 + ndum:
                    self._connect_lr(row_idx, col_idx, nx, ndum, tp_idx, bp_idx, hcon_idx_list,
                                     vcon_idx_list, xm_bot_idx)

        # connect to ground
        self._connect_tb(ndum - 1, ndum, ndum, tp_idx, hcon_idx_list,
                         vcon_idx_list, xm_bot_idx, mode=-1)
        # connect to supplies
        self._connect_ground(nx, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)
        self._connect_power(ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)

        # connect horizontal dummies
        for row_idx in range(ny + 2 * ndum):
            if row_idx < ndum or row_idx >= ny + ndum:
                col_iter = range(nx + 2 * ndum)
            else:
                col_iter = chain(range(ndum), range(nx + ndum, nx + 2 * ndum))
            for col_idx in col_iter:
                conn_tb = col_idx < ndum or col_idx >= nx + ndum
                self._connect_dummy(row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list,
                                    vcon_idx_list)

    def _connect_power(self, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(ny + ndum, ndum)
        _, vm_prev, _, _ = self.get_track_offsets(ndum, ndum - 1)
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        hconn = hcon_idx_list[0]
        vm_idx_list = [vm_off + vcon_idx_list[2], vm_prev + vcon_idx_list[-3],
                       vm_prev + vcon_idx_list[-2]]
        xm_idx_list = [xm_off + xm_bot_idx + idx for idx in range(num_xm_sup)]
        for vm_idx in vm_idx_list:
            # connect supply to vm layer
            self.add_via_on_grid(hm_layer, hm_off + hconn, vm_idx)
            # connect supply to xm layer
            for xm_idx in xm_idx_list:
                self.add_via_on_grid(vm_layer, vm_idx, xm_idx)

    def _connect_ground(self, nx, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        xm_prev = self.get_track_offsets(ndum - 1, ndum)[2]
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(ndum, ndum)
        _, vm_prev, _, _ = self.get_track_offsets(ndum, ndum - 1)
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        hconn = hcon_idx_list[0]

        # connect all dummies to ground
        self.add_via_on_grid(hm_layer, hm_off + hconn, vm_prev + vcon_idx_list[-4])

        vm_idx_list = [vm_off + vcon_idx_list[1], vm_off + vcon_idx_list[2],
                       vm_prev + vcon_idx_list[-3], vm_prev + vcon_idx_list[-2]]
        xm_idx_list = [xm_prev + xm_bot_idx + idx for idx in range(nx - num_xm_sup, nx)]
        xm_idx_list.append(xm_off + xm_bot_idx)
        for vm_idx in vm_idx_list:
            # connect supply to vm layer
            self.add_via_on_grid(hm_layer, hm_off + hconn, vm_idx)
            # connect supply to xm layer
            for xm_idx in xm_idx_list:
                self.add_via_on_grid(vm_layer, vm_idx, xm_idx)

    def _connect_dummy(self, row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list,
                       vcon_idx_list):
        hm_off, vm_off, _, _ = self.get_track_offsets(row_idx, col_idx)
        hm_layer = self.bot_layer_id
        self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + hcon_idx_list[1], vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + hcon_idx_list[1], vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[-4])
        if conn_tb:
            self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + vcon_idx_list[1])
            self.add_via_on_grid(hm_layer, hm_off + bp_idx, vm_off + vcon_idx_list[1])

    def _connect_lr(self, row_idx, col_idx, nx, ndum, tp_idx, bp_idx, hcon_idx_list,
                    vcon_idx_list, xm_bot_idx):
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(row_idx, col_idx)
        _, vm_next, _, _ = self.get_track_offsets(row_idx, col_idx + 1)
        hm_layer = self.bot_layer_id
        col_real = col_idx - ndum
        row_real = row_idx - ndum
        if col_real % 2 == 0:
            port = bp_idx
            conn = hcon_idx_list[1]
        else:
            port = tp_idx
            conn = hcon_idx_list[0]
        self.add_via_on_grid(hm_layer, hm_off + port, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_off + vcon_idx_list[-4])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_off + vcon_idx_list[-1])
        self.add_via_on_grid(hm_layer, hm_off + conn, vm_next + vcon_idx_list[3])
        self.add_via_on_grid(hm_layer, hm_off + port, vm_next + vcon_idx_list[3])

        # connect to output port
        vm_layer = hm_layer + 1
        if row_real % 2 == 0:
            xm_idx = xm_bot_idx + col_real + 1
        else:
            xm_idx = xm_bot_idx + (nx - 1 - col_real)
        self.add_via_on_grid(vm_layer, vm_off + vcon_idx_list[-1], xm_off + xm_idx)

    def _connect_tb(self, row_idx, col_idx, ndum, tp_idx, hcon_idx_list,
                    vcon_idx_list, xm_bot_idx, mode=0):
        # mode = 0 is normal connection, mode = 1 is vdd connection, mode = -1 is vss connection
        hm_off, vm_off, _, _ = self.get_track_offsets(row_idx, col_idx)
        hm_next, _, xm_next, _ = self.get_track_offsets(row_idx + 1, col_idx)
        hm_layer = self.bot_layer_id
        if col_idx == ndum:
            conn1 = vcon_idx_list[1]
            tap = vcon_idx_list[2]
            conn2 = vcon_idx_list[3]
        else:
            conn1 = vcon_idx_list[-2]
            tap = vcon_idx_list[-3]
            conn2 = vcon_idx_list[-4]
        if mode >= 0:
            self.add_via_on_grid(hm_layer, hm_off + tp_idx, vm_off + conn1)
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + conn1)
        if mode == 0:
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + tap)

            # connect to output port
            vm_layer = hm_layer + 1
            self.add_via_on_grid(vm_layer, vm_off + tap, xm_next + xm_bot_idx)
        if mode <= 0:
            self.add_via_on_grid(hm_layer, hm_next + hcon_idx_list[0], vm_off + conn2)
            self.add_via_on_grid(hm_layer, hm_next + tp_idx, vm_off + conn2)

    def _draw_metal_tracks(self, nx, ny, ndum, hcon_space):
        num_h_tracks, num_v_tracks, num_x_tracks = self.num_tracks[0:3]
        xm_bot_idx = (num_x_tracks - nx) / 2

        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        hm_dtr = hcon_space + 1
        if tp_idx + hm_dtr >= num_h_tracks or bp_idx - hm_dtr < 0:
            # use inner hm tracks instead.
            hm_dtr *= -1
        bcon_idx = bp_idx - hm_dtr
        tcon_idx = tp_idx + hm_dtr

        # get via extensions
        grid = self.grid
        hm_layer = self.bot_layer_id
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        hm_ext, vm_ext = grid.get_via_extensions(hm_layer, 1, 1, unit_mode=True)
        vmx_ext, _ = grid.get_via_extensions(vm_layer, 1, 1, unit_mode=True)

        vm_tidx = [-0.5, 0.5, 1.5, 2.5, num_v_tracks - 3.5, num_v_tracks - 2.5,
                   num_v_tracks - 1.5, num_v_tracks - 0.5]

        # get unit block size
        blk_w, blk_h = self.res_unit_size

        # find top X layer track index that can be connected to supply.
        hm_off, vm_off, xm_off, _ = self.get_track_offsets(0, 0)
        vm_y1 = max(grid.get_wire_bounds(hm_layer, hm_off + max(bp_idx, bcon_idx),
                                         unit_mode=True)[1] + vm_ext,
                    grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                         unit_mode=True)[1] + vmx_ext)
        xm_vdd_top_idx = grid.find_next_track(xm_layer, vm_y1 - vmx_ext, half_track=True,
                                              mode=-1, unit_mode=True)
        num_xm_sup = int(xm_vdd_top_idx - xm_bot_idx - xm_off + 1)

        # get lower/upper bounds of output ports.
        xm_lower = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[0], unit_mode=True)[0]
        vm_off = self.get_track_offsets(0, nx + 2 * ndum - 1)[1]
        xm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-1], unit_mode=True)[1]

        # expand range by +/- 1 to draw metal pattern on dummies too
        for row_idx in range(ny + 2 * ndum):
            for col_idx in range(nx + 2 * ndum):
                hm_off, vm_off, xm_off, _ = self.get_track_offsets(row_idx, col_idx)

                # extend port tracks on hm layer
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[1], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-2], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bp_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=tp_idx - bp_idx, unit_mode=True)

                # draw hm layer bridge
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[0], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[3], unit_mode=True)
                pitch = tcon_idx - bcon_idx
                self.add_wires(hm_layer, hm_off + bcon_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)
                hm_lower, _ = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-4], unit_mode=True)
                _, hm_upper = grid.get_wire_bounds(vm_layer, vm_off + vm_tidx[-1], unit_mode=True)
                self.add_wires(hm_layer, hm_off + bcon_idx, hm_lower - hm_ext, hm_upper + hm_ext,
                               num=2, pitch=pitch, unit_mode=True)

                # draw vm layer bridges
                vm_lower = min(grid.get_wire_bounds(hm_layer, hm_off + min(bp_idx, bcon_idx),
                                                    unit_mode=True)[0] - vm_ext,
                               grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                                    unit_mode=True)[0] - vmx_ext)
                vm_upper = max(grid.get_wire_bounds(hm_layer, hm_off + max(tp_idx, tcon_idx),
                                                    unit_mode=True)[1] + vm_ext,
                               grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx + nx - 1,
                                                    unit_mode=True)[1] + vmx_ext)
                self.add_wires(vm_layer, vm_off + vm_tidx[0], vm_lower, vm_upper,
                               num=2, pitch=3, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-4], vm_lower, vm_upper,
                               num=2, pitch=3, unit_mode=True)

                vm_y1 = max(grid.get_wire_bounds(hm_layer, hm_off + max(bp_idx, bcon_idx),
                                                 unit_mode=True)[1] + vm_ext,
                            grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx,
                                                 unit_mode=True)[1] + vmx_ext)
                vm_y2 = min(grid.get_wire_bounds(hm_layer, hm_off + min(tp_idx, tcon_idx),
                                                 unit_mode=True)[0] - vm_ext,
                            grid.get_wire_bounds(xm_layer, xm_off + xm_bot_idx + nx - 1,
                                                 unit_mode=True)[0] - vmx_ext)
                self.add_wires(vm_layer, vm_off + vm_tidx[1], vm_y2 - blk_h, vm_y1,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[1], vm_y2, vm_y1 + blk_h,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-3], vm_y2 - blk_h, vm_y1,
                               num=2, pitch=1, unit_mode=True)
                self.add_wires(vm_layer, vm_off + vm_tidx[-3], vm_y2, vm_y1 + blk_h,
                               num=2, pitch=1, unit_mode=True)

        # draw and export output ports
        for row in range(ny + 2 * ndum):
            tr_off = self.get_track_offsets(row, 0)[2]
            for tidx in range(nx):
                warr = self.add_wires(xm_layer, tr_off + xm_bot_idx + tidx, lower=xm_lower,
                                      upper=xm_upper,
                                      fill_type='VSS', unit_mode=True)
                if row < ndum or (row == ndum and tidx == 0):
                    net_name = 'VSS'
                elif row >= ny + ndum:
                    net_name = 'VDD'
                else:
                    net_name = 'out<%d>' % (tidx + (row - ndum) * nx)
                self.add_pin(net_name, warr, show=False)

        return [bcon_idx, tcon_idx], vm_tidx, xm_bot_idx, num_xm_sup


class ResLadder(TemplateBase):
    """An template for creating termination resistors.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            ndum=1,
            res_type='standard',
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
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            ndum='number of dummy resistors.',
            res_type='the resistor type.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']
        res_type = res_params['res_type']

        res_master = self.new_template(params=res_params, temp_cls=ResLadderCore)

        # draw contact and move array up
        top_layer, nx_arr, ny_arr = res_master.size
        w_pitch, h_pitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
        sub_params = dict(
            top_layer=top_layer,
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            well_width=res_master.get_well_width(),
            show_pins=False,
            is_passive=True,
            tot_width_parity=nx_arr % 2,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        _, nx_sub, ny_sub = sub_master.size

        nx_shift = (nx_arr - nx_sub) // 2

        xpitch, ypitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
        bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(nx_shift * xpitch, 0),
                                     unit_mode=True)
        res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_sub * h_pitch),
                                     unit_mode=True)
        top_yo = (ny_arr + 2 * ny_sub) * h_pitch
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(nx_shift * xpitch, top_yo),
                                     orient='MX', unit_mode=True)

        # connect implant layers of resistor array and substrate contact together
        for lay in self.grid.tech_info.get_implant_layers(sub_type, res_type=res_type):
            self.add_rect(lay, self.get_rect_bbox(lay))

        # recompute array_box/size
        self.size = top_layer, nx_arr, ny_arr + 2 * ny_sub
        self.array_box = bot_inst.array_box.merge(top_inst.array_box)
        self.add_cell_boundary(self.bound_box)

        # gather supply and re-export outputs
        sup_table = {'VDD': [], 'VSS': []}
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sup_table[port_name].extend(bot_inst.get_all_port_pins(port_name))
        sup_table[port_name].extend(top_inst.get_all_port_pins(port_name))

        for port_name in res_inst.port_names_iter():
            if port_name in sup_table:
                sup_table[port_name].extend(res_inst.get_all_port_pins(port_name))
            else:
                self.reexport(res_inst.get_port(port_name), show=False)

        vdd_warrs, vss_warrs = sup_table['VDD'], sup_table['VSS']
        sup_layer = vdd_warrs[0].layer_id + 1
        # get power fill width and spacing
        sup_width = 1
        sup_spacing = self.grid.get_num_space_tracks(sup_layer, sup_width)
        num_sup_tracks = res_master.num_tracks[-1]
        # make sure every resistor sees the same power fill
        if sup_width + sup_spacing > num_sup_tracks:
            raise ValueError('Cannot draw power fill with width = %d' % sup_width)
        while num_sup_tracks % (sup_width + sup_spacing) != 0:
            sup_spacing += 1

        vdd_warrs, vss_warrs = self.do_power_fill(sup_layer, vdd_warrs, vss_warrs,
                                                  sup_width=sup_width,
                                                  fill_margin=0.5, edge_margin=0.2,
                                                  sup_spacing=sup_spacing)
        self.add_pin('VDD', vdd_warrs, show=False)
        self.add_pin('VSS', vss_warrs, show=False)
