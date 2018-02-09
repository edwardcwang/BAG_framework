# -*- coding: utf-8 -*-

"""This module defines ResArrayBase, an abstract class that draws resistor arrays.

This module also define some simple subclasses of ResArrayBase.
"""

import abc
from typing import Dict, Set, Tuple, Union, Any, Optional
from itertools import chain

from bag.layout.routing import TrackID, WireArray, Port
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.core import TechInfo

from .base import ResTech, AnalogResCore, AnalogResBoundary
from ..analog_core.substrate import SubstrateContact


# noinspection PyAbstractClass
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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(ResArrayBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        tech_params = self.grid.tech_info.tech_params
        self._tech_cls = tech_params['layout']['res_tech_class']  # type: ResTech
        self._bot_port = None  # type: Port
        self._top_port = None  # type: Port
        self._core_offset = None  # type: Tuple[int, int]
        self._core_pitch = None  # type: Tuple[int, int]
        self._num_tracks = None  # type: Tuple[int, ...]
        self._num_corner_tracks = None  # type: Tuple[int, ...]
        self._w_tracks = None  # type: Tuple[int, ...]
        self._hm_layer = self._tech_cls.get_bot_layer()
        self._well_width = None

    @classmethod
    def get_port_layer_id(cls, tech_info):
        # type: (TechInfo) -> int
        return tech_info.tech_params['layout']['res_tech_class'].get_bot_layer()

    @property
    def num_tracks(self):
        # type: () -> Tuple[int, ...]
        """Returns the number of tracks per resistor block on each routing layer."""
        return self._num_tracks

    @property
    def bot_layer_id(self):
        # type: () -> int
        """Returns the bottom resistor routing layer ID."""
        return self._hm_layer

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
        res = self.grid.resolution
        dx = self._core_offset[0] + self._core_pitch[0] * col_idx
        dy = self._core_offset[1] + self._core_pitch[1] * row_idx
        loc = dx * res, dy * res
        bot_port = self._bot_port.transform(self.grid, loc=loc)
        top_port = self._top_port.transform(self.grid, loc=loc)
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
            pitch = self.grid.get_track_pitch(lay, unit_mode=True)
            offsets.append(dim / pitch)

        return tuple(offsets)

    def get_abs_track_index(self, layer_id, cell_idx, tr_idx):
        # type: (int, int, Union[float, int]) -> float
        """Compute absolute track index from relative track index.

        Parameters
        ----------
        layer_id : int
            the horizontal layer ID.
        cell_idx : int
            the row or column index.  0 is the bottom row/left-most column.
        tr_idx : Union[int, float]
            the track index within the given cell.

        Returns
        -------
        abs_idx : float
            the absolute track index in this template.
        """
        dim_idx = 1 if self.grid.get_direction(layer_id) == 'x' else 0
        delta = self._core_offset[dim_idx] + self._core_pitch[dim_idx] * cell_idx

        half_pitch = self.grid.get_track_pitch(layer_id, unit_mode=True) // 2
        htr = int(round(2 * tr_idx)) + delta // half_pitch
        if htr % 2 == 0:
            return htr // 2
        return htr / 2

    # noinspection PyUnusedLocal
    def draw_array(self,  # type: ResArrayBase
                   l,  # type: float
                   w,  # type: float
                   sub_type,  # type: str
                   threshold,  # type: str
                   nx=1,  # type: int
                   ny=1,  # type: int
                   min_tracks=None,  # type: Optional[Tuple[int, ...]]
                   res_type='reference',  # type: str
                   em_specs=None,  # type: Optional[Dict[str, Any]]
                   grid_type='standard',  # type: str
                   ext_dir='',  # type: str
                   max_blk_ext=100,  # type: int
                   options=None,  # type: Optional[Dict[str, Any]]
                   top_layer=None,  # type: Optional[int]
                   **kwargs
                   ):
        # type: (...) -> None
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
        min_tracks : Optional[Tuple[int, ...]]
            minimum number of tracks per layer in the resistor unit cell.
            If None, Defaults to all 1's.
        res_type : str
            the resistor type.
        em_specs : Optional[Dict[str, Any]]
            resistor EM specifications dictionary.
        grid_type : str
            the resistor private routing grid name.
        ext_dir : str
            resistor core extension direction.
        max_blk_ext : int
            maximum number of block pitches we can extend for primitives.
        options : Optional[Dict[str, Any]]
            custom options for resistor primitives.
        top_layer : Optional[int]
            The top metal layer this block will use.  Defaults to the layer above private routing grid.
            If the top metal layer is equal to the default layer, then this will be a primitive template;
            self.size will be None, and only one dimension is quantized.
            If the top metal layer is above the default layer, then this will be a standard template, and
            both width and height will be quantized according to the block size.
        **kwargs :
            optional arguments.
        """
        if em_specs is None:
            em_specs = {}
        # modify resistor layer routing grid.
        grid_layers = self.grid.tech_info.tech_params['layout']['analog_res'][grid_type]
        min_tracks_default = []
        for lay_id, tr_w, tr_sp, tr_dir in grid_layers:
            self.grid.add_new_layer(lay_id, tr_sp, tr_w, tr_dir, override=True)
            min_tracks_default.append(1)

        self.grid.update_block_pitch()

        if min_tracks is None:
            min_tracks = tuple(min_tracks_default)
        if top_layer is None:
            top_layer = self.grid.top_private_layer

        # find location of the lower-left resistor core
        res = self.grid.resolution
        lay_unit = self.grid.layout_unit
        l_unit = int(round(l / lay_unit / res))
        w_unit = int(round(w / lay_unit / res))
        res_info = self._tech_cls.get_res_info(self.grid, l_unit, w_unit, res_type, sub_type, threshold,
                                               min_tracks, em_specs, ext_dir, max_blk_ext=max_blk_ext,
                                               options=options)
        w_edge, h_edge = res_info['w_edge'], res_info['h_edge']
        w_core, h_core = res_info['w_core'], res_info['h_core']
        self._core_offset = w_edge, h_edge
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
                cur_loc = (w_edge + col * w_core, h_edge + row * h_core)
                self.add_instance(core_master, inst_name=cur_name, loc=cur_loc, unit_mode=True)
                if row == 0 and col == 0:
                    self._bot_port = core_master.get_port('bot')
                    self._top_port = core_master.get_port('top')

        # place boundaries
        # bottom-left corner
        inst_bl = self.add_instance(corner_master, inst_name='XBL')
        # bottom edge
        self.add_instance(tb_master, inst_name='XB', loc=(w_edge, 0), nx=nx, spx=w_core, unit_mode=True)
        # bottom-right corner
        loc = (2 * w_edge + nx * w_core, 0)
        self.add_instance(corner_master, inst_name='XBR', loc=loc, orient='MY', unit_mode=True)
        # left edge
        loc = (0, h_edge)
        well_xl = lr_master.get_well_left(unit_mode=True)
        self.add_instance(lr_master, inst_name='XL', loc=loc, ny=ny, spy=h_core, unit_mode=True)
        # right edge
        loc = (2 * w_edge + nx * w_core, h_edge)
        well_xr = loc[0] - well_xl
        self._well_width = well_xr - well_xl
        self.add_instance(lr_master, inst_name='XR', loc=loc, orient='MY', ny=ny, spy=h_core, unit_mode=True)
        # top-left corner
        loc = (0, 2 * h_edge + ny * h_core)
        self.add_instance(corner_master, inst_name='XTL', loc=loc, orient='MX', unit_mode=True)
        # top edge
        loc = (w_edge, 2 * h_edge + ny * h_core)
        self.add_instance(tb_master, inst_name='XT', loc=loc, orient='MX', nx=nx, spx=w_core, unit_mode=True)
        # top-right corner
        loc = (2 * w_edge + nx * w_core, 2 * h_edge + ny * h_core)
        inst_tr = self.add_instance(corner_master, inst_name='XTR', loc=loc, orient='R180', unit_mode=True)

        # set array box and size
        self.array_box = inst_bl.array_box.merge(inst_tr.array_box)
        bnd_box = inst_bl.bound_box.merge(inst_tr.bound_box)
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
        super(TerminationCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
            em_specs={},
            ext_dir='',
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
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.  Must be even.',
            ny='number of resistors in a column.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
            show_pins='True to show pins.',
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

        self.draw_array(em_specs=div_em_specs, grid_type='low_res', **self.params)

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
                        tidx = self.grid.coord_to_nearest_track(cur_lay, cur_warrs[0].middle, half_track=True)
                        tid = TrackID(cur_lay, tidx, width=cur_w)
                        port_wires[warrs_idx] = [self.connect_to_tracks(cur_warrs, tid)]
                else:
                    # draw one horizontal wire in middle of each row, then connect last vertical wire to it
                    # this way we distribute currents evenly.
                    cur_p = self.num_tracks[lay_idx]
                    # relative base index.  Round down if we have half-integer number of tracks
                    base_idx_rel = (int(round(cur_p * 2)) // 2 - 1) / 2
                    base_idx = self.get_abs_track_index(cur_lay, 0, base_idx_rel)
                    tid = TrackID(cur_lay, base_idx, width=cur_w, num=ny, pitch=cur_p)
                    for warrs_idx in range(3):
                        port_wires[warrs_idx] = self.connect_to_tracks(port_wires[warrs_idx], tid, min_len_mode=0)
            else:
                # layer direction is the same.  Strap wires to current layer.
                for warrs_idx in range(3):
                    cur_warrs = port_wires[warrs_idx]
                    new_warrs = [self.strap_wires(warr, cur_lay, tr_w_list=[cur_w], min_len_mode_list=[0])
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
        super(Termination, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
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
            em_specs='EM specifications for the termination network.',
            ext_dir='resistor core extension direction.',
            show_pins='True to show pins.',
        )

    def draw_layout(self):
        # type: () -> None

        res_params = self.params.copy()
        res_type = res_params['res_type']
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        sub_type = self.params['sub_type']
        show_pins = self.params['show_pins']

        res_master = self.new_template(params=res_params, temp_cls=TerminationCore)
        if sub_w == 0:
            # do not draw substrate contact.
            inst = self.add_instance(res_master, inst_name='XRES', loc=(0, 0), unit_mode=True)
            for port_name in inst.port_names_iter():
                self.reexport(inst.get_port(port_name), show=show_pins)
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

            bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(sub_x, 0), unit_mode=True)
            res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_shift * h_pitch), unit_mode=True)
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
        super(ResLadderCore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
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
        hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup = self._draw_metal_tracks(nx, ny, ndum, hcon_space)

        self._connect_ladder(nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup)

    def _connect_ladder(self, nx, ny, ndum, hcon_idx_list, vcon_idx_list, xm_bot_idx, num_xm_sup):
        tp_idx = self.top_port_idx
        bp_idx = self.bot_port_idx
        # connect main ladder
        for row_idx in range(ndum, ny + ndum):
            rmod = row_idx - ndum
            for col_idx in range(ndum, nx + ndum):
                if (col_idx == ndum and rmod % 2 == 1) or (col_idx == nx - 1 + ndum and rmod % 2 == 0):
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
                self._connect_dummy(row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list, vcon_idx_list)

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

    def _connect_dummy(self, row_idx, col_idx, conn_tb, tp_idx, bp_idx, hcon_idx_list, vcon_idx_list):
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
                warr = self.add_wires(xm_layer, tr_off + xm_bot_idx + tidx, lower=xm_lower, upper=xm_upper,
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
        super(ResLadder, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
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
        bot_inst = self.add_instance(sub_master, inst_name='XBSUB', loc=(nx_shift * xpitch, 0), unit_mode=True)
        res_inst = self.add_instance(res_master, inst_name='XRES', loc=(0, ny_sub * h_pitch), unit_mode=True)
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

        vdd_warrs, vss_warrs = self.do_power_fill(sup_layer, vdd_warrs, vss_warrs, sup_width=sup_width,
                                                  fill_margin=0.5, edge_margin=0.2, sup_spacing=sup_spacing)
        self.add_pin('VDD', vdd_warrs, show=False)
        self.add_pin('VSS', vss_warrs, show=False)
