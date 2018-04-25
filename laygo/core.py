# -*- coding: utf-8 -*-

"""This module defines LaygoBase, a base template class for generic digital layout topologies."""

import abc
from typing import TYPE_CHECKING, Dict, Any, Set, Tuple, List, Optional, Iterable

import bisect

from bag.math import lcm
from bag.util.interval import IntervalSet
from bag.util.cache import DesignMaster

from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing import TrackID, WireArray

from .tech import LaygoTech
from .base import LaygoPrimitive, LaygoSubstrate, LaygoEndRow, LaygoSpace
from ..analog_core.placement import WireGroup, WireTree

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB
    from bag.layout.objects import Instance
    from bag.layout.routing import RoutingGrid


class DigitalExtInfo(object):
    """The extension information object for DigitalBase."""

    def __init__(self, ext_list):
        self._ext_list = ext_list

    def reverse(self):
        return DigitalExtInfo([ext.reverse() for ext in reversed(self._ext_list)])

    def ext_iter(self):
        return self._ext_list


class LaygoEdgeInfo(object):
    """The edge information object for LaygoBase."""

    def __init__(self, row_end_list, ext_end_list):
        self._row_end_list = row_end_list
        self._ext_end_list = ext_end_list

    def get_immutable_key(self):
        return DesignMaster.to_immutable_id((self._row_end_list, self._ext_end_list))

    def master_infos_iter(self, row_edge_infos, y0=0, flip=False):
        for y, edge_params in self._ext_end_list:
            yield (y0 - y if flip else y0 + y, flip, edge_params)

        for (end, lay_info), (y, flip_ud, re_params) in zip(self._row_end_list, row_edge_infos):
            edge_params = re_params.copy()
            edge_params['layout_info'] = lay_info
            edge_params['adj_blk_info'] = end
            yield (y0 - y if flip else y0 + y, flip != flip_ud, edge_params)

    def row_end_iter(self):
        for val in self._row_end_list:
            yield val


class DigitalEdgeInfo(object):
    """The edge information object for DigitalBase."""

    def __init__(self, row_y_list, lay_edge_list, ext_end_list):
        self._row_y_list = row_y_list
        self._lay_edge_list = lay_edge_list
        self._ext_end_list = ext_end_list

    def master_infos_iter(self, row_edge_infos):
        for val in self._ext_end_list:
            if val is not None:
                yield (val[0], False, val[1])

        for idx, (row_y, lay_edge_info) in enumerate(zip(self._row_y_list, self._lay_edge_list)):
            yield from lay_edge_info.master_infos_iter(row_edge_infos, y0=row_y, flip=idx % 2 == 1)

    def get_laygo_edge(self, idx):
        return self._lay_edge_list[idx]

    def get_ext_end(self, bot_idx):
        return self._ext_end_list[bot_idx + 1]


class LaygoIntvSet(object):
    """A data structure that keeps track of used laygo columns in a laygo row.

    This class is used to automatically fill empty spaces, and also get
    left/right/top/bottom layout information needed to create space blocks
    and extension rows.

    Note: We intentionally did not keep track of total number of columns in
    thie object.  This makes it possible to dynamically size a laygo row.

    Parameters
    ----------
    default_end_info : Any
        the default left/right edge layout information object to use.
    """

    def __init__(self, default_end_info):
        # type: (Any) -> None
        self._intv = IntervalSet()
        self._end_flags = {}
        self._default_end_info = default_end_info

    def add(self, intv, ext_info, endl, endr):
        # type: (Tuple[int, int], Any, Any, Any) -> bool
        """Add a new interval to this data structure.

        Parameters
        ----------
        intv : Tuple[int, int]
            the laygo interval as (start_column, stop_column) tuple.
        ext_info : Any
            the top/bottom extension information object of this interval.
        endl : Any
            the left edge layout information object.
        endr : Any
            the right edge layout information object.

        Returns
        -------
        success : bool
            True if the given interval is successfully added.  False if it
            overlaps with existing blocks.
        """
        ans = self._intv.add(intv, val=ext_info)
        if ans:
            start, stop = intv
            if start in self._end_flags:
                del self._end_flags[start]
            else:
                self._end_flags[start] = endl
            if stop in self._end_flags:
                del self._end_flags[stop]
            else:
                self._end_flags[stop] = endr
            return True
        else:
            return False

    def values(self):
        # type: () -> Iterable[Any]
        """Returns an iterator over extension information objects stored in this row."""
        return self._intv.values()

    def get_complement(self, total_intv, endl_info, endr_info):
        # type: (Tuple[int, int], Any, Any) -> Tuple[List[Tuple[int, int]], List[Tuple[Any, Any]]]
        """Returns a list of unused column intervals.

        Parameters
        ----------
        total_intv : Tuple[int, int]
            A (start, stop) tuple that indicates how many columns are in this row.
        endl_info : Any
            the left-most edge layout information object of this row.
        endr_info : Any
            the right-most edge layout information object of this row.

        Returns
        -------
        intv_list : List[Tuple[int, int]]
            a list of unused column intervals.
        end_list : List[Tuple[Any, Any]]
            a list of left/right edge layout information object corresponding to each
            unused interval.
        """
        compl_intv = self._intv.get_complement(total_intv)
        intv_list = []
        end_list = []
        for intv in compl_intv:
            intv_list.append(intv)
            end_list.append((self._end_flags.get(intv[0], endl_info),
                             self._end_flags.get(intv[1], endr_info)))
        return intv_list, end_list

    def get_end_info(self, num_col):
        # type: (int) -> Tuple[Any, Any]
        """Returns the left-most and right-most edge layout information object of this row.

        Parameters
        ----------
        num_col : int
            number of columns in this row.

        Returns
        -------
        endl_info : Any
            the left-most edge layout information object of this row.
        endr_info : Any
            the right-most edge layout information object of this row.
        """
        if 0 not in self._end_flags:
            endl_info = self._default_end_info
        else:
            endl_info = self._end_flags[0]

        if num_col not in self._end_flags:
            endr_info = self._default_end_info
        else:
            endr_info = self._end_flags[num_col]

        return endl_info, endr_info

    def get_end(self):
        # type: () -> int
        """Returns the end column index of the last used interval."""
        if not self._intv:
            return 0
        return self._intv.get_end()


class LaygoBaseInfo(object):
    """A class that provides information to assist in LaygoBase layout calculations.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    config : Dict[str, Any]
        the LaygoBase configuration dictionary.
    top_layer : Optional[int]
        the LaygoBase top layer ID.
    guard_ring_nf : int
        guard ring width in number of fingers.  0 to disable.
    draw_boundaries : bool
        True if boundary cells should be drawn around this LaygoBase.
    end_mode : int
        right/left/top/bottom end mode flag.  This is a 4-bit integer.  If bit 0 (LSB) is 1, then
        we assume there are no blocks abutting the bottom.  If bit 1 is 1, we assume there are no
        blocks abutting the top.  bit 2 and bit 3 (MSB) corresponds to left and right, respectively.
        The default value is 15, which means we assume this AnalogBase is surrounded by empty
        spaces.
    num_col : Optional[int]
        number of columns in this LaygoBase.  This must be specified if draw_boundaries is True.
    """

    def __init__(self, grid, config, top_layer=None, guard_ring_nf=0, draw_boundaries=False,
                 end_mode=0, num_col=None):
        # type: (RoutingGrid, Dict[str, Any], Optional[int], int, bool, int, Optional[int]) -> None
        self._tech_cls = grid.tech_info.tech_params['layout']['laygo_tech_class']  # type: LaygoTech

        # error checking
        dig_top_layer = config['tr_layers'][-1]
        if dig_top_layer != self._tech_cls.get_dig_top_layer():
            raise ValueError('Top tr_layers must be layer %d' % self._tech_cls.get_dig_top_layer())

        # update routing grid
        lch_unit = int(round(config['lch'] / grid.layout_unit / grid.resolution))
        self.grid = grid.copy()
        self._lch_unit = lch_unit
        self._config = config

        sd_pitch = self._tech_cls.get_sd_pitch(lch_unit)
        vm_layer = self._tech_cls.get_dig_conn_layer()
        vm_space, vm_width = self._tech_cls.get_laygo_conn_track_info(self._lch_unit)
        self.grid.add_new_layer(vm_layer, vm_space, vm_width, 'y', override=True, unit_mode=True)
        tdir = 'x'
        for lay, w, sp in zip(self._config['tr_layers'], self._config['tr_widths'],
                              self._config['tr_spaces']):
            self.grid.add_new_layer(lay, sp, w, tdir, override=True, unit_mode=True)
            if tdir == 'y':
                pitch = w + sp
                if pitch % sd_pitch != 0:
                    raise ValueError('laygo vertical routing pitch must '
                                     'be multiples of %d' % sd_pitch)
                tdir = 'x'
            else:
                tdir = 'y'
        self.grid.update_block_pitch()

        # update routing grid width overrides
        w_override = self._config.get('w_override', None)
        if w_override:
            for layer_id, w_lookup in w_override.items():
                for width_ntr, w_unit in w_lookup.items():
                    self.grid.add_width_override(layer_id, width_ntr, w_unit, unit_mode=True)

        # initialize parameters
        self.guard_ring_nf = guard_ring_nf
        self.top_layer = dig_top_layer + 1 if top_layer is None else top_layer
        self.end_mode = end_mode
        self._col_width = self._tech_cls.get_sd_pitch(self._lch_unit)
        self.draw_boundaries = draw_boundaries

        # set number of columns
        self._num_col = None
        self._core_col = None
        self._edge_margins = None
        self._edge_widths = None
        self.set_num_col(num_col)

    @property
    def tech_cls(self):
        return self._tech_cls

    @property
    def conn_layer(self):
        return self._tech_cls.get_dig_conn_layer()

    @property
    def fg2d_s_short(self):
        return self._tech_cls.get_laygo_fg2d_s_short()

    @property
    def sub_columns(self):
        return self._tech_cls.get_sub_columns(self._lch_unit)

    @property
    def sub_port_columns(self):
        return self._tech_cls.get_sub_port_columns(self._lch_unit)

    @property
    def min_sub_space(self):
        return self._tech_cls.get_min_sub_space_columns(self._lch_unit)

    @property
    def mos_pitch(self):
        return self._tech_cls.get_mos_pitch(unit_mode=True)

    @property
    def lch_unit(self):
        return self._lch_unit

    @property
    def lch(self):
        return self._lch_unit * self.grid.layout_unit * self.grid.resolution

    @property
    def col_width(self):
        return self._col_width

    @property
    def unit_num_col(self):
        blk_w = self.grid.get_block_size(self._tech_cls.get_dig_top_layer(), unit_mode=True)[0]
        col_width = self.col_width
        return lcm([blk_w, col_width]) // col_width

    @property
    def tot_height_pitch(self):
        return lcm([self.grid.get_block_size(self.top_layer, unit_mode=True)[1], self.mos_pitch])

    @property
    def num_col(self):
        return self._num_col

    @property
    def edge_margins(self):
        return self._edge_margins

    @property
    def edge_widths(self):
        return self._edge_widths

    @property
    def tot_width(self):
        if self._edge_margins is None:
            raise ValueError('number of columns not set; cannot compute total width.')
        return (self._edge_margins[0] + self._edge_margins[1] + self._edge_widths[0] +
                self._edge_widths[1] + self._num_col * self._col_width)

    @property
    def core_width(self):
        if self._edge_widths is None:
            raise ValueError('number of columns not set; cannot compute core width.')
        return self._edge_widths[0] + self._edge_widths[1] + self._num_col * self._col_width

    @property
    def core_col(self):
        return self._core_col

    def get_placement_info(self, num_col):
        left_end = (self.end_mode & 4) != 0
        right_end = (self.end_mode & 8) != 0
        return self._tech_cls.get_placement_info(self.grid, self.top_layer, num_col, self._lch_unit,
                                                 self.guard_ring_nf, left_end, right_end, True)

    def set_num_col(self, new_num_col):
        if new_num_col is not None:
            if new_num_col % self.unit_num_col != 0:
                raise ValueError('num_col = %d must be '
                                 'multiple of %d' % (new_num_col, self.unit_num_col))
        self._num_col = new_num_col

        if self.draw_boundaries:
            if new_num_col is None:
                self._edge_margins = None
                self._edge_widths = None
                self._core_col = None
            else:
                place_info = self.get_placement_info(new_num_col)
                self._edge_margins = place_info.edge_margins
                self._edge_widths = place_info.edge_widths
                self._core_col = place_info.core_fg
        else:
            self._core_col = new_num_col
            self._edge_margins = (0, 0)
            self._edge_widths = (0, 0)

    def __getitem__(self, item):
        return self._config[item]

    def col_to_coord(self, col_idx, unit_mode=False):
        if self._edge_margins is None:
            raise ValueError('Edge margins is not defined.  Did you set number of columns?')

        ans = self._edge_margins[0] + self._edge_widths[0] + col_idx * self._col_width

        if unit_mode:
            return ans
        return ans * self.grid.resolution

    def col_to_track(self, layer_id, col_idx):
        # error checking
        if self.grid.get_direction(layer_id) == 'x':
            raise ValueError('col_to_track() only works on vertical routing layers.')

        coord = self.col_to_coord(col_idx, unit_mode=True)
        return self.grid.coord_to_track(layer_id, coord, unit_mode=True)

    def col_to_nearest_rel_track(self, layer_id, col_idx, half_track=False, mode=0):
        # error checking
        if self.grid.get_direction(layer_id) == 'x':
            raise ValueError('col_to_nearest_rel_track() only works on vertical routing layers.')

        x_rel = col_idx * self._col_width

        pitch = self.grid.get_track_pitch(layer_id, unit_mode=True)
        offset = pitch // 2
        if half_track:
            pitch //= 2

        q, r = divmod(x_rel - offset, pitch)
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
            return q
        elif q % 2 == 0:
            return q // 2
        else:
            return q / 2

    def coord_to_col(self, coord, unit_mode=False):
        if self._edge_margins is None:
            raise ValueError('Edge margins is not defined.  Did you set number of columns?')

        if not unit_mode:
            coord = int(round(coord / self.grid.resolution))

        col_width = self._col_width
        offset = self._edge_margins[0] + self._edge_widths[0]
        if (coord - offset) % col_width != 0:
            raise ValueError('Coordinate %d is not on pitch.' % coord)
        return (coord - offset) // col_width

    def coord_to_nearest_col(self, coord, mode=0, unit_mode=False):
        if self._edge_margins is None:
            raise ValueError('Edge margins is not defined.  Did you set number of columns?')

        if not unit_mode:
            coord = int(round(coord / self.grid.resolution))

        col_width = self._col_width
        offset = self._edge_margins[0] + self._edge_widths[0]

        coord -= offset
        if mode == 0:
            n = int(round(coord / col_width))
        elif mode > 0:
            if coord % col_width == 0 and mode == 2:
                coord += 1
            n = -(-coord // col_width)
        else:
            if coord % col_width == 0 and mode == -2:
                coord -= 1
            n = coord // col_width

        return n

    def rel_track_to_nearest_col(self, layer_id, rel_tid, mode=0):
        # error checking
        if self.grid.get_direction(layer_id) == 'x':
            raise ValueError('rel_track_to_nearest_col() only works on vertical routing layers.')

        pitch = self.grid.get_track_pitch(layer_id, unit_mode=True)
        x_rel = pitch // 2 + int(round(rel_tid * pitch))

        col_width = self.col_width

        if mode == 0:
            n = int(round(x_rel / col_width))
        elif mode > 0:
            if x_rel % col_width == 0 and mode == 2:
                x_rel += 1
            n = -(-x_rel // col_width)
        else:
            if x_rel % col_width == 0 and mode == -2:
                x_rel -= 1
            n = x_rel // col_width

        return n


class LaygoBase(TemplateBase, metaclass=abc.ABCMeta):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None

        hidden_params = kwargs.pop('hidden_params', {}).copy()
        hidden_params['laygo_edgel'] = None
        hidden_params['laygo_edger'] = None

        TemplateBase.__init__(self, temp_db, lib_name, params, used_names,
                              hidden_params=hidden_params, **kwargs)

        self._laygo_info = LaygoBaseInfo(self.grid, self.params['config'])
        self.grid = self._laygo_info.grid
        self._tech_cls = self._laygo_info.tech_cls

        # initialize attributes
        self._tr_manager = None
        self._row_layout_info = None
        self._row_prop_list = None
        self._row_info_list = None
        self._laygo_size = None
        self._ext_params = None
        self._used_list = None  # type: List[LaygoIntvSet]
        self._bot_end_master = None
        self._top_end_master = None
        self._lr_edge_info = None
        self._tb_ext_info = None
        self._bot_sub_extw = 0
        self._top_sub_extw = 0
        self._laygo_edgel = None
        self._laygo_edger = None

    @property
    def num_rows(self):
        # type: () -> int
        return len(self._row_prop_list)

    @property
    def num_cols(self):
        # type: () -> int
        return self._laygo_size[0]

    @property
    def laygo_info(self):
        # type: () -> LaygoBaseInfo
        return self._laygo_info

    @property
    def laygo_size(self):
        return self._laygo_size

    @property
    def digital_size(self):
        return self._laygo_size[0], 1

    @property
    def conn_layer(self):
        return self._tech_cls.get_dig_conn_layer()

    @property
    def fg2d_s_short(self):
        return self._laygo_info.fg2d_s_short

    @property
    def sub_columns(self):
        return self._laygo_info.sub_columns

    @property
    def min_sub_space(self):
        return self._laygo_info.min_sub_space

    @property
    def tot_height(self):
        return self._row_prop_list[-1]['row_y'][3]

    @property
    def row_layout_info(self):
        return self._row_layout_info

    @property
    def tb_ext_info(self):
        return self._tb_ext_info

    @property
    def lr_edge_info(self):
        return self._lr_edge_info

    def _set_row_layout_info(self):
        top_layer = self._laygo_info.top_layer
        guard_ring_nf = self._laygo_info.guard_ring_nf

        row_edge_infos = []
        for ridx, (rinfo, rprop) in enumerate(zip(self._row_info_list, self._row_prop_list)):
            flip_ud = (rprop['orient'] == 'MX')
            row_y = rprop['row_y']
            y = row_y[2] if flip_ud else row_y[1]

            row_edge_params = dict(
                name_id=rinfo['row_name_id'],
                top_layer=top_layer,
                guard_ring_nf=guard_ring_nf,
                row_info=rinfo,
                is_laygo=True,
            )
            row_edge_infos.append((y, flip_ud, row_edge_params))

        self._row_layout_info = dict(
            config=self.params['config'],
            top_layer=self._laygo_info.top_layer,
            guard_ring_nf=self._laygo_info.guard_ring_nf,
            draw_boundaries=self._laygo_info.draw_boundaries,
            end_mode=self._laygo_info.end_mode,
            row_prop_list=self._row_prop_list,
            row_info_list=self._row_info_list,
            ext_params=self._ext_params,
            bot_sub_extw=self._bot_sub_extw,
            top_sub_extw=self._top_sub_extw,
            row_edge_infos=row_edge_infos,
        )

    def _set_endlr_infos(self, num_rows):
        default_end_info = (self._tech_cls.get_default_end_info(), None)
        def_edge_info = LaygoEdgeInfo([(default_end_info, None)] * num_rows, [])
        self._laygo_edgel = self.params['laygo_edgel']
        if self._laygo_edgel is None:
            self._laygo_edgel = def_edge_info
        self._laygo_edger = self.params['laygo_edger']
        if self._laygo_edger is None:
            self._laygo_edger = def_edge_info

    def set_rows_direct(self, layout_info, num_col=None, end_mode=None):
        top_layer = layout_info['top_layer']
        guard_ring_nf = layout_info['guard_ring_nf']
        draw_boundaries = layout_info['draw_boundaries']
        row_prop_list = layout_info['row_prop_list']
        if end_mode is None:
            end_mode = layout_info['end_mode']

        num_rows = len(row_prop_list)

        # set LaygoInfo
        self._laygo_info.top_layer = top_layer
        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode
        self._laygo_info.set_num_col(num_col)

        # set row information
        default_end_info = (self._tech_cls.get_default_end_info(), None)
        self._used_list = [LaygoIntvSet(default_end_info) for _ in range(num_rows)]

        # set left and right end information list
        self._set_endlr_infos(num_rows)

        # make end masters
        if draw_boundaries:
            bot_type = row_prop_list[0]['mos_type']
            top_type = row_prop_list[-1]['mos_type']
            bot_thres = row_prop_list[0]['threshold']
            top_thres = row_prop_list[-1]['threshold']
            self._create_end_masters(end_mode, bot_type, top_type, bot_thres, top_thres,
                                     top_layer, 0)

        self._row_prop_list = row_prop_list
        if 'row_info_list' in layout_info:
            self._row_info_list = layout_info['row_info_list']
            self._ext_params = layout_info['ext_params']
        else:
            self._ext_params, self._row_info_list = self.compute_row_info(self._laygo_info,
                                                                          row_prop_list)

        # compute laygo size if we know the number of columns
        if num_col is not None:
            self.set_laygo_size(num_col)
        self._set_row_layout_info()

    def set_row_types(self, row_types, row_widths, row_orientations, row_thresholds,
                      draw_boundaries, end_mode, num_g_tracks=None, num_gb_tracks=None,
                      num_ds_tracks=None, row_min_tracks=None, top_layer=None,
                      guard_ring_nf=0, row_kwargs=None, num_col=None, row_sub_widths=None,
                      **kwargs):

        self._tr_manager = kwargs.get('tr_manager', None)
        wire_names = kwargs.get('wire_names', None)
        min_height = kwargs.get('min_height', 0)

        # error checking
        if (row_types[0] == 'ptap' or row_types[0] == 'ntap') and row_orientations[0] != 'R0':
            raise ValueError('bottom substrate orientation must be R0')
        if (row_types[-1] == 'ptap' or row_types[-1] == 'ntap') and row_orientations[-1] != 'MX':
            raise ValueError('top substrate orientation must be MX')
        if len(row_types) < 2:
            raise ValueError('Must draw at least 2 rows.')
        if len(row_types) != len(row_widths) or len(row_types) != len(row_orientations):
            raise ValueError('row_types/row_widths/row_orientations length mismatch.')
        if draw_boundaries and num_col is None:
            raise ValueError('Must specify total number of columns if drawing boundary.')
        if not row_sub_widths:
            row_sub_widths = row_widths
        elif len(row_sub_widths) != len(row_widths):
            raise ValueError('row_widths and row_sub_widths must have same length.')

        # set default values
        num_rows = len(row_types)
        if not draw_boundaries:
            end_mode = 0
        if row_kwargs is None:
            row_kwargs = [{}] * num_rows
        if row_min_tracks is None:
            row_min_tracks = [{}] * num_rows
        if top_layer is None:
            top_layer = self._laygo_info.top_layer

        # set LaygoInfo
        self._laygo_info.top_layer = top_layer
        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode
        self._laygo_info.set_num_col(num_col)

        # set row information
        default_end_info = (self._tech_cls.get_default_end_info(), None)
        self._used_list = [LaygoIntvSet(default_end_info) for _ in range(num_rows)]

        # set left and right end information list
        self._set_endlr_infos(num_rows)

        # compute remaining row information
        tot_height_pitch = self._laygo_info.tot_height_pitch
        if draw_boundaries:
            ybot, min_height = self._create_end_masters(end_mode, row_types[0], row_types[-1],
                                                        row_thresholds[0], row_thresholds[-1],
                                                        top_layer, min_height)
        else:
            ybot = 0

        tmp = self._get_place_info(row_types, row_widths, row_sub_widths, row_orientations,
                                   row_thresholds, row_min_tracks, row_kwargs, num_g_tracks,
                                   num_gb_tracks, num_ds_tracks, self._tr_manager, wire_names)
        self._row_prop_list, self._row_info_list, pinfo_list, wire_tree = tmp

        self._ext_params = self._place_rows(ybot, tot_height_pitch, self._row_prop_list,
                                            self._row_info_list, pinfo_list, wire_tree, min_height)

        # compute laygo size if we know the number of columns
        if num_col is not None:
            self.set_laygo_size(num_col)
        self._set_row_layout_info()

    def _create_end_masters(self, end_mode, bot_row_type, top_row_type, bot_row_thres,
                            top_row_thres, top_layer, min_height):
        bot_end = (end_mode & 1) != 0
        top_end = (end_mode & 2) != 0

        if bot_row_type != 'ntap' and bot_row_type != 'ptap':
            raise ValueError('Bottom row must be substrate.')
        if top_row_type != 'ntap' and top_row_type != 'ptap':
            raise ValueError('Top row must be substrate.')

        # create boundary masters
        params = dict(
            lch=self._laygo_info.lch,
            mos_type=bot_row_type,
            threshold=bot_row_thres,
            is_end=bot_end,
            top_layer=top_layer,
        )
        self._bot_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
        params = dict(
            lch=self._laygo_info.lch,
            mos_type=top_row_type,
            threshold=top_row_thres,
            is_end=top_end,
            top_layer=top_layer,
        )
        self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
        ybot = self._bot_end_master.bound_box.height_unit

        return ybot, min_height - self._top_end_master.bound_box.height_unit

    def _get_place_info(self, row_types, row_widths, row_sub_widths, row_orientations,
                        row_thresholds, row_min_tracks, row_kwargs, num_g_tracks,
                        num_gb_tracks, num_ds_tracks, tr_manager, wire_names):
        lch = self._laygo_info.lch
        lch_unit = int(round(lch / self.grid.layout_unit / self.grid.resolution))
        mos_pitch = self._laygo_info.mos_pitch
        tcls = self._tech_cls
        conn_layer = tcls.get_dig_conn_layer()
        hm_layer = conn_layer + 1
        le_sp_tr = self.grid.get_line_end_space_tracks(conn_layer, hm_layer, 1, half_space=True)

        rprop_list, rinfo_list, pinfo_list = [], [], []
        wire_tree = WireTree(mirror=(row_types[0] == 'nch' or row_types[0] == 'pch'))
        for row_idx, (row_type, row_w, row_wsub, row_orient,
                      row_thres, min_tracks, kwargs) in \
                enumerate(zip(row_types, row_widths, row_sub_widths, row_orientations,
                              row_thresholds, row_min_tracks, row_kwargs)):
            if row_idx == 0:
                bot_row_type = row_type
            else:
                bot_row_type = row_types[row_idx - 1]
            if row_idx == len(row_types) - 1:
                top_row_type = row_type
            else:
                top_row_type = row_types[row_idx + 1]

            if row_orient != 'R0':
                bot_row_type, top_row_type = top_row_type, bot_row_type

            # get information dictionary
            if row_type == 'nch' or row_type == 'pch':
                row_info = tcls.get_laygo_mos_row_info(lch_unit, row_w, row_wsub, row_type,
                                                       row_thres, bot_row_type, top_row_type,
                                                       **kwargs)
            elif row_type == 'ptap' or row_type == 'ntap':
                row_info = tcls.get_laygo_sub_row_info(lch_unit, row_w, row_type,
                                                       row_thres, **kwargs)
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            # get row pitch
            row_pitch = min_row_height = mos_pitch
            for layer, num_tr in min_tracks.items():
                tr_pitch = self.grid.get_track_pitch(layer, unit_mode=True)
                min_row_height = max(min_row_height, num_tr * tr_pitch)
                row_pitch = lcm([row_pitch, tr_pitch // 2])

            # get bottom/top Y coordinates
            blk_h = row_info['arr_y'][1]
            g_conn_y = row_info.get('g_conn_y', (0, 0))
            gb_conn_y = row_info['gb_conn_y']
            ds_conn_y = row_info['ds_conn_y']
            bot_conn_y, top_conn_y, bot_wires, top_wires = [], [], [], []
            if row_orient == 'R0':
                bext_info = row_info['ext_bot_info']
                text_info = row_info['ext_top_info']
                if wire_names is None:
                    ng = num_g_tracks[row_idx]
                    ngb = num_gb_tracks[row_idx]
                    nds = num_ds_tracks[row_idx]
                    if ng >= 1:
                        bot_conn_y.append(g_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'g', ng, space=le_sp_tr))
                    if ngb >= 1:
                        top_conn_y.append(gb_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'gb', ngb, space=le_sp_tr))
                    if nds >= 1:
                        top_conn_y.append(ds_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'ds', nds, space=le_sp_tr))
                else:
                    wnames = wire_names[row_idx]
                    if wnames.get('g', False):
                        bot_conn_y.append(g_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'g', tr_manager=tr_manager,
                                                   name_list=wnames['g']))
                    if wnames.get('gb', False):
                        top_conn_y.append(gb_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'gb', tr_manager=tr_manager,
                                                   name_list=wnames['gb']))
                    if wnames.get('ds', False):
                        top_conn_y.append(ds_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'ds', tr_manager=tr_manager,
                                                   name_list=wnames['ds']))
            else:
                bext_info = row_info['ext_top_info']
                text_info = row_info['ext_bot_info']
                g_conn_y = blk_h - g_conn_y[1], blk_h - g_conn_y[0]
                gb_conn_y = blk_h - gb_conn_y[1], blk_h - gb_conn_y[0]
                ds_conn_y = blk_h - ds_conn_y[1], blk_h - ds_conn_y[0]

                if wire_names is None:
                    ng = num_g_tracks[row_idx]
                    ngb = num_gb_tracks[row_idx]
                    nds = num_ds_tracks[row_idx]
                    if ng >= 1:
                        top_conn_y.append(g_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'g', ng, space=le_sp_tr))
                    if ngb >= 1:
                        bot_conn_y.append(gb_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'gb', ngb, space=le_sp_tr))
                    if nds >= 1:
                        bot_conn_y.append(ds_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'ds', nds, space=le_sp_tr))
                else:
                    wnames = wire_names[row_idx]
                    if wnames.get('g', False):
                        top_conn_y.append(g_conn_y)
                        top_wires.append(WireGroup(hm_layer, 'g', tr_manager=tr_manager,
                                                   name_list=wnames['g']))
                    if wnames.get('gb', False):
                        bot_conn_y.append(gb_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'gb', tr_manager=tr_manager,
                                                   name_list=wnames['gb']))
                    if wnames.get('ds', False):
                        bot_conn_y.append(ds_conn_y)
                        bot_wires.append(WireGroup(hm_layer, 'ds', tr_manager=tr_manager,
                                                   name_list=wnames['ds']))

            if bot_wires:
                wire_tree.add_wires(bot_wires, (row_idx, 0))
            if top_wires:
                wire_tree.add_wires(top_wires, (row_idx, 1))

            rprop_list.append(dict(
                mos_type=row_type,
                orient=row_orient,
                w=row_w,
                threshold=row_thres,
                kwargs=kwargs,
            ))
            rinfo_list.append(row_info)
            pinfo_list.append((bot_conn_y, top_conn_y, blk_h, bext_info, text_info,
                               min_row_height, row_pitch))

        return rprop_list, rinfo_list, pinfo_list, wire_tree

    def _place_with_num_tracks(self, tr_next, hm_layer, bot_conn_y, bot_wires, ytop_prev,
                               conn_delta, mos_pitch, tr_manager, last_track):

        # determine block placement
        ycur = ytop_prev
        tr_last_info = []
        for btr_info, (yb, yt) in zip(bot_wires, bot_conn_y):
            if isinstance(btr_info, int) or isinstance(btr_info, float):
                bot_ntr = btr_info
                if bot_ntr >= 1:
                    tr_last_info.append(tr_next + bot_ntr - 1)
            else:
                if btr_info:
                    bot_ntr, bot_loc = tr_manager.place_wires(hm_layer, btr_info, start_idx=tr_next)
                    tr_last_info.append((bot_loc[-1], btr_info[-1]))
                else:
                    bot_ntr = 0

            if bot_ntr >= 1:
                y_ttr = self.grid.track_to_coord(hm_layer, tr_next + bot_ntr - 1, unit_mode=True)
                ycur = max(ycur, y_ttr - yt + conn_delta)

        if not tr_last_info:
            tr_last_info = last_track

        # round Y coordinate to mos_pitch
        ycur = -(-ycur // mos_pitch) * mos_pitch
        return ycur, tr_last_info

    def _place_mirror_or_sub(self, row_type, row_thres, lch_unit, mos_pitch, ydelta, ext_info):
        # find substrate parameters
        sub_type = 'ntap' if row_type == 'pch' or row_type == 'ntap' else 'ptap'
        w_sub = self._laygo_info['w_sub']
        min_sub_tracks = self._laygo_info['min_sub_tracks']
        sub_info = self._tech_cls.get_laygo_sub_row_info(lch_unit, w_sub, sub_type, row_thres)
        sub_ext_info = sub_info['ext_top_info']

        # quantize substrate height to top layer pitch.
        sub_height = sub_info['arr_y'][1]
        min_sub_height = mos_pitch
        top_pitch = self.grid.get_track_pitch(self._laygo_info.top_layer, unit_mode=True)
        sub_pitch = lcm([mos_pitch, top_pitch])
        for layer, num_tr in min_sub_tracks:
            tr_pitch = self.grid.get_track_pitch(layer, unit_mode=True)
            min_sub_height = max(min_sub_height, num_tr * tr_pitch)
            sub_pitch = lcm([sub_pitch, tr_pitch])

        real_sub_height = max(sub_height, min_sub_height)
        real_sub_height = -(-real_sub_height // sub_pitch) * sub_pitch
        sub_extw = (real_sub_height - sub_height) // mos_pitch

        # repeat until we satisfy both substrate and mirror row constraint
        ext_w = -(-ydelta // mos_pitch)
        ext_w_valid = False
        while not ext_w_valid:
            ext_w_valid = True
            # check we satisfy substrate constraint
            valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, sub_ext_info,
                                                                     ext_info)
            ext_w_test = ext_w + sub_extw
            if ext_w_test < valid_widths[-1] and ext_w_test not in valid_widths:
                # did not pass substrate constraint, update extension width
                ext_w_valid = False
                ext_w_test = valid_widths[bisect.bisect_left(valid_widths, ext_w_test)]
                ext_w = ext_w_test - sub_extw
                continue

            # check we satisfy mirror extension constraint
            valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_info, ext_info)
            ext_w_test = ext_w * 2
            if ext_w_test < valid_widths[-1] and ext_w_test not in valid_widths:
                # did not pass extension constraint, update extension width.
                ext_w_valid = False
                ext_w_test = valid_widths[bisect.bisect_left(valid_widths, ext_w_test)]
                ext_w = -(-ext_w_test // 2)

        return ext_w, sub_extw

    def _place_rows(self, ybot, tot_height_pitch, rprop_list, rinfo_list, pinfo_list,
                    wire_tree, min_htot):
        lch_unit = self._laygo_info.lch_unit

        grid = self.grid
        mos_pitch = self._tech_cls.get_mos_pitch(unit_mode=True)
        vm_layer = self._tech_cls.get_dig_conn_layer()
        hm_layer = vm_layer + 1
        vm_le_sp = grid.get_line_end_space(vm_layer, 1, unit_mode=True)

        num_rows = len(rprop_list)
        ext_params_list = []
        prev_ext_info = None
        prev_ext_h = 0
        ytop_prev = ybot
        ytop_vm_prev = None
        # first pass: determine Y coordinates of each row.
        for idx, (rprop, pinfo) in enumerate(zip(rprop_list, pinfo_list)):
            row_type = rprop['mos_type']
            row_thres = rprop['threshold']

            (bot_conn_y, top_conn_y, blk_height, ext_bot_info, ext_top_info,
             min_row_height, row_pitch) = pinfo

            is_sub = (row_type == 'ptap' or row_type == 'ntap')

            # find Y coordinate of current block from track/mirror placement constraints
            if idx == 0 and is_sub:
                # bottom substrate has orientation R0 and no gate tracks, just abut to bottom.
                ycur = ytop_prev
                cur_bot_ext_h = 0
            else:
                ycur = ytop_prev
                wire_groups = wire_tree.get_wire_groups((idx, 0))
                if wire_groups is not None:
                    # find Y coordinate that allows us to connect to top bottom track
                    for (_, yt), wg in zip(bot_conn_y, wire_groups):
                        _, tr_idx, tr_w = wg.last_track
                        via_ext = grid.get_via_extensions(vm_layer, 1, tr_w, unit_mode=True)[0]
                        y_ttr = grid.get_wire_bounds(hm_layer, tr_idx, width=tr_w,
                                                     unit_mode=True)[1]
                        ycur = max(ycur, y_ttr + via_ext - yt)
                    ycur = -(-ycur // mos_pitch) * mos_pitch

                # if previous row has top wires, make sure vm line-end spacing constraint is met
                if ytop_vm_prev is not None and bot_conn_y:
                    conn_yb = min((yintv[0] for yintv in bot_conn_y))
                    ycur = max(ycur, ytop_vm_prev + vm_le_sp - conn_yb)
                    ycur = -(-ycur // mos_pitch) * mos_pitch

                # make sure extension constraints is met
                if idx != 0:
                    valid_widths = self._tech_cls.get_valid_extension_widths(lch_unit, ext_bot_info,
                                                                             prev_ext_info)
                    cur_bot_ext_h = (ycur - ytop_prev) // mos_pitch
                    ext_h = prev_ext_h + cur_bot_ext_h
                    if ext_h < valid_widths[-1] and ext_h not in valid_widths:
                        # make sure extension height is valid
                        ext_h = valid_widths[bisect.bisect_left(valid_widths, ext_h)]
                        cur_bot_ext_h = ext_h - prev_ext_h
                else:
                    # nmos/pmos at bottom row.  Need to check we can draw mirror image row.
                    tmp = self._place_mirror_or_sub(row_type, row_thres, lch_unit, mos_pitch,
                                                    ycur - ybot, ext_bot_info)
                    cur_bot_ext_h, self._bot_sub_extw = tmp

                ycur = ytop_prev + cur_bot_ext_h * mos_pitch

            # move top tracks and find top coordinate
            ytop = max(ycur + blk_height, ytop_prev + min_row_height)
            wire_groups = wire_tree.get_wire_groups((idx, 1))
            ytop_vm = None
            if wire_groups is not None:
                # move the top tracks so we can connect to them
                for (yb, _), wg in zip(top_conn_y, wire_groups):
                    _, tr_idx, tr_w = wg.first_track
                    via_ext = grid.get_via_extensions(vm_layer, 1, tr_w, unit_mode=True)[0]
                    idx_targ = grid.find_next_track(hm_layer, ycur + yb + via_ext,
                                                    tr_width=tr_w, half_track=True,
                                                    mode=1, unit_mode=True)
                    if tr_idx < idx_targ:
                        wg.move_by(idx_targ - tr_idx, propagate=True)
                    # update ytop
                    _, last_idx, last_w = wg.last_track
                    ytop_wire_cur = grid.get_wire_bounds(hm_layer, last_idx, width=last_w,
                                                         unit_mode=True)[1]
                    via_ext = grid.get_via_extensions(vm_layer, 1, last_w, unit_mode=True)[0]
                    if ytop_vm is None:
                        ytop_vm = ytop_wire_cur + via_ext
                    else:
                        ytop_vm = max(ytop_vm, ytop_wire_cur + via_ext)
                    ytop = max(ytop, ytop_wire_cur)

            ytop = -(-ytop // row_pitch) * row_pitch
            if idx == num_rows - 1:
                # this is the last row, quantize total height
                ytop = max(ytop, min_htot)
                tot_height = -(-(ytop - ybot) // tot_height_pitch) * tot_height_pitch
                ytop = ybot + tot_height
                if is_sub:
                    # last row is substrate, abut to top edge
                    ycur = ytop - blk_height
                    cur_bot_ext_h = (ycur - ytop_prev) // mos_pitch
                    cur_top_ext_h = 0
                else:
                    # last row is transistor, make sure mirror extension constraint passes
                    pass_mirror = False
                    while not pass_mirror:
                        tmp = self._place_mirror_or_sub(row_type, row_thres, lch_unit, mos_pitch,
                                                        ytop - ycur - blk_height, ext_top_info)
                        cur_top_ext_h, self._top_sub_extw = tmp
                        ytest = ycur + blk_height + cur_top_ext_h * mos_pitch
                        if ytest != ytop:
                            ytop = -(-ytest // row_pitch) * row_pitch
                            # step 4: round to total height pitch
                            tot_height = -(-(ytop - ybot) // tot_height_pitch) * tot_height_pitch
                            ytop = ybot + tot_height
                        else:
                            pass_mirror = True

                    cur_top_ext_h = (ytop - ycur - blk_height) // mos_pitch
            else:
                # this is not the last row, move the next tracks outside of this row
                cur_top_ext_h = (ytop - ycur - blk_height) // mos_pitch
                wire_groups = wire_tree.get_wire_groups((idx + 1, 0))
                if wire_groups is not None:
                    for wg in wire_groups:
                        _, tr_idx, tr_w = wg.first_track
                        idx_targ = grid.find_next_track(hm_layer, ytop,
                                                        tr_width=tr_w,
                                                        half_track=True,
                                                        mode=1, unit_mode=True)
                        if tr_idx < idx_targ:
                            wg.move_by(idx_targ - tr_idx, propagate=True)

            # record information
            rprop['row_y'] = ytop_prev, ycur, ycur + blk_height, ytop
            ext_y = ybot if idx == 0 else rprop_list[idx - 1]['row_y'][2]
            ext_params_list.append((prev_ext_h + cur_bot_ext_h, ext_y))

            # update previous row information
            ytop_prev = ytop
            ytop_vm_prev = ytop_vm
            prev_ext_info = ext_top_info
            prev_ext_h = cur_top_ext_h

        # second pass: move tracks to minimize resistance, then record track intervals.
        for idx in range(len(pinfo_list) - 1, -1, -1):
            bot_conn_y = pinfo_list[idx][0]
            rinfo = rinfo_list[idx]
            ycur = rprop_list[idx]['row_y'][1]

            bot_wire_groups = wire_tree.get_wire_groups((idx, 0))
            top_wire_groups = wire_tree.get_wire_groups((idx, 1))
            if bot_wire_groups is not None:
                for (yb, yt), wg in zip(bot_conn_y, bot_wire_groups):
                    _, tr_idx, tr_w = wg.last_track
                    via_ext = grid.get_via_extensions(vm_layer, 1, tr_w, unit_mode=True)[0]
                    idx_max = grid.find_next_track(hm_layer, ycur + yt - via_ext,
                                                   tr_width=tr_w, half_track=True,
                                                   mode=-1, unit_mode=True)
                    if idx_max > tr_idx:
                        wg.move_up(idx_max - tr_idx)

                    rinfo['%s_intv' % wg.type] = wg.interval
                    rinfo['%s_wires' % wg.type] = (wg.names, wg.locations)

            if top_wire_groups is not None:
                for wg in top_wire_groups:
                    rinfo['%s_intv' % wg.type] = wg.interval
                    rinfo['%s_wires' % wg.type] = (wg.names, wg.locations)

            for wtype in ('g', 'gb', 'ds'):
                key = '%s_intv' % wtype
                if key not in rinfo:
                    rinfo[key] = (0, 0)
                    rinfo['%s_wires' % wtype] = (None, None)

        return ext_params_list

    @classmethod
    def compute_row_info(cls, laygo_info, rprop_list, dy=0):
        lch_unit = laygo_info.lch_unit
        tcls = laygo_info.tech_cls
        grid = laygo_info.grid

        mos_pitch = tcls.get_mos_pitch(unit_mode=True)
        vm_layer = tcls.get_dig_conn_layer()
        hm_layer = vm_layer + 1
        via_ext = grid.get_via_extensions(vm_layer, 1, 1, unit_mode=True)[0]

        ext_params_list = []
        rinfo_list = []
        # first pass: determine Y coordinates of each row.
        for idx, rprop in enumerate(rprop_list):
            yb_row, yb_cur, yt_cur, yt_row = rprop['row_y']
            row_orient = rprop['orient']
            row_type = rprop['mos_type']
            row_w = rprop['w']
            row_thres = rprop['threshold']
            kwargs = rprop.get('kwargs', None)
            if kwargs is None:
                kwargs = {}

            if row_type == 'nch' or row_type == 'pch':
                row_info = tcls.get_laygo_mos_row_info(lch_unit, row_w, row_w, row_type,
                                                       row_thres, '', '', **kwargs)
            elif row_type == 'ptap' or row_type == 'ntap':
                row_info = tcls.get_laygo_sub_row_info(lch_unit, row_w, row_type,
                                                       row_thres, **kwargs)
            else:
                raise ValueError('Unknown row type: %s' % row_type)

            # record information
            ext_y = yb_row if idx == 0 else rprop_list[idx - 1]['row_y'][2]
            rinfo_list.append(row_info)
            ext_params_list.append(((yb_cur - ext_y) // mos_pitch, ext_y))

            # record track intervals
            btr = grid.find_next_track(hm_layer, dy + yb_row, half_track=True,
                                       mode=1, unit_mode=True)
            ttr = grid.find_next_track(hm_layer, dy + yt_row, half_track=True,
                                       mode=-1, unit_mode=True)
            g_conn_y = row_info.get('g_conn_y', (0, 0))
            gb_conn_y = row_info['gb_conn_y']
            ds_conn_y = row_info['ds_conn_y']
            if row_orient == 'R0':
                yt_g = dy + yb_cur + g_conn_y[1]
                yb_gb = dy + yb_cur + gb_conn_y[0]
                yb_ds = dy + yb_cur + ds_conn_y[0]
                gtr = grid.find_next_track(hm_layer, yt_g - via_ext, half_track=True,
                                           mode=-1, unit_mode=True)
                gbtr = grid.find_next_track(hm_layer, yb_gb + via_ext, half_track=True,
                                            mode=1, unit_mode=True)
                dstr = grid.find_next_track(hm_layer, yb_ds + via_ext, half_track=True,
                                            mode=1, unit_mode=True)
                row_info['g_intv'] = (btr, max(btr, gtr + 1))
                row_info['gb_intv'] = (gbtr, max(ttr + 1, gbtr))
                row_info['ds_intv'] = (dstr, max(ttr + 1, dstr))
            else:
                yb_g = dy + yt_cur - g_conn_y[1]
                yt_gb = dy + yt_cur - gb_conn_y[0]
                yt_ds = dy + yt_cur - ds_conn_y[0]
                gtr = grid.find_next_track(hm_layer, yb_g + via_ext, half_track=True,
                                           mode=1, unit_mode=True)
                gbtr = grid.find_next_track(hm_layer, yt_gb - via_ext, half_track=True,
                                            mode=-1, unit_mode=True)
                dstr = grid.find_next_track(hm_layer, yt_ds - via_ext, half_track=True,
                                            mode=-1, unit_mode=True)
                row_info['g_intv'] = (gtr, max(ttr + 1, gtr))
                row_info['gb_intv'] = (btr, max(btr, gbtr + 1))
                row_info['ds_intv'] = (btr, max(btr, dstr + 1))

        return ext_params_list, rinfo_list

    def get_row_info(self, row_idx):
        # type : (int) -> Dict[str, Any]
        """Returns the row layout information dictionary."""
        return self._row_info_list[row_idx]

    def get_num_tracks(self, row_idx, tr_type):
        row_info = self._row_info_list[row_idx]
        intv = row_info['%s_intv' % tr_type]
        return intv[1] - intv[0]

    def get_track_index(self, row_idx, tr_type, tr_idx):
        row_info = self._row_info_list[row_idx]
        orient = self._row_prop_list[row_idx]['orient']
        intv = row_info['%s_intv' % tr_type]
        ntr = intv[1] - intv[0]
        if tr_idx >= ntr:
            raise IndexError('tr_idx = %d >= %d' % (tr_idx, ntr))
        if tr_idx < 0:
            tr_idx += ntr
        if tr_idx < 0:
            raise IndexError('index out of range.')

        if orient == 'R0':
            return intv[0] + tr_idx
        else:
            return intv[1] - 1 - tr_idx

    def get_wire_id(self, row_idx, tr_type, wire_idx=0, wire_name=''):
        # type: (int, str, int, str) -> TrackID
        row_info = self._row_info_list[row_idx]
        name_list, loc_list = row_info['%s_wires' % tr_type]
        hm_layer = self.conn_layer + 1
        if wire_name:
            idx = -1
            for j in range(wire_idx + 1):
                idx = name_list.index(wire_name, idx + 1)
            cur_name = wire_name
            cur_loc = loc_list[idx]
        else:
            cur_name = name_list[wire_idx]
            cur_loc = loc_list[wire_idx]

        cur_width = self._tr_manager.get_width(hm_layer, cur_name)
        return TrackID(hm_layer, cur_loc, width=cur_width)

    def get_track_interval(self, row_idx, tr_type):
        row_info = self._row_info_list[row_idx]
        return row_info['%s_intv' % tr_type]

    def make_track_id(self, row_idx, tr_type, tr_idx, width=1, num=1, pitch=0):
        tid = self.get_track_index(row_idx, tr_type, tr_idx)
        return TrackID(self.conn_layer + 1, tid, width=width, num=num, pitch=pitch)

    def set_laygo_size(self, num_col=None):
        if self._laygo_size is None:
            # compute total number of columns
            if num_col is None:
                if self._laygo_info.num_col is not None:
                    num_col = self._laygo_info.num_col
                else:
                    num_col = 0
                    for intv in self._used_list:
                        num_col = max(num_col, intv.get_end())

            self._laygo_info.set_num_col(num_col)
            self._laygo_size = num_col, self.num_rows

            top_layer = self._laygo_info.top_layer
            draw_boundaries = self._laygo_info.draw_boundaries

            width = self._laygo_info.tot_width
            height = self.tot_height
            if draw_boundaries:
                height += self._top_end_master.bound_box.height_unit
            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            if not draw_boundaries:
                self.array_box = self.bound_box
            self.add_cell_boundary(bound_box)

    def add_laygo_mos(self, row_idx, col_idx, seg, gate_loc='d',
                      stack=False, is_sub=False, flip=False, **kwargs):
        # type: (int, int, int, str, bool, bool, bool, **kwargs) -> Dict[str, WireArray]
        """Adds a laygo transistor at the given location.

        Parameters
        ----------
        row_idx : int
            the row index.
        col_idx : int
            the left-most column index.
        seg : int
            number of segments.  For stacked transistors,
            number of fingers = 2 * number of segments.
        gate_loc : str
            gate alignment location.  Either 'd' or 's'.
        stack : bool
            True to draw a 2-stack transistor.
        is_sub : bool
            True to draw a substrate contact.
        flip : bool
            True to flip source/drain/gate left and right.  Useful if this transistor should
            be a mirror image.
        **kwargs :
            optional Laygo primitive arguments.

        Returns
        -------
        ports : Dict[str, WireArray]
            a dictionary from port names to port WireArray objects.
        """
        if seg <= 0:
            raise ValueError('Cannot draw non-positive segments.')

        row_info = self._row_info_list[row_idx]
        row_type = row_info['row_type']

        if gate_loc != 'd' and gate_loc != 's':
            raise ValueError("gate_loc must be 'd' or 's'.")

        ports = {}
        if is_sub or row_type == 'ntap' or row_type == 'ptap':
            if seg % 2 == 1:
                raise ValueError('Cannot draw odd segments of substrate connection.')
            nx = seg // 2
            inst = self._add_laygo_primitive('sub', loc=(col_idx, row_idx), nx=nx, spx=2, **kwargs)
            port_name = 'VDD' if row_type == 'ntap' or row_type == 'pch' else 'VSS'
            for name in (port_name, port_name + '_s', port_name + '_d'):
                ports[name] = WireArray.list_to_warr(inst.get_all_port_pins(name))
        else:
            if stack:
                blk_type = 'stack2' + gate_loc
                num2 = seg
                num1 = False
            else:
                blk_type = 'fg2' + gate_loc
                num2 = seg // 2
                num1 = (seg % 2 == 1)

            ne = (num2 + 1) // 2
            no = num2 - ne
            if num2 > 0:
                inst = [self._add_laygo_primitive(blk_type, loc=(col_idx, row_idx), nx=ne, spx=4,
                                                  **kwargs)]
                if no > 0:
                    inst.append(self._add_laygo_primitive(blk_type, loc=(col_idx + 2, row_idx),
                                                          flip=True, nx=no, spx=4, **kwargs))
            else:
                inst = []

            if num1:
                col = col_idx + num2 * 2
                blk_type = 'fg1' + gate_loc
                inst1 = self._add_laygo_primitive(blk_type, loc=(col, row_idx), **kwargs)
            else:
                inst1 = None

            if gate_loc == 's':
                names = ['s', 'd', 'g', 'g0', 'g1']
            else:
                names = ['s', 'd', 'g']

            for name in names:
                if flip:
                    if name == 'd':
                        cur_name = 's'
                    elif name == 's':
                        cur_name = 'd'
                    elif name == 'g0':
                        cur_name = 'g1'
                    elif name == 'g1':
                        cur_name = 'g0'
                    else:
                        cur_name = name
                else:
                    cur_name = name

                if inst:
                    pins = inst[0].get_all_port_pins(cur_name)
                    if len(inst) > 1:
                        pins.extend(inst[1].get_all_port_pins(cur_name))
                else:
                    pins = []

                if inst1 is not None:
                    if cur_name == 'g0' or cur_name == 'g1':
                        if num2 % 2 == 0 and cur_name == 'g0' or num2 % 2 == 1 and cur_name == 'g1':
                            pins.extend(inst1.port_pins_iter('g'))
                    else:
                        pins.extend(inst1.port_pins_iter(cur_name))

                if pins:
                    ports[name] = WireArray.list_to_warr(pins)

        return ports

    def _add_laygo_primitive(self, blk_type, loc=(0, 0), flip=False, nx=1, spx=0, **kwargs):
        # type: (str, Tuple[int, int], bool, int, int, **kwargs) -> Instance

        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= self.num_rows:
            raise ValueError('Cannot add primitive at row %d' % row_idx)

        row_info = self._row_info_list[row_idx]
        row_type = row_info['row_type']
        wblk = kwargs.pop('w', row_info['w_max'])

        col_width = self._laygo_info.col_width

        rprop = self._row_prop_list[row_idx]
        row_orient = rprop['orient']
        _, ycur, ytop, _ = rprop['row_y']
        options = rprop.get('kwargs', None)

        if options is not None:
            for key, val in options.items():
                if key not in kwargs:
                    kwargs[key] = val

        # make master
        params = dict(
            row_info=row_info,
            options=kwargs,
        )
        if row_type == 'ntap' or row_type == 'ptap':
            master = self.new_template(params=params, temp_cls=LaygoSubstrate)
        else:
            params['blk_type'] = blk_type
            params['w'] = wblk
            master = self.new_template(params=params, temp_cls=LaygoPrimitive)

        num_col = master.num_col
        intv = self._used_list[row_idx]
        lay_info = master.layout_info
        endl, endr = master.lr_edge_info
        endb, endt = master.tb_ext_info
        x0 = self._laygo_info.col_to_coord(col_idx, unit_mode=True)
        if row_orient == 'MX':
            endb, endt = endt, endb
        if flip:
            x0 += num_col * col_width
            endl, endr = endr, endl
            endb = endb.reverse()
            endt = endt.reverse()

        ext_info = endb, endt
        endl_info = (endl, lay_info)
        endr_info = (endr, lay_info)
        for inst_num in range(nx):
            intv_offset = col_idx + spx * inst_num
            inst_intv = intv_offset, intv_offset + num_col
            if not intv.add(inst_intv, ext_info, endl_info, endr_info):
                raise ValueError('Cannot add primitive on row %d, '
                                 'column [%d, %d).' % (row_idx, inst_intv[0], inst_intv[1]))

        if row_orient == 'R0':
            y0 = ycur
            orient = 'MY' if flip else 'R0'
        else:
            y0 = ytop
            orient = 'R180' if flip else 'MX'

        # convert horizontal pitch to resolution units
        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx * col_width, unit_mode=True)

    def fill_space(self):
        if self._laygo_size is None:
            raise ValueError('laygo_size must be set before filling spaces.')

        num_col = self._laygo_size[0]
        # add space blocks
        total_intv = (0, num_col)
        endl_iter = self._laygo_edgel.row_end_iter()
        endr_iter = self._laygo_edger.row_end_iter()
        for row_idx, (intv, endl, endr) in enumerate(zip(self._used_list, endl_iter, endr_iter)):
            for (start, end), end_info in zip(*intv.get_complement(total_intv, endl, endr)):
                self._add_laygo_space(end_info, num_blk=end - start, loc=(start, row_idx))

        # draw extensions
        ext_endl_infos, ext_endr_infos = [], []
        laygo_info = self._laygo_info
        tech_cls = laygo_info.tech_cls
        for bot_ridx in range(0, self.num_rows - 1):
            w, yext = self._ext_params[bot_ridx + 1]
            bot_ext_list = self._get_ext_info_row(bot_ridx, 1)
            top_ext_list = self._get_ext_info_row(bot_ridx + 1, 0)
            edgel, edger = tech_cls.draw_extensions(self, laygo_info, w, yext,
                                                    bot_ext_list, top_ext_list)
            ext_endl_infos.append(edgel)
            ext_endr_infos.append(edger)

        # set edge information
        endl_list, endr_list = [], []
        for intv in self._used_list:
            endl, endr = intv.get_end_info(num_col)
            endl_list.append(endl)
            endr_list.append(endr)

        self._lr_edge_info = (DigitalEdgeInfo([0], [LaygoEdgeInfo(endl_list, ext_endl_infos)], []),
                              DigitalEdgeInfo([0], [LaygoEdgeInfo(endr_list, ext_endr_infos)], []))
        self._tb_ext_info = (DigitalExtInfo(self._get_ext_info_row(self.num_rows - 1, 1)),
                             DigitalExtInfo(self._get_ext_info_row(0, 0)))

        # draw boundaries and return guard ring supplies in boundary cells
        return self._draw_boundary_cells()

    def _get_ext_info_row(self, row_idx, ext_idx):
        intv = self._used_list[row_idx]
        return [ext_info[ext_idx] for ext_info in intv.values()]

    def _add_laygo_space(self, adj_end_info, num_blk=1, loc=(0, 0), **kwargs):
        col_idx, row_idx = loc
        row_info = self._row_info_list[row_idx]
        rprop = self._row_prop_list[row_idx]
        intv = self._used_list[row_idx]

        row_y = rprop['row_y']
        row_orient = rprop['orient']

        params = dict(
            row_info=row_info,
            num_blk=num_blk,
            left_blk_info=adj_end_info[0][0],
            right_blk_info=adj_end_info[1][0],
        )
        params.update(kwargs)
        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        master = self.new_template(params=params, temp_cls=LaygoSpace)

        # update used interval
        lay_info = master.layout_info
        endl, endr = master.lr_edge_info
        endb, endt = master.tb_ext_info
        if row_orient == 'MX':
            endb, endt = endt, endb

        ext_info = endb, endt
        endl_info = (endl, lay_info)
        endr_info = (endr, lay_info)
        inst_intv = (col_idx, col_idx + num_blk)
        if not intv.add(inst_intv, ext_info, endl_info, endr_info):
            raise ValueError('Cannot add space on row %d, '
                             'column [%d, %d)' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = self._laygo_info.col_to_coord(col_idx, unit_mode=True)
        y0 = row_y[1] if row_orient == 'R0' else row_y[2]
        self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=row_orient,
                          unit_mode=True)

    def _draw_boundary_cells(self):
        if self._laygo_info.draw_boundaries:
            if self._laygo_size is None:
                raise ValueError('laygo_size must be set before drawing boundaries.')

            end_mode = self._laygo_info.end_mode
            row_edge_infos = self._row_layout_info['row_edge_infos']
            if end_mode & 8 != 0:
                edgel_infos = self._lr_edge_info[0].master_infos_iter(row_edge_infos)
            else:
                edgel_infos = []
            if end_mode & 4 != 0:
                edger_infos = self._lr_edge_info[1].master_infos_iter(row_edge_infos)
            else:
                edger_infos = []
            yt = self.bound_box.top_unit
            tmp = self._tech_cls.draw_boundaries(self, self._laygo_info, self._laygo_size[0], yt,
                                                 self._bot_end_master, self._top_end_master,
                                                 edgel_infos, edger_infos)
            arr_box, vdd_warrs, vss_warrs = tmp
            self.array_box = arr_box
            return vdd_warrs, vss_warrs

        return [], []
