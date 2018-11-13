# -*- coding: utf-8 -*-

from typing import Dict, Any, Set, List, Tuple

import abc
from itertools import chain

from bag.layout.util import BBox
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase, TemplateDB

from ..analog_core.base import AnalogBaseEdgeInfo

from ..laygo.base import LaygoEndRow, LaygoSubstrate
from ..laygo.core import LaygoBase, LaygoBaseInfo, LaygoIntvSet, DigitalEdgeInfo, DigitalExtInfo


class DigitalSpace(LaygoBase):
    """A space block in digital template.

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
        LaygoBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict(
            config='laygo configuration dictionary.',
            layout_info='LaygoBase layout information dictionary.',
            num_col='number of columns.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        layout_info = self.params['layout_info']
        num_col = self.params['num_col']

        self.set_rows_direct(layout_info, num_col=num_col, draw_boundaries=False, end_mode=0)

        self.fill_space()


class DigitalBase(TemplateBase, metaclass=abc.ABCMeta):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None

        hidden_params = kwargs.pop('hidden_params', {}).copy()
        hidden_params['digital_edgel'] = None
        hidden_params['digital_edger'] = None

        TemplateBase.__init__(self, temp_db, lib_name, params, used_names,
                              hidden_params=hidden_params, **kwargs)
        # initialize attributes
        tech_info = temp_db.grid.tech_info
        self._tech_cls = tech_info.tech_params['layout']['laygo_tech_class']
        self._laygo_info = None  # type: LaygoBaseInfo
        self._num_rows = 0
        self._row_layout_info = None
        self._row_height = 0
        self._dig_size = None
        self._ext_params = None
        self._used_list = None  # type: List[LaygoIntvSet]
        self._ext_end_list = None
        self._bot_end_master = None
        self._top_end_master = None
        self._bot_sub_master = None
        self._top_sub_master = None
        self._lr_edge_info = None
        self._tb_ext_info = None
        self._ybot = None
        self._ytop = None
        self._digital_edgel = None
        self._digital_edger = None

    @classmethod
    def get_sub_columns(cls, tech_info, lch_unit):
        tech_cls = tech_info.tech_params['layout']['laygo_tech_class']
        return tech_cls.get_sub_columns(lch_unit)

    @classmethod
    def get_sub_port_columns(cls, tech_info, lch_unit):
        tech_cls = tech_info.tech_params['layout']['laygo_tech_class']
        return tech_cls.get_sub_port_columns(lch_unit)

    @property
    def lch_unit(self):
        if 'config' in self.params:
            lch = self.params['config']['lch']
            grid = self.grid
            return int(round(lch / (grid.layout_unit * grid.resolution)))
        else:
            return self._laygo_info.lch_unit

    @property
    def sub_columns(self):
        return self.get_sub_columns(self.grid.tech_info, self.lch_unit)

    @property
    def conn_layer(self):
        # type: () -> int
        return self._tech_cls.get_dig_conn_layer()

    @property
    def num_cols(self):
        # type: () -> int
        return self._dig_size[0]

    @property
    def digital_size(self):
        # type: () -> Tuple[int, int]
        return self._dig_size

    @property
    def laygo_info(self):
        # type: () -> LaygoBaseInfo
        return self._laygo_info

    @property
    def row_layout_info(self):
        return self._row_layout_info

    @property
    def tb_ext_info(self):
        return self._tb_ext_info

    @property
    def lr_edge_info(self):
        return self._lr_edge_info

    def initialize(self, layout_info, num_rows, num_cols=None, draw_boundaries=False, end_mode=0,
                   guard_ring_nf=0):

        bot_sub_extw = layout_info['bot_sub_extw']
        top_sub_extw = layout_info['top_sub_extw']
        row_prop_list = layout_info['row_prop_list']

        num_laygo_rows = len(row_prop_list)
        bot_row_y = row_prop_list[0]['row_y']
        top_row_y = row_prop_list[-1]['row_y']

        self._laygo_info = LaygoBaseInfo(self.grid, layout_info['config'])
        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode
        self._laygo_info.set_num_col(num_cols)
        self.grid = self._laygo_info.grid
        top_layer = self._laygo_info.top_layer

        self._num_rows = num_rows

        self._row_layout_info = layout_info.copy()
        self._row_layout_info['top_layer'] = top_layer
        self._row_layout_info['guard_ring_nf'] = guard_ring_nf
        self._row_layout_info['draw_boundaries'] = draw_boundaries
        self._row_layout_info['end_mode'] = end_mode

        self._row_height = top_row_y[3]

        tech_cls = self._laygo_info.tech_cls
        default_end_info = tech_cls.get_default_end_info()
        def_edge_info = AnalogBaseEdgeInfo([(default_end_info, None)] * num_laygo_rows, [])
        self._used_list = [LaygoIntvSet(def_edge_info) for _ in range(num_rows)]
        self._ext_end_list = [[0, None, None] for _ in range(num_rows - 1)]

        lch = self._laygo_info.lch
        mos_pitch = self._laygo_info.mos_pitch
        tot_height = self._row_height * num_rows

        laygo_bot_extw = (bot_row_y[1] - bot_row_y[0]) // mos_pitch
        laygo_top_extw = (top_row_y[3] - top_row_y[2]) // mos_pitch
        bot_extw_tot = laygo_bot_extw + bot_sub_extw

        # set left and right end informations
        self._digital_edgel = self.params['digital_edgel']
        default_endlr_value = [def_edge_info] * num_rows
        if self._digital_edgel is None:
            self._digital_edgel = default_endlr_value
        self._digital_edger = self.params['digital_edger']
        if self._digital_edger is None:
            self._digital_edger = default_endlr_value

        self._ext_params = [None] * (num_rows + 1)   # type: List[Tuple[int, int]]
        if draw_boundaries:
            lch_unit = self._laygo_info.lch_unit
            w_sub = self._laygo_info['w_sub']
            bot_end = (end_mode & 1) != 0
            top_end = (end_mode & 2) != 0

            # create end row and substrate masters
            mtype = row_prop_list[0]['mos_type']
            thres = row_prop_list[0]['threshold']
            sub_type = 'ptap' if mtype == 'nch' else 'ntap'
            params = dict(
                lch=lch,
                mos_type=sub_type,
                threshold=thres,
                is_end=bot_end,
                top_layer=top_layer,
            )
            self._bot_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
            sub_info = tech_cls.get_laygo_sub_row_info(lch_unit, w_sub, sub_type, thres)
            params = dict(
                row_info=sub_info,
                options={},
            )
            self._bot_sub_master = self.new_template(params=params, temp_cls=LaygoSubstrate)

            if num_rows % 2 == 0:
                # because of mirroring, top and bottom masters are the same,
                # except for is_end parameter.
                params = dict(
                    lch=lch,
                    mos_type=sub_type,
                    threshold=thres,
                    is_end=top_end,
                    top_layer=top_layer,
                )
                self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
                self._top_sub_master = self._bot_sub_master
                top_extw = laygo_bot_extw
                top_extw_tot = top_extw + bot_sub_extw
            else:
                mtype = row_prop_list[-1]['mos_type']
                thres = row_prop_list[-1]['threshold']
                sub_type = 'ptap' if mtype == 'nch' else 'ntap'
                params = dict(
                    lch=lch,
                    mos_type=sub_type,
                    threshold=thres,
                    is_end=top_end,
                    top_layer=top_layer,
                )
                self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
                sub_info = tech_cls.get_laygo_sub_row_info(lch_unit, w_sub, sub_type, thres)
                params = dict(
                    row_info=sub_info,
                    options={},
                )
                self._top_sub_master = self.new_template(params=params, temp_cls=LaygoSubstrate)
                top_extw = laygo_top_extw
                top_extw_tot = top_extw + top_sub_extw

            y0 = self._bot_end_master.bound_box.height_unit
            y1 = y0 + self._bot_sub_master.bound_box.height_unit + bot_sub_extw * mos_pitch
            self._ybot = (y0, y1)
            bot_yext = y1 - bot_sub_extw * mos_pitch
            top_yext = y1 + tot_height - top_extw * mos_pitch
            y0 = top_yext + top_extw_tot * mos_pitch + self._top_sub_master.bound_box.height_unit
            y1 = y0 + self._top_end_master.bound_box.height_unit
            self._ytop = (y0, y1)

            # add extension between substrate and edge rows
            self._ext_params[0] = (bot_extw_tot, bot_yext)
            self._ext_params[num_rows] = (top_extw_tot, top_yext)
        else:
            self._ybot = (0, 0)
            self._ytop = (tot_height, tot_height)

        # add rest of extension parameters
        ycur = self._ybot[1] + self._row_height
        for bot_row_idx in range(num_rows - 1):
            w = laygo_top_extw if bot_row_idx % 2 == 0 else laygo_bot_extw

            self._ext_params[bot_row_idx + 1] = (w * 2, ycur - w * mos_pitch)
            ycur += self._row_height

        if num_cols is not None:
            self.set_digital_size(num_cols=num_cols)

    def set_digital_size(self, num_cols=None):
        if self._dig_size is None:
            if num_cols is None:
                num_cols = 0
                for intv in self._used_list:
                    num_cols = max(num_cols, intv.get_end())

            self._laygo_info.set_num_col(num_cols)
            self._dig_size = num_cols, self._num_rows

            top_layer = self._laygo_info.top_layer

            width = self._laygo_info.tot_width
            height = self._ytop[1]
            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            self.add_cell_boundary(bound_box)
            if not self._laygo_info.draw_boundaries:
                self.array_box = bound_box

    def get_track_index(self, row_idx, tr_type, tr_idx, dig_row_idx=0):
        row_prop = self._row_layout_info['row_prop_list'][row_idx]
        row_info = self._row_layout_info['row_info_list'][row_idx]

        orient = row_prop['orient']
        intv = row_info['%s_intv' % tr_type]
        ntr = intv[1] - intv[0]
        if tr_idx >= ntr:
            raise IndexError('tr_idx = %d >= %d' % (tr_idx, ntr))
        if tr_idx < 0:
            tr_idx += ntr
        if tr_idx < 0:
            raise IndexError('index out of range.')

        if orient == 'R0':
            ans = intv[0] + tr_idx
        else:
            ans = intv[1] - 1 - tr_idx

        if dig_row_idx != 0:
            # TODO: figure this out
            raise ValueError('Not supported yet.')
        else:
            dtr = self.grid.find_next_track(self.conn_layer + 1, self._ybot[1], mode=1,
                                            unit_mode=True)
            ans += dtr
        return ans

    def make_track_id(self, row_idx, tr_type, tr_idx, width=1, num=1, pitch=0, dig_row_idx=0):
        tidx = self.get_track_index(row_idx, tr_type, tr_idx, dig_row_idx=dig_row_idx)
        return TrackID(self.conn_layer + 1, tidx, width=width, num=num, pitch=pitch)

    def get_num_x_tracks(self, layer_id, half_int=False):
        row_height = self._row_height
        tr_pitch2 = self.grid.get_track_pitch(layer_id, unit_mode=True) // 2
        if row_height % tr_pitch2 != 0:
            raise ValueError('row height = %d not divisible '
                             'by pitch on layer %d' % (row_height, layer_id))

        num = row_height // tr_pitch2
        return num if half_int else (num // 2 if num % 2 == 0 else num / 2)

    def get_x_track_index(self, layer_id, dig_row_idx, tr_idx):
        row_height = self._row_height
        tr_pitch2 = self.grid.get_track_pitch(layer_id, unit_mode=True) // 2
        if row_height % tr_pitch2 != 0:
            raise ValueError('row height = %d not divisible '
                             'by pitch on layer %d' % (row_height, layer_id))

        y0 = dig_row_idx * self._row_height + self._ybot[1]
        tr_off = self.grid.coord_to_track(layer_id, y0, unit_mode=True)
        htr2 = int(round(2 * (tr_off + tr_idx))) + 1
        return htr2 // 2 if htr2 % 2 == 0 else htr2 / 2

    def make_x_track_id(self, layer_id, dig_row_idx, tr_idx, width=1, num=1, pitch=0):
        tidx = self.get_x_track_index(layer_id, dig_row_idx, tr_idx)
        return TrackID(layer_id, tidx, width=width, num=num, pitch=pitch)

    def add_digital_block(self, master, loc, flip=False, nx=1, spx=0):
        col_idx, row_idx = loc
        num_cols, num_rows = master.digital_size
        y0 = row_idx * self._row_height + self._ybot[1]
        endt, endb = master.tb_ext_info
        endl, endr = master.lr_edge_info
        if row_idx % 2 == 0:
            orient = 'MY' if flip else 'R0'
            rbot, rtop = row_idx, row_idx + num_rows
        else:
            orient = 'R180' if flip else 'MX'
            rbot, rtop = row_idx + 1 - num_rows, row_idx + 1
            y0 += self._row_height
            endl = endl.reverse()
            endr = endr.reverse()
            endt, endb = endb, endt

        if rbot < 0 or rtop > self._num_rows:
            raise ValueError('Cannot add block at row %d' % row_idx)

        col_width = self._laygo_info.col_width
        x0 = self._laygo_info.col_to_coord(col_idx, unit_mode=True)

        if flip:
            x0 += num_cols * col_width
            endl, endr = endr, endl
            endb = endb.reverse()
            endt = endt.reverse()

        if spx >= 0:
            coll, colr = col_idx, col_idx + (nx - 1) * spx + num_cols
        else:
            coll, colr = col_idx + (nx - 1) * spx, col_idx + num_cols
        for yidx in range(num_rows):
            rcur = rbot + yidx
            intv = self._used_list[rcur]
            if yidx == 0:
                if yidx == num_rows - 1:
                    cur_ext_info = endb, endt
                else:
                    cur_ext_info = endb, num_cols
            else:
                if yidx == num_rows - 1:
                    cur_ext_info = num_cols, endt
                else:
                    cur_ext_info = num_cols, num_cols

            # update ext_end_list
            if yidx < num_rows - 1:
                ext_end_cur = self._ext_end_list[rcur]
                if coll == 0:
                    ext_end_cur[1] = endl.get_ext_end(yidx)
                if colr > ext_end_cur[0]:
                    ext_end_cur[0] = colr
                    ext_end_cur[2] = endr.get_ext_end(yidx)

            edgel = endl.get_laygo_edge(yidx)
            edger = endr.get_laygo_edge(yidx)
            for inst_num in range(nx):
                intv_offset = col_idx + spx * inst_num
                inst_intv = intv_offset, intv_offset + num_cols
                if not intv.add(inst_intv, cur_ext_info, edgel, edger):
                    raise ValueError('Cannot add block on row %d, column '
                                     '[%d, %d).' % (rbot + yidx, inst_intv[0], inst_intv[1]))

        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx * col_width, unit_mode=True)

    def fill_space(self, port_cols=None):
        if self._dig_size is None:
            raise ValueError('digital size must be set before filling spaces.')

        # TODO: need to update laygo_endl_infos/laygo_endr_infos/
        # TODO: digital_endl_infos/digital_endr_infos parameters
        # TODO: for all instances in this block.

        # add spaces
        num_cols, num_rows = self._dig_size
        total_intv = (0, num_cols)
        for row_idx, (intv, ledgel, ledger) in enumerate(zip(self._used_list, self._digital_edgel,
                                                             self._digital_edger)):
            for (start, end), end_info in zip(*intv.get_complement(total_intv, ledgel, ledger)):
                space_params = dict(
                    config=self._row_layout_info['config'],
                    layout_info=self._row_layout_info,
                    num_col=end - start,
                    laygo_edgel=end_info[0],
                    laygo_edger=end_info[1],
                )
                space_master = self.new_template(params=space_params, temp_cls=DigitalSpace)
                self.add_digital_block(space_master, loc=(start, row_idx))

        # draw extensions
        ext_endl_infos, ext_endr_infos = [], []
        laygo_info = self._laygo_info
        tech_cls = laygo_info.tech_cls
        for bot_ridx in range(-1, num_rows):
            ext_val = self._ext_params[bot_ridx + 1]
            if ext_val is not None:
                w, yext = ext_val
                bot_ext_list = self._get_ext_info_row(bot_ridx, 1)
                top_ext_list = self._get_ext_info_row(bot_ridx + 1, 0)
                edgel, edger = tech_cls.draw_extensions(self, laygo_info, num_cols, w, yext,
                                                        bot_ext_list, top_ext_list)
                if 0 <= bot_ridx < num_rows - 1:
                    if edgel is None:
                        edgel = (yext, self._ext_end_list[bot_ridx][1])
                    if edger is None:
                        edger = (yext, self._ext_end_list[bot_ridx][2])
            else:
                edgel = edger = None
            ext_endl_infos.append(edgel)
            ext_endr_infos.append(edger)

        # set edge information
        row_y_list, lendl_list, lendr_list = [], [], []
        for row_idx, intv in enumerate(self._used_list):
            lendl, lendr = intv.get_end_info(num_cols)
            lendl_list.append(lendl)
            lendr_list.append(lendr)
            y0 = row_idx * self._row_height + self._ybot[1]
            if row_idx % 2 == 1:
                y0 += self._row_height
            row_y_list.append(y0)

        self._lr_edge_info = (DigitalEdgeInfo(row_y_list, lendl_list, ext_endl_infos),
                              DigitalEdgeInfo(row_y_list, lendr_list, ext_endr_infos))
        self._tb_ext_info = (DigitalExtInfo(self._get_ext_info_row(num_rows - 1, 1)),
                             DigitalExtInfo(self._get_ext_info_row(0, 0)))

        return self._draw_boundary_cells(port_cols)

    def _get_ext_info_row(self, row_idx, ext_idx):
        num_col, num_row = self._dig_size
        if row_idx == -1:
            ext_info = self._bot_sub_master.tb_ext_info[0]
            return [ext_info] * num_col
        elif row_idx == num_row:
            ext_info = self._top_sub_master.tb_ext_info[0]
            return [ext_info] * num_col
        else:
            intv = self._used_list[row_idx]
            ext_info_row = []
            for val in intv.values():
                test = val[ext_idx]
                if isinstance(test, int):
                    ext_info_row.append(test)
                else:
                    ext_info_row.extend(test.ext_iter())
            return ext_info_row

    def _draw_end_substrates(self, port_cols):
        laygo_info = self._laygo_info
        top_layer = laygo_info.top_layer
        guard_ring_nf = laygo_info.guard_ring_nf
        end_mode = laygo_info.end_mode

        num_col = self._dig_size[0]

        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0

        if port_cols is None:
            port_cols = set(range(0, num_col, 2))
            bot_sub2 = top_sub2 = None
        else:
            port_cols = set(port_cols)
            # get substrate master with no ports
            bot_sub2 = self._bot_sub_master.new_template_with(options=dict(export=False))
            top_sub2 = self._top_sub_master.new_template_with(options=dict(export=False))

        # add substrate blocks in substrate rows
        bot_warrs = []
        top_warrs = []
        ybot = self._ybot[0]
        ytop = self._ytop[0]
        for warrs, m1, m2, y, orient, name in ((bot_warrs, self._bot_sub_master, bot_sub2, ybot,
                                                'R0', 'XBSUB%d'),
                                               (top_warrs, self._top_sub_master, top_sub2, ytop,
                                                'MX', 'XTSUB%d')):
            port_name = 'VSS' if m1.has_port('VSS') else 'VDD'
            for col_idx in range(0, num_col, 2):
                xcur = laygo_info.col_to_coord(col_idx, unit_mode=True)
                if col_idx in port_cols:
                    inst = self.add_instance(m1, inst_name=name % col_idx, loc=(xcur, y),
                                             orient=orient, unit_mode=True)

                    warrs.extend(inst.get_all_port_pins(port_name))
                else:
                    self.add_instance(m2, inst_name=name % col_idx, loc=(xcur, y),
                                      orient=orient, unit_mode=True)

        edgel_infos, edger_infos = [], []
        for master, y, flip in ((self._bot_sub_master, ybot, False),
                                (self._top_sub_master, ytop, True)):
            layout_info = master.layout_info
            endl, endr = master.lr_edge_info
            rinfo = master.row_info
            if left_end:
                edge_params = dict(
                    top_layer=top_layer,
                    guard_ring_nf=guard_ring_nf,
                    is_end=True,
                    name_id=rinfo['row_name_id'],
                    layout_info=layout_info,
                    adj_blk_info=endl,
                    is_laygo=True,
                )
                edgel_infos.append((y, flip, edge_params))
            if right_end:
                edge_params = dict(
                    top_layer=top_layer,
                    guard_ring_nf=guard_ring_nf,
                    is_end=True,
                    name_id=rinfo['row_name_id'],
                    layout_info=layout_info,
                    adj_blk_info=endr,
                    is_laygo=True,
                )
                edger_infos.append((y, flip, edge_params))

        return edgel_infos, edger_infos, bot_warrs, top_warrs

    def _draw_boundary_cells(self, port_cols):
        laygo_info = self._laygo_info

        if laygo_info.draw_boundaries:
            if self._dig_size is None:
                raise ValueError('digital_size must be set before drawing boundaries.')

            # draw end substrates
            edgel_infos, edger_infos, bot_warrs, top_warrs = self._draw_end_substrates(port_cols)
            end_mode = self._laygo_info.end_mode
            row_edge_infos = self._row_layout_info['row_edge_infos']
            if end_mode & 8 != 0:
                edgel_infos = chain(edgel_infos,
                                    self._lr_edge_info[0].master_infos_iter(row_edge_infos))
            if end_mode & 4 != 0:
                edger_infos = chain(edger_infos,
                                    self._lr_edge_info[1].master_infos_iter(row_edge_infos))

            yt = self.bound_box.top_unit
            tmp = laygo_info.tech_cls.draw_boundaries(self, laygo_info, self._dig_size[0], yt,
                                                      self._bot_end_master, self._top_end_master,
                                                      edgel_infos, edger_infos)
            arr_box, gr_vdd_warrs, gr_vss_warrs = tmp
            self.array_box = arr_box
            return bot_warrs, top_warrs, gr_vdd_warrs, gr_vss_warrs

        return [], [], [], []
