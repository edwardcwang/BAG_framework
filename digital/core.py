# -*- coding: utf-8 -*-

import abc
from typing import Dict, Any, Set, List

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB

from ..laygo.base import LaygoEndRow, LaygoSubstrate
from ..laygo.core import LaygoBase, LaygoBaseInfo, LaygoIntvSet


class DigitalSpace(LaygoBase):
    """Stack driver substrate contact between core devices.

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
        super(DigitalSpace, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            config='laygo configuration dictionary.',
            dig_row_info='digital row information dictionary.',
            num_col='number of columns.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        dig_row_info = self.params['dig_row_info']
        num_col = self.params['num_col']

        self.set_rows_direct(dig_row_info)

        self.set_laygo_size(num_col=num_col)
        self.fill_space()


class DigitalBase(TemplateBase, metaclass=abc.ABCMeta):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None

        hidden_params = kwargs.pop('hidden_params', {}).copy()
        hidden_params['digital_endl_infos'] = None
        hidden_params['digital_endr_infos'] = None

        super(DigitalBase, self).__init__(temp_db, lib_name, params, used_names, hidden_params=hidden_params, **kwargs)
        self._laygo_info = None

        # initialize attributes
        self._num_rows = 0
        self._dig_size = None
        self._row_info = None
        self._row_height = 0
        self._used_list = None  # type: List[LaygoIntvSet]
        self._bot_end_master = None
        self._top_end_master = None
        self._bot_sub_master = None
        self._top_sub_master = None
        self._ybot = None
        self._ytop = None
        self._ext_params = None
        self._ext_edge_infos = None
        self._endl_infos = None
        self._endr_infos = None

    @property
    def digital_size(self):
        return self._dig_size

    @property
    def laygo_info(self):
        return self._laygo_info

    def initialize(self, row_info, num_rows, draw_boundaries, end_mode, guard_ring_nf=0):

        laygo_row_infos = row_info['row_infos']

        self._laygo_info = LaygoBaseInfo(self.grid, row_info['config'])
        self.grid = self._laygo_info.grid
        self._row_info = row_info
        self._num_rows = num_rows
        self._row_height = row_info['row_height']

        num_laygo_rows = len(laygo_row_infos)
        self._laygo_info.guard_ring_nf = guard_ring_nf
        self._laygo_info.draw_boundaries = draw_boundaries
        self._laygo_info.end_mode = end_mode

        tech_cls = self._laygo_info.tech_cls
        default_end_info = tech_cls.get_default_end_info()
        default_dig_end_info = [default_end_info] * num_laygo_rows

        self._used_list = [LaygoIntvSet(default_dig_end_info) for _ in range(num_rows)]

        lch = self._laygo_info.lch
        top_layer = self._laygo_info.top_layer
        mos_pitch = self._laygo_info.mos_pitch
        tot_height = self._row_height * num_rows

        bot_extw = row_info['bot_extw']
        bot_sub_extw = row_info['bot_sub_extw']
        bot_extw_tot = bot_extw + bot_sub_extw

        # set left and right end informations
        self._endl_infos = self.params['digital_endl_infos']
        if self._endl_infos is None:
            self._endl_infos = [[default_end_info] * num_laygo_rows] * num_rows
        self._endr_infos = self.params['digital_endr_infos']
        if self._endr_infos is None:
            self._endr_infos = [[default_end_info] * num_laygo_rows] * num_rows

        self._ext_params = []
        if draw_boundaries:
            lch_unit = self._laygo_info.lch_unit
            w_sub = self._laygo_info['w_sub']
            bot_end = (end_mode & 1) != 0
            top_end = (end_mode & 2) != 0

            # create end row and substrate masters
            mtype = laygo_row_infos[0]['row_type']
            thres = laygo_row_infos[0]['threshold']
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
            sub_ext_info = sub_info['ext_top_info']
            bot_ext_params = dict(
                lch=lch,
                w=bot_extw_tot,
                fg=1,
                top_ext_info=row_info['bot_ext_info'],
                bot_ext_info=sub_ext_info,
                is_laygo=True,
            )

            if num_rows % 2 == 0:
                # because of mirroring, top and bottom masters are the same, except for is_end parameter.
                params = dict(
                    lch=lch,
                    mos_type=sub_type,
                    threshold=thres,
                    is_end=top_end,
                    top_layer=top_layer,
                )
                self._top_end_master = self.new_template(params=params, temp_cls=LaygoEndRow)
                self._top_sub_master = self._bot_sub_master
                top_extw = bot_extw
                top_ext_params = bot_ext_params.copy()
                top_ext_params['bot_ext_info'] = bot_ext_params['top_ext_info']
                top_ext_params['top_ext_info'] = bot_ext_params['bot_ext_info']
                top_extw_tot = bot_extw_tot
            else:
                mtype = laygo_row_infos[-1]['row_type']
                thres = laygo_row_infos[-1]['threshold']
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
                top_extw = row_info['top_extw']
                top_sub_extw = row_info['top_sub_extw']
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
            self._ext_params.append((0, bot_extw_tot, bot_yext))
            self._ext_params.append((self._num_rows, top_extw_tot, top_yext))

        else:
            self._ybot = (0, 0)
            self._ytop = (tot_height, tot_height)

        # add rest of extension parameters
        ycur = self._ybot[1] + self._row_height
        for row_idx in range(num_rows - 1):
            if row_idx % 2 == 0:
                w = row_info['top_extw']
            else:
                w = bot_extw

            self._ext_params.append((row_idx + 1, w * 2, ycur - w * mos_pitch))
            ycur += self._row_height

    def set_digital_size(self, num_col=None):
        if self._dig_size is None:
            if num_col is None:
                num_col = 0
                for intv in self._used_list:
                    num_col = max(num_col, intv.get_end())

            self._laygo_info.set_num_col(num_col)
            self._dig_size = num_col, self._num_rows

            top_layer = self._laygo_info.top_layer

            width = self._laygo_info.tot_width
            height = self._ytop[1]

            bound_box = BBox(0, 0, width, height, self.grid.resolution, unit_mode=True)
            self.set_size_from_bound_box(top_layer, bound_box)
            self.add_cell_boundary(bound_box)

    def add_digital_block(self, master, loc=(0, 0), flip=False, nx=1, spx=0):
        col_idx, row_idx = loc
        if row_idx < 0 or row_idx >= self._num_rows:
            raise ValueError('Cannot add block at row %d' % row_idx)

        col_width = self._laygo_info.col_width

        intv = self._used_list[row_idx]
        inst_endl = master.get_left_edge_info()
        inst_endr = master.get_right_edge_info()
        if flip:
            inst_endl, inst_endr = inst_endr, inst_endl

        num_inst_col = master.laygo_size[0]
        ext_info = master.get_ext_bot_info(), master.get_ext_top_info()
        if row_idx % 2 == 1:
            ext_info = ext_info[1], ext_info[0]

        for inst_num in range(nx):
            intv_offset = col_idx + spx * inst_num
            inst_intv = intv_offset, intv_offset + num_inst_col
            if not intv.add(inst_intv, ext_info, inst_endl, inst_endr):
                raise ValueError('Cannot add primitive on row %d, '
                                 'column [%d, %d).' % (row_idx, inst_intv[0], inst_intv[1]))

        x0 = self._laygo_info.col_to_coord(col_idx, 's', unit_mode=True)
        if flip:
            x0 += master.digital_size[0]

        y0 = row_idx * self._row_height + self._ybot[1]
        if row_idx % 2 == 0:
            orient = 'MY' if flip else 'R0'
        else:
            y0 += self._row_height
            orient = 'R180' if flip else 'MX'

        # convert horizontal pitch to resolution units
        spx *= col_width

        inst_name = 'XR%dC%d' % (row_idx, col_idx)
        return self.add_instance(master, inst_name=inst_name, loc=(x0, y0), orient=orient,
                                 nx=nx, spx=spx, unit_mode=True)

    def fill_space(self, port_cols=None):
        if self._dig_size is None:
            raise ValueError('digital size must be set before filling spaces.')

        # TODO: need to update laygo_endl_infos/laygo_endr_infos/digital_endl_infos/digital_endr_infos parameters
        # TODO: for all instances in this block.

        # add spaces
        num_col = self._dig_size[0]
        total_intv = (0, num_col)
        for row_idx, (intv, endl_info, endr_info) in \
                enumerate(zip(self._used_list, self._endl_infos, self._endr_infos)):
            for (start, end), end_info in zip(*intv.get_complement(total_intv, endl_info, endr_info)):
                space_params = dict(
                    config=self._row_info['config'],
                    dig_row_info=self._row_info,
                    num_col=end-start,
                    laygo_endl_infos=end_info[0],
                    laygo_endr_infos=end_info[1],
                )
                space_master = self.new_template(params=space_params, temp_cls=DigitalSpace)
                self.add_digital_block(space_master, loc=(start, row_idx))

        # draw extensions
        self._ext_edge_infos = []
        laygo_info = self._laygo_info
        tech_cls = laygo_info.tech_cls
        for top_ridx, w, yext in self._ext_params:
            bot_ext_list = self._get_ext_info_row(top_ridx - 1, 1)
            top_ext_list = self._get_ext_info_row(top_ridx, 0)
            self._ext_edge_infos.extend(tech_cls.draw_extensions(self, laygo_info, w, yext, bot_ext_list,
                                                                 top_ext_list))

        return self._draw_boundary_cells(port_cols)

    def get_ext_bot_info(self):
        return self._get_ext_info_row(0, 0)

    def get_ext_top_info(self):
        return self._get_ext_info_row(self._num_rows - 1, 1)

    def _get_ext_info_row(self, row_idx, ext_idx):
        num_col, num_row = self._dig_size
        if row_idx == -1:
            ext_info = self._bot_sub_master.get_ext_top_info()
            return [ext_info] * num_col
        elif row_idx == num_row:
            ext_info = self._top_sub_master.get_ext_top_info()
            return [ext_info] * num_col
        else:
            intv = self._used_list[row_idx]
            ext_info_row = []
            for ext_info_inst in intv.values():
                ext_info_row.extend(ext_info_inst[ext_idx])
            return ext_info_row

    def get_left_edge_info(self):
        endl_list = []
        num_col = self._dig_size[0]
        for intv in self._used_list:
            endl, endr = intv.get_end_info(num_col)
            endl_list.append(endl)

        return endl_list

    def get_right_edge_info(self):
        endr_list = []
        num_col = self._dig_size[0]
        for intv in self._used_list:
            endl, endr = intv.get_end_info(num_col)
            endr_list.append(endr)

        return endr_list

    def _get_end_info_row(self, row_idx):
        num_col = self._dig_size[0]
        endl, endr = self._used_list[row_idx].get_end_info(num_col)
        return endl, endr

    @staticmethod
    def _flip_ud(orient):
        if orient == 'R0':
            return 'MX'
        elif orient == 'MX':
            return 'R0'
        elif orient == 'MY':
            return 'R180'
        elif orient == 'R180':
            return 'MY'
        else:
            raise ValueError('Unknonw orientation: %s' % orient)

    def _draw_end_substrates(self, port_cols):
        num_col = self._dig_size[0]
        top_layer = self._laygo_info.top_layer
        guard_ring_nf = self._laygo_info.guard_ring_nf
        spx = self._laygo_info.col_width
        end_mode = self._laygo_info.end_mode
        xr = self.bound_box.right_unit
        x0 = self._laygo_info.col_to_coord(0, 's', unit_mode=True)
        tech_cls = self._laygo_info.tech_cls

        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0

        if port_cols is None:
            port_cols = set(range(num_col))
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
        for warrs, m1, m2, y, orient, name in ((bot_warrs, self._bot_sub_master, bot_sub2, ybot, 'R0', 'XBSUB%d'),
                                               (top_warrs, self._top_sub_master, top_sub2, ytop, 'MX', 'XTSUB%d')):
            port_name = 'VSS' if m1.has_port('VSS') else 'VDD'
            xcur = x0
            for col_idx in range(num_col):
                if col_idx in port_cols:
                    inst = self.add_instance(m1, inst_name=name % col_idx, loc=(xcur, y), orient=orient, unit_mode=True)

                    warrs.extend(inst.get_all_port_pins(port_name))
                else:
                    self.add_instance(m2, inst_name=name % col_idx, loc=(xcur, y), orient=orient, unit_mode=True)
                xcur += spx

        edge_infos = []
        for master, y, orient in ((self._bot_sub_master, ybot, 'R0'), (self._top_sub_master, ytop, 'MX')):
            endl = master.get_left_edge_info()
            endr = master.get_right_edge_info()
            rinfo = master.row_info
            test_blk_info = tech_cls.get_laygo_blk_info('fg2d', rinfo['w_max'], rinfo)
            for x, is_end, flip_lr, end_flag in ((0, left_end, False, endl), (xr, right_end, True, endr)):
                edge_params = dict(
                    top_layer=top_layer,
                    guard_ring_nf=guard_ring_nf,
                    is_end=is_end,
                    name_id=rinfo['row_name_id'],
                    layout_info=test_blk_info['layout_info'],
                    adj_blk_info=end_flag,
                    is_laygo=True,
                )
                if orient == 'R0':
                    eorient = 'MY' if flip_lr else 'R0'
                else:
                    eorient = 'R180' if flip_lr else 'MX'
                edge_infos.append((x, y, eorient, edge_params))

        return edge_infos, bot_warrs, top_warrs

    def _draw_boundary_cells(self, port_cols):
        if self._laygo_info.draw_boundaries:
            if self._dig_size is None:
                raise ValueError('digital_size must be set before drawing boundaries.')

            # compute row edge information
            num_col = self._dig_size[0]
            end_mode = self._laygo_info.end_mode
            tech_cls = self._laygo_info.tech_cls
            xr = self.bound_box.right_unit

            left_end = (end_mode & 4) != 0
            right_end = (end_mode & 8) != 0

            # get edge information for each row
            ext_edge_infos = self._row_info['ext_edge_infos']
            row_edge_infos = self._row_info['row_edge_infos']

            # draw end substrates
            edge_infos, bot_warrs, top_warrs = self._draw_end_substrates(port_cols)

            # add extension edge in digital block
            for y, orient, edge_params in self._ext_edge_infos:
                tmp_copy = edge_params.copy()
                if orient == 'R0':
                    x = 0
                    tmp_copy['is_end'] = left_end
                else:
                    x = xr
                    tmp_copy['is_end'] = right_end
                edge_infos.append((x, y, orient, tmp_copy))

            for ridx in range(self._num_rows):
                endl_list, endr_list = self._get_end_info_row(ridx)
                if ridx % 2 == 0:
                    yscale = 1
                    yoff = self._ybot[1] + ridx * self._row_height
                else:
                    yscale = -1
                    yoff = self._ybot[1] + (ridx + 1) * self._row_height
                # add extension edges
                for y, orient, ee_params in ext_edge_infos:
                    tmp_copy = ee_params.copy()
                    if orient == 'R0':
                        x = 0
                        tmp_copy['is_end'] = left_end
                    else:
                        x = xr
                        tmp_copy['is_end'] = right_end
                    if yscale < 0:
                        orient = self._flip_ud(orient)
                    edge_infos.append((x, yscale * y + yoff, orient, tmp_copy))
                # add row edges
                for (y, row_orient, re_params), endl, endr in zip(row_edge_infos, endl_list, endr_list):
                    cur_row_info = re_params['row_info']
                    test_blk_info = tech_cls.get_laygo_blk_info('fg2d', cur_row_info['w_max'], cur_row_info)
                    for x, is_end, flip_lr, end_flag in ((0, left_end, False, endl), (xr, right_end, True, endr)):
                        edge_params = re_params.copy()
                        del edge_params['row_info']
                        edge_params['is_end'] = is_end
                        edge_params['name_id'] = cur_row_info['row_name_id']
                        edge_params['layout_info'] = test_blk_info['layout_info']
                        edge_params['adj_blk_info'] = end_flag
                        if flip_lr:
                            eorient = 'MY' if row_orient == 'R0' else 'R180'
                        else:
                            eorient = row_orient
                        if yscale < 0:
                            eorient = self._flip_ud(eorient)
                        edge_infos.append((x, yscale * y + yoff, eorient, edge_params))

            yt = self.bound_box.top_unit
            gr_vdd_warrs, gr_vss_warrs = tech_cls.draw_boundaries(self, self._laygo_info, num_col, yt,
                                                                  self._bot_end_master, self._top_end_master,
                                                                  edge_infos)

            return bot_warrs, top_warrs, gr_vdd_warrs, gr_vss_warrs

        return [], [], [], []
