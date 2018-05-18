# -*- coding: utf-8 -*-

"""This module defines dummy/power fill related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any, Tuple, List

import numpy as np

from bag.util.search import BinaryIterator
from bag.layout.util import BBox
from bag.layout.template import TemplateBase

from ..analog_core.base import AnalogBase, AnalogBaseInfo

if TYPE_CHECKING:
    from bag.layout.objects import Instance
    from bag.layout.template import TemplateDB


class PowerFill(TemplateBase):
    """A power fill template.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            fill_config='the fill configuration dictionary.',
            bot_layer='the bottom fill layer.',
            show_pins='True to show pins.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            show_pins=True,
        )

    def get_layout_basename(self):
        bot_lay = self.params['bot_layer']
        return 'power_fill_m%dm%d' % (bot_lay, bot_lay + 1)

    def draw_layout(self):
        # type: () -> None
        fill_config = self.params['fill_config']
        bot_layer = self.params['bot_layer']
        show_pins = self.params['show_pins']

        top_layer = bot_layer + 1
        blk_w, blk_h = self.grid.get_fill_size(top_layer, fill_config, unit_mode=True)
        bnd_box = BBox(0, 0, blk_w, blk_h, self.grid.resolution, unit_mode=True)
        self.set_size_from_bound_box(top_layer, bnd_box)
        self.array_box = bnd_box

        vdd_list, vss_list = None, None
        for lay in range(bot_layer, top_layer + 1):
            fill_width, fill_space, space, space_le = fill_config[lay]
            vdd_list, vss_list = self.do_power_fill(lay, space, space_le, vdd_warrs=vdd_list,
                                                    vss_warrs=vss_list, fill_width=fill_width,
                                                    fill_space=fill_space, unit_mode=True)
            if lay == bot_layer:
                self.add_pin('VDD_b', vdd_list, show=False)
                self.add_pin('VSS_b', vss_list, show=False)

        self.add_pin('VDD', vdd_list, show=show_pins)
        self.add_pin('VSS', vss_list, show=show_pins)

    @classmethod
    def get_fill_orient(cls, orient_mode):
        if orient_mode == 0:
            return 'R0'
        elif orient_mode == 1:
            return 'MY'
        elif orient_mode == 2:
            return 'MX'
        elif orient_mode == 3:
            return 'R180'
        else:
            raise ValueError('Unknown orientation mode: %d' % orient_mode)

    @classmethod
    def add_fill_blocks(cls,
                        template,  # type: TemplateBase
                        bound_box,  # type: BBox
                        fill_config,  # type: Dict[int, Tuple[int, int, int, int]]
                        bot_layer,  # type: int
                        top_layer,  # type: int
                        orient_mode=0,  # type: int
                        ):
        # type: (...) -> List[List[Instance]]
        # TODO: This method does not work when if fill size changes as layer changes.
        # TODO: Fix in the future.

        # number of wire types per fill block
        ntype = 2

        # error checking
        if top_layer <= bot_layer:
            raise ValueError('Must have top_layer > bot_layer.')

        grid = template.grid
        blk_w, blk_h = grid.get_fill_size(top_layer, fill_config, unit_mode=True)

        xl = bound_box.left_unit
        yb = bound_box.bottom_unit
        xr = bound_box.right_unit
        yt = bound_box.top_unit
        if xl % blk_w != 0 or xr % blk_w != 0 or yb % blk_h != 0 or yt % blk_h != 0:
            raise ValueError('%s is not on power fill grid.' % bound_box)

        # figure out where we can draw fill blocks.
        tot_w = xr - xl
        tot_h = yt - yb
        nx = tot_w // blk_w
        ny = tot_h // blk_h
        use_fill_list = []
        shape = (nx, ny)
        inst_info_list2 = []
        for layer in range(bot_layer, top_layer + 1):
            fill_w, fill_sp, sp, sp_le = fill_config[layer]
            cur_dir = grid.get_direction(layer)
            cur_pitch = grid.get_track_pitch(layer, unit_mode=True)

            fill_pitch = fill_w + fill_sp
            is_horiz = cur_dir == 'x'
            uf_mat = np.ones(shape, dtype=bool)
            if is_horiz:
                perp_dir = 'y'
                blk_dim = blk_w
                num_tr = tot_h // (cur_pitch * fill_pitch)
                tr_c0 = yb
                spx = sp_le
                spy = sp
                uf_mat_set = uf_mat.transpose()
            else:
                perp_dir = 'x'
                blk_dim = blk_h
                num_tr = tot_w // (cur_pitch * fill_pitch)
                tr_c0 = xl
                spx = sp
                spy = sp_le
                uf_mat_set = uf_mat

            cur_tr = grid.coord_to_track(layer, tr_c0, unit_mode=True) + fill_pitch / 2
            for idx in range(num_tr):
                blk_idx = idx // ntype
                wl, wu = grid.get_wire_bounds(layer, cur_tr, width=fill_w, unit_mode=True)
                test_box = bound_box.with_interval(perp_dir, wl, wu, unit_mode=True)
                for block_box in template.blockage_iter(layer, test_box, spx=spx, spy=spy):
                    bl, bu = block_box.get_interval(cur_dir, unit_mode=True)
                    nstart = max(bl - tr_c0, 0) // blk_dim
                    nstop = max(bu - tr_c0, 0) // blk_dim
                    uf_mat_set[blk_idx, nstart:nstop + 1] = False

                cur_tr += fill_pitch

            if layer > bot_layer:
                prev_uf_mat = use_fill_list[-1]
                uf_tot = prev_uf_mat & uf_mat
                inst_info_list = []
                for x0, y0, nx, ny in cls._get_fill_mosaics(uf_tot):
                    inst_info_list.append((x0, y0, nx, ny))
                inst_info_list2.append(inst_info_list)

            use_fill_list.append(uf_mat)

        inst_params = dict(
            fill_config=fill_config,
            show_pins=False
        )
        xinc = 0 if (orient_mode & 1 == 0) else 1
        yinc = 0 if (orient_mode & 2 == 0) else 1
        inst_list2 = []
        orient = cls.get_fill_orient(orient_mode)
        for idx, inst_info_list in enumerate(inst_info_list2):
            inst_list = []
            inst_params['bot_layer'] = bot_layer + idx
            master = template.new_template(params=inst_params, temp_cls=PowerFill)
            for x0, y0, nx, ny in inst_info_list:
                loc = xl + (x0 + xinc) * blk_w, yb + (y0 + yinc) * blk_h
                inst = template.add_instance(master, loc=loc, orient=orient, nx=nx, ny=ny,
                                             spx=blk_w, spy=blk_h, unit_mode=True)
                inst_list.append(inst)
            inst_list2.append(inst_list)
        return inst_list2

    @classmethod
    def _get_fill_mosaics(cls, uf_mat):
        # TODO: use Eppestein's Polygon dissection instead of greedy algorithm
        nx, ny = uf_mat.shape
        idx_mat = np.full((nx, ny, 2), -1)
        for xidx in range(nx):
            for yidx in range(ny):
                if uf_mat[xidx, yidx]:
                    if xidx > 0 and idx_mat[xidx - 1, yidx, 1] == yidx:
                        cur_xl = idx_mat[xidx, yidx, 0] = idx_mat[xidx - 1, yidx, 0]
                        idx_mat[xidx - 1, yidx, :] = -1
                    else:
                        cur_xl = idx_mat[xidx, yidx, 0] = xidx
                    if yidx > 0 and idx_mat[xidx, yidx - 1, 0] == cur_xl:
                        cur_yb = idx_mat[xidx, yidx, 1] = idx_mat[xidx, yidx - 1, 1]
                        idx_mat[xidx, yidx - 1, :] = -1
                        if xidx > 0 and idx_mat[xidx - 1, yidx, 1] == cur_yb:
                            idx_mat[xidx, yidx, 0] = idx_mat[xidx - 1, yidx, 0]
                            idx_mat[xidx - 1, yidx, :] = -1
                    else:
                        idx_mat[xidx, yidx, 1] = yidx

        x_list, y_list = np.nonzero(idx_mat[:, :, 0] >= 0)
        for xidx, yidx in zip(x_list, y_list):
            x0, y0 = idx_mat[xidx, yidx, :]
            nx = xidx - x0 + 1
            ny = yidx - y0 + 1
            yield x0, y0, nx, ny


class DecapFillCore(AnalogBase):
    """A decap cell used for power fill

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            wp='PMOS width.',
            wn='NMOS width.',
            thp='PMOS threshold.',
            thn='NMOS threshold.',
            nx='number of horizontal blocks of fill.',
            ny='number of vertical blocks of fill.',
            fill_config='the fill configuration dictionary.',
            top_layer='Top power fill layer',
            sup_width='Supply track width.',
            options='other AnalogBase options',
            show_pins='True to create pin labels.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            sup_width=2,
            options=None,
            show_pins=True,
        )

    def get_layout_basename(self):
        lay = self.params['top_layer']
        nx = self.params['nx']
        ny = self.params['ny']
        return 'decap_fill_core_lay%d_%dx%d' % (lay, nx, ny)

    def draw_layout(self):
        # type: () -> None
        lch = self.params['lch']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        wp = self.params['wp']
        wn = self.params['wn']
        thp = self.params['thp']
        thn = self.params['thn']
        nx = self.params['nx']
        ny = self.params['ny']
        fill_config = self.params['fill_config']
        top_layer = self.params['top_layer']
        sup_width = self.params['sup_width']
        options = self.params['options']
        show_pins = self.params['show_pins']

        if options is None:
            options = {}

        # get power fill size
        w_tot, h_tot = self.grid.get_fill_size(top_layer, fill_config, unit_mode=True)
        w_tot *= nx
        h_tot *= ny
        # get number of fingers
        info = AnalogBaseInfo(self.grid, lch, 0, top_layer=top_layer)
        bin_iter = BinaryIterator(2, None)
        while bin_iter.has_next():
            fg_cur = bin_iter.get_next()
            w_cur = info.get_placement_info(fg_cur).tot_width
            if w_cur < w_tot:
                bin_iter.save()
                bin_iter.up()
            elif w_cur > w_tot:
                bin_iter.down()
            else:
                bin_iter.save()
                break

        fg_tot = bin_iter.get_last_save()
        if fg_tot is None:
            raise ValueError('Decaep cell width exceed fill width.')
        self.draw_base(lch, fg_tot, ptap_w, ntap_w, [wn], [thn], [wp], [thp],
                       ng_tracks=[1], pg_tracks=[1], n_orientations=['MX'],
                       p_orientations=['R0'], top_layer=top_layer, min_height=h_tot,
                       **options)

        if self.bound_box.height_unit > h_tot:
            raise ValueError('Decap cell height exceed fill height.')

        nmos = self.draw_mos_conn('nch', 0, 0, fg_tot, 0, 0)
        pmos = self.draw_mos_conn('pch', 0, 0, fg_tot, 2, 2, gate_pref_loc='s')

        vss_tid = self.make_track_id('pch', 0, 'g', 0)
        vdd_tid = self.make_track_id('nch', 0, 'g', 0)

        self.connect_to_substrate('ptap', nmos['d'])
        self.connect_to_substrate('ntap', pmos['s'])
        vss_g = self.connect_to_tracks([nmos['s'], pmos['g']], vss_tid)
        vdd_g = self.connect_to_tracks([pmos['d'], nmos['g']], vdd_tid)

        vss, vdd = self.fill_dummy(vdd_width=sup_width, vss_width=sup_width)
        vss.append(vss_g)
        vdd.append(vdd_g)
        self.add_pin('VSS', vss, label='VSS:', show=show_pins)
        self.add_pin('VDD', vdd, label='VDD:', show=show_pins)


class DecapFill(TemplateBase):
    """A power fill cell containing decap

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            fill_config='the fill configuration dictionary.',
            decap_params='decap parameters.',
            nx='number of horizontal blocks of fill.',
            ny='number of vertical blocks of fill.',
            top_layer='Top power fill layer',
            show_pins='True to show pins.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            show_pins=True,
        )

    def get_layout_basename(self):
        lay = self.params['top_layer']
        nx = self.params['nx']
        ny = self.params['ny']
        return 'decap_fill_lay%d_%dx%d' % (lay, nx, ny)

    def draw_layout(self):
        # type: () -> None
        fill_config = self.params['fill_config']
        decap_params = self.params['decap_params']
        nx = self.params['nx']
        ny = self.params['ny']
        top_layer = self.params['top_layer']
        show_pins = self.params['show_pins']

        params = decap_params.copy()
        params['nx'] = nx
        params['ny'] = ny
        params['fill_config'] = fill_config
        params['top_layer'] = top_layer

        master_cap = self.new_template(params=params, temp_cls=DecapFillCore)

        w_blk, h_blk = self.grid.get_fill_size(top_layer, fill_config, unit_mode=True)
        w_tot = w_blk * nx
        h_tot = h_blk * ny
        dx = (w_tot - master_cap.bound_box.width_unit) // 2
        cap_inst = self.add_instance(master_cap, 'XCAP', (dx, 0), unit_mode=True)
        hm_layer = master_cap.mos_conn_layer + 1

        if top_layer <= hm_layer:
            raise ValueError('top layer must be at least %d' % (hm_layer + 1))

        # set size
        res = self.grid.resolution
        self.array_box = bnd_box = BBox(0, 0, w_tot, h_tot, res, unit_mode=True)
        self.set_size_from_bound_box(top_layer, bnd_box)
        self.add_cell_boundary(bnd_box)

        # do power fill
        ym_layer = hm_layer + 1
        vdd_list = cap_inst.get_all_port_pins('VDD')
        vss_list = cap_inst.get_all_port_pins('VSS')
        fill_width, fill_space, space, space_le = fill_config[ym_layer]
        vdd_list, vss_list = self.do_power_fill(ym_layer, space, space_le, vdd_warrs=vdd_list,
                                                vss_warrs=vss_list, fill_width=fill_width,
                                                fill_space=fill_space, unit_mode=True)
        if top_layer > ym_layer:
            params = dict(fill_config=fill_config, show_pins=False)
            inst = None
            for bot_layer in range(ym_layer, top_layer):
                params['bot_layer'] = bot_layer
                master = self.new_template(params=params, temp_cls=PowerFill)
                inst = self.add_instance(master, 'X%d' % bot_layer, nx=nx, ny=ny,
                                         spx=w_blk, spy=h_blk, unit_mode=True)
            vdd_list = self.connect_wires(inst.get_all_port_pins('VDD'))
            vss_list = self.connect_wires(inst.get_all_port_pins('VSS'))

        self.add_pin('VDD', vdd_list, show=show_pins)
        self.add_pin('VSS', vss_list, show=show_pins)
