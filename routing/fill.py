# -*- coding: utf-8 -*-

"""This module defines dummy/power fill related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any, Tuple, List

import numpy as np

from bag.layout.util import BBox
from bag.layout.template import TemplateBase

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

    @classmethod
    def add_fill_blocks(cls,
                        template,  # type: TemplateBase
                        bound_box,  # type: BBox
                        fill_config,  # type: Dict[int, Tuple[int, int, int, int]]
                        bot_layer,  # type: int
                        top_layer,  # type: int
                        ):
        # type: (...) -> List[List[Instance]]
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
                blk_dim = blk_h
                num_tr = tot_h // (cur_pitch * fill_pitch)
                tr_c0 = yb
                spx = sp_le
                spy = sp
                uf_mat_set = uf_mat.transpose()
            else:
                perp_dir = 'x'
                blk_dim = blk_w
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
                for x0, y0, nx, ny in cls.get_fill_mosaics(uf_tot):
                    inst_info_list.append((x0, y0, nx, ny))
                inst_info_list2.append(inst_info_list)

            use_fill_list.append(uf_mat)

        inst_params = dict(
            fill_config=fill_config,
            show_pins=False
        )
        inst_list2 = []
        for idx, inst_info_list in enumerate(inst_info_list2):
            inst_list = []
            inst_params['bot_layer'] = bot_layer + idx
            master = template.new_template(params=inst_params, temp_cls=PowerFill)
            for x0, y0, nx, ny in inst_info_list:
                loc = xl + x0 * blk_w, yb + y0 * blk_h
                inst = template.add_instance(master, loc=loc, nx=nx, ny=ny, spx=blk_w,
                                             spy=blk_h, unit_mode=True)
                inst_list.append(inst)
            inst_list2.append(inst_list)
        return inst_list2

    @classmethod
    def get_fill_mosaics(cls, uf_mat):
        # TODO: use Eppestein's Polygon dissection instead of greedy algorithm
        nx, ny = uf_mat.shape
        idx_mat = np.full((nx, ny, 2), -1)
        for xidx in range(nx):
            for yidx in range(ny):
                if uf_mat[xidx, yidx]:
                    if xidx > 0 and idx_mat[xidx-1, yidx, 1] == yidx:
                        cur_xl = idx_mat[xidx, yidx, 0] = idx_mat[xidx-1, yidx, 0]
                        idx_mat[xidx-1, yidx, :] = -1
                    else:
                        cur_xl = idx_mat[xidx, yidx, 0] = xidx
                    if yidx > 0 and idx_mat[xidx, yidx-1, 0] == cur_xl:
                        cur_yb = idx_mat[xidx, yidx, 1] = idx_mat[xidx, yidx-1, 1]
                        idx_mat[xidx, yidx-1, :] = -1
                        if xidx > 0 and idx_mat[xidx-1, yidx, 1] == cur_yb:
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

        self.add_pin('VDD', vdd_list, show=show_pins)
        self.add_pin('VSS', vss_list, show=show_pins)
