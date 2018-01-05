# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Optional

import math
from collections import namedtuple

from bag.math import lcm
from bag.util.search import BinaryIterator
from bag.layout.util import BBox
from bag.layout.routing import WireArray, TrackID
from bag.layout.template import TemplateBase
from bag.layout.routing.fill import fill_symmetric_min_density_info, fill_symmetric_interval

from .core import MOSTech

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig

ExtInfo = namedtuple('ExtInfo', ['od_margin', 'po_margin', 'mx_margin', 'mtype', 'thres', 'imp_min_w',
                                 'm1_sub_w', 'od_w'])
RowInfo = namedtuple('RowInfo', ['od_x', 'od_y', 'od_type', 'po_y', ])
AdjRowInfo = namedtuple('AdjRowInfo', ['od_y', 'od_type', 'po_y', ])


class MOSTechPlanarGeneric(MOSTech):
    """A generic implementation of MOSTech for planar technologies."""

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        MOSTech.__init__(self, config, tech_info)

    def get_edge_info(self, lch_unit, guard_ring_nf, is_end, **kwargs):
        # type: (int, int, bool, **kwargs) -> Dict[str, Any]

        dnw_margins = self.config['dnw_margins']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        imp_od_encx = mos_constants['imp_od_encx']
        nw_dnw_ovl = mos_constants['nw_dnw_ovl']
        nw_dnw_ext = mos_constants['nw_dnw_ext']
        sd_pitch = mos_constants['sd_pitch']
        edge_margin = mos_constants['edge_margin']
        fg_gr_min = mos_constants['fg_gr_min']
        fg_outer_min = mos_constants['fg_outer_min']
        po_od_extx = mos_constants['po_od_extx']

        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')

        if 0 < guard_ring_nf < fg_gr_min:
            raise ValueError('guard_ring_nf = %d < %d' % (guard_ring_nf, fg_gr_min))
        if is_sub_ring and guard_ring_nf <= 0:
            raise ValueError('guard_ring_nf = %d must be positive in substrate ring' % guard_ring_nf)

        # step 0: figure out implant/OD enclosure and outer edge margin
        outer_margin = edge_margin
        if dnw_mode:
            od_w = (fg_gr_min - 1) * sd_pitch + lch_unit + 2 * po_od_extx
            imp_od_encx = max(imp_od_encx, (nw_dnw_ovl + nw_dnw_ext - od_w) // 2)
            outer_margin = dnw_margins[dnw_mode] - nw_dnw_ext

        # calculate implant left X coordinate distance from right edge
        od_delta = po_od_extx - (sd_pitch - lch_unit) // 2
        imp_delta = od_delta + imp_od_encx

        # compute number of finger needed to have correct implant enclosure
        fg_od_margin = -(-imp_delta // sd_pitch)
        fg_outer = max(fg_od_margin, fg_outer_min)

        if guard_ring_nf == 0:
            fg_gr_sub = 0
            fg_gr_sep = 0
        else:
            if is_sub_ring:
                fg_gr_sep = -(-edge_margin // sd_pitch)
            else:
                fg_gr_sep = fg_outer
            fg_outer = 0
            fg_gr_sub = guard_ring_nf + 2 * fg_od_margin

        return dict(
            edge_num_fg=fg_outer + fg_gr_sub + fg_gr_sep,
            edge_margin=outer_margin if is_end else 0,
            fg_outer=fg_outer,
            fg_gr_sub=fg_gr_sub,
            fg_gr_sep=fg_gr_sep,
            fg_od_margin=fg_od_margin,
        )

    def get_md_min_len(self, lch_unit):
        # type: () -> int
        """Returns minimum drain wire length."""
        mos_conn_layer = self.get_mos_conn_layer()

        mos_constants = self.get_mos_tech_constants(lch_unit)
        w_conn_d = mos_constants['w_conn_d']

        mx_min_len = 0
        for layer_id in range(1, mos_conn_layer + 1):
            layer_type = self.tech_info.layer_id_to_type(layer_id)
            mx_min_len = max(mx_min_len, self.tech_info.get_min_length_unit(layer_type, w_conn_d[layer_id]))

        return mx_min_len

    def get_mos_info(self, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]
        """Get transistor layout information

        Layout placement strategy:

        1. find gate Y coordinates from PO spacing and CO enclosure rule.
        2. find drain and OD coordinates by using gate-drain metal and gate-drain OD
           spacing constraints.
        3. get top PO Y coordinates and wrap up.
        """

        # get transistor constants
        mos_constants = self.get_mos_tech_constants(lch_unit)

        sd_pitch = mos_constants['sd_pitch']
        sp_po = mos_constants['sp_po']
        sp_gd_mx = mos_constants['sp_gd_mx']
        sp_gd_od = mos_constants['sp_gd_od']
        w_conn_g = mos_constants['w_conn_g']
        po_od_exty = mos_constants['po_od_exty']

        # get via constants
        via_g = mos_constants['via_g']
        via_d = mos_constants['via_d']

        ds_dummy = kwargs.get('ds_dummy', False)

        # convert w to resolution units
        layout_unit = self.config['layout_unit']
        res = self.res
        w_unit = int(round(w / layout_unit / res))

        # get minimum metal lengths
        md_min_len = self.get_md_min_len(lch_unit)

        # compute gate location, based on PO-PO spacing
        po_yb = sp_po // 2
        g_co_yb = po_yb + via_g['bot_enc_le'][0]
        g_co_yt = g_co_yb + via_g['dim'][0][1]
        g_co_yc = (g_co_yb + g_co_yt) // 2
        g_m1_yb = g_co_yc - w_conn_g[1] // 2
        g_m1_yt = g_m1_yb + w_conn_g[1]
        g_mx_yt = g_co_yc + via_g['dim'][1][1] + via_g['top_enc_le'][1]
        g_mx_yb = g_mx_yt - md_min_len
        g_y_list = [(g_m1_yb, g_m1_yt), (g_mx_yb, g_mx_yt)]

        # compute drain/source location
        # first, get OD location from sp_gd_od
        od_yb = g_m1_yt + sp_gd_od
        od_yt = od_yb + w_unit
        od_yc = (od_yb + od_yt) // 2
        # get number of vias
        d_v0_h = via_d['dim'][0][1]
        d_v0_sp = via_d['sp'][0]
        d_v0_od_ency = via_d['bot_enc_le'][0]
        d_v0_m1_ency = via_d['top_enc_le'][0]
        d_v0_n = (w_unit - 2 * d_v0_od_ency + d_v0_sp) // (d_v0_h + d_v0_sp)
        d_v0_arrh = d_v0_n * (d_v0_h + d_v0_sp) - d_v0_sp
        # get metal length and bottom metal coordinate
        mx_h = max(md_min_len, d_v0_arrh + 2 * d_v0_m1_ency)
        d_mx_yb = od_yc - mx_h // 2
        # check sp_gd_m1 spec, move everything up if necessary
        delta = sp_gd_mx - (d_mx_yb - g_mx_yt)
        if delta > 0:
            d_mx_yb += delta
            od_yt += delta
            od_yb += delta
            od_yc += delta
        # compute final locations
        d_mx_yt = d_mx_yb + mx_h
        d_y_list = [(od_yb, od_yt), (d_mx_yb, d_mx_yt)]

        # find PO and block top Y coordinate
        po_yt = od_yt + po_od_exty
        blk_yt = po_yt + sp_po // 2
        arr_y = 0, blk_yt

        # compute extension information
        mtype = mos_type, mos_type
        lay_info_list = []
        for imp_name in self.get_mos_layers(mos_type, threshold):
            lay_info_list.append((imp_name, 0, 0, blk_yt))

        ext_top_info = ExtInfo(
            od_margin=blk_yt - od_yt,
            po_margin=blk_yt - po_yt,
            mx_margin=blk_yt - d_mx_yt,
            mtype=mtype,
            thres=threshold,
            imp_min_w=0,
            m1_sub_w=0,
            od_w=od_yt - od_yb,
        )
        ext_bot_info = ExtInfo(
            od_margin=od_yb,
            po_margin=po_yb,
            mx_margin=g_mx_yb,
            mtype=mtype,
            thres=threshold,
            imp_min_w=0,
            m1_sub_w=0,
            od_w=od_yt - od_yb,
        )

        sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
        layout_info = dict(
            blk_type='mos',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=arr_y,
            draw_od=not ds_dummy,
            row_info_list=[RowInfo(od_x=(0, fg),
                                   od_y=(od_yb, od_yt),
                                   od_type=('mos', sub_type),
                                   po_y=(po_yb, po_yt), )],
            lay_info_list=lay_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=[(mos_type, threshold, 0, blk_yt, 0, blk_yt)],
            is_sub_ring=False,
            dnw_mode='',
            # MosConnection parameters
            g_y_list=g_y_list,
            d_y_list=d_y_list,
        )

        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=None,
            right_edge_info=None,
            sd_yc=od_yc,
            g_conn_y=g_y_list[-1],
            d_conn_y=d_y_list[-1],
        )

    def get_valid_extension_widths(self, lch_unit, top_ext_info, bot_ext_info):
        # type: (int, ExtInfo, ExtInfo) -> List[int]
        """Compute a list of valid extension widths.

        The DRC rules that we consider are:

        1. wire line-end space
        # implant/threshold layers minimum width.
        #. max OD space
        #. implant/threshold layers to draw

        Of these rules, only the first three sets the minimum extension width.  However,
        if the maximum extension width with no dummy OD is smaller than 1 minus the minimum
        extension width with dummy OD, then that implies there exists some extension widths
        that need dummy OD but can't draw it.

        so our layout strategy is:

        1. compute minimum extension width from wire line-end/minimum implant width.
        #. Compute the maximum extension width that we don't need to draw dummy OD.
        #. Compute the minimum extension width that we can draw DRC clean dummy OD.
        #. Return the list of valid extension widths
        """

        mos_pitch = self.get_mos_pitch(unit_mode=True)
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sp_od_max = mos_constants['sp_od_max']
        od_w_min = mos_constants['od_fill_w'][0]
        imp_od_ency = mos_constants['imp_od_ency']
        imp_po_ency = mos_constants['imp_po_ency']
        w_conn_d = mos_constants['w_conn_d']
        sp_po = mos_constants['sp_po']
        po_od_exty = mos_constants['po_od_exty']
        po_h_min = mos_constants['po_h_min']

        # determine wire line-end spacing
        mos_conn_layer = self.get_mos_conn_layer()
        mx_spy = 0
        for mx_id in range(1, mos_conn_layer + 1):
            lay_type = self.tech_info.layer_id_to_type(mx_id)
            mx_w = w_conn_d[mx_id]
            mx_spy = max(mx_spy, self.tech_info.get_min_line_end_space_unit(lay_type, mx_w))

        bot_imp_min_w = bot_ext_info.imp_min_w  # type: int
        top_imp_min_w = top_ext_info.imp_min_w  # type: int

        # step 1: get minimum extension width
        # step 1a: consider mx line-end spacing
        mx_margin = top_ext_info.mx_margin + bot_ext_info.mx_margin  # type: int
        min_ext_w = max(0, -(-(mx_spy - mx_margin) // mos_pitch))
        # step 1b: consider minimum implant width
        min_ext_w = max(min_ext_w, -(-(bot_imp_min_w + top_imp_min_w) // mos_pitch))
        # step 1c: consider substrate m1 rectangle spacing
        m1_sub_w_max = max(top_ext_info.m1_sub_w, bot_ext_info.m1_sub_w)
        if m1_sub_w_max > 0:
            m1_type = self.tech_info.layer_id_to_type(1)
            m1_sub_spy = self.tech_info.get_min_space_unit(m1_type, m1_sub_w_max)
            min_ext_w = max(min_ext_w, -(-(m1_sub_spy - mx_margin) // mos_pitch))

        # step 2: get maximum extension width without dummy OD
        od_sp_margin = top_ext_info.od_margin + bot_ext_info.od_margin
        max_ext_w_no_od = (sp_od_max - od_sp_margin) // mos_pitch

        # step 3: find minimum extension width with dummy OD
        # now, the tricky part is that we need to make sure OD can be drawn in such a way
        # that we can satisfy both minimum implant width constraint and implant-OD/PO enclosure
        # constraint.  Currently, we compute minimum size so we can split implant either above
        # or below OD and they'll both be DRC clean.  This is a little sub-optimal, but
        # makes layout algorithm much much easier.

        # get od_bot_margin from PO spacing and minimum implant width spec.
        po_h = max(po_h_min, od_w_min + 2 * po_od_exty)
        po_od_exty = (po_h - od_w_min) // 2
        dum_po_yb = -bot_ext_info.po_margin + sp_po
        od_bot_margin = max(dum_po_yb + po_od_exty, bot_imp_min_w + imp_od_ency,
                            bot_imp_min_w + imp_po_ency + po_od_exty)

        # get od_top_margin assuming yt = 0
        dum_po_yt = top_ext_info.po_margin - sp_po
        od_top_margin = -min(dum_po_yt - po_od_exty, -top_imp_min_w - imp_od_ency,
                             -top_imp_min_w - imp_po_ency - po_od_exty)

        # get minimum extension width from OD related spacing rules
        min_ext_w_od = -(-(od_w_min + od_bot_margin + od_top_margin) // mos_pitch)

        if min_ext_w_od <= max_ext_w_no_od + 1:
            # we can transition from no-dummy to dummy seamlessly
            return [min_ext_w]
        else:
            # there exists extension widths such that we need dummies but cannot draw it
            width_list = list(range(min_ext_w, max_ext_w_no_od + 1))
            width_list.append(min_ext_w_od)
            return width_list

    def _get_dummy_po_y_list(self, lch_unit, bot_ext_info, top_ext_info, yblk):
        mos_constants = self.get_mos_tech_constants(lch_unit)
        od_w_min, od_w_max = mos_constants['od_fill_w']
        sp_od_max = mos_constants['sp_od_max']
        od_min_density = mos_constants['od_min_density']

        po_h_min = mos_constants['po_h_min']
        po_od_exty = mos_constants['po_od_exty']
        sp_po = mos_constants['sp_po']

        # step 1A: compute PO/OD bounds.
        bot_od_yt = -bot_ext_info.od_margin
        bot_po_yt = -bot_ext_info.po_margin
        top_od_yb = yblk + top_ext_info.od_margin
        top_po_yb = yblk + top_ext_info.po_margin
        # step 1B: calculate PO area bounds.  Include OD bounds because substrate rows don't have PO
        po_area_offset = max(bot_od_yt, bot_po_yt)
        po_area_tot = min(top_po_yb, top_od_yb) - po_area_offset
        # step 1B: compute target OD area needed from density rules
        od_area_tot = top_od_yb - bot_od_yt + (top_ext_info.od_w + bot_ext_info.od_w) // 2
        od_area_targ = int(math.ceil(od_area_tot * od_min_density))
        # step 1C: binary search on target PO area to achieve target OD area
        po_area_min = min(po_area_tot, od_area_targ + 2 * po_od_exty)
        po_area_iter = BinaryIterator(po_area_min, po_area_tot + 1)
        n_min_po = max(po_h_min, od_w_min + 2 * po_od_exty)
        n_max_po = od_w_max + 2 * po_od_exty
        sp_po_max = sp_od_max - 2 * po_od_exty
        while po_area_iter.has_next():
            po_area_targ = po_area_iter.get_next()
            fill_info = fill_symmetric_min_density_info(po_area_tot, po_area_targ, n_min_po, n_max_po, sp_po,
                                                        sp_max=sp_po_max, fill_on_edge=False, cyclic=False)
            po_area_cur, nfill = fill_info[0][:2]
            od_area_cur = po_area_cur - nfill * 2 * po_od_exty
            if od_area_cur >= od_area_targ:
                po_area_iter.save_info(fill_info)
                po_area_iter.down()
            else:
                if po_area_cur < po_area_targ or po_area_targ == po_area_tot:
                    # we cannot do any better by increasing po_area_targ
                    po_area_iter.save_info(fill_info)
                    break
                else:
                    po_area_iter.up()
        fill_info = po_area_iter.get_last_save_info()
        return fill_symmetric_interval(*fill_info[0][2], offset=po_area_offset, invert=fill_info[1])[0]

    def get_ext_info(self, lch_unit, w, fg, top_ext_info, bot_ext_info):
        # type: (int, int, int, ExtInfo, ExtInfo) -> Dict[str, Any]
        """Draw extension block.

        extension block has zero or more rows of dummy transistors, we specify
        the maximum OD spacing to meet OD density rule.  Most layout
        is straight-forward, but getting the implant right is very tricky.

        Extension implant strategy:

        constraints are:
        1. we cannot have checker-board pattern PP/NP.
        2. PP/NP has minimum width constraint.
        3. OD cannot intersect multiple types of implant.

        To solve these constraints, we use the following strategy (note that in
        LaygoBase, a transistor row can have both transistor or substrate):

        cases:
        1. top and bottom are same flavor transistor / sub (e.g. nch + nch or nch + ptap).
           split at middle, draw more dummy OD on substrate side.
        2. top and bottom are same flavor sub.
           split at middle.  The split point is chosen based on threshold alphabetical
           comparison, so we make sure we consistently favor one threshold over another.
        3. top and bottom are same flavor transistor.
           split at middle.  If there's OD, we force to use transistor implant.
           This avoid constraint 3.
        4. top and bottom row are different flavor sub.
           split at middle, draw more dummy OD on ptap side.
        5. top and bottom are different flavor, transistor and sub.
           we use transistor implant
        6. top and bottom are different transistor.
           split, force to use transistor implant to avoid constraint 1.
        """

        mos_pitch = self.get_mos_pitch(unit_mode=True)
        mos_constants = self.get_mos_tech_constants(lch_unit)
        imp_od_ency = mos_constants['imp_od_ency']
        imp_po_ency = mos_constants['imp_po_ency']
        imp_layers_info_struct = mos_constants['imp_layers']
        thres_layers_info_struct = mos_constants['thres_layers']
        sd_pitch = mos_constants['sd_pitch']
        po_od_exty = mos_constants['po_od_exty']

        yt = w * mos_pitch
        yc = yt // 2

        if w == 0:
            # empty extension
            top_mtype, top_row_type = top_ext_info.mtype
            bot_mtype, bot_row_type = bot_ext_info.mtype
            bot_sub = 'ptap' if (bot_row_type == 'nch' or bot_row_type == 'ptap') else 'ntap'
            top_sub = 'ptap' if (top_row_type == 'nch' or top_row_type == 'ptap') else 'ntap'
            layout_info = dict(
                blk_type='ext',
                lch_unit=lch_unit,
                sd_pitch=sd_pitch,
                fg=fg,
                arr_y=(0, 0),
                draw_od=True,
                row_info_list=[],
                lay_info_list=[],
                # edge parameters
                sub_type=bot_sub if bot_sub == top_sub else None,
                imp_params=None,
                is_sub_ring=False,
                dnw_mode='',
            )

            return dict(
                layout_info=layout_info,
                left_edge_info=None,
                right_edge_info=None,
            )

        po_y_list = self._get_dummy_po_y_list(lch_unit, bot_ext_info, top_ext_info, yt)

        lay_info_list = []
        if not po_y_list:
            # no dummy OD
            num_dod = 0
            od_y_list = []

            thres_split_y = imp_split_y = (yc, yc)
        else:
            # has dummy OD
            # get PO Y coordinates
            num_dod = len(po_y_list)
            od_y_list = []
            for po_yb, po_yt in po_y_list:
                od_yb = po_yb + po_od_exty
                od_yt = po_yt - po_od_exty
                od_y_list.append((od_yb, od_yt))

            # compute implant split Y coordinates
            if num_dod % 2 == 0:
                # we can split exactly in middle
                thres_split_y = imp_split_y = (yc, yc)
            else:
                mid_od_idx = num_dod // 2
                od_yb, od_yt = od_y_list[mid_od_idx]
                if num_dod > 1:
                    od_ytb = od_y_list[mid_od_idx - 1][1]
                    od_ybt = od_y_list[mid_od_idx + 1][0]
                    thres_split_y = imp_split_y = ((od_ytb + od_yb) // 2, (od_yt + od_ybt) // 2)
                else:
                    if imp_po_ency is None:
                        thres_split_y = imp_split_y = (od_yb - imp_od_ency, od_yt + imp_od_ency)
                    else:
                        po_yb, po_yt = po_y_list[mid_od_idx]
                        thres_split_y = imp_split_y = (min(od_yb - imp_od_ency, po_yb - imp_po_ency),
                                                       max(od_yt + imp_od_ency, po_yt + imp_po_ency))

        # compute implant and threshold layer information
        top_mtype, top_row_type = top_ext_info.mtype
        top_thres = top_ext_info.thres
        bot_mtype, bot_row_type = bot_ext_info.mtype
        bot_thres = bot_ext_info.thres
        bot_imp = 'nch' if (bot_row_type == 'nch' or bot_row_type == 'ptap') else 'pch'
        top_imp = 'nch' if (top_row_type == 'nch' or top_row_type == 'ptap') else 'pch'
        bot_tran = (bot_row_type == 'nch' or bot_row_type == 'pch')
        top_tran = (top_row_type == 'nch' or top_row_type == 'pch')
        # figure out where to separate top/bottom implant/threshold.
        if bot_imp == top_imp:
            sub_type = 'ptap' if bot_imp == 'nch' else 'ntap'
            if bot_tran != top_tran:
                # case 1
                sep_idx = 0 if bot_tran else 1
            elif bot_tran:
                # case 3
                sep_idx = 0 if bot_thres <= top_thres else 1
                if num_dod > 0:
                    bot_mtype = top_mtype = bot_imp
            else:
                # case 2
                sep_idx = 0 if bot_thres <= top_thres else 1
        else:
            sub_type = None
            if bot_tran != top_tran:
                # case 5
                if bot_tran:
                    top_mtype = bot_imp
                    top_thres = bot_thres
                    sep_idx = 1
                else:
                    bot_mtype = top_imp
                    bot_thres = top_thres
                    sep_idx = 0
            elif bot_tran:
                # case 6
                bot_mtype = bot_imp
                top_mtype = top_imp
                sep_idx = 1 if bot_imp == 'nch' else 0
            else:
                # case 4
                sep_idx = 1 if bot_imp == 'nch' else 0

        # add implant layers
        imp_ysep = imp_split_y[sep_idx]
        thres_ysep = thres_split_y[sep_idx]
        imp_params = [(bot_mtype, bot_thres, 0, imp_ysep, 0, thres_ysep),
                      (top_mtype, top_thres, imp_ysep, yt, thres_ysep, yt)]

        for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
            imp_layers_info = imp_layers_info_struct[mtype]
            thres_layers_info = thres_layers_info_struct[mtype][thres]
            for cur_yb, cur_yt, lay_info in [(imp_yb, imp_yt, imp_layers_info),
                                             (thres_yb, thres_yt, thres_layers_info)]:

                for lay_name in lay_info:
                    lay_info_list.append((lay_name, 0, cur_yb, cur_yt))

        # construct row_info_list, now we know where the implant splits
        row_info_list = []
        for od_y, po_y in zip(od_y_list, po_y_list):
            cur_mtype = bot_mtype if max(od_y[0], od_y[1]) < imp_ysep else top_mtype
            cur_sub_type = 'ptap' if cur_mtype == 'nch' or cur_mtype == 'ptap' else 'ntap'
            row_info_list.append(RowInfo(od_x=(0, fg), od_y=od_y, od_type=('dum', cur_sub_type), po_y=po_y))

        # create layout information dictionary
        between_gr = (top_row_type == 'ntap' and bot_row_type == 'ptap') or \
                     (top_row_type == 'ptap' and bot_row_type == 'ntap')
        layout_info = dict(
            blk_type='ext',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, yt),
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=lay_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=imp_params,
            is_sub_ring=False,
            between_gr=between_gr,
            dnw_mode='',
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
        )

    def get_sub_ring_ext_info(self, sub_type, height, fg, end_ext_info, **kwargs):
        # type: (str, int, int, ExtInfo, **kwargs) -> Dict[str, Any]
        lch = self.get_substrate_ring_lch()
        lch_unit = int(round(lch / self.config['layout_unit'] / self.res))

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        po_od_exty = mos_constants['po_od_exty']

        dnw_mode = kwargs.get('dnw_mode', '')

        po_y_list = self._get_dummy_po_y_list(lch_unit, end_ext_info, end_ext_info, height)

        if not po_y_list:
            # no dummy OD
            od_y_list = []
        else:
            # has dummy OD
            # get PO Y coordinates
            od_y_list = []
            for po_yb, po_yt in po_y_list:
                od_yb = po_yb + po_od_exty
                od_yt = po_yt - po_od_exty
                od_y_list.append((od_yb, od_yt))

        # construct row_info_list
        row_info_list = []
        for od_y, po_y in zip(od_y_list, po_y_list):
            row_info_list.append(RowInfo(od_x=(0, 0), od_y=od_y, od_type=('dum', sub_type), po_y=po_y))

        layout_info = dict(
            blk_type='ext_subring',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, height),
            draw_od=False,
            row_info_list=row_info_list,
            lay_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=[(sub_type, end_ext_info.thres, 0, height, 0, height), ],
            is_sub_ring=True,
            dnw_mode=dnw_mode,
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
        )

    def get_substrate_info(self, lch_unit, w, sub_type, threshold, fg, blk_pitch=1, **kwargs):
        # type: (int, float, str, str, int, int, **kwargs) -> Dict[str, Any]
        """Get substrate layout information.

        Strategy:

        1. Find bottom OD coordinates from spacing rules.
        #. Find template top coordinate by enforcing symmetry around OD center.
        #. Round up template height to blk_pitch, then recenter OD.
        #. make sure M1 are centered on OD.
        """

        mos_pitch = self.get_mos_pitch(unit_mode=True)
        md_min_len = self.get_md_min_len(lch_unit)
        mos_constants = self.get_mos_tech_constants(lch_unit)
        imp_od_ency = mos_constants['imp_od_ency']
        via_d = mos_constants['via_d']
        nw_dnw_ovl = mos_constants['nw_dnw_ovl']
        nw_dnw_ext = mos_constants['nw_dnw_ext']
        dnw_layers = mos_constants['dnw_layers']
        sd_pitch = mos_constants['sd_pitch']
        sub_m1_enc_le = mos_constants['sub_m1_enc_le']
        sub_m1_extx = mos_constants['sub_m1_extx']

        layout_unit = self.config['layout_unit']
        res = self.res
        od_h = int(round(w / layout_unit / (2 * res))) * 2

        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')
        guard_ring_nf = kwargs.get('guard_ring_nf', 0)

        # step 0: figure out implant/OD enclosure
        if dnw_mode:
            imp_od_ency = max(imp_od_ency, (nw_dnw_ovl + nw_dnw_ext - od_h) // 2)

        # step 1: find OD coordinate
        od_yb = imp_od_ency
        od_yt = od_yb + od_h
        blk_yt = od_yt + imp_od_ency
        # fix substrate height quantization, then recenter OD location
        blk_pitch = lcm([blk_pitch, mos_pitch])
        blk_yt = -(-blk_yt // blk_pitch) * blk_pitch
        od_yb = (blk_yt - od_h) // 2
        od_yt = od_yb + od_h
        od_yc = (od_yb + od_yt) // 2

        # step 2: find metal height
        d_v0_h = via_d['dim'][0][1]
        d_v0_sp = via_d['sp'][0]
        d_v0_od_ency = via_d['bot_enc_le'][0]
        d_v0_n = (od_h - 2 * d_v0_od_ency + d_v0_sp) // (d_v0_h + d_v0_sp)
        d_v0_arrh = d_v0_n * (d_v0_h + d_v0_sp) - d_v0_sp
        mx_h = max(md_min_len, d_v0_arrh + 2 * sub_m1_enc_le)
        d_mx_yb = od_yc - mx_h // 2
        d_mx_yt = d_mx_yb + mx_h

        # step 5: compute layout information
        lay_info_list = []
        for imp_name in self.get_mos_layers(sub_type, threshold):
            lay_info_list.append((imp_name, 0, 0, blk_yt))
        if dnw_mode:
            for lay in dnw_layers:
                lay_info_list.append((lay, 0, blk_yt - nw_dnw_ovl, blk_yt))

        mtype = (sub_type, sub_type)
        m1_sub_w = d_mx_yt - d_mx_yb
        if guard_ring_nf > 0:
            m1_sub_w = max(m1_sub_w, guard_ring_nf * sd_pitch + 2 * sub_m1_extx)
        ext_top_info = ExtInfo(
            od_margin=blk_yt - od_yt,
            po_margin=blk_yt,
            mx_margin=blk_yt - d_mx_yt,
            mtype=mtype,
            thres=threshold,
            imp_min_w=0,
            m1_sub_w=m1_sub_w,
            od_w=od_yt - od_yb,
        )
        ext_bot_info = ExtInfo(
            od_margin=od_yb,
            po_margin=blk_yt,
            mx_margin=d_mx_yb,
            mtype=mtype,
            thres=threshold,
            imp_min_w=0,
            m1_sub_w=m1_sub_w,
            od_w=od_yt - od_yb,
        )

        layout_info = dict(
            blk_type='sub',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, blk_yt),
            draw_od=True,
            row_info_list=[RowInfo(od_x=(0, fg),
                                   od_y=(od_yb, od_yt),
                                   od_type=('sub', sub_type),
                                   po_y=(0, 0), )],
            lay_info_list=lay_info_list,
            # substrate connection parameters
            sub_fg=(0, fg),
            sub_y_list=[(od_yb, od_yt), (d_mx_yb, d_mx_yt), (d_mx_yb, d_mx_yt)],
            # edge parameters
            sub_type=sub_type,
            imp_params=None,
            is_sub_ring=is_sub_ring,
            dnw_mode=dnw_mode,
        )

        return dict(
            layout_info=layout_info,
            sd_yc=od_yc,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=None,
            right_edge_info=None,
            blk_height=blk_yt,
            gb_conn_y=(d_mx_yb, d_mx_yt),
            ds_conn_y=(d_mx_yb, d_mx_yt),
        )

    def get_analog_end_info(self, lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        """Just draw nothing, but compute height so edge margin is met."""

        dnw_margins = self.config['dnw_margins']

        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')

        mos_pitch = self.get_mos_pitch(unit_mode=True)
        mos_constants = self.get_mos_tech_constants(lch_unit)
        nw_dnw_ext = mos_constants['nw_dnw_ext']
        sd_pitch = mos_constants['sd_pitch']
        edge_margin = mos_constants['edge_margin']
        if dnw_mode:
            edge_margin = dnw_margins[dnw_mode] - nw_dnw_ext

        if is_end:
            # step 1: figure out Y coordinates of CPO
            blk_pitch = lcm([blk_pitch, mos_pitch])
            # first assume top Y coordinate is 0
            arr_yt = -(-edge_margin // blk_pitch) * blk_pitch
        else:
            # allow substrate row abutment
            arr_yt = 0

        layout_info = dict(
            blk_type='end',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, arr_yt),
            draw_od=True,
            row_info_list=[],
            lay_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=None,
            is_sub_ring=is_sub_ring,
            dnw_mode=dnw_mode,
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
        )

    def get_sub_ring_end_info(self, sub_type, threshold, fg, end_ext_info, **kwargs):
        # type: (str, str, int, ExtInfo, **kwargs) -> Dict[str, Any]
        """Empty block, just reserve space for margin."""

        mos_pitch = self.get_mos_pitch(unit_mode=True)
        lch = self.get_substrate_ring_lch()
        lch_unit = int(round(lch / self.config['layout_unit'] / self.res))

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        edge_margin = mos_constants['edge_margin']

        dnw_mode = kwargs.get('dnw_mode', '')

        arr_yt = -(-edge_margin // mos_pitch) * mos_pitch

        ext_info = ExtInfo(
            od_margin=arr_yt + end_ext_info.od_margin,
            po_margin=arr_yt + end_ext_info.po_margin,
            mx_margin=arr_yt + end_ext_info.mx_margin,
            mtype=end_ext_info.mtype,
            thres=threshold,
            imp_min_w=0,
            m1_sub_w=0,
            od_w=0,
        )

        layout_info = dict(
            blk_type='end_subring',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, arr_yt),
            draw_od=True,
            row_info_list=[],
            lay_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=[(sub_type, threshold, 0, arr_yt, 0, arr_yt), ],
            is_sub_ring=True,
            dnw_mode=dnw_mode,
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
            ext_info=ext_info,
        )

    def get_outer_edge_info(self, guard_ring_nf, layout_info, is_end, adj_blk_info):
        # type: (int, Dict[str, Any], bool, Optional[Any]) -> Dict[str, Any]
        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        arr_y = layout_info['arr_y']
        row_info_list = layout_info['row_info_list']
        blk_type = layout_info['blk_type']
        dnw_mode = layout_info['dnw_mode']

        edge_info = self.get_edge_info(lch_unit, guard_ring_nf, is_end, dnw_mode=dnw_mode)
        fg_outer = edge_info['fg_outer']

        # compute new lay_info_list
        lay_info_list = layout_info['lay_info_list']
        if guard_ring_nf == 0:
            # keep all implant layers
            new_lay_list = lay_info_list
        else:
            # in guard ring mode, no outer edge block
            new_lay_list = []

        new_row_list = []
        if blk_type != 'sub' and guard_ring_nf == 0:
            # add dummy PO
            for row_info in row_info_list:
                new_row_list.append(RowInfo(od_x=(0, 0), od_y=(0, 0), od_type=row_info.od_type, po_y=row_info.po_y))

        return dict(
            blk_type='edge' if guard_ring_nf == 0 else 'gr_edge',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg_outer,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=new_row_list,
            lay_info_list=new_lay_list,
        )

    def get_gr_sub_info(self, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]

        imp_layers_info_struct = self.mos_config['imp_layers']
        thres_layers_info_struct = self.mos_config['thres_layers']
        dnw_layers = self.mos_config['dnw_layers']
        nw_dnw_ovl = self.mos_config['nw_dnw_ovl']

        sd_pitch = layout_info['sd_pitch']
        blk_type = layout_info['blk_type']
        sub_type = layout_info['sub_type']
        lch_unit = layout_info['lch_unit']
        arr_y = layout_info['arr_y']
        lay_info_list = layout_info['lay_info_list']
        imp_params = layout_info['imp_params']
        dnw_mode = layout_info['dnw_mode']
        between_gr = layout_info.get('between_gr', False)

        edge_info = self.get_edge_info(lch_unit, guard_ring_nf, True, dnw_mode=dnw_mode)
        fg_gr_sub = edge_info['fg_gr_sub']
        fg_od_margin = edge_info['fg_od_margin']

        # compute new row_info_list
        if blk_type == 'sub':
            sub_y_list = layout_info['sub_y_list']
            od_y = sub_y_list[0]
            mx_y = sub_y_list[1]
            gr_type = 'gr_sub_sub'
            od_x_main = (fg_od_margin, fg_gr_sub)
        elif blk_type == 'end' or between_gr:
            od_y = mx_y = (arr_y[1], arr_y[1])
            gr_type = 'gr_sub'
            od_x_main = (0, 0)
        else:
            od_y = mx_y = arr_y
            gr_type = 'gr_sub'
            od_x_main = (fg_od_margin, fg_od_margin + guard_ring_nf)

        sub_y_list = [od_y, mx_y, (mx_y[0], arr_y[1])]
        row_info_list = [RowInfo(od_x=od_x_main,
                                 od_y=od_y,
                                 od_type=('sub', sub_type),
                                 po_y=(0, 0))]
        if blk_type == 'sub':
            row_info_list.append(RowInfo(od_x=(fg_od_margin, fg_od_margin + guard_ring_nf), od_y=(od_y[1], arr_y[1]),
                                         od_type=('sub', sub_type), po_y=(0, 0)))

        # compute new lay_info_list
        wblk = fg_gr_sub * sd_pitch
        if imp_params is None:
            # copy implant layers, but update left coordinate of DNW layers
            new_lay_list = []
            for lay_name, xl, yb, yt in lay_info_list:
                if lay_name in dnw_layers:
                    new_lay_list.append((lay_name, wblk - nw_dnw_ovl, yb, yt))
                else:
                    new_lay_list.append((lay_name, xl, yb, yt))
        else:
            # we need to convert implant layers to substrate implants
            # compute substrate implant layers
            new_lay_list = []
            for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
                sub_type = 'ptap' if mtype == 'nch' or mtype == 'ptap' else 'ntap'
                imp_layers_info = imp_layers_info_struct[sub_type]
                thres_layers_info = thres_layers_info_struct[sub_type][thres]
                for cur_yb, cur_yt, lay_info in [(imp_yb, imp_yt, imp_layers_info),
                                                 (thres_yb, thres_yt, thres_layers_info)]:

                    for lay_name in lay_info:
                        new_lay_list.append((lay_name, 0, cur_yb, cur_yt))
            if dnw_mode:
                # add DNW layers
                # NOTE: since substrate has imp_params = None, if we're here we know that we're not
                # next to substrate, so DNW should span the entire height of this template
                for lay_name in dnw_layers:
                    new_lay_list.append((lay_name, wblk - nw_dnw_ovl, 0, arr_y[1]))

        return dict(
            blk_type=gr_type,
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg_gr_sub,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=new_lay_list,
            # substrate connection parameters
            sub_type=sub_type,
            sub_y_list=sub_y_list,
            sub_fg=(fg_od_margin, fg_od_margin + guard_ring_nf),
        )

    def get_gr_sep_info(self, layout_info, adj_blk_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        arr_y = layout_info['arr_y']
        lay_info_list = layout_info['lay_info_list']
        sub_type = layout_info['sub_type']
        blk_type = layout_info['blk_type']
        row_info_list = layout_info['row_info_list']
        is_sub_ring = layout_info['is_sub_ring']
        dnw_mode = layout_info['dnw_mode']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fg_gr_min = mos_constants['fg_gr_min']
        edge_constants = self.get_edge_info(lch_unit, fg_gr_min, True, is_sub_ring=is_sub_ring, dnw_mode=dnw_mode)
        fg_gr_sep = edge_constants['fg_gr_sep']

        new_row_list = []
        od_type = ('sub', sub_type)
        if blk_type == 'sub':
            cur_blk_type = 'gr_sep_sub'
            sub_y_list = layout_info['sub_y_list']
            # add OD that connects substrate to guard ring
            new_row_list.append(RowInfo(od_x=(0, fg_gr_sep), od_y=sub_y_list[0],
                                        od_type=od_type, po_y=(0, 0)))
        else:
            cur_blk_type = 'gr_sep'
            sub_y_list = None
            if is_sub_ring:
                # remove geometries
                lay_info_list = []
            else:
                # add dummy PO
                for row_info in row_info_list:
                    new_row_list.append(RowInfo(od_x=(0, 0), od_y=(0, 0), od_type=od_type, po_y=row_info.po_y))

        return dict(
            blk_type=cur_blk_type,
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg_gr_sep,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=new_row_list,
            lay_info_list=lay_info_list,
            # substrate connection parameters
            sub_type=sub_type,
            sub_y_list=sub_y_list,
        )

    def draw_mos(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor related layout.

        the layout information dictionary should contain the following entries:

        blk_type
            a string describing the type of this block.
        lch_unit
            channel length in resolution units
        sd_pitch
            the source/drain pitch of this template.
        fg
            the width of this template in number of fingers
        arr_y
            array box Y coordinates as two-element integer tuple.
        draw_od
            If False, we will not draw OD in this template.  This is used for
            supporting the ds_dummy option.
        row_info_list
            a list of named tuples for each OD row we need to draw in
            this template.

            a transistor row is defines as a row of OD/PO that either acts
            as an active device or used for guard ring/dummy fill purposes.
            Each named tuple should have the following entries:

            od_x
                OD X interval in finger index.
            od_y
                OD Y coordinates as two-element integer tuple.
            od_type
                two-element string tuple describing the OD type.  First element describes
                the purpose (mos/sub/dummy), the second element describes the substrate
                type associated with this OD (ptap/ntap).
            po_y
                PO Y coordinates as two-element integer tuple.
        lay_info_list
            a list of layers to draw.  Each layer information is a tuple
            of (imp_layer, xl, yb, yt).
        sub_y_list
            optional entry.  A list of substrate Y coordinates used to draw substrate contacts.
        sub_fg
            optional entry. substrate contact X interval in finger index.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the layout in.
        layout_info : Dict[str, Any]
            the layout information dictionary.
        """
        res = template.grid.resolution

        mos_layer_table = self.config['mos_layer_table']
        lay_name_table = self.config['layer_name']

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        arr_yb, arr_yt = layout_info['arr_y']
        draw_od = layout_info['draw_od']
        row_info_list = layout_info['row_info_list']
        lay_info_list = layout_info['lay_info_list']

        sub_y_list = layout_info.get('sub_y_list', None)
        sub_fg = layout_info.get('sub_fg', (0, 0))

        mos_constants = self.get_mos_tech_constants(lch_unit)
        po_od_extx = mos_constants['po_od_extx']

        blk_w = fg * sd_pitch

        # figure out transistor layout settings
        od_lay = mos_layer_table['OD']
        po_lay = mos_layer_table['PO']
        od_dum_lay = mos_layer_table['OD_dummy']
        po_dum_lay = mos_layer_table['PO_dummy']

        po_xc = sd_pitch // 2
        # draw transistor rows
        for row_info in row_info_list:
            # draw OD
            od_type = row_info.od_type[0]
            if od_type == 'dum' or od_type is None:
                od_lay_cur = od_dum_lay
            else:
                od_lay_cur = od_lay

            od_start, od_stop = row_info.od_x
            od_yb, od_yt = row_info.od_y
            po_yb, po_yt = row_info.po_y

            if od_yt > od_yb and draw_od:
                od_xl = po_xc - lch_unit // 2 + od_start * sd_pitch - po_od_extx
                od_xr = po_xc + lch_unit // 2 + (od_stop - 1) * sd_pitch + po_od_extx
                template.add_rect(od_lay_cur, BBox(od_xl, od_yb, od_xr, od_yt, res, unit_mode=True))

            # draw PO
            if po_yt > po_yb:
                for idx in range(fg):
                    po_xl = po_xc + idx * sd_pitch - lch_unit // 2
                    po_xr = po_xl + lch_unit
                    cur_od_type = od_type if od_start <= idx < od_stop else None
                    lay = po_lay if (cur_od_type == 'mos' or cur_od_type == 'mos_fake' or
                                     cur_od_type == 'sub') else po_dum_lay
                    template.add_rect(lay, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True))

        # draw other layers
        for imp_lay, xl, yb, yt in lay_info_list:
            box = BBox(xl, yb, blk_w, yt, res, unit_mode=True)
            if box.is_physical():
                template.add_rect(imp_lay, box)

        # draw M1 in substrate/guard ring blocks
        if blk_type == 'sub' or blk_type == 'gr_sub' or blk_type == 'gr_sub_sub' or blk_type == 'gr_sep_sub':
            sub_m1_extx = mos_constants['sub_m1_extx']

            if blk_type == 'sub' or blk_type == 'gr_sep_sub':
                # M1 spans the whole block
                m1_xl = -sub_m1_extx
                m1_xr = fg * sd_pitch + sub_m1_extx
            else:
                m1_xl = sub_fg[0] * sd_pitch - sub_m1_extx
                sub_fgr = fg if blk_type == 'gr_sub_sub' else sub_fg[1]
                m1_xr = sub_fgr * sd_pitch + sub_m1_extx

            m1_yb, m1_yt = sub_y_list[1]
            m1_name = lay_name_table[1]
            m1_box = BBox(m1_xl, m1_yb, m1_xr, m1_yt, res, unit_mode=True)
            if m1_box.is_physical():
                template.add_rect(m1_name, m1_box)
            if blk_type == 'gr_sub_sub':
                # connect to guard ring on top
                m1_xr = sub_fg[1] * sd_pitch + sub_m1_extx
                m1_box = BBox(m1_xl, m1_yt, m1_xr, arr_yt, res, unit_mode=True)
                if m1_box.is_physical():
                    template.add_rect(m1_name, m1_box)

        # set size and add PR boundary
        arr_box = BBox(0, arr_yb, blk_w, arr_yt, res, unit_mode=True)
        bound_box = arr_box.extend(x=0, y=0, unit_mode=True)
        template.array_box = arr_box
        template.prim_bound_box = bound_box
        if bound_box.is_physical():
            template.add_cell_boundary(bound_box)

    def draw_substrate_connection(self, template, layout_info, port_tracks, dum_tracks, dummy_only,
                                  is_laygo, is_guardring):
        # type: (TemplateBase, Dict[str, Any], List[int], List[int], bool, bool, bool) -> bool
        # note: we just export all tracks, as we can draw wires on all of them

        blk_type = layout_info['blk_type']
        sub_type = layout_info['sub_type']
        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        sub_fg = layout_info['sub_fg']
        sub_y_list = layout_info['sub_y_list']
        mx_yb, mx_yt = sub_y_list[2]

        dum_conn_layer = self.get_dum_conn_layer()
        mos_conn_layer = self.get_mos_conn_layer()

        if mx_yt > mx_yb:
            if sub_type is None:
                raise ValueError('Cannot draw substrate connection if substrate type is unknown')
            port_name = 'VDD' if sub_type == 'ntap' else 'VSS'

            # draw vias
            x0 = sub_fg[0] * sd_pitch
            num_via = sub_fg[1] - sub_fg[0] + 1
            m1_yb, m1_yt = sub_y_list[1]
            od_yb, od_yt = sub_y_list[0]
            top_layer = dum_conn_layer if dummy_only else mos_conn_layer
            if blk_type == 'sub' or blk_type == 'gr_sub_sub':
                # we can use draw_vertical_vias method to draw all vias
                via_drawn = self._draw_vertical_vias(template, lch_unit, x0, num_via, sd_pitch,
                                                     m1_yb, m1_yt, 0, top_layer=top_layer, is_sub=True,
                                                     mbot_yb=od_yb, mbot_yt=od_yt)
            else:
                # we need to make sure substrate vias abut with top and bottom row
                via_drawn = self._draw_vertical_vias(template, lch_unit, x0, num_via, sd_pitch,
                                                     mx_yb, mx_yt, 0, top_layer=top_layer, via_abut=True, is_sub=True,
                                                     mbot_yb=od_yb, mbot_yt=od_yt)

            # add pins if vias are drawn
            if via_drawn:
                dum_warrs = self._get_wire_array(dum_conn_layer, sub_fg[0] - 0.5, num_via, m1_yb, m1_yt)
                template.add_pin(port_name, dum_warrs, show=False)
                if not dummy_only:
                    mos_warrs = self._get_wire_array(mos_conn_layer, sub_fg[0] - 0.5, num_via, m1_yb, m1_yt)
                    template.add_pin(port_name, mos_warrs, show=False)

                return True

        return False

    def draw_mos_connection(self, template, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                            min_ds_cap, is_diff, diode_conn, options):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, Dict[str, Any]) -> None

        # note: ignore min_ds_cap
        if is_diff:
            raise ValueError('Differential connection not supported yet.')

        layout_info = mos_info['layout_info']
        sd_yc = mos_info['sd_yc']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        g_y_list = layout_info['g_y_list']
        d_y_list = layout_info['d_y_list']

        has_od = not options.get('ds_dummy', False)
        stack = options.get('stack', 1)

        if fg % stack != 0:
            raise ValueError('stack = %d must evenly divide fg = %d' % (stack, fg))

        seg = fg // stack
        width = fg * sd_pitch
        wire_pitch = sd_pitch * stack

        # determine gate locations
        if diode_conn:
            gloc = 'd'
        else:
            if ddir == 0 and sdir == 0:
                raise ValueError('drain and source cannot both go down.')
            if ddir == 0:
                gloc = 's'
            elif sdir == 0:
                gloc = 'd'
            else:
                gloc = gate_pref_loc

        # determine gate track IDs
        if gloc == 's':
            if seg > 2:
                tid_list = [a - 0.5 for a in range(2 * stack, fg, 2 * stack)]
            else:
                tid_list = [-0.5, fg - 0.5]
        else:
            tid_list = [a - 0.5 for a in range(stack, fg, 2 * stack)]

        # extend gate M1 if necessary
        m1_xl = m1_xr = None
        if gate_ext_mode % 2 == 1:
            m1_xl = 0
        if gate_ext_mode // 2 == 1:
            m1_xr = width

        # draw gate vias
        _, g_warrs = self._draw_g_vias(template, layout_info, fg, g_y_list, tid_list, sd_pitch // 2,
                                       m1_xl=m1_xl, m1_xr=m1_xr, has_od=has_od, dy=-sd_yc)

        # draw drain/source vias
        num_s = (seg + 2) // 2
        num_d = seg + 1 - num_s
        mx_yb, mx_yt = d_y_list[-1][0] - sd_yc, d_y_list[-1][1] - sd_yc
        od_yb, od_yt = d_y_list[0][0] - sd_yc, d_y_list[0][1] - sd_yc
        self._draw_vertical_vias(template, lch_unit, 0, num_s, wire_pitch * 2, mx_yb, mx_yt, 0,
                                 mbot_yb=od_yb, mbot_yt=od_yt)
        self._draw_vertical_vias(template, lch_unit, wire_pitch, num_d, wire_pitch * 2, mx_yb, mx_yt, 0,
                                 mbot_yb=od_yb, mbot_yt=od_yt)

        # get drain/source wire arrays
        mos_conn_layer = self.get_mos_conn_layer()
        s_warr = template.add_wires(mos_conn_layer, -0.5, mx_yb, mx_yt, num=num_s, pitch=stack * 2, unit_mode=True)
        d_warr = template.add_wires(mos_conn_layer, stack - 0.5, mx_yb, mx_yt,
                                    num=num_d, pitch=stack * 2, unit_mode=True)

        if diode_conn:
            d_warr = WireArray.list_to_warr(template.connect_wires([d_warr, ] + g_warrs))
            template.add_pin('g', d_warr, show=False)
        else:
            template.add_pin('g', WireArray.list_to_warr(g_warrs), show=False)

        template.add_pin('d', d_warr, show=False)
        template.add_pin('s', s_warr, show=False)

    def draw_dum_connection(self, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[int], Dict[str, Any]) -> None

        layout_info = mos_info['layout_info']
        sd_yc = mos_info['sd_yc']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        g_y_list = layout_info['g_y_list']
        d_y_list = layout_info['d_y_list']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        dum_m1_encx = mos_constants['dum_m1_encx']

        width = fg * sd_pitch
        has_od = not options.get('ds_dummy', False)

        xc = sd_pitch
        num_ds_tot = fg - 1
        m1_xl = m1_xr = None
        if edge_mode % 2 == 1:
            xc = 0
            num_ds_tot += 1
            m1_xl = -dum_m1_encx
        if edge_mode // 2 == 1:
            num_ds_tot += 1
            m1_xr = width + dum_m1_encx

        # get dummy ports
        dum_warrs, _ = self._draw_g_vias(template, layout_info, fg, g_y_list, gate_tracks, sd_pitch // 2,
                                         m1_xl=m1_xl, m1_xr=m1_xr, has_od=has_od, dy=-sd_yc,
                                         top_layer=self.get_dum_conn_layer())

        # draw drain/source vias and short to M1
        mx_yb, mx_yt = d_y_list[-1][0] - sd_yc, d_y_list[-1][1] - sd_yc
        m1_yb = g_y_list[0][0] - sd_yc
        od_yb, od_yt = d_y_list[0][0] - sd_yc, d_y_list[0][1] - sd_yc
        self._draw_vertical_vias(template, lch_unit, xc, num_ds_tot, sd_pitch, mx_yb, mx_yt, 0,
                                 top_layer=1, m1_yb=m1_yb, mbot_yb=od_yb, mbot_yt=od_yt)

        # add pins
        template.add_pin('dummy', dum_warrs, show=False)

    def draw_decap_connection(self, template, mos_info, sdir, ddir, gate_ext_mode, export_gate, options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
        raise ValueError('Decap connection is not supported in this technology.')

    def _draw_g_vias(self, template, layout_info, fg, g_y_list, tid_list, po_xc,
                     m1_xl=None, m1_xr=None, has_od=True, dy=0, top_layer=None):

        res = self.res
        mos_layer_table = self.config['mos_layer_table']
        lay_name_table = self.config['layer_name']
        via_id_table = self.config['via_id']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        via_g = mos_constants['via_g']
        w_conn_g = mos_constants['w_conn_g']

        dum_conn_layer = self.get_dum_conn_layer()
        mos_conn_layer = self.get_mos_conn_layer()
        if top_layer is None:
            top_layer = mos_conn_layer

        # draw via to M1
        po_lay = mos_layer_table['PO']
        m1_yb, m1_yt = g_y_list[0]
        m1_name = lay_name_table[1]
        m1_yc = (m1_yb + m1_yt) // 2
        v0_w, v0_h = via_g['dim'][0]
        via_type = via_id_table[(po_lay, m1_name)]
        po_ency = via_g['bot_enc_le'][0]
        m1_encx = via_g['top_enc_le'][0]

        m1_w = w_conn_g[1]
        via_enc1 = (lch_unit - v0_w) // 2
        via_enc2 = (m1_w - v0_h) // 2
        enc1 = [via_enc1, via_enc1, po_ency, po_ency]
        enc2 = [m1_encx, m1_encx, via_enc2, via_enc2]
        if has_od:
            # draw via to PO only if this is not drain/source dummy.
            template.add_via_primitive(via_type, loc=[po_xc, m1_yc + dy], enc1=enc1, enc2=enc2,
                                       cut_width=v0_w, cut_height=v0_h, nx=fg, spx=sd_pitch, unit_mode=True)

        # draw M1 rectangle
        if m1_xl is None:
            m1_xl = po_xc - v0_w // 2 - m1_encx
        if m1_xr is None:
            m1_xr = po_xc + (fg - 1) * sd_pitch + v0_w // 2 + m1_encx
        template.add_rect(m1_name, BBox(m1_xl, m1_yb + dy, m1_xr, m1_yt + dy, res, unit_mode=True))

        m2_w = w_conn_g[2]
        m2_name = lay_name_table[2]
        if tid_list[0] == -0.5 and top_layer == mos_conn_layer:
            # we only get here if we're drawing 2 segments with gate on source side
            # draw horizontal M2 to avoid DRC errors
            via_type = via_id_table[(m1_name, m2_name)]
            v1_w, v1_h = via_g['dim'][1]
            m1_encx = via_g['bot_enc_le'][1]
            m2_encx = via_g['top_enc_le'][1]
            m1_ency = (m1_w - v1_h) // 2
            m2_ency = (m2_w - v1_h) // 2
            enc1 = [m1_encx, m1_encx, m1_ency, m1_ency]
            enc2 = [m2_encx, m2_encx, m2_ency, m2_ency]
            template.add_via_primitive(via_type, loc=[fg * sd_pitch // 2, m1_yc + dy], enc1=enc1, enc2=enc2,
                                       num_cols=fg - 1, sp_cols=sd_pitch - v1_w, cut_width=v1_w,
                                       cut_height=v1_h, unit_mode=True)
            m2_yb = m1_yc + dy - m2_w // 2
            m2_yt = m1_yc + dy + m2_w // 2

            # draw horizontal M2 bar
            v2_w, v2_h = via_g['dim'][2]
            m2_encx = via_g['bot_enc_le'][2]
            m2_xl = -v2_w // 2 - m2_encx
            m2_xr = fg * sd_pitch + v2_w // 2 + m2_encx
            template.add_rect(m2_name, BBox(m2_xl, m2_yb, m2_xr, m2_yt, res, unit_mode=True))
            horiz_layer = 2
            mh_w = m2_w
        else:
            horiz_layer = 1
            mh_w = m1_w

        # draw vertical wires to the top
        dum_warrs, conn_warrs = [], []
        # draw horizontal to vertical vias and bars
        mv_w = w_conn_g[horiz_layer + 1]
        mh_name = lay_name_table[horiz_layer]
        mv_name = lay_name_table[horiz_layer + 1]
        via_type = via_id_table[(mh_name, mv_name)]
        vh_w, vh_h = via_g['dim'][horiz_layer]
        mh_encx = via_g['bot_enc_le'][horiz_layer]
        mv_ency = via_g['top_enc_le'][horiz_layer]
        mh_ency = (mh_w - vh_h) // 2
        mv_encx = (mv_w - vh_w) // 2

        grid = template.grid
        enc1 = [mh_encx, mh_encx, mh_ency, mh_ency]
        enc2 = [mv_encx, mv_encx, mv_ency, mv_ency]
        mx_yb = g_y_list[-1][0] + dy
        mx_yt = g_y_list[-1][1] + dy

        for tid in tid_list:
            xc = grid.track_to_coord(mos_conn_layer, tid, unit_mode=True)
            template.add_via_primitive(via_type, loc=[xc, m1_yc + dy], enc1=enc1, enc2=enc2,
                                       cut_width=vh_w, cut_height=vh_h, unit_mode=True)

            # draw rest of vertical wires to the top
            self._draw_vertical_vias(template, lch_unit, xc, 1, 0, mx_yb, mx_yt, horiz_layer + 1, top_layer=top_layer)

            # collect gate ports
            if horiz_layer != dum_conn_layer:
                warr = template.add_wires(dum_conn_layer, tid, mx_yb, mx_yt, unit_mode=True)
                dum_warrs.append(warr)
            if horiz_layer != mos_conn_layer and mos_conn_layer <= top_layer:
                warr = template.add_wires(mos_conn_layer, tid, mx_yb, mx_yt, unit_mode=True)
                conn_warrs.append(warr)

        return dum_warrs, conn_warrs

    def _draw_vertical_vias(self,
                            template,  # type: TemplateBase
                            lch_unit,  # type: int
                            x0,  # type: int
                            num,  # type: int
                            pitch,  # type: int
                            mx_yb,  # type: int
                            mx_yt,  # type: int
                            start_layer,  # type: int
                            via_abut=False,  # type: bool
                            is_sub=False,  # type: bool
                            top_layer=None,  # type: Optional[int]
                            m1_yb=None,  # type: Optional[int]
                            mbot_yb=None,  # type: Optional[int]
                            mbot_yt=None,  # type: Optional[int]
                            ):
        # type: (...) -> bool

        res = self.res
        via_id_table = self.config['via_id']
        lay_name_table = self.config['layer_name']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        via_d = mos_constants['via_d']
        w_conn_d = mos_constants['w_conn_d']
        sub_m1_enc_le = mos_constants['sub_m1_enc_le']

        mx_yc = (mx_yt + mx_yb) // 2
        via_drawn = True

        if top_layer is None:
            top_layer = self.get_mos_conn_layer()
        if mbot_yb is None:
            mbot_yb = mx_yb
        if mbot_yt is None:
            mbot_yt = mx_yt

        mtop_h = mx_yt - mx_yb
        for bot_lay_id in range(start_layer, top_layer):
            via_enc_le_bot = via_d['bot_enc_le'][bot_lay_id]
            if bot_lay_id == 0:
                mos_layer_table = self.config['mos_layer_table']
                od_lay = mos_layer_table['OD']
                m1_name = lay_name_table[1]
                via_type = via_id_table[(od_lay, m1_name)]
            else:
                via_type = via_id_table[(lay_name_table[bot_lay_id], lay_name_table[bot_lay_id + 1])]

            w_bot = w_conn_d[bot_lay_id]
            w_top = w_conn_d[bot_lay_id + 1]

            via_w, via_h = via_d['dim'][bot_lay_id]
            via_sp = via_d['sp'][bot_lay_id]

            if via_abut:
                # force these vias to have line end of ceil(sp / 2) or more
                via_enc_le_bot = max(via_enc_le_bot, -(-via_sp // 2))
            if is_sub:
                if bot_lay_id == 1:
                    via_enc_le_bot = max(via_enc_le_bot, sub_m1_enc_le)

            mbot_h = mbot_yt - mbot_yb
            num_via = (mbot_h - 2 * via_enc_le_bot + via_sp) // (via_w + via_sp)
            if num_via > 0:
                via_harr = num_via * (via_w + via_sp) - via_sp
                via_enc_le_top = (mtop_h - via_harr) // 2
                via_enc_le_bot = (mbot_h - via_harr) // 2
                via_enc1 = (w_bot - via_w) // 2
                via_enc2 = (w_top - via_w) // 2

                enc1 = [via_enc1, via_enc1, via_enc_le_bot, via_enc_le_bot]
                enc2 = [via_enc2, via_enc2, via_enc_le_top, via_enc_le_top]
                template.add_via_primitive(via_type, loc=[x0, mx_yc], num_rows=num_via, sp_rows=via_sp,
                                           enc1=enc1, enc2=enc2, cut_width=via_w, cut_height=via_h,
                                           nx=num, spx=pitch, unit_mode=True)
                if m1_yb is not None and start_layer == 0:
                    m1_name = lay_name_table[1]
                    m1_xl = x0 - w_top // 2
                    m1_xr = x0 + w_top // 2
                    template.add_rect(m1_name, BBox(m1_xl, m1_yb, m1_xr, mx_yt, res, unit_mode=True),
                                      nx=num, spx=pitch * res)
            else:
                via_drawn = False

            mbot_yb, mbot_yt = mx_yb, mx_yt

        return via_drawn

    def _get_wire_array(self, layer_id, tr0, num, lower, upper, pitch=1):
        tid = TrackID(layer_id, tr0, num=num, pitch=pitch)
        return WireArray(tid, lower * self.res, upper * self.res)
