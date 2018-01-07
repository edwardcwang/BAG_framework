# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Optional

import abc
import math
from collections import namedtuple

from bag.math import lcm
from bag.util.search import BinaryIterator
from bag.layout.util import BBox
from bag.layout.routing import WireArray
from bag.layout.routing.fill import fill_symmetric_min_density_info, fill_symmetric_interval, fill_symmetric_const_space
from bag.layout.template import TemplateBase

from .core import MOSTech

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig

ExtInfo = namedtuple('ExtInfo', ['margins', 'od_w', 'imp_min_w', 'mtype', 'thres', 'po_types',
                                 'edgel_info', 'edger_info'])
RowInfo = namedtuple('RowInfo', ['od_x_list', 'od_y', 'od_type', 'po_y', 'md_y'])
AdjRowInfo = namedtuple('AdjRowInfo', ['po_y', 'po_types'])
EdgeInfo = namedtuple('EdgeInfo', ['od_type', 'draw_layers'])
FillInfo = namedtuple('FillInfo', ['layer', 'exc_layer', 'x_intv_list', 'y_intv_list'])


class MOSTechFinfetBase(MOSTech, metaclass=abc.ABCMeta):
    """Base class for implementations of MOSTech in Finfet technologies.

    This class for now handles all DRC rules and drawings related to PO, OD, CPO,
    and MD. The rest needs to be implemented by subclasses.
    """

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        MOSTech.__init__(self, config, tech_info)

    @abc.abstractmethod
    def get_mos_yloc_info(self, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]
        """Computes Y coordinates of various layers in the transistor row.

        The returned dictionary should have the following entries:

        blk :
            a tuple of row bottom/top Y coordinates.
        od :
            a tuple of OD bottom/top Y coordinates.
        md :
            a tuple of MD bottom/top Y coordinates.
        top_margins :
            a dictionary of top extension margins, which is
            (blk_yt - lay_yt) of each layer.
        bot_margins :
            a dictionary of bottom extension margins, which is
            (lay_yb - blk_yb) of each layer.
        fill_info :
            a dictionary from metal layer tuple to tuple of exclusion
            layer name and list of metal fill Y intervals.
        g_y_list :
            a list of gate wire Y intervals on each layer.
        d_y_list :
            a list of drain wire Y intervals on each layer.
        s_y_list :
            a list of source wire Y intervals on each layer.
        """
        return {}

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
        cpo_po_extx = mos_constants['cpo_po_extx']

        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')

        if 0 < guard_ring_nf < fg_gr_min:
            raise ValueError('guard_ring_nf = %d < %d' % (guard_ring_nf, fg_gr_min))
        if is_sub_ring and guard_ring_nf <= 0:
            raise ValueError('guard_ring_nf = %d must be positive in substrate ring' % guard_ring_nf)

        # step 0: figure out implant/OD enclosure and outer edge margin
        outer_margin = edge_margin
        if dnw_mode:
            od_w = (fg_gr_min + 1) * sd_pitch + lch_unit
            imp_od_encx = max(imp_od_encx, (nw_dnw_ovl + nw_dnw_ext - od_w) // 2)
            outer_margin = dnw_margins[dnw_mode] - nw_dnw_ext

        # calculate implant left X coordinate distance from right edge
        od_delta = (sd_pitch + lch_unit) // 2
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

        # compute edge margin and cpo_xl
        if is_end:
            edge_margin = cpo_xl = 0
        else:
            edge_margin = outer_margin
            cpo_xl = outer_margin + (sd_pitch - lch_unit) // 2 - cpo_po_extx

        return dict(
            edge_num_fg=fg_outer + fg_gr_sub + fg_gr_sep,
            edge_margin=edge_margin,
            cpo_xl=cpo_xl,
            fg_outer=fg_outer,
            fg_gr_sub=fg_gr_sub,
            fg_gr_sep=fg_gr_sep,
            fg_od_margin=fg_od_margin,
        )

    def get_mos_info(self, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]

        ds_dummy = kwargs.get('ds_dummy', False)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        md_w = mos_constants['md_w']

        # Compute Y coordinates of various layers
        yloc_info = self.get_mos_yloc_info(lch_unit, w, mos_type, threshold, fg, **kwargs)
        blk_yb, blk_yt = yloc_info['blk']
        od_yloc = yloc_info['od']
        md_yloc = yloc_info['md']
        top_margins = yloc_info['top_margins']
        bot_margins = yloc_info['bot_margins']
        fill_info = yloc_info['fill_info']
        g_y_list = yloc_info['g_y_list']
        d_y_list = yloc_info['d_y_list']
        s_y_list = yloc_info['s_y_list']

        od_yc = (od_yloc[0] + od_yloc[1]) // 2

        # Compute extension information
        lr_edge_info = EdgeInfo(od_type='mos', draw_layers={})

        mtype = (mos_type, mos_type)
        sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
        po_types = (1,) * fg
        ext_top_info = ExtInfo(
            margins=top_margins,
            od_w=w,
            imp_min_w=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )
        ext_bot_info = ExtInfo(
            margins=bot_margins,
            od_w=w,
            imp_min_w=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )

        # Compute layout information
        lay_info_list = [(lay, 0, blk_yb, blk_yt) for lay in self.get_mos_layers(mos_type, threshold)]
        fill_info_list = [FillInfo(layer=layer, exc_layer=info[0], x_intv_list=[],
                                   y_intv_list=info[1]) for layer, info in fill_info.items()]

        layout_info = dict(
            blk_type='mos',
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(blk_yb, blk_yt),
            draw_od=not ds_dummy,
            row_info_list=[RowInfo(od_x_list=[(0, fg)],
                                   od_y=od_yloc,
                                   od_type=('mos', sub_type),
                                   po_y=(blk_yb, blk_yt),
                                   md_y=md_yloc), ],
            lay_info_list=lay_info_list,
            fill_info_list=fill_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=[(mos_type, threshold, blk_yb, blk_yt, blk_yb, blk_yt)],
            is_sub_ring=False,
            dnw_mode='',
            md_w=md_w,
            # MosConnection parameters
            g_y_list=g_y_list,
            d_y_list=d_y_list,
            s_y_list=s_y_list,
            # adjacent block information list
            adj_row_list=[],
            left_blk_info=None,
            right_blk_info=None,
        )

        # step 8: return results
        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=(lr_edge_info, []),
            right_edge_info=(lr_edge_info, []),
            sd_yc=od_yc,
            g_conn_y=g_y_list[-1],
            d_conn_y=d_y_list[-1],
            s_conn_y=s_y_list[-1],
        )

    def get_valid_extension_widths(self, lch_unit, top_ext_info, bot_ext_info):
        # type: (int, ExtInfo, ExtInfo) -> List[int]
        """Compute a list of valid extension widths.

        The DRC rules that we consider are:

        1. wire line-end space
        #. MD space
        # implant/threshold layers minimum width.
        #. CPO space
        #. max OD space
        #. lower metal fill
        #. implant/threshold layers to draw

        Of these rules, only the first three sets the minimum extension width.  However,
        if the maximum extension width with no dummy OD is smaller than 1 minus the minimum
        extension width with dummy OD, then that implies there exists some extension widths
        that need dummy OD but can't draw it.

        so our layout strategy is:

        1. compute minimum extension width from wire line-end/MD spaces/minimum implant width.
        #. Compute the maximum extension width that we don't need to draw dummy OD.
        #. Compute the minimum extension width that we can draw DRC clean dummy OD.
        #. Return the list of valid extension widths
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']  # type: int
        fin_p = mos_constants['mos_pitch']  # type: int
        od_sp_nfin_max = mos_constants['od_sp_nfin_max']
        od_nfin_min = mos_constants['od_fill_w'][0]
        imp_od_ency = mos_constants['imp_od_ency']
        cpo_h = mos_constants['cpo_h']
        cpo_od_sp = mos_constants['cpo_od_sp']
        cpo_spy = mos_constants['cpo_spy']
        md_h_min = mos_constants['md_h_min']
        md_od_exty = mos_constants['md_od_exty']
        md_spy = mos_constants['md_spy']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        bot_imp_min_w = bot_ext_info.imp_min_w  # type: int
        top_imp_min_w = top_ext_info.imp_min_w  # type: int

        # step 1: get minimum extension width from vertical spacing rule
        min_ext_w = max(0, -(-(bot_imp_min_w + top_imp_min_w) // fin_p))
        for name in top_ext_info.margins:
            cur_spy = mos_constants['%s_spy'] % name
            tot_margin = cur_spy - (top_ext_info.margins[name] + bot_ext_info.margins[name])
            min_ext_w = max(min_ext_w, -(-tot_margin // fin_p))

        # step 2: get maximum extension width without dummy OD
        od_space_nfin = (top_ext_info.margins['od'] + bot_ext_info.margins['od'] + fin_h) // fin_p
        max_ext_w_no_od = od_sp_nfin_max - od_space_nfin

        # step 3: find minimum extension width with dummy OD
        # now, the tricky part is that we need to make sure OD can be drawn in such a way
        # that we can satisfy both minimum implant width constraint and implant-OD enclosure
        # constraint.  Currently, we compute minimum size so we can split implant either above
        # or below OD and they'll both be DRC clean.  This is a little sub-optimal, but
        # makes layout algorithm much much easier.

        # get od_yb_max1, round to fin grid.
        dum_md_yb = -bot_ext_info.margins['md'] + md_spy
        od_yb_max1 = max(dum_md_yb + md_od_exty, cpo_h // 2 + cpo_od_sp)
        od_yb_max1 = -(-(od_yb_max1 - fin_p2 + fin_h2) // fin_p)
        # get od_yb_max2, round to fin grid.
        od_yb_max = bot_imp_min_w + imp_od_ency
        od_yb_max = max(od_yb_max1, -(-(od_yb_max - fin_p2 + fin_h2) // fin_p))

        # get od_yt_min1 assuming yt = 0, round to fin grid.
        dum_md_yt = top_ext_info.margins['md'] - md_spy
        od_yt_min1 = min(dum_md_yt - md_od_exty, -(cpo_h // 2) - cpo_od_sp)
        od_yt_min1 = (od_yt_min1 - fin_p2 - fin_h2) // fin_p
        # get od_yt_min2, round to fin grid.
        od_yt_min = -top_imp_min_w - imp_od_ency
        od_yt_min = min(od_yt_min1, (od_yt_min - fin_p2 - fin_h2) // fin_p)

        # get minimum extension width from OD related spacing rules
        min_ext_w_od = max(0, od_nfin_min - (od_yt_min - od_yb_max) - 1) * fin_p
        # check to see CPO spacing rule is satisfied
        min_ext_w_od = max(min_ext_w_od, cpo_spy + cpo_h)
        # check to see MD minimum height rule is satisfied
        min_ext_w_od = max(min_ext_w_od, md_h_min - (dum_md_yt - dum_md_yb))
        # round min_ext_w_od to fin grid.
        min_ext_w_od = -(-min_ext_w_od // fin_p)

        if min_ext_w_od <= max_ext_w_no_od + 1:
            # we can transition from no-dummy to dummy seamlessly
            return [min_ext_w]
        else:
            # there exists extension widths such that we need dummies but cannot draw it
            width_list = list(range(min_ext_w, max_ext_w_no_od + 1))
            width_list.append(min_ext_w_od)
            return width_list

    def _get_dummy_od_yloc(self, lch_unit, bot_ext_info, top_ext_info, yblk):
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_nfin_min, od_nfin_max = mos_constants['od_fill_w']
        od_spy_min = mos_constants['od_spy_min']
        od_spy_max = mos_constants['od_spy_max']
        od_min_density = mos_constants['od_min_density']

        od_yb_offset = (fin_p - fin_h) // 2
        od_yt_offset = od_yb_offset + fin_h
        od_spy_nfin_min = (od_spy_min - (fin_p - fin_h)) // fin_p
        od_spy_nfin_max = (od_spy_max - (fin_p - fin_h)) // fin_p

        # compute MD/OD locations.
        bot_od_yt = -bot_ext_info.margins['od']
        top_od_yb = yblk + top_ext_info.margins['od']
        bot_od_fidx = (bot_od_yt - od_yt_offset) // fin_p
        top_od_fidx = (top_od_yb - od_yb_offset) // fin_p

        # compute OD fill area needed to meet density
        bot_od_w = (bot_ext_info.od_w - 1) * fin_p + fin_h
        top_od_w = (top_ext_info.od_w - 1) * fin_p + fin_h
        od_area_adj = (bot_od_w + top_od_w) // 2
        od_area_tot = top_od_yb - bot_od_yt + od_area_adj
        od_area_targ = int(math.ceil(od_area_tot * od_min_density)) - od_area_adj
        od_fin_area_min = -(-(od_area_targ - fin_h) // fin_p) + 1
        od_fin_area_tot = top_od_fidx - bot_od_fidx - 1
        od_fin_area_iter = BinaryIterator(od_fin_area_min, od_fin_area_tot + 1)

        # binary search on OD fill area
        while od_fin_area_iter.has_next():
            # compute fill with fin-area target
            od_fin_area_targ_cur = od_fin_area_iter.get_next()
            fill_info = fill_symmetric_min_density_info(od_fin_area_tot, od_fin_area_targ_cur,
                                                        od_nfin_min, od_nfin_max, od_spy_nfin_min,
                                                        sp_max=od_spy_nfin_max, fill_on_edge=False,
                                                        cyclic=False)
            od_nfin_tot_cur = fill_info[0][0]

            # compute actual OD area
            od_intv_list = fill_symmetric_interval(*fill_info[0][2], offset=bot_od_fidx + 1, invert=fill_info[1])[0]
            od_area_cur = sum(((start - stop - 1) * fin_p + fin_h for start, stop in od_intv_list))
            if od_area_cur >= od_area_targ:
                od_fin_area_iter.save_info(od_intv_list)
                od_fin_area_iter.down()
            else:
                if od_nfin_tot_cur < od_fin_area_targ_cur or od_fin_area_targ_cur == od_fin_area_tot:
                    # we cannot do any better by increasing od_fin_area_targ
                    od_fin_area_iter.save_info(od_intv_list)
                    break
                else:
                    od_fin_area_iter.up()

        # convert fin interval to Y coordinates
        od_intv_list = od_fin_area_iter.get_last_save_info()
        return [(start * fin_p + od_yb_offset, stop * fin_p + od_yt_offset) for start, stop in od_intv_list]

    def _get_dummy_yloc(self, lch_unit, bot_ext_info, top_ext_info, yblk):
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_nfin_min, od_nfin_max = mos_constants['od_fill_w']
        od_spy_min = mos_constants['od_spy_min']
        md_h_min = mos_constants['md_h_min']
        md_od_exty = mos_constants['md_od_exty']
        md_spy = mos_constants['md_spy']

        od_yb_offset = (fin_p - fin_h) // 2
        od_yt_offset = od_yb_offset + fin_h
        od_h_min = (od_nfin_min - 1) * fin_p + fin_h

        # compute MD/OD locations.
        bot_md_yt = -bot_ext_info.margins['md']
        top_md_yb = yblk + top_ext_info.margins['md']

        # get dummy OD/MD intervals
        od_y_list = self._get_dummy_od_yloc(lch_unit, bot_ext_info, top_ext_info, yblk)
        md_y_list = []
        for od_yb, od_yt in od_y_list:
            md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
            md_yb = (od_yb + od_yt - md_h) // 2
            md_y_list.append((md_yb, md_yb + md_h))

        # check and fix bottom MD spacing violation
        if md_y_list[0][0] < bot_md_yt + md_spy:
            od_yt = od_y_list[0][1]
            od_bot_fidx = -(-(bot_md_yt + md_spy + md_od_exty - od_yb_offset) // fin_p)
            od_yb = od_bot_fidx * fin_p + od_yb_offset
            od_yt = max(od_yb + od_h_min, od_yt)
            od_y_list[0] = od_yb, od_yt
            md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
            md_yb = max((od_yb + od_yt - md_h) // 2, bot_md_yt + md_spy)
            md_y_list[0] = md_yb, md_yb + md_h
        # check and fix MD spacing violation
        if md_y_list[-1][1] > top_md_yb - md_spy:
            od_yb = od_y_list[-1][0]
            od_top_fidx = (top_md_yb - md_spy - md_od_exty - od_yt_offset) // fin_p
            od_yt = od_top_fidx * fin_p + od_yt_offset
            od_yb = min(od_yt - od_h_min, od_yb)
            od_y_list[-1] = od_yb, od_yt
            md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
            md_yt = min((od_yb + od_yt + md_h) // 2, top_md_yb - md_spy)
            md_y_list[0] = md_yt - md_h, md_yt

        if md_y_list[0][0] < bot_md_yt + md_spy:
            # bottom MD spacing rule violated.  This only happens if we have exactly
            # one dummy OD, and there is no solution that works for both top and bottom
            # MD spacing rules.
            raise ValueError('Cannot draw dummy OD and meet MD spacing constraints.  '
                             'See developer.')
        if len(md_y_list) > 1:
            # check inner MD and OD spacing rules are met.
            # I don't think these rules will ever be broken, so I'm not fixing it now.
            # However, if there does need to be a fix, you probably need to recompute
            # inner dummy OD Y coordinates.
            if (md_y_list[0][1] + md_spy > md_y_list[1][0] or
                    md_y_list[-2][1] + md_spy > md_y_list[-1][0]):
                raise ValueError('inner dummy MD spacing rule not met.  See developer.')
            if (od_y_list[0][1] + od_spy_min > od_y_list[1][0] or
                    od_y_list[-2][1] + od_spy_min > od_y_list[1][0]):
                raise ValueError('inner dummy OD spacing rule not met.  See developer.')

        return od_y_list, md_y_list

    def get_ext_info(self, lch_unit, w, fg, top_ext_info, bot_ext_info):
        # type: (int, int, int, ExtInfo, ExtInfo) -> Dict[str, Any]
        """Draw extension block.

        extension block has zero or more rows of dummy transistors, which are
        drawn to meet OD maximum spacing rule.  Most layout is straight-forward,
        but getting the implant right is very tricky.

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
           we use transistor implant.
        6. top and bottom are different transistor.
           split, force to use transistor implant to avoid constraint 1.
        """
        mos_layer_table = self.config['mos_layer_table']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_nfin_min, od_nfin_max = mos_constants['od_fill_w']
        od_sp_nfin_max = mos_constants['od_sp_nfin_max']
        cpo_spy = mos_constants['cpo_spy']
        imp_od_ency = mos_constants['imp_od_ency']
        md_h_min = mos_constants['md_h_min']
        cpo_h = mos_constants['cpo_h']
        md_w = mos_constants['md_w']
        md_od_exty = mos_constants['md_od_exty']
        m1_w = mos_constants['mos_conn_w']
        sd_pitch = mos_constants['sd_pitch']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2
        yt = w * fin_p
        yc = yt // 2

        lr_edge_info = EdgeInfo(od_type='dum', draw_layers={})
        if w == 0:
            # just draw CPO
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
                lay_info_list=[(mos_layer_table['CPO'], 0, -cpo_h // 2, cpo_h // 2)],
                fill_info_list=[],
                # edge parameters
                sub_type=bot_sub if bot_sub == top_sub else None,
                imp_params=None,
                is_sub_ring=False,
                dnw_mode='',
                md_w=md_w,
                # adjacent block information list
                adj_row_list=[],
                left_blk_info=None,
                right_blk_info=None,
            )

            return dict(
                layout_info=layout_info,
                left_edge_info=(lr_edge_info, []),
                right_edge_info=(lr_edge_info, []),
            )

        # step 1: compute OD Y coordinates
        bot_od_yt = -bot_ext_info.od_margin
        bot_od_yt_fin = (bot_od_yt - fin_p2 - fin_h2) // fin_p
        top_od_yb = yt + top_ext_info.od_margin
        top_od_yb_fin = (top_od_yb - fin_p2 + fin_h2) // fin_p
        area = top_od_yb_fin - bot_od_yt_fin
        od_fin_list = fill_symmetric_const_space(area, od_sp_nfin_max, od_nfin_min, od_nfin_max, offset=bot_od_yt_fin)

        # check if we draw one or two CPO.  Compute threshold split Y coordinates accordingly.
        cpo2_w = -(-(cpo_spy + cpo_h) // fin_p)  # type: int
        one_cpo = (w < cpo2_w)

        # calculate fill
        m1_sp_max = self.tech_constants['m1_sp_max']
        m1_fill_lmin = self.tech_constants['m1_fill_lmin']
        m1_fill_lmax = self.tech_constants['m1_fill_lmax']
        area_yb = -bot_ext_info.m1_margin
        area_yt = yt + top_ext_info.m1_margin
        fill_y_list = fill_symmetric_const_space(area_yt - area_yb, m1_sp_max, m1_fill_lmin,
                                                 m1_fill_lmax, offset=area_yb)

        lay_info_list = []
        cpo_lay = ('CutPoly', 'drawing')
        if not od_fin_list:
            # no dummy OD
            od_x_list = []
            od_y_list = md_y_list = [(0, 0)]

            if one_cpo:
                po_y_list = [(0, 0)]
                imp_split_y = (yc, yc)
                # compute adjacent row geometry
                adj_edgel_infos = [bot_ext_info.edgel_info, top_ext_info.edgel_info]
                adj_edger_infos = [bot_ext_info.edger_info, top_ext_info.edger_info]
                adj_row_list = [AdjRowInfo(po_types=bot_ext_info.po_types,
                                           po_y=(0, yc),
                                           ),
                                AdjRowInfo(po_types=top_ext_info.po_types,
                                           po_y=(yc, yt),
                                           )]
                lay_info_list.append((cpo_lay, 0, yc - cpo_h // 2, yc + cpo_h // 2))
            else:
                po_y_list = [(0, yt)]
                imp_split_y = (0, yt)
                adj_row_list = []
                adj_edgel_infos = []
                adj_edger_infos = []
                lay_info_list.append((cpo_lay, 0, -cpo_h // 2, cpo_h // 2))
                lay_info_list.append((cpo_lay, 0, yt - cpo_h // 2, yt + cpo_h // 2))

        else:
            # has dummy OD, compute OD X/Y coordinates
            adj_row_list = []
            adj_edgel_infos = []
            adj_edger_infos = []
            od_x_list = [(0, fg)]
            # add OD and CPO layout information, also calculate fill
            num_dod = len(od_fin_list)
            if num_dod == 1:
                # if we only have 1 dummy, use manual algorithm to make sure we satisfy MD-MD and CPO-MD rules
                bot_cpo_yt = cpo_h // 2
                top_cpo_yb = yt - cpo_h // 2
                bot_md_yt = -bot_ext_info.md_margin
                top_md_yb = yt + top_ext_info.md_margin
                bot_imp_y = bot_ext_info.imp_min_w
                top_imp_y = yt - top_ext_info.imp_min_w
                od_y_list, md_y_list = self._get_ext_dummy_loc(lch_unit, bot_od_yt_fin, top_od_yb_fin,
                                                               bot_cpo_yt, top_cpo_yb, bot_md_yt, top_md_yb,
                                                               bot_imp_y, top_imp_y)
            else:
                # for 2 for more dummy, od_sp_nfin_max (12) and od_nfin_max (20) values guarantee that
                # all dummy locations are DRC clean
                od_y_list = []
                md_y_list = []
                for a, b in od_fin_list:
                    od_yb = fin_p2 - fin_h2 + a * fin_p
                    od_yt = fin_p2 + fin_h2 + b * fin_p
                    od_yc = (od_yb + od_yt) // 2
                    md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
                    md_yb = od_yc - md_h // 2
                    md_yt = od_yc + md_h // 2
                    od_y_list.append((od_yb, od_yt))
                    md_y_list.append((md_yb, md_yt))

            # get PO/CPO locations
            cpo_yc = 0
            cpo_yc_list = []
            po_y_list = []
            for idx, (od_yb, od_yt) in enumerate(od_y_list):
                # find next CPO coordinates
                if idx + 1 < num_dod:
                    next_cpo_yc = (od_yt + od_y_list[idx + 1][0]) // 2
                else:
                    next_cpo_yc = yt

                # record PO Y coordinates
                po_y_list.append((cpo_yc, next_cpo_yc))

                # add CPO
                lay_info_list.append((cpo_lay, 0, cpo_yc - cpo_h // 2, cpo_yc + cpo_h // 2))
                cpo_yc_list.append(cpo_yc)
                # update CPO coordinates.
                cpo_yc = next_cpo_yc

            # add last CPO
            lay_info_list.append((cpo_lay, 0, yt - cpo_h // 2, yt + cpo_h // 2))
            cpo_yc_list.append(yt)

            # compute implant split Y coordinates
            if num_dod % 2 == 0:
                # we can split exactly in middle
                imp_split_y = (yc, yc)
            else:
                # Find the two middle CPO coordinates.
                top_cpo_idx = num_dod // 2
                od_yb, od_yt = od_y_list[top_cpo_idx]
                imp_split_y = (od_yb - imp_od_ency, od_yt + imp_od_ency)

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
            if bot_tran != top_tran:
                # case 1
                sep_idx = 0 if bot_tran else 1
            elif bot_tran:
                # case 3
                sep_idx = 0 if bot_thres <= top_thres else 1
                if od_x_list:
                    bot_mtype = top_mtype = bot_imp
            else:
                # case 2
                sep_idx = 0 if bot_thres <= top_thres else 1
        else:
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
        imp_params = [(bot_mtype, bot_thres, 0, imp_ysep, 0, imp_ysep),
                      (top_mtype, top_thres, imp_ysep, yt, imp_ysep, yt)]

        for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
            for lay in self.get_mos_layers(mtype, thres):
                if lay[0].startswith('VT'):
                    cur_yb, cur_yt = thres_yb, thres_yt
                else:
                    cur_yb, cur_yt = imp_yb, imp_yt
                lay_info_list.append((lay, 0, cur_yb, cur_yt))

        # construct row_info_list, now we know where the implant splits
        row_info_list = []
        for od_y, po_y, md_y in zip(od_y_list, po_y_list, md_y_list):
            cur_mtype = bot_mtype if max(od_y[0], od_y[1]) < imp_ysep else top_mtype
            cur_sub_type = 'ptap' if cur_mtype == 'nch' or cur_mtype == 'ptap' else 'ntap'
            row_info_list.append(RowInfo(od_x_list=od_x_list, od_y=od_y, od_type=('dum', cur_sub_type),
                                         po_y=po_y, md_y=md_y))

        # compute metal 1 fill locations
        fill_x_list = [(idx * sd_pitch - m1_w // 2, idx * sd_pitch + m1_w // 2)
                       for idx in range(0, fg + 1)]
        fill_info = FillInfo(layer=('M1', 'drawing'), exc_layer=None,
                             x_intv_list=fill_x_list, y_intv_list=fill_y_list)
        # create layout information dictionary
        layout_info = dict(
            lch_unit=lch_unit,
            md_w=md_w,
            fg=fg,
            sd_pitch=sd_pitch,
            array_box_xl=0,
            array_box_y=(0, yt),
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=lay_info_list,
            adj_info_list=adj_row_list,
            left_blk_info=None,
            right_blk_info=None,
            fill_info_list=[fill_info],

            # information needed for computing edge block layout
            blk_type='ext',
            imp_params=imp_params,
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=(lr_edge_info, adj_edgel_infos),
            right_edge_info=(lr_edge_info, adj_edger_infos),
        )

    @classmethod
    def get_sub_ring_ext_info(self, sub_type, height, fg, end_ext_info, **kwargs):
        # type: (str, int, int, ExtInfo, **kwargs) -> Dict[str, Any]
        # TODO: add actual implementation.
        raise NotImplementedError('Not implemented yet.')

    @classmethod
    def _get_sub_m1_y(self, lch_unit, od_yc, od_nfin):
        """Get M1 Y coordinates for substrate connection."""
        via_info = self.get_ds_via_info(lch_unit, od_nfin)
        m1_h = via_info['m1_h']

        m1_yb = od_yc - m1_h // 2
        m1_yt = od_yc + m1_h // 2

        return m1_yb, m1_yt

    @classmethod
    def get_substrate_info(self, lch_unit, w, sub_type, threshold, fg, blk_pitch=1, **kwargs):
        # type: (int, int, str, str, int, int, **kwargs) -> Dict[str, Any]
        """Get substrate layout information.

        Layout is quite simple.  We use M0PO to short adjacent S/D together, so dummies can be
        connected using only M2 or below.

        Strategy:

        1. Find bottom M0_PO and bottom OD coordinates from spacing rules.
        #. Find template top coordinate by enforcing symmetry around OD center.
        #. Round up template height to blk_pitch, then recenter OD.
        #. make sure MD/M1 are centered on OD.
        """

        md_od_exty = self.tech_constants['md_od_exty']
        fin_h = self.tech_constants['fin_h']
        fin_p = self.tech_constants['fin_pitch']
        mp_h = self.tech_constants['mp_h']
        mp_cpo_sp = self.tech_constants['mp_cpo_sp']
        mp_md_sp = self.tech_constants['mp_md_sp']
        mp_spy = self.tech_constants['mp_spy']
        cpo_h = self.tech_constants['cpo_h']
        md_h_min = self.tech_constants['md_h_min']
        md_w = self.tech_constants['md_w']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        # figure out od/md height
        od_h = (w - 1) * fin_p + fin_h
        md_h = max(md_h_min, od_h + 2 * md_od_exty)
        md_od_exty = (md_h - od_h) // 2

        # step 1: figure out Y coordinate of bottom CPO
        cpo_bot_yt = cpo_h // 2

        # step 2: find bottom M0_PO coordinate
        mp_yb = max(mp_spy // 2, cpo_bot_yt + mp_cpo_sp)
        mp_yt = mp_yb + mp_h

        # step 3: find OD coordinate
        od_bfin_yc = mp_yt + mp_md_sp + md_od_exty
        od_bfin_yc = -(-(od_bfin_yc - fin_p2) // fin_p) * fin_p + fin_p2
        od_yb = od_bfin_yc - fin_h2
        od_yt = od_yb + od_h
        cpo_top_yc = od_yt + od_yb
        # fix substrate height quantization, then recenter OD location
        blk_pitch = lcm([blk_pitch, fin_p])
        blk_h = -(-cpo_top_yc // blk_pitch) * blk_pitch
        cpo_top_yc = blk_h
        od_yb = blk_h // 2 - od_h // 2
        od_yb = (od_yb - fin_p2 + fin_h2) // fin_p * fin_p + fin_p2 - fin_h2
        od_yt = od_yb + od_h
        od_yc = (od_yb + od_yt) // 2

        # step 3: find MD Y coordinates
        md_yb = (od_yb + od_yt - md_h) // 2
        md_yt = md_yb + md_h

        # step 3.5: update MP Y coordinates, compute M1 upper and lower bound
        # bottom MP
        via_info = self.get_ds_via_info(lch_unit, w)
        gv0_h = via_info['h'][0]
        top_ency = via_info['top_ency'][0]
        gm1_delta = gv0_h // 2 + top_ency
        mp_yt = md_yb - mp_md_sp
        mp_yb = mp_yt - mp_h
        mp_yc = (mp_yt + mp_yb) // 2
        g_m1_yb = mp_yc - gm1_delta
        # top MP
        mp_yb = md_yt + mp_md_sp
        mp_yt = mp_yb + mp_h
        mp_yc = (mp_yb + mp_yt) // 2
        g_m1_yt = mp_yc + gm1_delta

        # step 4: compute PO locations
        po_yb, po_yt = 0, cpo_top_yc

        # step 5: compute layout information
        lay_info_list = [(lay, 0, po_yb, po_yt) for lay in self.get_mos_layers(sub_type, threshold)]
        lr_edge_info = EdgeInfo(od_type='sub')

        ds_conn_y = (g_m1_yb, g_m1_yt)
        fill_info = FillInfo(layer=('M1', 'drawing'), exc_layer=None,
                             x_intv_list=[], y_intv_list=[ds_conn_y])

        layout_info = dict(
            # information needed for draw_mos
            lch_unit=lch_unit,
            md_w=md_w,
            fg=fg,
            sd_pitch=sd_pitch,
            array_box_xl=0,
            array_box_y=(po_yb, po_yt),
            draw_od=True,
            row_info_list=[RowInfo(od_x_list=[(0, fg)],
                                   od_y=(od_yb, od_yt),
                                   od_type=('sub', sub_type),
                                   po_y=(po_yb, po_yt),
                                   md_y=(md_yb, md_yt)), ],
            lay_info_list=lay_info_list,
            adj_info_list=[],
            left_blk_info=None,
            right_blk_info=None,
            fill_info_list=[fill_info],

            # information needed for computing edge block layout
            blk_type='sub',
            imp_params=None,
        )

        mtype = (sub_type, sub_type)
        po_types = (1,) * fg
        ext_top_info = ExtInfo(
            mx_margin=po_yt - g_m1_yt,
            od_margin=po_yt - od_yt,
            md_margin=po_yt - md_yt,
            m1_margin=po_yt - g_m1_yt,
            imp_min_w=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )
        ext_bot_info = ExtInfo(
            mx_margin=g_m1_yb - po_yb,
            od_margin=od_yb - po_yb,
            md_margin=md_yb - po_yb,
            m1_margin=g_m1_yb - po_yb,
            imp_min_w=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )

        return dict(
            layout_info=layout_info,
            sd_yc=od_yc,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=(lr_edge_info, []),
            right_edge_info=(lr_edge_info, []),
            blk_height=po_yt,
            gb_conn_y=ds_conn_y,
            ds_conn_y=ds_conn_y,
        )

    @classmethod
    def get_analog_end_info(self, lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        """Get substrate end layout information

        Layout is quite simple.  We draw the right CPO width, and extend PO so PO-CPO overlap
        rule is satisfied.

        Strategy:
        If is not end (array abutment), just draw CPO.  If is end:
        1. find margin between bottom coordinate and array box bottom, round up to block pitch.
        #. Compute CPO location, and PO coordinates if we need to draw PO.
        #. Compute implant location.
        """
        fin_h = self.tech_constants['fin_h']
        fin_p = self.tech_constants['fin_pitch']
        cpo_po_ency = self.tech_constants['cpo_po_ency']
        md_w = self.tech_constants['md_w']
        cpo_h = self.tech_constants['cpo_h']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        lr_edge_info = EdgeInfo(od_type='sub')
        if is_end:
            # step 1: figure out Y coordinates of CPO
            blk_pitch = lcm([blk_pitch, fin_p])
            # first assume top Y coordinate is 0
            arr_yt = 0
            cpo_bot_yt = arr_yt + cpo_h // 2
            cpo_bot_yb = cpo_bot_yt - cpo_h
            finbound_yb = arr_yt - fin_p2 - fin_h2
            min_yb = min(finbound_yb, cpo_bot_yb)
            # make sure all layers are in first quadrant
            if min_yb < 0:
                yshift = -(min_yb // blk_pitch) * blk_pitch
                arr_yt += yshift
                cpo_bot_yt += yshift
                cpo_bot_yb += yshift
                finbound_yb += yshift

            finbound_yt = arr_yt + fin_p2 + fin_h2
            cpo_bot_yc = (cpo_bot_yb + cpo_bot_yt) // 2
            po_yt = arr_yt
            po_yb = cpo_bot_yt - cpo_po_ency
            if po_yt > po_yb:
                adj_info_list = [AdjRowInfo(po_y=(po_yb, po_yt), po_types=(1,) * fg)]
                adj_edge_infos = [lr_edge_info]
            else:
                adj_info_list = []
                adj_edge_infos = []

            lay_info_list = [(('CutPoly', 'drawing'), 0, cpo_bot_yb, cpo_bot_yt)]
            for lay in self.get_mos_layers(sub_type, threshold):
                if lay[0] == 'FinArea':
                    yb, yt = finbound_yb, finbound_yt
                else:
                    yb, yt = min(po_yb, cpo_bot_yc), arr_yt
                if yt > yb:
                    lay_info_list.append((lay, 0, yb, yt))
        else:
            # we just draw CPO
            arr_yt = 0
            cpo_h = mos_constants['cpo_h']
            lay_info_list = [(('CutPoly', 'drawing'), 0, -cpo_h // 2, cpo_h // 2)]
            adj_info_list = []
            adj_edge_infos = []

        layout_info = dict(
            # information needed for draw_mos
            lch_unit=lch_unit,
            md_w=md_w,
            fg=fg,
            sd_pitch=sd_pitch,
            array_box_xl=0,
            array_box_y=(0, arr_yt),
            draw_od=True,
            row_info_list=[],
            lay_info_list=lay_info_list,
            adj_info_list=adj_info_list,
            left_blk_info=None,
            right_blk_info=None,
            fill_info_list=[],

            # information needed for computing edge block layout
            blk_type='end',
            imp_params=None,
            left_edge_info=(lr_edge_info, adj_edge_infos),
            right_edge_info=(lr_edge_info, adj_edge_infos),
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=(lr_edge_info, adj_edge_infos),
            right_edge_info=(lr_edge_info, adj_edge_infos),
        )

    @classmethod
    def get_sub_ring_end_info(self, sub_type, threshold, fg, end_ext_info, **kwargs):
        # type: (str, str, int, ExtInfo, **kwargs) -> Dict[str, Any]
        # TODO: add actual implementation
        raise NotImplementedError('not implemented yet.')

    @classmethod
    def get_outer_edge_info(self, guard_ring_nf, layout_info, is_end, adj_blk_info):
        # type: (int, Dict[str, Any], bool, Optional[Any]) -> Dict[str, Any]
        lch_unit = layout_info['lch_unit']
        edge_info = self.get_edge_info(lch_unit, guard_ring_nf, is_end)
        outer_fg = edge_info['outer_fg']
        cpo_xl = edge_info['cpo_xl']
        # compute new row_info_list
        # noinspection PyProtectedMember
        row_info_list = [rinfo._replace(od_x_list=[]) for rinfo in layout_info['row_info_list']]

        # compute new lay_info_list
        lay_info_list = layout_info['lay_info_list']
        imp_params = layout_info['imp_params']
        if guard_ring_nf == 0 or imp_params is None:
            # we keep all implant layers, just update left coordinate.
            new_lay_list = [(lay, cpo_xl if lay[0] == 'CutPoly' else 0, yb, yt)
                            for lay, _, yb, yt in lay_info_list]
        else:
            # we need to convert implant layers to substrate implants
            # first, get all CPO layers
            new_lay_list = [(lay, cpo_xl, yb, yt) for lay, _, yb, yt in lay_info_list if lay[0] == 'CutPoly']
            # compute substrate implant layers
            for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
                sub_type = 'ptap' if mtype == 'nch' or mtype == 'ptap' else 'ntap'
                for lay in self.get_mos_layers(sub_type, thres):
                    cur_yb, cur_yt = imp_yb, imp_yt
                    new_lay_list.append((lay, 0, cur_yb, cur_yt))

        # compute new adj_info_list
        adj_info_list = layout_info['adj_info_list']
        if adj_blk_info is None:
            adj_blk_info = (None, [None] * len(adj_info_list))

        new_adj_list = []
        if outer_fg > 0:
            for adj_edge_info, adj_info in zip(adj_blk_info[1], adj_info_list):
                if adj_edge_info is not None and (adj_edge_info.od_type == 'mos' or adj_edge_info.od_type == 'sub'):
                    po_types = (0,) * (outer_fg - 1) + (1,)
                else:
                    po_types = (0,) * outer_fg
                # noinspection PyProtectedMember
                new_adj_list.append(adj_info._replace(po_types=po_types))

        # compute new fill information
        sd_pitch = layout_info['sd_pitch']
        if outer_fg > 0:
            mos_constants = self.get_mos_tech_constants(lch_unit)
            m1_w = mos_constants['mos_conn_w']
            x_intv_list = [(sd_pitch - m1_w // 2, sd_pitch + m1_w // 2)]
        else:
            x_intv_list = []
        # noinspection PyProtectedMember
        fill_info_list = [f._replace(x_intv_list=x_intv_list) for f in layout_info['fill_info_list']]

        return dict(
            lch_unit=lch_unit,
            md_w=layout_info['md_w'],
            fg=outer_fg,
            sd_pitch=sd_pitch,
            array_box_xl=0,
            array_box_y=layout_info['array_box_y'],
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=new_lay_list,
            adj_info_list=new_adj_list,
            left_blk_info=EdgeInfo(od_type=None),
            right_blk_info=adj_blk_info[0],
            fill_info_list=fill_info_list,

            blk_type='edge' if guard_ring_nf == 0 else 'gr_edge',
        )

    @classmethod
    def get_gr_sub_info(self, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]

        lch_unit = layout_info['lch_unit']
        edge_constants = self.get_edge_tech_constants(lch_unit)
        gr_nf_min = edge_constants['gr_nf_min']
        gr_sub_fg_margin = edge_constants['gr_sub_fg_margin']

        if guard_ring_nf < gr_nf_min:
            raise ValueError('guard_ring_nf = %d < %d' % (guard_ring_nf, gr_nf_min))

        fg = gr_nf_min + 2 + 2 * gr_sub_fg_margin

        # compute new row_info_list
        od_x_list = [(gr_sub_fg_margin + 1, gr_sub_fg_margin + 1 + gr_nf_min)]
        # noinspection PyProtectedMember
        row_info_list = [rinfo._replace(od_x_list=od_x_list, od_type=('sub', rinfo.od_type[1]))
                         for rinfo in layout_info['row_info_list']]

        # compute new lay_info_list
        lay_info_list = layout_info['lay_info_list']
        imp_params = layout_info['imp_params']
        if imp_params is None:
            # don't have to recompute implant layers
            new_lay_list = lay_info_list
        else:
            # we need to convert implant layers to substrate implants
            # first, get all CPO layers
            new_lay_list = [lay_info for lay_info in lay_info_list if lay_info[0][0] == 'CutPoly']
            # compute substrate implant layers
            for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
                sub_type = 'ptap' if mtype == 'nch' or mtype == 'ptap' else 'ntap'
                for lay in self.get_mos_layers(sub_type, thres):
                    cur_yb, cur_yt = imp_yb, imp_yt
                    new_lay_list.append((lay, 0, cur_yb, cur_yt))

        # compute new adj_info_list
        po_types = (0,) * gr_sub_fg_margin + (1,) * (gr_nf_min + 2) + (0,) * gr_sub_fg_margin
        # noinspection PyProtectedMember
        adj_info_list = [ar_info._replace(po_types=po_types) for ar_info in layout_info['adj_info_list']]

        # compute new fill information
        # noinspection PyProtectedMember
        fill_info_list = [f._replace(x_intv_list=[]) for f in layout_info['fill_info_list']]

        return dict(
            lch_unit=lch_unit,
            md_w=layout_info['md_w'],
            fg=fg,
            sd_pitch=layout_info['sd_pitch'],
            array_box_xl=0,
            array_box_y=layout_info['array_box_y'],
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=new_lay_list,
            adj_info_list=adj_info_list,
            left_blk_info=None,
            right_blk_info=None,
            fill_info_list=fill_info_list,

            blk_type='gr_sub',
        )

    @classmethod
    def get_gr_sep_info(self, layout_info, adj_blk_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]

        lch_unit = layout_info['lch_unit']
        edge_constants = self.get_edge_tech_constants(lch_unit)
        fg = edge_constants['gr_sep_fg']

        # compute new row_info_list
        # noinspection PyProtectedMember
        row_info_list = [rinfo._replace(od_x_list=[]) for rinfo in layout_info['row_info_list']]

        # compute new adj_info_list
        adj_info_list = layout_info['adj_info_list']
        new_adj_list = []
        for adj_edge_info, adj_info in zip(adj_blk_info[1], adj_info_list):
            if adj_edge_info.od_type == 'mos' or adj_edge_info.od_type == 'sub':
                po_types = (0,) * (fg - 1) + (1,)
            else:
                po_types = (0,) * fg
            # noinspection PyProtectedMember
            new_adj_list.append(adj_info._replace(po_types=po_types))

        # compute new fill information
        mos_constants = self.get_mos_tech_constants(lch_unit)
        m1_w = mos_constants['mos_conn_w']
        x_intv_list = [(-m1_w // 2, m1_w // 2)]
        # noinspection PyProtectedMember
        fill_info_list = [f._replace(x_intv_list=x_intv_list) for f in layout_info['fill_info_list']]

        return dict(
            lch_unit=lch_unit,
            md_w=layout_info['md_w'],
            fg=fg,
            sd_pitch=layout_info['sd_pitch'],
            array_box_xl=0,
            array_box_y=layout_info['array_box_y'],
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=layout_info['lay_info_list'],
            adj_info_list=new_adj_list,
            left_blk_info=None,
            right_blk_info=adj_blk_info[0],
            fill_info_list=fill_info_list,

            blk_type='gr_sep',
        )

    @classmethod
    def draw_mos(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor related layout.

        the layout information dictionary should contain the following entries:


        blk_type
            a string describing the type of this block.
        lch_unit
            channel length in resolution units
        md_w
            M0OD width in resolution units
        fg
            the width of this template in number of fingers
        sd_pitch
            the source/drain pitch of this template.
        array_box_xl
            array box left coordinate.  All PO X coordinates are calculated
            relative to this point.
        array_box_y
            array box Y coordinates as two-element integer tuple.
        od_type
            the OD type in this template.  Either 'mos', 'sub', or 'dum'.
        draw_od
            If False, we will not draw OD in this template.  This is used for
            supporting the ds_dummy option.
        row_info_list
            a list of named tuples for each transistor row we need to draw in
            this template.

            a transistor row is defines as a row of OD/PO/MD that either acts
            as an active device or used for dummy fill purposes.  Each named tuple
            should have the following entries:

            od_x_list
                A list of transistor X intervals in finger index.
            od_y
                OD Y coordinates as two-element integer tuple.
            po_y
                PO Y coordinates as two-element integer tuple.
            md_y
                MD Y coordinates as two-element integer tuple.
        lay_info_list
            a list of layers to draw.  Each layer information is a tuple
            of (imp_layer, xl, yb, yt).
        adj_info_list
            a list of named tuples for geometries belonging to adjacent
            rows.  Each named tuple should contain:

            po_y
                PO Y coordinates as two-element integer tuple.
            po_types
                list of po types.  1 for drawing, 0 for dummy.
        left_blk_info
            a tuple of (EdgeInfo, List[EdgeInfo]) that represents edge information
            of the left adjacent block.  These influences the geometry abutting the
            left block.  If None, assume default behavior.
        right_blk_info
            same as left_blk_info, but for the right edge.
        fill_info_list:
            a list of fill information named tuple.  Each tuple contains:

            layer
                the fill layer
            exc_layer
                the fill exclusion layer
            x_intv_list
                a list of X intervals of the fill
            y_intv_list
                a list of Y intervals of the fill

        Parameters
        ----------
        template : TemplateBase
            the template to draw the layout in.
        layout_info : Dict[str, Any]
            the layout information dictionary.
        """
        res = template.grid.resolution

        fin_pitch = self.tech_constants['fin_pitch']
        fin_h = self.tech_constants['fin_h']

        fin_pitch2 = fin_pitch // 2
        fin_h2 = fin_h // 2

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        md_w = layout_info['md_w']
        fg = layout_info['fg']
        sd_pitch = layout_info['sd_pitch']
        arr_xl = layout_info['array_box_xl']
        arr_yb, arr_yt = layout_info['array_box_y']
        draw_od = layout_info['draw_od']
        row_info_list = layout_info['row_info_list']
        lay_info_list = layout_info['lay_info_list']
        adj_info_list = layout_info['adj_info_list']
        left_blk_info = layout_info['left_blk_info']
        right_blk_info = layout_info['right_blk_info']
        fill_info_list = layout_info['fill_info_list']

        default_edge_info = EdgeInfo(od_type=None)
        if left_blk_info is None:
            if fg == 1 and right_blk_info is not None:
                # make sure if we only have one finger, PO purpose is still chosen correctly.
                left_blk_info = right_blk_info
            else:
                left_blk_info = default_edge_info
        if right_blk_info is None:
            if fg == 1:
                # make sure if we only have one finger, PO purpose is still chosen correctly.
                right_blk_info = left_blk_info
            else:
                right_blk_info = default_edge_info

        blk_w = fg * sd_pitch + arr_xl

        # figure out transistor layout settings
        od_dum_lay = ('Active', 'dummy')
        po_dum_lay = ('Poly', 'dummy')
        md_lay = ('LiAct', 'drawing')

        po_xc = arr_xl + sd_pitch // 2
        # draw transistor rows
        for row_info in row_info_list:
            od_type = row_info.od_type[0]
            if od_type == 'dum' or od_type is None:
                od_lay = od_dum_lay
            else:
                od_lay = ('Active', 'drawing')
            od_x_list = row_info.od_x_list
            od_yb, od_yt = row_info.od_y
            po_yb, po_yt = row_info.po_y
            md_yb, md_yt = row_info.md_y

            po_on_od = [False] * fg
            md_on_od = [False] * (fg + 1)
            if od_yt > od_yb:
                # draw OD and figure out PO/MD info
                for od_start, od_stop in od_x_list:
                    # mark PO/MD indices that are on OD
                    if od_start - 1 >= 0:
                        po_on_od[od_start - 1] = True
                    for idx in range(od_start, od_stop + 1):
                        md_on_od[idx] = True
                        if idx < fg:
                            po_on_od[idx] = True

                    if draw_od:
                        od_xl = po_xc - lch_unit // 2 + (od_start - 1) * sd_pitch
                        od_xr = po_xc + lch_unit // 2 + od_stop * sd_pitch
                        template.add_rect(od_lay, BBox(od_xl, od_yb, od_xr, od_yt, res, unit_mode=True))

            # draw PO
            if po_yt > po_yb:
                for idx in range(fg):
                    po_xl = po_xc + idx * sd_pitch - lch_unit // 2
                    po_xr = po_xl + lch_unit
                    if po_on_od[idx]:
                        cur_od_type = od_type
                        is_edge = False
                    else:
                        if idx == 0:
                            cur_od_type = left_blk_info.od_type
                            is_edge = True
                        elif idx == fg - 1:
                            cur_od_type = right_blk_info.od_type
                            is_edge = True
                        else:
                            cur_od_type = None
                            is_edge = False

                    if is_edge and cur_od_type is not None:
                        lay = ('Poly', 'edge')
                    elif cur_od_type == 'mos' or cur_od_type == 'sub':
                        lay = ('Poly', 'drawing')
                    else:
                        lay = ('Poly', 'dummy')
                    template.add_rect(lay, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True))

            # draw MD if it's physical
            if md_yt > md_yb and fg > 0:
                md_range = range(1, fg) if blk_type == 'gr_sub' else range(fg + 1)
                for idx in md_range:
                    md_xl = arr_xl + idx * sd_pitch - md_w // 2
                    md_xr = md_xl + md_w
                    if md_on_od[idx]:
                        template.add_rect(md_lay, BBox(md_xl, md_yb, md_xr, md_yt, res, unit_mode=True))

        # draw other layers
        for imp_lay, xl, yb, yt in lay_info_list:
            if imp_lay[0] == 'FinArea':
                # round to fin grid
                yb = (yb - fin_pitch2 + fin_h2) // fin_pitch * fin_pitch + fin_pitch2 - fin_h2
                yt = -(-(yt - fin_pitch2 - fin_h2) // fin_pitch) * fin_pitch + fin_pitch2 + fin_h2
            box = BBox(xl, yb, blk_w, yt, res, unit_mode=True)
            if box.is_physical():
                template.add_rect(imp_lay, box)

        # draw adjacent row geometries
        for adj_info in adj_info_list:
            po_yb, po_yt = adj_info.po_y
            for idx, po_type in enumerate(adj_info.po_types):
                lay = po_dum_lay if po_type == 0 else ('Poly', 'drawing')
                po_xl = po_xc + idx * sd_pitch - lch_unit // 2
                po_xr = po_xl + lch_unit
                template.add_rect(lay, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True))

        # set size and add PR boundary
        arr_box = BBox(arr_xl, arr_yb, blk_w, arr_yt, res, unit_mode=True)
        bound_box = arr_box.extend(x=0, y=0, unit_mode=True)
        template.array_box = arr_box
        template.prim_bound_box = bound_box
        if bound_box.is_physical():
            template.add_cell_boundary(bound_box)

            # draw metal fill.  This only needs to be done if the template has nonzero area.
            for fill_info in fill_info_list:
                exc_lay = fill_info.exc_layer
                lay = fill_info.layer
                x_intv_list = fill_info.x_intv_list
                y_intv_list = fill_info.y_intv_list
                if exc_lay is not None:
                    template.add_rect(exc_lay, bound_box)
                for xl, xr in x_intv_list:
                    for yb, yt in y_intv_list:
                        template.add_rect(lay, BBox(xl, yb, xr, yt, res, unit_mode=True))

    @classmethod
    def draw_substrate_connection(self, template, layout_info, port_tracks, dum_tracks, dummy_only,
                                  is_laygo, is_guardring):
        # type: (TemplateBase, Dict[str, Any], List[int], List[int], bool, bool, bool) -> bool

        fin_h = self.tech_constants['fin_h']
        fin_p = self.tech_constants['fin_pitch']
        mp_md_sp = self.tech_constants['mp_md_sp']
        mp_h = self.tech_constants['mp_h']
        mp_po_ovl = self.tech_constants['mp_po_ovl']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        row_info_list = layout_info['row_info_list']

        sd_pitch2 = sd_pitch // 2

        has_od = False
        for row_info in row_info_list:
            od_yb, od_yt = row_info.od_y
            if od_yt > od_yb:
                has_od = True
                # find current port name
                od_start, od_stop = row_info.od_x_list[0]
                fg = od_stop - od_start
                xshift = od_start * sd_pitch
                sub_type = row_info.od_type[1]
                port_name = 'VDD' if sub_type == 'ntap' else 'VSS'

                # draw substrate connection only if OD exists.
                od_yc = (od_yb + od_yt) // 2
                w = (od_yt - od_yb - fin_h) // fin_p + 1

                via_info = self.get_ds_via_info(lch_unit, w, compact=is_guardring)

                # find X locations of M1/M3.
                # we can export all dummy tracks.
                m1_x_list = [idx * sd_pitch for idx in range(fg + 1)]
                if dummy_only:
                    # find X locations to draw vias
                    m3_x_list = []
                else:
                    # first, figure out port/dummy tracks.
                    # Try to add as many unused tracks to port tracks as possible, while making sure we don't end
                    # up with adjacent port tracks.  This improves substrate connection resistance to supply.

                    # use half track indices so we won't have rounding errors.
                    phtr_set = set((int(2 * v + 1) for v in port_tracks))
                    # add as many unused tracks as possible to port tracks
                    for htr in range(0, 2 * fg + 1, 2):
                        if htr + 2 not in phtr_set and htr - 2 not in phtr_set:
                            phtr_set.add(htr)
                    # find X coordinates
                    m3_x_list = [sd_pitch2 * v for v in sorted(phtr_set)]

                m1_warrs, m3_warrs = self._draw_ds_via(template, sd_pitch, od_yc, fg, via_info, 1, 1,
                                                       m1_x_list, m3_x_list, xshift=xshift)
                template.add_pin(port_name, m1_warrs, show=False)
                template.add_pin(port_name, m3_warrs, show=False)

                if not is_guardring:
                    md_yb, md_yt = row_info.md_y
                    # draw M0PO connections
                    res = template.grid.resolution
                    gv0_h = via_info['h'][0]
                    gv0_w = via_info['w'][0]
                    top_encx = via_info['top_encx'][0]
                    top_ency = via_info['top_ency'][0]
                    gm1_delta = gv0_h // 2 + top_ency
                    m1_w = gv0_w + 2 * top_encx
                    bot_encx = (m1_w - gv0_w) // 2
                    bot_ency = (mp_h - gv0_h) // 2
                    # bottom MP
                    mp_yt = md_yb - mp_md_sp
                    mp_yb = mp_yt - mp_h
                    mp_yc = (mp_yt + mp_yb) // 2
                    m1_yb = mp_yc - gm1_delta
                    mp_y_list = [(mp_yb, mp_yt)]
                    # top MP
                    mp_yb = md_yt + mp_md_sp
                    mp_yt = mp_yb + mp_h
                    mp_yc = (mp_yb + mp_yt) // 2
                    m1_yt = mp_yc + gm1_delta
                    mp_y_list.append((mp_yb, mp_yt))

                    # draw MP, M1, and VIA0
                    via_type = 'M1_LiPo'
                    mp_dx = sd_pitch // 2 - lch_unit // 2 + mp_po_ovl
                    enc1 = [bot_encx, bot_encx, bot_ency, bot_ency]
                    enc2 = [top_encx, top_encx, top_ency, top_ency]
                    for idx in range(0, fg + 1, 2):
                        mp_xl = xshift + idx * sd_pitch - mp_dx
                        mp_xr = xshift + idx * sd_pitch + mp_dx
                        for mp_yb, mp_yt in mp_y_list:
                            template.add_rect('LiPo', BBox(mp_xl, mp_yb, mp_xr, mp_yt, res, unit_mode=True))
                        m1_xc = xshift + idx * sd_pitch
                        template.add_rect('M1', BBox(m1_xc - m1_w // 2, m1_yb, m1_xc + m1_w // 2, m1_yt, res,
                                                     unit_mode=True))
                        for mp_yb, mp_yt in mp_y_list:
                            mp_yc = (mp_yb + mp_yt) // 2
                            template.add_via_primitive(via_type, [m1_xc, mp_yc], enc1=enc1, enc2=enc2, unit_mode=True)

        return has_od

    @classmethod
    def draw_mos_connection(self, template, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                            min_ds_cap, is_diff, diode_conn, options):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, Dict[str, Any]) -> None

        # NOTE: ignore min_ds_cap.
        if is_diff:
            raise ValueError('differential connection not supported yet.')

        fin_h = self.tech_constants['fin_h']
        fin_pitch = self.tech_constants['fin_pitch']

        gate_yc = mos_info['gate_yc']
        layout_info = mos_info['layout_info']
        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        sd_pitch = layout_info['sd_pitch']
        od_yb, od_yt = layout_info['row_info_list'][0].od_y

        w = (od_yt - od_yb - fin_h) // fin_pitch + 1
        ds_via_info = self.get_ds_via_info(lch_unit, w)

        g_via_info = self.get_gate_via_info(lch_unit)

        stack = options.get('stack', 1)
        wire_pitch = stack * sd_pitch
        if fg % stack != 0:
            raise ValueError('AnalogMosConn: stack = %d must evenly divides fg = %d' % (stack, fg))
        num_seg = fg // stack

        s_x_list = list(range(0, num_seg * wire_pitch + 1, 2 * wire_pitch))
        d_x_list = list(range(wire_pitch, num_seg * wire_pitch + 1, 2 * wire_pitch))
        sd_yc = (od_yb + od_yt) // 2
        if diode_conn:
            if fg == 1:
                raise ValueError('1 finger transistor connection not supported.')

            sloc = 0 if sdir <= 1 else 2
            dloc = 2 - sloc

            # draw source
            _, sarr = self._draw_ds_via(template, wire_pitch, 0, num_seg, ds_via_info, sloc, sdir,
                                        s_x_list, s_x_list)
            # draw drain
            m1d, darr = self._draw_ds_via(template, wire_pitch, 0, num_seg, ds_via_info, dloc, ddir,
                                          d_x_list, d_x_list)
            # draw gate
            m1g, _ = self._draw_g_via(template, lch_unit, fg, sd_pitch, gate_yc - sd_yc, g_via_info, [],
                                      gate_ext_mode=gate_ext_mode)
            m1_yt = m1d[0].upper
            m1_yb = m1g[0].lower
            template.add_wires(1, m1d[0].track_id.base_index, m1_yb, m1_yt, num=len(m1d), pitch=2 * stack)
            template.add_pin('g', WireArray.list_to_warr(darr), show=False)
            template.add_pin('d', WireArray.list_to_warr(darr), show=False)
            template.add_pin('s', WireArray.list_to_warr(sarr), show=False)
        else:
            # determine gate location
            if sdir == 0:
                gloc = 'd'
            elif ddir == 0:
                gloc = 's'
            else:
                gloc = gate_pref_loc

            if (gloc == 's' and num_seg == 2) or gloc == 'd':
                sloc, dloc = 0, 2
            else:
                sloc, dloc = 2, 0

            if gloc == 'd':
                g_x_list = list(range(wire_pitch, num_seg * wire_pitch, 2 * wire_pitch))
            else:
                if num_seg != 2:
                    g_x_list = list(range(2 * wire_pitch, num_seg * wire_pitch, 2 * wire_pitch))
                else:
                    g_x_list = [0, 2 * wire_pitch]

            # draw gate
            _, garr = self._draw_g_via(template, lch_unit, fg, sd_pitch, gate_yc - sd_yc, g_via_info,
                                       g_x_list, gate_ext_mode=gate_ext_mode)
            # draw source
            _, sarr = self._draw_ds_via(template, wire_pitch, 0, num_seg, ds_via_info, sloc, sdir,
                                        s_x_list, s_x_list)
            # draw drain
            _, darr = self._draw_ds_via(template, wire_pitch, 0, num_seg, ds_via_info, dloc, ddir,
                                        d_x_list, d_x_list)

            template.add_pin('s', WireArray.list_to_warr(sarr), show=False)
            template.add_pin('d', WireArray.list_to_warr(darr), show=False)
            template.add_pin('g', WireArray.list_to_warr(garr), show=False)

    @classmethod
    def draw_dum_connection(self, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[int], Dict[str, Any]) -> None

        fin_h = self.tech_constants['fin_h']
        fin_pitch = self.tech_constants['fin_pitch']
        m1_dum_h = self.tech_constants['m1_dum_h']

        gate_yc = mos_info['gate_yc']
        layout_info = mos_info['layout_info']
        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        sd_pitch = layout_info['sd_pitch']
        od_yb, od_yt = layout_info['row_info_list'][0].od_y

        sd_yc = (od_yb + od_yt) // 2
        w = (od_yt - od_yb - fin_h) // fin_pitch + 1
        ds_via_info = self.get_ds_via_info(lch_unit, w)

        g_via_info = self.get_gate_via_info(lch_unit)

        left_edge = edge_mode % 2 == 1
        right_edge = edge_mode // 2 == 1
        if left_edge:
            ds_x_start = 0
        else:
            ds_x_start = sd_pitch
        if right_edge:
            ds_x_stop = fg * sd_pitch
        else:
            ds_x_stop = (fg - 1) * sd_pitch

        ds_x_list = list(range(ds_x_start, ds_x_stop + 1, sd_pitch))

        # draw gate
        m1g, _ = self._draw_g_via(template, lch_unit, fg, sd_pitch, gate_yc - sd_yc, g_via_info, [])
        # draw drain/source
        m1d, _ = self._draw_ds_via(template, sd_pitch, 0, fg, ds_via_info, 1, 1, ds_x_list, [], draw_m2=False)

        # connect gate and drain/source together
        res = template.grid.resolution
        m1_yb = int(round(m1g[0].lower / res))
        m1_yt = m1_yb + m1_dum_h
        template.connect_wires(m1g + m1d, lower=m1_yb, unit_mode=True)
        if ds_x_stop > ds_x_start:
            template.add_rect('M1', BBox(ds_x_start, m1_yb, ds_x_stop, m1_yt, res, unit_mode=True))

        template.add_pin('dummy', m1g, show=False)

    @classmethod
    def draw_decap_connection(self, template, mos_info, sdir, ddir, gate_ext_mode, export_gate, options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
        raise NotImplementedError('Not implemented')

    @classmethod
    def _draw_g_via(self, template, lch_unit, fg, sd_pitch, gate_yc, via_info, m3_x_list,
                    gate_ext_mode=0, dx=0):

        res = self.tech_constants['resolution']
        mp_po_ovl = self.tech_constants['mp_po_ovl']
        mp_h = self.tech_constants['mp_h']
        mx_area_min = self.tech_constants['mx_area_min']

        w_list = via_info['w']
        h_list = via_info['h']
        bot_encx = via_info['bot_encx']
        top_encx = via_info['top_encx']
        bot_ency = via_info['bot_ency']
        top_ency = via_info['top_ency']
        m1_h = via_info['m1_h']
        m2_h = via_info['m2_h']
        m3_h = via_info['m3_h']

        m1_top_ency = top_ency[0]
        v0_h = h_list[0]
        m2_top_encx = top_encx[1]
        v1_w = w_list[1]

        m1_yt = gate_yc + v0_h // 2 + m1_top_ency
        m1_yb = m1_yt - m1_h

        # compute minimum M2 width from area rule, make sure it's even.
        m2_w_min = -(-mx_area_min // (2 * m2_h)) * 2
        m2_yb = gate_yc - m2_h // 2
        m2_yt = m2_yb + m2_h

        if fg % 2 == 0:
            gate_fg_list = [2] * (fg // 2)
        else:
            if fg == 1:
                raise ValueError('cannot connect 1 finger transistor')
            if fg <= 5:
                gate_fg_list = [fg]
            else:
                num_mp_half = (fg - 3) // 2
                gate_fg_list = [2] * num_mp_half
                gate_fg_list.append(3)
                gate_fg_list.extend((2 for _ in range(num_mp_half)))

        m2_xl = m2_xr = dx + fg * sd_pitch // 2
        # extend gate in left/right direction with M2 if necessary.
        if gate_ext_mode % 2 == 1:
            m2_xl = dx
        if gate_ext_mode // 2 == 1:
            m2_xr = dx + fg * sd_pitch

        # connect gate to M2.
        v0_enc1 = [bot_encx[0], bot_encx[0], bot_ency[0], bot_ency[0]]
        v0_enc2 = [top_encx[0], top_encx[0], top_ency[0], top_ency[0]]
        v1_enc1 = [bot_encx[1], bot_encx[1], bot_ency[1], bot_ency[1]]
        v1_enc2 = [top_encx[1], top_encx[1], top_ency[1], top_ency[1]]
        tot_fg = 0
        m1_warrs = []
        for num_fg in gate_fg_list:
            via_xoff = dx + (tot_fg + 1) * sd_pitch
            cur_xc = dx + tot_fg * sd_pitch + num_fg * sd_pitch // 2
            # draw MP
            mp_w = (num_fg - 1) * sd_pitch - lch_unit + 2 * mp_po_ovl
            mp_xl = cur_xc - mp_w // 2
            mp_xr = mp_xl + mp_w
            mp_yb = gate_yc - mp_h // 2
            mp_yt = mp_yb + mp_h
            template.add_rect(('LiPo', 'drawing'), BBox(mp_xl, mp_yb, mp_xr, mp_yt, res, unit_mode=True))
            # draw V0, M1, and V1
            for idx in range(num_fg - 1):
                via_xc = via_xoff + idx * sd_pitch
                vloc = [via_xc, gate_yc]
                cur_tidx = template.grid.coord_to_track(1, via_xc, unit_mode=True)
                template.add_via_primitive('M1_LiPo', vloc, enc1=v0_enc1, enc2=v0_enc2, unit_mode=True)
                m1_warrs.append(template.add_wires(1, cur_tidx, m1_yb, m1_yt, unit_mode=True))
                if m3_x_list:
                    template.add_via_primitive('M2_M1', vloc, enc1=v1_enc1, enc2=v1_enc2, unit_mode=True)

            m2_xl = min(via_xoff - v1_w // 2 - m2_top_encx, m2_xl)
            m2_xr = max(via_xoff + (num_fg - 2) * sd_pitch + v1_w // 2 + m2_top_encx, m2_xr)
            tot_fg += num_fg

        # fix M2 area rule
        m2_xc = (m2_xl + m2_xr) // 2
        m2_xl = min(m2_xl, m2_xc - m2_w_min // 2)
        m2_xl = max(dx, m2_xl)
        m2_xr = max(m2_xr, m2_xl + m2_w_min)
        if m3_x_list:
            template.add_rect('M2', BBox(m2_xl, m2_yb, m2_xr, m2_yt, res, unit_mode=True))

        # connect gate to M3
        m3_warrs = []
        v2_h = h_list[2]
        v2_enc1 = [bot_encx[2], bot_encx[2], bot_ency[2], bot_ency[2]]
        v2_enc2 = [top_encx[2], top_encx[2], top_ency[2], top_ency[2]]
        m3_yt = gate_yc + v2_h // 2 + v2_enc2[2]
        m3_yb = m3_yt - m3_h
        for xc in m3_x_list:
            template.add_via_primitive('M3_M2', [xc, gate_yc], enc1=v2_enc1, enc2=v2_enc2,
                                       cut_height=v2_h, unit_mode=True)
            tr_idx = template.grid.coord_to_track(3, xc, unit_mode=True)
            m3_warrs.append(template.add_wires(3, tr_idx, m3_yb, m3_yt, unit_mode=True))

        return m1_warrs, m3_warrs

    @classmethod
    def _draw_ds_via(self, template, wire_pitch, od_yc, num_seg, via_info, m2_loc, m3_dir,
                     m1_x_list, m3_x_list, xshift=0, draw_m2=True):
        res = self.tech_constants['resolution']
        v0_sp = self.tech_constants['v0_sp']
        mx_area_min = self.tech_constants['mx_area_min']

        nv0 = via_info['num_v0']
        m1_h = via_info['m1_h']
        m2_h = via_info['m2_h']
        m3_h = via_info['m3_h']
        md_encx, m1_bot_encx, m2_bot_encx = via_info['bot_encx']
        m1_encx, m2_encx, m3_encx = via_info['top_encx']
        md_ency, m1_bot_ency, m2_bot_ency = via_info['bot_ency']
        m1_ency, m2_ency, m3_ency = via_info['top_ency']
        v0_h, v1_h, v2_h = via_info['h']

        # draw via to M1
        via_type = 'M1_LiAct'
        enc1 = [md_encx, md_encx, md_ency, md_ency]
        enc2 = [m1_encx, m1_encx, m1_ency, m1_ency]
        template.add_via_primitive(via_type, [xshift, od_yc], num_rows=nv0, sp_rows=v0_sp,
                                   enc1=enc1, enc2=enc2, nx=num_seg + 1, spx=wire_pitch, unit_mode=True)

        # find M2 location
        m1_yb = od_yc - m1_h // 2
        m1_yt = m1_yb + m1_h
        if m2_loc == 0:
            via_yc = m1_yb + m1_bot_ency + v1_h // 2
        elif m2_loc == 1:
            via_yc = od_yc
        else:
            via_yc = m1_yt - m1_bot_ency - v1_h // 2

        # add M1 and M2
        m2_xl, m2_xr = None, None
        m1_warrs = []
        v1_enc1 = [m1_bot_encx, m1_bot_encx, m1_bot_ency, m1_bot_ency]
        v1_enc2 = [m2_encx, m2_encx, m2_ency, m2_ency]
        for xloc in m1_x_list:
            tidx = template.grid.coord_to_track(1, xloc, unit_mode=True)
            cur_warr = template.add_wires(1, tidx, m1_yb, m1_yt, unit_mode=True)
            m1_warrs.append(cur_warr)
            if draw_m2:
                template.add_via_primitive('M2_M1', [xloc, via_yc], cut_height=v1_h, enc1=v1_enc1,
                                           enc2=v1_enc2, unit_mode=True)
                m2_xl = xloc if m2_xl is None else min(xloc, m2_xl)
                m2_xr = xloc if m2_xr is None else max(xloc, m2_xr)

        # draw via to M3 and add metal/ports
        v2_enc1 = [m2_bot_encx, m2_bot_encx, m2_bot_ency, m2_bot_ency]
        v2_enc2 = [m3_encx, m3_encx, m3_ency, m3_ency]
        m3_warrs = []
        for xloc in m3_x_list:
            if m3_dir == 0:
                m_yt = via_yc + v2_h // 2 + v2_enc2[2]
                m_yb = m_yt - m3_h
            elif m3_dir == 2:
                m_yb = via_yc - v2_h // 2 - v2_enc2[3]
                m_yt = m_yb + m3_h
            else:
                m_yb = via_yc - m3_h // 2
                m_yt = m_yb + m3_h

            cur_xc = xshift + xloc
            loc = [cur_xc, via_yc]
            template.add_via_primitive('M3_M2', loc, cut_height=v2_h, enc1=v2_enc1, enc2=v2_enc2, unit_mode=True)
            tr_idx = template.grid.coord_to_track(3, cur_xc, unit_mode=True)
            m3_warrs.append(template.add_wires(3, tr_idx, m_yb, m_yt, unit_mode=True))

        if m2_xl is not None and m2_xr is not None:
            # fix M2 area rule
            m2_w_min = -(-mx_area_min // (2 * m2_h)) * 2
            m2_xc = (m2_xl + m2_xr) // 2
            m2_xl = min(m2_xl, m2_xc - m2_w_min // 2)
            m2_xr = max(m2_xr, m2_xl + m2_w_min)

            m2_yb = via_yc - m2_h // 2
            m2_yt = m2_yb + m2_h
            template.add_rect('M2', BBox(m2_xl, m2_yb, m2_xr, m2_yt, res, unit_mode=True))

        return m1_warrs, m3_warrs
