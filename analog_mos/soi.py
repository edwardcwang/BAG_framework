# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, Union, List, Optional
from itertools import chain
from collections import namedtuple

from bag.layout.util import BBox
from bag.layout.routing import WireArray, TrackID
from bag.layout.template import TemplateBase

from .core import MOSTech

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig

ExtInfo = namedtuple('ExtInfo', ['mx_margin', 'imp_margins', 'mtype', 'thres'])


class MOSTechSOIGenericBC(MOSTech):
    """A generic implementation of MOSTech for SOI technologies with body-connected transistors."""

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        MOSTech.__init__(self, config, tech_info)

    @classmethod
    def get_analog_unit_fg(cls):
        # type: () -> int
        return 2

    @classmethod
    def draw_zero_extension(cls):
        # type: () -> bool
        return False

    @classmethod
    def floating_dummy(cls):
        # type: () -> bool
        return True

    @classmethod
    def abut_analog_mos(cls):
        # type: () -> bool
        return False

    @classmethod
    def get_substrate_ring_lch(cls):
        # type: () -> float
        return 56e-9

    @classmethod
    def get_dum_conn_pitch(cls):
        # type: () -> int
        return 1

    @classmethod
    def get_dum_conn_layer(cls):
        # type: () -> int
        return _config['mos_analog']['dum_layer']

    @classmethod
    def get_mos_conn_layer(cls):
        # type: () -> int
        return _config['mos_analog']['conn_layer']

    @classmethod
    def get_dig_conn_layer(cls):
        # type: () -> int
        raise ValueError('Not supported')

    @classmethod
    def get_dig_top_layer(cls):
        # type: () -> int
        raise ValueError('Not supported')

    @classmethod
    def get_min_fg_decap(cls, lch_unit):
        # type: (int) -> int
        return 0

    @classmethod
    def get_min_fg_sep(cls, lch_unit):
        # type: (int) -> int
        return 2

    @classmethod
    def get_tech_constant(cls, name):
        # type: (str) -> Any
        return _config[name]

    @classmethod
    def get_mos_pitch(cls, unit_mode=False):
        # type: (bool) -> Union[float, int]
        if unit_mode:
            return 1
        return _config['resolution']

    @classmethod
    def get_edge_info(cls, lch_unit, guard_ring_nf, is_end, **kwargs):
        # type: (int, int, bool) -> Dict[str, Any]

        mos_constants = cls.get_mos_tech_constants(lch_unit)
        return dict(
            edge_num_fg=0,
            edge_margin=mos_constants['edge_margin'],
        )

    @classmethod
    def get_mos_info(cls, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]

        # get transistor constants
        mos_conn_layer = cls.get_mos_conn_layer()
        mos_constants = cls.get_mos_tech_constants(lch_unit)

        sd_pitch = mos_constants['sd_pitch']
        sp_bpo_v0 = mos_constants['sp_bpo_v0']
        sp_gb_m1 = mos_constants['sp_gb_m1']
        sp_gd_m1 = mos_constants['sp_gd_m1']
        sp_gd_od = mos_constants['sp_gd_od']
        h_body = mos_constants['h_body']
        body_po_loc = mos_constants['body_po_loc']
        num_via_body = mos_constants['num_via_body']
        w_conn_g = mos_constants['w_conn_g']
        w_conn_d = mos_constants['w_conn_d']

        # get via constants
        mos_lay_table = _config['mos_layer_table']
        mos_analog_config = _config['mos_analog']
        via_g = mos_analog_config['via_g']
        via_d = mos_analog_config['via_d']
        via_b = mos_analog_config['via_b']
        imp_layers_info = mos_analog_config['imp_layers'][mos_type]
        thres_layers_info = mos_analog_config['thres_layers'][mos_type][threshold]

        # convert w to resolution units
        layout_unit = _config['layout_unit']
        res = _config['resolution']
        w_unit = int(round(w / layout_unit / res))

        # get minimum metal lengths
        mx_min_len = 0
        for layer_id in range(2, mos_conn_layer + 1):
            layer_type = TechInfoSOIGeneric.layer_id_to_type(layer_id)
            mx_min_len = max(mx_min_len, TechInfoSOIGeneric.get_min_length_unit(layer_type, w_conn_g[layer_id]))

        m1_type = TechInfoSOIGeneric.layer_id_to_type(1)
        m1_min_len = TechInfoSOIGeneric.get_min_length_unit(m1_type, w_conn_d[1])
        md_min_len = max(m1_min_len, mx_min_len)

        # compute gate location
        g_v1_h = via_g['dim'][1][1]
        g_v1_m2_ency = via_g['top_enc_le'][1]
        g_po_h = w_conn_g[0]
        g_m1_h = w_conn_g[1]
        g_m1_yb = sp_gb_m1 // 2
        g_m1_yt = g_m1_yb + g_m1_h
        g_m1_yc = (g_m1_yb + g_m1_yt) // 2
        g_po_yb = g_m1_yc - g_po_h // 2
        g_po_yt = g_po_yb + g_po_h
        g_mx_yt = g_m1_yc + g_v1_h // 2 + g_v1_m2_ency
        g_mx_yb = g_mx_yt - mx_min_len
        g_y_list = [(g_po_yb, g_po_yt), (g_m1_yb, g_m1_yt), (g_mx_yb, g_mx_yt)]

        # compute drain/source location
        # first, get OD and body bar location from sp_gd_od
        d_od_yb = g_po_yt + sp_gd_od
        d_od_yt = d_od_yb + w_unit
        b_od_yb = d_od_yt
        b_po_yb = b_od_yb + body_po_loc[0]
        # get top edge of drain/source contact
        d_v0_yt = b_po_yb - sp_bpo_v0
        # get number of vias
        d_v0_h = via_d['dim'][0][1]
        d_v0_sp = via_d['sp'][0]
        d_v0_od_ency = via_d['bot_enc_le'][0]
        d_v0_m1_ency = via_d['top_enc_le'][0]
        d_v0_arrh = d_v0_yt - d_od_yb - d_v0_od_ency
        d_v0_n = (d_v0_arrh + d_v0_sp) // (d_v0_h + d_v0_sp)
        # get metal length and bottom metal coordinate
        d_v0_arrh = d_v0_n * (d_v0_h + d_v0_sp) - d_v0_sp
        mx_h = max(md_min_len, d_v0_arrh + 2 * d_v0_m1_ency)
        d_v0_m1_ency = (mx_h - d_v0_arrh) // 2
        d_mx_yb = d_v0_yt + d_v0_m1_ency - mx_h
        # check sp_gd_m1 spec, move everything up if necessary
        delta = sp_gd_m1 - (d_mx_yb - g_m1_yt)
        if delta > 0:
            d_mx_yb += delta
            d_v0_yt += delta
            b_po_yb += delta
            b_od_yb += delta
            d_od_yt += delta
            d_od_yb += delta

        # compute locations
        d_mx_yt = d_mx_yb + mx_h
        d_od_yc = (d_od_yb + d_od_yt) // 2
        d_y_list = [(d_od_yb, d_od_yt), (d_mx_yb, d_mx_yt)]

        # compute body location
        b_v0_w, b_v0_h = via_b['dim']
        b_v0_sp = via_b['sp']
        b_v0_enc1 = via_b['enc1']
        b_v0_enc2 = via_b['enc2']
        b_od_yt = b_od_yb + h_body
        b_v0_arrh = num_via_body * (b_v0_h + b_v0_sp) - b_v0_sp
        b_m1_yt = b_od_yt - b_v0_enc1[2] + b_v0_enc2[2]
        b_m1_yb = b_m1_yt - b_v0_enc2[2] - b_v0_enc2[3] - b_v0_arrh
        b_y_list = [(b_od_yb, b_od_yt), (b_m1_yb, b_m1_yt)]
        b_po_yt = b_od_yb + body_po_loc[1]
        b_po_yc = (b_po_yb + b_po_yt) // 2
        b_po_y_list = [(b_po_yb, b_po_yt), (b_po_yb, b_po_yc), (b_po_yc, b_po_yt)]
        arr_y = (0, b_m1_yt + sp_gb_m1 // 2)

        # compute extension information
        imp_info, top_imp_margins, bot_imp_margins = {}, {}, {}
        y_lookup = {'b': b_y_list, 'g': g_y_list, 'd': d_y_list, 'bpo': b_po_y_list}
        od_name = mos_lay_table['OD']
        bag_purpose = 'bag_' + mos_type
        top_imp_margins[od_name] = top_imp_margins[(od_name[0], bag_purpose)] = arr_y[1] - b_od_yt
        bot_imp_margins[od_name] = bot_imp_margins[(od_name[0], bag_purpose)] = d_od_yb
        for imp_name, (bot_enc, top_enc) in chain(imp_layers_info.items(), thres_layers_info.items()):
            bot_type, bot_idx, bot_delta = bot_enc
            top_type, top_idx, top_delta = top_enc
            imp_yb = y_lookup[bot_type][bot_idx][0] - bot_delta
            imp_yt = y_lookup[top_type][top_idx][1] + top_delta
            imp_info[imp_name] = (imp_yb, imp_yt)
            top_imp_margins[imp_name] = arr_y[1] - imp_yt
            bot_imp_margins[imp_name] = imp_yb

        top_mx_margin = arr_y[1] - d_y_list[-1][-1]
        bot_mx_margin = g_y_list[-1][0]
        ext_top_info = ExtInfo(mx_margin=top_mx_margin, imp_margins=top_imp_margins, mtype=mos_type, thres=threshold)
        ext_bot_info = ExtInfo(mx_margin=bot_mx_margin, imp_margins=bot_imp_margins, mtype=mos_type, thres=threshold)

        layout_info = dict(
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=arr_y,
            b_po_y_list=b_po_y_list,
            g_y_list=g_y_list,
            d_y_list=d_y_list,
            b_y_list=b_y_list,
            imp_info=imp_info,

            blk_type='mos',
        )
        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=None,
            right_edge_info=None,
            sd_yc=d_od_yc,
            g_conn_y=g_y_list[-1],
            d_conn_y=d_y_list[-1],
        )

    @classmethod
    def get_valid_extension_widths(cls, lch_unit, top_ext_info, bot_ext_info):
        # type: (int, ExtInfo, ExtInfo) -> List[int]

        # get implant spacing info
        imp_sp_info = _config['mos_analog']['imp_spaces']

        # get transistor constants
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        w_conn_d = mos_constants['w_conn_d']

        # determine wire line-end spacing
        mos_conn_layer = cls.get_mos_conn_layer()
        mx_sp = 0
        for mx_id in range(1, mos_conn_layer + 1):
            lay_type = TechInfoSOIGeneric.layer_id_to_type(mx_id)
            mx_w = w_conn_d[mx_id]
            mx_sp = max(mx_sp, TechInfoSOIGeneric.get_min_line_end_space_unit(lay_type, mx_w))

        # get minimum extension width from wire line-end spacing
        w_min = max(0, mx_sp - top_ext_info.mx_margin - bot_ext_info.mx_margin)

        # update minimum extension width from implant spacing
        bot_imp_info = bot_ext_info.imp_margins
        top_imp_info = top_ext_info.imp_margins
        for bot_name, bot_margin in bot_imp_info.items():
            for top_name, top_margin in top_imp_info.items():
                if top_name == bot_name:
                    if bot_name in imp_sp_info:
                        # check same implant spacing
                        imp_sp, join_flag = imp_sp_info[bot_name]
                        if join_flag == 0:
                            w_min = max(w_min, imp_sp - bot_margin - top_margin)
                else:
                    key = (top_name, bot_name) if top_name < bot_name else (bot_name, top_name)
                    if key in imp_sp_info:
                        imp_sp = imp_sp_info[key]
                        w_min = max(w_min, imp_sp - bot_margin - top_margin)

        return [w_min]

    @classmethod
    def get_ext_info(cls, lch_unit, w, fg, top_ext_info, bot_ext_info):
        # type: (int, int, int, ExtInfo, ExtInfo) -> Dict[str, Any]

        # get implant spacing info
        imp_sp_info = _config['mos_analog']['imp_spaces']

        # get transistor constants
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        # extension block just have implant layers that needs to be merged
        # find those implant layers
        imp_info = {}
        bot_imp_info = bot_ext_info.imp_margins
        top_imp_info = top_ext_info.imp_margins
        for bot_name, bot_margin in bot_imp_info.items():
            if bot_name in top_imp_info and bot_name in imp_sp_info:
                top_margin = top_imp_info[bot_name]
                imp_sp, join_flag = imp_sp_info[bot_name]
                if join_flag == 2 or join_flag == 1 and bot_margin + top_margin + w < imp_sp:
                    imp_info[bot_name] = (-bot_margin, w + top_margin)

        layout_info = dict(
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, w),
            imp_info=imp_info,

            blk_type='ext',
        )
        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
        )

    @classmethod
    def get_sub_ring_ext_info(cls, sub_type, height, fg, end_ext_info, **kwargs):
        # type: (str, int, int, Any, **kwargs) -> Dict[str, Any]
        raise NotImplementedError('Not implemented yet')

    @classmethod
    def get_substrate_info(cls, lch_unit, w, sub_type, threshold, fg, blk_pitch=1, **kwargs):
        # type: (int, int, str, str, int, int, **kwargs) -> Dict[str, Any]

        # convert width to resolution units
        layout_unit = _config['layout_unit']
        res = _config['resolution']
        w_unit = int(round(w / layout_unit / (2 * res))) * 2

        # get transistor constants
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        sub_m1_extx = mos_constants['sub_m1_extx']

        # substrate row is just a slab of M1
        m1_type = TechInfoSOIGeneric.layer_id_to_type(1)
        m1_sp = TechInfoSOIGeneric.get_min_space_unit(m1_type, w_unit)

        # compute extension information
        height = w_unit + 2 * m1_sp
        height = -(-height // blk_pitch) * blk_pitch
        m1_yb = (height - w_unit) // 2
        m1_yt = height - m1_yb
        ext_top_info = ExtInfo(mx_margin=m1_yb, imp_margins={}, mtype=sub_type, thres=threshold)
        ext_bot_info = ExtInfo(mx_margin=m1_yb, imp_margins={}, mtype=sub_type, thres=threshold)

        layout_info = dict(
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, height),
            imp_info={},
            mx_y=(m1_yb, m1_yt),
            m1_extx=sub_m1_extx,
            port_name='VDD' if sub_type == 'ntap' else 'VSS',

            blk_type='sub',
        )

        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            sd_yc=height // 2,
            left_edge_info=None,
            right_edge_info=None,
        )

    @classmethod
    def get_analog_end_info(cls, lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]

        # get transistor constants
        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        # analog_end is empty
        layout_info = dict(
            lch_unit=lch_unit,
            sd_pitch=sd_pitch,
            fg=fg,
            arr_y=(0, 0),
            imp_info={},

            blk_type='end',
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=None,
            right_edge_info=None,
        )

    @classmethod
    def get_sub_ring_end_info(cls, sub_type, threshold, fg, end_ext_info, **kwargs):
        # type: (str, str, int, Any, **kwargs) -> Dict[str, Any]
        raise NotImplementedError('Not implemented yet.')

    @classmethod
    def get_outer_edge_info(cls, guard_ring_nf, layout_info, is_end, adj_blk_info):
        # type: (int, Dict[str, Any], bool, Optional[Any]) -> Dict[str, Any]

        # outer edge is empty
        return dict(
            lch_unit=layout_info['lch_unit'],
            sd_pitch=layout_info['sd_pitch'],
            fg=0,
            arr_y=layout_info['arr_y'],
            imp_info={},

            blk_type='edge',
        )

    @classmethod
    def get_gr_sub_info(cls, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]
        raise ValueError('Guard ring is not supported in this technology.')

    @classmethod
    def get_gr_sep_info(cls, layout_info, adj_blk_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]
        raise ValueError('Guard ring is not supported in this technology.')

    @classmethod
    def draw_mos(cls, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor geometries.

        Note: because body-connected transistors in SOI technology cannot be
        drawn on the same row, this method only draws implant and threshold
        layers.  The actual transistor is drawn in draw_mos_connection() method.

        For substrate row, this method just draws a large M1 rectangle to short
        all body connections together.
        """

        res = _config['resolution']

        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        arr_yb, arr_yt = layout_info['arr_y']
        imp_info = layout_info['imp_info']
        blk_type = layout_info['blk_type']

        # compute array/bounding box size
        width = fg * sd_pitch
        template.array_box = BBox(0, arr_yb, width, arr_yt, res, unit_mode=True)
        template.prim_bound_box = template.array_box
        template.prim_top_layer = cls.get_mos_conn_layer()

        # draw implant layers
        for imp_name, (imp_yb, imp_yt) in imp_info.items():
            imp_box = BBox(0, imp_yb, width, imp_yt, res, unit_mode=True)
            template.add_rect(imp_name, imp_box)

        if blk_type == 'sub':
            # draw M1 rectangle for substrate contact
            m1_extx = layout_info['m1_extx']
            mx_yb, mx_yt = layout_info['mx_y']
            sub_box = BBox(-m1_extx, mx_yb, width + m1_extx, mx_yt, res, unit_mode=True)
            m1_name = _config['layer_name'][1]
            template.add_rect(m1_name, sub_box)

    @classmethod
    def draw_substrate_connection(cls, template, layout_info, port_tracks, dum_tracks, dummy_only,
                                  is_laygo, is_guardring):
        # type: (TemplateBase, Dict[str, Any], List[int], List[int], bool, bool, bool) -> bool

        # note: we just export all tracks, as we can draw wires on all of them
        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        mx_yb, mx_yt = layout_info['mx_y']
        port_name = layout_info['port_name']

        if mx_yt > mx_yb:
            # add pins
            dum_conn_layer = cls.get_dum_conn_layer()
            dum_warrs = cls._get_wire_array(dum_conn_layer, -0.5, fg + 1, mx_yb, mx_yt)
            template.add_pin(port_name, dum_warrs, show=False)

            if dummy_only:
                end_layer = dum_conn_layer
            else:
                end_layer = cls.get_mos_conn_layer()
                mos_warrs = cls._get_wire_array(end_layer, -0.5, fg + 1, mx_yb, mx_yt)
                template.add_pin(port_name, mos_warrs, show=False)

            # draw vias
            cls._draw_vertical_vias(template, lch_unit, 0, fg + 1, sd_pitch, mx_yb, mx_yt, 1, end_layer=end_layer)

            return True

        return False

    @classmethod
    def draw_mos_connection(cls, template, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                            min_ds_cap, is_diff, diode_conn, options):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, Dict[str, Any]) -> None

        # note: ignore gate_ext_mode, min_ds_cap, is_diff
        if diode_conn:
            raise ValueError('Diode connection not supported yet.')

        res = _config['resolution']
        via_id_table = _config['via_id']
        mos_lay_table = _config['mos_layer_table']
        lay_name_table = _config['layer_name']
        mos_analog_config = _config['mos_analog']

        gate_layers = mos_analog_config['gate_layers']
        via_g = mos_analog_config['via_g']
        via_b = mos_analog_config['via_b']

        layout_info = mos_info['layout_info']
        sd_yc = mos_info['sd_yc']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        b_po_y_list = layout_info['b_po_y_list']
        g_y_list = layout_info['g_y_list']
        d_y_list = layout_info['d_y_list']
        b_y_list = layout_info['b_y_list']

        mos_constants = cls.get_mos_tech_constants(lch_unit)
        w_conn_g = mos_constants['w_conn_g']
        w_conn_d = mos_constants['w_conn_d']
        w_delta = mos_constants['w_delta']
        w_body = mos_constants['w_body']
        body_po_extx = mos_constants['body_po_extx']
        num_via_body = mos_constants['num_via_body']

        width = fg * sd_pitch

        # draw OD
        # draw main OD
        od_name = mos_lay_table['OD']
        w_od = w_conn_d[0]
        d_od_yb, d_od_yt = d_y_list[0]
        d_od_yt += w_delta
        od_box = BBox(-w_od // 2, d_od_yb, width + w_od // 2, d_od_yt, res, unit_mode=True)
        od_box = od_box.move_by(dy=-sd_yc, unit_mode=True)
        template.add_rect(od_name, od_box)
        # draw body OD
        b_od_yb = d_od_yt
        b_od_yt = b_y_list[0][1]
        b_od_xc = sd_pitch // 2
        od_box = BBox(-w_body // 2, b_od_yb, w_body // 2, b_od_yt, res, unit_mode=True)
        od_box = od_box.move_by(dx=b_od_xc, dy=-sd_yc, unit_mode=True)
        template.add_rect(od_name, od_box, nx=fg, spx=sd_pitch * res)

        # draw PO
        # draw gate PO bar
        po_name = mos_lay_table['PO']
        po_xc = sd_pitch // 2
        po_yb, po_yt = g_y_list[0]
        g_po_yc = (po_yb + po_yt) // 2
        po_box = BBox(po_xc - lch_unit // 2, po_yb, po_xc + (fg - 1) * sd_pitch + lch_unit // 2,
                      po_yt, res, unit_mode=True)
        po_box = po_box.move_by(dy=-sd_yc, unit_mode=True)
        template.add_rect(po_name, po_box)
        # draw gate PO
        po_yb, po_yt = po_yt, d_y_list[0][1]
        po_box = BBox(-lch_unit // 2, po_yb, lch_unit // 2, po_yt, res, unit_mode=True)
        po_box = po_box.move_by(dx=po_xc, dy=-sd_yc, unit_mode=True)
        template.add_rect(po_name, po_box, nx=fg, spx=sd_pitch * res)
        # draw gate layers
        po_yb = d_y_list[0][0]
        gate_box = BBox(po_box.left_unit, po_yb, po_box.right_unit, po_yt, res, unit_mode=True)
        gate_box = gate_box.move_by(dy=-sd_yc, unit_mode=True)
        for gate_lay in gate_layers:
            template.add_rect(gate_lay, gate_box, nx=fg, spx=sd_pitch * res)
        # draw body PO bar
        po_xl = -body_po_extx
        po_xr = width + body_po_extx
        po_yb, po_yt = b_po_y_list[0]
        po_box = BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True)
        po_box = po_box.move_by(dy=-sd_yc, unit_mode=True)
        template.add_rect(po_name, po_box)

        # draw gate connections
        # draw via to M1
        m1_name = lay_name_table[1]
        via_xc = width // 2
        via_yc = g_po_yc
        via_w, via_h = via_g['dim'][0]
        via_type = via_id_table[(po_name, m1_name)]
        via_enc_le = via_g['bot_enc_le'][0]
        m1_w_g = w_conn_g[1]
        via_enc1 = (w_conn_g[0] - via_h) // 2
        via_enc2 = (m1_w_g - via_h) // 2
        enc1 = [via_enc_le, via_enc_le, via_enc1, via_enc1]
        enc2 = [via_enc_le, via_enc_le, via_enc2, via_enc2]
        template.add_via_primitive(via_type, loc=[via_xc, via_yc - sd_yc], num_cols=fg - 1, sp_cols=sd_pitch - via_w,
                                   enc1=enc1, enc2=enc2, cut_width=via_w, cut_height=via_h, unit_mode=True)
        # draw gate M1 bar
        m1_bbox = BBox(0, -m1_w_g // 2, width, m1_w_g // 2, res, unit_mode=True)
        m1_bbox = m1_bbox.move_by(dy=via_yc - sd_yc, unit_mode=True)
        template.add_rect(m1_name, m1_bbox)

        # figure out gate location and number of gate wires
        if sdir == 0:
            g_x0 = sd_pitch
        elif ddir == 0:
            g_x0 = 0
        else:
            g_x0 = 0 if gate_pref_loc == 's' else sd_pitch

        num_s = fg // 2 + 1
        num_d = (fg + 1) // 2
        num_g = num_s if g_x0 == 0 else num_d

        # draw gate M1 to M2 vias
        m2_name = lay_name_table[2]
        m2_w = w_conn_g[2]
        via_w, via_h = via_g['dim'][1]
        via_enc_le1 = via_g['bot_enc_le'][1]
        via_enc_top2 = via_g['top_enc_le'][1]
        via_enc_bot2 = via_yc - via_h // 2 - g_y_list[2][0]
        via_enc1 = (m1_w_g - via_h) // 2
        via_enc2 = (m2_w - via_w) // 2
        enc1 = [via_enc_le1, via_enc_le1, via_enc1, via_enc1]
        enc2 = [via_enc2, via_enc2, via_enc_top2, via_enc_bot2]
        via_type = via_id_table[(m1_name, m2_name)]
        template.add_via_primitive(via_type, loc=[g_x0, via_yc - sd_yc], enc1=enc1, enc2=enc2, cut_width=via_w,
                                   cut_height=via_h, nx=num_g, spx=2 * sd_pitch, unit_mode=True)
        # draw gate vias to connection layer
        cls._draw_vertical_vias(template, lch_unit, g_x0, num_g, 2 * sd_pitch,
                                g_y_list[2][0] - sd_yc, g_y_list[2][1] - sd_yc, 2)

        # draw drain vias to connection layer
        d_yb, d_yt = d_y_list[-1]
        cls._draw_vertical_vias(template, lch_unit, 0, fg + 1, sd_pitch,
                                d_yb - sd_yc, d_yt - sd_yc, 0)

        # draw body vias to M1
        via_type = via_id_table[(od_name, m1_name)]
        via_w, via_h = via_b['dim']
        via_sp = via_b['sp']
        enc1 = via_b['enc1']
        enc2 = via_b['enc2']
        b_m1_yb, b_m1_yt = b_y_list[1]
        via_yc = (b_m1_yb + b_m1_yt) // 2
        template.add_via_primitive(via_type, loc=[sd_pitch // 2, via_yc - sd_yc], num_rows=num_via_body,
                                   sp_rows=via_sp, enc1=enc1, enc2=enc2, cut_width=via_w,
                                   cut_height=via_h, nx=fg, spx=sd_pitch, unit_mode=True)
        # draw body M1 bar
        m1_box = BBox(0, b_m1_yb, width, b_m1_yt, res, unit_mode=True)
        m1_box = m1_box.move_by(dy=-sd_yc, unit_mode=True)
        template.add_rect(m1_name, m1_box)

        # add ports
        mos_conn_layer = cls.get_mos_conn_layer()
        gtr0 = template.grid.coord_to_track(mos_conn_layer, g_x0, unit_mode=True)
        g_yb, g_yt = g_y_list[-1]
        g_warrs = cls._get_wire_array(mos_conn_layer, gtr0, num_g, g_yb - sd_yc, g_yt - sd_yc, pitch=2)
        s_warrs = cls._get_wire_array(mos_conn_layer, -0.5, num_s, d_yb - sd_yc, d_yt - sd_yc, pitch=2)
        d_warrs = cls._get_wire_array(mos_conn_layer, 0.5, num_d, d_yb - sd_yc, d_yt - sd_yc, pitch=2)

        template.add_pin('g', g_warrs, show=False)
        template.add_pin('d', d_warrs, show=False)
        template.add_pin('s', s_warrs, show=False)

    @classmethod
    def draw_dum_connection(cls, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[int], Dict[str, Any]) -> None

        res = _config['resolution']
        mos_lay_table = _config['mos_layer_table']
        lay_name_table = _config['layer_name']

        layout_info = mos_info['layout_info']
        sd_yc = mos_info['sd_yc']

        lch_unit = layout_info['lch_unit']
        sd_pitch = layout_info['sd_pitch']
        fg = layout_info['fg']
        b_po_y_list = layout_info['b_po_y_list']
        g_y_list = layout_info['g_y_list']
        b_y_list = layout_info['b_y_list']

        mos_constants = cls.get_mos_tech_constants(lch_unit)
        sp_gb_po = mos_constants['sp_gb_po']

        width = fg * sd_pitch

        # get dummy PO X coordinates
        po_x_list = []
        if (edge_mode & 1) == 0:
            po_x_list.append(sd_pitch // 2)
        if (edge_mode & 2) == 0:
            po_x_list.append(width - sd_pitch // 2)

        if po_x_list:
            # draw dummy PO
            po_name = mos_lay_table['PO']
            po_yb = g_y_list[0][0]
            po_yt = b_po_y_list[0][0] - sp_gb_po
            po_box = BBox(-lch_unit // 2, po_yb, lch_unit // 2, po_yt, res, unit_mode=True)
            po_box = po_box.move_by(dy=-sd_yc, unit_mode=True)
            for po_xc in po_x_list:
                po_box = po_box.move_by(dx=po_xc - po_box.xc_unit, unit_mode=True)
                template.add_rect(po_name, po_box)

        # draw body M1
        m1_name = lay_name_table[1]
        m1_yb, m1_yt = b_y_list[1]
        m1_box = BBox(0, m1_yb, width, m1_yt, res, unit_mode=True)
        m1_box = m1_box.move_by(dy=-sd_yc, unit_mode=True)
        template.add_rect(m1_name, m1_box)

        # add body ports
        dum_layer = cls.get_dum_conn_layer()
        for tidx in gate_tracks:
            warr = WireArray(TrackID(dum_layer, tidx), m1_box.bottom, m1_box.top)
            template.add_pin('dummy', warr, show=False)

    @classmethod
    def _draw_vertical_vias(cls, template, lch_unit, x0, num, pitch, mx_yb, mx_yt, start_layer, end_layer=None):
        # type: (TemplateBase, int, int, int, int, int, int, int, int) -> None

        via_d = _config['mos_analog']['via_d']
        via_id_table = _config['via_id']
        mos_lay_table = _config['mos_layer_table']
        lay_name_table = _config['layer_name']

        mos_constants = cls.get_mos_tech_constants(lch_unit)
        w_conn_d = mos_constants['w_conn_d']

        if end_layer is None:
            end_layer = cls.get_mos_conn_layer()

        mx_h = mx_yt - mx_yb
        mx_yc = (mx_yt + mx_yb) // 2
        for bot_lay_id in range(start_layer, end_layer):
            if bot_lay_id == 0:
                od_name = mos_lay_table['OD']
                m1_name = lay_name_table[1]
                via_type = via_id_table[(od_name, m1_name)]
                via_enc_le = via_d['top_enc_le'][bot_lay_id]
            else:
                via_type = via_id_table[(lay_name_table[bot_lay_id], lay_name_table[bot_lay_id + 1])]
                via_enc_le = max(via_d['bot_enc_le'][bot_lay_id], via_d['top_enc_le'][bot_lay_id])

            w_bot = w_conn_d[bot_lay_id]
            w_top = w_conn_d[bot_lay_id + 1]

            via_w, via_h = via_d['dim'][bot_lay_id]
            via_sp = via_d['sp'][bot_lay_id]

            num_via = (mx_h - 2 * via_enc_le + via_sp) // (via_w + via_sp)
            via_harr = num_via * (via_w + via_sp) - via_sp
            via_enc_le = (mx_h - via_harr) // 2
            via_enc1 = (w_bot - via_w) // 2
            via_enc2 = (w_top - via_w) // 2

            enc1 = [via_enc1, via_enc1, via_enc_le, via_enc_le]
            enc2 = [via_enc2, via_enc2, via_enc_le, via_enc_le]
            template.add_via_primitive(via_type, loc=[x0, mx_yc], num_rows=num_via, sp_rows=via_sp,
                                       enc1=enc1, enc2=enc2, cut_width=via_w, cut_height=via_h,
                                       nx=num, spx=pitch, unit_mode=True)

    @classmethod
    def _get_wire_array(cls, layer_id, tr0, num, lower, upper, pitch=1):
        res = _config['resolution']
        tid = TrackID(layer_id, tr0, num=num, pitch=pitch)
        return WireArray(tid, lower * res, upper * res)

    @classmethod
    def draw_decap_connection(cls, template, mos_info, sdir, ddir, gate_ext_mode, export_gate, options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
        raise ValueError('Decap connection is not supported in this technology.')
