# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Tuple, Union

import abc

from bag import float_to_si_string
from bag.layout.routing.fill import fill_symmetric_max_density

from .tech import LaygoTech
from ..analog_mos.planar import ExtInfo, RowInfo, EdgeInfo, MOSTechPlanarGeneric

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig
    from bag.layout.routing import WireArray
    from bag.layout.template import TemplateBase


class LaygoTechPlanarBase(MOSTechPlanarGeneric, LaygoTech, metaclass=abc.ABCMeta):
    """Base class for implementations of LaygoTech in Finfet technologies.

    This class for now handles all DRC rules and drawings related to PO, OD, CPO,
    and MD. The rest needs to be implemented by subclasses.

    Parameters
    ----------
    config : Dict[str, Any]
        the technology configuration dictionary.
    tech_info : TechInfo
        the TechInfo object.
    mos_entry_name : str
        name of the entry that contains technology parameters for transistors in
        the given configuration dictionary.
    """

    def __init__(self, config, tech_info, mos_entry_name='mos'):
        # type: (Dict[str, Any], TechInfoConfig, str) -> None
        LaygoTech.__init__(self, config, tech_info, mos_entry_name=mos_entry_name)

    def get_laygo_row_yloc_info(self, lch_unit, w, is_sub, **kwargs):
        # type: (int, int, bool, **kwargs) -> Dict[str, Any]
        analog = kwargs.pop('analog', False)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        od_spy = mos_constants['od_spy']
        po_spy = mos_constants['po_spy']

        g_conn_info = self.get_conn_drc_info(lch_unit, 'g', is_laygo=True)
        g_m1_sple = g_conn_info[1]['sp_le']

        if is_sub:
            yloc_info = self.get_sub_yloc_info(lch_unit, w, **kwargs)
        else:
            yloc_info = self.get_mos_yloc_info(lch_unit, w, **kwargs)

        blk_yb, blk_yt = blk_y = yloc_info['blk']
        po_yb, po_yt = po_y = yloc_info['po']
        od_yb, od_yt = od_y = yloc_info['od']

        # get wire coordinates
        conn_yloc_info = self.get_laygo_conn_yloc_info(lch_unit, od_y, is_sub)
        d_yb, d_yt = d_y = conn_yloc_info['d_y']
        g_yb, g_yt = g_y = conn_yloc_info['g_y']

        # step 2: compute top CPO location.
        return dict(
            blk=blk_y,
            po=po_y,
            od=od_y,
            top_margins=dict(
                od=(blk_yt - od_yt, od_spy),
                po=(blk_yt - po_yt, po_spy),
                m1=(blk_yt - d_yt, g_m1_sple),
            ),
            bot_margins=dict(
                od=(od_yb - blk_yb, od_spy),
                po=(blk_yt - po_yt, po_spy),
                m1=(g_yb - blk_yb, g_m1_sple),
            ),
            fill_info={},
            g_conn_y=g_y,
            gb_conn_y=d_y,
            ds_conn_y=d_y,
        )

    @abc.abstractmethod
    def get_laygo_blk_yloc_info(self, w, blk_type, row_info, **kwargs):
        # type: (int, str, Dict[str, Any], **kwargs) -> Dict[str, Any]
        """Computes Y coordinates of various layers in the laygo block.

        The returned dictionary should have the following entries:

        od :
            a tuple of OD bottom/top Y coordinates.
        md :
            a tuple of MD bottom/top Y coordinates.
        """
        return {}

    @abc.abstractmethod
    def draw_laygo_g_connection(self, template, mos_info, g_loc, num_fg, **kwargs):
        # type: (TemplateBase, Dict[str, Any], str, int, **kwargs) -> List[WireArray]
        """Draw laygo gate connections.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the block layout information dictionary.
        g_loc : str
            gate wire alignment location.  Either 'd' or 's'.
        num_fg : int
            number of gate fingers.
        **kwargs :
            optional parameters.

        Returns
        -------
        warr_list : List[WireArray]
            list of port wires as single-wire WireArrays.
        """
        return []

    @abc.abstractmethod
    def draw_laygo_ds_connection(self, template, mos_info, tidx_list, **kwargs):
        # type: (TemplateBase, Dict[str, Any], List[Union[float, int]], **kwargs) -> List[WireArray]
        """Draw laygo drain/source connections.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the block layout information dictionary.
        tidx_list : List[Union[float, int]]
            list of track index to draw drain/source wires.
        **kwargs :
            optional parameters.

        Returns
        -------
        warr_list : List[WireArray]
            list of port wires as single-wire WireArrays.
        """
        return []

    @abc.abstractmethod
    def draw_laygo_sub_connection(self, template, mos_info, **kwargs):
        # type: (TemplateBase, Dict[str, Any], **kwargs) -> List[WireArray]
        """Draw laygo substrate connections.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the block layout information dictionary.
        **kwargs :
            optional parameters.

        Returns
        -------
        warr_list : List[WireArray]
            list of port wires as single-wire WireArrays.
        """
        return []

    def get_default_end_info(self):
        # type: () -> Any
        return EdgeInfo(od_type=None, draw_layers={}, y_intv={}), []

    def get_laygo_mos_row_info(self,  # type: LaygoTechFinfetBase
                               lch_unit,  # type: int
                               w_max,  # type: int
                               w_sub,  # type: int
                               mos_type,  # type: str
                               threshold,  # type: str
                               bot_row_type,  # type: str
                               top_row_type,  # type: str
                               **kwargs):
        # type: (...) -> Dict[str, Any]

        # figure out various properties of the current laygo block
        is_sub = (mos_type == 'ptap' or mos_type == 'ntap')
        sub_type = 'ptap' if mos_type == 'nch' or mos_type == 'ptap' else 'ntap'

        # get Y coordinate information dictionary
        row_yloc_info = self.get_laygo_row_yloc_info(lch_unit, w_max, is_sub, **kwargs)
        blk_yb, blk_yt = row_yloc_info['blk']
        po_yloc = row_yloc_info['po']
        od_yloc = row_yloc_info['od']
        md_yloc = row_yloc_info['md']
        top_margins = row_yloc_info['top_margins']
        bot_margins = row_yloc_info['bot_margins']
        fill_info = row_yloc_info['fill_info']
        g_conn_y = row_yloc_info['g_conn_y']
        gb_conn_y = row_yloc_info['gb_conn_y']
        ds_conn_y = row_yloc_info['ds_conn_y']

        # compute extension information
        mtype = (mos_type, mos_type)
        po_type = 'PO_sub' if is_sub else 'PO'
        po_types = (po_type, po_type)
        lr_edge_info = EdgeInfo(od_type='sub' if is_sub else 'mos', draw_layers={}, y_intv={})
        ext_top_info = ExtInfo(margins=top_margins,
                               od_h=w_max,
                               imp_min_h=0,
                               mtype=mtype,
                               thres=threshold,
                               po_types=po_types,
                               edgel_info=lr_edge_info,
                               edger_info=lr_edge_info,
                               )
        ext_bot_info = ExtInfo(margins=bot_margins,
                               od_h=w_max,
                               imp_min_h=0,
                               mtype=mtype,
                               thres=threshold,
                               po_types=po_types,
                               edgel_info=lr_edge_info,
                               edger_info=lr_edge_info,
                               )

        lay_info_list = [(lay, 0, blk_yb, blk_yt)
                         for lay in self.get_mos_layers(mos_type, threshold)]
        imp_params = [(mos_type, threshold, blk_yb, blk_yt, blk_yb, blk_yt)],

        fill_info_list = [FillInfo(layer=layer, exc_layer=info[0], x_intv_list=[],
                                   y_intv_list=info[1]) for layer, info in fill_info.items()]

        blk_y = (blk_yb, blk_yt)
        lch = lch_unit * self.res * self.tech_info.layout_unit
        lch_str = float_to_si_string(lch)
        row_name_id = '%s_l%s_w%d_%s' % (mos_type, lch_str, w_max, threshold)
        return dict(
            w_max=w_max,
            w_sub=w_sub,
            lch_unit=lch_unit,
            row_type=mos_type,
            sub_type=sub_type,
            threshold=threshold,
            arr_y=blk_y,
            od_y=od_yloc,
            po_y=po_yloc,
            md_y=md_yloc,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            lay_info_list=lay_info_list,
            imp_params=imp_params,
            fill_info_list=fill_info_list,
            g_conn_y=g_conn_y,
            gb_conn_y=gb_conn_y,
            ds_conn_y=ds_conn_y,
            row_name_id=row_name_id,
        )

    def get_laygo_sub_row_info(self, lch_unit, w, mos_type, threshold, **kwargs):
        # type: (int, int, str, str, **kwargs) -> Dict[str, Any]
        return self.get_laygo_mos_row_info(lch_unit, w, w, mos_type, threshold, '', '', **kwargs)

    def get_laygo_blk_info(self, blk_type, w, row_info, **kwargs):
        # type: (str, int, Dict[str, Any], **kwargs) -> Dict[str, Any]

        arr_y = row_info['arr_y']
        po_y = row_info['po_y']
        lch_unit = row_info['lch_unit']
        row_type = row_info['row_type']
        sub_type = row_info['sub_type']
        threshold = row_info['threshold']
        row_ext_top = row_info['ext_top_info']
        row_ext_bot = row_info['ext_bot_info']
        lay_info_list = row_info['lay_info_list']
        imp_params = row_info['imp_params']
        fill_info_list = row_info['fill_info_list']

        # get Y coordinate information dictionary
        yloc_info = self.get_laygo_blk_yloc_info(w, blk_type, row_info, **kwargs)
        od_yloc = yloc_info['od']
        md_yloc = yloc_info['md']

        # figure out various properties of the current laygo block
        is_sub = (row_type == sub_type)
        y_intv = dict(od=od_yloc, md=md_yloc)
        if blk_type.startswith('fg1'):
            mtype = (row_type, row_type)
            od_type = 'mos'
            fg = 1
            od_intv = (0, 1)
            edgel_info = edger_info = EdgeInfo(od_type=od_type, draw_layers={}, y_intv=y_intv)
            po_types = ('PO',)
        elif blk_type == 'sub':
            mtype = (sub_type, row_type)
            od_type = 'sub'
            if is_sub:
                fg = 2
                od_intv = (0, 2)
                edgel_info = edger_info = EdgeInfo(od_type=od_type, draw_layers={}, y_intv=y_intv)
                po_types = ('PO_sub', 'PO_sub')
            else:
                mos_constants = self.get_mos_tech_constants(lch_unit)
                imp_od_ency = mos_constants['imp_od_ency']
                imp_yb = od_yloc[0] - imp_od_ency
                imp_yt = od_yloc[1] + imp_od_ency
                arr_yb, arr_yt = arr_y
                if kwargs.get('imp_min_g', False):
                    row_yb = arr_yb
                    row_yt = max(imp_yb, arr_yb)
                    sub_yb = row_yt
                    sub_yt = arr_yt
                elif kwargs.get('imp_min_d', False):
                    sub_yb = arr_yb
                    sub_yt = min(imp_yt, arr_yt)
                    row_yb = sub_yt
                    row_yt = arr_yt
                else:
                    row_yb = row_yt = arr_yb
                    sub_yb = arr_yb
                    sub_yt = arr_yt
                lay_info_list = [(lay, 0, sub_yb, sub_yt)
                                 for lay in self.get_mos_layers(sub_type, threshold)]
                if row_yt > row_yb:
                    lay_info_list.extend((lay, 0, row_yb, row_yt)
                                         for lay in self.get_mos_layers(row_type, threshold))
                fg = self.get_sub_columns(lch_unit)
                od_intv = (2, fg - 2)
                edgel_info = edger_info = EdgeInfo(od_type=None, draw_layers={}, y_intv=y_intv)
                po_types = ('PO_dummy', 'PO_edge_sub') + ('PO_sub',) * (fg - 4) + \
                           ('PO_edge_sub', 'PO_dummy',)
        else:
            mtype = (row_type, row_type)
            od_type = 'mos'
            fg = 2
            od_intv = (0, 2)
            edgel_info = edger_info = EdgeInfo(od_type=od_type, draw_layers={}, y_intv=y_intv)
            po_types = ('PO', 'PO')

        # update extension information
        # noinspection PyProtectedMember
        ext_top_info = row_ext_top._replace(mtype=mtype, po_types=po_types,
                                            edgel_info=edgel_info, edger_info=edger_info)
        # noinspection PyProtectedMember
        ext_bot_info = row_ext_bot._replace(mtype=mtype, po_types=po_types,
                                            edgel_info=edgel_info, edger_info=edger_info)

        layout_info = dict(
            is_sub_row=is_sub,
            blk_type='sub' if is_sub else 'mos',
            lch_unit=lch_unit,
            fg=fg,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=[RowInfo(od_x_list=[od_intv],
                                   od_y=od_yloc,
                                   od_type=(od_type, sub_type),
                                   row_y=arr_y,
                                   po_y=po_y,
                                   md_y=md_yloc), ],
            lay_info_list=lay_info_list,
            fill_info_list=fill_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=imp_params,
            is_sub_ring=False,
            dnw_mode='',
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
            left_edge_info=(edgel_info, []),
            right_edge_info=(edger_info, []),
        )

    def get_laygo_end_info(self, lch_unit, mos_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        return self.get_analog_end_info(lch_unit, mos_type, threshold, fg, is_end,
                                        blk_pitch, **kwargs)

    def get_laygo_space_info(self, row_info, num_blk, left_blk_info, right_blk_info):
        # type: (Dict[str, Any], int, Any, Any) -> Dict[str, Any]

        od_y = row_info['od_y']
        po_y = row_info['po_y']
        md_y = row_info['md_y']
        arr_y = row_info['arr_y']
        lch_unit = row_info['lch_unit']
        row_type = row_info['row_type']
        sub_type = row_info['sub_type']
        row_ext_top = row_info['ext_top_info']
        row_ext_bot = row_info['ext_bot_info']
        lay_info_list = row_info['lay_info_list']
        fill_info_list = row_info['fill_info_list']
        imp_params = row_info['imp_params']

        is_sub = (row_type == sub_type)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        od_spx = mos_constants['od_spx']
        od_fill_w_max = mos_constants['od_fill_w_max']

        od_spx_fg = -(-(od_spx - sd_pitch + lch_unit) // sd_pitch) + 2

        # get OD fill X interval
        area = num_blk - 2 * od_spx_fg
        if area > 0:
            if od_fill_w_max is None:
                od_x_list = [(od_spx_fg, num_blk - od_spx_fg)]
            else:
                od_fg_max = (od_fill_w_max - lch_unit) // sd_pitch - 1
                od_x_list = fill_symmetric_max_density(area, area, 2, od_fg_max, od_spx_fg,
                                                       offset=od_spx_fg, fill_on_edge=True,
                                                       cyclic=False)[0]
        else:
            od_x_list = []
        # get row OD list
        row_info_list = [RowInfo(od_x_list=od_x_list, od_y=od_y, od_type=('dum', sub_type),
                                 row_y=arr_y, po_y=po_y, md_y=md_y), ]

        # update extension information
        cur_edge_info = EdgeInfo(od_type=None, draw_layers={}, y_intv=dict(od=od_y, md=md_y))
        # figure out poly types per finger
        po_types = []
        od_intv_idx = 0
        for cur_idx in range(num_blk):
            if cur_idx == 0 or cur_idx == num_blk - 1:
                od_type = left_blk_info[0].od_type if cur_idx == 0 else right_blk_info[0].od_type
                if od_type == 'mos':
                    po_types.append('PO_edge')
                elif od_type == 'sub':
                    po_types.append('PO_edge_sub')
                elif od_type == 'dum':
                    po_types.append('PO_edge_dummy')
                else:
                    po_types.append('PO_dummy')
            elif cur_idx < od_spx_fg or cur_idx >= num_blk - od_spx_fg:
                po_types.append('PO_dummy')
            elif od_intv_idx < len(od_x_list):
                cur_od_intv = od_x_list[od_intv_idx]
                if cur_od_intv[1] == cur_idx:
                    po_types.append('PO_edge_dummy')
                    od_intv_idx += 1
                elif cur_od_intv[0] <= cur_idx < cur_od_intv[1]:
                    po_types.append('PO_gate_dummy')
                elif cur_idx == cur_od_intv[0] - 1:
                    po_types.append('PO_edge_dummy')
                else:
                    if cur_idx > cur_od_intv[1]:
                        od_intv_idx += 1
                    po_types.append('PO_dummy')
            else:
                po_types.append('PO_dummy')

        # noinspection PyProtectedMember
        ext_top_info = row_ext_top._replace(po_types=po_types, edgel_info=cur_edge_info,
                                            edger_info=cur_edge_info)
        # noinspection PyProtectedMember
        ext_bot_info = row_ext_bot._replace(po_types=po_types, edgel_info=cur_edge_info,
                                            edger_info=cur_edge_info)

        lr_edge_info = (cur_edge_info, [])
        layout_info = dict(
            is_sub_row=is_sub,
            blk_type='sub' if is_sub else 'mos',
            lch_unit=lch_unit,
            fg=num_blk,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=lay_info_list,
            fill_info_list=fill_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=imp_params,
            is_sub_ring=False,
            dnw_mode='',
            # adjacent block information list
            adj_row_list=[],
            left_blk_info=left_blk_info[0],
            right_blk_info=right_blk_info[0],
        )

        # step 8: return results
        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=lr_edge_info,
            right_edge_info=lr_edge_info,
        )

    def get_row_extension_info(self,  # type: LaygoTechFinfetBase
                               bot_ext_list,  # type: List[Union[int, ExtInfo]]
                               top_ext_list,  # type: List[Union[int, ExtInfo]]
                               ):
        # type: (...) -> List[Tuple[int, int, ExtInfo, ExtInfo]]
        # merge list of bottom and top extension informations into a list of
        # bottom/top extension tuples
        bot_idx = top_idx = 0
        bot_len = len(bot_ext_list)
        top_len = len(top_ext_list)
        ext_groups = []
        cur_fg = bot_off = top_off = 0
        while bot_idx < bot_len and top_idx < top_len:
            bot_info = bot_ext_list[bot_idx]  # type: Union[int, ExtInfo]
            top_info = top_ext_list[top_idx]  # type: Union[int, ExtInfo]
            if isinstance(bot_info, int) and isinstance(top_info, int):
                cur_fg += bot_info
                bot_off = top_off = cur_fg
                bot_idx += 1
                top_idx += 1
            else:
                bot_ptype = bot_info.po_types
                top_ptype = top_info.po_types
                bot_stop = bot_off + len(bot_ptype)
                top_stop = top_off + len(top_ptype)
                stop_idx = min(bot_stop, top_stop)

                # create new bottom/top extension information objects for the
                # current overlapping block
                bot_po_types = bot_ptype[cur_fg - bot_off:stop_idx - bot_off]
                top_po_types = top_ptype[cur_fg - top_off:stop_idx - top_off]
                # noinspection PyProtectedMember
                cur_bot_info = bot_info._replace(po_types=bot_po_types)
                # noinspection PyProtectedMember
                cur_top_info = top_info._replace(po_types=top_po_types)
                # append tuples of current number of fingers and bottom/top
                # extension information object
                ext_groups.append((cur_fg, stop_idx - cur_fg, cur_bot_info, cur_top_info))

                cur_fg = stop_idx
                if stop_idx == bot_stop:
                    bot_off = cur_fg
                    bot_idx += 1
                if stop_idx == top_stop:
                    top_off = cur_fg
                    top_idx += 1

        return ext_groups

    def draw_laygo_connection(self, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None

        layout_info = mos_info['layout_info']
        sub_type = layout_info['sub_type']

        if blk_type in ('fg2d', 'fg2s', 'stack2d', 'stack2s', 'fg1d', 'fg1s'):
            g_loc = blk_type[-1]
            num_fg = int(blk_type[-2])
            if blk_type.startswith('fg2'):
                didx_list = [0.5]
                sidx_list = [-0.5, 1.5]
            elif blk_type.startswith('stack2'):
                didx_list = [1.5]
                sidx_list = [-0.5]
            else:
                didx_list = [0.5]
                sidx_list = [-0.5]
            g_warrs = self.draw_laygo_g_connection(template, mos_info, g_loc, num_fg, **options)
            d_warrs = self.draw_laygo_ds_connection(template, mos_info, didx_list, **options)
            s_warrs = self.draw_laygo_ds_connection(template, mos_info, sidx_list, **options)

            for name, warr_list in (('g', g_warrs), ('d', d_warrs), ('s', s_warrs)):
                template.add_pin(name, warr_list, show=False)
                if len(warr_list) > 1:
                    for idx, warr in enumerate(warr_list):
                        template.add_pin('%s%d' % (name, idx), warr, show=False)
        elif blk_type == 'sub':
            warrs = self.draw_laygo_sub_connection(template, mos_info, **options)
            port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
            s_warrs = warrs[0::2]
            d_warrs = warrs[1::2]
            template.add_pin(port_name, s_warrs, show=False)
            template.add_pin(port_name + '_s', s_warrs, show=False)
            template.add_pin(port_name + '_d', d_warrs, show=False)

        else:
            raise ValueError('Unsupported laygo primitive type: %s' % blk_type)
