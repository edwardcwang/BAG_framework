# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Tuple

import abc

from bag import float_to_si_string
from bag.layout.routing.fill import fill_symmetric_max_density

from .tech import LaygoTech
from ..analog_mos.finfet import MOSTechFinfetBase
from ..analog_mos.finfet import ExtInfo, RowInfo, EdgeInfo, FillInfo


if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig
    from bag.layout.template import TemplateBase


class LaygoTechFinfetBase(MOSTechFinfetBase, LaygoTech, metaclass=abc.ABCMeta):
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
        MOSTechFinfetBase.__init__(self, config, tech_info, mos_entry_name=mos_entry_name)

    @abc.abstractmethod
    def get_laygo_yloc_info(self, lch_unit, w, is_sub, **kwargs):
        # type: (int, int, bool, **kwargs) -> Dict[str, Any]
        """Computes Y coordinates of various layers in the laygo row.

        The returned dictionary should have the following entries:

        blk :
            a tuple of row bottom/top Y coordinates.
        od :
            a tuple of OD bottom/top Y coordinates.
        md :
            a tuple of MD bottom/top Y coordinates.
        top_margins :
            a dictionary of top extension margins and minimum space,
            which is ((blk_yt - lay_yt), spy) of each layer.
        bot_margins :
            a dictionary of bottom extension margins and minimum space,
            which is ((lay_yb - blk_yb), spy) of each layer.
        fill_info :
            a dictionary from metal layer tuple to tuple of exclusion
            layer name and list of metal fill Y intervals.
        """
        return {}

    @abc.abstractmethod
    def draw_laygo_space_geometries(self, template, space_info, left_blk_info, right_blk_info):
        # type: (TemplateBase, Dict[str, Any], Any, Any) -> None
        """Draw any geometries necessary in the given laygo space template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        space_info : Dict[str, Any]
            the layout information dictionary.
        left_blk_info : Any
            left block information.
        right_blk_info : Any
            right block information.
        """

    def get_default_end_info(self):
        # type: () -> Any
        return EdgeInfo(od_type=None, draw_layers={})

    def get_laygo_mos_info(self, lch_unit, w, mos_type, threshold, blk_type, bot_row_type, top_row_type, **kwargs):
        # type: (int, int, str, str, str, str, str, **kwargs) -> Dict[str, Any]

        mos_constants = self.get_mos_tech_constants(lch_unit)
        pode_is_poly = mos_constants['pode_is_poly']

        # figure out various properties of the current laygo block
        is_sub = (mos_type == 'ptap' or mos_type == 'ntap')
        sub_type = 'ptap' if mos_type == 'nch' or mos_type == 'ptap' else 'ntap'
        po_edge_code = 2 if pode_is_poly else 1
        if blk_type.startswith('fg1'):
            mtype = (mos_type, mos_type)
            od_type = 'mos'
            fg = 1
            od_intv = (0, 1)
            edgel_info = EdgeInfo(od_type=od_type, draw_layers={})
            edger_info = EdgeInfo(od_type=None, draw_layers={})
            po_types = (1, po_edge_code)
        elif blk_type == 'sub':
            mtype = (sub_type, mos_type)
            od_type = 'sub'
            if is_sub:
                fg = 2
                od_intv = (0, 2)
                edgel_info = edger_info = EdgeInfo(od_type=od_type, draw_layers={})
                po_types = (1, 1)
            else:
                fg = self.get_sub_columns(lch_unit) * 2
                od_intv = (2, fg - 2)
                edgel_info = edger_info = EdgeInfo(od_type=None, draw_layers={})
                po_types = (0, po_edge_code) + (1,) * (fg - 4) + (po_edge_code, 0,)
        else:
            mtype = (mos_type, mos_type)
            od_type = 'mos'
            fg = 2
            od_intv = (0, 2)
            edgel_info = edger_info = EdgeInfo(od_type=od_type, draw_layers={})
            po_types = (1, 1)

        # get Y coordinate information dictionary
        yloc_info = self.get_laygo_yloc_info(lch_unit, w, is_sub, **kwargs)
        blk_yb, blk_yt = yloc_info['blk']
        od_yloc = yloc_info['od']
        md_yloc = yloc_info['md']
        top_margins = yloc_info['top_margins']
        bot_margins = yloc_info['bot_margins']
        fill_info = yloc_info['fill_info']
        g_conn_y = yloc_info['g_conn_y']
        gb_conn_y = yloc_info['gb_conn_y']
        ds_conn_y = yloc_info['ds_conn_y']

        od_yc = (od_yloc[0] + od_yloc[1]) // 2

        # compute extension information
        ext_top_info = ExtInfo(margins=top_margins,
                               od_w=w,
                               imp_min_w=0,
                               mtype=mtype,
                               thres=threshold,
                               po_types=po_types,
                               edgel_info=edgel_info,
                               edger_info=edger_info,
                               )
        ext_bot_info = ExtInfo(margins=bot_margins,
                               od_w=w,
                               imp_min_w=0,
                               mtype=mtype,
                               thres=threshold,
                               po_types=po_types,
                               edgel_info=edgel_info,
                               edger_info=edger_info,
                               )

        # compute layout information
        lay_info_list = [(lay, 0, blk_yb, blk_yt) for lay in self.get_mos_layers(mos_type, threshold)]

        fill_info_list = [FillInfo(layer=layer, exc_layer=info[0], x_intv_list=[],
                                   y_intv_list=info[1]) for layer, info in fill_info.items()]

        blk_y = (blk_yb, blk_yt)
        layout_info = dict(
            blk_type='mos',
            lch_unit=lch_unit,
            fg=fg,
            arr_y=blk_y,
            draw_od=True,
            row_info_list=[RowInfo(od_x_list=[od_intv],
                                   od_y=od_yloc,
                                   od_type=(od_type, sub_type),
                                   po_y=blk_y,
                                   md_y=md_yloc), ],
            lay_info_list=lay_info_list,
            fill_info_list=fill_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=[(mos_type, threshold, blk_yb, blk_yt, blk_yb, blk_yt)],
            is_sub_ring=False,
            dnw_mode='',
            # adjacent block information list
            adj_info_list=[],
            left_blk_info=None,
            right_blk_info=None,
            # laygo connections information
            w=w,
        )

        # step 8: return results
        lch = lch_unit * self.res * self.tech_info.layout_unit
        lch_str = float_to_si_string(lch)
        row_name_id = '%s_l%s_w%d_%s' % (mos_type, lch_str, w, threshold)
        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=(edgel_info, []),
            right_edge_info=(edger_info, []),
            sd_yc=od_yc,
            g_conn_y=g_conn_y,
            gb_conn_y=gb_conn_y,
            ds_conn_y=ds_conn_y,
            row_name_id=row_name_id,
        )

    def get_laygo_sub_info(self, lch_unit, w, mos_type, threshold, **kwargs):
        # type: (int, int, str, str, **kwargs) -> Dict[str, Any]
        return self.get_laygo_mos_info(lch_unit, w, mos_type, threshold, 'sub', '', '', **kwargs)

    def get_laygo_end_info(self, lch_unit, mos_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        return self.get_analog_end_info(lch_unit, mos_type, threshold, fg, is_end, blk_pitch, **kwargs)

    def get_laygo_space_info(self, row_info, num_blk, left_blk_info, right_blk_info):
        # type: (Dict[str, Any], int, Any, Any) -> Dict[str, Any]

        ans = row_info.copy()
        layout_info = row_info['layout_info'].copy()

        lch_unit = layout_info['lch_unit']
        row_info_list = layout_info['row_info_list']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        od_spx = mos_constants['od_spx']
        od_fill_w_max = mos_constants['od_fill_w_max']

        laygo_unit_fg = self.get_laygo_unit_fg()
        od_fg_max = (od_fill_w_max - lch_unit) // sd_pitch - 1
        od_spx_fg = -(-(od_spx - sd_pitch + lch_unit) // sd_pitch) + 2

        # get OD fill X interval
        num_fg = laygo_unit_fg * num_blk
        area = num_fg - 2 * od_spx_fg
        if area > 0:
            od_x_list = fill_symmetric_max_density(area, area, laygo_unit_fg, od_fg_max, od_spx_fg,
                                                   offset=od_spx_fg, fill_on_edge=True, cyclic=False)[0]
        else:
            od_x_list = []
        # get row OD list
        row_info_list = [RowInfo(od_x_list=od_x_list, od_y=row_info.od_y,
                                 od_type=('dum', row_info.od_type[1]),
                                 po_y=row_info.po_y, md_y=row_info.md_y)
                         for row_info in row_info_list]

        # update layout and result information dictionary.
        layout_info['fg'] = num_fg
        layout_info['row_info_list'] = row_info_list
        layout_info['left_blk_info'] = left_blk_info[0]
        layout_info['right_blk_info'] = right_blk_info[0]

        lr_edge_info = (EdgeInfo(od_type=None, draw_layers={}), [])
        ans['layout_info'] = layout_info
        ans['left_edge_info'] = lr_edge_info
        ans['right_edge_info'] = lr_edge_info

        return ans

    def get_row_extension_info(self, bot_ext_list, top_ext_list):
        # type: (List[ExtInfo], List[ExtInfo]) -> List[Tuple[int, ExtInfo, ExtInfo]]

        # merge list of bottom and top extension informations into a list of bottom/top extension tuples
        bot_idx = top_idx = 0
        bot_len = len(bot_ext_list)
        top_len = len(top_ext_list)
        ext_groups = []
        cur_fg = bot_off = top_off = 0
        while bot_idx < bot_len and top_idx < top_len:
            bot_info = bot_ext_list[bot_idx]
            top_info = top_ext_list[top_idx]
            bot_ptype = bot_info.po_types
            top_ptype = top_info.po_types
            bot_stop = bot_off + len(bot_ptype)
            top_stop = top_off + len(top_ptype)
            stop_idx = min(bot_stop, top_stop)

            # create new bottom/top extension information objects for the current overlapping block
            # noinspection PyProtectedMember
            cur_bot_info = bot_info._replace(po_types=bot_ptype[cur_fg - bot_off:stop_idx - bot_off])
            # noinspection PyProtectedMember
            cur_top_info = top_info._replace(po_types=top_ptype[cur_fg - top_off:stop_idx - top_off])
            # append tuples of current number of fingers and bottom/top extension information object
            ext_groups.append((stop_idx - cur_fg, cur_bot_info, cur_top_info))

            cur_fg = stop_idx
            if stop_idx == bot_stop:
                bot_off = cur_fg
                bot_idx += 1
            if stop_idx == top_stop:
                top_off = cur_fg
                top_idx += 1

        return ext_groups

    def draw_laygo_space_connection(self, template, space_info, left_blk_info, right_blk_info):
        # type: (TemplateBase, Dict[str, Any], Any, Any) -> Tuple[Any, Any]

        # draw geometries
        self.draw_laygo_space_geometries(template, space_info, left_blk_info, right_blk_info)

        layout_info = space_info['layout_info']
        ext_top_info = space_info['ext_top_info']
        ext_bot_info = space_info['ext_bot_info']

        fg = layout_info['fg']
        lch_unit = layout_info['lch_unit']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        pode_is_poly = mos_constants['pode_is_poly']

        # figure out poly types per finger
        od_type_list = ('mos', 'sub', 'mos_fake')
        po_edge_code = 2 if pode_is_poly else 1
        po_types = [po_edge_code if left_blk_info[0].od_type in od_type_list else 0]
        po_types.extend((0 for _ in range(fg - 2)))
        po_types.append(po_edge_code if right_blk_info[0].od_type in od_type_list else 0)

        # noinspection PyProtectedMember
        ext_top_info = ext_top_info._replace(po_types=po_types)
        # noinspection PyProtectedMember
        ext_bot_info = ext_bot_info._replace(po_types=po_types)

        # return new extension information objects
        return ext_bot_info, ext_top_info

    def draw_laygo_connection(self, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None
        pass
