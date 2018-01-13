# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Tuple

import abc

from bag import float_to_si_string

from .tech import LaygoTech
from ..analog_mos.finfet import MOSTechFinfetBase
from ..analog_mos.finfet import ExtInfo, RowInfo, AdjRowInfo, EdgeInfo, FillInfo


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
        return {}

    def get_laygo_end_info(self, lch_unit, mos_type, threshold, fg, is_end, blk_pitch):
        # type: (int, str, str, int, bool, int) -> Dict[str, Any]
        return {}

    def get_laygo_space_info(self, row_info, num_blk, left_blk_info, right_blk_info):
        # type: (Dict[str, Any], int, Any, Any) -> Dict[str, Any]
        pass

    def draw_laygo_connection(self, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None
        pass

    def draw_laygo_space_connection(self, template, space_info, left_blk_info, right_blk_info):
        # type: (TemplateBase, Dict[str, Any], Any, Any) -> Tuple[Any, Any]
        pass

    def get_row_extension_info(self, bot_ext_list, top_ext_list):
        # type: (List[Any], List[Any]) -> List[Tuple[int, Any, Any]]
        return []
