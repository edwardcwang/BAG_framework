# -*- coding: utf-8 -*-

"""This module defines abstract analog mosfet template classes.
"""

from typing import Dict, Any, Tuple, List, TYPE_CHECKING

from bag.layout.template import TemplateBase
from bag.layout.routing import WireArray

import abc

from ..analog_mos.core import MOSTech
from ..analog_mos.mos import AnalogMOSExt
from ..analog_mos.edge import AnalogEdge

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig
    from .base import LaygoEndRow
    from .core import LaygoBaseInfo


class LaygoTech(MOSTech, metaclass=abc.ABCMeta):
    """An abstract class for drawing transistor related layout for custom digital circuits.

    This class defines various methods use to draw layouts used by LaygoBase.

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
        MOSTech.__init__(self, config, tech_info, mos_entry_name=mos_entry_name)

    @abc.abstractmethod
    def get_default_end_info(self):
        # type: () -> Any
        """Returns the default end_info object."""
        return 0

    @abc.abstractmethod
    def get_laygo_mos_info(self, lch_unit, w, mos_type, threshold, blk_type, bot_row_type, top_row_type, **kwargs):
        # type: (int, int, str, str, str, str, str, **kwargs) -> Dict[str, Any]
        """Returns the transistor information dictionary for laygo blocks.

        The returned dictionary must have the following entries:

        layout_info: the layout information dictionary.
        ext_top_info: a tuple of values used to compute extension layout above the transistor.
        ext_bot_info : a tuple of values used to compute extension layout below the transistor.
        sd_yc : the Y coordinate of the center of source/drain junction.
        top_gtr_yc : maximum Y coordinate of the center of top gate track.
        bot_dstr_yc : minimum Y coordinate of the center of bottom drain/source track.
        max_bot_tr_yc : maximum Y coordinate of the center of bottom block tracks.
        min_top_tr_yc : minimum Y coordinate of the center of top block tracks.
        blk_height : the block height in mos pitches.

        The 4 track Y coordinates (top_gtr_yc, bot_dstr_yc, max_bot_tr_yc, min_top_tr_yc)
        should be independent of the transistor width.  In this way, you can use different
        width transistors in the same row.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        blk_type : str
            the digital block type.
        bot_row_type : str
            the bottom (next to gate) laygo row type.
        top_row_type: str
            the top (next to drain/source) laygo row type.
        **kwargs
            optional keyword arguments.

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_laygo_sub_info(self, lch_unit, w, mos_type, threshold, **kwargs):
        # type: (int, int, str, str, **kwargs) -> Dict[str, Any]
        """Returns the transistor information dictionary for laygo blocks.

        The returned dictionary must have the following entries:

        layout_info: the layout information dictionary.
        ext_top_info: a tuple of values used to compute extension layout above the transistor.
        ext_bot_info : a tuple of values used to compute extension layout below the transistor.
        sd_yc : the Y coordinate of the center of source/drain junction.
        top_gtr_yc : maximum Y coordinate of the center of top gate track.
        bot_dstr_yc : minimum Y coordinate of the center of bottom drain/source track.
        max_bot_tr_yc : maximum Y coordinate of the center of bottom block tracks.
        min_top_tr_yc : minimum Y coordinate of the center of top block tracks.
        blk_height : the block height in mos pitches.

        The 4 track Y coordinates (top_gtr_yc, bot_dstr_yc, max_bot_tr_yc, min_top_tr_yc)
        should be independent of the transistor width.  In this way, you can use different
        width transistors in the same row.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        **kwargs
            optional keyword arguments

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_laygo_end_info(self, lch_unit, mos_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        """Returns the LaygoBase end row layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        mos_type : str
            the transistor type, one of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        is_end : bool
            True if there are no block abutting the bottom.
        blk_pitch : int
            height quantization pitch, in resolution units.
        **kwargs :
            optional parameters.

        Returns
        -------
        end_info : Dict[str, Any]
            the laygo end row information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_laygo_space_info(self, row_info, num_blk, left_blk_info, right_blk_info):
        # type: (Dict[str, Any], int, Any, Any) -> Dict[str, Any]
        """Returns a new layout information dictionary for drawing LaygoBase space blocks.

        Parameters
        ----------
        row_info : Dict[str, Any]
            the Laygo row information dictionary.
        num_blk : int
            number of space blocks.
        left_blk_info : Any
            left block information.
        right_blk_info : Any
            right block information.

        Returns
        -------
        space_info : Dict[str, Any]
            the space layout information dictionary.
        """
        pass

    @abc.abstractmethod
    def draw_laygo_connection(self, template, mos_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None
        """Draw digital transistor connection in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        blk_type : str
            the digital block type.
        options : Dict[str, Any]
            any additional connection options.
        """
        pass

    @abc.abstractmethod
    def draw_laygo_space_connection(self, template, space_info, left_blk_info, right_blk_info):
        # type: (TemplateBase, Dict[str, Any], Any, Any) -> Tuple[Any, Any]
        """Draw digital transistor connection in the given template.

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

        Returns
        -------
        bot_ext_info : Any
            space block bottom extension information.
        top_ext_info : Any
            space block top extension information.
        """
        pass

    @abc.abstractmethod
    def get_row_extension_info(self, bot_ext_list, top_ext_list):
        # type: (List[Any], List[Any]) -> List[Tuple[int, Any, Any]]
        """Compute the list of bottom/top extension information pair needed to create Laygo extension row.

        Parameters
        ----------
        bot_ext_list : List[Any]
            list of bottom extension information objects.
        top_ext_list : List[Any]
            list of top extension information objects.

        Returns
        -------
        ext_combo_list : List[Tuple[int, Any, Any]]
            list of number of fingers and bottom/top extension information objects for each extension primitive.
        """
        return []

    def get_laygo_fg2d_s_short(self):
        # type: () -> bool
        """Returns True if the two source wires of fg2d is shorted together in the primitive.

        Returns
        -------
        s_short : bool
            True if the two source wires of fg2d is shorted together
        """
        return self.mos_config['laygo_fg2d_s_short']

    def get_laygo_unit_fg(self):
        # type: () -> int
        """Returns LaygoBase unit cell width in number of fingers.

        Returns
        -------
        num_fg : int
            LaygoBase unit cell width in number of fingers.
        """
        return self.mos_config['laygo_unit_fg']

    def get_sub_columns(self, lch_unit):
        # type: (int) -> int
        """Returns the number of columns per substrate block.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        num_cols : int
            number of columns per substrate block.
        """
        return self.get_mos_tech_constants(lch_unit)['laygo_sub_ncol']

    def get_sub_port_columns(self, lch_unit):
        # type: (int) -> List[int]
        """Returns the columns indices that have ports in substrate block.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        port_cols : List[int]
            the columns indices that have ports in substrate block.
        """
        return self.get_mos_tech_constants(lch_unit)['laygo_sub_port_cols']

    def get_min_sub_space_columns(self, lch_unit):
        # type: (int) -> int
        """Returns the minimum number of space columns needed around substrate blocks.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        num_cols : int
            minimum number of space columns.
        """
        return self.get_mos_tech_constants(lch_unit)['laygo_sub_spx']

    def get_laygo_conn_track_info(self, lch_unit):
        # type: (int) -> Tuple[int, int]
        """Returns dummy connection layer space and width.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units.

        Returns
        -------
        dum_sp : int
            space between dummy tracks in resolution units.
        dum_w : int
            width of dummy tracks in resolution units.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        laygo_conn_w = mos_constants['laygo_conn_w']
        laygo_num_sd_per_track = mos_constants['laygo_num_sd_per_track']
        return sd_pitch * laygo_num_sd_per_track - laygo_conn_w, laygo_conn_w

    def draw_extensions(self,  # type: LaygoTech
                        template,  # type: TemplateBase
                        laygo_info,  # type: LaygoBaseInfo
                        w,  # type: int
                        yext,  # type: int
                        bot_ext_list,  # type: List[Tuple[int, Any]]
                        top_ext_list,  # type: List[Tuple[int, Any]]
                        ):
        # type: (...) -> List[Tuple[int, str, Dict[str, Any]]]
        """Draw extension rows in the given LaygoBase/DigitalBase template.

        Parameters
        ----------
        template : TemplateBase
            the LaygoBase/DigitalBase object to draw layout in.
        laygo_info : LaygoBaseInfo
            the LaygoBaseInfo object.
        w : int
            extension width in number of mos pitches.
        yext : int
            Y coordinate of the extension block.
        bot_ext_list : List[Tuple[int, Any]]
            list of tuples of end finger index and bottom extension information
        top_ext_list : List[Tuple[int, Any]]
            list of tuples of end finger index and top extension information

        Returns
        -------
        ext_edges : List[Tuple[int, str, Dict[str, Any]]]
            a list of Y coordinate, orientation, and parameters for extension edge blocks.
            empty list if draw_boundaries is False.
        """
        lch = laygo_info.lch
        top_layer = laygo_info.top_layer
        guard_ring_nf = laygo_info.guard_ring_nf

        ext_groups = self.get_row_extension_info(bot_ext_list, top_ext_list)
        num_ext = len(ext_groups)

        curx = laygo_info.col_to_coord(0, 's', unit_mode=True)
        ext_edges = []
        for idx, (fg, bot_info, top_info) in enumerate(ext_groups):
            if w > 0 or self.draw_zero_extension():
                ext_params = dict(
                    lch=lch,
                    w=w,
                    fg=fg,
                    top_ext_info=top_info,
                    bot_ext_info=bot_info,
                    is_laygo=True,
                )
                ext_master = template.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                template.add_instance(ext_master, loc=(curx, yext), unit_mode=True)
                curx += ext_master.prim_bound_box.width_unit

                if idx == 0 or idx == num_ext - 1:
                    adj_blk_info = ext_master.get_left_edge_info() if idx == 0 else ext_master.get_right_edge_info()
                    # compute edge parameters
                    cur_ext_edge_params = dict(
                        top_layer=top_layer,
                        guard_ring_nf=guard_ring_nf,
                        name_id=ext_master.get_layout_basename(),
                        layout_info=ext_master.get_edge_layout_info(),
                        adj_blk_info=adj_blk_info,
                        is_laygo=True,
                    )
                    edge_orient = 'R0' if idx == 0 else 'MY'
                    ext_edges.append((yext, edge_orient, cur_ext_edge_params))

        return ext_edges

    def draw_boundaries(self,  # type: LaygoTech
                        template,  # type: TemplateBase
                        laygo_info,  # type: LaygoBaseInfo
                        num_col,  # type: int
                        yt,  # type: int
                        bot_end_master,  # type: LaygoEndRow
                        top_end_master,  # type: LaygoEndRow
                        edge_infos,  # type: List[Tuple[int, int, str, Dict[str, Any]]]
                        ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw boundaries for LaygoBase/DigitalBase.

        Parameters
        ----------
        template : TemplateBase
            the LaygoBase/DigitalBase object to draw layout in.
        laygo_info : LaygoBaseInfo
            the LaygoBaseInfo object.
        num_col : int
            number of primitive columns in the template.
        yt : int
            the top Y coordinate of the template.  Used to determine top end row placement.
        bot_end_master: LaygoEndRow
            the bottom LaygoEndRow master.
        top_end_master : LaygoEndRow
            the top LaygoEndRow master.
        edge_infos:  List[Tuple[int, int, str, Dict[str, Any]]]
            a list of X/Y coordinate, orientation, and parameters for all edge blocks.

        Returns
        -------
        vdd_warrs : List[WireArray]
            any VDD wires in the edge block due to guard ring.
        vss_warrs : List[WireArray]
            any VSS wires in the edge block due to guard ring.
        """
        end_mode = laygo_info.end_mode
        guard_ring_nf = laygo_info.guard_ring_nf
        col_width = laygo_info.col_width

        nx = num_col
        spx = col_width

        emargin_l, emargin_r = laygo_info.edge_margins
        ewidth_l, ewidth_r = laygo_info.edge_widths
        xoffset = emargin_l + ewidth_l

        # draw top and bottom end row
        template.add_instance(bot_end_master, inst_name='XRBOT', loc=(xoffset, 0),
                              nx=nx, spx=spx, unit_mode=True)
        template.add_instance(top_end_master, inst_name='XRBOT', loc=(xoffset, yt),
                              orient='MX', nx=nx, spx=spx, unit_mode=True)
        # draw corners
        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0
        edge_inst_list = []
        xr = laygo_info.tot_width
        for orient, y, master in (('R0', 0, bot_end_master), ('MX', yt, top_end_master)):
            for x, is_end, flip_lr in ((emargin_l, left_end, False), (xr - emargin_r, right_end, True)):
                edge_params = dict(
                    is_end=is_end,
                    guard_ring_nf=guard_ring_nf,
                    adj_blk_info=master.get_left_edge_info(),
                    name_id=master.get_layout_basename(),
                    layout_info=master.get_edge_layout_info(),
                    is_laygo=True,
                )
                edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
                if flip_lr:
                    eorient = 'MY' if orient == 'R0' else 'R180'
                else:
                    eorient = orient
                edge_inst_list.append(template.add_instance(edge_master, orient=eorient, loc=(x, y), unit_mode=True))

        # draw edge blocks
        for x, y, orient, edge_params in edge_infos:
            edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
            edge_inst_list.append(template.add_instance(edge_master, orient=orient, loc=(x, y), unit_mode=True))

        gr_vss_warrs = []
        gr_vdd_warrs = []
        conn_layer = self.get_dig_conn_layer()
        for inst in edge_inst_list:
            if inst.has_port('VDD'):
                gr_vdd_warrs.extend(inst.get_all_port_pins('VDD', layer=conn_layer))
            elif inst.has_port('VSS'):
                gr_vss_warrs.extend(inst.get_all_port_pins('VSS', layer=conn_layer))

        # connect body guard rings together
        gr_vdd_warrs = template.connect_wires(gr_vdd_warrs)
        gr_vss_warrs = template.connect_wires(gr_vss_warrs)

        return gr_vdd_warrs, gr_vss_warrs
