# -*- coding: utf-8 -*-

"""This module defines abstract analog mosfet template classes.
"""

from typing import Dict, Any, Tuple, List, TYPE_CHECKING

from bag.layout.util import BBox
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
    def get_laygo_mos_row_info(self,  # type: LaygoTech
                               lch_unit,  # type: int
                               w_max,  # type: int
                               w_sub,  # type: int
                               mos_type,  # type: str
                               threshold,  # type: str
                               bot_row_type,  # type: str
                               top_row_type,  # type: str
                               **kwargs):
        # type: (...) -> Dict[str, Any]
        """Returns the information dictionary for laygo transistor row.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w_max : int
            the maximum transistor width in number of fins/resolution units.
            Must be greater than or equal to w_sub.
        w_sub : int
            the substrate width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        bot_row_type : str
            the bottom (next to gate) laygo row type.
        top_row_type: str
            the top (next to drain/source) laygo row type.
        **kwargs
            optional keyword arguments.

        Returns
        -------
        row_info : Dict[str, Any]
            the row information dictionary.  Must have the following entries:

            w_max : int
                maximum transistor width in this row.
            w_sub : int
                substrate width in this row.
            lch_unit : int
                the channel length.
            row_type : str
                the row transistor/substrate type.
            sub_type : str
                the row substrate type.
            threshold : str
                the threshold flavor.
            arr_y : Tuple[int, int]
                the array box Y interval.
            od_y : Tuple[int, int]
                the worst case OD Y interval.
            po_y : Tuple[int, int]
                the PO Y interval.
            md_y : Tuple[int, int]
                the worst case MD Y interval.
            ext_top_info : Any
                an object used to compute extension layout above this row.
            ext_bot_info : Any
                an object used to compute extension layout below this row.
            lay_info_list : List[Any]
                the default layer information list.
            imp_params : Any
                the defeault implant parameters of this row.
            fill_info_list : List[Any]
                the fill information list.
            g_conn_y : Tuple[int, int]
                the gate connection Y coordinates.
            gb_conn_y : Tuple[int, int]
                the gate-bar connection Y coordinates.
            ds_conn_y : Tuple[int, int]
                the drain-source connection Y coordinates.
            row_name_id : str
                the name ID for this row.
        """
        return {}

    @abc.abstractmethod
    def get_laygo_sub_row_info(self, lch_unit, w, mos_type, threshold, **kwargs):
        # type: (int, int, str, str, **kwargs) -> Dict[str, Any]
        """Returns the information dictionary for laygo substrate row.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the substrate width in number of fins/resolution units.
        mos_type : str
            the transistor/substrate type.  One of 'pch', 'nch', 'ptap', or 'ntap'.
        threshold : str
            the transistor threshold flavor.
        **kwargs
            optional keyword arguments

        Returns
        -------
        row_info : Dict[str, Any]
            the row information dictionary.  Must have the following entries:

            w_max : int
                maximum transistor width in this row.
            w_sub : int
                substrate width in this row.
            lch_unit : int
                the channel length.
            row_type : str
                the row transistor/substrate type.
            sub_type : str
                the row substrate type.
            threshold : str
                the threshold flavor.
            arr_y : Tuple[int, int]
                the array box Y interval.
            od_y : Tuple[int, int]
                the worst case OD Y interval.
            po_y : Tuple[int, int]
                the PO Y interval.
            md_y : Tuple[int, int]
                the worst case MD Y interval.
            ext_top_info : Any
                an object used to compute extension layout above this row.
            ext_bot_info : Any
                an object used to compute extension layout below this row.
            lay_info_list : List[Any]
                the default layer information list.
            imp_params : Any
                the defeault implant parameters of this row.
            fill_info_list : List[Any]
                the fill information list.
            g_conn_y : Tuple[int, int]
                the gate connection Y coordinates.
            gb_conn_y : Tuple[int, int]
                the gate-bar connection Y coordinates.
            ds_conn_y : Tuple[int, int]
                the drain-source connection Y coordinates.
            row_name_id : str
                the name ID for this row.
        """
        return {}

    @abc.abstractmethod
    def get_laygo_blk_info(self, blk_type, w, row_info, **kwargs):
        # type: (str, int, Dict[str, Any], **kwargs) -> Dict[str, Any]
        """Returns the layout information dictionary for the given laygo block.

        Parameters
        ----------
        blk_type : str
            the laygo block type.
        w : int
            the transistor width.
        row_info : Dict[str, Any]
            the row layout information object.
        **kwargs
            optional keyword arguments.

        Returns
        -------
        blk_info : Dict[str, Any]
            the block information dictionary.  Contains the following entries:

            layout_info : Dict[str, Any]
                the block layout information dictionary.
            ext_top_info : Any
                an object used to compute extension layout above this block.
            ext_bot_info : Any
                an object used to compute extension layout below this block.
            left_edge_info : Any
                an object used to compute layout on the left of this block.
            right_edge_info : Any
                an object used to compute layout on the right of this block.
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
    def get_row_extension_info(self, bot_ext_list, top_ext_list):
        # type: (List[Any], List[Any]) -> List[Tuple[int, int, Any, Any]]
        """Compute the list of bottom/top extension information pair to create Laygo extension row.

        Parameters
        ----------
        bot_ext_list : List[Any]
            list of bottom extension information objects.
        top_ext_list : List[Any]
            list of top extension information objects.

        Returns
        -------
        ext_combo_list : List[Tuple[int, int, Any, Any]]
            list of number of fingers and bottom/top extension information objects for each
            extension primitive.
        """
        return []

    @abc.abstractmethod
    def draw_laygo_space_connection(self, template, space_info, left_blk_info, right_blk_info):
        # type: (TemplateBase, Dict[str, Any], Any, Any) -> None
        """Draw any space geometries necessary in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        space_info : Dict[str, Any]
            the laygo space block information dictionary.
        left_blk_info : Any
            left block information.
        right_blk_info : Any
            right block information.
        """
        pass

    @abc.abstractmethod
    def draw_laygo_connection(self, template, blk_info, blk_type, options):
        # type: (TemplateBase, Dict[str, Any], str, Dict[str, Any]) -> None
        """Draw digital transistor connection in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        blk_info : Dict[str, Any]
            the laygo block information dictionary.
        blk_type : str
            the digital block type.
        options : Dict[str, Any]
            any additional connection options.
        """
        pass

    def get_laygo_fg2d_s_short(self):
        # type: () -> bool
        """Returns True if the two source wires of fg2d is shorted together in the primitive.

        Returns
        -------
        s_short : bool
            True if the two source wires of fg2d is shorted together
        """
        return self.mos_config['laygo_fg2d_s_short']

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
                        num_cols,  # type: int
                        w,  # type: int
                        yext,  # type: int
                        bot_ext_list,  # type: List[Tuple[int, Any]]
                        top_ext_list,  # type: List[Tuple[int, Any]]
                        ):
        # type: (...) -> Tuple[Any, Any]
        """Draw extension rows in the given LaygoBase/DigitalBase template.

        Parameters
        ----------
        template : TemplateBase
            the LaygoBase/DigitalBase object to draw layout in.
        laygo_info : LaygoBaseInfo
            the LaygoBaseInfo object.
        num_cols : int
            number of columns.
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
        edgesl : Optional[Tuple[int, str, Dict[str, Any]]]
            a tuple of Y coordinate, orientation, and parameters for left edge.
        edgesr : Optional[Tuple[int, str, Dict[str, Any]]]
            a tuple of Y coordinate, orientation, and parameters for right edge.
        """
        lch = laygo_info.lch
        top_layer = laygo_info.top_layer
        guard_ring_nf = laygo_info.guard_ring_nf

        ext_groups = self.get_row_extension_info(bot_ext_list, top_ext_list)

        edgesl, edgesr = None, None
        if w > 0 or self.draw_zero_extension():
            for idx, (fg_off, fg, bot_info, top_info) in enumerate(ext_groups):
                ext_params = dict(
                    lch=lch,
                    w=w,
                    fg=fg,
                    top_ext_info=top_info,
                    bot_ext_info=bot_info,
                    is_laygo=True,
                )
                curx = laygo_info.col_to_coord(fg_off, unit_mode=True)
                ext_master = template.new_template(params=ext_params, temp_cls=AnalogMOSExt)
                template.add_instance(ext_master, loc=(curx, yext), unit_mode=True)

                if fg_off == 0:
                    adj_blk_info = ext_master.get_left_edge_info()
                    # compute edge parameters
                    cur_ext_edge_params = dict(
                        top_layer=top_layer,
                        guard_ring_nf=guard_ring_nf,
                        name_id=ext_master.get_layout_basename(),
                        layout_info=ext_master.get_edge_layout_info(),
                        adj_blk_info=adj_blk_info,
                        is_laygo=True,
                    )
                    edgesl = (yext, cur_ext_edge_params)
                if fg_off + fg == num_cols:
                    adj_blk_info = ext_master.get_right_edge_info()
                    # compute edge parameters
                    cur_ext_edge_params = dict(
                        top_layer=top_layer,
                        guard_ring_nf=guard_ring_nf,
                        name_id=ext_master.get_layout_basename(),
                        layout_info=ext_master.get_edge_layout_info(),
                        adj_blk_info=adj_blk_info,
                        is_laygo=True,
                    )
                    edgesr = (yext, cur_ext_edge_params)

        return edgesl, edgesr

    def draw_boundaries(self,  # type: LaygoTech
                        template,  # type: TemplateBase
                        laygo_info,  # type: LaygoBaseInfo
                        num_col,  # type: int
                        yt,  # type: int
                        bot_end_master,  # type: LaygoEndRow
                        top_end_master,  # type: LaygoEndRow
                        edgel_infos,  # type: List[Tuple[int, str, Dict[str, Any]]]
                        edger_infos,  # type: List[Tuple[int, str, Dict[str, Any]]]
                        ):
        # type: (...) -> Tuple[BBox, List[WireArray], List[WireArray]]
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
        edgel_infos:  List[Tuple[int, str, Dict[str, Any]]]
            a list of Y coordinate, orientation, and parameters for left edge blocks.
        edger_infos:  List[Tuple[int, str, Dict[str, Any]]]
            a list of Y coordinate, orientation, and parameters for right edge blocks.

        Returns
        -------
        arr_box : BBox
            the array box.
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

        inst = template.add_instance(bot_end_master, inst_name='XRBOT', loc=(xoffset, 0),
                                     nx=nx, spx=spx, unit_mode=True)
        arr_box = inst.array_box
        inst = template.add_instance(top_end_master, inst_name='XRBOT', loc=(xoffset, yt),
                                     orient='MX', nx=nx, spx=spx, unit_mode=True)
        arr_box = arr_box.merge(inst.array_box)
        # draw corners
        left_end = (end_mode & 4) != 0
        right_end = (end_mode & 8) != 0
        edge_inst_list = []
        xr = laygo_info.tot_width
        for orient, y, master in (('R0', 0, bot_end_master), ('MX', yt, top_end_master)):
            for x, is_end, flip_lr in ((emargin_l, left_end, False),
                                       (xr - emargin_r, right_end, True)):
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
                edge_inst_list.append(template.add_instance(edge_master, orient=eorient,
                                                            loc=(x, y), unit_mode=True))

        # draw edge blocks
        for y, flip_ud, edge_params in edgel_infos:
            orient = 'MX' if flip_ud else 'R0'
            edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
            edge_inst_list.append(template.add_instance(edge_master, orient=orient,
                                                        loc=(emargin_l, y), unit_mode=True))
        for y, flip_ud, edge_params in edger_infos:
            orient = 'R180' if flip_ud else 'MY'
            edge_master = template.new_template(params=edge_params, temp_cls=AnalogEdge)
            edge_inst_list.append(template.add_instance(edge_master, orient=orient,
                                                        loc=(xr - emargin_r, y), unit_mode=True))

        gr_vss_warrs = []
        gr_vdd_warrs = []
        conn_layer = self.get_dig_conn_layer()
        for inst in edge_inst_list:
            if inst.has_port('VDD'):
                gr_vdd_warrs.extend(inst.get_all_port_pins('VDD', layer=conn_layer))
            elif inst.has_port('VSS'):
                gr_vss_warrs.extend(inst.get_all_port_pins('VSS', layer=conn_layer))
            arr_box = arr_box.merge(inst.array_box)

        # connect body guard rings together
        gr_vdd_warrs = template.connect_wires(gr_vdd_warrs)
        gr_vss_warrs = template.connect_wires(gr_vss_warrs)

        arr_box_x = laygo_info.get_placement_info(num_col).arr_box_x
        arr_box = BBox(arr_box_x[0], arr_box.bottom_unit, arr_box_x[1], arr_box.top_unit,
                       arr_box.resolution, unit_mode=True)
        return arr_box, gr_vdd_warrs, gr_vss_warrs
