# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Union, Any, List, Optional, Tuple

import abc
import math
from collections import namedtuple

from bag.math import lcm
from bag.util.search import BinaryIterator
from bag.layout.util import BBox
from bag.layout.routing import WireArray
from bag.layout.routing.fill import fill_symmetric_min_density_info
from bag.layout.routing.fill import fill_symmetric_interval
from bag.layout.routing.fill import fill_symmetric_max_density
from bag.layout.template import TemplateBase

from .core import MOSTech

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig

RowInfo = namedtuple('RowInfo', ['od_x_list', 'od_type', 'row_y', 'od_y', 'po_y', 'md_y'])
AdjRowInfo = namedtuple('AdjRowInfo', ['row_y', 'po_y', 'po_types'])
EdgeInfo = namedtuple('EdgeInfo', ['od_type', 'draw_layers', 'y_intv'])
FillInfo = namedtuple('FillInfo', ['layer', 'exc_layer', 'x_intv_list', 'y_intv_list'])


class ExtInfo(namedtuple('ExtInfoBase', ['margins', 'od_h', 'imp_min_h', 'mtype', 'thres',
                                         'po_types', 'edgel_info', 'edger_info'])):
    __slots__ = ()

    def reverse(self):
        return self._replace(po_types=tuple(reversed(self.po_types)),
                             edgel_info=self.edger_info,
                             edger_info=self.edgel_info)


class MOSTechFinfetBase(MOSTech, metaclass=abc.ABCMeta):
    """Base class for implementations of MOSTech in Finfet technologies.

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
        MOSTech.__init__(self, config, tech_info, mos_entry_name=mos_entry_name)
        self.ignore_vm_layers = set()

    @abc.abstractmethod
    def get_mos_yloc_info(self, lch_unit, w, **kwargs):
        # type: (int, int, **kwargs) -> Dict[str, Any]
        """Computes Y coordinates of various layers in the transistor row.

        The returned dictionary should have the following entries:

        blk :
            a tuple of row bottom/top Y coordinates.
        po :
            a tuple of PO bottom/top Y coordinates that's outside of CPO.
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
        g_conn_y :
            Y coordinate interval where horizontal gate wires can contact
            to gate.
        d_conn_y :
            Y coordinate interval where horizontal drain/source wire can
            contact to drain/source.
        """
        return {}

    @abc.abstractmethod
    def get_sub_yloc_info(self, lch_unit, w, **kwargs):
        # type: (int, int, **kwargs) -> Dict[str, Any]
        """Computes Y coordinates of various layers in the substrate row.

        The returned dictionary should have the following entries:

        blk :
            a tuple of row bottom/top Y coordinates.
        po :
            a tuple of PO bottom/top Y coordinates that's outside of CPO.
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
        g_conn_y :
            Y coordinate interval where horizontal gate wires can contact
            to gate.
        d_conn_y :
            Y coordinate interval where horizontal drain/source wire can
            contact to drain/source.
        """
        return {}

    @abc.abstractmethod
    def draw_ds_connection(self,
                           template,  # type: TemplateBase
                           lch_unit,  # type: int
                           fg,  # type: int
                           wire_pitch,  # type: int
                           xc,  # type: int
                           od_y,  # type: Tuple[int, int]
                           md_y,  # type: Tuple[int, int]
                           dum_x_list,  # type: List[int]
                           conn_x_list,  # type: List[int]
                           align_gate,  # type: bool
                           wire_dir,  # type: int
                           ds_code,  # type: int
                           **kwargs
                           ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw drain/source connections on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the connection in.
        lch_unit : int
            the channel length in resolution units.
        fg : int
            number of fingers of the connection.
        wire_pitch : int
            the source/drain wire pitch.
        xc : int
            the center X coordinate of left-most source/drain.
        od_y : Tuple[int, int]
            the OD Y interval tuple.
        md_y : Tuple[int, int]
            the MD Y interval tuple.
        dum_x_list : List[int]
            list of center X coordinates to export dummy connection port.
        conn_x_list : List[int]
            list of center X coordinates to export connection port.
        align_gate : bool
            True if this drain/source connection is in the same column as gate.
        wire_dir : int
            the wire direction.  2 for up, 1 for middle, 0 for down.
        ds_code : int
            the drain/source code.  1 to draw source, 2 to draw drain, 3 to draw substrate
            connnection, 4 to draw substrate connection in guard ring.
        **kwargs :
            optional parameters.  Must support:

            source_parity : int
                the parity number of the source.  This is used if source/drain connections need to
                alternate between parity.
            ud_parity : int
                the up/down parity.  This is used if the source/drain connections is not symmetric
                across the middle horizontal line.

        Returns
        -------
        dum_warrs : List[WireArray]
            dummy wires as single-wire WireArrays.
        conn_warrs : List[WireArray]
            connection wires as single-wire WireArrays.
        """
        return [], []

    @abc.abstractmethod
    def draw_g_connection(self,
                          template,  # type: TemplateBase
                          lch_unit,  # type: int
                          fg,  # type: int
                          sd_pitch,  # type: int
                          xc,  # type: int
                          od_y,  # type: Tuple[int, int]
                          md_y,  # type: Tuple[int, int]
                          conn_x_list,  # type: List[int]
                          is_sub=False,  # type: bool
                          **kwargs
                          ):
        # type: (...) -> List[WireArray]
        """Draw gate connections on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the connection in.
        lch_unit : int
            the channel length in resolution units.
        fg : int
            number of fingers of the connection.
        sd_pitch : int
            the source/drain pitch.
        xc : int
            the center X coordinate of left-most source/drain.
        od_y : Tuple[int, int]
            the OD Y interval tuple.
        md_y : Tuple[int, int]
            the MD Y interval tuple.
        conn_x_list : List[int]
            list of center X coordinates to export connection port.
        is_sub : bool
            True if this is gate connection for substrate.
        **kwargs :
            optional parameters.

        Returns
        -------
        gate_warr : List[WireArray]
            gate wires as single-wire WireArrays.
        """
        return []

    @abc.abstractmethod
    def draw_dum_connection_helper(self,
                                   template,  # type: TemplateBase
                                   lch_unit,  # type: int
                                   fg,  # type: int
                                   sd_pitch,  # type: int
                                   xc,  # type: int
                                   od_y,  # type: Tuple[int, int]
                                   md_y,  # type: Tuple[int, int]
                                   ds_x_list,  # type: List[int]
                                   gate_tracks,  # type: List[Union[float, int]]
                                   left_edge,  # type: bool
                                   right_edge,  # type: bool
                                   options,  # type: Dict[str, Any]
                                   ):
        # type: (...) -> List[WireArray]
        """Draw dummy connections on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the connection in.
        lch_unit : int
            the channel length in resolution units.
        fg : int
            number of fingers of the connection.
        sd_pitch : int
            the source/drain pitch.
        xc : int
            the center X coordinate of left-most source/drain.
        od_y : Tuple[int, int]
            the OD Y interval tuple.
        md_y : Tuple[int, int]
            the MD Y interval tuple.
        ds_x_list : List[int]
            list of center X coordinates to draw drain/source dummy connections.
        gate_tracks : List[int]
            tracks to export dummy gate connections.
        left_edge : bool
            True if this dummy is on the left-most edge.
        right_edge : bool
            True if this dummy is on the right-most edge.
        options : Dict[str, Any]
            the dummy connection options.

        Returns
        -------
        dum_warrs : List[WireArray]
            dummy wires
        """
        return []

    @abc.abstractmethod
    def draw_decap_connection_helper(self,
                                     template,  # type: TemplateBase
                                     lch_unit,  # type: int
                                     fg,  # type: int
                                     sd_pitch,  # type: int
                                     xc,  # type: int
                                     od_y,  # type: Tuple[int, int]
                                     md_y,  # type: Tuple[int, int]
                                     gate_ext_mode,  # type: int
                                     export_gate,  # type: bool
                                     ):
        # type: (...) -> Tuple[Optional[WireArray], List[WireArray]]
        """Draw dummy connections on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the connection in.
        lch_unit : int
            the channel length in resolution units.
        fg : int
            number of fingers of the connection.
        sd_pitch : int
            the source/drain pitch.
        xc : int
            the center X coordinate of left-most source/drain.
        od_y : Tuple[int, int]
            the OD Y interval tuple.
        md_y : Tuple[int, int]
            the MD Y interval tuple.
        gate_ext_mode : int
            gate extension mode.  2-bit integer where LSB is 1 to
            extend left, MSB is 1 to extend right.
        export_gate : bool
            True to export gate on mos connection lay.

        Returns
        -------
        g_warr : Optional[WireArray]
            the gate wires.  None if gate does not need to be exported.
        sup_warrs : List[WireArray]
            list of supply wires.  Each supply wire will be exported
            individually.  wires with the same pitch should be grouped
            in a single WireArray.
        """
        return None, []

    def postprocess_mos_tech_constants(self, lch_unit, mos_constants):
        # type: (int, Dict[str, Any]) -> None
        """Optional method subclasses can override to add more entries to mos_constants."""
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        offset, h_scale, p_scale = mos_constants.get('od_fin_exty_constants', (0, 0, 0))
        mos_constants['od_fin_exty'] = (offset + int(round(h_scale * fin_h)) +
                                        int(round(p_scale * fin_p)))

    def get_od_w(self, lch_unit, fg):
        # type: (int, int) -> int
        """Calculate OD width."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        po_od_extx = mos_constants['po_od_extx']
        return (fg - 1) * sd_pitch + lch_unit + 2 * po_od_extx

    def get_od_w_inverse(self, lch_unit, w, round_up=None):
        # type: (int, int, bool) -> int
        """Calculate OD width."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        po_od_extx = mos_constants['po_od_extx']

        q = w - 2 * po_od_extx - lch_unit
        if round_up is None and q % sd_pitch != 0:
            raise ValueError('OD width %d is not correct.' % w)
        elif round_up:
            return 1 - (-q // sd_pitch)
        else:
            return 1 + (q // sd_pitch)

    def get_od_h(self, lch_unit, w):
        # type: (int, int) -> int
        """Calculate OD height."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_fin_exty = mos_constants['od_fin_exty']
        return (w - 1) * fin_p + fin_h + 2 * od_fin_exty

    def get_od_h_inverse(self, lch_unit, od_h, round_up=None):
        # type: (int, int) -> int
        """Calculate number of fins from OD height."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_fin_exty = mos_constants['od_fin_exty']

        q = od_h - 2 * od_fin_exty - fin_h
        if round_up is None and q % fin_p != 0:
            raise ValueError('OD height %d is not on fin grid.' % od_h)
        elif round_up:
            w = -(-q // fin_p) + 1
        else:
            w = q // fin_p + 1
        return w

    def get_od_spy_nfin(self, lch_unit, sp, round_up=True):
        # type: (int, int) -> int
        """Calculate OD vertical space in number of fin pitches, rounded up.

        Space of 0 means no fins are between the two OD.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_fin_exty = mos_constants['od_fin_exty']

        q = sp + fin_h + 2 * od_fin_exty - fin_p
        return -(-q // fin_p) if round_up else q // fin_p

    def get_od_spx_fg(self, lch_unit, sp):
        """Calculate OD horizontal space in number of fingers, rounded up.

        Space of 0 means no PO are between the PODEs.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        po_od_extx = mos_constants['po_od_extx']

        return -(-(sp + 2 * po_od_extx + lch_unit - 3 * sd_pitch) // sd_pitch)

    def get_fin_idx(self, lch_unit, od_y, top_edge, round_up=None):
        # type: (int, int, bool, Optional[bool]) -> int
        """Get fin index from OD top/bottom edge coordinate."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_fin_exty = mos_constants['od_fin_exty']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2
        delta = fin_h2 + od_fin_exty
        if not top_edge:
            delta *= -1

        quantity = od_y - delta - fin_p2
        if round_up is None and quantity % fin_p != 0:
            raise ValueError('OD coordinate %d is not on fin grid.' % od_y)
        elif round_up:
            fin_idx = -(-quantity // fin_p)
        else:
            fin_idx = quantity // fin_p

        return fin_idx

    def get_od_edge(self, lch_unit, fin_idx, top_edge):
        # type: (int, int, bool) -> int
        """Get OD edge Y coordinate from fin index."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_fin_exty = mos_constants['od_fin_exty']

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2
        delta = fin_h2 + od_fin_exty
        if not top_edge:
            delta *= -1

        return fin_idx * fin_p + fin_p2 + delta

    def snap_od_edge(self, lch_unit, od_y, top_edge, round_up):
        # type: (int, int, bool, bool) -> int
        """Snap the OD horizontal edge to fin grid."""
        fin_idx = self.get_fin_idx(lch_unit, od_y, top_edge, round_up=round_up)
        return self.get_od_edge(lch_unit, fin_idx, top_edge)

    def get_edge_info(self, lch_unit, guard_ring_nf, is_end, **kwargs):
        # type: (int, int, bool, **kwargs) -> Dict[str, Any]
        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')

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
        po_od_extx = mos_constants['po_od_extx']
        fg_outer_gr = mos_constants.get('fg_outer_gr', 0)
        substrate_planar = mos_constants.get('substrate_planar', False)

        if 0 < guard_ring_nf < fg_gr_min:
            raise ValueError('guard_ring_nf = %d < %d' % (guard_ring_nf, fg_gr_min))
        if is_sub_ring and guard_ring_nf <= 0:
            raise ValueError('guard_ring_nf = %d must be positive '
                             'in substrate ring' % guard_ring_nf)

        # step 0: figure out implant/OD enclosure and outer edge margin
        outer_margin = edge_margin
        if dnw_mode:
            od_w = self.get_od_w(lch_unit, fg_gr_min)
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
            fg_gr_sub = guard_ring_nf + 2 * fg_od_margin
            if substrate_planar:
                fg_gr_sep = fg_outer
                fg_outer = 0
            else:
                if is_sub_ring:
                    fg_gr_sep = -(-edge_margin // sd_pitch)
                else:
                    fg_gr_sep = fg_od_margin
                fg_outer = fg_outer_gr

        # compute edge margin and cpo_xl
        if is_end:
            edge_margin = outer_margin
            cpo_xl = (sd_pitch - lch_unit) // 2 - cpo_po_extx
        else:
            edge_margin = cpo_xl = 0

        return dict(
            edge_num_fg=fg_outer + fg_gr_sub + fg_gr_sep,
            edge_margin=edge_margin,
            cpo_xl=cpo_xl,
            fg_outer=fg_outer,
            fg_gr_sub=fg_gr_sub,
            fg_gr_sep=fg_gr_sep,
            fg_od_margin=fg_od_margin,
        )

    def _get_mos_blk_info(self, lch_unit, fg, w, mos_type, sub_type, threshold, **kwargs):
        # type: (int, int, int, str, str, str, **kwargs) -> Dict[str, Any]

        dnw_mode = kwargs.get('dnw_mode', '')
        is_sub_ring = kwargs.get('is_sub_ring', False)
        ds_dummy = kwargs.get('ds_dummy', False)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        nw_dnw_ovl = mos_constants['nw_dnw_ovl']
        dnw_layers = mos_constants['dnw_layers']
        substrate_planar = mos_constants.get('substrate_planar', False)
        has_cpo = mos_constants['has_cpo']
        cpo_h = mos_constants['cpo_h']

        is_sub = (mos_type == sub_type)
        blk_type = 'sub' if is_sub else 'mos'
        od_type = 'mos_fake' if ds_dummy else blk_type

        if is_sub:
            yloc_info = self.get_sub_yloc_info(lch_unit, w, **kwargs)
        else:
            yloc_info = self.get_mos_yloc_info(lch_unit, w, **kwargs)

        # Compute Y coordinates of various layers
        blk_yb, blk_yt = yloc_info['blk']
        po_yloc = yloc_info['po']
        od_yloc = yloc_info['od']
        md_yloc = yloc_info['md']
        top_margins = yloc_info['top_margins']
        bot_margins = yloc_info['bot_margins']
        fill_info = yloc_info['fill_info']
        g_conn_y = yloc_info['g_conn_y']
        d_conn_y = yloc_info['d_conn_y']

        od_yc = (od_yloc[0] + od_yloc[1]) // 2

        # Compute extension information
        lr_edge_info = EdgeInfo(od_type=od_type, draw_layers={}, y_intv={})

        po_type = 'PO_sub' if is_sub else 'PO'
        po_types = (po_type,) * fg
        mtype = (mos_type, mos_type)
        od_h = self.get_od_h(lch_unit, w)
        ext_top_info = ExtInfo(
            margins=top_margins,
            od_h=od_h,
            imp_min_h=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )
        ext_bot_info = ExtInfo(
            margins=bot_margins,
            od_h=od_h,
            imp_min_h=0,
            mtype=mtype,
            thres=threshold,
            po_types=po_types,
            edgel_info=lr_edge_info,
            edger_info=lr_edge_info,
        )

        # Compute layout information
        lay_info_list = [(lay, 0, blk_yb, blk_yt) for lay in
                         self.get_mos_layers(mos_type, threshold)]
        if dnw_mode:
            lay_info_list.extend(((lay, 0, blk_yt - nw_dnw_ovl, blk_yt) for lay in dnw_layers))

        fill_info_list = [FillInfo(layer=layer, exc_layer=info[0], x_intv_list=[],
                                   y_intv_list=info[1]) for layer, info in fill_info.items()]

        layout_info = dict(
            blk_type=blk_type,
            lch_unit=lch_unit,
            fg=fg,
            arr_y=(blk_yb, blk_yt),
            draw_od=not ds_dummy,
            row_info_list=[RowInfo(od_x_list=[(0, fg)],
                                   od_y=od_yloc,
                                   od_type=(od_type, sub_type),
                                   row_y=(blk_yb, blk_yt),
                                   po_y=po_yloc,
                                   md_y=md_yloc), ],
            lay_info_list=lay_info_list,
            fill_info_list=fill_info_list,
            # edge parameters
            sub_type=sub_type,
            imp_params=None if is_sub else [(mos_type, threshold, blk_yb, blk_yt, blk_yb, blk_yt)],
            is_sub_ring=is_sub_ring,
            dnw_mode='',
            # adjacent block information list
            adj_row_list=[],
            left_blk_info=None,
            right_blk_info=None,
        )

        if substrate_planar and is_sub:
            layout_info['no_po_region'] = set(range(fg))

        # step 8: return results
        if has_cpo:
            po_y = (blk_yb - cpo_h // 2, blk_yt + cpo_h // 2)
        else:
            po_y = po_yloc
        return dict(
            layout_info=layout_info,
            ext_top_info=ext_top_info,
            ext_bot_info=ext_bot_info,
            left_edge_info=(lr_edge_info, []),
            right_edge_info=(lr_edge_info, []),
            sd_yc=od_yc,
            po_y=po_y,
            od_y=od_yloc,
            g_conn_y=g_conn_y,
            d_conn_y=d_conn_y,
        )

    def get_mos_info(self, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]
        sub_type = 'ptap' if mos_type == 'nch' else 'ntap'
        return self._get_mos_blk_info(lch_unit, fg, w, mos_type, sub_type, threshold, **kwargs)

    def get_valid_extension_widths(self, lch_unit, top_ext_info, bot_ext_info, **kwargs):
        # type: (int, ExtInfo, ExtInfo, **kwargs) -> List[int]
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
        guard_ring_nf = kwargs.get('guard_ring_nf', 0)
        ignore_vm = kwargs.get('ignore_vm', False)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_p = mos_constants['mos_pitch']  # type: int
        od_spy_dum = mos_constants.get('od_spy_dum', mos_constants['od_spy'])
        od_nfin_min = mos_constants['od_fill_h'][0]
        imp_od_ency = mos_constants['imp_od_ency']
        has_cpo = mos_constants['has_cpo']
        cpo_h = mos_constants['cpo_h']
        cpo_od_sp = mos_constants['cpo_od_sp']
        cpo_spy = mos_constants['cpo_spy']
        po_spy = mos_constants['po_spy']
        po_od_exty = mos_constants['po_od_exty']
        md_h_min = mos_constants['md_h_min']
        md_od_exty = mos_constants['md_od_exty']
        md_spy = mos_constants['md_spy']
        if guard_ring_nf == 0 or 'od_spy_max_gr' not in mos_constants:
            od_spy_max = mos_constants['od_spy_max']
        else:
            od_spy_max = mos_constants['od_spy_max_gr']

        od_spy_nfin_max = self.get_od_spy_nfin(lch_unit, od_spy_max, round_up=False)

        bot_imp_min_h = bot_ext_info.imp_min_h  # type: int
        top_imp_min_h = top_ext_info.imp_min_h  # type: int

        # step 1: get minimum extension width from vertical spacing rule
        min_ext_h = max(0, -(-(bot_imp_min_h + top_imp_min_h) // fin_p))
        for name, (tm, cur_spy) in top_ext_info.margins.items():
            if not ignore_vm or name not in self.ignore_vm_layers:
                tot_margin = cur_spy - (tm + bot_ext_info.margins[name][0])
                min_ext_h = max(min_ext_h, -(-tot_margin // fin_p))

        # step 2: get maximum extension width without dummy OD
        od_bot_yt = -bot_ext_info.margins['od'][0]
        od_top_yb = top_ext_info.margins['od'][0]
        od_space_nfin = self.get_od_spy_nfin(lch_unit, od_top_yb - od_bot_yt, round_up=True)
        max_ext_w_no_od = od_spy_nfin_max - od_space_nfin

        # step 3: find minimum extension width with dummy OD
        # now, the tricky part is that we need to make sure OD can be drawn in such a way
        # that we can satisfy both minimum implant width constraint and implant-OD enclosure
        # constraint.  Currently, we compute minimum size so we can split implant either above
        # or below OD and they'll both be DRC clean.  This is a little sub-optimal, but
        # makes layout algorithm much much easier.

        # get od_yb_max1, round to fin grid.
        dum_md_yb = -bot_ext_info.margins['md'][0] + md_spy
        od_edge_spy = cpo_h // 2 + cpo_od_sp if has_cpo else po_spy // 2 + po_od_exty
        od_yb_max1 = max(od_edge_spy, dum_md_yb + md_od_exty, od_bot_yt + od_spy_dum)
        od_yb_max1 = self.snap_od_edge(lch_unit, od_yb_max1, False, True)
        # get od_yb_max2, round to fin grid.
        od_yb_max = bot_imp_min_h + imp_od_ency
        od_yb_max = max(od_yb_max1, self.snap_od_edge(lch_unit, od_yb_max, False, True))

        # get od_yt_min1 assuming yt = 0, round to fin grid.
        dum_md_yt = top_ext_info.margins['md'][0] - md_spy
        od_yt_min1 = min(-od_edge_spy, dum_md_yt - md_od_exty, od_top_yb - od_spy_dum)
        od_yt_min1 = self.snap_od_edge(lch_unit, od_yt_min1, True, False)
        # get od_yt_min2, round to fin grid.
        od_yt_min = -top_imp_min_h - imp_od_ency
        od_yt_min = min(od_yt_min1, self.snap_od_edge(lch_unit, od_yt_min, True, False))

        # get minimum extension width from OD related spacing rules
        min_ext_w_od = max(0, self.get_od_h(lch_unit, od_nfin_min) - (od_yt_min - od_yb_max))
        # check to see CPO spacing rule is satisfied
        if has_cpo:
            min_ext_w_od = max(min_ext_w_od, cpo_spy + cpo_h)
        # check to see MD minimum height rule is satisfied
        min_ext_w_od = max(min_ext_w_od, md_h_min - (dum_md_yt - dum_md_yb))
        # round min_ext_w_od to fin grid.
        min_ext_w_od = -(-min_ext_w_od // fin_p)

        if min_ext_w_od <= max_ext_w_no_od + 1:
            # we can transition from no-dummy to dummy seamlessly
            return [min_ext_h]
        else:
            # there exists extension widths such that we need dummies but cannot draw it
            width_list = list(range(min_ext_h, max_ext_w_no_od + 1))
            width_list.append(min_ext_w_od)
            return width_list

    def _get_dummy_od_yloc(self,  # type: MOSTechFinfetBase
                           lch_unit,  # type: int
                           yblk,  # type: int
                           bot_od_h,  # type: Optional[int]
                           bot_od_yt,  # type: Optional[int]
                           top_od_h,  # type: Optional[int]
                           top_od_yb,  # type: Optional[int]
                           **kwargs):
        # type: (...) -> List[Tuple[int, int]]
        """Compute dummy OD Y intervals in extension block.

        This method use fill algorithm to make sure both maximum OD spacing and
        minimum OD density rules are met.
        """
        guard_ring_nf = kwargs.get('guard_ring_nf', 0)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        od_min_density = kwargs.get('od_min_density', mos_constants['od_min_density'])
        od_spy = mos_constants['od_spy']
        od_nfin_min, od_nfin_max = mos_constants['od_fill_h']
        has_cpo = mos_constants['has_cpo']
        po_spy = mos_constants['po_spy']
        po_od_exty = mos_constants['po_od_exty']
        dpo_edge_spy = mos_constants['dpo_edge_spy']

        if guard_ring_nf == 0 or 'od_spy_max_gr' not in mos_constants:
            od_spy_max = mos_constants['od_spy_max']
        else:
            od_spy_max = mos_constants['od_spy_max_gr']

        if not has_cpo:
            od_spy = max(od_spy, 2 * po_od_exty + po_spy)
        od_spy_nfin_min = self.get_od_spy_nfin(lch_unit, od_spy, round_up=True)
        od_spy_nfin_max = self.get_od_spy_nfin(lch_unit, od_spy_max, round_up=False)

        # compute MD/OD locations.
        # check if we can just not draw anything
        if yblk == 0 or (top_od_yb is not None and top_od_yb - bot_od_yt <= od_spy_max):
            return []
        if top_od_yb is not None:
            # we are filling between two rows
            bot_od_fidx = self.get_fin_idx(lch_unit, bot_od_yt, True)
            top_od_fidx = self.get_fin_idx(lch_unit, top_od_yb, False)

            # compute OD fill area needed to meet density
            od_area_adj = (bot_od_h + top_od_h) // 2
            od_area_tot = top_od_yb - bot_od_yt + od_area_adj
            od_area_targ = int(math.ceil(od_area_tot * od_min_density)) - od_area_adj
            od_fin_area_min = self.get_od_h_inverse(lch_unit, od_area_targ, round_up=True)
            od_fin_area_tot = top_od_fidx - bot_od_fidx - 1

            offset = bot_od_fidx + 1
            foe = False
        else:
            # we are filling in empty area
            # compute fill bounds
            fill_yb = dpo_edge_spy
            fill_yt = yblk - dpo_edge_spy
            fill_h = fill_yt - fill_yb

            # check if we can draw anything at all
            dum_h_min = self.get_od_h(lch_unit, od_nfin_min) + 2 * po_od_exty
            # worst case, we will allow po_spy / 2 margin on edge
            if yblk < po_spy + dum_h_min:
                return []
            # check if we can just draw one dummy
            if fill_h < dum_h_min * 2 + po_spy:
                # get od fins. round up to try to meet min edge distance rule
                nfin = min(od_nfin_max, self.get_od_h_inverse(lch_unit, fill_h - 2 * po_od_exty,
                                                              round_up=True))
                od_h = self.get_od_h(lch_unit, nfin)
                od_yb = self.snap_od_edge(lch_unit, (fill_yb + fill_yt - od_h) // 2,
                                          False, round_up=False)
                return [(od_yb, od_yb + od_h)]

            # compute OD fill area needed to meet density
            bot_od_fidx = self.get_fin_idx(lch_unit, fill_yb + po_od_exty, False, round_up=True)
            top_od_fidx = self.get_fin_idx(lch_unit, fill_yt - po_od_exty, True, round_up=False)

            od_area_tot = yblk
            od_area_targ = int(math.ceil(od_area_tot * od_min_density))
            od_fin_area_min = self.get_od_h_inverse(lch_unit, od_area_targ, round_up=True)
            od_fin_area_tot = top_od_fidx - bot_od_fidx + 1

            offset = bot_od_fidx
            foe = True

        od_fin_area_iter = BinaryIterator(od_fin_area_min, od_fin_area_tot + 1)
        # binary search on OD fill area
        while od_fin_area_iter.has_next():
            # compute fill with fin-area target
            od_fin_area_targ_cur = od_fin_area_iter.get_next()
            fill_info = fill_symmetric_min_density_info(od_fin_area_tot, od_fin_area_targ_cur,
                                                        od_nfin_min, od_nfin_max, od_spy_nfin_min,
                                                        sp_max=od_spy_nfin_max, fill_on_edge=foe,
                                                        cyclic=False)
            od_nfin_tot_cur = fill_info[0][0]

            # compute actual OD area
            od_intv_list = fill_symmetric_interval(*fill_info[0][2], offset=offset,
                                                   invert=fill_info[1])[0]
            od_area_cur = sum((self.get_od_h(lch_unit, stop - start)
                               for start, stop in od_intv_list))
            if od_area_cur >= od_area_targ:
                od_fin_area_iter.save_info(od_intv_list)
                od_fin_area_iter.down()
            else:
                if (od_nfin_tot_cur < od_fin_area_targ_cur or
                        od_fin_area_targ_cur == od_fin_area_tot):
                    # we cannot do any better by increasing od_fin_area_targ
                    od_fin_area_iter.save_info(od_intv_list)
                    break
                else:
                    od_fin_area_iter.up()

        # convert fin interval to Y coordinates
        od_intv_list = od_fin_area_iter.get_last_save_info()
        return [(self.get_od_edge(lch_unit, start, False),
                 self.get_od_edge(lch_unit, stop - 1, True)) for start, stop in od_intv_list]

    def _get_dummy_yloc(self, lch_unit, bot_ext_info, top_ext_info, yblk, **kwargs):
        """Compute dummy OD/MD/PO/CPO Y intervals in extension block.

        This method gets OD coordinates from _get_dummy_od_yloc(), then modify the results
        if MD spacing rules are violated.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        od_nfin_min, od_nfin_max = mos_constants['od_fill_h']
        od_spy = mos_constants['od_spy']
        md_h_min = mos_constants['md_h_min']
        md_od_exty = mos_constants['md_od_exty']
        md_spy = mos_constants['md_spy']
        has_cpo = mos_constants['has_cpo']
        cpo_h = mos_constants['cpo_h']
        cpo_od_sp = mos_constants['cpo_od_sp']
        po_od_exty = mos_constants['po_od_exty']

        od_h_min = self.get_od_h(lch_unit, od_nfin_min)

        # compute MD/OD locations.
        bot_md_yt = -bot_ext_info.margins['md'][0]
        top_md_yb = yblk + top_ext_info.margins['md'][0]

        # get dummy OD/MD intervals
        bot_od_yt = -bot_ext_info.margins['od'][0]
        top_od_yb = yblk + top_ext_info.margins['od'][0]
        bot_od_h = bot_ext_info.od_h
        top_od_h = top_ext_info.od_h
        od_y_list = self._get_dummy_od_yloc(lch_unit, yblk, bot_od_h, bot_od_yt,
                                            top_od_h, top_od_yb, **kwargs)
        if not od_y_list:
            po_y_list = [(cpo_h // 2, yblk - cpo_h // 2)] if has_cpo else []
            return [], [], [(0, yblk)], po_y_list, [0, yblk]

        md_y_list = []
        for od_yb, od_yt in od_y_list:
            md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
            md_yb = (od_yb + od_yt - md_h) // 2
            md_y_list.append((md_yb, md_yb + md_h))

        # check and fix MD spacing violation, which can only occur if we have CPO
        if has_cpo:
            cpo_bot_yt = cpo_h // 2
            cpo_top_yb = yblk - cpo_h // 2
            if md_y_list[0][0] < bot_md_yt + md_spy or od_y_list[0][0] < cpo_bot_yt + cpo_od_sp:
                od_yt = od_y_list[0][1]
                od_bot_correct = max(cpo_bot_yt + cpo_od_sp, bot_md_yt + md_spy + md_od_exty)
                od_yb = self.snap_od_edge(lch_unit, od_bot_correct, False, True)
                od_yt = max(od_yb + od_h_min, od_yt)
                od_y_list[0] = od_yb, od_yt
                md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
                md_yb = max((od_yb + od_yt - md_h) // 2, bot_md_yt + md_spy)
                md_y_list[0] = md_yb, md_yb + md_h
            if md_y_list[-1][1] > top_md_yb - md_spy or od_y_list[-1][1] > cpo_top_yb - cpo_od_sp:
                od_yb = od_y_list[-1][0]
                od_top_correct = min(cpo_top_yb - cpo_od_sp, top_md_yb - md_spy - md_od_exty)
                od_yt = self.snap_od_edge(lch_unit, od_top_correct, True, False)
                od_yb = min(od_yt - od_h_min, od_yb)
                od_y_list[-1] = od_yb, od_yt
                md_h = max(md_h_min, od_yt - od_yb + 2 * md_od_exty)
                md_yt = min((od_yb + od_yt + md_h) // 2, top_md_yb - md_spy)
                md_yb = md_yt - md_h
                # here actually MD bottom spacing could be violated again
                if bot_md_yt + md_spy > md_yb:
                    # first check if it's even possible to solve
                    if (md_h > top_md_yb - bot_md_yt - 2 * md_spy or
                            bot_md_yt + md_spy > od_yb - md_od_exty):
                        raise ValueError(
                            'Cannot draw dummy OD and meet MD spacing constraints.  See developer.')
                    md_yb = bot_md_yt + md_spy
                    md_yt = md_yb + md_h
                md_y_list[0] = md_yb, md_yt

            if md_y_list[0][0] < bot_md_yt + md_spy or od_y_list[0][0] < cpo_bot_yt + cpo_od_sp:
                # bottom MD spacing rule violated.  This only happens if we have exactly
                # one dummy OD, and there is no solution that works for both top and bottom
                # MD spacing rules.
                raise ValueError(
                    'Cannot draw dummy OD and meet MD spacing constraints.  See developer.')
            if len(md_y_list) > 1:
                # check inner MD and OD spacing rules are met.
                # I don't think these rules will ever be broken, so I'm not fixing it now.
                # However, if there does need to be a fix, you probably need to recompute
                # inner dummy OD Y coordinates.
                if (md_y_list[0][1] + md_spy > md_y_list[1][0] or
                        md_y_list[-2][1] + md_spy > md_y_list[-1][0]):
                    raise ValueError('inner dummy MD spacing rule not met.  See developer.')
                if (od_y_list[0][1] + od_spy > od_y_list[1][0] or
                        od_y_list[-2][1] + od_spy > od_y_list[1][0]):
                    raise ValueError('inner dummy OD spacing rule not met.  See developer.')

        # get PO/CPO locations
        cpo_yc = 0
        num_dod = len(od_y_list)
        row_y_list, po_y_list, cpo_yc_list = [], [], []
        for idx, (od_yb, od_yt) in enumerate(od_y_list):
            # find next CPO coordinates
            if idx + 1 < num_dod:
                next_cpo_yc = (od_yt + od_y_list[idx + 1][0]) // 2
            else:
                next_cpo_yc = yblk
            # record coordinates
            row_y_list.append((cpo_yc, next_cpo_yc))
            if has_cpo:
                po_y_list.append((cpo_yc + cpo_h // 2, next_cpo_yc - cpo_h // 2))
            else:
                po_y_list.append((od_yb - po_od_exty, od_yt + po_od_exty))
            cpo_yc_list.append(cpo_yc)
            cpo_yc = next_cpo_yc
        # add last CPO
        cpo_yc_list.append(yblk)

        return od_y_list, md_y_list, row_y_list, po_y_list, cpo_yc_list

    def _get_ext_adj_split_info(self, lch_unit, w, bot_ext_info, top_ext_info, od_y_list,
                                cpo_yc_list):
        """Compute adjacent block information and Y split coordinate in extension block."""
        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_p = mos_constants['mos_pitch']
        cpo_spy = mos_constants['cpo_spy']
        imp_od_ency = mos_constants['imp_od_ency']
        has_cpo = mos_constants['has_cpo']
        cpo_h = mos_constants['cpo_h']
        no_sub_dummy = mos_constants.get('no_sub_dummy', False)

        _, top_row_type = top_ext_info.mtype
        _, bot_row_type = bot_ext_info.mtype
        top_is_sub = (top_row_type == 'ptap' or top_row_type == 'ntap')
        bot_is_sub = (bot_row_type == 'ptap' or bot_row_type == 'ntap')
        yt = w * fin_p
        yc = yt // 2

        # check if we draw one or two CPO.  Compute threshold split Y coordinates accordingly.
        cpo2_w = -(-(cpo_spy + cpo_h) // fin_p)  # type: int
        one_cpo = has_cpo and (w < cpo2_w)

        num_dod = len(od_y_list)
        if not od_y_list:
            # no dummy OD
            if one_cpo and w > 0:
                thres_split_y = imp_split_y = yc, yc
                adj_edgel_infos = [bot_ext_info.edgel_info, top_ext_info.edgel_info]
                adj_edger_infos = [bot_ext_info.edger_info, top_ext_info.edger_info]
                adj_row_list = [AdjRowInfo(po_types=bot_ext_info.po_types,
                                           row_y=(0, yc),
                                           po_y=(-(cpo_h // 2), yc - cpo_h // 2),
                                           ),
                                AdjRowInfo(po_types=top_ext_info.po_types,
                                           row_y=(yc, yt),
                                           po_y=(yc + cpo_h // 2, yt + cpo_h // 2),
                                           )]
            else:
                thres_split_y = imp_split_y = 0, yt
                adj_row_list = []
                adj_edgel_infos = []
                adj_edger_infos = []
        else:
            # has dummy OD
            adj_row_list = []
            adj_edgel_infos = []
            adj_edger_infos = []

            if no_sub_dummy:
                if top_is_sub and bot_is_sub:
                    raise ValueError('Impossible to draw extensions; both rows are substrates.')
                all_or_nothing = top_is_sub or bot_is_sub
            else:
                all_or_nothing = False

            if num_dod % 2 == 0 and not all_or_nothing:
                thres_split_y = imp_split_y = yc, yc
            else:
                if all_or_nothing:
                    bod_yb, _ = od_y_list[0]
                    _, tod_yt = od_y_list[-1]
                    imp_split_y = bod_yb - imp_od_ency, tod_yt + imp_od_ency
                    thres_split_y = cpo_yc_list[0], cpo_yc_list[-1]
                else:
                    mid_od_idx = num_dod // 2
                    od_yb, od_yt = od_y_list[mid_od_idx]
                    imp_split_y = od_yb - imp_od_ency, od_yt + imp_od_ency
                    thres_split_y = cpo_yc_list[mid_od_idx], cpo_yc_list[mid_od_idx + 1]

        return adj_row_list, adj_edgel_infos, adj_edger_infos, thres_split_y, imp_split_y

    def get_ext_info(self, lch_unit, w, fg, top_ext_info, bot_ext_info, **kwargs):
        # type: (int, int, int, ExtInfo, ExtInfo, **kwargs) -> Dict[str, Any]
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
           split at middle, draw more dummy OD on transistor side.
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
        imp_layers_info_struct = mos_constants['imp_layers']
        thres_layers_info_struct = mos_constants['thres_layers']
        fin_p = mos_constants['mos_pitch']
        has_cpo = mos_constants['has_cpo']
        cpo_spy = mos_constants['cpo_spy']
        cpo_h = mos_constants['cpo_h']
        cpo_h_end = mos_constants['cpo_h_end']
        sd_pitch = mos_constants['sd_pitch']
        od_spx = mos_constants['od_spx']
        od_fill_w_max = mos_constants['od_fill_w_max']
        substrate_planar = mos_constants.get('substrate_planar', False)
        cpo_po_ency = mos_constants['cpo_po_ency']

        yt = w * fin_p
        yc = yt // 2

        top_mtype, top_row_type = top_ext_info.mtype
        bot_mtype, bot_row_type = bot_ext_info.mtype
        top_thres = top_ext_info.thres
        bot_thres = bot_ext_info.thres
        bot_tran = (bot_row_type == 'nch' or bot_row_type == 'pch')
        top_tran = (top_row_type == 'nch' or top_row_type == 'pch')
        bot_imp = 'nch' if bot_row_type == 'nch' or bot_row_type == 'ptap' else 'pch'
        top_imp = 'nch' if top_row_type == 'nch' or top_row_type == 'ptap' else 'pch'

        # get dummy fill locations
        tmp = self._get_dummy_yloc(lch_unit, bot_ext_info, top_ext_info, yt, **kwargs)
        od_y_list, md_y_list, row_y_list, po_y_list, cpo_yc_list = tmp
        # get adjacent block/split information
        tmp = self._get_ext_adj_split_info(lch_unit, w, bot_ext_info, top_ext_info,
                                           od_y_list, cpo_yc_list)
        adj_row_list, adj_edgel_infos, adj_edger_infos, thres_split_y, imp_split_y = tmp

        # check if we draw one or two CPO
        cpo2_w = -(-(cpo_spy + cpo_h) // fin_p)  # type: int
        one_cpo = has_cpo and (w < cpo2_w)

        lay_info_list = []
        num_dod = len(od_y_list)
        cpo_lay = mos_layer_table['CPO']
        if not od_y_list:
            # no dummy OD
            lr_edge_info = EdgeInfo(od_type=None, draw_layers={}, y_intv={})
            od_x_list = []
            od_y_list = md_y_list = [(0, 0)]
            if one_cpo:
                cpo_yc_list = [yc]
                row_y_list = po_y_list = [(0, 0)]
            else:
                cpo_yc_list = [0, yt]
                thres_split_y = imp_split_y = 0, yt
        else:
            # has dummy OD
            lr_edge_info = EdgeInfo(od_type='dum', draw_layers={}, y_intv={})
            # get OD horizontal partitioning
            if od_fill_w_max is None or fg == 1:
                # force dummy OD if 1 finger, so that Laygo extensions will have dummy OD.
                od_x_list = [(0, fg)]
            else:
                od_fg_min = self.get_analog_unit_fg()
                od_fg_max = (od_fill_w_max - lch_unit) // sd_pitch - 1
                od_spx_fg = self.get_od_spx_fg(lch_unit, od_spx) + 2
                od_x_list = fill_symmetric_max_density(fg, fg, od_fg_min, od_fg_max, od_spx_fg,
                                                       fill_on_edge=True, cyclic=False)[0]

        # compute implant and threshold layer information
        # figure out where to separate top/bottom implant/threshold.
        add_row_yb = add_row_yt = add_po_yb = add_po_yt = 0
        if bot_imp == top_imp:
            sub_type = 'ptap' if bot_imp == 'nch' else 'ntap'
            if bot_tran != top_tran:
                # case 1
                if bot_tran:
                    if substrate_planar:
                        cpo_yb = cpo_yc_list[-1] - (cpo_h // 2)
                        cpo_yt = cpo_yb + cpo_h_end
                        if one_cpo:
                            add_row_yb = 0
                            add_po_yb = -cpo_h // 2
                            add_po_yt = cpo_yb
                        else:
                            add_row_yb = cpo_yc_list[-1]
                            add_po_yb = add_po_yt = 0
                        add_row_yt = cpo_yb + cpo_po_ency
                        adj_edgel_infos = [bot_ext_info.edgel_info]
                        adj_edger_infos = [bot_ext_info.edger_info]
                        imp_ysep = thres_ysep = max(add_row_yt, (cpo_yb + cpo_yt) // 2)
                    else:
                        imp_ysep = imp_split_y[1]
                        thres_ysep = thres_split_y[1]
                else:
                    if substrate_planar:
                        cpo_yt = cpo_yc_list[0] + cpo_h // 2
                        cpo_yb = cpo_yt - cpo_h_end
                        if one_cpo:
                            add_row_yt = yt
                            add_po_yt = yt + (cpo_h // 2)
                            add_po_yb = cpo_yt
                        else:
                            add_row_yt = cpo_yc_list[0]
                            add_po_yb = add_po_yt = 0
                        add_row_yb = cpo_yt - cpo_po_ency
                        adj_edgel_infos = [top_ext_info.edgel_info]
                        adj_edger_infos = [top_ext_info.edger_info]
                        imp_ysep = thres_ysep = min(add_row_yb, (cpo_yb + cpo_yt) // 2)
                    else:
                        imp_ysep = imp_split_y[0]
                        thres_ysep = thres_split_y[0]
            elif bot_tran:
                # case 3
                sep_idx = 0 if bot_thres <= top_thres else 1
                if num_dod > 0:
                    bot_mtype = top_mtype = bot_imp
                imp_ysep = imp_split_y[sep_idx]
                thres_ysep = thres_split_y[sep_idx]
            else:
                # case 2
                if substrate_planar:
                    cpo_yc_list = []
                sep_idx = 0 if bot_thres <= top_thres else 1
                imp_ysep = imp_split_y[sep_idx]
                thres_ysep = thres_split_y[sep_idx]
        else:
            sub_type = None
            if bot_tran != top_tran:
                # case 5
                if bot_tran:
                    if substrate_planar:
                        cpo_yb = cpo_yc_list[-1] - (cpo_h // 2)
                        cpo_yt = cpo_yb + cpo_h_end
                        if one_cpo:
                            add_row_yb = 0
                            add_po_yb = -cpo_h // 2
                            add_po_yt = cpo_yb
                        else:
                            add_row_yb = cpo_yc_list[-1]
                            add_po_yb = add_po_yt = 0
                        add_row_yt = cpo_yb + cpo_po_ency
                        adj_edgel_infos = [bot_ext_info.edgel_info]
                        adj_edger_infos = [bot_ext_info.edger_info]
                        imp_ysep = thres_ysep = max(add_row_yt, (cpo_yb + cpo_yt) // 2)
                    else:
                        top_mtype = bot_imp
                        top_thres = bot_thres
                        imp_ysep = imp_split_y[1]
                        thres_ysep = thres_split_y[1]
                else:
                    if substrate_planar:
                        cpo_yt = cpo_yc_list[0] + cpo_h // 2
                        cpo_yb = cpo_yt - cpo_h_end
                        if one_cpo:
                            add_row_yt = yt
                            add_po_yt = yt + (cpo_h // 2)
                            add_po_yb = cpo_yt
                        else:
                            add_row_yt = cpo_yc_list[0]
                            add_po_yb = add_po_yt = 0
                        add_row_yb = cpo_yt - cpo_po_ency
                        adj_edgel_infos = [top_ext_info.edgel_info]
                        adj_edger_infos = [top_ext_info.edger_info]
                        imp_ysep = thres_ysep = min(add_row_yb, (cpo_yb + cpo_yt) // 2)
                    else:
                        bot_mtype = top_imp
                        bot_thres = top_thres
                        imp_ysep = imp_split_y[0]
                        thres_ysep = thres_split_y[0]
            elif bot_tran:
                # case 6
                bot_mtype = bot_imp
                top_mtype = top_imp
                sep_idx = 1 if bot_imp == 'nch' else 0
                imp_ysep = imp_split_y[sep_idx]
                thres_ysep = thres_split_y[sep_idx]
            else:
                # case 4
                if substrate_planar:
                    cpo_yc_list = []
                sep_idx = 1 if bot_imp == 'nch' else 0
                imp_ysep = imp_split_y[sep_idx]
                thres_ysep = thres_split_y[sep_idx]

        # add implant layers
        imp_params = [(bot_mtype, bot_thres, 0, imp_ysep, 0, thres_ysep),
                      (top_mtype, top_thres, imp_ysep, yt, thres_ysep, yt)]

        for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
            imp_layers_info = imp_layers_info_struct[mtype]
            thres_layers_info = thres_layers_info_struct[mtype][thres]
            for cur_yb, cur_yt, lay_info in [(imp_yb, imp_yt, imp_layers_info),
                                             (thres_yb, thres_yt, thres_layers_info)]:

                for lay_name in lay_info:
                    lay_info_list.append((lay_name, 0, cur_yb, cur_yt))

        # add CPO layers
        if has_cpo:
            for idx, cpo_yc in enumerate(cpo_yc_list):
                # find next CPO coordinates
                if substrate_planar:
                    if idx == 0 and not bot_tran:
                        cur_cpo_yt = cpo_yc + cpo_h // 2
                        cur_cpo_yb = cur_cpo_yt - cpo_h_end
                    elif idx == len(cpo_yc_list) - 1 and not top_tran:
                        cur_cpo_yb = cpo_yc - cpo_h // 2
                        cur_cpo_yt = cur_cpo_yb + cpo_h_end
                    else:
                        cur_cpo_yb = cpo_yc - cpo_h // 2
                        cur_cpo_yt = cpo_yc + cpo_h // 2
                else:
                    cur_cpo_yb = cpo_yc - cpo_h // 2
                    cur_cpo_yt = cpo_yc + cpo_h // 2
                lay_info_list.append((cpo_lay, 0, cur_cpo_yb, cur_cpo_yt))

        # modify adjacent row geometries if substrate planar is true,
        # and we are next to a substrate row.
        if substrate_planar and (not bot_tran or not top_tran):
            if add_row_yt > add_row_yb:
                po_type = 'PO' if one_cpo else 'PO_dummy'
                adj_row_list = [AdjRowInfo(row_y=(add_row_yb, add_row_yt),
                                           po_y=(add_po_yb, add_po_yt),
                                           po_types=(po_type,) * fg)]
            else:
                adj_row_list = adj_edgel_infos = adj_edger_infos = []

        # construct row_info_list, now we know where the implant splits
        row_info_list = []
        for od_y, row_y, po_y, md_y in zip(od_y_list, row_y_list, po_y_list, md_y_list):
            cur_mtype = bot_mtype if max(od_y[0], od_y[1]) < imp_ysep else top_mtype
            cur_sub_type = 'ptap' if cur_mtype == 'nch' or cur_mtype == 'ptap' else 'ntap'
            row_info_list.append(RowInfo(od_x_list=od_x_list, od_y=od_y,
                                         od_type=('dum', cur_sub_type),
                                         row_y=row_y, po_y=po_y, md_y=md_y))

        # create layout information dictionary
        between_gr = (top_row_type == 'ntap' and bot_row_type == 'ptap') or \
                     (top_row_type == 'ptap' and bot_row_type == 'ntap')
        layout_info = dict(
            blk_type='ext',
            lch_unit=lch_unit,
            fg=fg,
            arr_y=(0, yt),
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=lay_info_list,
            # TODO: figure out how to do fill in extension block.
            fill_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=imp_params,
            is_sub_ring=False,
            between_gr=between_gr,
            dnw_mode='',
            # adjacent block information list
            adj_row_list=adj_row_list,
            left_blk_info=None,
            right_blk_info=None,
        )

        return dict(
            layout_info=layout_info,
            sub_ysep=(imp_ysep, thres_ysep),
            left_edge_info=(lr_edge_info, adj_edgel_infos),
            right_edge_info=(lr_edge_info, adj_edger_infos),
        )

    def get_sub_ring_ext_info(self, sub_type, height, fg, end_ext_info, **kwargs):
        # type: (str, int, int, ExtInfo, **kwargs) -> Dict[str, Any]
        dnw_mode = kwargs.get('dnw_mode', '')

        lch = self.get_substrate_ring_lch()
        lch_unit = int(round(lch / self.config['layout_unit'] / self.res))

        tmp = self._get_dummy_yloc(lch_unit, end_ext_info, end_ext_info, height)
        od_y_list, md_y_list, row_y_list, po_y_list, cpo_yc_list = tmp
        tmp = self._get_ext_adj_split_info(lch_unit, height, end_ext_info, end_ext_info,
                                           od_y_list, cpo_yc_list)
        adj_row_list, adj_edgel_infos, adj_edger_infos, _, _ = tmp

        # construct row_info_list
        row_info_list = []
        for od_y, md_y, row_y, po_y in zip(od_y_list, md_y_list, row_y_list, po_y_list):
            row_info_list.append(RowInfo(od_x_list=(0, 0), od_type=('dum', sub_type),
                                         od_y=od_y, row_y=row_y, po_y=po_y, md_y=md_y))

        lr_edge_info = EdgeInfo(od_type='dum', draw_layers={}, y_intv={})
        layout_info = dict(
            blk_type='ext_subring',
            lch_unit=lch_unit,
            fg=fg,
            arr_y=(0, height),
            draw_od=False,
            row_info_list=row_info_list,
            lay_info_list=[],
            # TODO: figure out how to do fill in extension block.
            fill_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=[(sub_type, end_ext_info.thres, 0, height, 0, height), ],
            is_sub_ring=True,
            dnw_mode=dnw_mode,
            # adjacent block information list
            adj_row_list=adj_row_list,
            left_blk_info=None,
            right_blk_info=None,
        )

        return dict(
            layout_info=layout_info,
            left_edge_info=(lr_edge_info, adj_edgel_infos),
            right_edge_info=(lr_edge_info, adj_edger_infos),
        )

    def get_substrate_info(self, lch_unit, w, sub_type, threshold, fg, blk_pitch=1, **kwargs):
        # type: (int, int, str, str, int, int, **kwargs) -> Dict[str, Any]
        return self._get_mos_blk_info(lch_unit, fg, w, sub_type, sub_type, threshold,
                                      blk_pitch=blk_pitch, **kwargs)

    def _get_end_blk_info(self, lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **kwargs):
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
        is_sub_ring = kwargs.get('is_sub_ring', False)
        dnw_mode = kwargs.get('dnw_mode', '')
        end_ext_info = kwargs.get('end_ext_info', None)

        is_sub_ring_end = (end_ext_info is not None)

        dnw_margins = self.config['dnw_margins']
        mos_layer_table = self.config['mos_layer_table']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        has_cpo = mos_constants['has_cpo']
        cpo_h = mos_constants['cpo_h']
        cpo_h_end = mos_constants['cpo_h_end']
        cpo_po_ency = mos_constants['cpo_po_ency']
        nw_dnw_ext = mos_constants['nw_dnw_ext']
        edge_margin = mos_constants['edge_margin']
        substrate_planar = mos_constants.get('substrate_planar', False)

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2
        if dnw_mode and not is_sub_ring_end:
            edge_margin = dnw_margins[dnw_mode] - nw_dnw_ext

        lr_edge_info = EdgeInfo(od_type='sub', draw_layers={}, y_intv={})
        cpo_lay = mos_layer_table['CPO']
        finbound_lay = mos_layer_table['FB']
        if is_end:
            blk_pitch = lcm([blk_pitch, fin_p])
            # first assume top Y coordinate is 0
            if substrate_planar or not has_cpo:
                arr_yt = -(-edge_margin // blk_pitch) * blk_pitch
                imp_yb = arr_yt
                finbound_yb = arr_yt - fin_p2 - fin_h2
                lay_info_list = []
                adj_row_list = []
                adj_edge_infos = []
            else:
                arr_yt = 0
                cpo_bot_yt = arr_yt + cpo_h // 2
                cpo_bot_yb = cpo_bot_yt - cpo_h_end
                finbound_yb = arr_yt - fin_p2 - fin_h2
                po_yb = cpo_bot_yt - cpo_po_ency
                imp_yb = min(po_yb, (cpo_bot_yt + cpo_bot_yb) // 2)
                min_yb = min(finbound_yb, cpo_bot_yb, imp_yb - edge_margin)
                # make sure all layers are in first quadrant
                if is_sub_ring_end:
                    yshift = -min_yb
                else:
                    yshift = -(min_yb // blk_pitch) * blk_pitch
                arr_yt += yshift
                cpo_bot_yt += yshift
                cpo_bot_yb += yshift
                finbound_yb += yshift
                lay_info_list = [(cpo_lay, 0, cpo_bot_yb, cpo_bot_yt), ]
                cpo_bot_yc = (cpo_bot_yb + cpo_bot_yt) // 2
                po_yt = arr_yt
                po_yb = cpo_bot_yt - cpo_po_ency
                imp_yb = min(po_yb, cpo_bot_yc)
                if po_yt > po_yb:
                    adj_row_list = [AdjRowInfo(row_y=(po_yb, po_yt), po_y=(0, 0),
                                               po_types=('PO_sub',) * fg)]
                    adj_edge_infos = [lr_edge_info]
                else:
                    adj_row_list = []
                    adj_edge_infos = []

            finbound_yt = arr_yt + fin_p2 + fin_h2
            for lay in self.get_mos_layers(sub_type, threshold):
                if lay == finbound_lay:
                    yb, yt = finbound_yb, finbound_yt
                else:
                    yb, yt = imp_yb, arr_yt
                if yt > yb:
                    lay_info_list.append((lay, 0, yb, yt))
        else:
            # we just draw CPO
            imp_yb = arr_yt = 0
            if substrate_planar or not has_cpo:
                lay_info_list = []
            else:
                lay_info_list = [(cpo_lay, 0, -cpo_h // 2, cpo_h // 2)]
            adj_row_list = []
            adj_edge_infos = []

        blk_type = 'end_subring' if is_sub_ring_end else 'end'
        layout_info = dict(
            blk_type=blk_type,
            lch_unit=lch_unit,
            fg=fg,
            arr_y=(0, arr_yt),
            draw_od=True,
            row_info_list=[],
            lay_info_list=lay_info_list,
            fill_info_list=[],
            # edge parameters
            sub_type=sub_type,
            imp_params=None,
            is_sub_ring=is_sub_ring,
            dnw_mode=dnw_mode,
            # adjacent block information list
            adj_row_list=adj_row_list,
            left_blk_info=None,
            right_blk_info=None,
        )

        ans = dict(
            layout_info=layout_info,
            sub_ysep=(imp_yb, imp_yb),
            left_edge_info=(lr_edge_info, adj_edge_infos),
            right_edge_info=(lr_edge_info, adj_edge_infos),
        )
        if is_sub_ring_end:
            ans['ext_info'] = ExtInfo(
                margins={key: (val[0] + arr_yt, val[1])
                         for key, val in end_ext_info.margins.items()},
                od_h=end_ext_info.od_h,
                imp_min_h=0,
                mtype=end_ext_info.mtype,
                thres=threshold,
                po_types=('PO_sub',) * fg,
                edgel_info=lr_edge_info,
                edger_info=lr_edge_info,
            )

        return ans

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
        return self._get_end_blk_info(lch_unit, sub_type, threshold, fg, is_end,
                                      blk_pitch, **kwargs)

    def get_sub_ring_end_info(self, sub_type, threshold, fg, end_ext_info, **kwargs):
        # type: (str, str, int, ExtInfo, **kwargs) -> Dict[str, Any]
        """Empty block, just reserve space for margin."""
        lch = self.get_substrate_ring_lch()
        lch_unit = int(round(lch / self.config['layout_unit'] / self.res))

        kwargs['end_ext_info'] = end_ext_info
        return self._get_end_blk_info(lch_unit, sub_type, threshold, fg, True, 1, **kwargs)

    def get_outer_edge_info(self, guard_ring_nf, layout_info, is_end, adj_blk_info):
        # type: (int, Dict[str, Any], bool, Optional[Any]) -> Dict[str, Any]
        mos_layer_table = self.config['mos_layer_table']
        imp_layers_info_struct = self.mos_config['imp_layers']
        thres_layers_info_struct = self.mos_config['thres_layers']

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        arr_y = layout_info['arr_y']
        row_info_list = layout_info['row_info_list']
        lay_info_list = layout_info['lay_info_list']
        adj_row_list = layout_info['adj_row_list']
        imp_params = layout_info['imp_params']
        dnw_mode = layout_info['dnw_mode']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        imp_edge_dx = mos_constants.get('imp_edge_dx', {})
        sd_pitch = mos_constants['sd_pitch']
        substrate_planar = mos_constants.get('substrate_planar', False)

        edge_info = self.get_edge_info(lch_unit, guard_ring_nf, is_end, dnw_mode=dnw_mode)
        fg_outer = edge_info['fg_outer']
        cpo_xl = edge_info['cpo_xl']

        # compute new lay_info_list
        cpo_lay = mos_layer_table['CPO']
        if guard_ring_nf == 0 or imp_params is None:
            # we keep all implant layers, just update CPO left coordinate.
            new_lay_list = []
            for lay, _, yb, yt in lay_info_list:
                if lay == cpo_lay:
                    new_lay_list.append((lay, cpo_xl, yb, yt))
                elif lay in imp_edge_dx:
                    offset, lch_scale, sd_scale = imp_edge_dx[lay]
                    cur_xl = (offset + int(round(lch_scale * lch_unit)) +
                              int(round(sd_scale * sd_pitch)))
                    new_lay_list.append((lay, cur_xl, yb, yt))
                else:
                    new_lay_list.append((lay, 0, yb, yt))
        else:
            # we need to convert implant layers to substrate implants
            # first, get CPO layers
            new_lay_list = []
            for lay, _, yb, yt in lay_info_list:
                if lay == cpo_lay:
                    new_lay_list.append((lay, cpo_xl, yb, yt))
            # get new implant layers
            for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
                sub_type = 'ptap' if mtype == 'nch' or mtype == 'ptap' else 'ntap'
                imp_layers_info = imp_layers_info_struct[sub_type]
                thres_layers_info = thres_layers_info_struct[sub_type][thres]
                for cur_yb, cur_yt, lay_info in [(imp_yb, imp_yt, imp_layers_info),
                                                 (thres_yb, thres_yt, thres_layers_info)]:
                    if cur_yt > cur_yb:
                        for lay_name in lay_info:
                            if lay_name in imp_edge_dx:
                                offset, lch_scale, sd_scale = imp_edge_dx[lay_name]
                                cur_xl = (offset + int(round(lch_scale * lch_unit)) +
                                          int(round(sd_scale * sd_pitch)))
                                new_lay_list.append((lay_name, cur_xl, cur_yb, cur_yt))
                            else:
                                new_lay_list.append((lay_name, 0, cur_yb, cur_yt))

        # compute new row_info_list
        # noinspection PyProtectedMember
        row_info_list = [rinfo._replace(od_x_list=[]) for rinfo in row_info_list]

        # compute new adj_row_list
        if adj_blk_info is None:
            adj_blk_info = (None, [None] * len(adj_row_list))

        # change PO type in adjacent row geometries
        new_adj_row_list = []
        if fg_outer > 0:
            for adj_edge_info, adj_info in zip(adj_blk_info[1], adj_row_list):
                if adj_edge_info is not None:
                    po_types = ('PO_dummy',) * (fg_outer - 1)
                    adj_od_type = adj_edge_info.od_type
                    if adj_od_type == 'mos':
                        po_types += ('PO_edge',)
                    elif adj_od_type == 'sub':
                        po_types += ('PO_edge_sub',)
                    elif adj_od_type == 'dum':
                        po_types += ('PO_edge_dummy',)
                    else:
                        po_types += ('PO_dummy',)
                else:
                    po_types = ('PO_dummy',) * fg_outer
                # noinspection PyProtectedMember
                new_adj_row_list.append(adj_info._replace(po_types=po_types))

        layout_info = dict(
            blk_type='edge' if guard_ring_nf == 0 else 'gr_edge',
            lch_unit=lch_unit,
            fg=fg_outer,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=new_lay_list,
            # TODO: figure out how to draw fill in outer edge block
            fill_info_list=[],
            # adjacent block information
            adj_row_list=new_adj_row_list,
            left_blk_info=EdgeInfo(od_type=None, draw_layers={}, y_intv={}),
            right_blk_info=adj_blk_info[0],
        )

        if blk_type == 'sub' and substrate_planar:
            layout_info['no_po_region'] = set(range(fg_outer))
            layout_info['no_md_region'] = set(range(fg_outer + 1))

        return layout_info

    def get_gr_sub_info(self, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]
        mos_layer_table = self.config['mos_layer_table']

        imp_layers_info_struct = self.mos_config['imp_layers']
        thres_layers_info_struct = self.mos_config['thres_layers']
        dnw_layers = self.mos_config['dnw_layers']
        nw_dnw_ovl = self.mos_config['nw_dnw_ovl']

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        arr_y = layout_info['arr_y']
        lay_info_list = layout_info['lay_info_list']
        row_info_list = layout_info['row_info_list']
        fill_info_list = layout_info['fill_info_list']
        adj_row_list = layout_info['adj_row_list']
        imp_params = layout_info['imp_params']
        dnw_mode = layout_info['dnw_mode']
        between_gr = layout_info.get('between_gr', False)

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        substrate_planar = mos_constants.get('substrate_planar', False)

        edge_info = self.get_edge_info(lch_unit, guard_ring_nf, True, dnw_mode=dnw_mode)
        fg_gr_sub = edge_info['fg_gr_sub']
        fg_od_margin = edge_info['fg_od_margin']

        # compute new row_info_list

        od_x_list = [(fg_od_margin, fg_od_margin + guard_ring_nf)]
        # noinspection PyProtectedMember
        row_info_list = [rinfo._replace(od_x_list=od_x_list, od_type=('sub', rinfo.od_type[1]))
                         for rinfo in row_info_list]
        if substrate_planar:
            if blk_type == 'sub':
                edge_blk_type = 'gr_sub_sub'
            elif blk_type == 'end' or between_gr:
                edge_blk_type = 'gr_sub_end'
                row_info_list = []
            else:
                edge_blk_type = 'gr_sub'
        else:
            edge_blk_type = 'gr_sub'

        # compute new lay_info_list
        cpo_lay = mos_layer_table['CPO']
        wblk = fg_gr_sub * sd_pitch
        if imp_params is None:
            # copy implant layers, but update left coordinate of DNW layers
            new_lay_list = []
            for lay_name, xl, yb, yt in lay_info_list:
                if lay_name in dnw_layers:
                    new_lay_list.append((lay_name, wblk - nw_dnw_ovl, yb, yt))
                else:
                    if not (substrate_planar and lay_name == cpo_lay):
                        new_lay_list.append((lay_name, xl, yb, yt))
        else:
            # we need to convert implant layers to substrate implants
            # first, get all CPO layers
            if substrate_planar:
                new_lay_list = []
            else:
                new_lay_list = [lay_info for lay_info in lay_info_list if lay_info[0] == cpo_lay]
            # compute substrate implant layers
            for mtype, thres, imp_yb, imp_yt, thres_yb, thres_yt in imp_params:
                sub_type = 'ptap' if mtype == 'nch' or mtype == 'ptap' else 'ntap'
                imp_layers_info = imp_layers_info_struct[sub_type]
                thres_layers_info = thres_layers_info_struct[sub_type][thres]
                for cur_yb, cur_yt, lay_info in [(imp_yb, imp_yt, imp_layers_info),
                                                 (thres_yb, thres_yt, thres_layers_info)]:
                    for lay_name in lay_info:
                        new_lay_list.append((lay_name, 0, cur_yb, cur_yt))
            # add DNW layers
            if dnw_mode:
                # add DNW layers
                # NOTE: since substrate has imp_params = None, if we're here we know that we're not
                # next to substrate, so DNW should span the entire height of this template
                for lay_name in dnw_layers:
                    new_lay_list.append((lay_name, wblk - nw_dnw_ovl, 0, arr_y[1]))

        # compute new adj_row_list
        po_types = ('PO_dummy',) * (fg_od_margin - 1) + ('PO_edge',) + \
                   ('PO_sub',) * guard_ring_nf + ('PO_edge',) + ('PO_dummy',) * (fg_od_margin - 1)
        # noinspection PyProtectedMember
        new_adj_row_list = [ar_info._replace(po_types=po_types) for ar_info in adj_row_list]

        # compute new fill information
        # noinspection PyProtectedMember
        fill_info_list = [f._replace(x_intv_list=[]) for f in fill_info_list]

        layout_info = dict(
            blk_type=edge_blk_type,
            lch_unit=lch_unit,
            fg=fg_gr_sub,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=row_info_list,
            lay_info_list=new_lay_list,
            fill_info_list=fill_info_list,
            # adjacent block information list
            adj_row_list=new_adj_row_list,
            left_blk_info=None,
            right_blk_info=None,
        )

        if substrate_planar:
            layout_info['no_po_region'] = set(range(fg_gr_sub))
            layout_info['no_md_region'] = set((idx for idx in range(fg_gr_sub + 1)
                                               if (idx < fg_od_margin or
                                                   idx > fg_od_margin + guard_ring_nf)))
        return layout_info

    def get_gr_sep_info(self, layout_info, adj_blk_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]
        mos_layer_table = self.config['mos_layer_table']

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        arr_y = layout_info['arr_y']
        lay_info_list = layout_info['lay_info_list']
        row_info_list = layout_info['row_info_list']
        adj_row_list = layout_info['adj_row_list']
        is_sub_ring = layout_info['is_sub_ring']
        dnw_mode = layout_info['dnw_mode']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fg_gr_min = mos_constants['fg_gr_min']
        substrate_planar = mos_constants.get('substrate_planar', False)

        edge_constants = self.get_edge_info(lch_unit, fg_gr_min, True, is_sub_ring=is_sub_ring,
                                            dnw_mode=dnw_mode)
        fg_gr_sep = edge_constants['fg_gr_sep']
        cpo_xl = edge_constants['cpo_xl']

        # compute new row_info_list
        if substrate_planar:
            cpo_lay = mos_layer_table['CPO']
            new_lay_list = [(lay, cpo_xl, yb, yt) if lay == cpo_lay else (lay, 0, yb, yt)
                            for lay, _, yb, yt in lay_info_list]
            if blk_type == 'sub':
                od_x_list = [(0, fg_gr_sep)]
            else:
                od_x_list = []
        else:
            new_lay_list = lay_info_list
            od_x_list = []
        # noinspection PyProtectedMember
        new_row_list = [rinfo._replace(od_x_list=od_x_list) for rinfo in row_info_list]

        # compute new adj_row_list
        new_adj_list = []
        for adj_edge_info, adj_info in zip(adj_blk_info[1], adj_row_list):
            po_types = ('PO_dummy',) * (fg_gr_sep - 1)
            adj_od_type = adj_edge_info.od_type
            if adj_od_type == 'mos':
                po_types += ('PO_edge',)
            elif adj_od_type == 'sub':
                po_types += ('PO_edge_sub',)
            elif adj_od_type == 'dum':
                po_types += ('PO_edge_dummy',)
            else:
                po_types += ('PO_dummy',)
            # noinspection PyProtectedMember
            new_adj_list.append(adj_info._replace(po_types=po_types))

        layout_info = dict(
            blk_type='gr_sep',
            lch_unit=lch_unit,
            fg=fg_gr_sep,
            arr_y=arr_y,
            draw_od=True,
            row_info_list=new_row_list,
            lay_info_list=new_lay_list,
            # TODO: figure out how to compute fill information
            fill_info_list=[],
            # adjacent block information list
            adj_row_list=new_adj_list,
            left_blk_info=None,
            right_blk_info=adj_blk_info[0],
        )

        if substrate_planar and blk_type == 'sub':
            layout_info['no_po_region'] = set(range(fg_gr_sep))
            layout_info['no_md_region'] = set(range(fg_gr_sep + 1))
        return layout_info

    # noinspection PyMethodMayBeStatic
    def draw_mos_rect(self, template, layer, bbox):
        # type: (TemplateBase, Tuple[str, str], BBox) -> None
        """This method draws the given transistor layer geometry.

        The default implementation is to just call the add_rect() method.  However, if the
        technology requires more complex handling, you can override this method to implement
        the correct drawing behavior.

        Parameters
        ----------
        template : TemplateBase
            the template.
        layer : Tuple[str, str]
            the layer/purpose pair.
        bbox : BBox
            the geometry bounding box.
        """
        template.add_rect(layer, bbox)

    def draw_od(self, template, od_type, bbox):
        # type: (TemplateBase, str, BBox) -> None
        """This method draws a transistor OD.

        By default, this method just calls draw_mos_rect() on the OD layer.

        Parameters
        ----------
        template : TemplateBase
            the template.
        od_type : str
            the OD type.
        bbox : BBox
            the geometry bounding box.
        """
        mos_layer_table = self.config['mos_layer_table']
        layer = mos_layer_table[od_type]
        self.draw_mos_rect(template, layer, bbox)

    # noinspection PyUnusedLocal
    def draw_poly(self,  # type: MOSTechFinfetBase
                  template,  # type: TemplateBase
                  mos_constants,  # type: Dict[str, Any]
                  po_type,  # type: str
                  po_x,  # type: Tuple[int, int]
                  row_y,  # type: Tuple[int, int]
                  po_y,  # type: Tuple[int, int]
                  od_y,  # type: Tuple[int, int]
                  ):
        # type: (...) -> None
        """This method draws a transistor poly.

        By default, this method does the following:

        1. draws a PO on row_y coordinates.
        2. if the PO is an edge PO or a substrate PO, then draw PODE on od_y coordinates.

        Parameters
        ----------
        template : TemplateBase
            the template.
        mos_constants : Dict[str, Any]
            the transistor constants dictionary
        po_type : str
            the PO type.
        po_x : Tuple[int, int]
            the PO X bounds.
        row_y : Tuple[int, int]
            the row Y bounds.
        po_y : Tuple[int, int]
            the PO Y bounds outside of CPO.
        od_y : Tuple[int, int]
            the OD Y bounds that intersects this PO.
        """
        mos_layer_table = self.config['mos_layer_table']
        has_cpo = mos_constants['has_cpo']

        po_lay = mos_layer_table[po_type]
        res = template.grid.resolution

        po_xl, po_xr = po_x
        if has_cpo:
            template.add_rect(po_lay, BBox(po_xl, row_y[0], po_xr, row_y[1], res, unit_mode=True))
        else:
            template.add_rect(po_lay, BBox(po_xl, po_y[0], po_xr, po_y[1], res, unit_mode=True))

        od_yb, od_yt = od_y
        if od_yt > od_yb and ('sub' in po_type or
                              ('edge' in po_type and po_type != 'PO_edge_dummy')):
            pode_lay = mos_layer_table.get('PODE', None)
            if pode_lay is not None:
                template.add_rect(pode_lay, BBox(po_xl, od_yb, po_xr, od_yt, res, unit_mode=True))

    def draw_mos(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor related layout.

        the layout information dictionary should contain the following entries:

        blk_type
            a string describing the type of this block.
        lch_unit
            channel length in resolution units
        fg
            the width of this template in number of fingers
        arr_y
            array box Y coordinates as two-element integer tuple.
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
                A list of OD X intervals in finger index.
            od_type
                a tuple of (OD type, substrate type).  OD type is one of 'mos', 'sub', or 'dum'.
            od_y
                OD Y interval.
            row_y
                the row Y interval.
            po_y
                PO Y interval; does not include PO in CPO.
            md_y
                MD Y interval.
        lay_info_list
            a list of layers to draw.  Each layer information is a tuple
            of (imp_layer, xl, yb, yt).
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
        adj_row_list
            a list of named tuples for geometries belonging to adjacent
            rows.  Each named tuple should contain:

            row_y
                the row Y interval.
            po_y
                PO Y interval; does not include PO in CPO.
            po_types
                list of po types corresponding to each PO.  0 for dummy, 1 for drawing,
                2 for PODE if PODE is a special poly layer.
        left_blk_info
            a tuple of (EdgeInfo, List[EdgeInfo]) that represents edge information
            of the left adjacent block.  These influences the geometry abutting the
            left edge.  If None, assume default behavior.
        right_blk_info
            same as left_blk_info, but for the right edge.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the layout in.
        layout_info : Dict[str, Any]
            the layout information dictionary.
        """
        res = self.res
        mos_layer_table = self.config['mos_layer_table']

        blk_type = layout_info['blk_type']
        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        arr_yb, arr_yt = layout_info['arr_y']
        draw_od = layout_info['draw_od']
        row_info_list = layout_info['row_info_list']
        lay_info_list = layout_info['lay_info_list']
        fill_info_list = layout_info['fill_info_list']
        adj_row_list = layout_info['adj_row_list']
        left_blk_info = layout_info['left_blk_info']
        right_blk_info = layout_info['right_blk_info']
        no_po_region = layout_info.get('no_po_region', set())
        no_md_region = layout_info.get('no_md_region', set())

        mos_constants = self.get_mos_tech_constants(lch_unit)
        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        sd_pitch = mos_constants['sd_pitch']
        md_w = mos_constants['md_w']
        po_od_extx = mos_constants['po_od_extx']
        substrate_planar = mos_constants.get('substrate_planar', False)

        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2

        default_edge_info = EdgeInfo(od_type=None, draw_layers={}, y_intv={})
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

        blk_w = fg * sd_pitch

        # figure out transistor layout settings
        md_lay = mos_layer_table['MD']
        md_dum_lay = mos_layer_table['MD_dummy']
        finbound_lay = mos_layer_table['FB']

        po_xc = sd_pitch // 2
        # draw transistor rows
        for row_info in row_info_list:
            od_type = row_info.od_type[0]
            if od_type == 'dum' or od_type is None:
                od_name = 'OD_dummy'
                md_lay_cur = md_dum_lay
            elif od_type == 'sub':
                od_name = 'OD_sub'
                md_lay_cur = md_lay
            else:
                od_name = 'OD'
                md_lay_cur = md_lay
            od_x_list = row_info.od_x_list
            od_yb, od_yt = row_info.od_y
            row_y = row_info.row_y
            po_y = row_info.po_y
            md_yb, md_yt = row_info.md_y

            # draw OD and figure out PO/MD info
            po_on_od = [False] * fg
            md_on_od = [False] * (fg + 1)
            po_is_edge = [False] * fg
            if od_yt > od_yb:
                for od_start, od_stop in od_x_list:
                    # mark PO/MD indices that are on OD
                    if od_start >= 1:
                        po_on_od[od_start - 1] = True
                        po_is_edge[od_start - 1] = True
                    for idx in range(od_start, od_stop + 1):
                        md_on_od[idx] = True
                        if idx < fg:
                            po_on_od[idx] = True
                            po_is_edge[idx] = idx == od_stop

                    if draw_od:
                        od_xl = po_xc - lch_unit // 2 + od_start * sd_pitch - po_od_extx
                        od_xr = po_xc + lch_unit // 2 + (od_stop - 1) * sd_pitch + po_od_extx
                        if substrate_planar:
                            # modify OD geometry inside planar guard ring
                            if blk_type == 'gr_sub_sub':
                                self.draw_od(template, od_name,
                                             BBox(od_xl, od_yt, od_xr, arr_yt, res, unit_mode=True))
                                od_xr = po_xc + lch_unit // 2 + (fg - 1) * sd_pitch + po_od_extx
                            elif blk_type == 'gr_sub':
                                od_yb = 0
                                od_yt = arr_yt
                        od_box = BBox(od_xl, od_yb, od_xr, od_yt, res, unit_mode=True)
                        self.draw_od(template, od_name, od_box)
            elif substrate_planar and blk_type == 'gr_sub' and arr_yt > arr_yb:
                od_start, od_stop = od_x_list[0]
                od_xl = po_xc - lch_unit // 2 + od_start * sd_pitch - po_od_extx
                od_xr = po_xc + lch_unit // 2 + (od_stop - 1) * sd_pitch + po_od_extx
                od_box = BBox(od_xl, arr_yb, od_xr, arr_yt, res, unit_mode=True)
                self.draw_od(template, od_name, od_box)

            # draw PO/PODE
            if row_y[1] > row_y[0]:
                for idx in range(fg):
                    if idx not in no_po_region:
                        po_xl = po_xc + idx * sd_pitch - lch_unit // 2
                        po_xr = po_xl + lch_unit
                        is_edge = po_is_edge[idx]
                        pode_y = row_info.od_y
                        if po_on_od[idx]:
                            cur_od_type = od_type
                        else:
                            if idx == 0:
                                cur_od_type = left_blk_info.od_type
                                is_edge = True
                                pode_y = left_blk_info.y_intv.get('od', row_info.od_y)
                            elif idx == fg - 1:
                                cur_od_type = right_blk_info.od_type
                                is_edge = True
                                pode_y = right_blk_info.y_intv.get('od', row_info.od_y)
                            else:
                                cur_od_type = None

                        if is_edge and cur_od_type is not None:
                            if cur_od_type == 'mos_fake':
                                lay = 'PO_dummy'
                            elif cur_od_type == 'dum':
                                lay = 'PO_edge_dummy'
                            elif cur_od_type == 'sub':
                                lay = 'PO_edge_sub'
                            else:
                                lay = 'PO_edge'
                        elif cur_od_type == 'mos':
                            lay = 'PO'
                        elif cur_od_type == 'sub':
                            lay = 'PO_sub'
                        elif cur_od_type == 'dum':
                            lay = 'PO_gate_dummy'
                        else:
                            lay = 'PO_dummy'

                        self.draw_poly(template, mos_constants, lay, (po_xl, po_xr), row_y,
                                       po_y, pode_y)

            # draw MD
            if md_yt > md_yb and fg > 0:
                for idx in range(fg + 1):
                    if idx not in no_md_region:
                        md_xl = idx * sd_pitch - md_w // 2
                        md_xr = md_xl + md_w
                        md_box = BBox(md_xl, md_yb, md_xr, md_yt, res, unit_mode=True)
                        if md_on_od[idx]:
                            self.draw_mos_rect(template, md_lay_cur, md_box)
                        else:
                            if (0 < idx < fg or
                                    idx == 0 and 'md' in left_blk_info.draw_layers or
                                    idx == fg and 'md' in right_blk_info.draw_layers):
                                self.draw_mos_rect(template, md_dum_lay, md_box)

        # draw other layers
        for imp_lay, xl, yb, yt in lay_info_list:
            if imp_lay == finbound_lay:
                # round to fin grid
                yb = (yb - fin_p2 + fin_h2) // fin_p * fin_p + fin_p2 - fin_h2
                yt = -(-(yt - fin_p2 - fin_h2) // fin_p) * fin_p + fin_p2 + fin_h2
            box = BBox(xl, yb, blk_w, yt, res, unit_mode=True)
            if box.is_physical():
                self.draw_mos_rect(template, imp_lay, box)

        # draw adjacent row geometries
        for adj_info in adj_row_list:
            row_y = adj_info.row_y
            po_y = adj_info.po_y
            if row_y[1] > row_y[0]:
                for idx, po_type in enumerate(adj_info.po_types):
                    if idx not in no_po_region:
                        po_xl = po_xc + idx * sd_pitch - lch_unit // 2
                        po_xr = po_xl + lch_unit
                        self.draw_poly(template, mos_constants, po_type, (po_xl, po_xr), row_y,
                                       po_y, (po_y[0], po_y[0]))

        # set size and add PR boundary
        arr_box = BBox(0, arr_yb, blk_w, arr_yt, res, unit_mode=True)
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
                    self.draw_mos_rect(template, exc_lay, bound_box)
                for xl, xr in x_intv_list:
                    for yb, yt in y_intv_list:
                        self.draw_mos_rect(template, lay, BBox(xl, yb, xr, yt, res, unit_mode=True))

    def draw_substrate_connection(self,  # type: MOSTechFinfetBase
                                  template,  # type: TemplateBase
                                  layout_info,  # type: Dict[str, Any]
                                  port_tracks,  # type: List[Union[float, int]]
                                  dum_tracks,  # type: List[Union[float, int]]
                                  exc_tracks,  # type: List[Union[float, int]]
                                  dummy_only,  # type: bool
                                  is_laygo,  # type: bool
                                  is_guardring,  # type: bool
                                  options,  # type: Dict[str, Any]
                                  ):
        # type: (...) -> bool

        sub_parity = options.get('sub_parity', 0)

        lch_unit = layout_info['lch_unit']
        row_info_list = layout_info['row_info_list']

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        sd_pitch2 = sd_pitch // 2

        exc_set = set(int(2 * v + 1) for v in exc_tracks)

        has_od = False
        for row_info in row_info_list:
            od_y = row_info.od_y
            md_y = row_info.md_y
            if od_y[1] > od_y[0]:
                has_od = True

                # find current port name
                od_start, od_stop = row_info.od_x_list[0]
                fg = od_stop - od_start
                xshift = od_start * sd_pitch
                sub_type = row_info.od_type[1]
                port_name = 'VDD' if sub_type == 'ntap' else 'VSS'

                # find X locations of M1/M3.
                if dummy_only:
                    # find X locations to draw vias
                    dum_x_list = [sd_pitch2 * int(2 * v + 1) for v in dum_tracks]
                    conn_x_list = []
                else:
                    # first, figure out port/dummy tracks
                    # To lower parasitics, we try to draw only as many dummy tracks as necessary.
                    # Also, every port track is also a dummy track (because some technology
                    # there's no horizontal short).  With these constraints, our track selection
                    # algorithm is as follows:
                    # 1. for every dummy track, if its not adjacent to any port tracks, add it to
                    #    port tracks (this improves dummy connection resistance to supply).
                    # 2. Try to add as many unused tracks to port tracks as possible, while making
                    #    sure we don't end up with adjacent port tracks.  This improves substrate
                    #    connection resistance to supply.

                    # use half track indices so we won't have rounding errors.
                    dum_htr_set = set((int(2 * v + 1) for v in dum_tracks))
                    conn_htr_set = set((int(2 * v + 1) for v in port_tracks))
                    # add as many dummy tracks as possible to port tracks
                    for d in dum_htr_set:
                        if d + 2 not in conn_htr_set and d - 2 not in conn_htr_set:
                            if d in exc_set:
                                dum_htr_set.add(d)
                            else:
                                conn_htr_set.add(d)
                    # add as many unused tracks as possible to port tracks
                    for htr in range(od_start * 2, 2 * od_stop + 1, 2):
                        if htr + 2 not in conn_htr_set and htr - 2 not in conn_htr_set:
                            if htr in exc_set:
                                dum_htr_set.add(htr)
                            else:
                                conn_htr_set.add(htr)
                    # add all port sets to dummy set
                    dum_htr_set.update(conn_htr_set)
                    # find X coordinates
                    dum_x_list = [sd_pitch2 * v for v in sorted(dum_htr_set)]
                    conn_x_list = [sd_pitch2 * v for v in sorted(conn_htr_set)]

                ds_code = 4 if is_guardring else 3
                dum_warrs, port_warrs = self.draw_ds_connection(template, lch_unit, fg, sd_pitch,
                                                                xshift, od_y, md_y,
                                                                dum_x_list, conn_x_list, True, 1,
                                                                ds_code,
                                                                ud_parity=sub_parity)
                template.add_pin(port_name, dum_warrs, show=False)
                template.add_pin(port_name, port_warrs, show=False)

                if not is_guardring:
                    self.draw_g_connection(template, lch_unit, fg, sd_pitch, xshift, od_y, md_y,
                                           conn_x_list, is_sub=True)
        return has_od

    def draw_mos_connection(self,  # type: MOSTechFinfetBase
                            template,  # type: TemplateBase
                            mos_info,  # type: Dict[str, Any]
                            sdir,  # type: int
                            ddir,  # type: int
                            gate_pref_loc,  # type: str
                            gate_ext_mode,  # type: int
                            min_ds_cap,  # type: bool
                            is_diff,  # type: bool
                            diode_conn,  # type: bool
                            options,  # type: Dict[str, Any]
                            ):
        # type: (...) -> None

        stack = options.get('stack', 1)
        source_parity = options.get('source_parity', 0)

        # NOTE: ignore min_ds_cap.
        if is_diff:
            raise ValueError('differential connection not supported yet.')

        layout_info = mos_info['layout_info']

        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        row_info = layout_info['row_info_list'][0]

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        mos_conn_modullus = mos_constants['mos_conn_modulus']

        if fg % stack != 0:
            raise ValueError('AnalogMosConn: stack = %d must evenly divides fg = %d' % (stack, fg))

        od_yb, od_yt = row_info.od_y
        md_yb, md_yt = row_info.md_y
        # shift Y interval so that OD centers at y=0
        od_yc = (od_yb + od_yt) // 2
        od_y = od_yb - od_yc, od_yt - od_yc
        md_y = md_yb - od_yc, md_yt - od_yc
        wire_pitch = stack * sd_pitch
        num_seg = fg // stack

        s_x_list = list(range(0, num_seg * wire_pitch + 1, 2 * wire_pitch))
        d_x_list = list(range(wire_pitch, num_seg * wire_pitch + 1, 2 * wire_pitch))

        drain_parity = (source_parity + 1) % mos_conn_modullus
        # determine drain/source via location
        if sdir == 0:
            ds_code = 2
        elif ddir == 0:
            ds_code = 1
        else:
            ds_code = 1 if gate_pref_loc == 's' else 2

        if diode_conn:
            if fg == 1:
                raise ValueError('1 finger transistor connection not supported.')

            # draw wires
            _, s_warrs = self.draw_ds_connection(template, lch_unit, num_seg, wire_pitch, 0, od_y,
                                                 md_y, s_x_list, s_x_list, ds_code == 1, sdir, 1,
                                                 source_parity=source_parity)
            _, d_warrs = self.draw_ds_connection(template, lch_unit, num_seg, wire_pitch, 0, od_y,
                                                 md_y, d_x_list, d_x_list, ds_code == 2, 0, 2,
                                                 source_parity=drain_parity)
            g_warrs = self.draw_g_connection(template, lch_unit, fg, sd_pitch, 0, od_y, md_y,
                                             d_x_list, is_sub=False, is_diode=True)

            g_warrs = WireArray.list_to_warr(g_warrs)
            d_warrs = WireArray.list_to_warr(d_warrs)
            s_warrs = WireArray.list_to_warr(s_warrs)
            template.connect_wires([g_warrs, d_warrs])
            template.add_pin('g', g_warrs, show=False)
            template.add_pin('d', d_warrs, show=False)
            template.add_pin('s', s_warrs, show=False)
        else:
            if not gate_pref_loc:
                gate_pref_loc = 'd' if ds_code == 2 else 's'
            if gate_pref_loc == 'd':
                # avoid drawing gate on the left-most source/drain if number of fingers is odd
                g_x_list = list(range(wire_pitch, num_seg * wire_pitch, 2 * wire_pitch))
            else:
                if num_seg != 2:
                    g_x_list = list(range(2 * wire_pitch, num_seg * wire_pitch, 2 * wire_pitch))
                else:
                    g_x_list = [0, 2 * wire_pitch]

            # draw wires
            _, s_warrs = self.draw_ds_connection(template, lch_unit, num_seg, wire_pitch, 0, od_y,
                                                 md_y, s_x_list, s_x_list, ds_code == 1, sdir, 1,
                                                 source_parity=source_parity)
            _, d_warrs = self.draw_ds_connection(template, lch_unit, num_seg, wire_pitch, 0, od_y,
                                                 md_y, d_x_list, d_x_list, ds_code == 2, ddir, 2,
                                                 source_parity=drain_parity)
            g_warrs = self.draw_g_connection(template, lch_unit, fg, sd_pitch, 0, od_y, md_y,
                                             g_x_list, is_sub=False)

            template.add_pin('s', WireArray.list_to_warr(s_warrs), show=False)
            template.add_pin('d', WireArray.list_to_warr(d_warrs), show=False)
            template.add_pin('g', WireArray.list_to_warr(g_warrs), show=False)

    def draw_dum_connection(self, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[Union[float, int]], Dict[str, Any]) -> None

        layout_info = mos_info['layout_info']

        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        row_info = layout_info['row_info_list'][0]

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        od_yb, od_yt = row_info.od_y
        md_yb, md_yt = row_info.md_y
        # shift Y interval so that OD centers at y=0
        od_yc = (od_yb + od_yt) // 2
        od_y = od_yb - od_yc, od_yt - od_yc
        md_y = md_yb - od_yc, md_yt - od_yc

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

        dum_warrs = self.draw_dum_connection_helper(template, lch_unit, fg, sd_pitch, 0,
                                                    od_y, md_y, ds_x_list, gate_tracks,
                                                    left_edge, right_edge, options)
        template.add_pin('dummy', dum_warrs, show=False)

    def draw_decap_connection(self, template, mos_info, sdir, ddir, gate_ext_mode, export_gate,
                              options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
        layout_info = mos_info['layout_info']

        lch_unit = layout_info['lch_unit']
        fg = layout_info['fg']
        row_info = layout_info['row_info_list'][0]

        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']

        od_yb, od_yt = row_info.od_y
        md_yb, md_yt = row_info.md_y
        # shift Y interval so that OD centers at y=0
        od_yc = (od_yb + od_yt) // 2
        od_y = od_yb - od_yc, od_yt - od_yc
        md_y = md_yb - od_yc, md_yt - od_yc

        g_warr, sup_warrs = self.draw_decap_connection_helper(template, lch_unit, fg, sd_pitch, 0,
                                                              od_y, md_y, gate_ext_mode,
                                                              export_gate)
        if g_warr is not None:
            template.add_pin('g', g_warr, show=False)
        for sup_warr in sup_warrs:
            template.add_pin('supply', sup_warr, show=False)

    def draw_active_fill(self, template, mos_type, threshold, w, h):
        # type: (TemplateBase, str, str, int, int) -> None

        mos_layer_table = self.config['mos_layer_table']
        lch_unit = self.mos_config['dum_lch']
        mos_constants = self.get_mos_tech_constants(lch_unit)

        fin_h = mos_constants['fin_h']
        fin_p = mos_constants['mos_pitch']
        od_min_density = mos_constants['od_min_density']
        od_spx = mos_constants['od_spx']
        dod_edge_spx = mos_constants['dod_edge_spx']
        dod_fg_min, dod_fg_max = mos_constants['dod_fill_fg']
        dpo_edge_spy = mos_constants['dpo_edge_spy']
        po_od_exty = mos_constants['po_od_exty']
        po_od_extx = mos_constants['po_od_extx']
        sd_pitch = mos_constants['sd_pitch']
        fb_od_encx = mos_constants['fb_od_encx']
        imp_od_encx = mos_constants['imp_od_encx']
        imp_po_ency = mos_constants['imp_od_ency']

        # compute fill X intervals
        fill_xl = dod_edge_spx
        fill_xr = w - dod_edge_spx
        fill_yb = dpo_edge_spy
        fill_yt = h - dpo_edge_spy
        fill_w = fill_xr - fill_xl

        # check if we can draw anything at all
        dum_w_min = self.get_od_w(lch_unit, dod_fg_min)
        # worst case, we will allow od_spx / 2 margin on edge
        if w < od_spx + dum_w_min:
            return
        # check if we can just draw one dummy
        if fill_w < dum_w_min * 2 + od_spx:
            # get number of fingers. round up to try to meet min edge distance rule
            fg = min(dod_fg_max, self.get_od_w_inverse(lch_unit, fill_w, round_up=True))
            od_w = self.get_od_w(lch_unit, fg)
            od_xl = (fill_xl + fill_xr - od_w) // 2
            od_x_list = [(od_xl, od_xl + od_w)]
            od_x_density = od_w / w
        else:
            # check if we can only draw two dummies
            fg = self.get_od_w_inverse(lch_unit, (fill_w - od_spx) // 2, round_up=False)
            if fg <= dod_fg_max:
                od_w = self.get_od_w(lch_unit, fg)
                od_x_list = [(fill_xl, fill_xl + od_w), (fill_xr - od_w, fill_xr)]
                od_x_density = (2 * od_w) / w
            else:
                # use maximum number of fingers for all fill
                od_w = self.get_od_w(lch_unit, dod_fg_max)
                od_x_list, od_area = fill_symmetric_max_density(fill_w, fill_w, od_w, od_w, od_spx,
                                                                offset=fill_xl, fill_on_edge=True)
                od_x_density = od_area / w

        # compute fill Y intervals
        od_y_density = od_min_density / od_x_density
        od_y_list = self._get_dummy_od_yloc(lch_unit, h, None, None, None, None,
                                            od_min_density=od_y_density)
        if not od_y_list:
            return

        # draw fills
        res = template.grid.resolution
        ny = len(od_y_list)
        po_lay = mos_layer_table['PO_dummy']
        for idx, (od_yb, od_yt) in enumerate(od_y_list):
            po_yb = fill_yb if idx == 0 else od_yb - po_od_exty
            po_yt = fill_yt if idx == ny - 1 else od_yt + po_od_exty
            for od_xl, od_xr in od_x_list:
                box = BBox(od_xl, od_yb, od_xr, od_yt, res, unit_mode=True)
                self.draw_od(template, 'OD_dummy', box)
                po_xl = od_xl + po_od_extx - sd_pitch
                po_xr = po_xl + lch_unit
                nx = 1 + ((od_xr - po_xr - po_od_extx + sd_pitch) // sd_pitch)
                template.add_rect(po_lay, BBox(po_xl, po_yb, po_xr, po_yt, res, unit_mode=True),
                                  nx=nx, spx=sd_pitch, unit_mode=True)

        # draw other layers
        od_xl = od_x_list[0][0]
        od_xr = od_x_list[-1][1]
        finbound_lay = mos_layer_table['FB']
        fin_p2 = fin_p // 2
        fin_h2 = fin_h // 2
        fin_xl = min(fill_xl, od_xl - fb_od_encx)
        fin_xr = max(fill_xr, od_xr + fb_od_encx)
        fin_yb = fin_p2 - fin_h2
        fin_yt = ((h - fin_p2 - fin_h2) // fin_p) * fin_p + fin_p2 + fin_h2
        imp_xl = min(fill_xl, od_xl - imp_od_encx)
        imp_xr = max(fill_xr, od_xr + imp_od_encx)
        imp_yb = fill_yb - imp_po_ency
        imp_yt = fill_yt + imp_po_ency
        fin_box = BBox(fin_xl, fin_yb, fin_xr, fin_yt, res, unit_mode=True)
        imp_box = BBox(imp_xl, imp_yb, imp_xr, imp_yt, res, unit_mode=True)
        for imp_lay in self.get_mos_layers(mos_type, threshold):
            if imp_lay == finbound_lay:
                box = fin_box
            else:
                box = imp_box
            if box.is_physical():
                self.draw_mos_rect(template, imp_lay, box)
