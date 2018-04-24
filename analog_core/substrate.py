# -*- coding: utf-8 -*-

"""This module defines various substrate related classes."""
# TODO: Add tech_cls switch support?

from typing import TYPE_CHECKING, Dict, Any, Set, Tuple, Optional, Union

from bag.util.search import BinaryIterator
from bag.layout.template import TemplateBase
from bag.layout.routing import TrackID
from bag.layout.util import BBox

from ..analog_mos.substrate import AnalogSubstrate
from ..analog_mos.edge import AnalogEdge, AnalogEndRow, SubRingEndRow
from ..analog_mos.mos import SubRingExt
from ..analog_mos.conn import AnalogSubstrateConn

from .base import AnalogBaseInfo

if TYPE_CHECKING:
    from bag.layout.routing import RoutingGrid
    from bag.layout.template import TemplateDB


class SubstrateContact(TemplateBase):
    """A template that draws a single substrate.

    Useful for resistor/capacitor body biasing.

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
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._fg_tot = None
        self._sub_bndx = None
        self._sub_bndy = None

    @property
    def fg_tot(self):
        return self._fg_tot

    @property
    def port_name(self):
        # type: () -> str
        return 'VDD' if self.params['sub_type'] == 'ntap' else 'VSS'

    @classmethod
    def get_substrate_height(cls, grid, top_layer, lch, w, sub_type, threshold,
                             end_mode=15, **kwargs):
        # type: (RoutingGrid, int, float, Union[int, float], str, str, int, **kwargs) -> int
        """Compute height of the substrate contact block, given parameters."""
        fg = 2

        tech_params = grid.tech_info.tech_params
        tech_cls = tech_params['layout']['mos_tech_class']

        res = grid.resolution
        lch_unit = int(round(lch / grid.layout_unit / res))

        if top_layer is None:
            top_layer = tech_cls.get_mos_conn_layer() + 1

        sub_pitch = AnalogSubstrate.get_block_pitch(grid, top_layer, **kwargs)
        info = tech_cls.get_substrate_info(lch_unit, w, sub_type, threshold, fg,
                                           blk_pitch=sub_pitch, **kwargs)

        arr_yb, arr_yt = info['layout_info']['arr_y']
        blk_h = arr_yt - arr_yb

        blk_pitch = grid.get_block_size(top_layer, unit_mode=True)[1]
        info = tech_cls.get_analog_end_info(lch_unit, sub_type, threshold, fg, True, blk_pitch,
                                            **kwargs)
        arr_yb, arr_yt = info['layout_info']['arr_y']
        end_h = arr_yt - arr_yb

        if end_mode & 1 != 0:
            blk_h += end_h
        if end_mode & 2 != 0:
            blk_h += end_h

        return blk_h

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            top_layer='the top layer of the template.',
            lch='channel length, in meters.',
            w='substrate width, in meters/number of fins.',
            port_width='port width in number of tracks.',
            sub_type='substrate type.',
            threshold='substrate threshold flavor.',
            well_width='Width of the well in layout units.',
            end_mode='The substrate end mode.',
            is_passive='True if this substrate is used as substrate contact for passive devices.',
            max_nxblk='Maximum width in number of blocks.  Negative to disable',
            port_tid='Substrate port (tr_idx, tr_width) tuple.',
            show_pins='True to show pin labels.',

        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            port_width=None,
            well_end_mode=0,
            end_mode=15,
            is_passive=False,
            max_nxblk=-1,
            port_tid=None,
            show_pins=False,
        )

    def get_substrate_box(self):
        # type: () -> Tuple[Optional[BBox], Optional[BBox]]
        """Returns the substrate tap bounding box."""
        (imp_yb, imp_yt), (thres_yb, thres_yt) = self._sub_bndy

        xl, xr = self._sub_bndx
        if xl is None or xr is None:
            return None, None

        res = self.grid.resolution
        if imp_yb is None or imp_yt is None:
            imp_box = None
        else:
            imp_box = BBox(xl, imp_yb, xr, imp_yt, res, unit_mode=True)
        if thres_yb is None or thres_yt is None:
            thres_box = None
        else:
            thres_box = BBox(xl, thres_yb, xr, thres_yt, res, unit_mode=True)

        return imp_box, thres_box

    def draw_layout(self):
        # type: () -> None

        top_layer = self.params['top_layer']
        lch = self.params['lch']
        w = self.params['w']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        port_width = self.params['port_width']
        well_width = self.params['well_width']
        end_mode = self.params['end_mode']
        is_passive = self.params['is_passive']
        max_nxblk = self.params['max_nxblk']
        port_tid = self.params['port_tid']
        show_pins = self.params['show_pins']

        res = self.grid.resolution
        well_width = int(round(well_width / res))
        right_end = (end_mode & 8) != 0
        left_end = (end_mode & 4) != 0
        top_end = (end_mode & 2) != 0
        bot_end = (end_mode & 1) != 0

        # get layout info, also set RoutingGrid to substrate grid.
        layout_info = AnalogBaseInfo(self.grid, lch, 0, top_layer=top_layer, end_mode=end_mode)
        # compute template width in number of sd pitches
        # find maximum number of fingers we can draw
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_fg = bin_iter.get_next()
            cur_pinfo = layout_info.get_placement_info(cur_fg)
            cur_core_width = cur_pinfo.core_width
            if cur_core_width == well_width:
                bin_iter.save_info(cur_pinfo)
                break
            elif cur_core_width < well_width:
                bin_iter.save_info(cur_pinfo)
                bin_iter.up()
            else:
                bin_iter.down()

        sub_fg_tot = bin_iter.get_last_save()
        if sub_fg_tot is None:
            raise ValueError('Cannot draw substrate that fit in width: %d' % well_width)

        # check width parity requirement
        if max_nxblk > 0:
            blkw = self.grid.get_block_size(top_layer, unit_mode=True)[0]
            place_info = bin_iter.get_last_save_info()
            cur_nxblk = place_info.tot_width // blkw
            while sub_fg_tot > 0 and (cur_nxblk > max_nxblk or (max_nxblk - cur_nxblk) % 2 != 0):
                sub_fg_tot -= 1
                place_info = layout_info.get_placement_info(sub_fg_tot)
                cur_nxblk = place_info.tot_width // blkw
            if sub_fg_tot <= 0:
                raise ValueError('Cannot draw substrate with width = %d, '
                                 'max_nxblk = %d' % (well_width, max_nxblk))

        layout_info.set_fg_tot(sub_fg_tot)
        self.grid = layout_info.grid

        place_info = layout_info.get_placement_info(sub_fg_tot)
        edgel_x0 = place_info.edge_margins[0]
        tot_width = place_info.tot_width

        # create masters
        master_list = [
            self.new_template(params=dict(lch=lch, fg=sub_fg_tot, sub_type=sub_type,
                                          threshold=threshold, is_end=bot_end,
                                          top_layer=top_layer,),
                              temp_cls=AnalogEndRow),
            self.new_template(params=dict(lch=lch, w=w, sub_type=sub_type, threshold=threshold,
                                          fg=sub_fg_tot, top_layer=top_layer,
                                          options=dict(is_passive=is_passive),),
                              temp_cls=AnalogSubstrate),
            self.new_template(params=dict(lch=lch, fg=sub_fg_tot, sub_type=sub_type,
                                          threshold=threshold, is_end=top_end,
                                          top_layer=top_layer,),
                              temp_cls=AnalogEndRow), ]

        ycur = 0
        array_box = BBox.get_invalid_bbox()
        sub_conn, inst = None, None
        for master, orient in zip(master_list, ['R0', 'R0', 'MX']):
            if orient == 'MX':
                ycur += master.array_box.top_unit

            name_id = master.get_layout_basename()
            edge_layout_info = master.get_edge_layout_info()
            xcur = edgel_x0
            if left_end:
                edge_info = master.get_left_edge_info()
                edge_params = dict(
                    is_end=True,
                    guard_ring_nf=0,
                    name_id=name_id,
                    layout_info=edge_layout_info,
                    adj_blk_info=edge_info,
                )
                edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)
                if not edge_master.is_empty:
                    edge_inst = self.add_instance(edge_master, loc=(edgel_x0, ycur),
                                                  orient=orient, unit_mode=True)
                    array_box = array_box.merge(edge_inst.array_box)
                    xcur = edge_inst.array_box.right_unit

            inst = self.add_instance(master, loc=(xcur, ycur), orient=orient, unit_mode=True)
            array_box = array_box.merge(inst.array_box)
            if isinstance(master, AnalogSubstrate):
                conn_params = dict(
                    layout_info=edge_layout_info,
                    layout_name=name_id + '_subconn',
                    is_laygo=False,
                )
                conn_master = self.new_template(params=conn_params, temp_cls=AnalogSubstrateConn)
                sub_conn = self.add_instance(conn_master, loc=(xcur, ycur),
                                             orient=orient, unit_mode=True)
            xcur = inst.array_box.right_unit

            if right_end:
                edge_info = master.get_right_edge_info()
                edge_params = dict(
                    is_end=True,
                    guard_ring_nf=0,
                    name_id=name_id,
                    layout_info=edge_layout_info,
                    adj_blk_info=edge_info,
                )
                edge_master = self.new_template(params=edge_params, temp_cls=AnalogEdge)
                if not edge_master.is_empty:
                    xcur += edge_master.array_box.right_unit
                    eor = 'MY' if orient == 'R0' else 'R180'
                    edge_inst = self.add_instance(edge_master, loc=(xcur, ycur), orient=eor,
                                                  unit_mode=True)
                    array_box = array_box.merge(edge_inst.array_box)

            if orient == 'R0':
                ycur += master.array_box.top_unit

        # calculate substrate Y coordinates
        imp_yb, thres_yb = master_list[0].sub_ysep
        imp_yt, thres_yt = master_list[2].sub_ysep
        self._sub_bndy = (imp_yb, ycur - imp_yt), (thres_yb, ycur - thres_yt)

        # get left/right substrate coordinates
        tot_imp_box = BBox.get_invalid_bbox()
        for lay in self.grid.tech_info.get_implant_layers('ptap'):
            tot_imp_box = tot_imp_box.merge(self.get_rect_bbox(lay))
        for lay in self.grid.tech_info.get_implant_layers('ntap'):
            tot_imp_box = tot_imp_box.merge(self.get_rect_bbox(lay))

        if not tot_imp_box.is_physical():
            self._sub_bndx = None, None
        else:
            self._sub_bndx = tot_imp_box.left_unit, tot_imp_box.right_unit

        # set array box and size
        self.array_box = array_box
        bound_box = BBox(0, 0, tot_width, inst.bound_box.top_unit, res, unit_mode=True)
        if self.grid.size_defined(top_layer):
            self.set_size_from_bound_box(top_layer, bound_box)
        else:
            self.prim_bound_box = bound_box
            self.prim_top_layer = top_layer
        self.add_cell_boundary(bound_box)

        hm_layer = layout_info.mconn_port_layer + 1
        if port_tid is None:
            # find center track index
            hm_mid = self.grid.coord_to_nearest_track(hm_layer, self.array_box.yc_unit, mode=0,
                                                      half_track=True, unit_mode=True)
            # connect to horizontal metal layer.
            hm_pitch = self.grid.get_track_pitch(hm_layer, unit_mode=True)
            ntr = self.array_box.height_unit // hm_pitch  # type: int
            if port_width is None:
                port_width = self.grid.get_max_track_width(hm_layer, 1, ntr, half_end_space=False)
            port_tid = TrackID(hm_layer, hm_mid, width=port_width)
        else:
            port_tid = TrackID(hm_layer, port_tid[0], width=port_tid[1])

        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sub_wires = self.connect_to_tracks(sub_conn.get_port(port_name).get_pins(hm_layer - 1),
                                           port_tid)
        self.add_pin(port_name, sub_wires, show=show_pins)

        self._fg_tot = sub_fg_tot


class SubstrateRing(TemplateBase):
    """A template that draws a ring of substrate taps around a given bounding box.

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
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        self._blk_loc = None

    @property
    def blk_loc_unit(self):
        return self._blk_loc

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            show_pins=False,
            dnw_mode='',
        )

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            top_layer='the top layer of the template.',
            bound_box='bounding box of the inner template',
            sub_type='the substrate type.',
            w='substrate tap width, in meters/number of fins.',
            fg_side='number of fingers in vertical substrate ring.',
            threshold='substrate threshold flavor.',
            show_pins='True to show pin labels.',
            dnw_mode='deep N-well mode string.  Empty string to disable.',
        )

    def draw_layout(self):
        # type: () -> None

        top_layer = self.params['top_layer']
        bound_box = self.params['bound_box']
        sub_type = self.params['sub_type']
        w = self.params['w']
        fg_side = self.params['fg_side']
        threshold = self.params['threshold']
        show_pins = self.params['show_pins']
        dnw_mode = self.params['dnw_mode']

        sub_end_mode = 15
        lch = self._tech_cls.get_substrate_ring_lch()

        # create layout masters
        box_w, box_h = bound_box.width_unit, bound_box.height_unit
        layout_info = AnalogBaseInfo(self.grid, lch, fg_side, top_layer=top_layer,
                                     end_mode=sub_end_mode, is_sub_ring=True, dnw_mode=dnw_mode)
        sd_pitch = layout_info.sd_pitch_unit
        mtop_lay = layout_info.mconn_port_layer + 1

        fg_tot = -(-box_w // sd_pitch)
        place_info = layout_info.get_placement_info(fg_tot)
        wtot = place_info.tot_width
        dx = place_info.edge_margins[0]
        arr_box_x = place_info.arr_box_x
        layout_info.set_fg_tot(fg_tot)
        self.grid = layout_info.grid

        if top_layer < mtop_lay:
            raise ValueError('top_layer = %d must be at least %d' % (top_layer, mtop_lay))

        htot, master_list, edge_list = self._make_masters(top_layer, mtop_lay, lch, fg_tot, w,
                                                          fg_side, sub_type, threshold, dnw_mode,
                                                          box_h)

        # arrange layout masters
        # first, compute edge margins so everything is quantized properly.
        m_sub, m_end1, m_end2, m_ext = master_list
        e_sub, e_end1, e_end2, e_ext = edge_list
        e1_h = m_end1.bound_box.height_unit
        e2_h = m_end2.bound_box.height_unit
        sub_h = m_sub.bound_box.height_unit
        sub_w = m_sub.bound_box.width_unit
        e_sub_w = e_sub.bound_box.width_unit

        # add masters at correct locations
        m_list = [e_end1, m_end1, e_end1, e_sub, m_sub, e_sub, e_end2, m_end2, e_end2]
        xl_list = [dx, dx + e_sub_w, dx + e_sub_w + sub_w]
        yl_list = [0, e1_h, e1_h + sub_h]
        o_list = ['R0', 'R0', 'MY']
        self._blk_loc = ((wtot - box_w) // 2, (htot - box_h) // 2)

        # substrate connection master
        conn_params = dict(
            layout_info=m_sub.get_edge_layout_info(),
            layout_name=m_sub.get_layout_basename() + '_subconn',
            is_laygo=False,
        )
        m_conn = self.new_template(params=conn_params, temp_cls=AnalogSubstrateConn)

        # add substrate row masters
        edge_inst_list = []
        conn_list = []
        tid_list = []
        hm_pitch = self.grid.get_track_pitch(mtop_lay, unit_mode=True)
        for name_fmt, flip_ud, yoff in (('XB%d', False, 0), ('XT%d', True, htot)):
            m_idx = 0
            for yidx, yl in enumerate(yl_list):
                for xidx, (xl, orient) in enumerate(zip(xl_list, o_list)):
                    master = m_list[m_idx]
                    cur_name = name_fmt % m_idx  # type: str
                    if not master.is_empty:
                        if flip_ud:
                            orient = 'MX' if orient == 'R0' else 'R180'
                        if orient == 'R0':
                            loc = (xl, yl)
                        elif orient == 'MY':
                            loc = (xl + master.bound_box.width_unit, yl)
                        elif orient == 'MX':
                            loc = (xl, yoff - yl)
                        elif orient == 'R180':
                            loc = (xl + master.bound_box.width_unit, yoff - yl)
                        else:
                            raise ValueError('Unsupported orientation: %s' % orient)
                        inst = self.add_instance(master, inst_name=cur_name, loc=loc,
                                                 orient=orient, unit_mode=True)
                        if xidx == 0 or xidx == 2:
                            edge_inst_list.append(inst)
                        elif xidx == 1 and yidx == 1:
                            # get supply TrackID
                            hm_tidx = self.grid.coord_to_track(mtop_lay, inst.bound_box.yc_unit,
                                                               unit_mode=True)
                            ntr = inst.bound_box.height_unit // hm_pitch  # type: int
                            tr_width = self.grid.get_max_track_width(mtop_lay, 1, ntr,
                                                                     half_end_space=False)
                            tid_list.append(TrackID(mtop_lay, hm_tidx, width=tr_width))
                            inst = self.add_instance(m_conn, inst_name=cur_name + '_CONN', loc=loc,
                                                     orient=orient, unit_mode=True)
                            conn_list.append(inst)
                    m_idx += 1

        # add left and right edge
        hsub = e1_h + e2_h + sub_h
        edge_inst_list.append(self.add_instance(e_ext, inst_name='XEL', loc=(dx, hsub),
                                                unit_mode=True))
        edge_inst_list.append(self.add_instance(e_ext, inst_name='XER', loc=(wtot - dx, hsub),
                                                orient='MY', unit_mode=True))

        # set size and array box
        res = self.grid.resolution
        bnd_box = BBox(0, 0, wtot, htot, res, unit_mode=True)
        if top_layer > mtop_lay:
            self.set_size_from_bound_box(top_layer, bnd_box)
        else:
            self.prim_top_layer = top_layer
            self.prim_bound_box = bnd_box
        self.array_box = BBox(arr_box_x[0], 0, arr_box_x[1], htot, res, unit_mode=True)
        self.add_cell_boundary(self.bound_box)

        if dnw_mode:
            # add overlay DNW layer
            for lay in self.grid.tech_info.get_dnw_layers():
                dnw_box = self.get_rect_bbox(lay)
                self.add_rect(lay, dnw_box)

        # connect to horizontal metal layer.
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        conn_warr_list, dum_warr_list = [], []
        dum_layer = self._tech_cls.get_dum_conn_layer()
        for inst in edge_inst_list:
            if inst.has_port(port_name):
                conn_warr_list.extend(inst.get_all_port_pins(port_name, layer=mtop_lay - 1))
                dum_warr_list.extend(inst.get_all_port_pins(port_name, layer=dum_layer))

        self.connect_wires(dum_warr_list)
        edge_warrs = self.connect_wires(conn_warr_list)

        for conn_inst, tid in zip(conn_list, tid_list):
            cur_warrs = edge_warrs + conn_inst.get_port(port_name).get_pins(mtop_lay - 1)
            sub_wires = self.connect_to_tracks(cur_warrs, tid)
            self.add_pin(port_name, sub_wires, show=show_pins)

    def _make_masters(self, top_layer, mtop_lay, lch, fg_tot, w, fg_side, sub_type, threshold,
                      dnw_mode, box_h):
        options1 = dict(is_sub_ring=True, dnw_mode=dnw_mode)
        options2 = dict(dnw_mode=dnw_mode)
        options3 = options1.copy()
        options3['integ_htr'] = True

        sub_params = dict(
            lch=lch,
            fg=fg_tot,
            w=w,
            sub_type=sub_type,
            threshold=threshold,
            top_layer=mtop_lay,
            options=options3,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=AnalogSubstrate)

        end1_params = dict(
            lch=lch,
            fg=fg_tot,
            sub_type=sub_type,
            threshold=threshold,
            is_end=True,
            top_layer=mtop_lay,
            options=options1,
        )
        end1_master = self.new_template(params=end1_params, temp_cls=AnalogEndRow)

        end2_params = dict(
            fg=fg_tot,
            sub_type=sub_type,
            threshold=threshold,
            end_ext_info=sub_master.get_ext_top_info(),
            options=options2,
        )
        end2_master = self.new_template(params=end2_params, temp_cls=SubRingEndRow)

        # compute extension height
        hsub = (sub_master.bound_box.height_unit + end1_master.bound_box.height_unit +
                end2_master.bound_box.height_unit)
        hmin = 2 * hsub + box_h
        blk_h = self.grid.get_block_size(top_layer, unit_mode=True)[1]
        if box_h % blk_h != 0:
            raise ValueError('Template height = %d is not multiples of %d' % (box_h, blk_h))

        htot = -(-hmin // blk_h) * blk_h
        if ((htot - box_h) // blk_h) % 2 == 1:
            # make sure template has integer number of blocks from top and bottom.
            htot += blk_h

        ext_params = dict(
            sub_type=sub_type,
            height=htot - 2 * hsub,
            fg=fg_tot,
            end_ext_info=end2_master.get_ext_info(),
            options=options2,
        )
        ext_master = self.new_template(params=ext_params, temp_cls=SubRingExt)

        master_list = [sub_master, end1_master, end2_master, ext_master]

        edge_list = []
        for master in master_list:
            edge_params = dict(
                is_end=True,
                guard_ring_nf=fg_side,
                name_id=master.get_layout_basename(),
                layout_info=master.get_edge_layout_info(),
                adj_blk_info=master.get_left_edge_info(),
            )
            edge_list.append(self.new_template(params=edge_params, temp_cls=AnalogEdge))

        return htot, master_list, edge_list


class DeepNWellRing(TemplateBase):
    """A template that draws a deep N-well double ring around a template.

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
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(DeepNWellRing, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        self._blk_loc = None

    @property
    def blk_loc_unit(self):
        return self._blk_loc

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            show_pins=False,
        )

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            top_layer='the top layer of the template.',
            bound_box='bounding box of the inner template',
            w='substrate tap width, in meters/number of fins.',
            fg_side='number of fingers in vertical substrate ring.',
            threshold='substrate threshold flavor.',
            show_pins='True to show pin labels.',
            dnw_mode='deep N-well mode string.  This determines the DNW space to adjacent blocks.',
        )

    def draw_layout(self):
        # type: () -> None

        top_layer = self.params['top_layer']
        bound_box = self.params['bound_box']
        w = self.params['w']
        fg_side = self.params['fg_side']
        threshold = self.params['threshold']
        show_pins = self.params['show_pins']
        dnw_mode = self.params['dnw_mode']

        # test top_layer
        hm_layer = self._tech_cls.get_mos_conn_layer() + 1
        if top_layer <= hm_layer:
            raise ValueError('top layer for DeepNWellRing must be >= %d' % (hm_layer + 1))

        # make masters
        dnw_params = dict(
            top_layer=top_layer,
            bound_box=bound_box,
            sub_type='ntap',
            w=w,
            fg_side=fg_side,
            threshold=threshold,
            show_pins=False,
            dnw_mode='compact',
        )
        dnw_master = self.new_template(params=dnw_params, temp_cls=SubstrateRing)
        dnw_blk_loc = dnw_master.blk_loc_unit

        sub_params = dict(
            top_layer=top_layer,
            bound_box=dnw_master.bound_box,
            sub_type='ptap',
            w=w,
            fg_side=fg_side,
            threshold=threshold,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateRing)
        sub_blk_loc = sub_master.blk_loc_unit

        # put masters at (0, 0)
        sub_inst = self.add_instance(sub_master, 'XSUB')
        dnw_inst = self.add_instance(dnw_master, 'XDNW', loc=sub_blk_loc, unit_mode=True)

        # check how much to move substrate rings by to achive the DNW margin.
        x_pitch, y_pitch = self.grid.get_block_size(top_layer, unit_mode=True)
        dnw_margin = self.grid.tech_info.get_dnw_margin_unit(dnw_mode)
        dnw_box = BBox.get_invalid_bbox()
        for dnw_lay in self.grid.tech_info.get_dnw_layers():
            dnw_box = dnw_box.merge(dnw_inst.get_rect_bbox(dnw_lay))

        dx = -(-max(0, dnw_margin - dnw_box.left_unit) // x_pitch) * x_pitch
        dy = -(-max(0, dnw_margin - dnw_box.bottom_unit) // x_pitch) * x_pitch
        self.move_all_by(dx=dx, dy=dy, unit_mode=True)

        # set size
        res = self.grid.resolution
        sub_w = sub_master.bound_box.width_unit
        sub_h = sub_master.bound_box.height_unit
        bnd_box = BBox(0, 0, sub_w + 2 * dx, sub_h + 2 * dy, res, unit_mode=True)
        self.set_size_from_bound_box(top_layer, bnd_box)
        self.array_box = bnd_box
        self.add_cell_boundary(bnd_box)

        # record block location
        dnw_loc = dnw_inst.location_unit
        self._blk_loc = dnw_loc[0] + dnw_blk_loc[0], dnw_loc[1] + dnw_blk_loc[1]

        # export supplies
        self.reexport(sub_inst.get_port('VSS'), show=show_pins)
        self.reexport(dnw_inst.get_port('VDD'), show=show_pins)
