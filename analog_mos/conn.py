# -*- coding: utf-8 -*-

"""This module defines abstract analog mosfet template classes.
"""

from typing import Dict, Any, Set

from bag import float_to_si_string
from bag.layout.template import TemplateBase, TemplateDB


class AnalogMOSConn(TemplateBase):
    """A template containing transistor connections.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            sdir='source connection direction.  0 for down, 1 for middle, 2 for up.',
            ddir='drain connection direction.  0 for down, 1 for middle, 2 for up.',
            min_ds_cap='True to minimize parasitic Cds.',
            gate_pref_loc="Preferred gate vertical track location.  Either 's' or 'd'.",
            is_diff='True to draw a differential pair connection instead (shared source).',
            diode_conn='True to short drain/gate',
            gate_ext_mode='connect gate using lower level metal to adjacent transistors.',
            options='Dictionary of transistor row options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            min_ds_cap=False,
            gate_pref_loc='',
            is_diff=False,
            diode_conn=False,
            gate_ext_mode=0,
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        # type: () -> str
        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        prefix = 'mconn'
        if self.params['is_diff']:
            prefix += '_diff'

        basename = '%s_l%s_w%s_fg%d_s%d_d%d_g%s' % (prefix, lch_str, w_str,
                                                    self.params['fg'],
                                                    self.params['sdir'],
                                                    self.params['ddir'],
                                                    self.params['gate_pref_loc'],
                                                    )

        if self.params['min_ds_cap']:
            basename += '_minds'
        if self.params['diode_conn']:
            basename += '_diode'
        gext = self.params['gate_ext_mode']
        if gext > 0:
            basename += '_gext%d' % gext
        return basename

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        sdir = self.params['sdir']
        ddir = self.params['ddir']
        gate_pref_loc = self.params['gate_pref_loc']
        gate_ext_mode = self.params['gate_ext_mode']
        min_ds_cap = self.params['min_ds_cap']
        is_diff = self.params['is_diff']
        diode_conn = self.params['diode_conn']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))
        guard_ring_nf = options.get('guard_ring_nf', 0)
        mos_info = tech_cls.get_mos_info(lch_unit, w, 'nch', 'standard', fg, guard_ring_nf=guard_ring_nf)
        tech_cls.draw_mos_connection(self, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                                     min_ds_cap, is_diff, diode_conn, options)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()


class AnalogMOSDummy(TemplateBase):
    """A template containing transistor dummy connections.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            edge_mode='Whether to connect to source/drain on left/right edges.',
            gate_tracks='list of track numbers to draw dummy gate connections.',
            options='Dictionary of transistor row options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        # type: () -> str
        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        fg = self.params['fg']
        edge_mode = self.params['edge_mode']

        basename = 'mdum_l%s_w%s_fg%d_edge%d' % (lch_str, w_str, fg, edge_mode)

        return basename

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        edge_mode = self.params['edge_mode']
        gate_tracks = self.params['gate_tracks']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))
        guard_ring_nf = options.get('guard_ring_nf', 0)
        mos_info = tech_cls.get_mos_info(lch_unit, w, 'nch', 'standard', fg, guard_ring_nf=guard_ring_nf)
        tech_cls.draw_dum_connection(self, mos_info, edge_mode, gate_tracks, options)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()


class AnalogMOSDecap(TemplateBase):
    """A template containing transistor decap connections.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            fg='number of fingers.',
            sdir='source connection direction.',
            ddir='drain connection direction.',
            gate_ext_mode='connect gate using lower level metal to adjacent transistors.',
            export_gate='True to export gate to higher level metal.',
            options='Dictionary of transistor row options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            gate_ext_mode=0,
            export_gate=False,
            sdir=1,
            ddir=1,
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        # type: () -> str
        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        gext = self.params['gate_ext_mode']
        basename = 'mdecap_l%s_w%s_fg%d_gext%d_s%d_d%d' % (lch_str, w_str, self.params['fg'], gext,
                                                           self.params['sdir'], self.params['ddir'])
        if self.params['export_gate']:
            basename += '_gport'
        return basename

    def draw_layout(self):
        lch = self.params['lch']
        w = self.params['w']
        fg = self.params['fg']
        sdir = self.params['sdir']
        ddir = self.params['ddir']
        gate_ext_mode = self.params['gate_ext_mode']
        export_gate = self.params['export_gate']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        res = self.grid.resolution
        lch_unit = int(round(lch / self.grid.layout_unit / res))
        guard_ring_nf = options.get('guard_ring_nf', 0)
        mos_info = tech_cls.get_mos_info(lch_unit, w, 'nch', 'standard', fg, guard_ring_nf=guard_ring_nf)
        tech_cls.draw_decap_connection(self, mos_info, sdir, ddir, gate_ext_mode,
                                       export_gate, options)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()


class AnalogSubstrateConn(TemplateBase):
    """A template containing substrate connections.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self.has_connection = False

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
            dummy_only='True if only dummy connections will be made to this substrate.',
            port_tracks='Substrate port must contain these track indices.',
            dum_tracks='Dummy port must contain these track indices.',
            exc_tracks='Do not draw tracks on these indices.',
            is_laygo='True if this is laygo substrate connection.',
            is_guardring='True if this is guardring substrate connection.',
            options='Additional substrate connection options.',
            tech_cls_name='Technology class name.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            dummy_only=False,
            port_tracks=[],
            dum_tracks=[],
            exc_tracks=[],
            is_laygo=False,
            is_guardring=False,
            options=None,
            tech_cls_name=None,
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def draw_layout(self):
        layout_info = self.params['layout_info']
        dummy_only = self.params['dummy_only']
        port_tracks = self.params['port_tracks']
        dum_tracks = self.params['dum_tracks']
        exc_tracks = self.params['exc_tracks']
        is_laygo = self.params['is_laygo']
        is_guardring = self.params['is_guardring']
        options = self.params['options']
        tech_cls_name = self.params['tech_cls_name']

        if options is None:
            options = {}
        if tech_cls_name is None:
            tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']
        else:
            tech_cls = self.grid.tech_info.tech_params['layout'][tech_cls_name]

        tmp = tech_cls.draw_substrate_connection(self, layout_info, port_tracks, dum_tracks,
                                                 exc_tracks, dummy_only, is_laygo,
                                                 is_guardring, options)
        self.prim_top_layer = tech_cls.get_mos_conn_layer()
        self.has_connection = tmp
