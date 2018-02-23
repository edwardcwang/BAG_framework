# -*- coding: utf-8 -*-

"""This module defines analog mosfet boundary primitive template classes.
"""

from typing import TYPE_CHECKING, Dict, Any, Set, Tuple, Optional

from bag import float_to_si_string
from bag.layout.template import TemplateBase

from .core import MOSTech
from .substrate import AnalogSubstrateCore
from .conn import AnalogSubstrateConn

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class AnalogEndRow(TemplateBase):
    """A primitive template of the top/bottom boundary row.

    This template must abut a substrate row.

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
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
        self._layout_info = None
        self._left_edge_info = None
        self._right_edge_info = None
        self._sub_ysep = None

    @property
    def sub_ysep(self):
        # type: () -> Tuple[Optional[int], Optional[int]]
        return self._sub_ysep

    def get_edge_layout_info(self):
        # type: () -> Dict[str, Any]
        return self._layout_info

    def get_left_edge_info(self):
        # type: () -> Any
        return self._left_edge_info

    def get_right_edge_info(self):
        # type: () -> Any
        return self._right_edge_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lch='channel length, in meters.',
            fg='number of fingers.',
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            is_end='True if there are no blocks abutting the end.',
            top_layer='The top routing layer.  Used to determine vertical pitch.',
            options='Optional layout parameters.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(options=None)

    def get_layout_basename(self):
        fmt = '%s_end_l%s_%s_lay%d_fg%d'
        sub_type = self.params['sub_type']
        lstr = float_to_si_string(self.params['lch'])
        th = self.params['threshold']
        top_layer = self.params['top_layer']
        fg = self.params['fg']

        basename = fmt % (sub_type, lstr, th, top_layer, fg)
        if self.params['is_end']:
            basename += '_end'

        return basename

    def compute_unique_key(self):
        key = self.get_layout_basename(), self.params['options']
        return self.to_immutable_id(key)

    def draw_layout(self):
        lch_unit = int(round(self.params['lch'] / self.grid.layout_unit / self.grid.resolution))
        fg = self.params['fg']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        is_end = self.params['is_end']
        top_layer = self.params['top_layer']
        options = self.params['options']
        if options is None:
            options = {}

        blk_pitch = self.grid.get_block_size(top_layer, unit_mode=True)[1]
        end_info = self._tech_cls.get_analog_end_info(lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **options)

        self._layout_info = end_info['layout_info']
        self._left_edge_info = end_info['left_edge_info']
        self._right_edge_info = end_info['right_edge_info']
        self._sub_ysep = end_info.get('sub_ysep', (None, None))
        self._tech_cls.draw_mos(self, self._layout_info)


class SubRingEndRow(TemplateBase):
    """A primitive template of the inner substrate boundary row inside a substrate ring.

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
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()
        self._layout_info = None
        self._left_edge_info = None
        self._right_edge_info = None
        self._sub_ysep = None
        self._ext_info = None

    @property
    def sub_ysep(self):
        # type: () -> Tuple[Optional[int], Optional[int]]
        return self._sub_ysep

    def get_ext_info(self):
        # type: () -> Any
        return self._ext_info

    def get_edge_layout_info(self):
        # type: () -> Dict[str, Any]
        return self._layout_info

    def get_left_edge_info(self):
        # type: () -> Any
        return self._left_edge_info

    def get_right_edge_info(self):
        # type: () -> Any
        return self._right_edge_info

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            fg='number of fingers.',
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            end_ext_info='substrate ring inner end row extension info.',
            options='Optional layout parameters.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(options=None)

    def get_layout_basename(self):
        fmt = '%s_subringend_%s_fg%d'
        sub_type = self.params['sub_type']
        th = self.params['threshold']
        fg = self.params['fg']
        basename = fmt % (sub_type, th, fg)
        return basename

    def compute_unique_key(self):
        key = self.get_layout_basename(), self.params['end_ext_info'], self.params['options']
        return self.to_immutable_id(key)

    def draw_layout(self):
        fg = self.params['fg']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        end_ext_info = self.params['end_ext_info']
        options = self.params['options']
        if options is None:
            options = {}

        end_info = self._tech_cls.get_sub_ring_end_info(sub_type, threshold, fg, end_ext_info, **options)

        self._layout_info = end_info['layout_info']
        self._left_edge_info = end_info['left_edge_info']
        self._right_edge_info = end_info['right_edge_info']
        self._ext_info = end_info['ext_info']
        self._sub_ysep = end_info.get('sub_ysep', (None, None))
        self._tech_cls.draw_mos(self, self._layout_info)


class AnalogOuterEdge(TemplateBase):
    """A primitive template of the outer-most left/right edge of analog mosfet.

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
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_mos(self, self.params['layout_info'])
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()


class AnalogGuardRingSep(TemplateBase):
    """A primitive template of the geometry between substrate/transistor row and left/right guard ring.

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
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            layout_name='name of the layout cell.',
            layout_info='the layout information dictionary.',
        )

    def get_layout_basename(self):
        return self.params['layout_name']

    def compute_unique_key(self):
        return self.to_immutable_id((self.params['layout_name'], self.params['layout_info']))

    def draw_layout(self):
        self._tech_cls.draw_mos(self, self.params['layout_info'])
        self.prim_top_layer = self._tech_cls.get_mos_conn_layer()


class AnalogEdge(TemplateBase):
    """A primitive template of the left/right analog mosfet edge block.

    This block will include guard ring if that option is enabled.

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
    kwargs : Dict[str, Any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._tech_cls = self.grid.tech_info.tech_params['layout']['mos_tech_class']  # type: MOSTech
        if self.params['is_laygo']:
            self.prim_top_layer = self._tech_cls.get_dig_conn_layer()
        else:
            self.prim_top_layer = self._tech_cls.get_mos_conn_layer()

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            is_end='True if this edge is at the end.',
            guard_ring_nf='number of guard ring fingers.',
            adj_blk_info='data structure storing layout information of adjacent block.',
            name_id='cell name ID.',
            layout_info='the layout information dictionary.',
            is_laygo='True if this extension is used in LaygoBase.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(is_laygo=False)

    def get_layout_basename(self):
        base = 'aedge_%s_gr%d' % (self.params['name_id'], self.params['guard_ring_nf'])
        if self.params['is_end']:
            base += '_end'
        if self.params['is_laygo']:
            base = 'laygo_' + base
        return base

    def compute_unique_key(self):
        base_name = self.get_layout_basename()
        return self.to_immutable_id((base_name, self.params['layout_info'],
                                     self.params['adj_blk_info'], self.grid.get_flip_parity()))

    def draw_layout(self):
        is_end = self.params['is_end']
        guard_ring_nf = self.params['guard_ring_nf']
        layout_info = self.params['layout_info']
        is_laygo = self.params['is_laygo']
        adj_blk_info = self.params['adj_blk_info']
        basename = self.get_layout_basename()

        if guard_ring_nf > 0:
            outer_adj_blk = None
        else:
            outer_adj_blk = adj_blk_info

        out_info = self._tech_cls.get_outer_edge_info(guard_ring_nf, layout_info, is_end, outer_adj_blk)
        # add outer edge
        out_params = dict(
            layout_name='%s_outer' % basename,
            layout_info=out_info,
        )
        master = self.new_template(params=out_params, temp_cls=AnalogOuterEdge)
        if not master.is_empty:
            self.add_instance(master, 'XOUTER')

        self.array_box = master.array_box
        self.prim_bound_box = master.prim_bound_box

        if guard_ring_nf > 0:
            # draw guard ring and guard ring separator
            x0 = self.array_box.right_unit
            sub_info = self._tech_cls.get_gr_sub_info(guard_ring_nf, layout_info)
            loc = x0, 0
            sub_params = dict(
                dummy_only=False,
                port_tracks=[],
                dum_tracks=[],
                layout_name='%s_sub' % basename,
                layout_info=sub_info,
            )
            master = self.new_template(params=sub_params, temp_cls=AnalogSubstrateCore)
            inst = self.add_instance(master, 'XSUB', loc=loc, unit_mode=True)
            conn_params = dict(
                layout_name='%s_subconn' % basename,
                layout_info=sub_info,
                is_laygo=is_laygo,
                is_guardring=True,
            )
            conn_master = self.new_template(params=conn_params, temp_cls=AnalogSubstrateConn)
            if conn_master.has_connection:
                conn_inst = self.add_instance(conn_master, loc=loc, unit_mode=True)
                for port_name in conn_inst.port_names_iter():
                    self.reexport(conn_inst.get_port(port_name), show=False)

            x0 = inst.array_box.right_unit
            sep_info = self._tech_cls.get_gr_sep_info(layout_info, adj_blk_info)
            sep_params = dict(
                layout_name='%s_sep' % basename,
                layout_info=sep_info,
            )
            master = self.new_template(params=sep_params, temp_cls=AnalogGuardRingSep)
            inst = self.add_instance(master, 'XSEP', loc=(x0, 0), unit_mode=True)
            self.array_box = self.array_box.merge(inst.array_box)
            self.prim_bound_box = self.prim_bound_box.merge(inst.translate_master_box(master.prim_bound_box))
