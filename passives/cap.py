# -*- coding: utf-8 -*-

"""This package defines various passives template classes.
"""

from typing import Dict, Set, Any

from bag.layout.util import BBox
from bag.layout.template import TemplateBase, TemplateDB

from ..analog_core import SubstrateContact


class MOMCap(TemplateBase):
    """differential bias resistor for differential high pass filter.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(MOMCap, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Dict[str, Any]
            dictionary of default parameter values.
        """
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
            cap_bot_layer='MOM cap bottom layer.',
            cap_top_layer='MOM cap top layer.',
            cap_width='MOM cap width.',
            cap_height='MOM cap height.',
            sub_lch='channel length, in meters.',
            sub_w='substrate width, in meters/number of fins.',
            sub_type='substrate type.',
            threshold='substrate threshold flavor.',
            show_pins='True to show pin labels.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, cap_bot_layer, cap_top_layer, cap_width, cap_height,
                            sub_lch, sub_w, sub_type, threshold, show_pins):
        res = self.grid.resolution
        cap_width = int(round(cap_width / res))
        cap_height = int(round(cap_height / res))

        blk_w, _ = self.grid.get_block_size(cap_top_layer, unit_mode=True)
        w_pitch, _ = self.grid.get_size_pitch(cap_top_layer, unit_mode=True)
        tot_width = -(-cap_width // blk_w) * blk_w

        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=threshold,
            top_layer=cap_top_layer,
            blk_width=tot_width // w_pitch,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        inst = self.add_instance(sub_master, inst_name='XSUB')
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        self.reexport(inst.get_port(port_name), show=show_pins)

        subw, subh = self.grid.get_size_dimension(sub_master.size, unit_mode=True)
        self.size = self.grid.get_size_tuple(cap_top_layer, tot_width, subh + cap_height, round_up=True, unit_mode=True)
        self.array_box = self.bound_box

        cap_xl = self.array_box.xc_unit - cap_width // 2
        cap_box = BBox(cap_xl, subh, cap_xl + cap_width, subh + cap_height, res, unit_mode=True)
        cap_ports = self.add_mom_cap(cap_box, cap_bot_layer, cap_top_layer - cap_bot_layer + 1, 2)
        cp, cn = cap_ports[cap_top_layer]
        self.add_pin('plus', cp, show=show_pins)
        self.add_pin('minus', cn, show=show_pins)


class MOMCapUnit(TemplateBase):
    """differential bias resistor for differential high pass filter.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(MOMCapUnit, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Dict[str, Any]
            dictionary of default parameter values.
        """
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
            cap_bot_layer='MOM cap bottom layer.',
            cap_top_layer='MOM cap top layer.',
            cap_width='MOM cap width.',
            cap_height='MOM cap height.',
            port_width='port track width.',
            show_pins='True to show pin labels.',
        )

    def draw_layout(self):
        # type: () -> None
        cap_bot_layer = self.params['cap_bot_layer']
        cap_top_layer = self.params['cap_top_layer']
        cap_width = self.params['cap_width']
        cap_height = self.params['cap_height']
        port_width = self.params['port_width']
        show_pins = self.params['show_pins']

        res = self.grid.resolution
        cap_width = int(round(cap_width / res))
        cap_height = int(round(cap_height / res))

        self.size = self.grid.get_size_tuple(cap_top_layer, cap_width, cap_height, round_up=True, unit_mode=True)
        self.array_box = self.bound_box

        cap_ports = self.add_mom_cap(self.array_box, cap_bot_layer, cap_top_layer - cap_bot_layer + 1,
                                     port_width, array=True)
        cp, cn = cap_ports[cap_top_layer]
        self.add_pin('plus', cp, show=show_pins)
        self.add_pin('minus', cn, show=show_pins)
