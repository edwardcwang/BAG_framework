# -*- coding: utf-8 -*-

"""This module defines dummy/power fill related templates."""

from typing import TYPE_CHECKING, Dict, Set, Any

from bag.layout.util import BBox
from bag.layout.template import TemplateBase

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class PowerFill(TemplateBase):
    """A power fill template.

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
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            fill_config='the fill configuration dictionary.',
            bot_layer='the bottom fill layer.',
            top_layer='the top layer.',
            show_pins='True to show pins.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            show_pins=True,
        )

    def draw_layout(self):
        # type: () -> None
        fill_config = self.params['fill_config']
        bot_layer = self.params['bot_layer']
        top_layer = self.params['top_layer']
        show_pins = self.params['show_pins']

        blk_w, blk_h = self.grid.get_fill_size(top_layer, fill_config, unit_mode=True)
        bnd_box = BBox(0, 0, blk_w, blk_h, self.grid.resolution, unit_mode=True)
        self.set_size_from_bound_box(top_layer, bnd_box)
        self.array_box = bnd_box

        vdd_list, vss_list = None, None
        for lay in range(bot_layer, top_layer + 1):
            fill_width, fill_space, space, space_le = fill_config[lay]
            vdd_list, vss_list = self.do_power_fill(lay, space, space_le, vdd_warrs=vdd_list,
                                                    vss_warrs=vss_list, fill_width=fill_width,
                                                    fill_space=fill_space, unit_mode=True)

        self.add_pin('VDD', vdd_list, show=show_pins)
        self.add_pin('VSS', vss_list, show=show_pins)
