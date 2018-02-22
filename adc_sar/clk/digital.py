# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Set, Any

from bag.layout.digital import StdCellTemplate, StdCellBase

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class Flop(StdCellBase):
    """A row of retiming latch.

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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        StdCellBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

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
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None

        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        self.set_draw_boundaries(True)

        tap_params = dict(cell_name='tap_pwr', config_file=config_file)
        tap_master = self.new_template(params=tap_params, temp_cls=StdCellTemplate)
        flop_name = 'dff_1x'
        flop_params = dict(cell_name=flop_name, config_file=config_file)
        flop_master = self.new_template(params=flop_params, temp_cls=StdCellTemplate)
        inv_name = 'inv_clk_16x'
        inv_params = dict(cell_name=inv_name, config_file=config_file)
        inv_master = self.new_template(params=inv_params, temp_cls=StdCellTemplate)

        flop_ncol = flop_master.std_size[0]
        tap_ncol = tap_master.std_size[0]
        inv_ncol = inv_master.std_size[0]
        space_ncol = 2
        show_pins = True

        tap_list = [self.add_std_instance(tap_master, 'XTAP00', loc=(0, 0)),
                    self.add_std_instance(tap_master, 'XTAP01', loc=(0, 1))]
        ff_bot = self.add_std_instance(flop_master, 'XFFB', loc=(tap_ncol + space_ncol, 0), nx=2, spx=flop_ncol)
        ff_top = self.add_std_instance(flop_master, 'XFFT', loc=(tap_ncol + space_ncol, 1), nx=2, spx=flop_ncol)
        inv_top = self.add_std_instance(inv_master, 'XINVT', loc=(2 * flop_ncol + tap_ncol + space_ncol, 1), nx=2,
                                        spx=inv_ncol)
        inv_bot = self.add_std_instance(inv_master, 'XINVB', loc=(2 * flop_ncol + tap_ncol + space_ncol, 0), nx=2,
                                        spx=inv_ncol)
        xcur = 2 * flop_ncol + tap_ncol + 2 * space_ncol + 2 * inv_ncol
        tap_list.append(self.add_std_instance(tap_master, 'XTAP00', loc=(xcur, 0)))
        tap_list.append(self.add_std_instance(tap_master, 'XTAP01', loc=(xcur, 1)))

        vdd_warrs, vss_warrs = [], []
        for inst in tap_list:
            vdd_warrs.extend(inst.get_all_port_pins('VDD'))
            vss_warrs.extend(inst.get_all_port_pins('VSS'))

        vdd_warrs = self.connect_wires(vdd_warrs)
        vss_warrs = self.connect_wires(vss_warrs)

        # set template size
        top_layer = self.std_routing_layers[-1]
        while not self.grid.size_defined(top_layer):
            top_layer += 1
        self.set_std_size((2 * (flop_ncol + tap_ncol + space_ncol + inv_ncol), 2), top_layer=top_layer)
        self.fill_space()

        # export supplies
        self.add_pin('VDD', vdd_warrs, show=show_pins)
        self.add_pin('VSS', vss_warrs, show=show_pins)
        # self.reexport(inv_bot.get_port('I'), 'IINVBOT', show=show_pins)
        # self.reexport(inv_bot.get_port('O'), 'OINVBOT', show=show_pins)
        # self.reexport(inv_top.get_port('I'), 'IINVTOP', show=show_pins)
        # self.reexport(inv_top.get_port('O'), 'OINVTOP', show=show_pins)
        for inst, name in [(ff_bot, 'BOT'), (ff_top, 'TOP')]:
            for idx in range(2):
                self.reexport(inst.get_port('I', col=idx), 'I%s%d' % (name, idx), show=show_pins)
                self.reexport(inst.get_port('O', col=idx), 'O%s%d' % (name, idx), show=show_pins)
                self.reexport(inst.get_port('CLK', col=idx), 'CLK%s%d' % (name, idx), show=show_pins)
        for inst, name in [(inv_bot, 'INVBOT'), (inv_top, 'INVTOP')]:
            for idx in range(2):
                self.reexport(inst.get_port('I', col=idx), 'I%s%d' % (name, idx), show=show_pins)
                self.reexport(inst.get_port('O', col=idx), 'O%s%d' % (name, idx), show=show_pins)

        self.draw_boundaries()
