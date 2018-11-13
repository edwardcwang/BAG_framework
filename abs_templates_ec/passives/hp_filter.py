# -*- coding: utf-8 -*-

"""This package defines various passives template classes.
"""
from typing import Dict, Set, Any

from bag.layout.util import BBox
from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase, TemplateDB

from ..resistor.core import ResArrayBase
from ..analog_core import SubstrateContact


class BiasResistor(ResArrayBase):
    """bias resistor for differential high pass filter.

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
        super(BiasResistor, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
            em_specs={},
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
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            nx='number of columns.',
            ny='number of rows.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
        )

    def connect_series_resistor(self, nx, ny):
        last_port = None
        port_out = None

        for cidx in range(0, nx):
            # connect all resistors in the column
            for ridx in range(1, ny):
                _, a = self.get_res_ports(ridx - 1, cidx)
                b, _ = self.get_res_ports(ridx, cidx)
                vlay = a.layer_id + 1
                vtr = self.grid.coord_to_nearest_track(vlay, a.middle, half_track=True)
                self.connect_to_tracks([a, b], TrackID(vlay, vtr))

            # snake between columns
            bot_warr, _ = self.get_res_ports(0, cidx)
            _, top_warr = self.get_res_ports(ny - 1, cidx)
            if last_port is None:
                last_port = top_warr, 1
                port_out = bot_warr
            else:
                if last_port[1] == 0:
                    self.connect_wires([last_port[0], bot_warr])
                    last_port = top_warr, 1
                else:
                    self.connect_wires([last_port[0], top_warr])
                    last_port = bot_warr, 0

        return port_out, last_port[0]

    def draw_layout(self):
        # type: () -> None

        kwargs = self.params.copy()
        nx = kwargs.pop('nx')
        ny = kwargs.pop('ny')

        if nx % 2 != 0:
            raise ValueError('nx = %d must be even.' % nx)

        # draw array
        self.draw_array(nx=nx, ny=ny, edge_space=True, **kwargs)

        # connect resistor in series
        port_out, port_bias = self.connect_series_resistor(nx, ny)

        vm_layer = self.bot_layer_id + 1
        vm_width = self.w_tracks[1]

        # connect output
        vtid = self.grid.coord_to_nearest_track(vm_layer, port_out.middle, half_track=True)
        vtid = TrackID(vm_layer, vtid, width=vm_width)
        warr = self.connect_to_tracks(port_out, vtid)
        self.add_pin('out', warr, show=False)

        # connect bias
        vtid = self.grid.coord_to_nearest_track(vm_layer, port_bias.middle, half_track=True)
        vtid = TrackID(vm_layer, vtid, width=vm_width)
        warr = self.connect_to_tracks(port_bias, vtid, min_len_mode=-1)
        self.add_pin('bias', warr, show=False)


class HighPassFilter(TemplateBase):
    """An template for creating high pass filter.

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
        super(HighPassFilter, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            res_type='reference',
            em_specs={},
            show_pins=True,
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
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            cap_edge_margin='margin between cap to block edge',
            nx='number of resistor columns.',
            ny='number of resistor rows.',
            num_cap_layer='number of layers to use for AC coupling cap.',
            cap_height='capacitor height.',
            io_width='input/output track width',
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            sup_width='supply track width.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None

        res = self.grid.resolution
        edge_margin = int(round(self.params['cap_edge_margin'] / res))
        cap_height = int(round(self.params['cap_height'] / res))
        num_cap_layer = self.params['num_cap_layer']
        show_pins = self.params['show_pins']

        # place instances
        io_layer, io_width, cap_yt, resout = self.place()

        # draw AC coupling caps bounding boxes
        # figure out cap left/right X coordinate
        blk_w = self.grid.get_size_dimension(self.size, unit_mode=True)[0]
        xl = edge_margin
        xr = blk_w - edge_margin

        # draw mom caps and get cap ports
        cap_yt = min(cap_yt, edge_margin + cap_height)
        cap_box = BBox(xl, edge_margin, xr, cap_yt, res, unit_mode=True)
        # make sure both left and right ports on vertical layers are in
        port_parity = {lay: (0, 1) for lay in range(io_layer, io_layer + num_cap_layer, 2)}
        for lay in range(io_layer + 1, io_layer + num_cap_layer, 2):
            port_parity[lay] = (1, 1)
        cap_ports = self.add_mom_cap(cap_box, io_layer, num_cap_layer,
                                     port_widths=io_width, port_parity=port_parity)

        out_layer = io_layer + num_cap_layer - 1
        self.connect_to_tracks(resout, cap_ports[io_layer][0][0].track_id)
        self.add_pin('out', cap_ports[out_layer][0], show=show_pins)
        self.add_pin('in', cap_ports[out_layer][1], show=show_pins)

    def place(self):
        res_params = self.params.copy()
        sub_lch = res_params.pop('sub_lch')
        sub_w = res_params.pop('sub_w')
        io_width = res_params.pop('io_width')
        sup_width = res_params.pop('sup_width')

        # compute resistor output track indices and create resistor
        io_layer = BiasResistor.get_port_layer_id(self.grid.tech_info) + 2

        res_master = self.new_template(params=res_params, temp_cls=BiasResistor)

        # draw contact and move array up
        sub_type = self.params['sub_type']
        show_pins = self.params['show_pins']
        top_layer, nx_arr, ny_arr = res_master.size
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            top_layer=top_layer,
            blk_width=nx_arr,
            show_pins=False,
        )
        _, blk_h = self.grid.get_size_pitch(top_layer)
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        ny_shift = sub_master.size[2]
        res_inst = self.add_instance(res_master, inst_name='XRES')
        top_yo = (ny_arr + ny_shift) * blk_h
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(0.0, top_yo), orient='MX')

        # recompute array_box/size
        self.size = top_layer, nx_arr, ny_arr + ny_shift
        self.array_box = top_inst.array_box.extend(y=0, unit_mode=True)

        # export supply
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        sup_wa = top_inst.get_all_port_pins(port_name)[0]
        hm_layer = sup_wa.layer_id
        vm_layer = hm_layer + 1
        vm_tr = self.grid.coord_to_nearest_track(vm_layer, sup_wa.middle, half_track=True)
        sup_wa = self.connect_to_tracks(sup_wa, TrackID(vm_layer, vm_tr, width=sup_width))
        self.add_pin(port_name, sup_wa, show=show_pins)

        # export bias ports
        bias_warr = res_inst.get_all_port_pins('bias')[0]
        self.add_pin('bias', bias_warr, show=show_pins)

        return io_layer, io_width, top_inst.array_box.bottom_unit, res_inst.get_all_port_pins('out')[0]
