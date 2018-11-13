# -*- coding: utf-8 -*-

from typing import Dict, Any, Set

from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing import TrackID, WireArray
from bag.layout.util import BBox

from ..resistor.core import ResArrayBase
from ..analog_core import AnalogBase, SubstrateContact
from ..passives.hp_filter import HighPassFilter


class DLevCap(TemplateBase):
    """An template for AC coupling clock arrays

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
        super(DLevCap, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_layer='Number of cap layers.',
            bot_layer='cap bottom layer.',
            port_widths='port widths',
            io_width='input/output width.',
            io_space='input/output spacing.',
            width='cap width.',
            height='cap height.',
            space='cap spacing.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, num_layer, bot_layer, port_widths, io_width,
                            io_space, width, height, space, show_pins):

        res = self.grid.resolution
        width = int(round(width / res))
        height = int(round(height / res))
        space = int(round(space / res))

        io_pitch = io_width + io_space
        io_layer = AnalogBase.get_mos_conn_layer(self.grid.tech_info) + 1
        vm_layer = io_layer + 1
        outp_tr = (io_width - 1) / 2
        inp_tr = outp_tr + io_pitch
        inn_tr = inp_tr + io_pitch
        outn_tr = inn_tr + io_pitch
        tr_list = [outp_tr, inp_tr, inn_tr, outn_tr]

        cap_yb = self.grid.get_wire_bounds(io_layer, outn_tr, width=io_width, unit_mode=True)[1]
        cap_yb += space

        # draw caps
        cap_bboxl = BBox(space, cap_yb, width + space, cap_yb + height, res, unit_mode=True)
        cap_bboxr = cap_bboxl.move_by(dx=width + space, unit_mode=True)
        capl_ports = self.add_mom_cap(cap_bboxl, bot_layer, num_layer, port_widths=port_widths)
        capr_ports = self.add_mom_cap(cap_bboxr, bot_layer, num_layer, port_widths=port_widths)
        # connect caps to dlev/summer inputs/outputs
        warr_list = [capl_ports[vm_layer][0], capl_ports[vm_layer][1],
                     capr_ports[vm_layer][0], capr_ports[vm_layer][1]]
        hwarr_list = self.connect_matching_tracks(warr_list, io_layer, tr_list, width=io_width)

        for name, warr in zip(('outp', 'inp', 'inn', 'outn'), hwarr_list):
            self.add_pin(name, warr, show=show_pins)

        # calculate size
        top_layer = bot_layer + num_layer - 1
        if self.grid.get_direction(top_layer) == 'x':
            yt = capr_ports[top_layer][1][0].get_bbox_array(self.grid).top_unit
            xr = capr_ports[top_layer - 1][1][0].get_bbox_array(self.grid).right_unit
        else:
            yt = capr_ports[top_layer - 1][1][0].get_bbox_array(self.grid).top_unit
            xr = capr_ports[top_layer][1][0].get_bbox_array(self.grid).right_unit

        self.size = self.grid.get_size_tuple(top_layer, xr + space, yt, round_up=True, unit_mode=True)
        self.array_box = self.bound_box

class RXClkArray(TemplateBase):
    """An template for AC coupling clock arrays

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
        super(RXClkArray, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._mid_tracks = None
        self._output_layer = None
        self._track_pitch = None

    @property
    def track_pitch(self):
        return self._track_pitch

    @property
    def output_layer(self):
        return self._output_layer

    @property
    def mid_tracks(self):
        return self._mid_tracks

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
            sup_width=3,
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
            passive_params='High-pass filter passives parameters.',
            io_width='input/output track width.',
            sup_width='supply track width.',
            clk_names='output clock names.',
            sub_types='substrate types.',
            clk_locs='output clock locations.',
            parity='input/output clock parity.',
            show_pins='True to draw pin layouts.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, passive_params, io_width, sup_width,
                            clk_names, sub_types, clk_locs, parity, show_pins):
        hpf_params = passive_params.copy()
        hpf_params['show_pins'] = False

        # get high pass filter size and io layer
        num_blocks = len(clk_names)
        hpf_master = self.new_template(params=hpf_params, temp_cls=HighPassFilter)
        hpfw, hpfh = self.grid.get_size_dimension(hpf_master.size, unit_mode=True)
        io_layer = hpf_master.get_port('in').get_pins()[0].layer_id + 1

        # calculate supply track index
        port_name = 'VDD' if passive_params['sub_type'] == 'ntap' else 'VSS'
        sup_warr = hpf_master.get_port(port_name).get_pins()[0]
        sup_layer = sup_warr.layer_id + 1
        vss_tr = self.grid.coord_to_nearest_track(sup_warr.layer_id + 1, sup_warr.middle, half_track=True)
        sup_space = self.grid.get_num_space_tracks(sup_layer, sup_width)
        vdd_tr = vss_tr + sup_width + sup_space

        # calculate output tracks
        num_tracks = self.grid.get_num_tracks(hpf_master.size, io_layer)
        ltr, mtr, rtr = self.grid.get_evenly_spaced_tracks(3, num_tracks, io_width, half_end_space=True)

        prefix = 'clkp' if parity == 1 else 'clkn'

        in_list = []
        sup_dict = {'VDD': [], 'VSS': []}
        self._output_layer = io_layer
        self._track_pitch = mtr - ltr
        self._mid_tracks = []
        for idx, out_name, sub_type, out_loc in zip(range(num_blocks), clk_names, sub_types, clk_locs):
            offset = num_tracks * idx
            iid = mtr + offset
            self._mid_tracks.append(iid)
            if out_name:
                hpf_params['sub_type'] = sub_type
                hpf_master = self.new_template(params=hpf_params, temp_cls=HighPassFilter)
                inst = self.add_instance(hpf_master, 'XHPF', loc=(hpfw * idx, 0), unit_mode=True)

                if out_loc[parity] == 0:
                    oid = iid
                elif out_loc[parity] > 0:
                    oid = rtr + offset
                else:
                    oid = ltr + offset

                inwarr = inst.get_all_port_pins('in')[0]
                outwarr = inst.get_all_port_pins('out')[0]
                inwarr = self.connect_to_tracks(inwarr, TrackID(io_layer, iid, width=io_width), min_len_mode=0)
                outwarr = self.connect_to_tracks(outwarr, TrackID(io_layer, oid, width=io_width), min_len_mode=-1)
                in_list.append(inwarr)
                self.add_pin(prefix + '_' + out_name, outwarr, show=show_pins)
                self.reexport(inst.get_port('bias'), net_name='bias_' + out_name, show=show_pins)
                port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
                sup_dict[port_name].extend(inst.get_all_port_pins(port_name))

        # export inputs
        self.add_pin(prefix, in_list, label=prefix + ':', show=show_pins)
        # export supplies
        for name, tidx in (('VSS', vss_tr), ('VDD', vdd_tr)):
            warr_list = sup_dict[name]
            if warr_list:
                tid = TrackID(sup_layer, tidx, width=sup_width)
                self.add_pin(name, self.connect_to_tracks(warr_list, tid), show=show_pins)

        # calculate size
        top_layer = io_layer
        self.size = self.grid.get_size_tuple(io_layer, hpfw * num_blocks, hpfh, round_up=True, unit_mode=True)
        self.array_box = self.bound_box


class BiasBusIO(TemplateBase):
    """An template for AC coupling clock arrays

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
        super(BiasBusIO, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            show_pins=True,
            track_width=1,
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
            io_names='names of wires to connect.',
            sup_name='supply port name.',
            reserve_tracks='list of name/layer/track/width to reserve.',
            bus_layer='bus wire layer.',
            show_pins='True to draw pin layout.',
            bus_margin='number of tracks to save as margins to adjacent blocks.',
            track_width='width of each track',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _get_bound(self, dim, bus_layer, mode):
        tr_lower = self.grid.find_next_track(bus_layer - 1, dim, mode=mode, unit_mode=True)
        tr_upper = self.grid.find_next_track(bus_layer + 1, dim, mode=mode, unit_mode=True)
        index = 1 if mode > 0 else 0
        dim_lower = self.grid.get_wire_bounds(bus_layer - 1, tr_lower, unit_mode=True)[index]
        dim_upper = self.grid.get_wire_bounds(bus_layer + 1, tr_upper, unit_mode=True)[index]
        if mode > 0:
            return max(dim_lower, dim_upper)
        else:
            return min(dim_lower, dim_upper)

    def _draw_layout_helper(self, io_names, sup_name, reserve_tracks, bus_layer, bus_margin, show_pins, track_width):
        # compute bus length
        track_space = self.grid.get_num_space_tracks(bus_layer, width_ntr=track_width)
        io_names = {name: idx for idx, name in enumerate(io_names)}
        bus_lower = None
        bus_upper = None
        reserve_list = []
        io_dict = {bus_layer - 1: [], bus_layer + 1: []}
        for name, layer, track, width in reserve_tracks:
            if layer == bus_layer - 1 or layer == bus_layer + 1:
                reserve_list.append((layer, track, width))
                num_space = self.grid.get_num_space_tracks(layer, width_ntr=width)
                lower = self.grid.get_wire_bounds(layer, track - num_space, width=width, unit_mode=True)[0]
                upper = self.grid.get_wire_bounds(layer, track + num_space, width=width, unit_mode=True)[1]
                if bus_lower is None:
                    bus_lower, bus_upper = lower, upper
                else:
                    bus_lower = min(bus_lower, lower)
                    bus_upper = max(bus_upper, upper)

                idx = io_names.get(name, -1)
                if idx >= 0:
                    io_dict[layer].append((name, idx, track, width))

        bus_upper = self._get_bound(bus_upper, bus_layer, 1)
        # draw input buses
        track_pitch = track_width + track_space
        track0 = bus_margin + track_space + (track_width + 1) / 2
        wire_layers = (bus_layer - 1, bus_layer + 1)
        for lay in wire_layers:
            for name, idx, track, width in io_dict[lay]:
                mid = self.grid.track_to_coord(lay, track, unit_mode=True)
                cur_tr_idx = track0 + idx * track_pitch
                w = self.add_wires(bus_layer, cur_tr_idx, 0, mid, width=track_width, unit_mode=True)
                w_in = self.connect_to_tracks(w, TrackID(lay, track, width=width), min_len_mode=0)
                pin_w = WireArray(TrackID(bus_layer, cur_tr_idx, width=track_width), 0,
                                  self.grid.get_min_length(bus_layer, 1, unit_mode=True),
                                  res=self.grid.resolution, unit_mode=True)
                self.add_pin(name, pin_w, show=show_pins)
                self.add_pin(name + '_in', w_in, show=show_pins)

        # compute size
        last_sup_track = track0 + (len(io_names) - 1) * track_pitch + (track_width + 1) / 2 + track_space
        bus_wl = self.grid.get_wire_bounds(bus_layer, bus_margin, unit_mode=True)[0]
        bus_wu = self.grid.get_wire_bounds(bus_layer, last_sup_track, unit_mode=True)[1]
        size_wu = self.grid.get_wire_bounds(bus_layer, last_sup_track + bus_margin, unit_mode=True)[1]
        if self.grid.get_direction(bus_layer) == 'x':
            cur_width, cur_height = bus_upper, size_wu
        else:
            cur_width, cur_height = size_wu, bus_upper
        self.size = self.grid.get_size_tuple(bus_layer + 2, cur_width, cur_height, round_up=True, unit_mode=True)
        self.array_box = self.bound_box

        # reserve tracks
        for lay, track, width in reserve_list:
            self.reserve_tracks(lay, track, width)

        # draw supply wires
        sup_warr_list = None
        sup_pitch = last_sup_track - bus_margin
        for lay in wire_layers:
            tr_max = self.grid.find_next_track(lay, bus_upper, mode=1, unit_mode=True)
            tr_idx_list = list(range(1, tr_max + 1, 2))
            avail_list = self.get_available_tracks(lay, tr_idx_list, bus_wl, bus_wu, unit_mode=True)
            # connect
            sup_warr_list = []
            for aidx in avail_list:
                sup_warr_list.append(self.add_wires(lay, aidx, bus_wl, bus_wu, unit_mode=True))
            self.connect_to_tracks(sup_warr_list, TrackID(bus_layer, bus_margin, width=1,
                                                          num=2, pitch=sup_pitch),
                                   track_lower=0)

        num_sup_tracks = self.grid.get_num_tracks(self.size, bus_layer + 2)
        num_top_sup = 1
        sup_w = self.grid.get_max_track_width(bus_layer + 2, num_top_sup, num_sup_tracks)
        # TODO: find best way to do this in process independent way
        while sup_w > 10:
            num_top_sup += 1
            sup_w = self.grid.get_max_track_width(bus_layer + 2, num_top_sup, num_sup_tracks)
        sup_tr_list = self.grid.get_evenly_spaced_tracks(num_top_sup, num_sup_tracks, sup_w)
        for sup_idx in sup_tr_list:
            sup_warr = self.connect_to_tracks(sup_warr_list, TrackID(bus_layer + 2, sup_idx, width=sup_w))
            self.add_pin(sup_name, sup_warr, show=show_pins)


class BiasBusCorner(TemplateBase):
    """An template for AC coupling clock arrays

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
        super(BiasBusCorner, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            io_names='names of wires to connect.',
            sup_name='supply port name.',
            hm_layer='horizontal wire layer.',
            vm_layer='vertical wire layer.',
            show_pins='True to draw pin layout.',
            bus_margin='number of tracks to save as margins to adjacent blocks.',
        )

    def draw_layout(self):
        # type: () -> None
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, io_names, sup_name, hm_layer, vm_layer, bus_margin, show_pins):
        num_wires = 2 + io_names

        # compute size
        top_layer = max(hm_layer, vm_layer) + 1
        width = self.grid.get_track_pitch(vm_layer, unit_mode=True) * (num_wires + 2 * bus_margin)
        height = self.grid.get_track_pitch(hm_layer, unit_mode=True) * (num_wires + 2 * bus_margin)
        self.size = self.grid.get_size_tuple(top_layer, width, height, round_up=True, unit_mode=True)
        self.array_box = self.bound_box
        xr = self.array_box.right_unit
        yt = self.array_box.top_unit

        # get via extension
        bext, text = self.grid.get_via_extensions(min(hm_layer, vm_layer), 1, 1, unit_mode=True)
        if hm_layer < vm_layer:
            hext, vext = bext, text
        else:
            hext, vext = text, bext

        # draw supply wires on signal layers
        xl = self.grid.get_wire_bounds(vm_layer, bus_margin, unit_mode=True)[0]
        xl -= hext
        supb = self.add_wires(hm_layer, bus_margin, xl, xr)
        supt = self.add_wires(hm_layer, bus_margin + num_wires - 1, xl, xr)
        supl, supr = self.connect_differential_tracks(supb, supt, vm_layer, bus_margin,
                                                      bus_margin + num_wires - 1, track_upper=yt)

        # draw signal wires
        for idx, name in enumerate(sup_name):
            tidx = bus_margin + 1 + idx
            xl = self.grid.get_wire_bounds(vm_layer, tidx, unit_mode=True)[0] - hext
            wh = self.add_wires(hm_layer, tidx, xl, xr)
            self.add_pin(name + '_h', wh, show=show_pins)
            wv = self.connect_to_tracks(wh, TrackID(vm_layer, tidx), track_upper=yt)
            self.add_pin(name + '_v', wv, show=show_pins)


class CTLECore(ResArrayBase):
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
        super(CTLECore, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_cap_layer='number of layers to use for AC coupling cap.',
            cap_port_widths='capacitor port widths.',
            cap_port_offset='capacitor port index offset from common mode wire.',
            cap_height='capacitor height.',
            num_r1='number of r1 segments.',
            num_r2='number of r2 segments.',
            num_dumc='number of dummy columns.',
            num_dumr='number of dummy rows.',
            io_width='input/output track width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            show_pins='True to draw pin layous.',
        )

    def draw_layout(self):
        # type: () -> None

        kwargs = self.params.copy()
        num_r1 = kwargs.pop('num_r1')
        num_r2 = kwargs.pop('num_r2')
        num_dumc = kwargs.pop('num_dumc')
        num_dumr = kwargs.pop('num_dumr')
        show_pins = kwargs.pop('show_pins')
        io_width = kwargs.pop('io_width')
        cap_edge_margin = kwargs.pop('cap_edge_margin')
        num_cap_layer = kwargs.pop('num_cap_layer')
        cap_port_widths = kwargs.pop('cap_port_widths')
        cap_port_offset = kwargs.pop('cap_port_offset')
        cap_height = int(round(kwargs.pop('cap_height') / self.grid.resolution))

        if num_r1 % 2 != 0 or num_r2 % 2 != 0:
            raise ValueError('num_r1 and num_r2 must be even.')
        if num_dumc <= 0 or num_dumr <= 0:
            raise ValueError('num_dumr and num_dumc must be greater than 0.')

        # draw array
        nr1 = num_r1 // 2
        nr2 = num_r2 // 2
        parent_grid = self.grid
        self.draw_array(nx=4 + num_dumc * 2, ny=2 * (max(nr1, nr2) + num_dumr),
                        edge_space=False, **kwargs)

        # connect wires
        sup_name = 'VDD' if kwargs['sub_type'] == 'ntap' else 'VSS'
        supt, supb = self._connect_dummies(nr1, nr2, num_dumr, num_dumc, sup_name, show_pins)
        inp, inn, outp, outn, outcm = self._connect_snake(nr1, nr2, num_dumr, num_dumc, io_width, show_pins)

        # calculate capacitor bounding box
        res = self.grid.resolution
        cap_edge_margin = int(round(cap_edge_margin / res))
        hm_layer = outcm.layer_id + 1
        io_layer = hm_layer + 2
        mid_coord = int(round(outcm.middle / res))
        cm_tr = self.grid.coord_to_track(io_layer, mid_coord, unit_mode=True)
        io_width = cap_port_widths[2]
        cap_yb = self.grid.get_wire_bounds(io_layer, cap_port_offset + cm_tr, width=io_width, unit_mode=True)[0]
        num_sp = self.grid.get_num_space_tracks(hm_layer, cap_port_widths[0])
        cap_yt = self.grid.get_wire_bounds(hm_layer, supt.track_id.base_index - num_sp - 1, unit_mode=True)[1]
        cap_yt = min(cap_yt, cap_yb + cap_height)
        cap_xl = cap_edge_margin
        cap_xr = self.array_box.right_unit - cap_edge_margin

        # construct port parity
        top_parity, bot_parity = {}, {}
        for cap_lay in range(hm_layer, hm_layer + num_cap_layer):
            if self.grid.get_direction(cap_lay) == 'x':
                top_parity[cap_lay] = (1, 0)
                bot_parity[cap_lay] = (0, 1)
            else:
                top_parity[cap_lay] = (0, 1)
                bot_parity[cap_lay] = (0, 1)

        cap_top = self.add_mom_cap(BBox(cap_xl, cap_yb, cap_xr, cap_yt, res, unit_mode=True), hm_layer,
                                   num_cap_layer, port_widths=cap_port_widths, port_parity=top_parity)
        cap_yb = 2 * mid_coord - cap_yb
        cap_yt = 2 * mid_coord - cap_yt
        cap_bot = self.add_mom_cap(BBox(cap_xl, cap_yt, cap_xr, cap_yb, res, unit_mode=True), hm_layer,
                                   num_cap_layer, port_widths=cap_port_widths, port_parity=bot_parity)

        self.connect_to_tracks(inp, cap_top[hm_layer][0][0].track_id)
        self.connect_to_tracks(outp, cap_top[hm_layer][1][0].track_id)
        self.connect_to_tracks(inn, cap_bot[hm_layer][0][0].track_id)
        self.connect_to_tracks(outn, cap_bot[hm_layer][1][0].track_id)

        top_layer = hm_layer + num_cap_layer - 1
        self.add_pin('inp', cap_top[top_layer][0], show=show_pins)
        self.add_pin('outp', cap_top[io_layer][1], show=show_pins)
        self.add_pin('inn', cap_bot[top_layer][0], show=show_pins)
        self.add_pin('outn', cap_bot[io_layer][1], show=show_pins)

        self.set_size_from_array_box(top_layer, parent_grid)

    def _connect_snake(self, nr1, nr2, ndumr, ndumc, io_width, show_pins):
        nrow_half = max(nr1, nr2) + ndumr
        for idx in range(nr1):
            if idx != 0:
                self._connect_mirror(nrow_half, (idx - 1, ndumc), (idx, ndumc), 1, 0)
                self._connect_mirror(nrow_half, (idx - 1, ndumc + 1), (idx, ndumc + 1), 1, 0)
            if idx == nr1 - 1:
                self._connect_mirror(nrow_half, (idx, ndumc), (idx, ndumc + 1), 1, 1)
        for idx in range(nr2):
            if idx != 0:
                self._connect_mirror(nrow_half, (idx - 1, ndumc + 2), (idx, ndumc + 2), 1, 0)
                self._connect_mirror(nrow_half, (idx - 1, ndumc + 3), (idx, ndumc + 3), 1, 0)
            if idx == nr2 - 1:
                self._connect_mirror(nrow_half, (idx, ndumc + 2), (idx, ndumc + 3), 1, 1)

        # connect outp/outn
        outpl = self.get_res_ports(nrow_half, ndumc + 1)[0]
        outpr = self.get_res_ports(nrow_half, ndumc + 2)[0]
        outp = self.connect_wires([outpl, outpr])[0]
        outnl = self.get_res_ports(nrow_half - 1, ndumc + 1)[1]
        outnr = self.get_res_ports(nrow_half - 1, ndumc + 2)[1]
        outn = self.connect_wires([outnl, outnr])[0]

        vm_layer = outp.layer_id + 1
        vm_tr = self.grid.coord_to_nearest_track(vm_layer, outp.middle, half_track=True)
        vm_tid = TrackID(vm_layer, vm_tr, width=io_width)
        outp = self.connect_to_tracks(outp, vm_tid, min_len_mode=1)
        outn = self.connect_to_tracks(outn, vm_tid, min_len_mode=-1)

        # connect inp/inn
        inp = self.get_res_ports(nrow_half, ndumc)[0]
        inn = self.get_res_ports(nrow_half - 1, ndumc)[1]
        mid = (self.get_res_ports(nrow_half, ndumc - 1)[0].middle + inp.middle) / 2
        vm_tr = self.grid.coord_to_nearest_track(vm_layer, mid, half_track=True)
        vm_tid = TrackID(vm_layer, vm_tr, width=io_width)
        inp = self.connect_to_tracks(inp, vm_tid, min_len_mode=1)
        inn = self.connect_to_tracks(inn, vm_tid, min_len_mode=-1)

        # connect outcm
        cmp = self.get_res_ports(nrow_half, ndumc + 3)[0]
        cmn = self.get_res_ports(nrow_half - 1, ndumc + 3)[1]
        vm_tr = self.grid.coord_to_nearest_track(vm_layer, cmp.middle, half_track=True)
        vm_tid = TrackID(vm_layer, vm_tr, width=io_width)
        outcm_v = self.connect_to_tracks([cmp, cmn], vm_tid)
        hm_layer = vm_layer + 1
        hm_tr = self.grid.coord_to_nearest_track(hm_layer, outcm_v.middle, half_track=True)
        outcm = self.connect_to_tracks(outcm_v, TrackID(hm_layer, hm_tr, width=io_width), track_lower=0)
        self.add_pin('outcm', outcm, show=show_pins)

        return inp, inn, outp, outn, outcm_v

    def _connect_mirror(self, offset, loc1, loc2, port1, port2):
        r1, c1 = loc1
        r2, c2 = loc2
        for sgn in (-1, 1):
            cur_r1 = offset + sgn * r1
            cur_r2 = offset + sgn * r2
            if sgn < 0:
                cur_r1 -= 1
                cur_r2 -= 1
            if sgn < 0:
                cur_port1 = 1 - port1
                cur_port2 = 1 - port2
            else:
                cur_port1 = port1
                cur_port2 = port2
            wa1 = self.get_res_ports(cur_r1, c1)[cur_port1]
            wa2 = self.get_res_ports(cur_r2, c2)[cur_port2]
            if wa1.track_id.base_index == wa2.track_id.base_index:
                self.connect_wires([wa1, wa2])
            else:
                vm_layer = wa1.layer_id + 1
                vm = self.grid.coord_to_nearest_track(vm_layer, wa1.middle, half_track=True)
                self.connect_to_tracks([wa1, wa2], TrackID(vm_layer, vm))

    def _connect_dummies(self, nr1, nr2, ndumr, ndumc, sup_name, show_pins):
        num_per_col = [0] * ndumc + [nr1, nr1, nr2, nr2] + [0] * ndumc
        nrow_half = max(nr1, nr2) + ndumr
        bot_warrs, top_warrs = [], []
        for col_idx, res_num in enumerate(num_per_col):
            if res_num == 0:
                cur_ndum = nrow_half * 2
                bot_idx_list = [0]
            else:
                cur_ndum = nrow_half - res_num
                bot_idx_list = [0, nrow_half + res_num]

            for bot_idx in bot_idx_list:
                top_idx = bot_idx + cur_ndum
                warr_list = []
                for ridx in range(bot_idx, top_idx):
                    bp, tp = self.get_res_ports(ridx, col_idx)
                    warr_list.append(bp)
                    warr_list.append(tp)
                vm_layer = warr_list[0].layer_id + 1
                vm = self.grid.coord_to_nearest_track(vm_layer, warr_list[0].middle, half_track=True)
                sup_warr = self.connect_to_tracks(warr_list, TrackID(vm_layer, vm))
                if bot_idx == 0:
                    bot_warrs.append(sup_warr)
                if bot_idx != 0 or res_num == 0:
                    top_warrs.append(sup_warr)

        hm_layer = bot_warrs[0].layer_id + 1
        hm_pitch = self.grid.get_track_pitch(hm_layer, unit_mode=True)
        num_hm_tracks = self.array_box.height_unit // hm_pitch
        btr = self.connect_to_tracks(bot_warrs, TrackID(hm_layer, 0))
        ttr = self.connect_to_tracks(top_warrs, TrackID(hm_layer, num_hm_tracks - 1))
        self.add_pin(sup_name + 'B', btr, show=show_pins)
        self.add_pin(sup_name + 'T', ttr, show=show_pins)

        return ttr, btr


class CTLE(TemplateBase):
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
        super(CTLE, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_cap_layer='number of layers to use for AC coupling cap.',
            cap_port_widths='capacitor port widths.',
            cap_port_offset='capacitor port index offset from common mode wire.',
            cap_height='capacitor height.',
            num_r1='number of r1 segments.',
            num_r2='number of r2 segments.',
            num_dumc='number of dummy columns.',
            num_dumr='number of dummy rows.',
            io_width='input/output track width.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            res_type='the resistor type.',
            em_specs='EM specifications for the termination network.',
            show_pins='True to draw pin layous.',
            sup_width='supply track width.',
            sub_lch='substrate contact channel length.',
            sub_w='substrate contact width.',
        )

    def draw_layout(self):
        # type: () -> None
        core_params = self.params.copy()
        sub_lch = core_params.pop('sub_lch')
        sub_w = core_params.pop('sub_w')
        sup_width = core_params.pop('sup_width')
        show_pins = self.params['show_pins']
        core_params['show_pins'] = False

        core_master = self.new_template(params=core_params, temp_cls=CTLECore)

        # draw contact and move array up
        sub_type = self.params['sub_type']
        top_layer, nx_arr, ny_arr = core_master.size
        w_pitch, h_pitch = self.grid.get_size_pitch(top_layer, unit_mode=True)
        sub_params = dict(
            lch=sub_lch,
            w=sub_w,
            sub_type=sub_type,
            threshold=self.params['threshold'],
            top_layer=top_layer,
            blk_width=nx_arr,
            show_pins=False,
        )
        sub_master = self.new_template(params=sub_params, temp_cls=SubstrateContact)
        bot_inst = self.add_instance(sub_master, inst_name='XBSUB')
        ny_shift = sub_master.size[2]
        core_inst = self.add_instance(core_master, loc=(0, ny_shift * h_pitch), inst_name='XRES', unit_mode=True)
        top_yo = (ny_arr + 2 * ny_shift) * h_pitch
        top_inst = self.add_instance(sub_master, inst_name='XTSUB', loc=(0, top_yo), orient='MX', unit_mode=True)

        # connect supplies
        port_name = 'VDD' if sub_type == 'ntap' else 'VSS'
        warrt = self._connect_supply(core_inst, top_inst, port_name, 'T', sup_width)
        warrb = self._connect_supply(core_inst, bot_inst, port_name, 'B', sup_width)
        self.add_pin(port_name, warrt, show=show_pins)
        self.add_pin(port_name, warrb, show=show_pins)

        # export ports
        for pname in core_inst.port_names_iter():
            if not pname.startswith(port_name):
                self.reexport(core_inst.get_port(pname), show=show_pins)

        # compute size
        self.array_box = top_inst.array_box.merge(bot_inst.array_box)
        self.set_size_from_array_box(top_layer)

    def _connect_supply(self, core, sub, name, suffix, sup_width):
        warr1 = sub.get_all_port_pins(name)[0]
        warr2 = core.get_all_port_pins(name + suffix)[0]
        hm_layer = warr1.layer_id
        vm_layer = hm_layer + 1
        vm_id = self.grid.coord_to_nearest_track(vm_layer, warr1.middle, half_track=True)
        return self.connect_to_tracks([warr1, warr2], TrackID(vm_layer, vm_id, sup_width))
