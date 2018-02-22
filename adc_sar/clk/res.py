# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Set, Any

from bag.layout.routing import TrackID

from abs_templates_ec.resistor.core import ResArrayBase

if TYPE_CHECKING:
    from bag.layout.template import TemplateDB


class ResFeedbackCore(ResArrayBase):
    """An template for creating inverter feedback resistors.

    Parameters
    ----------
    temp_db : TemplateDB
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        ResArrayBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            l='unit resistor length, in meters.',
            w='unit resistor width, in meters.',
            sub_type='the substrate type.',
            threshold='the substrate threshold flavor.',
            nx='number of resistors in a row.',
            ny='number of resistors in a column.',
            em_specs='EM specifications for the termination network.',
            show_pins='True to show pins.',
            res_options='Configuration dictionary for ResArrayBase.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            em_specs=None,
            show_pins=True,
            res_options=None,
        )

    def draw_layout(self):
        # type: () -> None
        l = self.params['l']
        w = self.params['w']
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        nx = self.params['nx']
        ny = self.params['ny']
        em_specs = self.params['em_specs']
        show_pins = self.params['show_pins']
        res_options = self.params['res_options']

        if nx <= 0:
            raise ValueError('number of resistors in a row must be positive.')
        if ny % 2 != 0:
            raise ValueError('number of resistors in a column must be even')
        if res_options is None:
            res_options = {}

        min_tracks = (1, 1, 1, 1)
        top_layer = self.bot_layer_id + len(min_tracks) - 1
        self.draw_array(l, w, sub_type, threshold, nx=nx, ny=ny, min_tracks=min_tracks,
                        em_specs=em_specs, top_layer=top_layer, connect_up=True, **res_options)

        # connect row resistors
        vm_layer = self.bot_layer_id + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        left_row = 0
        right_row = 1
        port_prev = None
        vm_width = self.w_tracks[1]
        for row_idx in range(ny):
            for col_idx in range(nx):
                ports_l = self.get_res_ports(row_idx, col_idx)
                if col_idx == nx - 1:
                    ports_r = self.get_res_ports(row_idx, col_idx)
                else:
                    ports_r = self.get_res_ports(row_idx, col_idx + 1)
                con_par = (col_idx + row_idx) % 2
                # self.connect_wires([ports_l[con_par], ports_r[con_par]])
                if col_idx == 0:
                    # column 0, left side
                    if row_idx > 0:
                        # not in the first row
                        ports_l_below = self.get_res_ports(row_idx - 1, col_idx)
                        if left_row == 1 and row_idx != ny // 2:
                            # connect the left side of this row downward
                            vl_id = self.grid.coord_to_nearest_track(vm_layer, ports_l[con_par].middle, mode=-1,
                                                                     half_track=True)
                            v_tid = TrackID(vm_layer, vl_id, width=vm_width)
                            cur_list_l = [ports_l[con_par], ports_l_below[1 - con_par]]
                            self.connect_to_tracks(cur_list_l, v_tid)
                else:
                    self.connect_wires([ports_l[con_par], port_prev[con_par]])
                port_prev = ports_l

                if col_idx == nx - 2:
                    if row_idx > 0:
                        ports_r_below = self.get_res_ports(row_idx - 1, col_idx + 1)
                        if right_row == 1 and row_idx != ny // 2:
                            vr_id = self.grid.coord_to_nearest_track(vm_layer, ports_r[con_par].middle, mode=1,
                                                                     half_track=True)
                            v_tid = TrackID(vm_layer, vr_id, width=vm_width)
                            # print 'should be making a grid connection'
                            cur_list_r = [ports_r[con_par], ports_r_below[1 - con_par]]
                            self.connect_to_tracks(cur_list_r, v_tid)
            if row_idx > 0:
                left_row = 1 - left_row
                right_row = 1 - right_row

        ym_width = self.w_tracks[3]
        min_len_mode_list = [0] * len(self.w_tracks)
        for row_idx, col_idx, par, name in [(0, 0, -1, 'in1'), ((ny // 2) - 1, nx - 1, 1, 'in2'),
                                            (ny // 2, nx - 1, 1, 'in4'), (ny - 1, 0, -1, 'in3')]:
            ports = self.get_res_ports(row_idx, col_idx)
            con_par = (row_idx + col_idx) % 2
            if par > 0:
                con_par = 1 - con_par
            vl_tidx = self.grid.coord_to_nearest_track(ym_layer, ports[con_par].middle, mode=par, half_track=True)
            vl_tid = TrackID(ym_layer, vl_tidx, width=ym_width)
            warrs = self.connect_with_via_stack(ports[con_par], vl_tid, min_len_mode_list=min_len_mode_list)
            self.add_pin(name, warrs[-1], show=show_pins)
