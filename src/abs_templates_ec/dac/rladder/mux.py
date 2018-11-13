# -*- coding: utf-8 -*-

"""This module contains classes for resistor ladder DAC output muxes.
"""

from typing import Dict, Set, Any

from bag.layout.template import TemplateDB
from bag.layout.digital import StdCellTemplate, StdCellBase
from bag.layout.routing import TrackID


class PassgateRow(StdCellBase):
    """A row of passgates.

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
        super(PassgateRow, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            col_nbits='number of column bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        col_nbits = self.params['col_nbits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        num_col = 2 ** col_nbits

        # get template masters
        pg_params = dict(cell_name='passgate_2x', config_file=config_file)
        pg_master = self.new_template(params=pg_params, temp_cls=StdCellTemplate)

        # add pass gates.
        xo = 0
        pg_inst = self.add_std_instance(pg_master, loc=(xo, 0), nx=num_col + 1, spx=pg_master.std_size[0])

        # set template size
        self.set_std_size((pg_master.std_size[0] * (num_col + 1), 1))

        # connect passgate sources
        s_list = pg_inst.get_all_port_pins('S')
        s_list = self.connect_wires(s_list)
        slay = s_list[0].layer_id + 1
        num_tr = pg_master.get_num_tracks(slay)
        s_tid = TrackID(slay, num_tr - 0.5, num=num_col, pitch=num_tr)
        s_tid = s_tid.transform(self.grid, pg_inst.location_unit, pg_inst.orientation, unit_mode=True)
        self.connect_to_tracks(s_list, s_tid)

        # export passgate ports
        for idx in range(num_col):
            self.reexport(pg_inst.get_port('D', col=idx), 'in<%d>' % idx, show=False)
            self.reexport(pg_inst.get_port('EN', col=idx), 'en<%d>' % idx, show=False)
            self.reexport(pg_inst.get_port('ENB', col=idx), 'enb<%d>' % idx, show=False)

        self.reexport(pg_inst.get_port('D', col=num_col), 'out', show=False)
        self.reexport(pg_inst.get_port('EN', col=num_col), 'en_row', show=False)
        self.reexport(pg_inst.get_port('ENB', col=num_col), 'enb_row', show=False)


class InputBuffer(StdCellBase):
    """resistor ladder mux input buffers.

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
        super(InputBuffer, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_bits='total number of bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        num_bits = self.params['num_bits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        inv_params = dict(cell_name='inv_2x', config_file=config_file)
        inv_master = self.new_template(params=inv_params, temp_cls=StdCellTemplate)

        port_layer = inv_master.get_port('O').get_pins()[0].layer_id + 1
        num_tracks = inv_master.get_num_tracks(port_layer)
        outb_tr = (num_tracks - 1) / 2
        in_tr = outb_tr + 1
        out_tr = outb_tr - 1

        # add inverters
        bot_inst = self.add_std_instance(inv_master, nx=num_bits, spx=inv_master.std_size[0])
        top_inst = self.add_std_instance(inv_master, loc=(0, 1), nx=num_bits, spx=inv_master.std_size[0])

        # connect inputs/outputs
        for idx in range(num_bits):
            tr_off = idx * num_tracks
            bot_in_pin = bot_inst.get_port('I', col=idx).get_pins()
            bot_out_pin = bot_inst.get_port('O', col=idx).get_pins()
            top_in_pin = top_inst.get_port('I', col=idx).get_pins()
            top_out_pin = top_inst.get_port('O', col=idx).get_pins()

            tr = TrackID(port_layer, tr_off + in_tr)
            in_warr = self.connect_to_tracks(bot_in_pin, tr, track_lower=0.0)
            self.add_pin('in<%d>' % (num_bits - 1 - idx), in_warr, show=False)
            tr = TrackID(port_layer, tr_off + out_tr)
            out_warr = self.connect_to_tracks(top_out_pin, tr)
            self.add_pin('out<%d>' % (num_bits - 1 - idx), out_warr, show=False)
            tr = TrackID(port_layer, tr_off + outb_tr)
            outb_warr = self.connect_to_tracks(bot_out_pin + top_in_pin, tr)
            self.add_pin('outb<%d>' % (num_bits - 1 - idx), outb_warr, show=False)

        # set template size
        self.set_std_size((inv_master.std_size[0] * num_bits, 2))


class RowDecoder(StdCellBase):
    """The row decoder.

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
        super(RowDecoder, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            row_nbits='number of column bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        row_nbits = self.params['row_nbits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        if row_nbits <= 0:
            raise ValueError('row_nbits must be positive.')
        num_row = 2 ** row_nbits

        dec_name = 'decoder%d_unit_diff_horiz_1x' % row_nbits
        dec_params = dict(cell_name=dec_name, config_file=config_file)
        dec_master = self.new_template(params=dec_params, temp_cls=StdCellTemplate)

        # add decoders
        inst_list = [self.add_std_instance(dec_master, ny=num_row // 2, spy=2),
                     self.add_std_instance(dec_master, loc=(0, 1), ny=num_row // 2, spy=2)]

        # get input pins and export outputs
        in_lists = [([], [], []) for _ in range(row_nbits)]
        for idx in range(num_row):
            inst = inst_list[idx % 2]
            ridx = idx // 2
            self.reexport(inst.get_port('O', row=ridx), net_name='out<%d>' % idx, show=False)
            self.reexport(inst.get_port('OB', row=ridx), net_name='outb<%d>' % idx, show=False)
            for bit_idx in range(row_nbits):
                in_lists[bit_idx][0].append(inst.get_port('IN<%d>' % bit_idx, row=ridx).get_pins()[0])
                if (idx & (1 << bit_idx)) >> bit_idx == 1:
                    in_lists[bit_idx][2].append(idx)
                else:
                    in_lists[bit_idx][1].append(idx)

        in_layer = in_lists[0][0][0].layer_id + 1
        for idx, (pin_list, outb_idx_list, out_idx_list) in enumerate(in_lists):
            outb_tr = self.grid.find_next_track(in_layer, pin_list[0].lower, mode=1, half_track=True)
            warr = self.connect_to_tracks([pin_list[idx] for idx in outb_idx_list], TrackID(in_layer, outb_tr))
            self.add_pin('inb<%d>' % idx, warr, show=False)
            warr = self.connect_to_tracks([pin_list[idx] for idx in out_idx_list], TrackID(in_layer, outb_tr + 1))
            self.add_pin('in<%d>' % idx, warr, show=False)

        # set template size
        self.set_std_size((dec_master.std_size[0], num_row))


class ColDecoder(StdCellBase):
    """The column decoder.

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
        super(ColDecoder, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            col_nbits='number of column bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        col_nbits = self.params['col_nbits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        num_col = 2 ** col_nbits

        dec_name = 'decoder%d_unit_diff_vert_1x' % col_nbits
        dec_params = dict(cell_name=dec_name, config_file=config_file)
        dec_master = self.new_template(params=dec_params, temp_cls=StdCellTemplate)

        # add decoders
        inst = self.add_std_instance(dec_master, nx=num_col, spx=dec_master.std_size[0])

        # get input pins and export outputs
        in_lists = [([], [], []) for _ in range(col_nbits)]
        for idx in range(num_col):
            self.reexport(inst.get_port('O', col=idx), net_name='out<%d>' % idx, show=False)
            self.reexport(inst.get_port('OB', col=idx), net_name='outb<%d>' % idx, show=False)
            for bit_idx in range(col_nbits):
                in_lists[bit_idx][0].append(inst.get_port('IN<%d>' % bit_idx, col=idx).get_pins()[0])
                if (idx & (1 << bit_idx)) >> bit_idx == 1:
                    in_lists[bit_idx][2].append(idx)
                else:
                    in_lists[bit_idx][1].append(idx)

        in_layer = in_lists[0][0][0].layer_id + 1
        for idx, (pin_list, outb_idx_list, out_idx_list) in enumerate(in_lists):
            outb_tr = self.grid.find_next_track(in_layer, pin_list[0].lower, half_track=True, mode=1)
            warr = self.connect_to_tracks([pin_list[idx] for idx in outb_idx_list], TrackID(in_layer, outb_tr))
            self.add_pin('inb<%d>' % idx, warr, show=False)
            warr = self.connect_to_tracks([pin_list[idx] for idx in out_idx_list], TrackID(in_layer, outb_tr + 1))
            self.add_pin('in<%d>' % idx, warr, show=False)

        # set template size
        self.set_std_size((dec_master.std_size[0] * num_col, dec_master.std_size[1]))


class RLadderMux(StdCellBase):
    """The column decoder.

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
        super(RLadderMux, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            col_nbits='number of column bits.',
            row_nbits='number of row bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        col_nbits = self.params['col_nbits']
        row_nbits = self.params['row_nbits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()

        if row_nbits <= 0 or col_nbits <= 0:
            raise ValueError('row_nbits and col_nbits must be positive.')

        num_row = 2 ** row_nbits
        num_col = 2 ** col_nbits

        tap_params = dict(cell_name='tap_pwr', config_file=config_file)
        tap_master = self.new_template(params=tap_params, temp_cls=StdCellTemplate)
        buf_params = dict(num_bits=col_nbits + row_nbits, config_file=config_file)
        buf_master = self.new_template(params=buf_params, temp_cls=InputBuffer)
        col_params = dict(col_nbits=col_nbits, config_file=config_file)
        col_master = self.new_template(params=col_params, temp_cls=ColDecoder)
        row_params = dict(row_nbits=row_nbits, config_file=config_file)
        row_master = self.new_template(params=row_params, temp_cls=RowDecoder)
        pgr_params = dict(col_nbits=col_nbits, row_nbits=row_nbits, num_space=0, config_file=config_file)
        pgr_master = self.new_template(params=pgr_params, temp_cls=PassgateRow)

        row_offset = col_master.std_size[1]
        if row_offset % 2 == 1:
            # add space between column decoder and passgate
            row_offset += 1

        xoff = tap_master.std_size[0]
        cdec_inst = self.add_std_instance(col_master, loc=(xoff, 0))
        buf_inst = self.add_std_instance(buf_master, loc=(xoff + col_master.std_size[0], 0))

        pgr_inst_list = [self.add_std_instance(pgr_master, loc=(xoff, row_offset),
                                               ny=num_row // 2, spy=2),
                         self.add_std_instance(pgr_master, loc=(xoff, row_offset + 1),
                                               ny=num_row // 2, spy=2),
                         ]

        # find row decoder location so input wires don't collide.
        rdec_in_pin = row_master.get_port('in<%d>' % (row_nbits - 1)).transform(self.grid, orient='MY').get_pins()[0]
        rdec_in_layer = rdec_in_pin.layer_id
        rdec_in_tr = rdec_in_pin.track_id.base_index
        out_tr = buf_inst.get_port('outb<%d>' % col_nbits).get_pins()[0].track_id.base_index
        # round up
        ntr_shift = int(max(out_tr + 2 - rdec_in_tr, 0) - row_master.std_size[0] + 0.5)
        if ntr_shift > 0:
            rdec_col_idx = self.get_num_columns(rdec_in_layer, ntr_shift)
            # round to multiples of min_space_width
            rdec_col_idx = -(-rdec_col_idx // self.min_space_width) * self.min_space_width
        else:
            rdec_col_idx = 0

        rdec_col_idx = max(rdec_col_idx, pgr_master.std_size[0])
        rdec_inst = self.add_std_instance(row_master, loc=(xoff + rdec_col_idx, row_offset), flip_lr=True)
        x_rtap = max(xoff + rdec_col_idx + row_master.std_size[0],
                     xoff + col_master.std_size[0] + buf_master.std_size[0])

        # add taps and set size
        ny_size = row_offset + row_master.std_size[1]
        nx_size = x_rtap + tap_master.std_size[0]
        tap_list = [self.add_std_instance(tap_master, loc=(0, 0), ny=ny_size // 2, spy=2),
                    self.add_std_instance(tap_master, loc=(0, 1), ny=ny_size // 2, spy=2),
                    self.add_std_instance(tap_master, loc=(x_rtap, 0), ny=ny_size // 2, spy=2),
                    self.add_std_instance(tap_master, loc=(x_rtap, 1), ny=ny_size // 2, spy=2),
                    ]
        self.set_std_size((nx_size, ny_size))
        # fill unused spaces
        self.fill_space()

        # export supplies
        for key in ('VDD', 'VSS'):
            vlist = []
            for inst in tap_list:
                vlist.extend(inst.get_all_port_pins(key))
            warr_list = self.connect_wires(vlist)
            self.add_pin(key, warr_list, show=False)

        # export code inputs
        for idx in range(col_nbits + row_nbits):
            code_warr = buf_inst.get_port('in<%d>' % idx).get_pins()[0]
            mid_tr = self.grid.find_next_track(code_warr.layer_id + 1, code_warr.lower, half_track=True)
            code_warr = self._up_two_layers(code_warr, mid_tr)
            self.add_pin('code<%d>' % idx, code_warr, show=False)

        # connect buffers to column decoder
        max_col_tid = -1
        for bit_idx in range(col_nbits):
            col_in = cdec_inst.get_port('in<%d>' % bit_idx).get_pins()[0]
            col_inb = cdec_inst.get_port('inb<%d>' % bit_idx).get_pins()[0]
            buf_out = buf_inst.get_port('out<%d>' % bit_idx).get_pins()
            buf_outb = buf_inst.get_port('outb<%d>' % bit_idx).get_pins()
            self.connect_to_tracks(buf_out, col_in.track_id, track_lower=col_in.lower)
            self.connect_to_tracks(buf_outb, col_inb.track_id, track_lower=col_inb.lower)
            max_col_tid = max(max_col_tid, col_in.track_id.base_index, col_inb.track_id.base_index)

        # connect buffers to row decoder.
        rdec_in_route_layer = rdec_in_layer + 1
        base_tr = max_col_tid + 1
        for bit_idx in range(col_nbits, col_nbits + row_nbits):
            row_in = rdec_inst.get_port('in<%d>' % (bit_idx - col_nbits)).get_pins()
            row_inb = rdec_inst.get_port('inb<%d>' % (bit_idx - col_nbits)).get_pins()
            buf_out = buf_inst.get_port('out<%d>' % bit_idx).get_pins()
            buf_outb = buf_inst.get_port('outb<%d>' % bit_idx).get_pins()
            self.connect_to_tracks(buf_outb + row_inb, TrackID(rdec_in_route_layer, base_tr))
            self.connect_to_tracks(buf_out + row_in, TrackID(rdec_in_route_layer, base_tr + 1))
            base_tr += 2

        # connect row decoder and passgates, export voltage inputs, and collect passgate ports
        col_port_list = [([], []) for _ in range(num_col)]
        vin_layer = pgr_master.get_port('in<0>').get_pins()[0].layer_id + 1
        tr_pitch = self.grid.get_track_pitch(vin_layer, unit_mode=True)
        row_h_unit = self.std_row_height_unit
        num_tr = row_h_unit // tr_pitch
        vin_bot_idx = (num_tr - num_col) / 2
        top_in_tr = vin_bot_idx + num_col - 1 + num_tr * (num_row - 1 + row_offset)
        for idx in range(num_row):
            pg_inst = pgr_inst_list[idx % 2]
            ridx = idx // 2
            pg_en = pg_inst.get_port('en_row', row=ridx).get_pins()
            pg_enb = pg_inst.get_port('enb_row', row=ridx).get_pins()
            self.connect_wires(pg_en + rdec_inst.get_port('out<%d>' % idx).get_pins())
            self.connect_wires(pg_enb + rdec_inst.get_port('outb<%d>' % idx).get_pins())
            for cidx in range(num_col):
                for cpidx, pfmt in enumerate(('en<%d>', 'enb<%d>')):
                    col_port_list[cidx][cpidx].append(pg_inst.get_port(pfmt % cidx, row=ridx).get_pins()[0])
                vin_pin = pg_inst.get_port('in<%d>' % cidx, row=ridx).get_pins()[0]
                vin_tid = TrackID(vin_layer, cidx + vin_bot_idx + num_tr * (idx + row_offset))
                vin_warr = self.connect_to_tracks(vin_pin, vin_tid, track_lower=0,
                                                  track_upper=self.array_box.right_unit, unit_mode=True)
                self.add_pin('in<%d>' % (cidx + idx * num_col), vin_warr, show=False)

        # connect column decoder and passgates
        for idx in range(num_col):
            col_out = cdec_inst.get_port('out<%d>' % idx).get_pins()[0]
            col_outb = cdec_inst.get_port('outb<%d>' % idx).get_pins()[0]
            self.connect_to_tracks(col_port_list[idx][0], col_out.track_id, track_lower=col_out.lower)
            self.connect_to_tracks(col_port_list[idx][1], col_outb.track_id, track_lower=col_outb.lower)

        # connect and export output
        out = pgr_inst_list[0].get_all_port_pins('out') + pgr_inst_list[1].get_all_port_pins('out')
        out = self.connect_wires(out, upper=self.array_box.top)
        # connect to horizontal layer
        out_tr = top_in_tr + 2
        out = self._up_two_layers(out[0], out_tr)
        self.add_pin('out', out, show=False)

    def _up_two_layers(self, warr, mid_tr):
        # connect to horizontal layer
        lay_id = warr.layer_id
        mid_coord = self.grid.track_to_coord(lay_id, warr.track_id.base_index)
        lay_id += 1
        min_len = self.grid.get_min_length(lay_id, 1)
        warr = self.connect_to_tracks(warr, TrackID(lay_id, mid_tr), track_lower=mid_coord - min_len / 2,
                                      track_upper=mid_coord + min_len / 2)
        # connect to vertical layer
        mid_coord = self.grid.track_to_coord(lay_id, warr.track_id.base_index)
        lay_id += 1
        warr_tr = self.grid.coord_to_nearest_track(lay_id, warr.middle, half_track=True)
        min_len = self.grid.get_min_length(lay_id, 1)
        return self.connect_to_tracks(warr, TrackID(lay_id, warr_tr), track_lower=mid_coord - min_len / 2,
                                      track_upper=mid_coord + min_len / 2)


class RLadderMuxArray(StdCellBase):
    """The column decoder.

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
        super(RLadderMuxArray, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_mux='number of muxes.',
            col_nbits='number of column bits.',
            row_nbits='number of row bits.',
            config_file='Standard cell configuration file.',
        )

    def draw_layout(self):
        # type: () -> None
        num_mux = self.params['num_mux']
        col_nbits = self.params['col_nbits']
        row_nbits = self.params['row_nbits']
        config_file = self.params['config_file']

        # use standard cell routing grid
        self.update_routing_grid()
        # enable standard cell boundaries
        self.set_draw_boundaries(True)

        # place muxes
        mux_params = dict(col_nbits=col_nbits, row_nbits=row_nbits, config_file=config_file)
        mux_master = self.new_template(params=mux_params, temp_cls=RLadderMux)
        mux_ncol, mux_nrow = mux_master.std_size
        mux_inst = self.add_std_instance(mux_master, nx=num_mux, spx=mux_ncol)

        # set size and draw boundaries
        top_layer = mux_master.get_port('VDD').get_pins()[0].layer_id + 2
        self.set_std_size((mux_ncol * num_mux, mux_nrow), top_layer=top_layer)
        self.draw_boundaries()

        # export inputs
        nbits_tot = col_nbits + row_nbits
        for idx in range(2 ** nbits_tot):
            name = 'in<%d>' % idx
            warrs = self.connect_wires(mux_inst.get_all_port_pins(name), lower=0.0)
            self.add_pin(name, warrs, show=False)

        # export outputs/code
        for idx in range(num_mux):
            self.reexport(mux_inst.get_port('out', col=idx), net_name='out<%d>' % idx, show=False)
            for bit_idx in range(nbits_tot):
                old_name = 'code<%d>' % bit_idx
                new_name = 'code<%d>' % (bit_idx + nbits_tot * idx)
                self.reexport(mux_inst.get_port(old_name, col=idx), net_name=new_name, show=False)

        # connect power and do fill
        vdd_list = mux_inst.get_all_port_pins('VDD')
        vss_list = mux_inst.get_all_port_pins('VSS')
        sup_layer = vdd_list[0].layer_id + 1
        vdd_list, vss_list = self.do_power_fill(sup_layer, vdd_list, vss_list, sup_width=2,
                                                fill_margin=0.2, edge_margin=0.2)
        sup_layer += 1
        vdd_list, vss_list = self.do_power_fill(sup_layer, vdd_list, vss_list, sup_width=2,
                                                fill_margin=0.2, edge_margin=0.2)

        self.add_pin('VDD', vdd_list, show=False)
        self.add_pin('VSS', vss_list, show=False)
