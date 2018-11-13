# -*- coding: utf-8 -*-

"""This module defines amplifier templates used in high speed links.
"""

from typing import Dict, Any, Set, Union, List, Optional

from bag.layout.template import TemplateDB

from .base import SerdesRXBase, SerdesRXBaseInfo


class DiffAmp(SerdesRXBase):
    """A single diff amp.

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
        super(DiffAmp, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._num_fg = -1

    @property
    def num_fingers(self):
        # type: () -> int
        return self._num_fg

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
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            th_dict={},
            top_layer=None,
            nduml=4,
            ndumr=4,
            min_fg_sep=0,
            gds_space=0,
            diff_space=1,
            hm_width=1,
            hm_cur_width=-1,
            show_pins=True,
            guard_ring_nf=0,
            tail_decap=False,
            flip_sd=False,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            top_layer='the top routing layer.',
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            fg_dict='NMOS/PMOS number of fingers dictionary.',
            nduml='Number of left dummy fingers.',
            ndumr='Number of right dummy fingers.',
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            tail_decap='True to draw tail decap transistors.',
            flip_sd='True to flip source drain.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: DiffAmp
                            top_layer,  # type: Optional[int]
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w_dict,  # type: Dict[str, Union[float, int]]
                            th_dict,  # type: Dict[str, str]
                            fg_dict,  # type: Dict[str, int]
                            nduml,  # type: int
                            ndumr,  # type: int
                            min_fg_sep,  # type: int
                            gds_space,  # type: int
                            diff_space,  # type: int
                            hm_width,  # type: int
                            hm_cur_width,  # type: int
                            show_pins,  # type: bool
                            guard_ring_nf,  # type: int
                            tail_decap,  # type: bool
                            flip_sd,  # type: bool
                            **kwargs
                            ):
        # type: (...) -> None

        serdes_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        diffamp_info = serdes_info.get_diffamp_info(fg_dict, flip_sd=flip_sd)
        fg_tot = diffamp_info['fg_tot'] + nduml + ndumr
        self._num_fg = fg_tot

        hm_layer = serdes_info.mconn_port_layer + 1
        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        hm_cur_space = self.grid.get_num_space_tracks(hm_layer, hm_cur_width)
        diff_space = max(hm_cur_space, diff_space)
        gds_space = max(hm_cur_space, gds_space)

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            top_layer=top_layer,
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            w_dict=w_dict,
            th_dict=th_dict,
            gds_space=gds_space,
            pg_tracks=[hm_width],
            pds_tracks=[2 * hm_cur_width + diff_space],
            min_fg_sep=min_fg_sep,
            guard_ring_nf=guard_ring_nf,
        )
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)

        gate_locs = {'inp': (hm_width - 1) / 2 + hm_width + diff_space,
                     'inn': (hm_width - 1) / 2}
        _, amp_ports = self.draw_diffamp(nduml, fg_dict, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                         diff_space=diff_space, gate_locs=gate_locs, tail_decap=tail_decap,
                                         flip_sd=flip_sd)

        vdd_warrs = None
        hide_pins = {'midp', 'midn', 'tail', 'foot'}
        for pname, warrs in amp_ports.items():
            if pname == 'vddt':
                vdd_warrs = self.connect_wires(warrs, unit_mode=True)
            else:
                self.add_pin(pname, warrs, show=show_pins and pname not in hide_pins)

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy(vdd_warrs=vdd_warrs, unit_mode=True, sup_margin=1)
        self.add_pin('VSS', ptap_wire_arrs)
        self.add_pin('VDD', ntap_wire_arrs)
        if vdd_warrs is not None:
            self.add_pin('VDD', vdd_warrs)


class IntegSummer(SerdesRXBase):
    """A single diff amp.

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
        super(IntegSummer, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._num_fg = -1

    @property
    def num_fingers(self):
        # type: () -> int
        return self._num_fg

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
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            th_dict={},
            nduml=4,
            ndumr=4,
            min_fg_sep=0,
            gds_space=1,
            diff_space=1,
            hm_width=1,
            hm_cur_width=-1,
            show_pins=True,
            guard_ring_nf=0,
            flip_sd_list=None,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            gm_fg_list='number of fingers for each gm stages.',
            sgn_list='gm sign list.',
            fg_load='number of load fingers.',
            nduml='Number of left dummy fingers.',
            ndumr='Number of right dummy fingers.',
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            flip_sd_list='List of whether to flip source/drain connections.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: DiffAmp
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w_dict,  # type: Dict[str, Union[float, int]]
                            th_dict,  # type: Dict[str, str]
                            gm_fg_list,  # type: List[Dict[str, int]]
                            sgn_list,  # type: List[int]
                            fg_load,  # type: int
                            nduml,  # type: int
                            ndumr,  # type: int
                            min_fg_sep,  # type: int
                            gds_space,  # type: int
                            diff_space,  # type: int
                            hm_width,  # type: int
                            hm_cur_width,  # type: int
                            show_pins,  # type: bool
                            guard_ring_nf,  # type: int
                            flip_sd_list  # type: Optional[List[bool]]
                            ):
        # type: (...) -> None

        serdes_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        # calculate total number of fingers.
        info = serdes_info.get_summer_info(fg_load, gm_fg_list)
        fg_tot = info['fg_tot'] + nduml + ndumr
        self._num_fg = fg_tot

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            w_dict=w_dict,
            th_dict=th_dict,
            gds_space=gds_space,
            pg_tracks=[hm_width],
            pds_tracks=[2 * hm_cur_width + diff_space],
            min_fg_sep=min_fg_sep,
            guard_ring_nf=guard_ring_nf,
        )
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)
        self.set_size_from_array_box(self.mos_conn_layer + 1)
        sup_lower, sup_upper = self.array_box.left_unit, self.array_box.right_unit

        gate_locs = {'inp': (hm_width - 1) / 2 + hm_width + diff_space,
                     'inn': (hm_width - 1) / 2}
        _, summer_ports = self.draw_gm_summer(nduml, fg_load, gm_fg_list, sgn_list=sgn_list,
                                              hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              flip_sd_list=flip_sd_list)

        vdd_warrs = None
        hide_pins = {'midp', 'midn', 'tail', 'foot'}
        sw_list = []
        for (pname, pidx), warrs in summer_ports.items():
            if pname in hide_pins:
                continue

            if pname == 'bias_casc':
                self.add_pin('bias_ffe', warrs, show=show_pins)
            elif pname == 'sw':
                sw_list.extend(warrs)
            elif pname == 'vddt':
                vdd_warrs = self.connect_wires(warrs, lower=sup_lower, upper=sup_upper, unit_mode=True)
            else:
                if pidx < 0:
                    self.add_pin(pname, warrs, show=show_pins)
                else:
                    self.add_pin('%s<%d>' % (pname, pidx), warrs, show=show_pins)

        warr = self.connect_wires(sw_list)
        self.add_pin('bias_switch', warr, show=show_pins)

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy(lower=sup_lower, upper=sup_upper, vdd_warrs=vdd_warrs,
                                                         unit_mode=True, sup_margin=1)
        self.add_pin('VSS', ptap_wire_arrs)
        self.add_pin('VDD', ntap_wire_arrs)
        if vdd_warrs is not None:
            self.add_pin('VDD', vdd_warrs)


class Tap1Summer(SerdesRXBase):
    """A single diff amp.

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
        super(Tap1Summer, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._num_fg = -1

    @property
    def num_fingers(self):
        # type: () -> int
        return self._num_fg

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
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            th_dict={},
            nduml=4,
            ndumr=4,
            min_fg_sep=0,
            gds_space=1,
            diff_space=1,
            hm_width=1,
            hm_cur_width=-1,
            show_pins=True,
            guard_ring_nf=0,
            flip_sd_list=None,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            gm_fg_list='number of fingers for each gm stages.',
            sgn_list='gm sign list.',
            fg_load='number of load fingers.',
            nduml='Number of left dummy fingers.',
            ndumr='Number of right dummy fingers.',
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            show_pins='True to create pin labels.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            flip_sd_list='List of whether to flip source/drain connections.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: DiffAmp
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w_dict,  # type: Dict[str, Union[float, int]]
                            th_dict,  # type: Dict[str, str]
                            gm_fg_list,  # type: List[Dict[str, int]]
                            sgn_list,  # type: List[int]
                            fg_load,  # type: int
                            nduml,  # type: int
                            ndumr,  # type: int
                            min_fg_sep,  # type: int
                            gds_space,  # type: int
                            diff_space,  # type: int
                            hm_width,  # type: int
                            hm_cur_width,  # type: int
                            show_pins,  # type: bool
                            guard_ring_nf,  # type: int
                            flip_sd_list  # type: Optional[List[bool]]
                            ):
        # type: (...) -> None

        serdes_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        # calculate total number of fingers.
        info = serdes_info.get_summer_info(fg_load, gm_fg_list)
        fg_tot = info['fg_tot'] + nduml + ndumr
        self._num_fg = fg_tot

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            w_dict=w_dict,
            th_dict=th_dict,
            gds_space=gds_space,
            pg_tracks=[hm_width],
            pds_tracks=[2 * hm_cur_width + diff_space],
            min_fg_sep=min_fg_sep,
            guard_ring_nf=guard_ring_nf,
        )
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)
        self.set_size_from_array_box(self.mos_conn_layer + 1)
        sup_lower, sup_upper = self.array_box.left_unit, self.array_box.right_unit

        gate_locs = {'inp': (hm_width - 1) / 2 + hm_width + diff_space,
                     'inn': (hm_width - 1) / 2}
        _, summer_ports = self.draw_gm_summer(nduml, fg_load, gm_fg_list, sgn_list=sgn_list,
                                              hm_width=hm_width, hm_cur_width=hm_cur_width,
                                              diff_space=diff_space, gate_locs=gate_locs,
                                              flip_sd_list=flip_sd_list)

        vdd_warrs = None
        hide_pins = {'midp', 'midn', 'tail', 'foot'}
        sw_list = []
        for (pname, pidx), warrs in summer_ports.items():
            if pname in hide_pins:
                continue

            if pname == 'sw':
                sw_list.extend(warrs)
            elif pname == 'vddt':
                vdd_warrs = self.connect_wires(warrs, lower=sup_lower, upper=sup_upper, unit_mode=True)
            else:
                if pidx < 0:
                    self.add_pin(pname, warrs, show=show_pins)
                else:
                    self.add_pin('%s<%d>' % (pname, pidx), warrs, show=show_pins)

        warr = self.connect_wires(sw_list)
        self.add_pin('bias_switch', warr, show=show_pins)

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy(lower=sup_lower, upper=sup_upper, vdd_warrs=vdd_warrs,
                                                         unit_mode=True, sup_margin=1)
        self.add_pin('VSS', ptap_wire_arrs)
        self.add_pin('VDD', ntap_wire_arrs)
        if vdd_warrs is not None:
            self.add_pin('VDD', vdd_warrs)
