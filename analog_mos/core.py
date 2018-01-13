# -*- coding: utf-8 -*-

"""This module defines abstract analog mosfet template classes.
"""

from typing import TYPE_CHECKING, Dict, Any, Union, Tuple, List, Optional

import abc
from itertools import chain
from collections import namedtuple

from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateBase

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig

PlaceInfo = namedtuple('PlaceInfo', ['tot_width', 'core_width', 'edge_margins', 'edge_widths', 'arr_box_x', ])


class MOSTech(object, metaclass=abc.ABCMeta):
    """An abstract class for drawing transistor related layout.
    
    This class defines various methods use to draw layouts used by AnalogBase.

    Parameters
    ----------
    config : Dict[str, Any]
        the technology configuration dictionary.
    tech_info : TechInfo
        the TechInfo object.
    mos_entry_name : str
        name of the entry that contains technology parameters for transistors in
        the given configuration dictionary.
    """

    def __init__(self, config, tech_info, mos_entry_name='mos'):
        # type: (Dict[str, Any], TechInfoConfig, str) -> None
        self.config = config
        self.mos_config = self.config[mos_entry_name]
        self.res = self.config['resolution']
        self.tech_info = tech_info
        self._lch_unit = None
        self._mos_constants = None

    @abc.abstractmethod
    def get_edge_info(self, lch_unit, guard_ring_nf, is_end, **kwargs):
        # type: (int, int, bool, **kwargs) -> Dict[str, Any]
        """Returns a dictionary containing transistor edge layout information.

        The returned dictionary must have two entries

        edge_num_fg : int
            edge block width in number of fingers.
        edge_margin : int
            the left/right margin needed around edge blocks, in resolution units.

        Parameters
        ----------
        lch_unit : int
            the channel length, in resolution units.
        guard_ring_nf : int
            guard ring width in number of fingers.
        is_end : bool
            True if there are no blocks abutting the left edge.
        **kwargs :
            Optional edge layout parameters.  Currently supported parameters are:

            is_sub_ring : bool
                True if this is a substrate ring edge.
            dnw_mode : str
                the deep N-well mode string, empty to disable.

        Returns
        -------
        edge_info : Dict[str, Any]
            edge layout information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_mos_info(self, lch_unit, w, mos_type, threshold, fg, **kwargs):
        # type: (int, int, str, str, int, **kwargs) -> Dict[str, Any]
        """Returns the transistor information dictionary.

        The returned dictionary must have the following entries:

        layout_info
            the layout information dictionary.
        ext_top_info
            a tuple of values used to compute extension layout above the transistor.
        ext_bot_info
            a tuple of values used to compute extension layout below the transistor.
        sd_yc
            the Y coordinate of the center of source/drain junction.
        g_conn_y
            a Tuple of bottom/top Y coordinates of gate wire on mos_conn_layer.  Used to determine
            gate tracks location.
        d_conn_y
            a Tuple of bottom/top Y coordinates of drain/source wire on mos_conn_layer.  Used to
            determine drain/source tracks location.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor w in number of fins/resolution units.
        mos_type : str
            the transistor type.  Either 'pch' or 'nch'.
        threshold : str
            the transistor threshold flavor.
        fg : int
            number of fingers in this transistor row.
        **kwargs :
            optional transistor row options.

        Returns
        -------
        mos_info : Dict[str, Any]
            the transistor information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_valid_extension_widths(self, lch_unit, top_ext_info, bot_ext_info):
        # type: (int, Any, Any) -> List[int]
        """Returns a list of valid extension widths in mos_pitch units.

        the list should be sorted in increasing order, and any extension widths greater than
        or equal to the last element should be valid.  For example, if the returned list
        is [0, 2, 5], then extension widths 0, 2, 5, 6, 7, ... are valid, while extension
        widths 1, 3, 4 are not valid.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        top_ext_info : Any
            layout information about the top block.
        bot_ext_info : Any
            layout information about the bottom block.
        """
        return [0]

    @abc.abstractmethod
    def get_ext_info(self, lch_unit, w, fg, top_ext_info, bot_ext_info):
        # type: (int, int, int, Any, Any) -> Dict[str, Any]
        """Returns the extension layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the extension width in number of fins/resolution units.
        fg : int
            total number of fingers.
        top_ext_info : Any
            layout information about the top block.
        bot_ext_info : Any
            layout information about the bottom block.

        Returns
        -------
        ext_info : Dict[str, Any]
            the extension information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_sub_ring_ext_info(self, sub_type, height, fg, end_ext_info, **kwargs):
        # type: (str, int, int, Any, **kwargs) -> Dict[str, Any]
        """Returns the SubstrateRing extension layout information dictionary.

        Parameters
        ----------
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        height : int
            the extension width in resolution units.
        fg : int
            total number of fingers.
        end_ext_info : Any
            layout extension information about the substrate inner end row.
        **kwargs :
            additional arguments.

        Returns
        -------
        ext_info : Dict[str, Any]
            the substrate ring extension information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_substrate_info(self, lch_unit, w, sub_type, threshold, fg, blk_pitch=1, **kwargs):
        # type: (int, int, str, str, int, int, int, **kwargs) -> Dict[str, Any]
        """Returns the substrate layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        w : int
            the transistor width in number of fins/resolution units.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        blk_pitch : int
            substrate height quantization pitch.  Defaults to 1 (no quantization).
        **kwargs :
            additional arguments.

        Returns
        -------
        sub_info : Dict[str, Any]
            the substrate information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_analog_end_info(self, lch_unit, sub_type, threshold, fg, is_end, blk_pitch, **kwargs):
        # type: (int, str, str, int, bool, int, **kwargs) -> Dict[str, Any]
        """Returns the AnalogBase end row layout information dictionary.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        is_end : bool
            True if there are no block abutting the bottom.
        blk_pitch : int
            substrate height quantization pitch.
        **kwargs :
            optional keyword arguments.

        Returns
        -------
        end_info : Dict[str, Any]
            the end row information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_sub_ring_end_info(self, sub_type, threshold, fg, end_ext_info, **kwargs):
        # type: (str, str, int, Any, **kwargs) -> Dict[str, Any]
        """Returns the SubstrateRing inner end row layout information dictionary.

        Parameters
        ----------
        sub_type : str
            the substrate type.  Either 'ptap' or 'ntap'.
        threshold : str
            the substrate threshold type.
        fg : int
            total number of fingers.
        end_ext_info : Any
            layout extension information about the substrate row.
        **kwargs :
            optional layout parameters.

        Returns
        -------
        end_info : Dict[str, Any]
            the end row information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_outer_edge_info(self, guard_ring_nf, layout_info, is_end, adj_blk_info):
        # type: (int, Dict[str, Any], bool, Optional[Any]) -> Dict[str, Any]
        """Returns the outer edge layout information dictionary.

        Parameters
        ----------
        guard_ring_nf : int
            guard ring width in number of fingers.  0 if there is no guard ring.
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.
        is_end : bool
            True if there are no blocks abutting the left edge.
        adj_blk_info : Optional[Any]
            data structure storing layout information of adjacent block.
            If None, will use default settings.

        Returns
        -------
        outer_edge_info : Dict[str, Any]
            the outer edge layout information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_gr_sub_info(self, guard_ring_nf, layout_info):
        # type: (int, Dict[str, Any]) -> Dict[str, Any]
        """Returns the guard ring substrate layout information dictionary.

        Parameters
        ----------
        guard_ring_nf : int
            guard ring width in number of fingers.  0 if there is no guard ring.
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.

        Returns
        -------
        gr_sub_info : Dict[str, Any]
            the guard ring substrate layout information dictionary.
        """
        return {}

    @abc.abstractmethod
    def get_gr_sep_info(self, layout_info, adj_blk_info):
        # type: (Dict[str, Any], Any) -> Dict[str, Any]
        """Returns the guard ring separator layout information dictionary.

        Parameters
        ----------
        layout_info : Dict[str, Any]
            layout information dictionary of the center block.
        adj_blk_info : Optional[Any]
            data structure storing layout information of adjacent block.
            If None, will use default settings.

        Returns
        -------
        gr_sub_info : Dict[str, Any]
            the guard ring separator layout information dictionary.
        """
        return {}

    @abc.abstractmethod
    def draw_mos(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        """Draw transistor layout structure in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        layout_info : Dict[str, Any]
            layout information dictionary for the transistor/substrate/extension/edge blocks.
        """
        pass

    @abc.abstractmethod
    def draw_substrate_connection(self, template, layout_info, port_tracks, dum_tracks, dummy_only,
                                  is_laygo, is_guardring):
        # type: (TemplateBase, Dict[str, Any], List[int], List[int], bool, bool, bool) -> bool
        """Draw substrate connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        layout_info : Dict[str, Any]
            the substrate layout information dictionary.
        port_tracks : List[int]
            list of port track indices that must be drawn on transistor connection layer.
        dum_tracks : List[int]
            list of dummy port track indices that must be drawn on dummy connection layer.
        dummy_only : bool
            True to only draw connections up to dummy connection layer.
        is_laygo : bool
            True if this is Laygo substrate connection.
        is_guardring : bool
            True if this is guardring substrate connection.

        Returns
        -------
        has_connection : bool
            True if connection is drawn.
        """
        pass

    @abc.abstractmethod
    def draw_mos_connection(self, template, mos_info, sdir, ddir, gate_pref_loc, gate_ext_mode,
                            min_ds_cap, is_diff, diode_conn, options):
        # type: (TemplateBase, Dict[str, Any], int, int, str, int, bool, bool, bool, Dict[str, Any]) -> None
        """Draw transistor connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        sdir : int
            source direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        ddir : int
            drain direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        gate_pref_loc : str
            preferred gate location flag, either 's' or 'd'.  This is only used if both source
            and drain did not go down.
        gate_ext_mode : int
            gate extension flag.  This is a 2 bit integer, the LSB is 1 if we should connect
            gate to the left adjacent block on lower metal layers, the MSB is 1 if we should
            connect gate to the right adjacent block on lower metal layers.
        min_ds_cap : bool
            True to minimize drain-to-source parasitic capacitance.
        is_diff : bool
            True if this is a differential pair connection.
        diode_conn : bool
            True to short gate and drain together.
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    @abc.abstractmethod
    def draw_dum_connection(self, template, mos_info, edge_mode, gate_tracks, options):
        # type: (TemplateBase, Dict[str, Any], int, List[Union[float, int]], Dict[str, Any]) -> None
        """Draw dummy connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        edge_mode : int
            the dummy edge mode flag.  This is a 2-bit integer, the LSB is 1 if there are no
            transistors on the left side of the dummy, the MSB is 1 if there are no transistors
            on the right side of the dummy.
        gate_tracks : List[Union[float, int]]
            list of dummy connection track indices.
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    @abc.abstractmethod
    def draw_decap_connection(self, template, mos_info, sdir, ddir, gate_ext_mode, export_gate, options):
        # type: (TemplateBase, Dict[str, Any], int, int, int, bool, Dict[str, Any]) -> None
        """Draw decoupling cap connection layout in the given template.

        Parameters
        ----------
        template : TemplateBase
            the TemplateBase object to draw layout in.
        mos_info : Dict[str, Any]
            the transistor layout information dictionary.
        sdir : int
            source direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        ddir : int
            drain direction flag.  0 to go down, 1 to stay in middle, 2 to go up.
        gate_ext_mode : int
            gate extension flag.  This is a 2 bit integer, the LSB is 1 if we should connect
            gate to the left adjacent block on lower metal layers, the MSB is 1 if we should
            connect gate to the right adjacent block on lower metal layers.
        export_gate : bool
            True to draw gate connections up to transistor connection layer.
        options : Dict[str, Any]
            a dictionary of transistor row options.
        """
        pass

    def get_conn_drc_info(self, lch_unit, wire_type, is_laygo=False):
        # type: (int, str, bool) -> Dict[int, Dict[str, Any]]
        """Get DRC information about gate/drain/source wire on each layer.

        Parameters
        ----------
        lch_unit : int
            channel length, in resolution units.
        wire_type : str
            the wire type, either 'g' or 'd'.
        is_laygo : bool
            True if this is for laygo connections.

        Returns
        -------
        drc_info : Dict[int, Dict[str, Any]]
            a dictionary from layer ID to DRC information dictionary.
        """
        if is_laygo:
            wire_type = 'laygo_' + wire_type
        mos_constants = self.get_mos_tech_constants(lch_unit)
        bot_layer = mos_constants[wire_type + '_bot_layer']
        widths = mos_constants[wire_type + '_conn_w']
        via_info = mos_constants[wire_type + '_via']
        dirs = mos_constants[wire_type + '_conn_dir']

        conn_info = {}
        layers = range(bot_layer, bot_layer + len(dirs))
        for lay, w, direction, vdim, vble, vtle in \
            zip(layers, widths, dirs, via_info['dim'],
                via_info['bot_enc_le'], via_info['top_enc_le']):
            vdim_le = vdim[0] if direction == 'x' else vdim[1]
            top_ext = vdim_le // 2 + vtle
            lay_name = self.tech_info.get_layer_name(lay)
            if isinstance(lay_name, tuple):
                lay_name = lay_name[0]
            lay_type = self.tech_info.get_layer_type(lay_name)
            min_len = self.tech_info.get_min_length_unit(lay_type, w)
            min_len = max(2 * top_ext, -(-min_len // 2) * 2)
            sp_le = self.tech_info.get_min_line_end_space_unit(lay_type, w)
            conn_info[lay] = dict(
                w=w,
                direction=direction,
                min_len=min_len,
                sp_le=sp_le,
                top_ext=top_ext,
                bot_ext=0,
            )

            if lay > bot_layer:
                vdim_le = vdim[0] if dirs[lay - bot_layer - 1] == 'x' else vdim[1]
                bot_ext = vdim_le // 2 + vble
                conn_info[lay - 1]['bot_ext'] = bot_ext
                conn_info[lay - 1]['min_len'] = max(conn_info[lay - 1]['min_len'], 2 * bot_ext)

        return conn_info

    def get_mos_layers(self, mos_type, threshold):
        # type: (str, str) -> List[Tuple[str, str]]
        """Returns a list of implant/well/threshold layers.

        Parameters
        ----------
        mos_type : str
            the transistor type.  Valid values are 'pch', 'nch', 'ntap', and 'ptap'.
        threshold : str
            the threshold flavor.

        Returns
        -------
        layer_list : List[Tuple[str, str]]
            a list of implant/well/threshold layer names.
        """
        imp_layers_info = self.mos_config['imp_layers'][mos_type]
        thres_layers_info = self.mos_config['thres_layers'][mos_type][threshold]

        return list(chain(imp_layers_info.keys(), thres_layers_info.keys()))

    def get_mos_tech_constants(self, lch_unit):
        # type: (int) -> Dict[str, Any]
        """Returns a dictionary of technology constants given transistor channel length.
        
        Must have the following entries:
        
        sd_pitch : the source/drain pitch of the transistor in resolution units.
        mos_conn_w : the transistor connection track width in resolution units.
        dum_conn_w : the dummy connection track width in resolution units.
        num_sd_per_track : number of transistor source/drain junction per vertical track.
        
        
        Parameters
        ----------
        lch_unit : int
            the channel length, in resolution units.
        
        Returns
        -------
        tech_dict : Dict[str, Any]
            a technology constants dictionary.
        """
        if lch_unit != self._lch_unit:
            # handle general channel-length dependent constants
            ans = self.mos_config.copy()
            for key, data in ans.items():
                if isinstance(data, dict) and 'lch' in data and 'val' in data:
                    for lch, val in zip(data['lch'], data['val']):
                        if lch_unit <= lch:
                            ans[key] = val
                            break

            # handle mos/dum_conn_w
            mos_layer = self.get_mos_conn_layer()
            dum_layer = self.get_dum_conn_layer()
            d_conn_w = ans['d_conn_w']
            d_bot_layer = ans['d_bot_layer']
            ans['mos_conn_w'] = d_conn_w[dum_layer - d_bot_layer]
            ans['dum_conn_w'] = d_conn_w[mos_layer - d_bot_layer]
            # handle laygo_conn_w
            if 'laygo_d_conn_w' in ans:
                d_conn_w = ans['laygo_d_conn_w']
                d_bot_layer = ans['laygo_d_bot_layer']
                laygo_layer = self.get_dig_conn_layer()
                ans['laygo_conn_w'] = d_conn_w[laygo_layer - d_bot_layer]

            # handle sd_pitch
            offset, scale = ans['sd_pitch_constants']
            ans['sd_pitch'] = offset + int(round(scale * lch_unit))
            self._mos_constants = ans
            self._lch_unit = lch_unit

        return self._mos_constants

    def get_analog_unit_fg(self):
        # type: () -> int
        """Returns the number of fingers in an AnalogBase row unit.

        Returns
        -------
        num_fg : int
            number of fingers in an AnalogBase row unit.
        """
        return self.mos_config['analog_unit_fg']

    def draw_zero_extension(self):
        # type: () -> bool
        """Returns True if we should draw 0 width extension.

        Returns
        -------
        draw_ext : bool
            True to draw 0 width extension.
        """
        return self.mos_config['draw_zero_extension']

    def floating_dummy(self):
        # type: () -> bool
        """Returns True if floating dummies are allowed.

        Returns
        -------
        float_dummy : bool
            True if floating dummies are allowed.
        """
        return self.mos_config['floating_dummy']

    def abut_analog_mos(self):
        # type: () -> bool
        """Returns True if abutting transistors in AnalogBase is allowed.

        Returns
        -------
        abut_analog_mos : bool
            True if abutting transistors in AnalogBase is allowed.
        """
        return self.mos_config['abut_analog_mos']

    def get_substrate_ring_lch(self):
        # type: () -> float
        """Returns substrate channel length used in substrate rings.

        Returns
        -------
        lch : float
            Substrate channel length, in meters.
        """
        return self.mos_config['sub_ring_lch']

    def get_dum_conn_pitch(self):
        # type: () -> int
        """Returns the minimum track pitch of dummy connections in number of tracks.

        Some technology can only draw dummy connections on every other track.  In that case,
        this method should return 2.

        Returns
        -------
        dum_conn_pitch : pitch between adjacent dummy connection.
        """
        return self.mos_config['dum_conn_pitch']

    def get_dum_conn_layer(self):
        # type: () -> int
        """Returns the dummy connection layer ID.  Must be vertical.
        
        Returns
        -------
        dum_layer : int
            the dummy connection layer ID.
        """
        return self.mos_config['dum_layer']

    def get_mos_conn_layer(self):
        # type: () -> int
        """Returns the transistor connection layer ID.  Must be vertical.
        
        Returns
        -------
        mos_layer : int
            the transistor connection layer ID.
        """
        return self.mos_config['ana_conn_layer']

    def get_dig_conn_layer(self):
        # type: () -> int
        """Returns the digital connection layer ID.  Must be vertical.

        Returns
        -------
        dig_layer : int
            the transistor connection layer ID.
        """
        return self.mos_config['dig_conn_layer']

    def get_dig_top_layer(self):
        # type: () -> int
        """Returns the digital top layer ID.  Must be vertical.

        Returns
        -------
        dig_layer : int
            the transistor connection layer ID.
        """
        return self.mos_config['dig_top_layer']

    def get_min_fg_decap(self, lch_unit):
        # type: (int) -> int
        """Returns the minimum number of fingers for decap connections.
        
        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.
        
        Returns
        -------
        num_fg : int
            minimum number of decap fingers.
        """
        return self.get_mos_tech_constants(lch_unit)['min_fg_decap']

    def get_min_fg_sep(self, lch_unit):
        # type: (int) -> int
        """Returns the minimum number of dummy fingers needed between active transistors in AnalogBase.

        Parameters
        ----------
        lch_unit : int
            the channel length in resolution units.

        Returns
        -------
        num_fg : int
            minimum number of dummy fingers.
        """
        return self.get_mos_tech_constants(lch_unit)['min_fg_sep']

    def get_tech_constant(self, name):
        # type: (str) -> Any
        """Returns the value of the given technology constant.
        
        Parameters
        ----------
        name : str
            constant name.
            
        Returns
        -------
        val : Any
            constant value.
        """
        return self.config[name]

    def get_mos_pitch(self, unit_mode=False):
        # type: (bool) -> Union[float, int]
        """Returns the transistor vertical placement quantization pitch.
        
        This is usually the fin pitch for finfet process.
        
        Parameters
        ----------
        unit_mode : bool
            True to return the pitch in resolution units.
            
        Returns
        -------
        mos_pitch : Union[float, int]
            the transistor vertical placement quantization pitch.
        """
        ans = self.mos_config['mos_pitch']
        if unit_mode:
            return ans
        return ans * self.res

    def get_dum_conn_track_info(self, lch_unit):
        # type: (int) -> Tuple[int, int]
        """Returns dummy connection layer space and width.
        
        Parameters
        ----------
        lch_unit : int
            channel length in resolution units.
            
        Returns
        -------
        dum_sp : int
            space between dummy tracks in resolution units.
        dum_w : int
            width of dummy tracks in resolution units.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        dum_conn_w = mos_constants['dum_conn_w']
        num_sd_per_track = mos_constants['num_sd_per_track']
        return sd_pitch * num_sd_per_track - dum_conn_w, dum_conn_w

    def get_mos_conn_track_info(self, lch_unit):
        # type: (int) -> Tuple[int, int]
        """Returns transistor connection layer space and width.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units.

        Returns
        -------
        tr_sp : int
            space between transistor connection tracks in resolution units.
        tr_w : int
            width of transistor connection tracks in resolution units.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        sd_pitch = mos_constants['sd_pitch']
        mos_conn_w = mos_constants['mos_conn_w']
        num_sd_per_track = mos_constants['num_sd_per_track']

        return sd_pitch * num_sd_per_track - mos_conn_w, mos_conn_w

    def get_num_fingers_per_sd(self, lch_unit):
        # type: (int) -> int
        """Returns the number of transistor source/drain junction per vertical track.
        
        Parameters
        ----------
        lch_unit : int
            channel length in resolution units
            
        Returns
        -------
        num_sd_per_track : number of source/drain junction per vertical track.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        return mos_constants['num_sd_per_track']

    def get_laygo_num_fingers_per_sd(self, lch_unit):
        # type: (int) -> int
        """Returns the number of transistor source/drain junction per vertical track.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units

        Returns
        -------
        num_sd_per_track : number of source/drain junction per vertical track.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        return mos_constants['laygo_num_sd_per_track']

    def get_sd_pitch(self, lch_unit):
        # type: (int) -> int
        """Returns the source/drain pitch in resolution units.

        Parameters
        ----------
        lch_unit : int
            channel length in resolution units

        Returns
        -------
        sd_pitch : the source/drain pitch in resolution units.
        """
        mos_constants = self.get_mos_tech_constants(lch_unit)
        return mos_constants['sd_pitch']

    def get_placement_info(self, grid, top_layer, fg_tot, lch_unit, guard_ring_nf,
                           left_end, right_end, is_laygo, **kwargs):
        # type: (RoutingGrid, int, int, int, int, bool, bool, bool, **kwargs) -> PlaceInfo
        """Compute edge block placement information.

        Parameters
        ----------
        grid: RoutingGrid
            the RoutingGrid object.
        top_layer : int
            the top routing layer ID.  Used to determine width quantization.
        fg_tot : int&
            total number of fingers.
        lch_unit : int
            channel length in resolution units
        guard_ring_nf : int
            guard ring width in number of fingers.
        left_end : bool
            True if there are no blocks abutting the left edge.
        right_end : bool
            True if there are no blocks abutting the right edge.
        is_laygo : bool
            True if we're getting placement information for LaygoBase.
        kwargs :
            Optional edge layout parameters.  Currently supports:

            is_sub_ring : bool
                True if this is a substrate ring edge.
            dnw_mode : str
                the deep N-well mode string, empty to disable.

        Returns
        -------
        place_info : PlaceInfo
            the placement information named tuple.
        """
        sd_pitch = self.get_sd_pitch(lch_unit)
        edgel_info = self.get_edge_info(lch_unit, guard_ring_nf, left_end, **kwargs)
        edgel_num_fg = edgel_info['edge_num_fg']
        edgel_margin = edgel_info['edge_margin']
        edger_info = self.get_edge_info(lch_unit, guard_ring_nf, right_end, **kwargs)
        edger_num_fg = edger_info['edge_num_fg']
        edger_margin = edger_info['edge_margin']

        if is_laygo:
            top_vm_layer = self.get_dig_top_layer()
        else:
            top_vm_layer = self.get_mos_conn_layer()

        prim_layer = top_vm_layer + 1
        core_width = (edgel_num_fg + edger_num_fg + fg_tot) * sd_pitch
        if top_layer <= prim_layer:
            # use private layer for horizontal quantization so that
            # array box can be defined.
            blk_w = grid.get_block_size(top_vm_layer, unit_mode=True)[0]
            edgel_margin = -(-edgel_margin // blk_w) * blk_w
            edger_margin = -(-edger_margin // blk_w) * blk_w
            arr_dxl = edgel_margin
            arr_dxr = edger_margin
        else:
            blk_w = grid.get_block_size(top_layer, unit_mode=True)[0]
            arr_dxl = 0
            arr_dxr = 0

        tot_width = core_width + edgel_margin + edger_margin
        tot_width = -(-tot_width // blk_w) * blk_w
        space = tot_width - core_width
        edge_margin_tot = edgel_margin + edger_margin
        if edge_margin_tot == 0:
            left_margin = space // 2
        else:
            left_margin = space * edgel_margin // edge_margin_tot
        right_margin = space - left_margin

        return PlaceInfo(tot_width=tot_width,
                         core_width=core_width,
                         edge_margins=(left_margin, right_margin),
                         edge_widths=(sd_pitch * edgel_num_fg, sd_pitch * edger_num_fg),
                         arr_box_x=(arr_dxl, tot_width - arr_dxr))
