# -*- coding: utf-8 -*-

"""This module defines SerdesRXBase, the base class of all analog high speed link templates.
"""

import abc
from typing import Dict, Any, List, Optional, Tuple, Union

from bag.layout.routing import WireArray, RoutingGrid

from ..analog_core import AnalogBase, AnalogBaseInfo

wtype = Union[float, int]


class SerdesRXBaseInfo(AnalogBaseInfo):
    """A class that calculates informations to assist in SerdesRXBase layout calculations.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    lch : float
        the channel length of AnalogBase, in meters.
    guard_ring_nf : int
        guard ring width in number of fingers.  0 to disable.
    top_layer : Optional[int]
        the top level routing layer ID.
    end_mode : int
        right/left/top/bottom end mode flag.  This is a 4-bit integer.  If bit 0 (LSB) is 1, then
        we assume there are no blocks abutting the bottom.  If bit 1 is 1, we assume there are no
        blocks abutting the top.  bit 2 and bit 3 (MSB) corresponds to left and right, respectively.
        The default value is 15, which means we assume this AnalogBase is surrounded by empty spaces.
    min_fg_sep : int
        minimum number of separation fingers.
    """

    def __init__(self, grid, lch, guard_ring_nf, top_layer=None, end_mode=15, min_fg_sep=0):
        # type: (RoutingGrid, float, int, Optional[int], int, int) -> None
        super(SerdesRXBaseInfo, self).__init__(grid, lch, guard_ring_nf,
                                               top_layer=top_layer, end_mode=end_mode, min_fg_sep=min_fg_sep)

    def get_gm_info(self, fg_params, flip_sd=False):
        # type: (Dict[str, int]) -> Dict[str, Any]
        """Return Gm layout information dictionary.

        Parameters
        ----------
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            but :
                number of fingers of butterfly transistor.
            casc :
                number of fingers of cascode transistor.
            in :
                nummber of fingers of input transistor.
            sw :
                number of fingers of tail switch transistor.
            en :
                number of fingers of enable transistor.
            tail :
                number of fingers of tail bias transistor.
            sep :
                number of fingers used as separation between P and N side.
            min :
                minimum number of fingers for this circuit.
        flip_sd : bool
            True to flip source/drain connections.

        Returns
        -------
        info : Dict[str, Any]
            the Gm stage layout information dictionary.
        """
        fg_min = fg_params.get('min', 0)
        valid_keys = ['but', 'casc', 'in', 'sw', 'en', 'tail']
        fg_max = max((fg_params.get(key, 0) for key in valid_keys))
        fg_ref = fg_params.get('ref', 0)
        if fg_ref > 0:
            fg_max = max(fg_max, fg_ref + fg_params['tail'])
        fg_tot = fg_max * 2 + self.min_fg_sep

        if fg_tot < fg_min:
            # add dummies to get to fg_min
            # TODO: figure out when to even/not even depending on technology
            if (fg_min - fg_tot) % 4 != 0:
                # this code makes sure number of dummies is always even
                fg_min = fg_min + 4 - ((fg_min - fg_tot) % 4)
            nduml = ndumr = (fg_min - fg_tot) // 2
            fg_tot = fg_min
        else:
            nduml = ndumr = 0

        # determine output source/drain type.
        fg_but = fg_params.get('but', 0)
        if (fg_but // 2) % 2 == 1:
            out_type = 's'
        else:
            out_type = 'd'

        if flip_sd:
            out_type = 's' if out_type == 'd' else 's'

        results = dict(
            fg_tot=fg_tot,
            fg_max=fg_max,
            fg_sep=self.min_fg_sep,
            nduml=nduml,
            ndumr=ndumr,
            out_type=out_type,
        )

        # calculate column offsets.
        col_offsets = {}
        for name in ('but', 'casc', 'in', 'sw', 'en', 'tail'):
            fg = fg_params.get(name, 0)
            if fg > 0:
                col_offsets[name] = (fg_max - fg) + nduml

        results['col_offsets'] = col_offsets

        return results

    def get_diffamp_info(self, fg_params, flip_sd=False):
        # type: (Dict[str, int]) -> Dict[str, Any]
        """Return DiffAmp layout information dictionary.

        Parameters
        ----------
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            load
                number of fingers of load transistor.  Only one of load/offset can be nonzero.
            offset
                number of fingers of offset cancellation transistor.  Only one of load/offset can be nonzero.
            but
                number of fingers of butterfly transistor.
            casc
                number of fingers of cascode transistor.
            in
                nummber of fingers of input transistor.
            sw
                number of fingers of tail switch transistor.
            en
                number of fingers of enable transistor.
            tail
                number of fingers of tail bias transistor.
            sep
                number of fingers used as separation between P and N side.
            min
                minimum number of fingers for this circuit.
        flip_sd : bool
            True to flip source/drain connections.

        Returns
        -------
        info : Dict[str, Any]
            the DiffAmp stage layout information dictionary.
        """
        fg_min = fg_params.get('min', 0)
        fg_load = fg_params.get('load', 0)
        fg_offset = fg_params.get('offset', 0)
        fg_pmos = max(fg_load, fg_offset)
        fg_pmos_tot = 2 * fg_pmos + self.min_fg_sep
        # this guarantees fg_gm_tot >= fg_load_tot
        fg_min = max(fg_min, fg_pmos_tot)
        gm_fg_params = fg_params.copy()
        gm_fg_params['min'] = fg_min
        gm_info = self.get_gm_info(gm_fg_params, flip_sd=flip_sd)
        fg_gm_tot = gm_info['fg_tot']
        nduml_pmos = (fg_gm_tot - fg_pmos_tot) // 2
        ndumr_pmos = fg_gm_tot - fg_pmos_tot - nduml_pmos

        results = dict(
            fg_tot=fg_gm_tot,
            fg_sep=self.min_fg_sep,
            fg_min=fg_min,
            nduml_pmos=nduml_pmos,
            ndumr_pmos=ndumr_pmos,
            out_type=gm_info['out_type'],
        )

        return results

    def get_sampler_info(self, fg_params):
        # type: (Dict[str, int]) -> Dict[str, Any]
        """Return sampler layout information dictionary.

        Parameters
        ----------
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            sample :
                number of fingers of sample transistor.
            min :
                minimum number of fingers for this circuit.

        Returns
        -------
        info : Dict[str, Any]
            the DiffAmp stage layout information dictionary.
        """
        fg_min = fg_params.get('min', 0)
        fg_samp = fg_params['sample']
        fg_pmos_tot = 2 * fg_samp + self.min_fg_sep
        fg_tot = max(fg_min, fg_pmos_tot)
        nduml = (fg_tot - fg_pmos_tot) // 2

        results = dict(
            nduml=nduml,
            ndumr=fg_tot - fg_pmos_tot - nduml,
            fg_tot=fg_tot,
            fg_sep=self.min_fg_sep,
            fg_min=fg_min,
        )

        return results

    def get_summer_info(self, fg_load, gm_fg_list, gm_sep_list=None, flip_sd_list=None):
        # type: (int, List[Dict[str, int]], Optional[List[int]], Optional[List[bool]]) -> Dict[str, Any]
        """Return GmSummer layout information dictionary.

        Parameters
        ----------
        fg_load : int
            number of pmos load fingers (single-sided).
        gm_fg_list : List[Dict[str, int]]
            list of Gm parameter dictionaries.
        gm_sep_list : Optional[List[int]]
            list of number of separator fingers between Gm stages.
            Defaults to minimum.
        flip_sd_list : Optional[List[bool]]
            list of whether to flip source/drain connections for each Gm cell.
            Defaults to False.

        Returns
        -------
        info : Dict[str, Any]
            the GmSummer stage layout information dictionary.
        """
        if flip_sd_list is None:
            flip_sd_list = [False] * (len(gm_fg_list))
        elif len(flip_sd_list) != len(gm_fg_list):
            raise ValueError('flip_sd_list length mismatch')

        if gm_sep_list is None:
            gm_sep_list = [self.min_fg_sep] * (len(gm_fg_list) - 1)
        else:
            # error checking
            if len(gm_sep_list) != len(gm_fg_list) - 1:
                raise ValueError('gm_sep_list length mismatch')
            gm_sep_list = [max(self.min_fg_sep, val) for val in gm_sep_list]
        # append dummy value so we can use zip later.
        gm_sep_list.append(0)

        gm_fg_cum_list = []
        gm_fg_tot = 0
        fg_load_list = []
        for gm_fg_dict, flip_sd in zip(gm_fg_list, flip_sd_list):
            gm_info = self.get_gm_info(gm_fg_dict, flip_sd=flip_sd)
            cur_fg_max = (gm_info['fg_max'] - gm_info['fg_sep']) // 2
            gm_fg_tot += cur_fg_max
            gm_fg_cum_list.append(cur_fg_max)
            cur_fg_tot = (gm_info['fg_tot'] - gm_info['fg_sep']) // 2
            if fg_load > 0:
                cur_fg_load = min(fg_load, cur_fg_tot)
                fg_load_list.append(cur_fg_load)
                fg_load -= cur_fg_load
            else:
                fg_load_list.append(0)

        # get each diffamp info and calculate total number of fingers.
        fg_tot = 0
        amp_info_list = []
        gm_offsets = []
        for gm_fg_dict, fg_load, fg_sep_gm, flip_sd in zip(gm_fg_list, fg_load_list, gm_sep_list, flip_sd_list):
            gm_offsets.append(fg_tot)
            amp_fg_dict = gm_fg_dict.copy()
            amp_fg_dict['load'] = fg_load
            amp_info = self.get_diffamp_info(amp_fg_dict, flip_sd=flip_sd)
            fg_tot += amp_info['fg_tot'] + fg_sep_gm
            amp_info_list.append(amp_info)

        results = dict(
            fg_tot=fg_tot,
            gm_sep_list=gm_sep_list,
            gm_offsets=gm_offsets,
            fg_load_list=fg_load_list,
            amp_info_list=amp_info_list,
        )
        return results

    def get_summer_offset_info(self, fg_load, fg_offset, gm_fg_list, gm_sep_list=None, flip_sd_list=None):
        # type: (int, int, List[Dict[str, int]], Optional[List[int]], Optional[List[bool]]) -> Dict[str, Any]
        """Return GmSummerOffset layout information dictionary.

        Parameters
        ----------
        fg_load : int
            number of pmos load fingers (single-sided).
        fg_offset : int
            number of pmos offset cancellation fingers (single-sided).
        gm_fg_list : List[Dict[str, int]]
            list of Gm parameter dictionaries.
        gm_sep_list : Optional[List[int]]
            list of number of separator fingers between Gm stages.
            Defaults to minimum.
        flip_sd_list : Optional[List[bool]]
            list of whether to flip source/drain connections for each Gm cell.
            Defaults to False.

        Returns
        -------
        info : Dict[str, Any]
            the GmSummer stage layout information dictionary.
        """
        if flip_sd_list is None:
            flip_sd_list = [False] * (len(gm_fg_list))
        elif len(flip_sd_list) != len(gm_fg_list):
            raise ValueError('flip_sd_list length mismatch')

        if gm_sep_list is None:
            gm_sep_list = [self.min_fg_sep] * (len(gm_fg_list) - 1)
        else:
            # error checking
            if len(gm_sep_list) != len(gm_fg_list) - 1:
                raise ValueError('gm_sep_list length mismatch')
            gm_sep_list = [max(self.min_fg_sep, val) for val in gm_sep_list]
        # append dummy value so we can use zip later.
        gm_sep_list.append(0)

        # use all offset cancellation fingers first, then load fingers
        fg_load_list = []
        fg_offset_list = []
        for gm_fg_dict, flip_sd in zip(gm_fg_list, flip_sd_list):
            gm_info = self.get_gm_info(gm_fg_dict, flip_sd=flip_sd)
            cur_fg_tot = (gm_info['fg_tot'] - gm_info['fg_sep']) // 2
            if fg_offset > 0:
                cur_fg_offset = min(fg_offset, cur_fg_tot)
                fg_offset_list.append(cur_fg_offset)
                fg_offset -= cur_fg_offset
                fg_load_list.append(0)
            elif fg_load > 0:
                cur_fg_load = min(fg_load, cur_fg_tot)
                fg_load_list.append(cur_fg_load)
                fg_load -= cur_fg_load
                fg_offset_list.append(0)
            else:
                fg_load_list.append(0)
                fg_offset_list.append(0)

        # get each diffamp info and calculate total number of fingers.
        fg_tot = 0
        amp_info_list = []
        gm_offsets = []
        for gm_fg_dict, fg_load, fg_offset, fg_sep_gm, flip_sd in \
                zip(gm_fg_list, fg_load_list, fg_offset_list, gm_sep_list, flip_sd_list):
            gm_offsets.append(fg_tot)
            amp_fg_dict = gm_fg_dict.copy()
            if fg_offset == 0:
                amp_fg_dict['load'] = fg_load
            else:
                amp_fg_dict['offset'] = fg_offset
            amp_info = self.get_diffamp_info(amp_fg_dict, flip_sd=flip_sd)
            fg_tot += amp_info['fg_tot'] + fg_sep_gm
            amp_info_list.append(amp_info)

        results = dict(
            fg_tot=fg_tot,
            gm_sep_list=gm_sep_list,
            gm_offsets=gm_offsets,
            fg_load_list=fg_load_list,
            fg_offset_list=fg_offset_list,
            amp_info_list=amp_info_list,
        )
        return results


# noinspection PyAbstractClass
class SerdesRXBase(AnalogBase, metaclass=abc.ABCMeta):
    """Subclass of AmplifierBase that draws serdes circuits.

    To use this class, :py:meth:`draw_rows` must be the first function called,
    which will call :py:meth:`draw_base` for you with the right arguments.

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
    **kwargs
        optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(SerdesRXBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._nrow_idx = None
        self._serdes_info = None  # type: SerdesRXBaseInfo

    @property
    def layout_info(self):
        return self._serdes_info

    def get_nmos_row_index(self, name):
        """Returns the index of the given nmos row type."""
        return self._nrow_idx.get(name, -1)

    def _get_gm_input_track_index(self, gate_locs, track_width, diff_space):
        in_ntr = self.get_num_tracks('nch', self._nrow_idx['in'], 'g')
        inp_tr = in_ntr - (track_width + 1) / 2
        inn_tr = inp_tr - track_width - diff_space
        ptr_idx = self.get_track_index('nch', self._nrow_idx['in'], 'g', gate_locs.get('inp', inp_tr))
        ntr_idx = self.get_track_index('nch', self._nrow_idx['in'], 'g', gate_locs.get('inn', inn_tr))
        return ptr_idx, ntr_idx

    def _get_diffamp_output_track_index(self, track_width, diff_space):
        out_ntr = self.get_num_tracks('pch', 0, 'ds')
        outn_tr = out_ntr - (track_width + 1) / 2
        outp_tr = outn_tr - track_width - diff_space
        ptr_idx = self.get_track_index('pch', 0, 'ds', outp_tr)
        ntr_idx = self.get_track_index('pch', 0, 'ds', outn_tr)
        return ptr_idx, ntr_idx

    def draw_gm(self,  # type: SerdesRXBase
                col_idx,  # type: int
                fg_params,  # type: Dict[str, int]
                hm_width=1,  # type: int
                hm_cur_width=-1,  # type: int
                diff_space=1,  # type: int
                gate_locs=None,  # type: Optional[Dict[str, int]]
                flip_sd=False,  # type: bool
                tail_decap=False,  # type: bool
                ):
        # type: (...) -> Tuple[int, Dict[str, List[WireArray]]]
        """Draw a differential gm stage.

        a separator is used to separate the positive half and the negative half of the gm stage.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            but
                number of fingers of butterfly transistor.
            casc
                number of fingers of cascode transistor.
            in
                nummber of fingers of input transistor.
            sw
                number of fingers of tail switch transistor.
            en
                number of fingers of enable transistor.
            tail
                number of fingers of tail bias transistor.
            min
                minimum number of fingers for this circuit.
        hm_width : int
            width of horizontal tracks.
        hm_cur_width : int
            width of horizontal current-carrying tracks.  If negative, defaults to hm_width.
        diff_space : int
            number of tracks to reserve as space between differential wires.
        gate_locs : Optional[Dict[str, int]]
            dictionary from gate names to relative track index.  If None uses default.
        flip_sd : bool
            True to flip source/drain.  This is to help draw layout where certain configuration
            of number of fingers and source/drain directions may not be possible.
        tail_decap : bool
            True to draw mos decap for tail gate bias.

        Returns
        -------
        fg_gm : int
            width of Gm stage in number of fingers.
        port_dict : Dict[str, List[WireArray]]
            a dictionary from connection name to WireArrays.  Outputs are on vertical layer,
            and rests are on the horizontal layer above that.
        """
        fg_in = fg_params['in']
        fg_tail = fg_params['tail']
        fg_but = fg_params.get('but', 0)
        fg_casc = fg_params.get('casc', 0)
        fg_sw = fg_params.get('sw', 0)
        fg_en = fg_params.get('en', 0)
        fg_ref = fg_params.get('ref', 0)
        if fg_ref > 0:
            # enable tail decap if using reference
            tail_decap = True

        # error checking
        if fg_in <= 0 or fg_tail <= 0:
            raise ValueError('tail/input number of fingers must be positive.')
        if fg_but > 0:
            # override fg_casc
            fg_casc = 0
            if fg_but % 2 == 1:
                raise ValueError('fg_but must be even.')

        for name in ('but', 'casc', 'en', 'sw'):
            fg = fg_params.get(name, 0)
            if fg > 0 and name not in self._nrow_idx:
                raise ValueError('nmos %s row is not drawn.' % name)

        if hm_cur_width < 0:
            hm_cur_width = hm_width

        gate_locs = gate_locs or {}

        # find number of fingers per row
        gm_info = self._serdes_info.get_gm_info(fg_params, flip_sd=flip_sd)
        out_type = gm_info['out_type']
        fg_sep = gm_info['fg_sep']
        fg_gm_tot = gm_info['fg_tot']

        # figure out source/drain directions and intermediate connections
        # load always drain down.
        sd_dir = {}
        conn = {}
        track = {}
        # butterfly, cascode and input
        if fg_but > 0:
            if out_type == 's':
                # for diff mode, 'drain' direction always mean output direction, so
                # it always goes up.
                sd_dir['but'] = (0, 2)
                # output on source wire
                sd_dir['in'] = (0, 2)
                btail_type = 'd'
            else:
                sd_dir['but'] = (0, 2)
                sd_dir['in'] = (2, 0)
                btail_type = 's'
            conn['butp'] = [('butp', 's'), ('inp', btail_type)]
            conn['butn'] = [('butn', 's'), ('inn', btail_type)]
            track['butp'] = ('nch', self._nrow_idx['but'], 'ds', (hm_cur_width - 1) / 2)
            track['butn'] = ('nch', self._nrow_idx['but'], 'ds', (hm_cur_width - 1) / 2)

            itail_type = 'd' if btail_type == 's' else 's'
            conn['tail'] = [('inp', itail_type), ('inn', itail_type)]
        elif fg_casc > 0:
            # if cascode, flip input source/drain
            if out_type == 'd':
                sd_dir['casc'] = (0, 2)
                sd_dir['in'] = (2, 0)
                mid_type = 's'
            else:
                sd_dir['casc'] = (2, 0)
                sd_dir['in'] = (0, 2)
                mid_type = 'd'
            conn['midp'] = [('cascp', mid_type), ('inp', mid_type)]
            conn['midn'] = [('cascn', mid_type), ('inn', mid_type)]
            track['midp'] = ('nch', self._nrow_idx['casc'], 'ds', (hm_cur_width - 1) / 2)
            track['midn'] = ('nch', self._nrow_idx['casc'], 'ds', (hm_cur_width - 1) / 2)

            conn['tail'] = [('inp', out_type), ('inn', out_type)]
            casc_ntr = self.get_num_tracks('nch', self._nrow_idx['casc'], 'g')
            conn['bias_casc'] = [('cascp', 'g'), ('cascn', 'g')]
            track['bias_casc'] = ('nch', self._nrow_idx['casc'], 'g',
                                  gate_locs.get('bias_casc', casc_ntr - (hm_width + 1) / 2))
        else:
            if out_type == 'd':
                sd_dir['in'] = (0, 2)
                tail_type = 's'
            else:
                sd_dir['in'] = (2, 0)
                tail_type = 'd'
            conn['tail'] = [('inp', tail_type), ('inn', tail_type)]

        # switch
        if fg_sw > 0:
            inst_g = [('swp', 'g'), ('swn', 'g')]
            inst_d = [('swp', 'd'), ('swn', 'd')]
            inst_s = [('swp', 's'), ('swn', 's')]

            # switch follows input direction
            conn['sw'] = inst_g
            if sd_dir['in'][0] == 0:
                sd_dir['sw'] = (0, 1)
                conn['vddt'] = inst_d
                conn['tail'].extend(inst_s)
            else:
                sd_dir['sw'] = (1, 0)
                conn['vddt'] = inst_s
                conn['tail'].extend(inst_d)

            track['vddt'] = ('nch', self._nrow_idx['sw'], 'ds', (hm_cur_width - 1) / 2)
            track['sw'] = ('nch', self._nrow_idx['sw'], 'g', gate_locs.get('sw', (hm_width - 1) / 2))

        # enable
        if fg_en > 0:
            inst_g = [('enp', 'g'), ('enn', 'g')]
            inst_d = [('enp', 'd'), ('enn', 'd')]
            inst_s = [('enp', 's'), ('enn', 's')]

            # enable is opposite of input direction
            conn['enable'] = inst_g
            if sd_dir['in'][0] == 0:
                sd_dir['en'] = (2, 0)
                conn['tail'].extend(inst_s)
                conn['foot'] = inst_d
            else:
                sd_dir['en'] = (0, 2)
                conn['tail'].extend(inst_d)
                conn['foot'] = inst_s

            track['enable'] = ('nch', self._nrow_idx['en'], 'g', gate_locs.get('enable', (hm_width - 1) / 2))
            track['tail'] = ('nch', self._nrow_idx['en'], 'ds', (hm_cur_width - 1) / 2)

        # tail
        if 'foot' in conn:
            # enable exists.  direction opposite of enable
            key = 'foot'
            comp = 'en'
        else:
            # direction opposite of in.
            key = 'tail'
            comp = 'in'

        inst_g = [('tailp', 'g'), ('tailn', 'g')]
        inst_d = [('tailp', 'd'), ('tailn', 'd')]
        inst_s = [('tailp', 's'), ('tailn', 's')]

        conn['bias_tail'] = inst_g
        if sd_dir[comp][0] == 0:
            sd_dir['tail'] = (2, 0)
            conn[key].extend(inst_s)
            conn['VSS'] = inst_d
        else:
            sd_dir['tail'] = (0, 2)
            conn[key].extend(inst_d)
            conn['VSS'] = inst_s

        if fg_ref > 0:
            conn['VSS'].append(('ref', 's'))
            conn['bias_tail'].append(('ref', 'g'))

        track['bias_tail'] = ('nch', self._nrow_idx['tail'], 'g', gate_locs.get('bias_tail', (hm_width - 1) / 2))
        track[key] = ('nch', self._nrow_idx['tail'], 'ds', (hm_cur_width - 1) / 2)

        # create mos connections
        mos_dict = {}
        col_offsets = gm_info['col_offsets']
        for name, fg in zip(('but', 'casc', 'in', 'sw', 'en', 'tail'),
                            (fg_but, fg_casc, fg_in, fg_sw, fg_en, fg_tail)):
            if fg > 0:
                col_start = col_idx + col_offsets[name]
                sdir, ddir = sd_dir[name]
                ridx = self._nrow_idx[name]
                is_diff = (name == 'but')
                lgate_ext_mode = rgate_ext_mode = 0
                if tail_decap and name == 'tail':
                    fgr = col_offsets[name]
                    min_fg_decap = self.min_fg_decap
                    # determine whether to draw left decap, and left tail transistor gate extension mode
                    fgl = fgr - fg_ref if fg_ref > 0 else fgr
                    if fgl < 0:
                        raise ValueError('Do not have room for reference current mirror.')
                    if fgl >= min_fg_decap:
                        self.draw_mos_decap('nch', ridx, col_idx, fgl, 2)
                        lgate_ext_mode += 1
                    elif fg_ref > 0:
                        lgate_ext_mode += 1

                    # draw reference if needed
                    if fg_ref > 0:
                        mos_dict['ref'] = self.draw_mos_conn('nch', ridx, col_idx + fgl, fg_ref, 0, 0, diode_conn=True,
                                                             gate_ext_mode=3 if fgl > 0 else 2)

                    if fg_sep >= min_fg_decap:
                        self.draw_mos_decap('nch', ridx, col_start + fg, fg_sep, 3)
                        lgate_ext_mode += 2
                        rgate_ext_mode += 1

                    if fgr >= min_fg_decap:
                        self.draw_mos_decap('nch', ridx, col_start + 2 * fg + fg_sep, fgr, 1)
                        rgate_ext_mode += 2

                mos_dict['%sp' % name] = self.draw_mos_conn('nch', ridx, col_start, fg, sdir, ddir,
                                                            is_diff=is_diff, gate_ext_mode=lgate_ext_mode)
                mos_dict['%sn' % name] = self.draw_mos_conn('nch', ridx, col_start + fg + fg_sep,
                                                            fg, sdir, ddir, is_diff=is_diff,
                                                            gate_ext_mode=rgate_ext_mode)

        # get output WireArrays
        port_dict = {}
        if fg_but > 0:
            op_sd = out_type + 'p'
            on_sd = out_type + 'n'
            port_dict['outp'] = [mos_dict['butp'][op_sd], mos_dict['butn'][op_sd]]
            port_dict['outn'] = [mos_dict['butp'][on_sd], mos_dict['butn'][on_sd]]

            # draw differential butterfly connection
            but_ntr = self.get_num_tracks('nch', self._nrow_idx['but'], 'g')
            ptr_idx = self.get_track_index('nch', self._nrow_idx['but'], 'g',
                                           gate_locs.get('sgnp', but_ntr - (hm_width + 1) / 2))
            ntr_idx = self.get_track_index('nch', self._nrow_idx['but'], 'g',
                                           gate_locs.get('sgnn', but_ntr - (hm_width + 1) / 2 - hm_width - diff_space))
            p_tr, n_tr = self.connect_differential_tracks([mos_dict['butp']['gp'], mos_dict['butn']['gn']],
                                                          [mos_dict['butp']['gn'], mos_dict['butn']['gp']],
                                                          self.mos_conn_layer + 1, ptr_idx, ntr_idx,
                                                          width=hm_width)
            port_dict['sgnp'] = [p_tr, ]
            port_dict['sgnn'] = [n_tr, ]
        elif fg_casc > 0:
            port_dict['outp'] = [mos_dict['cascn'][out_type], ]
            port_dict['outn'] = [mos_dict['cascp'][out_type], ]
        else:
            port_dict['outp'] = [mos_dict['inn'][out_type], ]
            port_dict['outn'] = [mos_dict['inp'][out_type], ]

        # draw differential input connection
        inp_warr = mos_dict['inp']['g']
        inn_warr = mos_dict['inn']['g']
        ptr_idx, ntr_idx = self._get_gm_input_track_index(gate_locs, hm_width, diff_space)
        p_tr, n_tr = self.connect_differential_tracks(inp_warr, inn_warr, self.mos_conn_layer + 1, ptr_idx, ntr_idx,
                                                      width=hm_width)
        port_dict['inp'] = [p_tr, ]
        port_dict['inn'] = [n_tr, ]

        # draw intermediate connections
        for conn_name, conn_list in conn.items():
            warr_list = [mos_dict[mos][sd] for mos, sd in conn_list]
            if conn_name == 'VSS':
                self.connect_to_substrate('ptap', warr_list)
            else:
                if conn_list[0][1] == 'g':
                    tr_width = hm_width
                else:
                    tr_width = hm_cur_width

                mos_type, ridx, tr_type, tr_idx = track[conn_name]
                tr_id = self.make_track_id(mos_type, ridx, tr_type, tr_idx, width=tr_width)
                sig_warr = self.connect_to_tracks(warr_list, tr_id)
                port_dict[conn_name] = [sig_warr, ]

        return fg_gm_tot, port_dict

    def draw_pmos_sampler(self,  # type: SerdesRXBase
                          col_idx,  # type: int
                          fg_params,  # type: Dict[str, int]
                          hm_width=1,  # type: int
                          hm_cur_width=-1,  # type: int
                          diff_space=1,  # type: int
                          gate_locs=None,  # type: Optional[Dict[str, float]]
                          io_space=1,  # type: int
                          to_gm_input=False,  # type: bool
                          ):
        # type: (...) -> Tuple[int, Dict[str, List[WireArray]]]
        """Draw a differential amplifier/dynamic latch.

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            sample
                number of sampler fingers of butterfly transistor.
            min
                minimum number of fingers for this circuit.
        hm_width : int
            width of horizontal tracks.
        hm_cur_width : int
            width of horizontal current-carrying tracks.  If negative, defaults to hm_width.
        diff_space : int
            number of tracks to reserve as space between differential wires.
        gate_locs : Optional[Dict[string, int]]
            dictionary from gate names to relative track index.  If None uses default.
            True to use load dummy transistors as load decaps.
        io_space : int
            space between input and output differential tracks.
        to_gm_input : bool
            True to connect output directly to gm input tracks.

        Returns
        -------
        fg_samp : int
            width of the sampler in number of fingers.
        port_dict : Dict[str, List[WireArray]]
            a dictionary from connection name to the horizontal track associated
            with the connection.
        """
        gate_locs = gate_locs or {}

        # get layout information
        results = self._serdes_info.get_sampler_info(fg_params)
        col_idx += results['nduml']
        fg_samp = fg_params['sample']
        fg_tot = results['fg_tot']  # type: int
        fg_sep = results['fg_sep']

        # get input/output tracks
        inp_tr, inn_tr = self._get_diffamp_output_track_index(hm_cur_width, diff_space)
        if to_gm_input:
            out_width = hm_width
            outp_tr, outn_tr = self._get_gm_input_track_index(gate_locs, hm_width, diff_space)
        else:
            out_width = hm_cur_width
            outp_tr = inn_tr - io_space - hm_cur_width
            outn_tr = outp_tr - diff_space - hm_cur_width

        # draw load transistors
        sdir, ddir = 2, 0
        loadp = self.draw_mos_conn('pch', 0, col_idx, fg_samp, sdir, ddir)
        loadn = self.draw_mos_conn('pch', 0, col_idx + fg_samp + fg_sep, fg_samp, sdir, ddir)

        # connect wires
        pgbot_tr = (hm_width - 1) / 2
        tr_id = self.make_track_id('pch', 0, 'g', gate_locs.get('sample_clk', pgbot_tr), width=hm_width)
        clk_warr = self.connect_to_tracks([loadp['g'], loadn['g']], tr_id)

        hm_layer = self.mos_conn_layer + 1
        inp_warr, inn_warr = self.connect_differential_tracks(loadp['s'], loadn['s'], hm_layer,
                                                              inp_tr, inn_tr, width=hm_cur_width)
        outp_warr, outn_warr = self.connect_differential_tracks(loadp['d'], loadn['d'], hm_layer,
                                                                outp_tr, outn_tr, width=out_width)
        # type checking
        if clk_warr is None:
            raise ValueError('no clock connection made.')
        if inp_warr is None:
            raise ValueError('no inp connection made.')
        if inn_warr is None:
            raise ValueError('no inn connection made.')
        if outp_warr is None:
            raise ValueError('no outp connection made.')
        if outn_warr is None:
            raise ValueError('no outn connection made.')

        return fg_tot, {'sample_clk': [clk_warr], 'inp': [inp_warr], 'inn': [inn_warr],
                        'outp': [outp_warr], 'outn': [outn_warr]}

    def draw_diffamp(self,  # type: SerdesRXBase
                     col_idx,  # type: int
                     fg_params,  # type: Dict[str, int]
                     hm_width=1,  # type: int
                     hm_cur_width=-1,  # type: int
                     diff_space=1,  # type: int
                     gate_locs=None,  # type: Optional[Dict[str, float]]
                     sign=1,  # type: int
                     flip_sd=False,  # type: bool
                     tail_decap=False,  # type: bool
                     load_decap=False,  # type: bool
                     ):
        # type: (...) -> Tuple[int, Dict[str, List[WireArray]]]
        """Draw a differential amplifier/dynamic latch.

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_params : Dict[str, int]
            a dictionary containing number of fingers per transistor type.
            Possible entries are:

            load
                number of fingers of load transistor.  Only one of load/offset can be nonzero.
            offset
                number of fingers of offset cancellation transistor.  Only one of load/offset can be nonzero.
            but
                number of fingers of butterfly transistor.
            casc
                number of fingers of cascode transistor.
            in
                nummber of fingers of input transistor.
            sw
                number of fingers of tail switch transistor.
            en
                number of fingers of enable transistor.
            tail
                number of fingers of tail bias transistor.
            sep
                number of fingers used as separation between P and N side.
            min
                minimum number of fingers for this circuit.
        hm_width : int
            width of horizontal tracks.
        hm_cur_width : int
            width of horizontal current-carrying tracks.  If negative, defaults to hm_width.
        diff_space : int
            number of tracks to reserve as space between differential wires.
        gate_locs : Optional[Dict[string, int]]
            dictionary from gate names to relative track index.  If None uses default.
        sign : int
            the sign of the gain.  If negative, flip output connection.
        flip_sd : bool
            True to flip source/drain.  This is to help draw layout where certain configuration
            of number of fingers and source/drain directions may not be possible.
        tail_decap : bool
            True to use tail dummy transistors as tail decaps.
        load_decap : bool
            True to use load dummy transistors as load decaps.
        Returns
        -------
        fg_amp : int
            width of amplifier in number of fingers.
        port_dict : Dict[str, List[WireArray]]
            a dictionary from connection name to the horizontal track associated
            with the connection.
        """
        fg_load = fg_params.get('load', 0)
        fg_offset = fg_params.get('offset', 0)
        fg_pmos = max(fg_load, fg_offset)

        fg_but = fg_params.get('but', 0)
        if fg_pmos > fg_but > 0:
            raise ValueError('fg_pmos > fg_but > 0 case not supported yet.')

        gate_locs = gate_locs or {}

        # compute Gm stage column index.
        results = self._serdes_info.get_diffamp_info(fg_params, flip_sd=flip_sd)
        # import pprint
        # print('draw diffamp at column %d, fg_tot = %d, fg_min = %d' % (col_idx, results['fg_tot'], results['fg_min']))
        # pprint.pprint(fg_params)
        fg_min = results['fg_min']
        offset_load = results['nduml_pmos']
        out_type = results['out_type']
        fg_sep = results['fg_sep']

        # draw Gm.
        gm_params = fg_params.copy()
        gm_params['min'] = max(gm_params.get('min', fg_min), fg_min)
        fg_amp_tot, port_dict = self.draw_gm(col_idx, gm_params, hm_width=hm_width, hm_cur_width=hm_cur_width,
                                             diff_space=diff_space, gate_locs=gate_locs, flip_sd=flip_sd,
                                             tail_decap=tail_decap)

        outp_warrs = port_dict['outp']
        outn_warrs = port_dict['outn']
        if fg_pmos > 0:
            if load_decap:
                # TODO: implement this feature
                raise ValueError('do not support load decap with nonzero load yet.')
            # draw load transistors
            load_col_idx = col_idx + offset_load
            if out_type == 'd':
                sdir, ddir = 2, 0
                sup_type = 's'
            else:
                sdir, ddir = 0, 2
                sup_type = 'd'
            loadn = self.draw_mos_conn('pch', 0, load_col_idx, fg_pmos, sdir, ddir)
            loadp = self.draw_mos_conn('pch', 0, load_col_idx + fg_pmos + fg_sep, fg_pmos, sdir, ddir)

            pgbot_tr = (hm_width - 1) / 2
            if fg_offset > 0:
                # connect offset cancellation gate bias
                pgtop_tr = pgbot_tr + hm_width
                if sign < 0:
                    opg_tr = pgbot_tr
                    ong_tr = pgtop_tr
                else:
                    opg_tr = pgtop_tr
                    ong_tr = pgbot_tr
                optr_id = self.make_track_id('pch', 0, 'g', gate_locs.get('bias_offp', opg_tr), width=hm_width)
                ontr_id = self.make_track_id('pch', 0, 'g', gate_locs.get('bias_offn', ong_tr), width=hm_width)
                pwarr = self.connect_to_tracks([loadp['g']], optr_id)
                nwarr = self.connect_to_tracks([loadn['g']], ontr_id)
                if sign < 0:
                    port_dict['bias_offp'] = [nwarr, ]
                    port_dict['bias_offn'] = [pwarr, ]
                else:
                    port_dict['bias_offp'] = [pwarr, ]
                    port_dict['bias_offn'] = [nwarr, ]
            else:
                # connect load gate bias
                tr_id = self.make_track_id('pch', 0, 'g', gate_locs.get('bias_load', pgbot_tr), width=hm_width)
                warr = self.connect_to_tracks([loadp['g'], loadn['g']], tr_id)
                port_dict['bias_load'] = [warr, ]

            # connect VDD
            self.connect_to_substrate('ntap', [loadp[sup_type], loadn[sup_type]])

            # collect pmos outputs
            outp_warrs.append(loadp[out_type])
            outn_warrs.append(loadn[out_type])
        elif load_decap:
            # use all load dummies as decaps
            load_decap = self.draw_mos_decap('pch', 0, col_idx, fg_amp_tot, 0, export_gate=True)
            tr_id = self.make_track_id('pch', 0, 'g', gate_locs.get('bias_load', (hm_width - 1) / 2), width=hm_width)
            warr = self.connect_to_tracks(load_decap['g'], tr_id)
            port_dict['bias_load'] = [warr, ]

        # connect differential outputs
        ptr_idx, ntr_idx = self._get_diffamp_output_track_index(hm_cur_width, diff_space)
        if sign < 0:
            # flip positive/negative wires.
            p_tr, n_tr = self.connect_differential_tracks(outn_warrs, outp_warrs, self.mos_conn_layer + 1,
                                                          ptr_idx, ntr_idx, width=hm_cur_width)
        else:
            p_tr, n_tr = self.connect_differential_tracks(outp_warrs, outn_warrs, self.mos_conn_layer + 1,
                                                          ptr_idx, ntr_idx, width=hm_cur_width)
        port_dict['outp'] = [p_tr, ]
        port_dict['outn'] = [n_tr, ]

        return fg_amp_tot, port_dict

    def draw_gm_summer(self,  # type: SerdesRXBase
                       col_idx,  # type: int
                       fg_load,  # type: int
                       gm_fg_list,  # type: List[Dict[str, int]]
                       gm_sep_list=None,  # type: Optional[List[int]]
                       sgn_list=None,  # type: Optional[List[int]]
                       hm_width=1,  # type: int
                       hm_cur_width=-1,  # type: int
                       diff_space=1,  # type: int
                       gate_locs=None,  # type: Optional[Dict[str, float]]
                       flip_sd_list=None,  # type: Optional[List[bool]]
                       decap_list=None,  # type: Optional[List[bool]]
                       load_decap_list=None,  # type: Optional[List[bool]]
                       ):
        # type: (...) -> Tuple[int, Dict[Tuple[str, int], List[WireArray]]]
        """Draw a differential Gm summer (multiple Gm stage connected to same load).

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_load : int
            number of pmos load fingers (single-sided).
        gm_fg_list : List[Dict[str, int]]
            a list of finger dictionaries for each Gm stage, from left to right.
        gm_sep_list : Optional[List[int]]
            list of number of separator fingers between Gm stages.
            Defaults to minimum.
        sgn_list : Optional[List[int]]
            a list of 1s or -1s representing the sign of each gm stage.  If None, defautls to all 1s.
        hm_width : int
            width of horizontal tracks.
        hm_cur_width : int
            width of horizontal current-carrying tracks.  If negative, defaults to hm_width.
        diff_space : int
            number of tracks to reserve as space between differential wires.
        gate_locs : Optional[Dict[str, int]]
            dictionary from gate names to relative track index.  If None uses default.
        flip_sd_list : Optional[List[bool]]
            list of whether to flip source/drain connections for each Gm cell.
            Defaults to False.
        decap_list : Optional[List[bool]]
            list of whether to draw tail decap for each Gm cell.
            Defaults to False.
        load_decap_list : Optional[List[bool]]
            list of whether to draw load decap for each Gm cell.
            Defaults to False.
        Returns
        -------
        fg_summer : int
            width of Gm summer in number of fingers.
        port_dict : dict[(str, int), :class:`~bag.layout.routing.WireArray`]
            a dictionary from connection name/index pair to the horizontal track associated
            with the connection.
        """
        if flip_sd_list is None:
            flip_sd_list = [False] * (len(gm_fg_list))
        elif len(flip_sd_list) != len(gm_fg_list):
            raise ValueError('flip_sd_list length mismatch')
        if decap_list is None:
            decap_list = [False] * (len(gm_fg_list))
        elif len(decap_list) != len(gm_fg_list):
            raise ValueError('decap_list length mismatch')
        if load_decap_list is None:
            load_decap_list = [False] * (len(gm_fg_list))
        elif len(load_decap_list) != len(gm_fg_list):
            raise ValueError('load_decap_list length mismatch')

        if sgn_list is None:
            sgn_list = [1] * len(gm_fg_list)

        # error checking
        if fg_load <= 0:
            raise ValueError('load transistors num. fingers must be positive.')

        summer_info = self._serdes_info.get_summer_info(fg_load, gm_fg_list, gm_sep_list=gm_sep_list,
                                                        flip_sd_list=flip_sd_list)

        if len(sgn_list) != len(gm_fg_list):
            raise ValueError('sign list and number of GM stages mistach.')

        fg_load_list = summer_info['fg_load_list']
        gm_offsets = summer_info['gm_offsets']
        # print('summer col: %d' % col_idx)
        # print('summer gm offsets: %s' % repr(gm_offsets))
        # draw each Gm stage and load.
        conn_dict = {'vddt': [], 'bias_load': [], 'outp': [], 'outn': [], 'bias_load_decap': []}
        port_dict = {}
        for idx, (cur_fg_load, gm_off, gm_fg_dict, sgn, flip_sd, tail_decap, load_decap) in \
                enumerate(zip(fg_load_list, gm_offsets, gm_fg_list, sgn_list,
                              flip_sd_list, decap_list, load_decap_list)):
            cur_amp_params = gm_fg_dict.copy()
            cur_amp_params['load'] = cur_fg_load
            _, cur_ports = self.draw_diffamp(col_idx + gm_off, cur_amp_params, hm_width=hm_width,
                                             hm_cur_width=hm_cur_width, diff_space=diff_space,
                                             gate_locs=gate_locs, sign=sgn, flip_sd=flip_sd,
                                             tail_decap=tail_decap, load_decap=load_decap)

            # register port
            for name, warr_list in cur_ports.items():
                if name == 'bias_load' and cur_fg_load == 0 and load_decap:
                    # separate bias_load and bias_load_decap
                    conn_dict['bias_load_decap'].extend(warr_list)
                elif name in conn_dict:
                    conn_dict[name].extend(warr_list)
                else:
                    port_dict[(name, idx)] = warr_list

        # connect tracks together
        for name, warr_list in conn_dict.items():
            if warr_list:
                conn_list = self.connect_wires(warr_list)
                if len(conn_list) != 1:
                    # error checking
                    raise ValueError('%s wire are on different tracks.' % name)
                port_dict[(name, -1)] = conn_list

        return summer_info['fg_tot'], port_dict

    def draw_gm_summer_offset(self,  # type: SerdesRXBase
                              col_idx,  # type: int
                              fg_load,  # type: int
                              fg_offset,  # type: int
                              gm_fg_list,  # type: List[Dict[str, int]]
                              gm_sep_list=None,  # type: Optional[List[int]]
                              sgn_list=None,  # type: Optional[List[int]]
                              hm_width=1,  # type: int
                              hm_cur_width=-1,  # type: int
                              diff_space=1,  # type: int
                              gate_locs=None  # type: Optional[Dict[str, float]]
                              ):
        # type: (...) -> Tuple[int, Dict[Tuple[str, int], List[WireArray]]]
        """Draw a differential Gm summer (multiple Gm stage connected to same load).

        a separator is used to separate the positive half and the negative half of the latch.
        For tail/switch/enable devices, the g/d/s of both halves are shorted together.

        Parameters
        ----------
        col_idx : int
            the left-most transistor index.  0 is the left-most transistor.
        fg_load : int
            number of pmos load fingers (single-sided).
        fg_offset : int
            number of pmos offset cancellation fingers (single-sided).
        gm_fg_list : List[Dict[str, int]]
            a list of finger dictionaries for each Gm stage, from left to right.
        gm_sep_list : Optional[List[int]]
            list of number of separator fingers between Gm stages.
            Defaults to minimum.
        sgn_list : Optional[List[int]]
            a list of 1s or -1s representing the sign of each gm stage.  If None, defautls to all 1s.
        hm_width : int
            width of horizontal tracks.
        hm_cur_width : int
            width of horizontal current-carrying tracks.  If negative, defaults to hm_width.
        diff_space : int
            number of tracks to reserve as space between differential wires.
        gate_locs : Optional[Dict[str, int]]
            dictionary from gate names to relative track index.  If None uses default.

        Returns
        -------
        fg_summer : int
            width of Gm summer in number of fingers.
        port_dict : dict[(str, int), :class:`~bag.layout.routing.WireArray`]
            a dictionary from connection name/index pair to the horizontal track associated
            with the connection.
        """
        if sgn_list is None:
            sgn_list = [1] * len(gm_fg_list)

        # error checking
        if fg_load <= 0:
            raise ValueError('load transistors num. fingers must be positive.')

        summer_info = self._serdes_info.get_summer_offset_info(fg_load, fg_offset, gm_fg_list, gm_sep_list=gm_sep_list)

        if len(sgn_list) != len(gm_fg_list):
            raise ValueError('sign list and number of GM stages mistach.')

        fg_load_list = summer_info['fg_load_list']
        fg_offset_list = summer_info['fg_offset_list']
        gm_offsets = summer_info['gm_offsets']
        # print('summer col: %d' % col_idx)
        # print('summer gm offsets: %s' % repr(gm_offsets))
        # draw each Gm stage and load.
        conn_dict = {'vddt': [], 'bias_load': [], 'outp': [], 'outn': [], 'bias_offp': [], 'bias_offn': []}
        port_dict = {}
        for idx, (cur_fg_load, cur_fg_offset, gm_off, gm_fg_dict, sgn) in enumerate(zip(fg_load_list, fg_offset_list,
                                                                                        gm_offsets, gm_fg_list,
                                                                                        sgn_list)):
            cur_amp_params = gm_fg_dict.copy()
            cur_amp_params['load'] = cur_fg_load
            cur_amp_params['offset'] = cur_fg_offset
            _, cur_ports = self.draw_diffamp(col_idx + gm_off, cur_amp_params, hm_width=hm_width,
                                             hm_cur_width=hm_cur_width, diff_space=diff_space,
                                             gate_locs=gate_locs, sign=sgn)
            # register port
            for name, warr_list in cur_ports.items():
                if name in conn_dict:
                    conn_dict[name].extend(warr_list)
                else:
                    port_dict[(name, idx)] = warr_list

        # connect tracks together
        for name, warr_list in conn_dict.items():
            if warr_list:
                conn_list = self.connect_wires(warr_list)
                if len(conn_list) != 1:
                    # error checking
                    raise ValueError('%s wire are on different tracks.' % name)
                port_dict[(name, -1)] = conn_list

        return summer_info['fg_tot'], port_dict

    def draw_rows(self, lch, fg_tot, ptap_w, ntap_w, w_dict, th_dict, **kwargs):
        # type: (float, int, wtype, wtype, Dict[str, wtype], Dict[str, str], **Any) -> None
        """Draw the transistors and substrate rows.

        Parameters
        ----------
        lch : float
            the transistor channel length, in meters
        fg_tot : int
            total number of fingers for each row.
        ptap_w : Union[float, int]
            pwell substrate contact width.
        ntap_w : Union[float, int]
            nwell substrate contact width.
        w_dict : Dict[str, Union[float, int]]
            dictionary from transistor type to row width.  Possible entries are:

            load
                width of load transistor.
            casc
                width of butterfly/cascode transistor.
            in
                width of input transistor.
            sw
                width of tail switch transistor.
            en
                width of enable transistor.
            tail
                width of tail bias transistor.

        th_dict : Dict[str, str]
            dictionary from transistor type to threshold flavor.  Possible entries are:

            load
                threshold of load transistor.
            casc
                threshold of butterfly/cascode transistor.
            in
                threshold of input transistor.
            sw
                threshold of tail switch transistor.
            en
                threshold of enable transistor.
            tail
                threshold of tail bias transistor.

        **kwargs
            any addtional parameters for AnalogBase's draw_base() method.
        """
        # error checking
        w_tail = w_dict['tail']
        w_in = w_dict['in']
        w_load = w_dict['load']
        th_load = th_dict['load']
        if w_tail <= 0 or w_in <= 0 or w_load <= 0:
            raise ValueError('tail/input/load transistors width must be positive.')

        self._serdes_info = SerdesRXBaseInfo(self.grid, lch, kwargs.get('guard_ring_nf', 0),
                                             min_fg_sep=kwargs.get('min_fg_sep', 0))

        # figure out row indices for each nmos row type,
        # and build nw_list/nth_list
        self._nrow_idx = {}
        nw_list = []
        nth_list = []
        cur_idx = 0
        for name in ('tail', 'en', 'sw', 'in', 'casc'):
            width = w_dict.get(name, 0)
            if width > 0:
                thres = th_dict[name]
                self._nrow_idx[name] = cur_idx
                nw_list.append(width)
                nth_list.append(thres)
                cur_idx += 1

        if 'casc' in self._nrow_idx:
            # butterfly switch and cascode share the same row.
            self._nrow_idx['but'] = self._nrow_idx['casc']

        # draw base
        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, [w_load], [th_load], **kwargs)
