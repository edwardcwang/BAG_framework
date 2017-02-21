# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""This module defines amplifier templates used in high speed links.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from .base import SerdesRXBase


class DynamicLatchChain(SerdesRXBase):
    """A chain of dynamic latches.

    Parameters
    ----------
    grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.  Must have the following entries:
    used_names : set[str]
        a set of already used cell names.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        SerdesRXBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @staticmethod
    def _rename_port(pname, idx, nstage):
        """Rename the given port."""
        if nstage == 1:
            return pname
        else:
            return '%s<%d>' % (pname, idx)

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self, lch, ptap_w, ntap_w, w_dict, th_dict, fg_dict,
                            nstage, fg_sep, nduml, ndumr, global_gnd_layer, global_gnd_name,
                            diff_space, cur_track_width, show_pins, **kwargs):
        if nstage <= 0:
            raise ValueError('nstage = %d must be greater than 0' % nstage)

        # calculate total number of fingers.
        fg_sep = max(fg_sep, self.get_min_fg_sep(self.grid.tech_info))
        fg_latch = max(fg_dict.values()) * 2 + fg_sep
        fg_tot = nstage * fg_latch + (nstage - 1) * fg_sep + nduml + ndumr

        # figure out number of tracks
        kwargs['pg_tracks'] = [1]
        kwargs['pds_tracks'] = [2 + diff_space]
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in' or (row_name == 'casc' and fg_dict.get('but', 0) > 0):
                    ng_tracks.append(2 + diff_space)
                else:
                    ng_tracks.append(1)
                nds_tracks.append(cur_track_width + kwargs['gds_space'])
        kwargs['ng_tracks'] = ng_tracks
        kwargs['nds_tracks'] = nds_tracks

        # draw rows with width/threshold parameters.
        for key, val in w_dict.items():
            kwargs['w_' + key] = val
        for key, val in th_dict.items():
            kwargs['th_' + key] = val
        del kwargs['rename_dict']
        self.draw_rows(lch, fg_tot, ptap_w, ntap_w, **kwargs)

        port_list = []

        da_kwargs = {'fg_' + key: val for key, val in fg_dict.items()}
        for idx in range(nstage):
            col_idx = (fg_latch + fg_sep) * idx + nduml
            _, pdict = self.draw_diffamp(col_idx, cur_track_width=cur_track_width, **da_kwargs)
            for pname, port_warr in pdict.items():
                pname = self.get_pin_name(pname)
                if pname:
                    pin_name = self._rename_port(pname, idx, nstage)
                    port_list.append((pin_name, port_warr))

        ptap_wire_arrs, ntap_wire_arrs = self.fill_dummy()
        # export supplies
        port_list.extend((('VSS', warr) for warr in ptap_wire_arrs))
        port_list.extend((('VDD', warr) for warr in ntap_wire_arrs))

        for pname, warr in port_list:
            self.add_pin(pname, warr, show=show_pins)

        # add global ground
        if global_gnd_layer is not None:
            _, global_gnd_box = next(ptap_wire_arrs[0].wire_iter(self.grid))
            self.add_pin_primitive(global_gnd_name, global_gnd_layer, global_gnd_box)

    @classmethod
    def get_default_param_values(cls):
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
            gds_space=1,
            diff_space=1,
            nstage=1,
            fg_sep=0,
            nduml=4,
            ndumr=4,
            cur_track_width=1,
            show_pins=True,
            rename_dict={},
            guard_ring_nf=0,
            global_gnd_layer=None,
            global_gnd_name='gnd!',
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
            fg_dict='NMOS/PMOS number of fingers dictionary.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            nstage='number of dynamic latch stages.',
            fg_sep='number of separator finger between stages',
            nduml='number of dummy fingers on the left.',
            ndumr='number of dummy fingers on the right.',
            cur_track_width='width of the current-carrying horizontal track wire in number of tracks.',
            show_pins='True to create pin labels.',
            rename_dict='port renaming dictionary',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            global_gnd_layer='layer of the global ground pin.  None to disable drawing global ground.',
            global_gnd_name='name of global ground pin.',
        )
