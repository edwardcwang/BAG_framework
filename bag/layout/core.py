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

"""This module defines the base template class.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from future.utils import with_metaclass

import abc
from typing import List, Iterator, Tuple
from itertools import chain

import bag
import bag.io
from .util import BBox
from .objects import Rect, Via, ViaInfo, Instance, InstanceInfo, PinInfo
from .objects import Path, Blockage, Boundary
from bag.util.search import BinaryIterator

# try to import cybagoa module
try:
    import cybagoa
except ImportError:
    cybagoa = None


class TechInfo(with_metaclass(abc.ABCMeta, object)):
    """A base class that create vias.

    This class provides the API for making vias.  Each process should subclass this class and
    implement the make_via method.

    Parameters
    ----------
    res : float
        the grid resolution of this technology.
    via_tech : string
        the via technology library name.  This is usually the PDK library name.
    process_params : dict[str, any]
        process specific parameters.

    Attributes
    ----------
    tech_params : dict[str, any]
        technology specific parameters.
    """

    def __init__(self, res, layout_unit, via_tech, process_params):
        self._resolution = res
        self._layout_unit = layout_unit
        self._via_tech = via_tech
        self.tech_params = process_params

    @classmethod
    @abc.abstractmethod
    def get_via_drc_info(cls, vname, vtype, mtype, mw_unit, is_bot):
        """Return data structures used to identify VIA DRC rules.

        Parameters
        ----------
        vname : string
            the via type name.
        vtype : string
            the via type, square/hrect/vrect/etc.
        mtype : string
            name of the metal layer via is connecting.  Can be either top or bottom.
        mw_unit : int
            width of the metal, in resolution units.
        is_bot : bool
            True if the given metal is the bottom metal.

        Returns
        -------
        sp : Tuple[int, int]
            horizontal/vertical space between adjacent vias, in resolution units.
        sp3 : Tuple[int, int] or None
            horizontal/vertical space between adjacent vias if the via has 3 or more neighbors.
            None if no constraint.
        dim : Tuple[int, int]
            the via width/height in resolution units.
        enc : List[Tuple[int, int]]
            a list of valid horizontal/vertical enclosure of the via on the given metal
            layer, in resolution units.
        arr_enc : List[Tuple[int, int]] or None
            a list of valid horizontal/vertical enclosure of the via on the given metal
            layer if this is a "via array", in layout units.
            None if no constraint.
        arr_test : callable or None
            a function that accepts two inputs, the number of via rows and number of via
            columns, and returns True if those numbers describe a "via array".
            None if no constraint.
        """
        return (0, 0), (0, 0), (0, 0), [(0, 0)], None, None

    @property
    def via_tech_name(self):
        """Returns the via technology library name."""
        return self._via_tech

    @property
    def resolution(self):
        """Returns the grid resolution."""
        return self._resolution

    @property
    def layout_unit(self):
        """Returns the layout unit length, in meters."""
        return self._layout_unit

    @abc.abstractmethod
    def get_min_space(self, layer_type, width):
        """Returns the minimum spacing needed around a wire on the given layer with the given width.

        Parameters
        ----------
        layer_type : str
            the wiring layer type.
        width : float
            the width of the wire, in layout units.

        Returns
        -------
        sp : float
            the minimum spacing needed.
        """
        return 0.0

    @abc.abstractmethod
    def get_min_length(self, layer_type, width):
        # type: (str, float) -> float
        """Returns the minimum length of a wire on the given layer with the given width.

        Parameters
        ----------
        layer_type : str
            the wiring layer type.
        width : float
            the width of the wire, in layout units.

        Returns
        -------
        min_length : float
            the minimum length.
        """
        return 0.0

    @abc.abstractmethod
    def get_layer_id(self, layer_name):
        """Return the layer id for the given layer name.

        Parameters
        ----------
        layer_name : string
            the layer name.

        Returns
        -------
        layer_id : int
            the layer ID.
        """
        return 0

    @abc.abstractmethod
    def get_layer_name(self, layer_id):
        """Return the layer name(s) for the given routing grid layer ID.

        Parameters
        ----------
        layer_id : int
            the routing grid layer ID.

        Returns
        -------
        name : string or Tuple[string]
            name of the layer.  Returns a tuple of names if this is a double
            patterning layer.
        """
        return ''

    @abc.abstractmethod
    def get_layer_type(self, layer_name):
        """Returns the metal type of the given wiring layer.

        Parameters
        ----------
        layer_name : str
            the wiring layer name.

        Returns
        -------
        metal_type : string
            the metal layer type.
        """
        return ''

    @abc.abstractmethod
    def get_via_name(self, bot_layer_id):
        """Returns the via type name of the given via.

        Parameters
        ----------
        bot_layer_id : int
            the via bottom layer ID

        Returns
        -------
        name : string
            the via type name.
        """
        return ''

    @abc.abstractmethod
    def get_metal_em_specs(self, layer_name, w, l=-1, vertical=False, **kwargs):
        """Returns a tuple of EM current/resistance specs of the given wire.

        Parameters
        ----------
        layer_name : str
            the metal layer name.
        w : float
            the width of the metal in layout units (dimension perpendicular to current flow).
        l : float
            the length of the metal in layout units (dimension parallel to current flow).
            If negative, disable length enhancement.
        vertical : bool
            True to compute vertical current.
        **kwargs :
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_via_em_specs(self, via_name, bm_layer, tm_layer, via_type='square',
                         bm_dim=(-1, -1), tm_dim=(-1, -1), array=False, **kwargs):
        """Returns a tuple of EM current/resistance specs of the given via.

        Parameters
        ----------
        via_name : string
            the via type name.
        bm_layer : str
            the bottom layer name.
        tm_layer : string
            the top layer name.
        via_type : string
            the via type, square/vrect/hrect/etc.
        bm_dim : Tuple[float, float]
            bottom layer metal width/length in layout units.  If negative,
            disable length/width enhancement.
        tm_dim : Tuple[float, float]
            top layer metal width/length in layout units.  If negative,
            disable length/width enhancement.
        array : bool
            True if this via is in a via array.
        **kwargs :
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current per via, in Amperes.
        iac_rms : float
            maximum AC RMS current per via, in Amperes.
        iac_peak : float
            maximum AC peak current per via, in Amperes.
        """
        return 0.0, float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_res_rsquare(self, res_type):
        """Returns R-square for the given resistor type.

        This is used to do some approximate resistor dimension calculation.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        rsquare : float
            resistance in Ohms per unit square of the given resistor type.
        """
        return 0.0

    @abc.abstractmethod
    def get_res_width_bounds(self, res_type):
        """Returns the maximum and minimum resistor width for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        wmin : float
            minimum resistor width, in layout units.
        wmax : float
            maximum resistor width, in layout units.
        """
        return 0.0, 0.0

    @abc.abstractmethod
    def get_res_length_bounds(self, res_type):
        """Returns the maximum and minimum resistor length for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        lmin : float
            minimum resistor length, in layout units.
        lmax : float
            maximum resistor length, in layout units.
        """
        return 0.0, 0.0

    @abc.abstractmethod
    def get_res_min_nsquare(self, res_type):
        """Returns the minimum allowable number of squares for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        nsq_min : flaot
            minimum number of squares needed.
        """
        return 1.0

    @abc.abstractmethod
    def get_res_em_specs(self, res_type, w, l=-1, **kwargs):
        # type: (str, float, float, **kwargs) -> Tuple[float, float, float]
        """Returns a tuple of EM current/resistance specs of the given resistor.

        Parameters
        ----------
        res_type : string
            the resistor type string.
        w : float
            the width of the metal in layout units (dimension perpendicular to current flow).
        l : float
            the length of the metal in layout units (dimension parallel to current flow).
            If negative, disable length enhancement.
        **kwargs :
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        return 0.0, float('inf'), float('inf')

    @abc.abstractmethod
    def get_res_info(self, res_type, w, l, **kwargs):
        """Returns a dictionary containing EM information of the given resistor.

        Parameters
        ----------
        res_type : string or (string, string)
            the resistor type.
        w : float
            the resistor width in layout units (dimension perpendicular to current flow).
        l : float
            the resistor length in layout units (dimension parallel to current flow).
        **kwargs :
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : dict[string, any]
            A dictionary of wire information.  Should have the following:

            resistance : float
                The resistance, in Ohms.
            idc : float
                The maximum allowable DC current, in Amperes.
            iac_rms : float
                The maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The maximum allowable AC peak current, in Amperes.
        """
        return None

    def get_best_via_array(self, vname, bmtype, tmtype, bot_dir, w, h):
        """Maximize the number of vias in the given area.

        Parameters
        ----------
        vname : string
            the via type name.
        bmtype : string
            the bottom metal type name.
        tmtype : string
            the top metal type name.
        bot_dir : string
            the bottom wire direction.  Either 'x' or 'y'.
        w : float
            width of the via array bounding box, in layout units.
        h : float
            height of the via array bounding box, in layout units.

        Returns
        -------
        best_nxy : Tuple[int, int]
            optimal number of vias per row/column.
        best_mdim_list : list[Tuple[int, int]]
            a list of bottom/top layer width/height, in resolution units.
        vtype : str
            the via type to draw, square/hrect/vrect/etc.
        vdim : Tuple[int, int]
            the via width/height, in resolution units.
        via_space : Tuple[int, int]
            the via horizontal/vertical spacing, in resolution units.
        via_arr_dim : Tuple[int, int]
            the via array width/height, in resolution units.
        """
        res = self._resolution
        w = int(round(w / res))
        h = int(round(h / res))

        if bot_dir == 'x':
            top_dir = 'y'
            bw = h
            tw = w
        else:
            top_dir = 'x'
            bw = w
            tw = h

        best_num = None
        best_nxy = [-1, -1]
        best_mdim_list = None
        best_type = None
        best_vdim = None
        best_sp = None
        best_adim = None
        for vtype, weight in [('square', 1), ('vrect', 2), ('hrect', 2)]:
            try:
                # get space and enclosure rules for top and bottom layer
                sp, sp3, dim, encb, arr_encb, arr_testb = self.get_via_drc_info(vname, vtype, bmtype, bw, True)
                _, _, _, enct, arr_enct, arr_testt = self.get_via_drc_info(vname, vtype, tmtype, tw, False)
                # print _get_via_params(vname, vtype, bmtype, bw)
                # print _get_via_params(vname, vtype, tmtype, tw)
            except ValueError:
                continue

            # compute maximum possible nx and ny
            spx_min, spy_min = sp
            if sp3 is not None:
                spx_min = min(sp[0], sp3[0])
                spy_min = min(sp[1], sp3[1])
            nx_max = (w + spx_min) // (dim[0] + spx_min)
            ny_max = (h + spy_min) // (dim[1] + spy_min)

            # print nx_max, ny_max, dim, w, h, spx_min, spy_min

            # generate list of possible nx/ny configuration
            nxy_list = [(a * b, a, b) for a in range(1, nx_max + 1) for b in range(1, ny_max + 1)]
            nxy_list = sorted(nxy_list, reverse=True)

            # find best nx/ny configuration
            opt_nxy = None
            opt_mdim_list = None
            opt_adim = None
            opt_sp = None
            for num, nx, ny in nxy_list:
                # check if we need to use sp3
                if sp3 is not None and nx > 1 and ny > 1 and max(nx, ny) > 2:
                    spx, spy = sp3
                else:
                    spx, spy = sp

                # get via array bounding box
                w_arr = nx * (spx + dim[0]) - spx
                h_arr = ny * (spy + dim[1]) - spy

                mdim_list = [None, None]
                # check if at least one enclosure rule is satisfied for both top and bottom layer
                for idx, (mdir, tot_enc_list, arr_enc, arr_test) in enumerate([(bot_dir, encb, arr_encb, arr_testb),
                                                                               (top_dir, enct, arr_enct, arr_testt)]):
                    # check if array enclosure rule applies
                    if arr_test is not None and arr_test(ny, nx):
                        tot_enc_list = tot_enc_list + arr_enc

                    if mdir == 'y':
                        enc_idx = 0
                        enc_dim = w_arr
                        ext_dim = h_arr
                        dim_lim = w
                        max_ext_dim = h
                    else:
                        enc_idx = 1
                        enc_dim = h_arr
                        ext_dim = w_arr
                        dim_lim = h
                        max_ext_dim = w

                    min_ext_dim = None
                    for enc in tot_enc_list:
                        if enc[enc_idx] * 2 + enc_dim <= dim_lim:
                            # enclosure rule passed.  Find minimum other dimension
                            cur_ext_dim = ext_dim + 2 * enc[1 - enc_idx]
                            if min_ext_dim is None or min_ext_dim > cur_ext_dim:
                                min_ext_dim = cur_ext_dim

                    if min_ext_dim is None:
                        # all enclosure rule failed.  Exit.
                        break
                    else:
                        # record metal dimension.
                        min_ext_dim = max(min_ext_dim, max_ext_dim)
                        mdim_list[idx] = [min_ext_dim, min_ext_dim]
                        mdim_list[idx][enc_idx] = dim_lim

                if mdim_list[0] is not None and mdim_list[1] is not None:
                    # passed
                    opt_mdim_list = mdim_list
                    opt_nxy = (nx, ny)
                    opt_adim = (w_arr, h_arr)
                    opt_sp = (spx, spy)
                    break

            if opt_nxy is not None:
                opt_num = weight * opt_nxy[0] * opt_nxy[1]
                if (best_num is None or opt_num > best_num or
                        (opt_num == best_num and self._via_better(opt_mdim_list, best_mdim_list))):
                    best_num = opt_num
                    best_nxy = opt_nxy
                    best_mdim_list = opt_mdim_list
                    best_type = vtype
                    best_vdim = dim
                    best_sp = opt_sp
                    best_adim = opt_adim

        if best_num is None:
            return None
        return best_nxy, best_mdim_list, best_type, best_vdim, best_sp, best_adim

    def _via_better(self, mdim_list1, mdim_list2):
        """Returns true if the via in mdim_list1 has smaller area compared with via in mdim_list2"""
        res = self._resolution
        better = False
        for mdim1, mdim2 in zip(mdim_list1, mdim_list2):
            area1 = int(round(mdim1[0] / res)) * int(round(mdim1[1] / res))
            area2 = int(round(mdim2[0] / res)) * int(round(mdim2[1] / res))
            if area1 < area2:
                better = True
            elif area1 > area2:
                return False
        return better

    # noinspection PyMethodMayBeStatic
    def get_via_id(self, bot_layer, top_layer):
        """Returns the via ID string given bottom and top layer name.

        Defaults to "<bot_layer>_<top_layer>"

        Parameters
        ----------
        bot_layer : string
            the bottom layer name.
        top_layer : string
            the top layer name.

        Returns
        -------
        via_id : string
            the via ID string.
        """
        return '%s_%s' % (top_layer, bot_layer)

    def get_via_info(self, bbox, bot_layer, top_layer, bot_dir, bot_len=-1, top_len=-1, **kwargs):
        """Create a via on the routing grid given the bounding box.

        Parameters
        ----------
        bbox : bag.layout.util.BBox
            the bounding box of the via.
        bot_layer : string or (string, string)
            the bottom layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        top_layer : string or (string, string)
            the top layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        bot_dir : string
            the bottom layer extension direction.  Either 'x' or 'y'
        bot_len : float
            length of bottom wire connected to this Via, in layout units.
            Used for length enhancement EM calculation.
        top_len : float
            length of top wire connected to this Via, in layout units.
            Used for length enhancement EM calculation.
        **kwargs :
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : dict[string, any]
            A dictionary of via information, or None if no solution.  Should have the following:

            resistance : float
                The total via array resistance, in Ohms.
            idc : float
                The total via array maximum allowable DC current, in Amperes.
            iac_rms : float
                The total via array maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The total via array maximum allowable AC peak current, in Amperes.
            params : dict[str, any]
                A dictionary of via parameters.
            top_box : bag.layout.util.BBox
                the top via layer bounding box, including extensions.
            bot_box : bag.layout.util.BBox
                the bottom via layer bounding box, including extensions.

        """
        # remove purpose
        if isinstance(bot_layer, tuple):
            bot_layer = bot_layer[0]
        if isinstance(top_layer, tuple):
            top_layer = top_layer[0]
        bot_layer = bag.io.fix_string(bot_layer)
        top_layer = bag.io.fix_string(top_layer)

        bot_id = self.get_layer_id(bot_layer)
        bmtype = self.get_layer_type(bot_layer)
        tmtype = self.get_layer_type(top_layer)
        vname = self.get_via_name(bot_id)

        via_result = self.get_best_via_array(vname, bmtype, tmtype, bot_dir, bbox.width, bbox.height)
        if via_result is None:
            # no solution found
            return None

        (nx, ny), mdim_list, vtype, vdim, (spx, spy), (warr_norm, harr_norm) = via_result

        res = self.resolution
        xc_norm = bbox.xc_unit
        yc_norm = bbox.yc_unit

        wbot_norm = mdim_list[0][0]
        hbot_norm = mdim_list[0][1]
        wtop_norm = mdim_list[1][0]
        htop_norm = mdim_list[1][1]

        # OpenAccess Via can't handle even + odd enclosure, so we truncate.
        enc1_x = (wbot_norm - warr_norm) // 2 * res
        enc1_y = (hbot_norm - harr_norm) // 2 * res
        enc2_x = (wtop_norm - warr_norm) // 2 * res
        enc2_y = (htop_norm - harr_norm) // 2 * res

        # compute EM rule dimensions
        if bot_dir == 'x':
            bw, tw = hbot_norm * res, wtop_norm * res
        else:
            bw, tw = wbot_norm * res, htop_norm * res

        bot_xl_norm = xc_norm - wbot_norm // 2
        bot_yb_norm = yc_norm - hbot_norm // 2
        top_xl_norm = xc_norm - wtop_norm // 2
        top_yb_norm = yc_norm - htop_norm // 2

        bot_box = BBox(bot_xl_norm, bot_yb_norm, bot_xl_norm + wbot_norm, bot_yb_norm + hbot_norm, res, unit_mode=True)
        top_box = BBox(top_xl_norm, top_yb_norm, top_xl_norm + wtop_norm, top_yb_norm + htop_norm, res, unit_mode=True)

        idc, irms, ipeak = self.get_via_em_specs(vname, bot_layer, top_layer, via_type=vtype,
                                                 bm_dim=(bw, bot_len), tm_dim=(tw, top_len),
                                                 array=nx > 1 or ny > 1, **kwargs)

        params = {'id': self.get_via_id(bot_layer, top_layer),
                  'loc': (xc_norm * res, yc_norm * res),
                  'orient': 'R0',
                  'num_rows': ny,
                  'num_cols': nx,
                  'sp_rows': spy * res,
                  'sp_cols': spx * res,
                  # increase left/bottom enclusion if off-center.
                  'enc1': [enc1_x, enc1_x, enc1_y, enc1_y],
                  'enc2': [enc2_x, enc2_x, enc2_y, enc2_y],
                  'cut_width': vdim[0] * res,
                  'cut_height': vdim[1] * res,
                  }

        ntot = nx * ny
        return dict(
            resistance=0.0,
            idc=idc * ntot,
            iac_rms=irms * ntot,
            iac_peak=ipeak * ntot,
            params=params,
            top_box=top_box,
            bot_box=bot_box,
        )

    def design_resistor(self, res_type, res_targ, idc=0.0, iac_rms=0.0,
                        iac_peak=0.0, num_even=True, **kwargs):
        """Finds the optimal resistor dimension that meets the given specs.

        Assumes resistor length does not effect EM specs.

        Parameters
        ----------
        res_type : string
            the resistor type.
        res_targ : float
            target resistor, in Ohms.
        idc : float
            maximum DC current spec, in Amperes.
        iac_rms : float
            maximum AC RMS current spec, in Amperes.
        iac_peak : float
            maximum AC peak current spec, in Amperes.
        num_even : int
            True to return even number of resistors.
        **kwargs :
            optional EM spec calculation parameters.

        Returns
        -------
        num_par : int
            number of resistors needed in parallel.
        num_ser : int
            number of resistors needed in series.
        w : float
            width of a unit resistor, in meters.
        l : float
            length of a unit resistor, in meters.
        """
        resolution = self.resolution
        rsq = self.get_res_rsquare(res_type)
        wmin, wmax = self.get_res_width_bounds(res_type)
        lmin, lmax = self.get_res_length_bounds(res_type)
        min_nsq = self.get_res_min_nsquare(res_type)

        # step 1: estimate number of resistors in parallel from EM specs
        res_idc, res_irms, res_ipeak = self.get_res_em_specs(res_type, wmax, **kwargs)
        num_par = 1
        if 0.0 < res_idc < idc:
            num_par = max(num_par, -(-idc // res_idc))
        if 0.0 < res_irms < iac_rms:
            num_par = max(num_par, -(-iac_rms // res_irms))
        if 0.0 < res_ipeak < iac_peak:
            num_par = max(num_par, -(-iac_rms // res_ipeak))
        num_par = int(round(num_par))
        if num_even and num_par % 2 == 1:
            num_par += 1
            num_par_inc = 2
        else:
            num_par_inc = 1

        # step 2: find width and length of unit resistor
        # note: due to possibility of violating lmin, we may need to increase
        # num_par, so do it in a while loop
        done = False
        wopt = lopt = None
        while not done:
            cur_res_targ = res_targ * num_par
            cur_idc = idc / num_par
            cur_iac_rms = iac_rms / num_par
            cur_iac_peak = iac_peak / num_par

            # step 3: binary search to find width of resistor.
            wmin_unit = int(round(wmin / resolution))
            wmax_unit = int(round(wmax / resolution))
            bin_iter = BinaryIterator(wmin_unit, wmax_unit + 1)
            while bin_iter.has_next():
                wcur = bin_iter.get_next() * resolution
                res_idc, res_irms, res_ipeak = self.get_res_em_specs(res_type, wcur, **kwargs)
                if res_idc > cur_idc and res_irms > cur_iac_rms and res_ipeak > cur_iac_peak:
                    bin_iter.save()
                    bin_iter.down()
                else:
                    bin_iter.up()

            wopt_unit = bin_iter.get_last_save()
            # use even width
            if wopt_unit % 2 == 1:
                wopt_unit += 1
            wopt = wopt_unit * resolution
            lopt = round(cur_res_targ / rsq * wopt / resolution) * resolution
            # print(num_par, wopt, lopt, res_iac_rms, cur_iac_rms, cur_res_targ)
            if lopt < max(lmin, min_nsq * wopt):
                # need to increase num_par and recalculate unit resistor dimensions
                num_par += num_par_inc
            else:
                done = True

        # step 3: fix maximum length violation by having resistor in series.
        if lopt > lmax:
            num_ser = -(-lopt // lmax)
            lopt = round(lopt / num_ser / resolution) * resolution
        else:
            num_ser = 1

        # step 4: return answer
        return num_par, num_ser, wopt * self.layout_unit, lopt * self.layout_unit


class DummyTechInfo(TechInfo):
    """A dummy TechInfo class.

    Parameters
    ----------
    tech_params : dict[str, any]
        technology parameters dictionary.
    """

    def __init__(self, tech_params):
        TechInfo.__init__(self, 0.001, 1e-6, '', tech_params)

    @classmethod
    def get_via_drc_info(cls, vname, vtype, mtype, mw_unit, is_bot):
        return (0, 0), (0, 0), (0, 0), [(0, 0)], None, None

    def get_min_space(self, layer_type, width):
        return 0.0

    def get_min_length(self, layer_type, width):
        return 0.0

    def get_layer_id(self, layer_name):
        return -1

    def get_layer_name(self, layer_id):
        return ''

    def get_layer_type(self, layer_name):
        return ''

    def get_via_name(self, bot_layer_id):
        return ''

    def get_metal_em_specs(self, layer_name, w, l=-1, vertical=False, **kwargs):
        return float('inf'), float('inf'), float('inf')

    def get_via_em_specs(self, via_name, bm_layer, tm_layer, via_type='square',
                         bm_dim=(-1, -1), tm_dim=(-1, -1), array=False, **kwargs):
        return 0.0, float('inf'), float('inf'), float('inf')

    def get_res_rsquare(self, res_type):
        return 0.0

    def get_res_width_bounds(self, res_type):
        return 0.0, 0.0

    def get_res_length_bounds(self, res_type):
        return 0.0, 0.0

    def get_res_min_nsquare(self, res_type):
        return 1.0

    def get_res_em_specs(self, res_type, w, l=-1, **kwargs):
        return 0.0, float('inf'), float('inf'), float('inf')

    def get_res_info(self, res_type, w, l, **kwargs):
        return None


class BagLayout(object):
    """This class contains layout information of a cell.

    Parameters
    ----------
    grid : :class:`bag.layout.routing.RoutingGrid`
        the routing grid instance.
    use_cybagoa : bool
        True to use cybagoa package to accelerate layout.
    pin_purpose : string
        Default pin purpose name.  Defaults to 'pin'.
    make_pin_rect : bool
        True to create pin object in addition to label.  Defaults to True.
    """

    def __init__(self, grid, use_cybagoa=False, pin_purpose='pin', make_pin_rect=True):
        self._res = grid.resolution
        self._via_tech = grid.tech_info.via_tech_name
        self._pin_purpose = pin_purpose
        self._make_pin_rect = make_pin_rect
        self._inst_list = []  # type: List[Instance]
        self._inst_primitives = []  # type: List[InstanceInfo]
        self._rect_list = []  # type: List[Rect]
        self._via_list = []  # type: List[Via]
        self._via_primitives = []  # type: List[ViaInfo]
        self._pin_list = []  # type: List[PinInfo]
        self._path_list = []  # type: List[Path]
        self._blockage_list = []  # type: List[Blockage]
        self._boundary_list = []  # type: List[Boundary]
        self._used_inst_names = set()
        self._used_pin_names = set()
        self._content = None
        self._finalized = False
        self._flat_inst_list = None
        self._flat_rect_list = None
        self._flat_via_list = None
        self._flat_path_list = None
        self._flat_blockage_list = None
        self._flat_boundary_list = None
        if use_cybagoa and cybagoa is not None:
            encoding = bag.io.get_encoding()
            self._oa_layout = cybagoa.PyLayout(encoding)
            self._flat_oa_layout = cybagoa.PyLayout(encoding)
        else:
            self._oa_layout = None
            self._flat_oa_layout = None

    @property
    def pin_purpose(self):
        """Returns the default pin layer purpose name."""
        return self._pin_purpose

    def inst_iter(self):
        # type: () -> Iterator[Instance]
        return iter(self._inst_list)

    def flatten(self):
        self._flat_rect_list = []
        self._flat_path_list = []
        self._flat_blockage_list = []
        self._flat_boundary_list = []
        self._flat_via_list = []  # type: List[ViaInfo]
        self._flat_inst_list = []  # type: List[InstanceInfo]

        # get rectangles
        for obj in self._rect_list:
            if obj.valid:
                obj_content = obj.content
                self._flat_rect_list.append(obj_content)
                if self._flat_oa_layout is not None:
                    self._flat_oa_layout.add_rect(**obj_content)

        for obj in self._path_list:
            if obj.valid:
                obj_content = obj.content
                self._flat_path_list.append(obj_content)
                if self._flat_oa_layout is not None:
                    self._flat_oa_layout.add_path(**obj_content)

        for obj in self._blockage_list:
            if obj.valid:
                obj_content = obj.content
                self._flat_blockage_list.append(obj_content)
                if self._flat_oa_layout is not None:
                    self._flat_oa_layout.add_blockage(**obj_content)

        for obj in self._boundary_list:
            if obj.valid:
                obj_content = obj.content
                self._flat_boundary_list.append(obj_content)
                if self._flat_oa_layout is not None:
                    self._flat_oa_layout.add_boundary(**obj_content)

        # get vias
        for obj in self._via_list:
            if obj.valid:
                obj_content = obj.content
                self._flat_via_list.append(obj_content)
                if self._flat_oa_layout is not None:
                    self._flat_oa_layout.add_via(**obj_content)

        # get via primitives
        self._flat_via_list.extend(self._via_primitives)
        if self._flat_oa_layout is not None:
            for via in self._via_primitives:
                self._flat_oa_layout.add_via(**via)

        # get instances
        for obj in self._inst_list:
            if obj.valid:
                # TODO: add support for flatten blockage/boundary
                finst_list, frect_list, fvia_list, fpath_list = obj.flatten()
                self._flat_inst_list.extend(finst_list)
                self._flat_rect_list.extend(frect_list)
                self._flat_via_list.extend(fvia_list)
                self._flat_path_list.extend(fpath_list)
                if self._flat_oa_layout is not None:
                    for fobj in finst_list:
                        self._flat_oa_layout.add_inst(**fobj)
                    for fobj in frect_list:
                        self._flat_oa_layout.add_rect(**fobj)
                    for fobj in fvia_list:
                        self._flat_oa_layout.add_via(**fobj)
                    for fobj in fpath_list:
                        self._flat_oa_layout.add_path(**fobj)
        # get instance primitives
        self._flat_inst_list.extend(self._inst_primitives)
        if self._flat_oa_layout is not None:
            for obj in self._inst_primitives:
                self._flat_oa_layout.add_inst(**obj)

        # add pins to oa layout
        if self._flat_oa_layout is not None:
            for pin in self._pin_list:
                self._flat_oa_layout.add_pin(**pin)

    def finalize(self, flatten=False):
        # type: (bool) -> None
        """Prevents any further changes to this layout.

        Parameters
        ----------
        flatten : bool
            True to compute flattened layout.
        """
        self._finalized = True

        # get rectangles
        rect_list = []
        for obj in self._rect_list:
            if obj.valid:
                if not obj.bbox.is_physical():
                    raise ValueError('rectangle with non-physical bounding box found.')
                obj_content = obj.content
                rect_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_rect(**obj_content)

        path_list = []
        for obj in self._path_list:
            if obj.valid:
                obj_content = obj.content
                path_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_path(**obj_content)

        blockage_list = []
        for obj in self._blockage_list:
            if obj.valid:
                obj_content = obj.content
                blockage_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_blockage(**obj_content)

        boundary_list = []
        for obj in self._boundary_list:
            if obj.valid:
                obj_content = obj.content
                boundary_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_boundary(**obj_content)

        # get vias
        via_list = []  # type: List[ViaInfo]
        for obj in self._via_list:
            if obj.valid:
                obj_content = obj.content
                via_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_via(**obj_content)
        # get via primitives
        via_list.extend(self._via_primitives)
        if self._oa_layout is not None:
            for via in self._via_primitives:
                self._oa_layout.add_via(**via)

        # get instances
        inst_list = []  # type: List[InstanceInfo]
        for obj in self._inst_list:
            if obj.valid:
                obj_content = self._format_inst(obj)
                inst_list.append(obj_content)
                if self._oa_layout is not None:
                    self._oa_layout.add_inst(**obj_content)
        # get instance primitives
        inst_list.extend(self._inst_primitives)
        if self._oa_layout is not None:
            for obj in self._inst_primitives:
                self._oa_layout.add_inst(**obj)

        # add pins to oa layout
        if self._oa_layout is not None:
            for pin in self._pin_list:
                self._oa_layout.add_pin(**pin)

        # TODO: add blockage/boundary support
        self._content = [inst_list,
                         rect_list,
                         via_list,
                         self._pin_list,
                         path_list,
                         ]

        if flatten:
            self.flatten()

    def get_flat_geometries(self):
        """Returns flattened geometries in this layout."""
        # TODO: add blockage/boundary support
        return self._flat_inst_list, self._flat_rect_list, self._flat_via_list, self._flat_path_list

    def get_masters_set(self):
        """Returns a set of all template master keys used in this layout."""
        return set((inst.master.key for inst in self._inst_list))

    def _get_unused_inst_name(self, inst_name):
        """Returns a new inst name."""
        if inst_name is None or inst_name in self._used_inst_names:
            cnt = 0
            inst_name = 'X%d' % cnt
            while inst_name in self._used_inst_names:
                cnt += 1
                inst_name = 'X%d' % cnt

        return inst_name

    def _format_inst(self, inst):
        # type: (Instance) -> InstanceInfo
        """Convert the given instance into dictionary representation."""
        content = inst.content
        inst_name = self._get_unused_inst_name(content.name)
        content.name = inst_name
        self._used_inst_names.add(inst_name)
        return content

    def get_content(self, cell_name, flatten=False):
        """returns a list describing geometries in this layout.

        Parameters
        ----------
        cell_name : str
            the layout top level cell name.
        flatten : bool
            True to flatten all instances

        Returns
        -------
        content :
            a list describing this layout, or PyOALayout if cybagoa package is enabled.
        """
        if not self._finalized:
            raise Exception('Layout is not finalized.')

        if flatten:
            # TODO: add blockage/boundary support
            if self._flat_oa_layout is not None:
                return cell_name, self._flat_oa_layout
            return [cell_name, self._flat_inst_list, self._flat_rect_list,
                    self._flat_via_list, self._pin_list, self._flat_path_list]
        else:
            if self._oa_layout is not None:
                return cell_name, self._oa_layout
            ans = [cell_name]
            ans.extend(self._content)
            return ans

    def add_instance(self, instance):
        """Adds the given instance to this layout.

        Parameters
        ----------
        instance : bag.layout.objects.Instance
            the instance to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # if isinstance(instance.nx, float) or isinstance(instance.ny, float):
        #     raise Exception('float nx/ny')

        self._inst_list.append(instance)

    def move_all_by(self, dx=0.0, dy=0.0):
        # type: (float, float) -> None
        """Move all layout objects in this layout by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        for obj in chain(self._inst_list, self._inst_primitives, self._rect_list,
                         self._via_primitives, self._via_list, self._pin_list,
                         self._path_list, self._blockage_list, self._boundary_list):
            obj.move_by(dx=dx, dy=dy)

    def add_instance_primitive(self, lib_name, cell_name, loc,
                               view_name='layout', inst_name=None, orient="R0",
                               num_rows=1, num_cols=1, sp_rows=0.0, sp_cols=0.0,
                               params=None, **kwargs):

        """Adds a new (arrayed) primitive instance to this layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        loc : tuple(float, float)
            instance location.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : str or None
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        orient : str
            instance orientation.  Defaults to "R0"
        num_rows : int
            number of rows.  Must be positive integer.
        num_cols : int
            number of columns.  Must be positive integer.
        sp_rows : float
            row spacing.  Used for arraying given instance.
        sp_cols : float
            column spacing.  Used for arraying given instance.
        params : dict[str, any]
            the parameter dictionary.  Used for adding pcell instance.
        **kwargs
            additional arguments.  Usually implementation specific.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # get unique instance name
        inst_name = self._get_unused_inst_name(inst_name)
        self._used_inst_names.add(inst_name)

        inst_info = InstanceInfo(self._res, lib=lib_name,
                                 cell=cell_name,
                                 view=view_name,
                                 name=inst_name,
                                 loc=[round(loc[0] / self._res) * self._res,
                                      round(loc[1] / self._res) * self._res],
                                 orient=orient,
                                 num_rows=num_rows,
                                 num_cols=num_cols,
                                 sp_rows=round(sp_rows / self._res) * self._res,
                                 sp_cols=round(sp_cols / self._res) * self._res)

        # if isinstance(num_rows, float) or isinstance(num_cols, float):
        #     raise Exception('float nx/ny')

        if params is not None:
            inst_info.params = params
        inst_info.update(kwargs)

        self._inst_primitives.append(inst_info)

    def add_rect(self, rect):
        """Add a new (arrayed) rectangle.

        Parameters
        ----------
        rect : bag.layout.objects.Rect
            the rectangle object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # if isinstance(rect.nx, float) or isinstance(rect.ny, float):
        #     raise Exception('float nx/ny')

        self._rect_list.append(rect)

    def add_path(self, path):
        # type: (Path) -> None
        """Add a new path.

        Parameters
        ----------
        path : Path
            the path object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')
        self._path_list.append(path)

    def add_blockage(self, blockage):
        # type: (Blockage) -> None
        """Add a new blockage.

        Parameters
        ----------
        blockage : Blockage
            the blockage object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')
        self._blockage_list.append(blockage)

    def add_boundary(self, boundary):
        # type: (Boundary) -> None
        """Add a new boundary.

        Parameters
        ----------
        boundary : Boundary
            the boundary object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')
        self._boundary_list.append(boundary)

    def add_via(self, via):
        """Add a new (arrayed) via.

        Parameters
        ----------
        via : bag.layout.objects.Via
            the via object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # if isinstance(via.nx, float) or isinstance(via.ny, float):
        #     raise Exception('float nx/ny')

        self._via_list.append(via)

    def add_via_primitive(self, via_type, loc, num_rows=1, num_cols=1, sp_rows=0.0, sp_cols=0.0,
                          enc1=None, enc2=None, orient='R0', cut_width=None, cut_height=None,
                          arr_nx=1, arr_ny=1, arr_spx=0.0, arr_spy=0.0):
        """Adds a primitive via by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        loc : list[float]
            the via location as a two-element list.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : float
            spacing between via cut rows.
        sp_cols : float
            spacing between via cut columns.
        enc1 : list[float]
            a list of left, right, top, and bottom enclosure values on bottom layer.  Defaults to all 0.
        enc2 : list[float]
            a list of left, right, top, and bottom enclosure values on top layer.  Defaults. to all 0.
        orient : str
            orientation of the via.
        cut_width : float or None
            via cut width.  This is used to create rectangle via.
        cut_height : float or None
            via cut height.  This is used to create rectangle via.
        arr_nx : int
            number of columns.
        arr_ny : int
            number of rows.
        arr_spx : float
            column pitch.
        arr_spy : float
            row pitch.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        if arr_nx > 0 and arr_ny > 0:
            if enc1 is None:
                enc1 = [0.0, 0.0, 0.0, 0.0]
            if enc2 is None:
                enc2 = [0.0, 0.0, 0.0, 0.0]

            # if isinstance(arr_nx, float) or isinstance(arr_ny, float):
            #     raise Exception('float nx/ny')

            par = ViaInfo(self._res, id=via_type, loc=loc, orient=orient, num_rows=num_rows, num_cols=num_cols,
                          sp_rows=sp_rows, sp_cols=sp_cols, enc1=enc1, enc2=enc2, )
            if cut_width is not None:
                par['cut_width'] = cut_width
            if cut_height is not None:
                par['cut_height'] = cut_height
            if arr_nx > 1 or arr_ny > 1:
                par['arr_nx'] = arr_nx
                par['arr_ny'] = arr_ny
                par['arr_spx'] = arr_spx
                par['arr_spy'] = arr_spy
            self._via_primitives.append(par)

    def add_pin(self, net_name, layer, bbox, pin_name=None, label=None):
        """Add a new pin.

        Parameters
        ----------
        net_name : str
            the net name associated with this pin.
        layer : string or (string, string)
            the layer name, or (layer, purpose) pair.
            if purpose is not specified, defaults to 'pin'.
        bbox : bag.layout.util.BBox
            the rectangle bounding box
        pin_name : str or None
            the pin name.  If None or empty, auto-generate from net name.
        label : str or None
            the pin label text.  If None or empty, will use net name as the text.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        if isinstance(layer, bytes):
            # interpret as unicode
            layer = layer.decode('utf-8')
        if isinstance(layer, str):
            layer = (layer, self._pin_purpose)
        else:
            layer = layer[0], layer[1]

        if not label:
            label = net_name

        pin_name = pin_name or net_name
        idx = 1
        while pin_name in self._used_pin_names:
            pin_name = '%s_%d' % (net_name, idx)
            idx += 1

        par = PinInfo(self._res, net_name=net_name,
                      pin_name=pin_name,
                      label=label,
                      layer=list(layer),
                      bbox=[[bbox.left, bbox.bottom], [bbox.right, bbox.top]],
                      make_rect=self._make_pin_rect)

        self._used_pin_names.add(pin_name)
        self._pin_list.append(par)
