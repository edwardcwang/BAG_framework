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


"""This module defines abstract finfet classes that subclass analog mosfet classes.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *
from future.utils import with_metaclass

import abc
import math

from bag import float_to_si_string
from bag.layout.template import TemplateBase
from bag.layout.util import BBox

from .core import AnalogMosBase


# noinspection PyAbstractClass
class AnalogFinfetFoundation(with_metaclass(abc.ABCMeta, TemplateBase)):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogFinfetFoundation, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    def draw_foundation(self, lch, nfin, fg, nduml, ndumr, arr_box_ext, tech_constants,
                        dx=0.0, no_cpo=False):
        """Draw the layout of this template.

        Override this method to create the layout.

        Parameters
        ----------
        lch : float
            the transistor channel length.
        nfin : float for int
            array box height in number of fins.
        fg : int
            number of polys to draw.
        nduml : int
            number of dummy polys on the left.
        ndumr : int
            number of dummy polys on the right.
        arr_box_ext : list[float]
            array box extension on the left, bottom, right, and top.
        tech_constants : dict[str, any]
            the technology constants dictionary.  Must have the following entries:

            mos_fin_pitch : float
                the pitch between fins.
            mos_cpo_h : float
                the height of CPO layer.
            mos_cpo_h_end : float
                the height of CPO layer at the end of substrate.
            sd_pitch : float
                source and drain pitch of the transistor.
            implant_layers : list[str]
                list of implant/threshold layers to draw.
            extra_layers : list[((str, str), list[float])]
                a list of extra technology layers to draw.  Each element is a tuple of (layer, purpose)
                and (left, bottom, right, top) extension over the array box.
        dx : float
            shift the layout by this amount horizontally.
        no_cpo : bool
            True to not draw CPO layer.  This is used when drawing substrate connections
            in extension guard ring.
        """
        if arr_box_ext is None:
            arr_box_ext = [0, 0, 0, 0]

        lch /= self.grid.layout_unit
        res = self.grid.resolution

        mos_fin_pitch = tech_constants['mos_fin_pitch']
        mos_cpo_h = tech_constants['mos_cpo_h']
        sd_pitch = tech_constants['sd_pitch']
        lay_list = tech_constants['implant_layers']
        extra_list = tech_constants['extra_layers']

        extl, extb, extr, extt = arr_box_ext

        # check if we're drawing substrate foundation.  If so use different CPO height for bottom.
        if extb > 0:
            mos_cpo_h_bot = tech_constants['mos_cpo_h_end']
        else:
            mos_cpo_h_bot = mos_cpo_h

        if extt > 0:
            mos_cpo_h_top = tech_constants['mos_cpo_h_end']
        else:
            mos_cpo_h_top = mos_cpo_h

        # +2 to account for 2 PODE polys.
        fg_tot = nduml + fg + ndumr
        bnd_box_w = fg_tot * sd_pitch

        # compute array box
        mos_cls = self.grid.tech_info.tech_params['layout']['mos_template']
        hm_layer_id = mos_cls.port_layer_id() + 1
        hm_track_pitch = self.grid.get_track_pitch(hm_layer_id)
        # pr_bnd_yext = mos_fin_pitch * (np.ceil(mos_cpo_h / mos_fin_pitch - 0.5) + 0.5)
        pr_bnd_yext = hm_track_pitch * math.ceil(mos_cpo_h / 2.0 / hm_track_pitch)
        arr_box_bot = pr_bnd_yext
        arr_box_top = arr_box_bot + nfin * mos_fin_pitch
        arr_box_left = dx
        arr_box_right = dx + bnd_box_w
        arr_box = BBox(arr_box_left, arr_box_bot, arr_box_right, arr_box_top, res)

        # draw CPO
        if not no_cpo:
            self.add_rect('CPO', BBox(arr_box_left - extl, arr_box.bottom - mos_cpo_h_bot / 2.0,
                                      arr_box_right + extr, arr_box.bottom + mos_cpo_h_bot / 2.0, res))
            self.add_rect('CPO', BBox(arr_box_left - extl, arr_box.top - mos_cpo_h_top / 2.0,
                                      arr_box_right + extr, arr_box.top + mos_cpo_h_top / 2.0, res))

        # draw DPO/PO
        dpo_lp = ('PO', 'dummy1')
        po_lp = ('PO', 'drawing')
        yb = arr_box.bottom - extb
        yt = arr_box.top + extt
        dl = lch / 2.0
        # draw DPO left
        xmid = dx + 0.5 * sd_pitch
        self.add_rect(dpo_lp, BBox(xmid - dl, yb, xmid + dl, yt, res),
                      nx=nduml, spx=sd_pitch)
        # draw PO
        xmid = dx + (nduml + 0.5) * sd_pitch
        self.add_rect(po_lp, BBox(xmid - dl, yb, xmid + dl, yt, res),
                      nx=fg, spx=sd_pitch)
        # draw DPO right
        xmid = dx + (nduml + fg + 0.5) * sd_pitch
        self.add_rect(dpo_lp, BBox(xmid - dl, yb, xmid + dl, yt, res),
                      nx=ndumr, spx=sd_pitch)

        # draw VT/implant
        imp_box = BBox(arr_box_left - extl, arr_box_bot - extb,
                       arr_box_right + extr, arr_box_top + extt, res)
        for lay in lay_list:
            self.add_rect(lay, imp_box)
        for lay_purp, (aextl, aextb, aextr, aextt) in extra_list:
            box = BBox(arr_box.left - aextl, arr_box.bottom - aextb,
                       arr_box.right + aextr, arr_box.top + aextt, arr_box.resolution)
            self.add_rect(lay_purp, box)

        # draw PR boundary
        self.add_rect(('prBoundary', 'boundary'),
                      BBox(arr_box_left - extl, 0.0, arr_box_right + extr,
                           arr_box_top + pr_bnd_yext, res))

        # set array box of this template
        self.array_box = arr_box


class AnalogFinfetExt(AnalogFinfetFoundation):
    """The template for finfet vertical extension block.  Used to add more routing tracks.

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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogFinfetFoundation.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        tech_params = self.grid.tech_info.tech_params
        self._gr_cls = tech_params['layout']['ext_guard_ring_template']

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
            mos_type="transistor type, either 'pch' or 'nch'.",
            threshold='transistor threshold flavor.',
            fg='number of fingers.',
            nfin='height of the extension in number of fins',
            tech_constants='technology constants dictionary.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  Use 0 for no guard ring.',
            is_end='True if this template is at the ends.',
        )

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
            guard_ring_nf=0,
            is_end=False,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        lch = self.params['lch']
        nfin = self.params['nfin']
        fg = self.params['fg']
        lch_str = float_to_si_string(lch)
        gr_nf = self.params['guard_ring_nf']
        main = '%s_%s_l%s_fin%d_fg%d' % (mos_type, threshold, lch_str, nfin, fg)
        name = 'ext_end' if self.params['is_end'] else 'ext'

        if gr_nf > 0:
            return '%s_gr%d_%s' % (main, gr_nf, name)
        else:
            return main + '_' + name

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self):
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        lch = self.params['lch']
        mos_type = self.params['mos_type']
        nfin = self.params['nfin']
        fg = self.params['fg']
        threshold = self.params['threshold']
        tech_constants = self.params['tech_constants']
        gr_nf = self.params['guard_ring_nf']
        is_end = self.params['is_end']

        if fg <= 0:
            raise ValueError('Number of fingers must be positive.')

        nfin_min = tech_constants['mos_ext_nfin_min']

        if nfin < nfin_min:
            raise ValueError('Extension must have a minimum of %d fins' % nfin_min)

        ndum = tech_constants['mos_edge_num_dpo']  # type: int
        xext = tech_constants['mos_edge_xext']  # type: float
        od_dummy_layer = tech_constants.get('od_dummy_layer', None)
        od_dummy_sp_nfin = tech_constants.get('od_dummy_sp_nfin', 0)

        # make sure CPO overlap rule is met at the ends
        if is_end:
            mos_core_cpo_po_ov = tech_constants['mos_core_cpo_po_ov']
            mos_cpo_h = tech_constants['mos_cpo_h_end']
            extb = max(mos_core_cpo_po_ov - mos_cpo_h / 2.0, 0.0)
        else:
            extb = 0.0

        gr_master = None
        xshift = 0.0
        arr_box_ext = [xext, extb, xext, 0.0]
        if gr_nf > 0:
            # create guard ring master.
            gr_params = dict(
                lch=lch,
                height=nfin,
                sub_type='ptap' if mos_type == 'nch' or mos_type == 'ptap' else 'ntap',
                threshold=threshold,
                fg=gr_nf,
                is_end=is_end,
            )
            gr_master = self.new_template(params=gr_params, temp_cls=self._gr_cls)  # type: AnalogFinfetGuardRingExt
            # add left guard ring
            lgr_inst = self.add_instance(gr_master, inst_name='XLGR')
            # re-export body pins
            self.reexport(lgr_inst.get_port(), show=False)
            # calculate shift
            xshift = gr_master.array_box.right
            # share dummy polys
            ndum //= 2

        # draw foundation
        self.draw_foundation(lch=lch, nfin=nfin, fg=fg + 2,
                             nduml=ndum, ndumr=ndum, arr_box_ext=arr_box_ext,
                             tech_constants=tech_constants, dx=xshift)

        if gr_master is not None:
            # draw right guard ring.
            rgr_inst = self.add_instance(gr_master, inst_name='XRGR', orient='MY')
            rgr_inst.move_by(dx=self.array_box.right + gr_master.array_box.right)
            # re-export body pins
            self.reexport(rgr_inst.get_port(), show=False)
            # update array box
            xl = gr_master.array_box.left
            wgr = gr_master.array_box.width
            wfund = self.array_box.width
            self.array_box = BBox(xl, self.array_box.bottom,
                                  xl + 2 * wgr + wfund, self.array_box.top, self.array_box.resolution)

        # draw dummy OD
        if od_dummy_layer is not None:
            sd_pitch = tech_constants['sd_pitch']
            mos_fin_h = tech_constants['mos_fin_h']
            mos_fin_pitch = tech_constants['mos_fin_pitch']
            lch_layout = lch / self.grid.layout_unit
            xl = xshift + (ndum + 0.5) * sd_pitch - lch_layout / 2.0
            xr = xl + lch_layout + (fg + 1) * sd_pitch
            yb = self.array_box.bottom + od_dummy_sp_nfin * mos_fin_pitch - mos_fin_h / 2.0
            yt = self.array_box.top - od_dummy_sp_nfin * mos_fin_pitch + mos_fin_h / 2.0
            self.add_rect(od_dummy_layer, BBox(xl, yb, xr, yt, self.grid.resolution))


# noinspection PyAbstractClass
class AnalogFinfetGuardRingExt(with_metaclass(abc.ABCMeta, TemplateBase)):
    """A guard ring for extension block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
        the template database.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogFinfetGuardRingExt, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            height='guard ring height, in meters/number of fins.',
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            fg='width of guard ring in number of fingers.',
            is_end='True if this template is at the ends.',
        )

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
            is_end=False,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        h_str = float_to_si_string(self.params['height'])
        main = '%s_%s_l%s_h%s_fg%d_guardring' % (self.params['sub_type'],
                                                 self.params['threshold'],
                                                 lch_str, h_str,
                                                 self.params['fg'])
        if self.params['is_end']:
            main += '_end'
        return main

    def compute_unique_key(self):
        return self.get_layout_basename()


class AnalogFinfetEdge(with_metaclass(abc.ABCMeta, AnalogFinfetFoundation)):
    """The template for finfet horizontal edge block.
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
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogFinfetEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        tech_params = self.grid.tech_info.tech_params
        self._gr_cls = tech_params['layout']['edge_guard_ring_template']

    @abc.abstractmethod
    def draw_od_edge(self, xshift, yc, w, tech_constants):
        """Draw od edge dummies.

        You can assume that self.array_box is already set.
        """
        pass

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
            mos_type="transistor type, either 'pch' or 'nch'.",
            threshold='transistor threshold flavor.',
            w='transistor width, in meters/number of fins.',
            bext='bottom extension in number of fins',
            text='top extension in number of fins',
            tech_constants='technology constants dictionary.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  Use 0 for no guard ring.',
            is_guard_ring='True if this is edge of a guard ring.',
            is_right="True if this is the right edge.",
            end_mode='An integer indicating whether top/bottom of this template is at the ends.',
            is_ds_dummy='True if this template is only used to create drain/source dummy metals.',
        )

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
            guard_ring_nf=0,
            is_guard_ring=False,
            is_right=False,
            end_mode=0,
            is_ds_dummy=False,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        lch = self.params['lch']
        w = self.params['w']
        bext = self.params['bext']
        text = self.params['text']
        lch_str = float_to_si_string(lch)
        gr_nf = self.params['guard_ring_nf']
        end_mode = self.params['end_mode']
        main = '%s_%s_l%s_w%d_bex%d_tex%d' % (mos_type, threshold, lch_str, w, bext, text)
        if self.params['is_guard_ring']:
            blk_name = 'gredge'
        else:
            blk_name = 'edge'
        if self.params['is_right']:
            blk_name += 'r'
        blk_name += '_end%d' % end_mode
        if self.params['is_ds_dummy']:
            blk_name += '_dsdummy'
        if gr_nf > 0:
            return '%s_gr%d_%s' % (main, gr_nf, blk_name)
        else:
            return main + '_' + blk_name

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self):
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        lch = self.params['lch']
        w = self.params['w']
        bext = self.params['bext']
        text = self.params['text']
        tech_constants = self.params['tech_constants']
        gr_nf = self.params['guard_ring_nf']
        end_mode = self.params['end_mode']
        is_guard_ring = self.params['is_guard_ring']
        is_right = self.params['is_right']

        res = self.grid.resolution

        mos_fin_h = tech_constants['mos_fin_h']
        mos_fin_pitch = tech_constants['mos_fin_pitch']
        ndum = tech_constants['mos_edge_num_dpo']
        xext = tech_constants['mos_edge_xext']
        sd_pitch = tech_constants['sd_pitch']
        nfin = tech_constants['nfin']
        mos_core_cpo_po_ov = tech_constants['mos_core_cpo_po_ov']

        if end_mode % 2 == 1:
            extb = max(mos_core_cpo_po_ov - tech_constants['mos_cpo_h_end'] / 2.0, 0.0)
        else:
            extb = 0.0
        if end_mode // 2 == 1:
            extt = max(mos_core_cpo_po_ov - tech_constants['mos_cpo_h_end'] / 2.0, 0.0)
        else:
            extt = 0.0

        gr_master = None
        xshift = 0.0
        arr_box_ext = [xext, extb, 0.0, extt]
        if is_guard_ring and is_right:
            # the right guard ring edge shares dummy poly, so we need to shrink
            # implant layers
            arr_box_ext[0] -= (ndum / 2) * sd_pitch

        if gr_nf > 0:
            # create guard ring master.
            gr_params = dict(
                lch=lch,
                sub_type='ptap' if mos_type == 'nch' or mos_type == 'ptap' else 'ntap',
                threshold=threshold,
                w=w,
                bext=bext,
                text=text,
                fg=gr_nf,
                end_mode=end_mode,
            )
            gr_master = self.new_template(params=gr_params, temp_cls=self._gr_cls)  # type: AnalogFinfetGuardRingExt
            # add left guard ring
            gr_inst = self.add_instance(gr_master, inst_name='XGR')
            # re-export body pins
            self.reexport(gr_inst.get_port(), show=False)
            # calculate shift
            xshift = gr_master.array_box.right
            # share dummy polys
            ndum //= 2

        # draw foundation, include 1 PODE poly
        self.draw_foundation(lch=lch, nfin=nfin + bext + text, fg=1,
                             nduml=ndum, ndumr=0, arr_box_ext=arr_box_ext,
                             tech_constants=tech_constants, dx=xshift,
                             no_cpo=is_guard_ring)

        if gr_master is not None:
            # update array box
            xl = gr_master.array_box.left
            wgr = gr_master.array_box.width
            wfund = self.array_box.width
            self.array_box = BBox(xl, self.array_box.bottom,
                                  xl + wgr + wfund, self.array_box.top, self.array_box.resolution)

        # set block size from array box
        tech_params = self.grid.tech_info.tech_params
        mos_cls = tech_params['layout']['mos_template']
        h_layer_id = mos_cls.port_layer_id() + 1
        self.set_size_from_array_box(h_layer_id)
        od_yc = self.array_box.bottom + tech_constants['mos_edge_od_dy'] + bext * mos_fin_pitch

        # draw OD/PODE
        if not self.params['is_ds_dummy']:
            lch_layout = lch / self.grid.layout_unit
            od_h = mos_fin_h + (w - 1) * mos_fin_pitch
            xmid = xshift + (ndum + 0.5) * sd_pitch
            xl = xmid - lch_layout / 2.0
            xr = xmid + lch_layout / 2.0
            box = BBox(xl, od_yc - od_h / 2.0, xr, od_yc + od_h / 2.0, res)
            self.add_rect('OD', box)
            self.add_rect(('PODE', 'dummy1'), box)

        # draw OD edge objects
        self.draw_od_edge(xshift, od_yc, w, tech_constants)


# noinspection PyAbstractClass
class AnalogFinfetGuardRingEdge(with_metaclass(abc.ABCMeta, TemplateBase)):
    """A guard ring for extension block.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
        the template database.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogFinfetGuardRingEdge, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            sub_type="substrate type, either 'ptap' or 'ntap'.",
            threshold='transistor threshold flavor.',
            w='transistor width, in meters/number of fins.',
            bext='bottom extension in number of fins',
            text='top extension in number of fins',
            fg='width of guard ring in number of fingers.',
            end_mode='An integer indicating whether top/bottom of this template is at the ends.',
        )

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
            end_mode=0,
        )

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        sub_type = self.params['sub_type']
        threshold = self.params['threshold']
        lch = self.params['lch']
        w = self.params['w']
        bext = self.params['bext']
        text = self.params['text']
        fg = self.params['fg']
        lch_str = float_to_si_string(lch)
        main = '%s_%s_l%s_w%d_fg_%d_bex%d_tex%d_guardring' % (sub_type, threshold,
                                                              lch_str, w, fg, bext, text)
        main += '_end%d' % self.params['end_mode']
        return main

    def compute_unique_key(self):
        return self.get_layout_basename()


class AnalogFinfetBase(with_metaclass(abc.ABCMeta, AnalogMosBase)):
    """An abstract subclass of AnalogMosBase for finfet technology.

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
    tech_constants : dict[str, any]
        the technology constants dictionary.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AnalogFinfetBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._sd_center = None, None
        self._ds_track_idx = None

    @classmethod
    @abc.abstractmethod
    def get_tech_constants(cls):
        """Returns the technology constants dictionary."""
        return {}

    @classmethod
    @abc.abstractmethod
    def get_ext_params(cls, lch, mos_type, threshold, fg, ext_nfin, guard_ring_nf, is_end):
        """Returns a dictionary of extension block parameters.

        Parameters
        ----------
        lch : float
            channel length, in meters.
        mos_type : string
            transistor type.
        threshold : string
            transistor threshold.
        fg : int
            number of fingers.
        ext_nfin : int
            extension height in number of fins.
        guard_ring_nf : int
            width of guard ring in number of fingers.  0 to disable.
        is_end : bool
            True if this template is at the ends.

        Returns
        -------
        params : dict[str, any]
            the extension block parameters.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_edge_params(cls, lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                        guard_ring_style, guard_ring_nf, is_guard_ring_transistor, is_right, end_mode,
                        is_ds_dummy):
        """Returns a dictionary of edge block parameters.

        Parameters
        ----------
        lch : float
            channel length, in meters.
        w : int or float
            transistor width, in meters or number of fins.
        mos_type : string
            transistor type.
        threshold : string
            transistor threshold.
        fg : int
            number of fingers.
        core_bot_ext : int
            core bottom extension in number of fins.
        core_top_ext : int
            core top extension in number of fins.
        guard_ring_style : bool
            True to draw guard ring style edge.
        guard_ring_nf : int
            width of guard ring in number of fingers.  0 to disable.
        is_guard_ring_transistor : bool
            True if this is a guard ring transistor edge.
        is_right : bool
            True if this is the right edge.
        end_mode : int
            An integer indicating whether top/bottom of this template is at the ends.
        is_ds_dummy : bool
            True if this transistor is only for drain/source dummy metal connection.

        Returns
        -------
        params : dict[str, any] or
            the edge block parameters.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_core_params(cls, lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                        guard_ring_style, is_guard_ring_transistor, end_mode, is_ds_dummy):
        """Returns a dictionary of edge block parameters.

        Parameters
        ----------
        lch : float
            channel length, in meters.
        w : int or float
            transistor width, in meters or number of fins.
        mos_type : string
            transistor type.
        threshold : string
            transistor threshold.
        fg : int
            number of fingers.
        core_bot_ext : int
            core bottom extension in number of fins.
        core_top_ext : int
            core top extension in number of fins.
        guard_ring_style : bool
            True to draw guard ring style transistor core.
        is_guard_ring_transistor : bool
            True if this is a guard ring transistor.
        end_mode : int
            An integer indicating whether top/bottom of this template is at the ends.
        is_ds_dummy : bool
            True if this transistor is only for drain/source dummy metal connection.

        Returns
        -------
        params : dict[str, any] or
            the core block parameters.
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def get_ds_conn_info(cls, lch, w, guard_ring_style=False):
        """Returns a dictionary containing information about drain/source connection.

        Parameters
        ----------
        lch : float
            the channel length, in meters.
        w : int
            the number of fins.
        guard_ring_style : bool
            True to return information for guard ring style transistor.

        Returns
        -------
        ds_conn_info : dict[str, any]
            a dictionary with the following key/values:

            nfin : int
                the core transistor array box height, in number of fin pitches.
            od_dy : float
                vertical distance between OD center and array box bottom.
            g_tr_nfin_max : int
                maximum distance between top of gate track and array box bottom, in number of fin pitches.
            ds_tr_nfin_min : int
                minimum distance between bottom drain/source track and array box bottom, in number of fins
            od_h : float
                height of OD layer, in layout units.
            m0_h : float
                height of drain/source M0, in layout units.
            m1_h : float
                height of drain/source M1, in layout units.
            m0_w : float
                width of drain/source M0, in layout units.
            m1_w : float
                width of drain/source M1, in layout units
            nvia0 : int
                number of drain/source VIA0.
            enc0_list : list[float]
                drain/source VIA0 M0 enclosure list.
            enc1_list : list[float]
                drain/source VIA0 M1 enclosure list.
        """
        return {}

    @classmethod
    def draw_transistor(cls, template, lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                        bot_ext_nfin, top_ext_nfin, guard_ring_nf, res,
                        is_guard_ring_transistor=False, guard_ring_style=False, end_mode=0,
                        is_ds_dummy=False):
        """Draw a transistor at the given location.
        """

        tech_params = template.grid.tech_info.tech_params
        edge_cls = tech_params['layout']['finfet_edge_template']
        core_cls = tech_params['layout']['finfet_core_template']
        core_end_mode = 0
        if end_mode % 2 == 1 and bot_ext_nfin == 0:
            core_end_mode += 1
        if end_mode // 2 == 1 and top_ext_nfin == 0:
            core_end_mode += 2
        if fg <= 0:
            raise ValueError('Number of fingers must be positive.')

        # get technology constants
        mos_fin_pitch = cls.get_tech_constants()['mos_fin_pitch']

        # get extension needed to fit integer number of tracks
        core_info = cls.get_ds_conn_info(lch, w, guard_ring_style=guard_ring_style)
        od_dy = core_info['od_dy']

        inst_list = []

        # draw left edge
        ledge_params = cls.get_edge_params(lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                                           guard_ring_style, guard_ring_nf, is_guard_ring_transistor,
                                           False, core_end_mode, is_ds_dummy=is_ds_dummy)
        ledge_master = template.new_template(params=ledge_params, temp_cls=edge_cls)
        ledge_inst = template.add_instance(ledge_master, inst_name='XLEDGE')
        inst_list.append(ledge_inst)
        ledge_arr_box = ledge_inst.array_box

        # draw bottom extension if needed, then compute lower-left array box coordinate.
        if bot_ext_nfin > 0:
            # draw bottom extension
            bot_ext_params = cls.get_ext_params(lch, mos_type, threshold, fg, bot_ext_nfin,
                                                guard_ring_nf, end_mode % 2 == 1)
            bot_ext_master = template.new_template(params=bot_ext_params,
                                                   temp_cls=AnalogFinfetExt)
            bot_ext_inst = template.add_instance(bot_ext_master, inst_name='XBEXT')
            inst_list.append(bot_ext_inst)
            bot_ext_arr_box = bot_ext_inst.array_box
            # move left edge up
            ledge_inst.move_by(dy=bot_ext_arr_box.top - ledge_arr_box.bottom)
            # update left edge array box
            ledge_arr_box = ledge_inst.array_box

            arr_box_left, arr_box_bottom = bot_ext_arr_box.left, bot_ext_arr_box.bottom
        else:
            arr_box_left, arr_box_bottom = ledge_arr_box.left, ledge_arr_box.bottom

        # draw core transistor.
        core_params = cls.get_core_params(lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                                          guard_ring_style, is_guard_ring_transistor,
                                          core_end_mode, is_ds_dummy=is_ds_dummy)
        core_master = template.new_template(params=core_params, temp_cls=core_cls)
        core_inst = template.add_instance(core_master, inst_name='XMOS')
        inst_list.append(core_inst)
        core_arr_box = core_inst.array_box
        # move core transistor
        core_inst.move_by(dx=ledge_arr_box.right - core_arr_box.left,
                          dy=ledge_arr_box.bottom - core_arr_box.bottom)
        # update core array box
        core_arr_box = core_inst.array_box
        # infer source/drain pitch from array box width
        sd_center = core_arr_box.left, core_arr_box.bottom + od_dy + core_bot_ext * mos_fin_pitch

        # draw right edge
        redge_params = cls.get_edge_params(lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                                           guard_ring_style, guard_ring_nf, is_guard_ring_transistor,
                                           True, core_end_mode, is_ds_dummy=is_ds_dummy)
        redge_master = template.new_template(params=redge_params, temp_cls=edge_cls)
        redge_inst = template.add_instance(redge_master, inst_name='XREDGE', orient='MY')
        inst_list.append(redge_inst)
        redge_arr_box = redge_inst.array_box
        redge_inst.move_by(dx=core_arr_box.right - redge_arr_box.left,
                           dy=core_arr_box.bottom - redge_arr_box.bottom)
        # update right edge array box
        redge_arr_box = redge_inst.array_box

        # draw top extension if needed, then calculate top right coordinate
        if top_ext_nfin > 0:
            # draw top extension
            top_ext_params = cls.get_ext_params(lch, mos_type, threshold, fg, top_ext_nfin,
                                                guard_ring_nf, end_mode // 2 == 1)
            top_ext_master = template.new_template(params=top_ext_params,
                                                   temp_cls=AnalogFinfetExt)
            top_ext_inst = template.add_instance(top_ext_master, inst_name='XTEXT')
            inst_list.append(top_ext_inst)
            top_ext_arr_box = top_ext_inst.array_box
            top_ext_inst.move_by(dy=core_arr_box.top - top_ext_arr_box.bottom)
            # update array box
            top_ext_arr_box = top_ext_inst.array_box
            arr_box_right = top_ext_arr_box.right
            arr_box_top = top_ext_arr_box.top
        else:
            arr_box_right = redge_arr_box.right
            arr_box_top = redge_arr_box.top

        # set array box of this template
        array_box = BBox(arr_box_left, arr_box_bottom, arr_box_right, arr_box_top, res)

        if guard_ring_nf > 0:
            # get body ports.
            port_list = [inst.get_port('b') for inst in inst_list if inst.has_port('b')]
        else:
            port_list = []
        return inst_list, array_box, sd_center, port_list

    def get_ds_track_index(self):
        """Returns the bottom drain/source track index.

        Returns
        -------
        tr_idx : int
            the bottom drain/source track index.
        """
        return self._ds_track_idx

    def get_left_sd_center(self):
        """Returns the center coordinate of the leftmost source/drain connection.

        Returns
        -------
        xc : float
            the center X coordinate of left-most source/drain.
        yc : float
            the center Y coordinate of left-most source/drain.
        """
        return self._sd_center

    def draw_layout(self):
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        lch = self.params['lch']
        w = self.params['w']
        mos_type = self.params['mos_type']
        threshold = self.params['threshold']
        fg = self.params['fg']
        g_tracks = self.params['g_tracks']
        ds_tracks = self.params['ds_tracks']
        gds_space = self.params['gds_space']
        guard_ring_nf = self.params['guard_ring_nf']
        end_mode = self.params['end_mode']
        is_ds_dummy = self.params['is_ds_dummy']

        if fg <= 0:
            raise ValueError('Number of fingers must be positive.')

        # get technology constants
        tech_constants = self.get_tech_constants()
        mos_fin_pitch = tech_constants['mos_fin_pitch']
        if guard_ring_nf == 0:
            mos_ext_nfin_min = tech_constants['mos_ext_nfin_min']
        else:
            mos_ext_nfin_min = tech_constants['mos_gring_ext_nfin_min']

        # express track pitch as number of fin pitches
        track_width, track_space = self.grid.get_track_info(self.port_layer_id() + 1)
        track_pitch = track_width + track_space
        track_nfin = int(round(track_pitch * 1.0 / mos_fin_pitch))
        if abs(track_pitch - track_nfin * mos_fin_pitch) >= self.grid.resolution:
            # check track_pitch is multiple of nfin.
            msg = 'track pitch = %.4g not multiples of fin pitch = %.4g' % (track_pitch, mos_fin_pitch)
            raise ValueError(msg)

        # get extension needed to fit integer number of tracks
        core_info = self.get_ds_conn_info(lch, w)
        core_nfin = core_info['nfin']
        g_tr_nfin_max = core_info['g_tr_nfin_max']
        ds_tr_nfin_min = core_info['ds_tr_nfin_min']

        # compute minimum number of tracks and needed bottom extension
        # make sure always have at least 2 tracks.
        ntrack_min = -(-core_nfin // track_nfin)
        bot_ext_nfin = track_nfin * ntrack_min - core_nfin

        # See if the first track is far enough from core transistor
        g_tr_top_nfin = int(math.ceil((track_space / 2.0 + track_width) / mos_fin_pitch))
        g_tr_nfin_max += bot_ext_nfin
        if g_tr_top_nfin > g_tr_nfin_max:
            # first track from bottom too close to core transistor
            # add an additional track to increase spacing.
            bot_ext_nfin += track_nfin
            ntrack_min += 1

        # add more gate tracks
        bot_ext_nfin += max(g_tracks - 1, 0) * track_nfin
        ds_tr_nfin_min += bot_ext_nfin
        ntrack_min += g_tracks - 1

        # find index of bottom drain/source track index
        self._ds_track_idx = int(math.ceil((ds_tr_nfin_min * mos_fin_pitch - track_space / 2.0) / track_pitch))
        self._ds_track_idx = max(self._ds_track_idx, g_tracks + gds_space)
        # find top extension needed to get drain/source tracks
        top_ext_nfin = max(0, self._ds_track_idx + ds_tracks - ntrack_min) * track_nfin

        # determine if we need top/bottom extension blocks, or we should simply extend the core.
        if bot_ext_nfin < mos_ext_nfin_min:
            core_bot_ext = bot_ext_nfin
            bot_ext_nfin = 0
        else:
            core_bot_ext = 0
        if top_ext_nfin < mos_ext_nfin_min:
            core_top_ext = top_ext_nfin
            top_ext_nfin = 0
        else:
            core_top_ext = 0

        results = self.draw_transistor(self, lch, w, mos_type, threshold, fg, core_bot_ext, core_top_ext,
                                       bot_ext_nfin, top_ext_nfin, guard_ring_nf,
                                       self.grid.resolution, is_guard_ring_transistor=False,
                                       guard_ring_style=False, end_mode=end_mode,
                                       is_ds_dummy=is_ds_dummy)
        self.array_box, self._sd_center = results[1], results[2]
        # set block size from array box
        self.set_size_from_array_box(self.port_layer_id() + 1)

        for body_port in results[3]:
            self.reexport(body_port, show=False)
