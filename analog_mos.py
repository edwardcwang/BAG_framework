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


"""This module defines abstract analog mosfet template classes.
"""

import abc
import numpy as np
from itertools import repeat, chain

from bag import float_to_si_string
from bag.layout.util import BBox
from bag.layout.template import MicroTemplate


class AnalogMosBase(MicroTemplate):
    """An abstract template for analog mosfet.

    Must have parameters mos_type, lch, w, threshold, fg.
    Instantiates a transistor with minimum G/D/S connections.

    Parameters
    ----------
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    tech_name : str
        the technology name.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lib_name, params, used_names, tech_name):
        MicroTemplate.__init__(self, lib_name, params, used_names)
        self.res_um = None
        self.tech_name = tech_name

    def get_layout_basename(self):
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """

        lch_str = float_to_si_string(self.params['lch'])
        w_str = float_to_si_string(self.params['w'])
        return '%s_%s_%s_%s_w%s_fg%d_g%d_ds%s_base' % (self.tech_name,
                                                       self.params['mos_type'],
                                                       self.params['threshold'],
                                                       lch_str, w_str,
                                                       self.params['fg'],
                                                       self.params['g_tracks'],
                                                       self.params['ds_tracks'])

    def compute_unique_key(self):
        return self.get_layout_basename()


class AnalogFinfetBase(AnalogMosBase):
    """An abstract subclass of AnalogMosBase for finfet technology.

    Parameters
    ----------
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    tech_name : str
        the technology name.
    core_cls : class
        the Template class used to generate core transistor.
    edge_cls : class
        the Template class used to generate transistor edge block.
    ext_cls : class
        the Template class used to generate extension block.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lib_name, params, used_names, tech_name, core_cls, edge_cls, ext_cls):
        AnalogMosBase.__init__(self, lib_name, params, used_names, tech_name)
        self.core_cls = core_cls
        self.edge_cls = edge_cls
        self.ext_cls = ext_cls

    @abc.abstractmethod
    def get_ext_params(self, loc, num_tracks):
        """Returns a dictionary of extension block parameters.

        Parameters
        ----------
        loc : str
            location of the extension block.  Either 'top' or 'bot'.
        num_tracks : int
            number of gate or source/drain tracks needed.

        Returns
        -------
        params : dict[str, any] or None
            the extension block parameters, or None if it is not needed.
        """
        return {}

    @abc.abstractmethod
    def get_edge_params(self):
        """Returns a dictionary of edge block parameters.

        Returns
        -------
        params : dict[str, any] or
            the edge block parameters.
        """
        return {}

    @abc.abstractmethod
    def get_core_params(self):
        """Returns a dictionary of core block parameters.

        Returns
        -------
        params : dict[str, any] or
            the core block parameters.
        """
        return {}

    def draw_layout(self, layout, temp_db, grid,
                    mos_type='nch', threshold='lvt', lch=16e-9, w=4, fg=4,
                    g_tracks=1, ds_tracks=1):
        """Draw the layout of this template.

        Override this method to create the layout.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance to draw the layout with.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
        mos_type : str
            the transistor type.
        threshold : str
            the transistor threshold flavor.
        lch : float
            the transistor channel length.
        w : float for int
            the transistor width, or number of fins.
        fg : int
            the number of fingers.
        g_tracks : int
            number of gate routing tracks.
        ds_tracks : int
            number of drain/source routing tracks.
        """
        if fg <= 0:
            raise ValueError('Number of fingers must be positive.')

        self.res_um = grid.get_resolution()

        # create left edge, but don't add it to layout yet
        edge_params = self.get_edge_params()
        edge_blk = temp_db.new_template(params=edge_params, temp_cls=self.edge_cls)  # type: MicroTemplate
        edge_arr_box = edge_blk.array_box

        # draw bottom extension (if needed) and left edge.  Also compute lower-left array box coordinate.
        bot_ext_params = self.get_ext_params('bot', g_tracks)
        if bot_ext_params is not None:
            blk = temp_db.new_template(params=bot_ext_params, temp_cls=self.ext_cls)  # type: MicroTemplate
            self.add_template(layout, blk, 'XBEXT')
            bot_ext_arr_box = blk.array_box

            dy = bot_ext_arr_box.top - edge_arr_box.bottom
            arr_box_left, arr_box_bottom = bot_ext_arr_box.left, bot_ext_arr_box.bottom
        else:
            dy = 0.0
            arr_box_left, arr_box_bottom = edge_arr_box.left, edge_arr_box.bottom

        self.add_template(layout, edge_blk, 'XLEDGE', loc=(0.0, dy))

        # draw transistor
        core_params = self.get_core_params()
        core_blk = temp_db.new_template(params=core_params, temp_cls=self.core_cls)  # type: MicroTemplate
        core_arr_box = core_blk.array_box
        dx = edge_arr_box.right - core_arr_box.left
        self.add_template(layout, core_blk, 'XMOS', loc=(dx, dy))

        # draw right edge and compute right array box coordinate.
        dx = dx + core_arr_box.right + edge_arr_box.width - edge_arr_box.left
        self.add_template(layout, edge_blk, 'XREDGE', loc=(dx, dy), orient='MY')
        arr_box_right = edge_arr_box.left + dx
        arr_box_top = edge_arr_box.top + dy

        # draw top extension if needed.  Also update upper array box coordinate.
        top_ext_params = self.get_ext_params('top', ds_tracks)
        if top_ext_params is not None:
            blk = temp_db.new_template(params=top_ext_params, temp_cls=self.ext_cls)  # type: MicroTemplate
            top_ext_arr_box = blk.array_box
            dy = dy + core_arr_box.top - top_ext_arr_box.bottom
            self.add_template(layout, blk, 'XTEXT', loc=(0.0, dy))
            arr_box_top = top_ext_arr_box.top + dy

        # set array box of this template
        self.array_box = BBox(arr_box_left, arr_box_bottom, arr_box_right, arr_box_top, grid.get_resolution())


class AnalogFinfetFoundation(MicroTemplate):
    """The abstract base class for finfet layout classes.

    This class provides the draw_foundation() method, which draws the poly array
    and implantation layers.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, lib_name, params, used_names):
        MicroTemplate.__init__(self, lib_name, params, used_names)

    def draw_foundation(self, layout, res, lch=16e-9, nfin=4, fg=4,
                        nduml=0, ndumr=0, arr_box_ext=None,
                        tech_constants=None):
        """Draw the layout of this template.

        Override this method to create the layout.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance to draw the layout with.
        res : float
            layout resolution.
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
            sd_pitch : float
                source and drain pitch of the transistor.
            implant_layers : list[str]
                list of implant/threshold layers to draw.
        """
        if arr_box_ext is None:
            arr_box_ext = [0, 0, 0, 0]

        mos_fin_pitch = tech_constants['mos_fin_pitch']
        mos_cpo_h = tech_constants['mos_cpo_h']
        sd_pitch = tech_constants['sd_pitch']
        lay_list = tech_constants['implant_layers']

        extl, extb, extr, extt = arr_box_ext

        # +2 to account for 2 PODE polys.
        fg_tot = nduml + fg + ndumr
        bnd_box_w = fg_tot * sd_pitch + extl + extr

        # compute array box
        pr_bnd_yext = mos_fin_pitch * (np.ceil(mos_cpo_h / mos_fin_pitch - 0.5) + 0.5)
        arr_box_bot = pr_bnd_yext
        arr_box_top = arr_box_bot + nfin * mos_fin_pitch
        arr_box_left = extl
        arr_box_right = bnd_box_w - extr
        arr_box = BBox(arr_box_left, arr_box_bot, arr_box_right, arr_box_top, res)

        # draw CPO
        layout.add_rect('CPO', BBox(0.0, arr_box.bottom - mos_cpo_h / 2.0,
                                    bnd_box_w, arr_box.bottom + mos_cpo_h / 2.0, res))
        layout.add_rect('CPO', BBox(0.0, arr_box.top - mos_cpo_h / 2.0,
                                    bnd_box_w, arr_box.top + mos_cpo_h / 2.0, res))

        # draw DPO/PO
        lch_um = lch * 1e6
        dpo_lp = ('PO', 'dummy1')
        po_lp = ('PO', 'drawing')
        for idx, (layer, purpose) in enumerate(chain(repeat(dpo_lp, nduml), repeat(po_lp, fg),
                                                     repeat(dpo_lp, ndumr))):
            xmid = (idx + 0.5) * sd_pitch + extl
            layout.add_rect(layer, BBox(xmid - lch_um / 2.0, arr_box.bottom - extb,
                                        xmid + lch_um / 2.0, arr_box.top + extt, res),
                            purpose=purpose)

        # draw VT/implant
        imp_box = BBox(0.0, arr_box.bottom - extb,
                       arr_box.right + extr, arr_box.top + extt, res)
        for lay in lay_list:
            layout.add_rect(lay, imp_box)

        # draw PR boundary
        layout.add_rect('prBoundary', BBox(0.0, 0.0, arr_box_right + extr,
                                           arr_box_top + pr_bnd_yext, res),
                        purpose='boundary')

        # set array box of this template
        self.array_box = arr_box


class AnalogFinfetExt(AnalogFinfetFoundation):
    """The template for finfet vertical extension block.  Used to add more routing tracks.

    Parameters
    ----------
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    """

    def __init__(self, lib_name, params, used_names):
        AnalogFinfetFoundation.__init__(self, lib_name, params, used_names)

    def get_default_params(self):
        """Returns the default parameter dictionary.

        Override this method to return a dictionary of default parameter values.
        This returned dictionary should not include port_specs

        Returns
        -------
        default_params : dict[str, any]
            the default parameters dictionary.
        """
        return dict(mos_type='nch',
                    threshold='ulvt',
                    lch=16e-9,
                    nfin=8,
                    fg=2,
                    tech_constants=None,
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
        tech_str = self.params['tech_constants']['name']
        lch_str = float_to_si_string(lch)
        return '%s_%s_%s_%s_fin%d_fg%d_ext' % (tech_str, mos_type, threshold, lch_str, nfin, fg)

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self, layout, temp_db, grid,
                    mos_type='nch', threshold='lvt', lch=16e-9,
                    nfin=4, fg=4, tech_constants=None):
        """Draw the layout of this template.

        Override this method to create the layout.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance to draw the layout with.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
        mos_type : str
            the transistor type.  Either 'nch' or 'pch'
        threshold : str
            the transistor threshold flavor.
        lch : float
            the transistor channel length.
        nfin : float for int
            array box height in number of fins.
        fg : int
            the transistor number of fingers.
        tech_constants : dict[str, any]
            the technology constants dictionary.  Must have the following entries:

            mos_fin_pitch : float
                the pitch between fins.
            mos_cpo_h : float
                the height of CPO layer.
            sd_pitch : float
                source and drain pitch of the transistor.
            implant_layers : list[str]
                list of implant/threshold layers to draw.
            name : str
                a string describing the process technology.  Used for identification purposes.
            mos_edge_num_dpo : int
                number of dummy polys at each edge.
            mos_edge_xext : float
                horizontal extension of CPO/implant layers over the array box left and right edges.
        """
        res = grid.get_resolution()

        if fg <= 0:
            raise ValueError('Number of fingers must be positive.')

        ndum = tech_constants['mos_edge_num_dpo']  # type: int
        xext = tech_constants['mos_edge_xext']  # type: float

        # include 2 PODE polys
        self.draw_foundation(layout, res, lch=lch, nfin=nfin, fg=fg + 2,
                             nduml=ndum, ndumr=ndum, arr_box_ext=[xext, 0.0, xext, 0.0],
                             tech_constants=tech_constants)


class AnalogFinfetEdge(AnalogFinfetFoundation):
    """The template for finfet vertical extension block.  Used to add more routing tracks.

    Parameters
    ----------
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lib_name, params, used_names):
        AnalogFinfetFoundation.__init__(self, lib_name, params, used_names)

    @abc.abstractmethod
    def draw_od_edge(self, layout, temp_db, grid, w, tech_constants):
        """Draw od edge dummies.

        You can assume that self.array_box is already set.
        """
        pass

    def get_default_params(self):
        """Returns the default parameter dictionary.

        Override this method to return a dictionary of default parameter values.
        This returned dictionary should not include port_specs

        Returns
        -------
        default_params : dict[str, any]
            the default parameters dictionary.
        """
        return dict(mos_type='nch',
                    threshold='ulvt',
                    lch=16e-9,
                    w=8,
                    tech_constants=None,
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
        tech_str = self.params['tech_constants']['name']
        lch_str = float_to_si_string(lch)
        return '%s_%s_%s_%s_w%d_edge' % (tech_str, mos_type, threshold, lch_str, w)

    def compute_unique_key(self):
        return self.get_layout_basename()

    def draw_layout(self, layout, temp_db, grid,
                    mos_type='nch', threshold='lvt', lch=16e-9,
                    w=4, tech_constants=None):
        """Draw the layout of this template.

        Override this method to create the layout.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance to draw the layout with.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
        mos_type : str
            the transistor type.  Either 'nch' or 'pch'
        threshold : str
            the transistor threshold flavor.
        lch : float
            the transistor channel length.
        w : float for int
            transistor width.
        tech_constants : dict[str, any]
            the technology constants dictionary.  Must have the following entries:

            mos_fin_pitch : float
                the pitch between fins.
            mos_cpo_h : float
                the height of CPO layer.
            sd_pitch : float
                source and drain pitch of the transistor.
            implant_layers : list[str]
                list of implant/threshold layers to draw.
            name : str
                a string describing the process technology.  Used for identification purposes.
            mos_edge_num_dpo : int
                number of dummy polys at each edge.
            mos_edge_xext : float
                horizontal extension of CPO/implant layers over the array box left and right edges.
            mos_fin_h : float
                the height of the fin.
            nfin : int
                array box height in number of fin pitches.
            mos_core_cpo_po_ov : float
                overlap between CPO and PO.
        """
        res = grid.get_resolution()

        mos_fin_h = tech_constants['mos_fin_h']
        mos_fin_pitch = tech_constants['mos_fin_pitch']
        ndum = tech_constants['mos_edge_num_dpo']
        xext = tech_constants['mos_edge_xext']
        sd_pitch = tech_constants['sd_pitch']
        nfin = tech_constants['nfin']
        mos_core_cpo_po_ov = tech_constants['mos_core_cpo_po_ov']
        mos_cpo_h = tech_constants['mos_cpo_h']

        if mos_type == 'ptap' or mos_type == 'ntap':
            extb = max(mos_core_cpo_po_ov - mos_cpo_h / 2.0, 0.0)
        else:
            extb = 0.0

        # draw foundation, include 1 PODE poly
        self.draw_foundation(layout, res, lch=lch, nfin=nfin, fg=1,
                             nduml=ndum, ndumr=0, arr_box_ext=[xext, extb, 0.0, 0.0],
                             tech_constants=tech_constants)

        # draw OD/PODE
        od_h = mos_fin_h + (w - 1) * mos_fin_pitch
        lch_um = lch * 1e6
        xmid = (ndum + 0.5) * sd_pitch + xext
        xl = xmid - lch_um / 2.0
        xr = xmid + lch_um / 2.0
        box = BBox(xl, self.array_box.yc - od_h / 2.0, xr, self.array_box.yc + od_h / 2.0, res)
        layout.add_rect('OD', box)
        layout.add_rect('PODE', box, purpose='dummy1')

        # draw OD edge objects
        self.draw_od_edge(layout, temp_db, grid, w, tech_constants)
