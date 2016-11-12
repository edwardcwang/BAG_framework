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


from .amplifier import AmplifierBase


class SerdesBase(AmplifierBase):
    """Subclass of AmplifierBase that draws serdes circuits.

    Parameters
    ----------
    grid : :class:`bag.layout.routing.RoutingGrid`
            the :class:`~bag.layout.routing.RoutingGrid` instance.
    lib_name : str
        the layout library name.
    params : dict
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    mos_cls : class
        the transistor template class.
    sub_cls : class
        the substrate template class.
    """

    def __init__(self, grid, lib_name, params, used_names,
                 mos_cls, sub_cls, mconn_cls, sep_cls, dum_cls):
        AmplifierBase.__init__(self, grid, lib_name, params, used_names,
                               mos_cls, sub_cls, mconn_cls, sep_cls, dum_cls)

    def draw_dynamic_latch(self, layout, temp_db, po_idx, nduml, ndumr, fg_load,
                           fg_in, fg_tail, fg_cas=0, fg_sw=0, fg_en=0):
        """Draw a dynamic latch.

        Parameters
        ----------
        layout : :class:`bag.layout.core.BagLayout`
            the BagLayout instance.
        temp_db : :class:`bag.layout.template.TemplateDB`
            the TemplateDB instance.  Used to create new templates.
        po_idx : int
            the poly index.  0 is the left-most poly.
        nduml : int
            number of left dummies.
        ndumr : int
            number of right dummies.
        fg_load : int
            load fingers per side.
        fg_in : int
            input fingers per side.
        fg_tail : int
            tail fingers per side.
        fg_cas : int
            cascode fingers per side.  0 to disable.
        fg_sw : int
            tail switch fingers per side.  0 to disable.
        fg_en : int
            enable fingers per side.  0 to disable.
        """
        pass
