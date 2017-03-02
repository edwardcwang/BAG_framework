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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'gm.yaml'))


# noinspection PyPep8Naming
class demo_templates__gm(Module):
    """Module for library demo_templates cell gm.

    Fill in high level description here.
    """
    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)
        self.parameters['lch'] = None
        self.parameters['win'] = None
        self.parameters['wt'] = None
        self.parameters['nf'] = None
        self.parameters['ndum_extra'] = None
        self.parameters['input_intent'] = None
        self.parameters['tail_intent'] = None

    def design(self, lch, win, wt, nf, ndum_extra, input_intent, tail_intent):
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        self.parameters['lch'] = lch
        self.parameters['win'] = win
        self.parameters['wt'] = wt
        self.parameters['nf'] = nf
        self.parameters['ndum_extra'] = ndum_extra
        self.parameters['input_intent'] = input_intent
        self.parameters['tail_intent'] = tail_intent

        self.instances['XP'].design(w=win, l=lch, nf=nf, intent=input_intent)
        self.instances['XN'].design(w=win, l=lch, nf=nf, intent=input_intent)
        self.instances['XT'].design(w=wt, l=lch, nf=nf*2, intent=tail_intent)

        self.instances['XD1'].design(w=win, l=lch, nf=4, intent=input_intent)
        self.instances['XD3'].design(w=win, l=lch, nf=ndum_extra*2, intent=input_intent)
        self.instances['XD2'].design(w=wt, l=lch, nf=4, intent=tail_intent)
        self.instances['XD4'].design(w=wt, l=lch, nf=ndum_extra*2, intent=tail_intent)

    def get_layout_params(self, **kwargs):
        """Returns a dictionary with layout parameters.

        This method computes the layout parameters used to generate implementation's
        layout.  Subclasses should override this method if you need to run post-extraction
        layout.

        Parameters
        ----------
        kwargs :
            any extra parameters you need to generate the layout parameters dictionary.
            Usually you specify layout-specific parameters here, like metal layers of
            input/output, customizable wire sizes, and so on.

        Returns
        -------
        params : dict[str, any]
            the layout parameters dictionary.
        """
        params = dict(
            lch=self.parameters['lch'] * 1e6,
            win=self.parameters['win'] * 1e6,
            wt=self.parameters['wt'] * 1e6,
            wsw=0.0,
            wen=0.0,
            wb=0.0,
            nf=self.parameters['nf'],
            nduml=self.parameters['ndum_extra']+1,
            ndumr=self.parameters['ndum_extra']+1,
            nmos_h=5.2,
            input_intent=self.parameters['input_intent'],
            tail_intent=self.parameters['tail_intent'],
            )
        
        params.update(kwargs)

        return params

    def get_layout_pin_mapping(self):
        """Returns the layout pin mapping dictionary.

        This method returns a dictionary used to rename the layout pins, in case they are different
        than the schematic pins.

        Returns
        -------
        pin_mapping : dict[str, str]
            a dictionary from layout pin names to schematic pin names.
        """
        return dict(CKT='BIAS')
