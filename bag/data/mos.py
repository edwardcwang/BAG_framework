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

"""This module defines classes for computing DC operating point.
"""

from typing import Dict

import numpy as np


def mos_y_to_ss(sim_data, char_freq, fg, ibias):
    # type: (Dict[str, np.ndarray], float, int, np.ndarray) -> Dict[str, np.ndarray]
    """Convert transistor Y parameters to small-signal parameters.

    This function computes MOSFET small signal parameters from 3-port
    Y parameter measurements done on gate, drain and source, with body
    bias fixed.  This functions fits the Y parameter to a capcitor-only
    small signal model using least-mean-square error.

    Parameters
    ----------
    sim_data : Dict[str, np.ndarray]
        A dictionary of Y parameters values stored as complex numpy arrays.
    char_freq : float
        the frequency Y parameters are measured at.
    fg : int
        number of transistor fingers used for the Y parameter measurement.
    ibias : np.ndarray
        the DC bias current of the transistor.  Always positive.

    Returns
    -------
    ss_dict : Dict[str, np.ndarray]
        A dictionary of small signal parameter values stored as numpy
        arrays.  These values are normalized to 1-finger transistor.
    """
    w = 2 * np.pi * char_freq

    gm = (sim_data['y21'].real - sim_data['y31'].real) / 2.0
    gds = (sim_data['y22'].real - sim_data['y32'].real) / 2.0
    gb = (sim_data['y33'].real - sim_data['y23'].real) / 2.0 - gm - gds

    cgd = -0.5 / w * (sim_data['y12'].imag + sim_data['y21'].imag)
    cgs = -0.5 / w * (sim_data['y13'].imag + sim_data['y31'].imag)
    cds = -0.5 / w * (sim_data['y23'].imag + sim_data['y32'].imag)
    cgb = sim_data['y11'].imag / w - cgd - cgs
    cdb = sim_data['y22'].imag / w - cds - cgd
    csb = sim_data['y33'].imag / w - cgs - cds

    ibias = ibias / fg  # type: np.ndarray
    gm = gm / fg  # type: np.ndarray
    gds = gds / fg  # type: np.ndarray
    gb = gb / fg  # type: np.ndarray
    cgd = cgd / fg  # type: np.ndarray
    cgs = cgs / fg  # type: np.ndarray
    cds = cds / fg  # type: np.ndarray
    cgb = cgb / fg  # type: np.ndarray
    cdb = cdb / fg  # type: np.ndarray
    csb = csb / fg  # type: np.ndarray

    return dict(
        ibias=ibias,
        gm=gm,
        gds=gds,
        gb=gb,
        cgd=cgd,
        cgs=cgs,
        cds=cds,
        cgb=cgb,
        cdb=cdb,
        csb=csb,
    )
