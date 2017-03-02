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

"""This module defines functions useful for digital verification/postprocessing.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from .core import Waveform


def de_bruijn(n, symbols=None, ):
    """Returns a De Bruijn sequence with subsequence of length n.

    a De Bruijn sequence with subsequence of length n is a sequence such that
    all possible subsequences of length n appear exactly once somewhere in the
    sequence.  This method is useful for simulating the worst case eye diagram
    given finite impulse response.

    Parameters
    ----------
    n : int
        length of the subsequence.
    symbols : list[flaot] or None
        the list of symbols.  If None, defaults to [0.0, 1.0].

    Returns
    -------
    seq : list[float]
        the de bruijn sequence.
    """
    symbols = symbols or [0.0, 1.0]
    k = len(symbols)

    a = [0] * (k * n)
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return [symbols[i] for i in sequence]


def get_flop_timing(tvec, d, q, clk, ttol, data_thres=0.5,
                    clk_thres=0.5, tstart=0.0, clk_edge='rising', tag=None, invert=False):
    """Calculate flop timing parameters given the associated waveforms.

    This function performs the following steps:

    1. find all valid clock edges.  Compute period of the clock (clock waveform
       must be periodic).
    
    2. For each valid clock edge:

        A. Check if the input changes in the previous cycle.  If so, compute tsetup.
           Otherwise, tsetup = tperiod.
    
        B. Check if input changes in the current cycle.  If so, compute thold.
           Otherwise, thold = tperiod.
  
        C. Check that output transition at most once and that output = input.
           Otherwise, record an error.

        D. record the output data polarity.

    3. For each output data polarity, compute the minimum tsetup and thold and any
       errors.  Return summary as a dictionary.

    
    The output is a dictionary with keys 'setup', 'hold', 'delay', and 'errors'.
    the setup/hold/delay entries contains 2-element tuples describing the worst
    setup/hold/delay time.  The first element is the setup/hold/delay time, and
    the second element is the clock edge time at which it occurs.  The errors field
    stores all clock edge times at which an error occurs.


    Parameters
    ----------
    tvec : numpy.ndarray
        the time data.
    d : numpy.ndarray
        the input data.
    q : numpy.ndarray
        the output data.
    clk : numpy.ndarray
        the clock data.
    ttol : float
        time resolution.
    data_thres : float
        the data threshold.
    clk_thres : float
        the clock threshold.
    tstart : float
        ignore data points before tstart.
    clk_edge : str
        the clock edge type.  Valid values are "rising", "falling", or "both".
    tag : obj
        an identifier tag to append to results.
    invert : bool
        if True, the flop output is inverted from the data.

    Returns
    -------
    data : dict[str, any]
        A dictionary describing the worst setup/hold/delay and errors, if any.
    """
    d_wv = Waveform(tvec, d, ttol)
    clk_wv = Waveform(tvec, clk, ttol)
    q_wv = Waveform(tvec, q, ttol)
    tend = tvec[-1]

    # get all clock sampling times and clock period
    samp_times = clk_wv.get_all_crossings(clk_thres, start=tstart, edge=clk_edge)
    tper = (samp_times[-1] - samp_times[0]) / (len(samp_times) - 1)
    # ignore last clock cycle if it's not a full cycle.
    if samp_times[-1] + tper > tend:
        samp_times = samp_times[:-1]

    # compute setup/hold/error for each clock period
    data = {'setup': (tper, -1), 'hold': (tper, -1), 'delay': (0.0, -1), 'errors': []}
    for t in samp_times:
        d_prev = d_wv.get_all_crossings(data_thres, start=t - tper, stop=t, edge='both')
        d_cur = d_wv.get_all_crossings(data_thres, start=t, stop=t + tper, edge='both')
        q_cur = q_wv.get_all_crossings(data_thres, start=t, stop=t + tper, edge='both')
        d_val = d_wv(t) > data_thres
        q_val = q_wv(t + tper) > data_thres

        # calculate setup/hold/delay
        tsetup = t - d_prev[-1] if d_prev else tper
        thold = d_cur[0] - t if d_cur else tper
        tdelay = q_cur[0] - t if q_cur else 0.0

        # check if flop has error
        error = (invert != (q_val != d_val)) or (len(q_cur) > 1)

        # record results
        if tsetup < data['setup'][0]:
            data['setup'] = (tsetup, t)
        if thold < data['hold'][0]:
            data['hold'] = (thold, t)
        if tdelay > data['delay'][0]:
            data['delay'] = (tdelay, t)
        if error:
            data['errors'].append(t)

    if tag is not None:
        data['setup'] += (tag, )
        data['hold'] += (tag, )
        data['delay'] += (tag, )
        data['errors'] = [(t, tag) for t in data['errors']]

    return data
