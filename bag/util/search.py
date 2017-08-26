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


"""This module provides search related utilities.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Optional

import numpy as np


class BinaryIterator(object):
    """A class that performs binary search over integers.

    This class supports both bounded or unbounded binary search, and
    you can also specify a step size.

    Parameters
    ----------
    low : int
        the lower bound (inclusive).
    high : Optional[int]
        the upper bound (exclusive).  None for unbounded binary search.
    step : int
        the step size.  All return values will be low + N * step
    """

    def __init__(self, low, high, step=1):
        # type: (int, Optional[int], int) -> None

        if not isinstance(low, int) or not isinstance(step, int):
            raise ValueError('low and step must be integers.')

        self._offset = low
        self._step = step
        self._low = 0

        if high is not None:
            if not isinstance(high, int):
                raise ValueError('high must be None or integer.')

            nmax = (high - low) // step
            if low + step * nmax < high:
                nmax += 1
            self._high = nmax
            self._current = (self._low + self._high) // 2
        else:
            self._high = None
            self._current = self._low

        self._save_marker = None

    def has_next(self):
        # type: () -> bool
        """returns True if this iterator is not finished yet."""
        return self._high is None or self._low < self._high

    def get_next(self):
        # type: () -> int
        """Returns the next value to look at."""
        return self._current * self._step + self._offset

    def up(self):
        # type: () -> None
        """Increment this iterator."""
        self._low = self._current + 1

        if self._high is not None:
            self._current = (self._low + self._high) // 2
        else:
            if self._current > 0:
                self._current *= 2
            else:
                self._current = 1

    def down(self):
        # type: () -> None
        """Decrement this iterator."""
        self._high = self._current
        self._current = (self._low + self._high) // 2

    def save(self):
        # type: () -> None
        """Save the current index"""
        self._save_marker = self._current

    def get_last_save(self):
        # type: () -> Optional[int]
        """Returns the last saved index."""
        if self._save_marker is None:
            return None
        return self._save_marker * self._step + self._offset


class FloatBinaryIterator(object):
    """A class that performs binary search over floating point numbers.

    This class supports both bounded or unbounded binary search, and terminates
    when we can guarantee the given error tolerance.

    Parameters
    ----------
    low : float
        the lower bound.
    high : Optional[float]
        the upper bound.  None for unbounded binary search.
    tol : float
        we will guarantee that the final solution will be within this
        tolerance.
    search_step : float
        for unbounded binary search, this is the initial step size when
        searching for upper bound.
    """

    def __init__(self, low, high, tol=1.0, search_step=1.0):
        # type: (float, Optional[float], float) -> None
        self._offset = low
        self._tol = tol
        self._low = 0
        self._search_step = search_step

        if high is not None:
            self._high = high - low
            self._current = self._high / 2
        else:
            self._high = None
            self._current = 0

        self._save_marker = None

    def has_next(self):
        # type: () -> bool
        """returns True if this iterator is not finished yet."""
        return self._high is None or self._low + 2 * self._tol < self._high

    def get_next(self):
        # type: () -> float
        """Returns the next value to look at."""
        return self._current + self._offset

    def up(self):
        # type: () -> None
        """Increment this iterator."""
        self._low = self._current

        if self._high is not None:
            self._current = (self._low + self._high) / 2
        else:
            if self._current != 0:
                self._current *= 2
            else:
                self._current = self._search_step

    def down(self):
        # type: () -> None
        """Decrement this iterator."""
        self._high = self._current
        self._current = (self._low + self._high) / 2

    def save(self):
        # type: () -> None
        """Save the current index"""
        self._save_marker = self._current

    def get_last_save(self):
        # type: () -> Optional[float]
        """Returns the last saved index."""
        if self._save_marker is None:
            return None
        return self._save_marker + self._offset


def minimize_brent_discrete(func, a, x, b, maxiter=500):
    """Find minimum of a discrete function given interval.

    This is a modification of the Brent's method so function arguments are
    quantized to integers.

    Parameters
    ----------
    func : callable
        function to minimize.  Must take a single integer and returns a scalar value.
    a : int
        lower bound of the interval.
    x : int
        a point in the interval for which f(x) < f(a), f(b).
    b: int
        upper bound of the interval.
    maxiter : int
        maximum number of iterations.

    Returns
    -------
    xmin : Optional[int]
        the minimum X solution, None if solution cannot be found within maxiter iterations.
    """
    # set up for optimization
    tol = 1e-8
    _mintol = 1e-2
    _cg = 0.3819660

    w = v = x
    fw = fv = fx = func(x)
    deltax = 0.0
    iter_cnt = 0
    rat = 0
    while iter_cnt < maxiter and (x > a + 1 or b > x + 1):
        tol1 = tol * np.abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # XXX In the first iteration, rat is only bound in the true case
        # of this conditional. This used to cause an UnboundLocalError
        # (gh-4140). It should be set before the if (but to what?).
        if np.abs(deltax) <= tol1:
            # noinspection PyTypeChecker
            if x >= xmid:
                deltax = a - x       # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:                              # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2        # if parabolic step is useful.
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid >= x:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                # noinspection PyTypeChecker
                if x >= xmid:
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if np.abs(rat) < 1:            # update by at least tol1
            if rat >= 0:
                u = max(int(round(x + tol1)), x + 1)
            else:
                u = min(int(round(x - tol1)), x - 1)
        else:
            u = int(round(x + rat))

        # if rounding to integer causes u to be equal to boundaries,
        # we change our picks of u to narrow the range as much as
        # possible.
        if u == b:
            if b > x + 1:
                u = (x + b) // 2
            else:
                u = int(round(x - (x - a) * _cg))
        elif u == a:
            if a < x - 1:
                u = (a + x) // 2
            else:
                u = int(round(x + (b - x) * _cg))

        fu = func(u)      # calculate new output value

        if fu > fx:                 # if it's bigger than current
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        iter_cnt += 1

    if x > a + 1 or b > x + 1:
        return None
    return x
