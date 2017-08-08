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
