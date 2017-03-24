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

from typing import Union, Optional


class BinaryIterator(object):
    """A class that performs binary search over integer or float range.

    This class simplifies binary search algorithm writing, whether it be bounded or unbounded.

    Parameters
    ----------
    low : Union[float, int]
        the lower index/bound (inclusive).
    high : Optional[Union[float, int]]
        the higher index (exclusive).  None for unbounded binary search.
    step : Union[float, int]
        the step size.
    is_float : bool
        True if this BinaryIterator is used for float values.
    """

    def __init__(self, low, high, step=1, is_float=False):
        # type: (Union[float, int], Optional[Union[float, int]], Union[float, int], bool) -> None
        self.low = low
        self.high = high
        self.step = step
        self.is_float = is_float
        if high is not None:
            if is_float:
                self.current = (low + high) / 2
            else:
                self.current = (low + high) // 2
        else:
            self.current = low
        self.save_marker = None

    def has_next(self):
        # type: () -> bool
        """returns True if this iterator is not finished yet."""
        return self.high is None or self.low + self.step <= self.high

    def get_next(self):
        # type: () -> Union[float, int]
        """Returns the next value to look at."""
        return self.current

    def up(self):
        # type: () -> None
        """Increment this iterator."""
        if self.is_float:
            self.low = self.current
        else:
            self.low = self.current + self.step
        if self.high is not None:
            if self.is_float:
                self.current = (self.low + self.high) / 2
            else:
                self.current = (self.low + self.high) // 2
        else:
            if self.current > 0:
                self.current *= 2
            else:
                self.current = self.step

    def down(self):
        # type: () -> None
        """Decrement this iterator."""
        self.high = self.current
        if self.is_float:
            self.current = (self.low + self.high) / 2
        else:
            self.current = (self.low + self.high) // 2

    def save(self):
        # type: () -> None
        """Save the current index"""
        self.save_marker = self.current

    def get_last_save(self):
        # type: () -> Union[float, int]
        """Returns the last saved index."""
        return self.save_marker
