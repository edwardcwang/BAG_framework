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


class BinaryIterator(object):
    """A class that returns the next index to evaluate for binary search.

    This class simplifies binary search algorithm writing, whether it be bounded or unbounded.

    Parameters
    ----------
    low : int
        the lower index (inclusive).
    high : int or None
        the higher index (exclusive).  None for unbounded binary search.
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high
        if high is not None:
            self.current = (low + high) // 2
        else:
            self.current = low
        self.save_index = None

    def has_next(self):
        """returns True if this iterator is not finished yet."""
        return self.high is None or self.low < self.high

    def get_next(self):
        """Returns the next index to look at."""
        if self.high is None or self.low < self.high:
            return self.current
        return None

    def up(self):
        """Increment this iterator."""
        self.low = self.current + 1
        if self.high is not None:
            self.current = (self.low + self.high) // 2
        else:
            if self.current > 0:
                self.current *= 2
            else:
                self.current = 1

    def down(self):
        """Decrement this iterator."""
        self.high = self.current
        self.current = (self.low + self.high) // 2

    def save(self):
        """Save the current index"""
        self.save_index = self.current

    def get_last_save(self):
        """Returns the saved index."""
        return self.save_index
