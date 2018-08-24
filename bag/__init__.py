# -*- coding: utf-8 -*-

"""This is the bag root package.
"""

import signal

__all__ = []

# make sure that SIGINT will always be catched by python.
signal.signal(signal.SIGINT, signal.default_int_handler)
