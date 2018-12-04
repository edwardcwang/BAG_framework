# -*- coding: utf-8 -*-

"""This module contains utility classes used for layout
"""

# noinspection PyUnresolvedReferences
from pybag.core import BBox, BBoxArray, BBoxCollection


class PortSpec(object):
    """Specification of a port.

    Parameters
    ----------
    ntr : int
        number of tracks the port should occupy
    idc : float
        DC current the port should support, in Amperes.
    """

    def __init__(self, ntr, idc):
        self._ntr = ntr
        self._idc = idc

    @property
    def ntr(self):
        """minimum number of tracks the port should occupy"""
        return self._ntr

    @property
    def idc(self):
        """minimum DC current the port should support, in Amperes"""
        return self._idc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        fmt_str = '%s(%d, %.4g)'
        return fmt_str % (self.__class__.__name__, self._ntr, self._idc)


class Pin(object):
    """A layout pin.

    Multiple pins can share the same terminal name.

    Parameters
    ----------
    pin_name : str
        the pin label.
    term_name : str
        the terminal name.
    layer : str
        the pin layer name.
    bbox : bag.layout.util.BBox
        the pin bounding box.
    """

    def __init__(self, pin_name, term_name, layer, bbox):
        if not bbox.is_physical():
            raise Exception('Non-physical pin bounding box: %s' % bbox)

        self._pin_name = pin_name
        self._term_name = term_name
        self._layer = layer
        self._bbox = bbox

    @property
    def pin_name(self):
        """the pin label."""
        return self._pin_name

    @property
    def term_name(self):
        """the terminal name."""
        return self._term_name

    @property
    def layer(self):
        """the pin layer name"""
        return self._layer

    @property
    def bbox(self):
        """the pin bounding box."""
        return self._bbox

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '%s(%s, %s, %s, %s)' % (self.__class__.__name__, self._pin_name,
                                       self._term_name, self._layer, self._bbox)
