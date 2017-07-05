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


"""This module defines Database, a class that keeps track of design module definitions and instantiates them.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from bag.util.libimport import ClassImporter

from .module import GenericModule


class Database(object):
    """A database of all design modules.

    This class is responsible for keeping track of design module libraries, and
    creating instances of design modules.

    Parameters
    ----------
    lib_defs : str
        path to the design library definition file.
    tech_info : :class:`bag.layout.core.TechInfo`
        the :class:`~bag.layout.core.TechInfo` instance.
    sch_exc_libs : List[str]
        list of libraries that are excluded from import.
    """

    def __init__(self, lib_defs, tech_info, sch_exc_libs):
        self._importer = ClassImporter(lib_defs)
        self._tech_info = tech_info
        self._exc_libs = set(sch_exc_libs)

    @property
    def tech_info(self):
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def append_library(self, lib_name, lib_path):
        """Adds a new library to the library definition file.

        Parameters
        ----------
        lib_name : str
            name of the library.
        lib_path : str
            path to this library.
        """
        self._importer.append_library(lib_name, lib_path)

    def get_library_path(self, lib_name):
        """Returns the location of the given library.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        lib_path : str or None
            the location of the library, or None if library not defined.
        """
        return self._importer.get_library_path(lib_name)

    def make_design_module(self, lib_name, cell_name, parent=None, prj=None, **kwargs):
        """Create a new design module instance.

        Parameters
        ----------
        lib_name : str
            design module library name.
        cell_name : str
            design module cell name
        parent : :class:`bag.design.Module` or None.
            the parent design module instance, or None for top level design.
        prj : :class:`bag.BagProject` or None
            the BagProject instance.  Used to implement design.
        **kwargs :
            optional parameters.

        Returns
        -------
        module : :class:`bag.design.Module`
            a new :class:`~bag.design.Module` instance.
        """
        if lib_name in self._exc_libs:
            return GenericModule(self, lib_name, cell_name, parent=parent, prj=prj, **kwargs)
        else:
            cls = self._importer.get_class(lib_name, cell_name)
            return cls(self, parent=parent, prj=prj, **kwargs)
