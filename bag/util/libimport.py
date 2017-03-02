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


"""This module defines ClassImporter, a class that dynamically import packages from a definition file.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import sys
import os
import importlib

from ..io import readlines_iter, write_file


class ClassImporter(object):
    """A class that dynamically imports Python class from a definition file.

    This class is used to import design modules and layout templates in other locations to enable
    code reuse and design collaboration.

    Parameters
    ----------
    lib_defs : str
        path to the design library definition file.
    """
    def __init__(self, lib_defs):
        """Create a new design database instance.
        """
        lib_defs = os.path.abspath(lib_defs)
        if not os.path.exists(lib_defs):
            raise Exception("design library definition file %s not found" % lib_defs)

        self.lib_defs = lib_defs
        self.libraries = {}
        for line in readlines_iter(lib_defs):
            line = line.strip()
            # ignore comments and empty lines
            if line and not line.startswith('#'):
                lib_name, lib_path = line.split()
                lib_path = os.path.abspath(os.path.expandvars(lib_path))
                check_path = os.path.join(lib_path, lib_name)
                if not os.path.exists(check_path):
                    raise Exception('Library %s not found.' % check_path)
                # make sure every library is on python path, so we can import it.
                if lib_path not in sys.path:
                    sys.path.append(lib_path)
                self.libraries[lib_name] = lib_path

    def append_library(self, lib_name, lib_path):
        """Adds a new library to the library definition file.

        Parameters
        ----------
        lib_name : str
            name of the library.
        lib_path : str
            path to this library.
        """
        if lib_name not in self.libraries:
            lib_path = os.path.abspath(lib_path)
            self.libraries[lib_name] = lib_path
            write_file(self.lib_defs, '%s %s\n' % (lib_name, lib_path), append=True)

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
        return self.libraries.get(lib_name, None)

    def get_class(self, lib_name, cell_name):
        """Returns the Python class with the given library and cell name.

        Parameters
        ----------
        lib_name : str
            design module library name.
        cell_name : str
            design module cell name

        Returns
        -------
        cls : class
            the corresponding Python class.
        """

        if lib_name not in self.libraries:
            raise Exception("Library %s not listed in definition file %s" % (lib_name, self.lib_defs))

        module_name = '%s.%s' % (lib_name, cell_name)
        module_cls = '%s__%s' % (lib_name, cell_name)

        lib_package = importlib.import_module(lib_name)
        cell_package = importlib.import_module(module_name, package=lib_package)
        return getattr(cell_package, module_cls)
