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


"""This module defines classes used to cache existing design masters
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from future.utils import with_metaclass

import sys
import os
import time
import importlib
import abc
from collections import OrderedDict

from typing import Sequence, Dict, Set, Any, Optional, Type, Callable

from ..io import readlines_iter, write_file, fix_string


class ClassImporter(object):
    """A class that dynamically imports Python class from a definition file.

    This class is used to import design modules to enable code reuse and design collaboration.

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


class DesignMaster(with_metaclass(abc.ABCMeta, object)):
    """A design master instance.

    This class represents a design master in the design database.

    Parameters
    ----------
    master_db : MasterDB
        the master database.
    lib_name : str
        the generated instance library name.
    params : Dict[str, Any]
        the parameters dictionary.
    used_names : Set[str]
        a set of already used cell names.
    """
    def __init__(self, master_db, lib_name, params, used_names):
        # type: (MasterDB, str, Dict[str, Any], Set[str]) -> None
        self._master_db = master_db
        self._lib_name = lib_name

        # set parameters
        params_info = self.get_params_info()
        default_params = self.get_default_param_values()
        self.params = {}
        if params_info is None:
            # compatibility with old schematics generators
            self._cell_name = None
            self._prelim_key = self.to_immutable_id((self._get_qualified_name(), params))
            self._key = None
        else:
            self.populate_params(params, params_info, default_params)

            # get unique cell name
            self._cell_name = self._get_unique_cell_name(used_names)
            self._prelim_key = self.compute_unique_key()
            self._key = self._prelim_key

        self.children = None
        self._finalized = False

    def populate_params(self, table, params_info, default_params, **kwargs):
        # type: (Dict[str, Any], Dict[str, str], Dict[str, Any], **kwargs) -> None
        """Fill params dictionary with values from table and default_params"""
        for key, desc in params_info.items():
            if key not in table:
                if key not in default_params:
                    raise ValueError('Parameter %s not specified.  Description:\n%s' % (key, desc))
                else:
                    self.params[key] = default_params[key]
            else:
                self.params[key] = table[key]

    @classmethod
    def to_immutable_id(cls, val):
        # type: (Any) -> Any
        """Convert the given object to an immutable type for use as keys in dictionary.
        """
        # python 2/3 compatibility: convert raw bytes to string
        val = fix_string(val)

        if val is None or isinstance(val, int) or isinstance(val, str) or isinstance(val, float):
            return val
        elif isinstance(val, list) or isinstance(val, tuple):
            return tuple((cls.to_immutable_id(item) for item in val))
        elif isinstance(val, dict):
            return tuple(((k, cls.to_immutable_id(val[k])) for k in sorted(val.keys())))
        elif hasattr(val, 'get_immutable_key') and callable(val.get_immutable_key):
            return val.get_immutable_key()
        else:
            raise Exception('Unrecognized value %s with type %s' % (str(val), type(val)))

    @classmethod
    @abc.abstractmethod
    def get_params_info(cls):
        # type: () -> Optional[Dict[str, str]]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return None

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Dict[str, Any]
            dictionary of default parameter values.
        """
        return {}

    @abc.abstractmethod
    def get_master_basename(self):
        # type: () -> str
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return ''

    @abc.abstractmethod
    def get_content(self, rename_fun):
        # type: (Callable[str, str]) -> Any
        """Returns the content of this master instance.

        Parameters
        ----------
        rename_fun : Callable[str, str]
            a function that renames design masters.

        Returns
        -------
        content : Any
            the master content data structure.
        """
        pass

    @property
    def master_db(self):
        # type: () -> MasterDB
        """Returns the database used to create design masters."""
        return self._master_db

    @property
    def lib_name(self):
        # type: () -> str
        """The master library name"""
        return self._lib_name

    @property
    def cell_name(self):
        # type: () -> str
        """The master cell name"""
        return self._cell_name

    @property
    def key(self):
        # type: () -> Optional[Any]
        """A unique key representing this master."""
        return self._key

    @property
    def finalized(self):
        # type: () -> bool
        """Returns True if this DesignMaster is finalized."""
        return self._finalized

    @property
    def prelim_key(self):
        # type: () -> Any
        """Returns a preliminary unique key.  For compatibility purposes with old schematic generators."""
        return self._prelim_key

    def _get_unique_cell_name(self, used_names):
        # type: (Set[str]) -> str
        """Returns a unique cell name.

        Parameters
        ----------
        used_names : Set[str]
            a set of used names.

        Returns
        -------
        cell_name : str
            the unique cell name.
        """
        counter = 0
        basename = self.get_master_basename()
        cell_name = basename
        while cell_name in used_names:
            counter += 1
            cell_name = '%s_%d' % (basename, counter)

        return cell_name

    def _get_qualified_name(self):
        # type: () -> str
        """Returns the qualified name of this class."""
        my_module = self.__class__.__module__
        if my_module is None or my_module == str.__class__.__module__:
            return self.__class__.__name__
        else:
            return my_module + '.' + self.__class__.__name__

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        self._finalized = True

    def compute_unique_key(self):
        # type: () -> Any
        """Returns a unique hashable object (usually tuple or string) that represents this instance.

        Returns
        -------
        unique_id : Any
            a hashable unique ID representing the given parameters.
        """
        return self.to_immutable_id((self._get_qualified_name(), self.params))


class MasterDB(with_metaclass(abc.ABCMeta, object)):
    """A database of existing design masters.

    This class keeps track of existing design masters and maintain design dependency hierarchy.

    Parameters
    ----------
    lib_name : str
        the cadence library to put all generated templates in.
    lib_defs : str
        generator library definition file path.  If empty, then assume user supplies Python class directly.
    name_prefix : str
        generated master name prefix.
    name_suffix : str
        generated master name suffix.
    """

    def __init__(self, lib_name, lib_defs='', name_prefix='', name_suffix=''):
        # type: (str, str, str, str) -> None

        self._lib_name = lib_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix

        self._used_cell_names = set()  # type: Set[str]
        self._importer = ClassImporter(lib_defs) if os.path.isfile(lib_defs) else None
        self._key_lookup = {}  # type: Dict[Any, Any]
        self._master_lookup = {}  # type: Dict[Any, DesignMaster]
        self._rename_dict = {}

    @abc.abstractmethod
    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[DesignMaster], str, Dict[str, Any], Set[str], **kwargs) -> DesignMaster
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[DesignMaster]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs
            optional arguments for the generator.

        Returns
        -------
        master : DesignMaster
            the non-finalized generated instance.
        """
        # noinspection PyTypeChecker
        return None

    @abc.abstractmethod
    def create_masters_in_db(self, lib_name, content_list, debug=False):
        # type: (str, Sequence[Any], bool) -> None
        """Create the masters in the design database.

        Parameters
        ----------
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        debug : bool
            True to print debug messages
        """
        pass

    @property
    def lib_name(self):
        # type: () -> str
        """Returns the layout library name."""
        return self._lib_name

    def format_cell_name(self, cell_name):
        # type: (str) -> str
        """Returns the formatted cell name.

        Parameters
        ----------
        cell_name : str
            the original cell name.

        Returns
        -------
        final_name : str
            the new cell name.
        """
        cell_name = self._rename_dict.get(cell_name, cell_name)
        return '%s%s%s' % (self._name_prefix, cell_name, self._name_suffix)

    def append_library(self, lib_name, lib_path):
        # type: (str, str) -> None
        """Adds a new library to the library definition file.

        Parameters
        ----------
        lib_name : str
            name of the library.
        lib_path : str
            path to this library.
        """
        if self._importer is None:
            raise ValueError('Cannot add generator library; library definition file not specified.')

        self._importer.append_library(lib_name, lib_path)

    def get_library_path(self, lib_name):
        # type: (str) -> Optional[str]
        """Returns the location of the given library.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        lib_path : Optional[str]
            the location of the library, or None if library not defined.
        """
        if self._importer is None:
            raise ValueError('Cannot get generator library path; library definition file not specified.')

        return self._importer.get_library_path(lib_name)

    def get_generator_class(self, lib_name, cell_name):
        # type: (str, str) -> Any
        """Returns the corresponding generator Python class.

        Parameters
        ----------
        lib_name : str
            template library name.
        cell_name : str
            generator cell name

        Returns
        -------
        temp_cls : Any
            the corresponding Python class.
        """
        if self._importer is None:
            raise ValueError('Cannot get generator class; library definition file not specified.')

        return self._importer.get_class(lib_name, cell_name)

    def new_master(self, lib_name='', cell_name='', params=None, gen_cls=None, debug=False, **kwargs):
        # type: (str, str, Optional[Dict[str, Any]], Optional[Type[DesignMaster]], bool, **kwargs) -> DesignMaster
        """Create a generator instance.

        Parameters
        ----------
        lib_name : str
            generator library name.
        cell_name : str
            generator name
        params : Optional[Dict[str, Any]]
            the parameter dictionary.
        gen_cls : Optional[Type[DesignMaster]]
            the generator class to instantiate.  Overrides lib_name and cell_name.
        debug : bool
            True to print debug messages.
        **kwargs
            optional arguments for generator.

        Returns
        -------
        master : DesignMaster
            the generator instance.
        """
        if params is None:
            params = {}

        if gen_cls is None:
            gen_cls = self.get_generator_class(lib_name, cell_name)

        master = self.create_master_instance(gen_cls, self._lib_name, params, self._used_cell_names, **kwargs)
        key = master.key

        if key is None:
            prelim_key = master.prelim_key
            if prelim_key in self._key_lookup:
                key = self._key_lookup[prelim_key]
                master = self._master_lookup[key]
                if debug:
                    print('master cached')
            else:
                if debug:
                    print('finalizing master')
                start = time.time()
                master.finalize()
                end = time.time()

                key = master.key
                self._key_lookup[prelim_key] = key
                if key in self._master_lookup:
                    master = self._master_lookup[key]
                else:
                    self._master_lookup[key] = master
                self._used_cell_names.add(master.cell_name)
                if debug:
                    print('finalizing master took %.4g seconds' % (end - start))
        else:
            if key in self._master_lookup:
                master = self._master_lookup[key]
                if debug:
                    print('master cached')
            else:
                if debug:
                    print('finalizing master')
                start = time.time()
                master.finalize()
                end = time.time()
                self._master_lookup[key] = master
                self._used_cell_names.add(master.cell_name)
                if debug:
                    print('finalizing master took %.4g seconds' % (end - start))

        return master

    def instantiate_masters(self, master_list, name_list=None, debug=False):
        # type: (Sequence[DesignMaster], Optional[Sequence[str]], bool) -> None
        """create all given masters in the database.

        Parameters
        ----------
        master_list : Sequence[DesignMaster]
            list of masters to instantiate.
        name_list : Optional[Sequence[str]]
            list of master cell names.  If not given, default names will be used.
        debug : bool
            True to print debugging messages
        """
        if name_list is None:
            name_list = [None] * len(master_list)
        else:
            if len(name_list) != len(master_list):
                raise ValueError("Master list and name list length mismatch.")

        # configure renaming dictionary
        self._rename_dict.clear()
        for master, name in zip(master_list, name_list):
            if name is not None and name != master.cell_name:
                if name in self._used_cell_names:
                    raise ValueError('master cell name = %s is already used.' % name)
                self._rename_dict[master.cell_name] = name

        print(self._rename_dict)

        if debug:
            print('Retrieving master contents')

        # use ordered dict so that children are created before parents.
        info_dict = OrderedDict()  # type: Dict[str, DesignMaster]
        start = time.time()
        for master, top_name in zip(master_list, name_list):
            self._instantiate_master_helper(info_dict, master)
        end = time.time()

        content_list = [master.get_content(self.format_cell_name) for master in info_dict.values()]

        if debug:
            print('master content retrieval took %.4g seconds' % (end - start))

        self.create_masters_in_db(self._lib_name, content_list, debug=debug)

    def _instantiate_master_helper(self, info_dict, master):
        # type: (Dict[str, DesignMaster], DesignMaster) -> None
        """Helper method for batch_layout().

        Parameters
        ----------
        info_dict : Dict[str, DesignMaster]
            dictionary from existing master cell name to master objects.
        master : DesignMaster
            the master object to create.
        """
        # get template master for all children
        for master_key in master.children:
            child_temp = self._master_lookup[master_key]
            if child_temp.cell_name not in info_dict:
                self._instantiate_master_helper(info_dict, child_temp)

        # get template master for this cell.
        info_dict[master.cell_name] = self._master_lookup[master.key]
