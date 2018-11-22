# -*- coding: utf-8 -*-

"""This module defines classes used to cache existing design masters
"""

from typing import Sequence, Dict, Set, Any, Optional, TypeVar, Type, Callable, Iterator

import abc
import time
import numbers
from collections import OrderedDict

from pybag.enum import DesignOutput

from ..io import fix_string
from .search import get_new_name


class DesignMaster(abc.ABC):
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
    **kwargs :
        optional parameters.

    Attributes
    ----------
    params : Dict[str, Any]
        the parameters dictionary.
    """

    def __init__(self, master_db, lib_name, params, used_names, **kwargs):
        # type: (MasterDB, str, Dict[str, Any], Set[str], **Any) -> None
        self._master_db = master_db
        self._lib_name = lib_name
        self._used_names = used_names
        self._children = set()
        self._finalized = False

        # set parameters
        params_info = self.get_params_info()
        default_params = self.get_default_param_values()
        self.params = {}  # type: Dict[str, Any]
        self.populate_params(params, params_info, default_params, **kwargs)

        # get unique cell name
        self._cell_name = get_new_name(self.get_master_basename(), self._used_names)
        self._key = self.compute_unique_key()

    def populate_params(self, table, params_info, default_params, **kwargs):
        # type: (Dict[str, Any], Dict[str, str], Dict[str, Any], **Any) -> None
        """Fill params dictionary with values from table and default_params"""
        hidden_params = kwargs.get('hidden_params', {})

        for key, desc in params_info.items():
            if key not in table:
                if key not in default_params:
                    raise ValueError('Parameter %s not specified.  Description:\n%s' % (key, desc))
                else:
                    self.params[key] = default_params[key]
            else:
                self.params[key] = table[key]

        # add hidden parameters
        for name, value in hidden_params.items():
            self.params[name] = table.get(name, value)

    @classmethod
    def to_immutable_id(cls, val):
        # type: (Any) -> Any
        """Convert the given object to an immutable type for use as keys in dictionary.
        """
        # python 2/3 compatibility: convert raw bytes to string
        val = fix_string(val)

        if val is None or isinstance(val, numbers.Number) or isinstance(val, str):
            return val
        elif isinstance(val, list) or isinstance(val, tuple):
            return tuple((cls.to_immutable_id(item) for item in val))
        elif isinstance(val, dict):
            return tuple(((k, cls.to_immutable_id(val[k])) for k in sorted(val.keys())))
        elif isinstance(val, set):
            return tuple((k for k in sorted(val)))
        elif hasattr(val, 'get_immutable_key') and callable(val.get_immutable_key):
            return val.get_immutable_key()
        else:
            raise Exception('Unrecognized value %s with type %s' % (str(val), type(val)))

    @classmethod
    @abc.abstractmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter names to descriptions.
        """
        return {}

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
        # type: (Callable[[str], str]) -> Any
        """Returns the content of this master instance.

        Parameters
        ----------
        rename_fun : Callable[[str], str]
            a function that renames design masters.

        Returns
        -------
        content : Any
            the master content data structure.
        """
        return None

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

    def add_child_key(self, child_key):
        # type: (object) -> None
        """Registers the given child key."""
        self._children.add(child_key)

    def children(self):
        # type: () -> Iterator[object]
        """Iterate over all children's key."""
        return iter(self._children)


MasterType = TypeVar('MasterType', bound=DesignMaster)


class MasterDB(abc.ABC):
    """A database of existing design masters.

    This class keeps track of existing design masters and maintain design dependency hierarchy.

    Parameters
    ----------
    lib_name : str
        the library to put all generated templates in.
    name_prefix : str
        generated master name prefix.
    name_suffix : str
        generated master name suffix.
    """

    def __init__(self, lib_name, name_prefix='', name_suffix=''):
        # type: (str, str, str) -> None

        self._lib_name = lib_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix

        self._used_cell_names = set()  # type: Set[str]
        self._key_lookup = {}  # type: Dict[Any, Any]
        self._master_lookup = {}  # type: Dict[Any, DesignMaster]
        self._rename_dict = {}  # type: Dict[str, str]

    @abc.abstractmethod
    def create_masters_in_db(self, output, lib_name, content_list, debug=False, **kwargs):
        # type: (DesignOutput, str, Sequence[Any], bool, **Any) -> None
        """Create the masters in the design database.

        Parameters
        ----------
        output : DesignOutput
            the output type.
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        debug : bool
            True to print debug messages
        **kwargs :  Any
            parameters associated with the given output type.
        """
        pass

    @property
    def lib_name(self):
        # type: () -> str
        """Returns the master library name."""
        return self._lib_name

    @property
    def cell_prefix(self):
        # type: () -> str
        """Returns the cell name prefix."""
        return self._name_prefix

    @cell_prefix.setter
    def cell_prefix(self, new_val):
        # type: (str) -> None
        """Change the cell name prefix."""
        self._name_prefix = new_val

    @property
    def cell_suffix(self):
        # type: () -> str
        """Returns the cell name suffix."""
        return self._name_suffix

    @property
    def used_cell_names(self):
        # type: () -> Set[str]
        return self._used_cell_names

    @cell_suffix.setter
    def cell_suffix(self, new_val):
        # type: (str) -> None
        """Change the cell name suffix."""
        self._name_suffix = new_val

    def clear(self):
        """Clear all existing schematic masters."""
        self._key_lookup.clear()
        self._master_lookup.clear()
        self._rename_dict.clear()

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[MasterType], str, Dict[str, Any], Set[str], **Any) -> MasterType
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[MasterType]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs : Any
            optional arguments for the generator.

        Returns
        -------
        master : MasterType
            the non-finalized generated instance.
        """
        return gen_cls(self, lib_name, params, used_cell_names, **kwargs)

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

    def new_master(self,  # type: MasterDB
                   gen_cls,  # type: Type[MasterType]
                   params=None,  # type: Optional[Dict[str, Any]]
                   debug=False,  # type: bool
                   **kwargs):
        # type: (...) -> MasterType
        """Create a generator instance.

        Parameters
        ----------
        gen_cls : Type[MasterType]
            the generator class to instantiate.  Overrides lib_name and cell_name.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.
        debug : bool
            True to print debug messages.
        **kwargs :
            optional arguments for generator.

        Returns
        -------
        master : MasterType
            the generator instance.
        """
        if params is None:
            params = {}

        master = self.create_master_instance(gen_cls, self._lib_name, params,
                                             self._used_cell_names, **kwargs)
        key = master.key
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
            self.register_master(key, master)
            if debug:
                print('finalizing master took %.4g seconds' % (end - start))

        return master

    def register_master(self, key, master):
        self._master_lookup[key] = master
        self._used_cell_names.add(master.cell_name)

    def instantiate_master(self, output, master, top_cell_name=None, **kwargs):
        # type: (DesignOutput, DesignMaster, Optional[str], **Any) -> None
        """Instantiate the given master.

        Parameters
        ----------
        output : DesignOutput
            the design output type.
        master : DesignMaster
            the :class:`~bag.layout.template.TemplateBase` to instantiate.
        top_cell_name : Optional[str]
            name of the top level cell.  If None, a default name is used.
        **kwargs : Any
            optional arguments for batch_output().
        """
        self.batch_output(output, [master], name_list=[top_cell_name], **kwargs)

    def batch_output(self,
                     output,  # type: DesignOutput
                     master_list,  # type: Sequence[DesignMaster]
                     name_list=None,  # type: Optional[Sequence[Optional[str]]]
                     lib_name='',  # type: str
                     debug=False,  # type: bool
                     rename_dict=None,  # type: Optional[Dict[str, str]]
                     **kwargs,  # type: Any
                     ):
        # type: (...) -> None
        """create all given masters in the database.

        Parameters
        ----------
        master_list : Sequence[DesignMaster]
            list of masters to instantiate.
        output : DesignOutput
            the output type.
        name_list : Optional[Sequence[Optional[str]]]
            list of master cell names.  If not given, default names will be used.
        lib_name : str
            Library to create the masters in.  If empty or None, use default library.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        **kwargs : Any
            parameters associated with the given output type.
        """
        if name_list is None:
            name_list = [None] * len(master_list)  # type: Sequence[Optional[str]]
        else:
            if len(name_list) != len(master_list):
                raise ValueError("Master list and name list length mismatch.")

        # configure renaming dictionary.  Verify that renaming dictionary is one-to-one.
        rename = self._rename_dict
        rename.clear()
        reverse_rename = {}  # type: Dict[str, str]
        if rename_dict:
            for key, val in rename_dict.items():
                if key != val:
                    if val in reverse_rename:
                        raise ValueError('Both %s and %s are renamed '
                                         'to %s' % (key, reverse_rename[val], val))
                    rename[key] = val
                    reverse_rename[val] = key

        for master, name in zip(master_list, name_list):
            if name is not None and name != master.cell_name:
                cur_name = master.cell_name
                if name in reverse_rename:
                    raise ValueError('Both %s and %s are renamed '
                                     'to %s' % (cur_name, reverse_rename[name], name))
                rename[cur_name] = name
                reverse_rename[name] = cur_name

                if name in self._used_cell_names:
                    # name is an already used name, so we need to rename it to something else
                    name2 = get_new_name(name, self._used_cell_names, reverse_rename)
                    rename[name] = name2
                    reverse_rename[name2] = name

        if debug:
            print('Retrieving master contents')

        # use ordered dict so that children are created before parents.
        info_dict = OrderedDict()  # type: Dict[str, DesignMaster]
        start = time.time()
        for master, top_name in zip(master_list, name_list):
            self._batch_output_helper(info_dict, master)
        end = time.time()

        if not lib_name:
            lib_name = self.lib_name
        if not lib_name:
            raise ValueError('master library name is not specified.')

        content_list = [master.get_content(self.format_cell_name)
                        for master in info_dict.values()]

        if debug:
            print('master content retrieval took %.4g seconds' % (end - start))

        self.create_masters_in_db(output, lib_name, content_list, debug=debug, **kwargs)

    def _batch_output_helper(self, info_dict, master):
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
        for master_key in master.children():
            child_temp = self._master_lookup[master_key]
            if child_temp.cell_name not in info_dict:
                self._batch_output_helper(info_dict, child_temp)

        # get template master for this cell.
        info_dict[master.cell_name] = self._master_lookup[master.key]
