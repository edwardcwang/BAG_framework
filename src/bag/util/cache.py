# -*- coding: utf-8 -*-

"""This module defines classes used to cache existing design masters
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Sequence, Dict, Set, Any, Optional, TypeVar, Type, Tuple, Iterator, List
)

import abc
import sys
import time
import numbers
from collections import OrderedDict

from pybag.enum import DesignOutput, is_netlist_type, is_model_type
from pybag.core import implement_yaml, implement_netlist, implement_gds

from sortedcontainers import SortedDict

from ..io import fix_string
from ..env import get_netlist_setup_file, get_gds_layer_map, get_gds_object_map
from .search import get_new_name

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.tech import TechInfo

MasterType = TypeVar('MasterType', bound='DesignMaster')
DBType = TypeVar('DBType', bound='MasterDB')


class Param(SortedDict):
    def __init__(self, *args, **kwargs) -> None:
        hash_val = kwargs.pop('_hash_value', 0)
        SortedDict.__init__(self, *args, **kwargs)
        self._hash_value = hash_val

    @classmethod
    def to_param(cls, table: Dict[Any, Any]):
        if isinstance(table, Param):
            return table
        ans = Param()
        for key, val in table.items():
            ans.assign(key, val)
        ans.update_hash()
        return ans

    @classmethod
    def get_hash(cls, val: object) -> int:
        if isinstance(val, list):
            seed = 0
            for item in val:
                seed = cls._combine_hash(seed, cls.get_hash(item))
            return seed
        else:
            return hash(val)

    @classmethod
    def _combine_hash(cls, seed: int, v: int) -> int:
        # boost::hash_combine port
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2)
        return seed & sys.maxsize

    def copy(self) -> Param:
        ans = SortedDict.copy(self)  # type: Param
        ans._hash_value = self._hash_value
        return ans

    def assign(self, name: object, value: object):
        if isinstance(value, Param) or not isinstance(value, dict):
            SortedDict.__setitem__(self, name, value)
        else:
            SortedDict.__setitem__(self, name, self.to_param(value))

    def update_hash(self) -> None:
        self._hash_value = 0
        for key, val in self.items():
            self._hash_value = self._combine_hash(self._hash_value, hash(key))
            self._hash_value = self._combine_hash(self._hash_value, self.get_hash(val))

    def __setitem__(self, name: object, value: object) -> None:
        raise TypeError('Cannot call __setitem__ on a Param object.')

    def __hash__(self) -> int:
        return self._hash_value


class DesignMaster(abc.ABC):
    """A design master instance.

    This class represents a design master in the design database.

    Parameters
    ----------
    master_db : MasterDB
        the master database.
    params : Param
        the parameters dictionary.
    key: Any
        If not None, the unique ID for this master instance.
    copy_state : Optional[Dict[str, Any]]
        If not None, set content of this master from this dictionary.

    Attributes
    ----------
    params : Param
        the parameters dictionary.
    """

    def __init__(self, master_db: DBType, params: Param, *,
                 key: Any = None, copy_state: Optional[Dict[str, Any]] = None) -> None:
        self._master_db = master_db  # type: DBType
        self._cell_name = ''

        if copy_state:
            self._children = copy_state['children']
            self._finalized = copy_state['finalized']
            self.params = copy_state['params']
            self._cell_name = copy_state['cell_name']
            self._key = copy_state['key']
        else:
            # use ordered dictionary so we have deterministic dependency order
            self._children = OrderedDict()
            self._finalized = False

            # set parameters
            self.params = params
            self._key = self.compute_unique_key(params) if key is None else key

            # update design master signature
            self._cell_name = get_new_name(self.get_master_basename(),
                                           self.master_db.used_cell_names)

    @classmethod
    def get_qualified_name(cls) -> str:
        """Returns the qualified name of this class."""
        my_module = cls.__module__
        if my_module is None or my_module == str.__class__.__module__:
            return cls.__name__
        else:
            return my_module + '.' + cls.__name__

    @classmethod
    def populate_params(cls, table: Dict[str, Any], params_info: Dict[str, str],
                        default_params: Dict[str, Any]) -> Param:
        """Fill params dictionary with values from table and default_params"""
        hidden_params = cls.get_hidden_params()

        result = Param()
        for key, desc in params_info.items():
            if key not in table:
                if key not in default_params:
                    raise ValueError('Parameter {} not specified.  '
                                     'Description:\n{}'.format(key, desc))
                else:
                    result.assign(key, default_params[key])
            else:
                result.assign(key, table[key])

        # add hidden parameters
        for name, value in hidden_params.items():
            result.assign(name, table.get(name, value))

        result.update_hash()
        return result

    @classmethod
    def compute_unique_key(cls, params: Param) -> Any:
        """Returns a unique hashable object (usually tuple or string) that represents this instance.

        Parameters
        ----------
        params : Param
            the parameters object.  All default and hidden parameters have been processed already.

        Returns
        -------
        unique_id : Any
            a hashable unique ID representing the given parameters.
        """
        return cls.get_qualified_name(), params

    @classmethod
    def process_params(cls, params: Dict[str, Any]) -> Tuple[Param, Any]:
        """Process the given parameters dictionary.

        This method computes the final parameters dictionary from the user given one by
        filling in default and hidden parameter values, and also compute the unique ID of
        this master instance.

        Parameters
        ----------
        params : Dict[str, Any]
            the parameter dictionary specified by the user.

        Returns
        -------
        unique_id : Any
            a hashable unique ID representing the given parameters.
        """
        params_info = cls.get_params_info()
        default_params = cls.get_default_param_values()
        params = cls.populate_params(params, params_info, default_params)
        return params, cls.compute_unique_key(params)

    def update_signature(self, key: Any) -> None:
        self._key = key
        self._cell_name = get_new_name(self.get_master_basename(), self.master_db.used_cell_names)

    def get_copy_state(self):
        # type: () -> Dict[str, Any]
        return {
            'children': self._children.copy(),
            'finalized': self._finalized,
            'params': self.params.copy(),
            'cell_name': self._cell_name,
            'key': self._key,
        }

    def get_copy(self):
        # type: () -> MasterType
        """Returns a copy of this master instance."""
        copy_state = self.get_copy_state()
        return self.__class__(self._master_db, None, copy_state=copy_state)

    @classmethod
    def to_immutable_id(cls, val):
        # type: (Any) -> Any
        """Convert the given object to an immutable type for use as keys in dictionary.
        """
        # python 2/3 compatibility: convert raw bytes to string
        val = fix_string(val)

        if (val is None or isinstance(val, numbers.Number) or isinstance(val, str) or
                isinstance(val, Param)):
            return val
        elif isinstance(val, list) or isinstance(val, tuple):
            return tuple((cls.to_immutable_id(item) for item in val))
        elif isinstance(val, dict):
            return Param.to_param(val)
        elif isinstance(val, set):
            return tuple((k for k in sorted(val)))
        elif hasattr(val, 'get_immutable_key') and callable(val.get_immutable_key):
            return val.get_immutable_key()
        else:
            raise Exception('Unrecognized value %s with type %s' % (str(val), type(val)))

    @classmethod
    def format_cell_name(cls, cell_name, rename_dict, name_prefix, name_suffix):
        # type: (str, Dict[str, str], str, str) -> str
        tmp = rename_dict.get(cell_name, cell_name)
        return name_prefix + tmp + name_suffix

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

    @classmethod
    def get_hidden_params(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary of hidden parameter values.

        hidden parameters are parameters are invisible to the user and only used
        and computed internally.

        Returns
        -------
        hidden_params : Dict[str, Any]
            dictionary of hidden parameter values.
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
    def get_content(self, output_type, rename_dict, name_prefix, name_suffix):
        # type: (DesignOutput, Dict[str, str], str, str) -> Tuple[str, Any]
        """Returns the content of this master instance.

        Parameters
        ----------
        output_type : DesignOutput
            the output type.
        rename_dict : Dict[str, str]
            the renaming dictionary.
        name_prefix : str
            the name prefix.
        name_suffix : str
            the name suffix.

        Returns
        -------
        cell_name : str
            the master cell name.
        content : Any
            the master content data structure.
        """
        return '', None

    @property
    def master_db(self):
        # type: () -> DBType
        """Returns the database used to create design masters."""
        return self._master_db

    @property
    def lib_name(self):
        # type: () -> str
        """The master library name"""
        return self._master_db.lib_name

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

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        self._finalized = True

    def add_child_key(self, child_key):
        # type: (object) -> None
        """Registers the given child key."""
        self._children[child_key] = None

    def clear_children_key(self):
        # type: () -> None
        """Remove all children keys."""
        self._children.clear()

    def children(self):
        # type: () -> Iterator[object]
        """Iterate over all children's key."""
        return iter(self._children)


class MasterDB(abc.ABC):
    """A database of existing design masters.

    This class keeps track of existing design masters and maintain design dependency hierarchy.

    Parameters
    ----------
    lib_name : str
        the library to put all generated templates in.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated master name prefix.
    name_suffix : str
        generated master name suffix.
    """

    def __init__(self, lib_name, prj=None, name_prefix='', name_suffix=''):
        # type: (str, Optional[BagProject], str, str) -> None

        self._prj = prj
        self._lib_name = lib_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix

        self._used_cell_names = set()  # type: Set[str]
        self._key_lookup = {}  # type: Dict[Any, Any]
        self._master_lookup = {}  # type: Dict[Any, DesignMaster]

    @property
    @abc.abstractmethod
    def tech_info(self) -> TechInfo:
        """TechInfo: the TechInfo object."""
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

    def create_masters_in_db(self, output, lib_name, content_list, debug=False, **kwargs):
        # type: (DesignOutput, str, List[Any], bool, **Any) -> None
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
        start = time.time()
        if output is DesignOutput.LAYOUT:
            if self._prj is None:
                raise ValueError('BagProject is not defined.')

            # create layouts
            self._prj.instantiate_layout(lib_name, content_list)
        elif output is DesignOutput.GDS:
            fname = kwargs['fname']
            res = self.tech_info.resolution
            user_unit = self.tech_info.layout_unit

            lay_map = get_gds_layer_map()
            obj_map = get_gds_object_map()

            implement_gds(fname, lib_name, lay_map, obj_map, res, user_unit, content_list)
        elif output is DesignOutput.SCHEMATIC:
            if self._prj is None:
                raise ValueError('BagProject is not defined.')

            self._prj.instantiate_schematic(lib_name, content_list)
        elif output is DesignOutput.YAML:
            fname = kwargs['fname']

            implement_yaml(fname, content_list)
        elif is_netlist_type(output) or is_model_type(output):
            fname = kwargs['fname']
            flat = kwargs.get('flat', True)
            shell = kwargs.get('shell', False)
            rmin = kwargs.get('rmin', 2000)

            prim_fname = get_netlist_setup_file()

            implement_netlist(fname, content_list, output, flat, shell, rmin, prim_fname)
        else:
            raise ValueError('Unknown design output type: {}'.format(output.name))
        end = time.time()

        if debug:
            print('design instantiation took %.4g seconds' % (end - start))

    def clear(self):
        """Clear all existing schematic masters."""
        self._key_lookup.clear()
        self._master_lookup.clear()

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

        master_params, key = gen_cls.process_params(params)
        test = self.find_master(key)
        if test is not None:
            if debug:
                print('master cached')
            return test

        if debug:
            print('finalizing master')
        master = gen_cls(self, master_params, key=key, **kwargs)
        start = time.time()
        master.finalize()
        end = time.time()
        self.register_master(key, master)
        if debug:
            print('finalizing master took %.4g seconds' % (end - start))

        return master

    def find_master(self, key):
        # type: (Any) -> Optional[MasterType]
        return self._master_lookup.get(key, None)

    def register_master(self, key, master):
        # type: (Any, MasterType) -> None
        self._master_lookup[key] = master
        self._used_cell_names.add(master.cell_name)

    def instantiate_master(self, output, master, top_cell_name='', **kwargs):
        # type: (DesignOutput, DesignMaster, str, **Any) -> None
        """Instantiate the given master.

        Parameters
        ----------
        output : DesignOutput
            the design output type.
        master : DesignMaster
            the :class:`~bag.layout.template.TemplateBase` to instantiate.
        top_cell_name : str
            name of the top level cell.  If empty, a default name is used.
        **kwargs : Any
            optional arguments for batch_output().
        """
        self.batch_output(output, [(master, top_cell_name)], **kwargs)

    def batch_output(self,
                     output,  # type: DesignOutput
                     info_list,  # type: Sequence[Tuple[DesignMaster, str]]
                     debug=False,  # type: bool
                     rename_dict=None,  # type: Optional[Dict[str, str]]
                     **kwargs,  # type: Any
                     ):
        # type: (...) -> None
        """create all given masters in the database.

        Parameters
        ----------
        output : DesignOutput
            The output type.
        info_list : Sequence[Tuple[DesignMaster, str]]
            Sequence of (master, cell_name) tuples to instantiate.
            Use empty string cell_name to use default names.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        **kwargs : Any
            parameters associated with the given output type.
        """
        # configure renaming dictionary.  Verify that renaming dictionary is one-to-one.
        rename = {}
        reverse_rename = {}  # type: Dict[str, str]
        if rename_dict:
            for key, val in rename_dict.items():
                if key != val:
                    if val in reverse_rename:
                        raise ValueError('Both %s and %s are renamed '
                                         'to %s' % (key, reverse_rename[val], val))
                    rename[key] = val
                    reverse_rename[val] = key

        for m, name in info_list:
            m_name = m.cell_name
            if name and name != m_name:
                if name in reverse_rename:
                    raise ValueError('Both %s and %s are renamed '
                                     'to %s' % (m_name, reverse_rename[name], name))
                rename[m_name] = name
                reverse_rename[name] = m_name

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
        for master, _ in info_list:
            self._batch_output_helper(info_dict, master)
        end = time.time()

        content_list = [master.get_content(output, rename, self._name_prefix, self._name_suffix)
                        for master in info_dict.values()]

        if debug:
            print('master content retrieval took %.4g seconds' % (end - start))

        self.create_masters_in_db(output, self.lib_name, content_list, debug=debug, **kwargs)

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
