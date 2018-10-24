# -*- coding: utf-8 -*-

"""This module defines classes representing various design instances.
"""

from typing import TYPE_CHECKING, Optional, Any, Tuple, Dict

from ..util.search import get_new_name

try:
    import pybag
except ImportError:
    raise ImportError('Cannot import pybag library.  Do you have the right shared library file?')

if TYPE_CHECKING:
    from .database import ModuleDB
    from .module import Module


class DesignInstance:
    """This class represents all design instances.

    Parameters
    ----------
    db : ModuleDB
        the design database.
    lib_name : str
        the generator library name.
    cell_name : str
        the generator cell name.
    """

    def __init__(self, db, lib_name, cell_name):
        # type: (ModuleDB, str, str) -> None
        self._db = db
        self._master = None  # type: Optional[Module]
        self._lib_name = lib_name
        self._cell_name = cell_name

    @property
    def master(self):
        # type: () -> Optional[Module]
        """Optional[Module]: the master object of this instance."""
        return self._master

    @master.setter
    def master(self, new_master):
        # type: (Optional[Module]) -> None
        self._master = new_master

    @property
    def database(self):
        # type: () -> ModuleDB
        """ModuleDB: the database storing this instance."""
        return self._db

    @property
    def gen_lib_name(self):
        # type: () -> str
        """str: the generator library name."""
        return self._lib_name

    @property
    def gen_cell_name(self):
        # type: () -> str
        """str: the generator cell name."""
        return self._cell_name

    @property
    def master_cell_name(self):
        # type: () -> str
        """str: the cell name of the master object"""
        return self.gen_cell_name if self.master is None else self.master.cell_name

    def set_param(self, key, val):
        # type: (str, Any) -> None
        """Sets the parameters of this instance.

        Parameters
        ----------
        key : str
            the parameter name.
        val : Any
            the parameter value.
        """
        raise Exception('Cannot set parameters on a DesignInstance; '
                        'DesignInstance with primitive masters are not allowed.')

    def design_specs(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Call the design method on master."""
        self._update_master('design_specs', args, kwargs)

    def design(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Call the design method on master."""
        self._update_master('design', args, kwargs)

    def _update_master(self, design_fun, args, kwargs):
        # type: (str, Tuple[Any], Dict[str, Any]) -> None
        """Update the underlying master object."""
        if args:
            key = get_new_name('args', kwargs)
            kwargs[key] = args
        else:
            key = None
        self._master = self._db.new_master(self.gen_lib_name, self.gen_cell_name,
                                           params=kwargs, design_args=key,
                                           design_fun=design_fun)

        if self._master.is_primitive():
            # update parameters
            for key, val in self._master.get_schematic_parameters().items():
                self.set_param(key, val)

    def implement_design(self, lib_name, top_cell_name='', prefix='', suffix='',
                         debug=False, rename_dict=None, lib_path='', output='schematic',
                         **kwargs):
        # type: (str, str, str, str, bool, Optional[Dict[str, str]], str, str, **Any)-> None
        """Implement this design.

        Parameters
        ----------
        lib_name : str
            the generated library name.
        top_cell_name : str
            the generated top design cell name.
        prefix : str
            prefix for all generated cell names.
        suffix : str
            suffix for all generated cell names.
        debug : bool
            True to enable debug messages.
        rename_dict : Optional[Dict[str, str]]
            optional renaming dictionary.
        lib_path : str
            path to create the generated library in.
        output : str
            the design output type.
        **kwargs : Any
            any additional parameters specific to the output type.
        """
        if not top_cell_name:
            top_cell_name = None

        if lib_path:
            self._db.lib_path = lib_path
        self._db.cell_prefix = prefix
        self._db.cell_suffix = suffix
        self._db.instantiate_masters([self._master], [top_cell_name], lib_name=lib_name,
                                     debug=debug, rename_dict=rename_dict, output=output,
                                     **kwargs)


class SchInstance(DesignInstance):
    """This class represents a schematic instance within another schematic.

    Parameters
    ----------
    db : ModuleDB
        the design database.
    inst_ptr : pybag.base.schematic.PySchInstRef
        a reference to the actual schematic instance object.
    is_static : bool
        rue if this instance points to a static/fixed schematic.
    """

    def __init__(self, db, inst_ptr, is_static):
        # type: (ModuleDB, pybag.base.schematic.PySchInstRef, bool) -> None
        DesignInstance.__init__(self, db, '', '')
        self._static = is_static
        self._ptr = inst_ptr

    @property
    def static(self):
        # type: () -> bool
        """bool: True if this instance points to a static/fixed schematic."""
        return self._static

    @property
    def gen_lib_name(self):
        # type: () -> str
        """str: the generator library name."""
        return self._ptr.lib_name

    @property
    def gen_cell_name(self):
        # type: () -> str
        """str: the generator cell name."""
        return self._ptr.cell_name

    @property
    def name(self):
        # type: () -> str
        """str: the name of this instance in the parent schematic."""
        return self._ptr.name

    @property
    def width(self):
        # type: () -> int
        """int: the instance symbol width."""
        return self._ptr.width

    @property
    def height(self):
        # type: () -> int
        """int: the instance symbol height."""
        return self._ptr.height

    @property
    def is_primitive(self):
        # type: () -> bool
        """bool: True if this is a primitive schematic instance."""
        if self._static:
            return True
        if self.master is None:
            raise ValueError(
                'Instance {} has no master.  Did you forget to call design()?'.format(self.name))
        return self.master.is_primitive()

    @property
    def should_delete(self):
        # type: () -> bool
        """bool: True if this instance should be deleted by the parent."""
        return self.master is not None and self.master.should_delete_instance()

    @property
    def master_key(self):
        # type: () -> Optional[Any]
        """Optional[Any]: A unique key identifying the master object."""
        if self.master is None:
            raise ValueError('Instance {} has no master; cannot get key')
        return self.master.key

    def update_primitive_flag(self):
        # type: () -> None
        """Update the is_primitive flag of the underlying instance reference."""
        self._ptr.is_primitive = self.is_primitive

    def _update_master(self, design_fun, args, kwargs):
        # type: (str, Tuple[Any], Dict[str, Any]) -> None
        """Update the underlying master object."""
        DesignInstance._update_master(self, design_fun, args, kwargs)
        # update instance cell name and primitive flag after master is updated
        self._ptr.cell_name = self.master_cell_name
        self._ptr.is_primitive = self.is_primitive

    def change_generator(self, gen_lib_name, gen_cell_name, static=False):
        # type: (str, str, bool) -> None
        """Change the circuit generator responsible for producing this instance.

        Parameter
        ---------
        gen_lib_name : str
            new generator library name.
        gen_cell_name : str
            new generator cell name.
        static : bool
            True if this is actually a fixed schematic, not a generator.
        """
        self.master = None
        self._ptr.lib_name = gen_lib_name
        self._ptr.cell_name = gen_cell_name
        self._static = static
        self._ptr.reset()

    def set_param(self, key, val):
        # type: (str, Any) -> None
        """Sets the parameters of this instance.

        Parameters
        ----------
        key : str
            the parameter name.
        val : Any
            the parameter value.
        """
        self._ptr.set_param(key, val)

    def update_connection(self, term_name, net_name):
        # type: (str, str) -> None
        """Update connections of this schematic instance.

        Parameters
        ----------
        term_name : str
            The terminal (in other words, port) of the instance.
        net_name : str
            The net to connect the terminal to.
        """
        self._ptr.update_connection(term_name, net_name)

    def get_master_lib_name(self, impl_lib):
        # type: (str) -> str
        """Returns the master library name.

        the master library could be different than the implementation library in
        the case of static schematic.

        Parameters
        ----------
        impl_lib : str
            implementation library name.

        Returns
        -------
        master_lib : str
            the master library name.

        """
        return self.gen_lib_name if self.is_primitive else impl_lib
