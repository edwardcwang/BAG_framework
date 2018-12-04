# -*- coding: utf-8 -*-

"""This module defines classes representing various design instances.
"""

from typing import TYPE_CHECKING, Type, Optional, Any

from ..util.cache import Param

if TYPE_CHECKING:
    from .database import ModuleDB
    from .module import Module

    try:
        from pybag.core import PySchInstRef
    except ImportError:
        raise ImportError(
            'Cannot import pybag library.  Do you have the right shared library file?')


class SchInstance:
    """This class represents an instance inside a schematic.

    Parameters
    ----------
    db : ModuleDB
        the design database.
    inst_ptr : PySchInstRef
        a reference to the actual schematic instance object.
    """

    def __init__(self, db, inst_ptr, master=None):
        # type: (ModuleDB, PySchInstRef, Optional[Module]) -> None
        self._db = db
        self._master = master
        self._ptr = inst_ptr

        # get schematic class object from master
        if master is None:
            lib_name = self._ptr.lib_name
            static = self._ptr.is_primitive and lib_name != 'BAG_prim'
            if static:
                self._sch_cls = None
            else:
                cell_name = self._ptr.cell_name
                self._sch_cls = db.get_schematic_class(lib_name, cell_name)
        else:
            self._sch_cls = master.__class__

    @property
    def database(self):
        # type: () -> ModuleDB
        """ModuleDB: the schematic database."""
        return self._db

    @property
    def master(self):
        # type: () -> Optional[Module]
        """Optional[Module]: the master object of this instance."""
        return self._master

    @property
    def master_class(self):
        # type: () -> Optional[Type[Module]]
        return self._sch_cls

    @property
    def lib_name(self):
        # type: () -> str
        """str: the generator library name."""
        return self._ptr.lib_name

    @property
    def cell_name(self):
        # type: () -> str
        """str: the generator cell name."""
        return self._ptr.cell_name

    @property
    def master_cell_name(self):
        # type: () -> str
        """str: the cell name of the master object"""
        return self.cell_name if self.master is None else self.master.cell_name

    @property
    def static(self):
        # type: () -> bool
        """bool: True if this instance points to a static/fixed schematic."""
        return self._sch_cls is None

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
        """bool: True if this is a primitive (static or in BAG_prim) schematic instance."""
        if self._sch_cls is None:
            return True
        if self.master is None:
            raise ValueError('Schematic instance has no master.  Did you forget to call design()?')
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

    def design(self, **kwargs):
        # type: (**Any) -> None
        """Call the design method on master."""
        if self._sch_cls is None:
            raise RuntimeError('Cannot call design() method on static instances.')

        self._master = self._db.new_master(self._sch_cls, params=kwargs)
        if self._master.is_primitive():
            # update parameters
            for key, val in self._master.get_schematic_parameters().items():
                self.set_param(key, val)
        else:
            self._ptr.lib_name = self._master.lib_name
        self._ptr.cell_name = self._master.cell_name

    def design_model(self, model_params):
        # type: (Param) -> None
        """Call design_model method on master."""
        if self._sch_cls is None:
            # static instance; assume model is defined in include files
            return

        self._master = self._db.new_model(self._master, model_params)
        self._ptr.cell_name = self._master.cell_name

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
        self._master = None
        if static:
            self._sch_cls = None
        else:
            self._sch_cls = self._db.get_schematic_class(gen_lib_name, gen_cell_name)
        self._ptr.update_master(gen_lib_name, gen_cell_name, prim=static)

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

    def update_connection(self, inst_name, term_name, net_name):
        # type: (str, str, str) -> None
        """Update connections of this schematic instance.

        Parameters
        ----------
        inst_name : str
            The instance name.
        term_name : str
            The terminal (in other words, port) of the instance.
        net_name : str
            The net to connect the terminal to.
        """
        self._ptr.update_connection(inst_name, term_name, net_name)

    def get_connection(self, term_name):
        # type: (str) -> str
        """Get the net name connected to the given terminal.

        Parameters
        ----------
        term_name : str
            the terminal name.

        Returns
        -------
        net_name : str
            the resulting net name.  Empty string if given terminal is not found.
        """
        return self._ptr.get_connection(term_name)

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
        return self.lib_name if self.is_primitive else impl_lib
