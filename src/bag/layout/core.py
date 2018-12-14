# -*- coding: utf-8 -*-

"""This module defines some core layout classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

# noinspection PyUnresolvedReferences
from pybag.core import PyLayInstRef, BBox

if TYPE_CHECKING:
    from .template import TemplateBase


class PyLayInstance:
    def __init__(self, parent: TemplateBase, master: TemplateBase, ref: PyLayInstRef) -> None:
        self._parent = parent
        self._master = master
        self._ref = ref

    @property
    def nx(self) -> int:
        """int: Number of columns."""
        return self._ref.nx

    @property
    def ny(self) -> int:
        """int: Number of rows."""
        return self._ref.ny

    @property
    def spx_unit(self) -> int:
        """int: The column pitch."""
        return self._ref.spx

    @property
    def spy_unit(self) -> int:
        """int: The row pitch."""
        return self._ref.spy

    @property
    def master(self) -> TemplateBase:
        """TemplateBase: the master of this instance."""
        return self._master

    @property
    def location_unit(self) -> Tuple[int, int]:
        """Tuple[int, int]: The instance location"""
        return self._ref.location

    @property
    def orientation(self) -> str:
        """str: The instance orientation"""
        return self._ref.orientation

    @property
    def bound_box(self) -> BBox:
        """BBox: Returns the overall bounding box of this instance."""
        return self._master.bound_box.tr
        return self._translate_master_box_w_array(self._master.bound_box)

    @property
    def array_box(self):
        """Returns the array box of this instance."""
        master_box = getattr(self._master, 'array_box', None)  # type: BBox
        if master_box is None:
            raise ValueError('Master template array box is not defined.')

        return self._translate_master_box_w_array(master_box)

    @property
    def fill_box(self):
        """Returns the fill box of this instance."""
        master_box = getattr(self._master, 'fill_box', None)  # type: BBox
        if master_box is None:
            raise ValueError('Master template fill box is not defined.')

        return self._translate_master_box_w_array(master_box)

    @nx.setter
    def nx(self, val: int) -> None:
        self._ref.nx = val

    @ny.setter
    def ny(self, val: int) -> None:
        self._ref.ny = val

    @spx_unit.setter
    def spx_unit(self, val: int) -> None:
        self._ref.spx = val

    @spy_unit.setter
    def spy_unit(self, val: int) -> None:
        self._ref.spy = val

    @location_unit.setter
    def location_unit(self, new_loc: Tuple[int, int]) -> None:
        self._ref.location = new_loc

    @orientation.setter
    def orientation(self, val: str) -> None:
        self._ref.orientation = val

    def get_item_location(self, row=0, col=0, unit_mode=True):
        # type: (int, int, bool) -> Tuple[int, int]
        """Returns the location of the given item in the array.

        Parameters
        ----------
        row : int
            the item row index.  0 is the bottom-most row.
        col : int
            the item column index.  0 is the left-most column.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        xo : Union[float, int]
            the item X coordinate.
        yo : Union[float, int]
            the item Y coordinate.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        if row < 0 or row >= self.ny or col < 0 or col >= self.nx:
            raise ValueError('Invalid row/col index: row=%d, col=%d' % (row, col))

        return col * self.spx_unit, row * self.spy_unit

    def get_bound_box_of(self, row=0, col=0):
        # type: (int, int) -> BBox
        """Returns the bounding box of an instance in this mosaic."""
        dx, dy = self.get_item_location(row=row, col=col)
        cdef
        BBox
        box = self._master.bound_box
        box = box.c_transform(self._ref.obj.xform)
        return box.move_by(dx, dy)

    def move_by(self, dx=0, dy=0, unit_mode=True):
        # type: (int, int, bool) -> None
        """Move this instance by the given amount.

        Parameters
        ----------
        dx : int
            the X shift.
        dy : int
            the Y shift.
        unit_mode : bool
            deprecated parameter.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        if self._ref.parent == NULL:
            raise ValueError('Cannot change object after commit')

        self._ref.obj.xform.move_by(dx, dy)

    def new_master_with(self, **kwargs):
        # type: (Any) -> None
        """Change the master template of this instance.

        This method will get the old master template layout parameters, update
        the parameter values with the given dictionary, then create a new master
        template with those parameters and associate it with this instance.

        Parameters
        ----------
        **kwargs
            a dictionary of new parameter values.
        """
        self._master = self._master.new_template_with(**kwargs)
        self._update_inst_master(self._master.layout_cellview)

    def translate_master_box(self, BBox box

        ):
        # type: (BBox) -> BBox
        """Transform the bounding box in master template.

        Parameters
        ----------
        box : BBox
            the BBox in master template coordinate.

        Returns
        -------
        new_box : BBox
            the corresponding BBox in instance coordinate.
        """
        return box.c_transform(self._ref.obj.xform)


    def translate_master_location(self, mloc, unit_mode=True):
        # type: (Tuple[int, int], bool) -> Tuple[int, int]
        """Returns the actual location of the given point in master template.

        Parameters
        ----------
        mloc : Tuple[int, int]
            the location in master coordinate.
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        loc : Tuple[int, int]
            The actual location.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        cdef
        coord_t
        x = mloc[0]
        cdef
        coord_t
        y = mloc[1]
        self._ref.obj.xform.transform(x, y)

        return x, y


    def translate_master_track(self, layer_id, track_idx):
        # type: (int, Union[float, int]) -> Union[float, int]
        """Returns the actual track index of the given track in master template.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        track_idx : Union[float, int]
            the track index.

        Returns
        -------
        new_idx : Union[float, int]
            the new track index.
        """
        dx, dy = self.location_unit
        return self._grid.transform_track(layer_id, track_idx, dx=dx, dy=dy,
                                          orient=self.orientation)


    def get_port(self, name='', row=0, col=0):
        # type: (Optional[str], int, int) -> Port
        """Returns the port object of the given instance in the array.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.

        Returns
        -------
        port : Port
            the port object.
        """
        dx, dy = self.get_item_location(row=row, col=col)
        xshift, yshift = self.location_unit
        loc = (xshift + dx, yshift + dy)
        return self._master.get_port(name).transform(self._grid, loc=loc,
                                                     orient=self.orientation)


    def get_pin(self, name='', row=0, col=0, layer=-1):
        # type: (Optional[str], int, int, int) -> Union[WireArray, BBox]
        """Returns the first pin with the given name.

        This is an efficient method if you know this instance has exactly one pin.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Returns
        -------
        pin : Union[WireArray, BBox]
            the first pin associated with the port of given name.
        """
        return self.get_port(name, row, col).get_pins(layer)[0]


    def port_pins_iter(self, name='', layer=-1):
        # type: (Optional[str], int) -> Iterable[WireArray]
        """Iterate through all pins of all ports with the given name in this instance array.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Yields
        ------
        pin : WireArray
            the pin as WireArray.
        """
        for col in range(self.nx):
            for row in range(self.ny):
                try:
                    port = self.get_port(name, row, col)
                except KeyError:
                    return
                for warr in port.get_pins(layer):
                    yield warr


    def get_all_port_pins(self, name='', layer=-1):
        # type: (Optional[str], int) -> List[WireArray]
        """Returns a list of all pins of all ports with the given name in this instance array.

        This method gathers ports from all instances in this array with the given name,
        then find all pins of those ports on the given layer, then return as list of WireArrays.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Returns
        -------
        pin_list : List[WireArray]
            the list of pins as WireArrays.
        """
        return list(self.port_pins_iter(name=name, layer=layer))


    def port_names_iter(self):
        # type: () -> Iterable[str]
        """Iterates over port names in this instance.

        Yields
        ------
        port_name : str
            name of a port in this instance.
        """
        return self._master.port_names_iter()


    def has_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this instance has the given port."""
        return self._master.has_port(port_name)


    def has_prim_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this instance has the given primitive port."""
        return self._master.has_prim_port(port_name)


    def commit(self):
        if self._grid.tech_info.use_flip_parity():
            # update track parity
            top_layer = self._master.top_layer
            bot_layer = self._grid.get_bot_common_layer(self._master.grid, top_layer)
            fp_dict = self._grid.get_flip_parity_at(bot_layer, top_layer, self.location_unit,
                                                    self.orientation)
            self.new_master_with(flip_parity=fp_dict)

        self._dep_set.add(self._master.key)
        self._ref.commit()