# -*- coding: utf-8 -*-

"""This module defines layout template classes.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Union, Dict, Any, List, TypeVar, Type, Optional, Tuple, Iterable,
    Sequence, cast
)
from bag.typing import PointType

import abc
import copy
from itertools import product

from ..util.cache import DesignMaster, MasterDB
from ..util.interval import IntervalSet
from ..util.math import HalfInt
from ..io.file import write_yaml
from .core import PyLayInstance
from .tech import TechInfo
from .routing.base import Port, TrackID, WireArray
from .routing.grid import RoutingGrid

from pybag.enum import (
    PathStyle, BlockageType, BoundaryType, GeometryMode, DesignOutput, Orient2D,
    Orientation, Direction, MinLenMode
)
from pybag.core import (
    BBox, BBoxArray, PyLayCellView, Transform, PyLayInstRef, PyPath, PyBlockage, PyBoundary,
    PyRect, PyVia, PyPolygon, PyPolygon90, PyPolygon45, ViaParam, COORD_MIN, COORD_MAX
)

GeoType = Union[PyRect, PyPolygon90, PyPolygon45, PyPolygon]
TemplateType = TypeVar('TemplateType', bound='TemplateBase')
DiffWarrType = Tuple[Optional[WireArray], Optional[WireArray]]

if TYPE_CHECKING:
    from bag.core import BagProject
    from bag.typing import TrackType, SizeType


class TemplateDB(MasterDB):
    """A database of all templates.

    This class is a subclass of MasterDB that defines some extra properties/function
    aliases to make creating layouts easier.

    Parameters
    ----------
    routing_grid : RoutingGrid
        the default RoutingGrid object.
    lib_name : str
        the cadence library to put all generated templates in.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    """

    def __init__(self, routing_grid: RoutingGrid, lib_name: str, prj: Optional[BagProject] = None,
                 name_prefix: str = '', name_suffix: str = '') -> None:
        MasterDB.__init__(self, lib_name, prj=prj, name_prefix=name_prefix, name_suffix=name_suffix)

        self._grid = routing_grid

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The global RoutingGrid instance."""
        return self._grid

    @property
    def tech_info(self) -> TechInfo:
        return self._grid.tech_info

    def new_template(self, temp_cls: Type[TemplateType], params: Optional[Dict[str, Any]] = None,
                     **kwargs: Any) -> TemplateType:
        """Alias for new_master() for backwards compatibility.
        """
        return self.new_master(temp_cls, params=params, **kwargs)

    def instantiate_layout(self, template: TemplateBase, top_cell_name: str = '',
                           output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for instantiate_master(), with default output type of LAYOUT.
        """
        self.instantiate_master(output, template, top_cell_name, **kwargs)

    def batch_layout(self, info_list: Sequence[Tuple[TemplateBase, str]],
                     output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for batch_output(), with default output type of LAYOUT.
        """
        self.batch_output(output, info_list, **kwargs)


class TemplateBase(DesignMaster, metaclass=abc.ABCMeta):
    """The base template class.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    params : Dict[str, Any]
        the parameter values.
    **kwargs : Any
        dictionary of the following optional parameters:

        grid : RoutingGrid
            the routing grid to use for this template.
        use_cybagoa : bool
            True to use cybagoa module to accelerate layout.
    """

    def __init__(self, temp_db: TemplateDB, params: Dict[str, Any], **kwargs: Any) -> None:
        # initialize template attributes
        self._parent_grid = kwargs.get('grid', temp_db.grid)
        self._grid = self._parent_grid.copy()  # type: RoutingGrid
        self._size = None  # type: SizeType
        self._ports = {}
        self._port_params = {}
        self._prim_ports = {}
        self._prim_port_params = {}
        self._array_box = None  # type: BBox
        self._fill_box = None  # type: BBox
        self.prim_top_layer = None
        self.prim_bound_box = None

        # add hidden parameters
        if 'hidden_params' in kwargs:
            hidden_params = kwargs['hidden_params'].copy()
        else:
            hidden_params = {}
        hidden_params['flip_parity'] = None

        DesignMaster.__init__(self, temp_db, params, hidden_params=hidden_params)
        # update RoutingGrid
        fp_dict = self.params['flip_parity']
        if fp_dict is not None:
            self._grid.flip_parity = fp_dict

        # create Cython wrapper object
        self._layout = PyLayCellView(self._grid, self.cell_name)

    @abc.abstractmethod
    def draw_layout(self) -> None:
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        pass

    def get_master_basename(self) -> str:
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self.get_layout_basename()

    def get_layout_basename(self) -> str:
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        return self.__class__.__name__

    def get_content(self, output_type: DesignOutput, rename_dict: Dict[str, str],
                    name_prefix: str, name_suffix: str) -> Tuple[str, Any]:
        if not self.finalized:
            raise ValueError('This template is not finalized yet')

        cell_name = self.format_cell_name(self._layout.cell_name, rename_dict,
                                          name_prefix, name_suffix)
        return name_prefix + cell_name + name_suffix, self._layout

    def finalize(self) -> None:
        """Finalize this master instance.
        """
        # create layout
        self.draw_layout()

        # finalize this template
        grid = self.grid
        grid.tech_info.finalize_template(self)

        # construct port objects
        for net_name, port_params in self._port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for wire_arr_list in pin_dict.values():
                    for warr in wire_arr_list:  # type: WireArray
                        self._layout.add_pin_arr(net_name, label, warr)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # construct primitive port objects
        for net_name, port_params in self._prim_port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for layer_name, box_list in pin_dict.items():
                    for box in box_list:
                        self._layout.add_pin(layer_name, net_name, label, box)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # call super finalize routine
        DesignMaster.finalize(self)

    @property
    def template_db(self) -> TemplateDB:
        """TemplateDB: The template database object"""
        # noinspection PyTypeChecker
        return self.master_db

    @property
    def is_empty(self) -> bool:
        """bool: True if this template is empty."""
        return self._layout.is_empty

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The RoutingGrid object"""
        return self._grid

    @grid.setter
    def grid(self, new_grid: RoutingGrid) -> None:
        if not self._finalized:
            self._grid = new_grid
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def array_box(self) -> Optional[BBox]:
        """Optional[BBox]: The array/abutment bounding box of this template."""
        return self._array_box

    @array_box.setter
    def array_box(self, new_array_box: BBox) -> None:
        if not self._finalized:
            self._array_box = new_array_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def fill_box(self) -> Optional[BBox]:
        """Optional[BBox]: The dummy fill bounding box of this template."""
        return self._fill_box

    @fill_box.setter
    def fill_box(self, new_box: BBox) -> None:
        if not self._finalized:
            self._fill_box = new_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def top_layer(self) -> int:
        """int: The top layer ID used in this template."""
        if self.size is None:
            if self.prim_top_layer is None:
                raise Exception('Both size and prim_top_layer are unset.')
            return self.prim_top_layer
        return self.size[0]

    @property
    def size(self) -> Optional[SizeType]:
        """Optional[SizeType]: The size of this template, in (layer, nx_blk, ny_blk) format."""
        return self._size

    @property
    def bound_box(self) -> Optional[BBox]:
        """Optional[BBox]: Returns the template BBox.  None if size not set yet."""
        mysize = self.size
        if mysize is None:
            if self.prim_bound_box is None:
                raise ValueError('Both size and prim_bound_box are unset.')
            return self.prim_bound_box

        wblk, hblk = self.grid.get_size_dimension(mysize)
        return BBox(0, 0, wblk, hblk)

    @size.setter
    def size(self, new_size: SizeType) -> None:
        if not self._finalized:
            self._size = new_size
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def layout_cellview(self) -> PyLayCellView:
        """PyLayCellView: The internal layout object."""
        return self._layout

    def set_geometry_mode(self, mode: GeometryMode) -> None:
        """Sets the geometry mode of this layout.

        Parameters
        ----------
        mode : GeometryMode
            the geometry mode.
        """
        self._layout.set_geometry_mode(mode.value)

    def get_rect_bbox(self, layer: str, purpose: str = '') -> BBox:
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.

        Parameters
        ----------
        layer : str
            the layer name.
        purpose : str
            the purpose name.

        Returns
        -------
        box : BBox
            the overall bounding box of the given layer.
        """
        return self._layout.get_rect_bbox(layer, purpose)

    def new_template_with(self, **kwargs: Any) -> TemplateBase:
        """Create a new template with the given parameters.

        This method will update the parameter values with the given dictionary,
        then create a new template with those parameters and return it.

        Parameters
        ----------
        **kwargs : Any
            a dictionary of new parameter values.

        Returns
        -------
        new_temp : TemplateBase
            A new layout master object.
        """
        # get new parameter dictionary.
        new_params = copy.deepcopy(self.params)
        for key, val in kwargs.items():
            if key in new_params:
                new_params[key] = val

        return self.template_db.new_template(params=new_params, temp_cls=self.__class__,
                                             grid=self._parent_grid)

    def set_size_from_bound_box(self, top_layer_id: int, bbox: BBox, *, round_up: bool = False,
                                half_blk_x: bool = True, half_blk_y: bool = True):
        """Compute the size from overall bounding box.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        bbox : BBox
            the overall bounding box
        round_up: bool
            True to round up bounding box if not quantized properly
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.
        """
        grid = self.grid

        if bbox.xl != 0 or bbox.yl != 0:
            raise ValueError('lower-left corner of overall bounding box must be (0, 0).')

        # noinspection PyAttributeOutsideInit
        self.size = grid.get_size_tuple(top_layer_id, bbox.w, bbox.h, round_up=round_up,
                                        half_blk_x=half_blk_x, half_blk_y=half_blk_y)

    def set_size_from_array_box(self, top_layer_id: int) -> None:
        """Automatically compute the size from array_box.

        Assumes the array box is exactly in the center of the template.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        """
        grid = self.grid

        array_box = self.array_box
        if array_box is None:
            raise ValueError("array_box is not set")

        dx = array_box.xl
        dy = array_box.yl
        if dx < 0 or dy < 0:
            raise ValueError('lower-left corner of array box must be in first quadrant.')

        # noinspection PyAttributeOutsideInit
        self.size = grid.get_size_tuple(top_layer_id, 2 * dx + self.array_box.width_unit,
                                        2 * dy + self.array_box.height_unit)

    def write_summary_file(self, fname: str, lib_name: str, cell_name: str) -> None:
        """Create a summary file for this template layout."""
        # get all pin information
        grid = self.grid
        tech_info = grid.tech_info
        pin_dict = {}
        res = grid.resolution
        for port_name in self.port_names_iter():
            pin_cnt = 0
            port = self.get_port(port_name)
            for pin_warr in port:
                for lay, _, bbox in pin_warr.wire_iter(grid):
                    if pin_cnt == 0:
                        pin_name = port_name
                    else:
                        pin_name = '%s_%d' % (port_name, pin_cnt)
                    pin_cnt += 1
                    pin_dict[pin_name] = dict(
                        layer=[lay, tech_info.pin_purpose],
                        netname=port_name,
                        xy0=[bbox.xl * res, bbox.yl * res],
                        xy1=[bbox.xh * res, bbox.yh * res],
                    )

        # get size information
        bnd_box = self.bound_box
        if bnd_box is None:
            raise ValueError("bound_box is not set")
        info = {
            lib_name: {
                cell_name: dict(
                    pins=pin_dict,
                    xy0=[0.0, 0.0],
                    xy1=[bnd_box.w * res, bnd_box.h * res],
                ),
            },
        }

        write_yaml(fname, info)

    def get_pin_name(self, name: str) -> str:
        """Get the actual name of the given pin from the renaming dictionary.

        Given a pin name, If this Template has a parameter called 'rename_dict',
        return the actual pin name from the renaming dictionary.

        Parameters
        ----------
        name : str
            the pin name.

        Returns
        -------
        actual_name : str
            the renamed pin name.
        """
        rename_dict = self.params.get('rename_dict', {})
        return rename_dict.get(name, name)

    def get_port(self, name: str = '') -> Port:
        """Returns the port object with the given name.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this template has only one port,
            then return it.

        Returns
        -------
        port : Port
            the port object.
        """
        if not name:
            if len(self._ports) != 1:
                raise ValueError('Template has %d ports != 1.' % len(self._ports))
            name = next(iter(self._ports))
        return self._ports[name]

    def has_port(self, port_name: str) -> bool:
        """Returns True if this template has the given port."""
        return port_name in self._ports

    def port_names_iter(self) -> Iterable[str]:
        """Iterates over port names in this template.

        Yields
        ------
        port_name : str
            name of a port in this template.
        """
        return self._ports.keys()

    def get_prim_port(self, name: str = '') -> Port:
        """Returns the primitive port object with the given name.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this template has only one port,
            then return it.

        Returns
        -------
        port : Port
            the primitive port object.
        """
        if not name:
            if len(self._prim_ports) != 1:
                raise ValueError('Template has %d ports != 1.' % len(self._prim_ports))
            name = next(iter(self._ports))
        return self._prim_ports[name]

    def has_prim_port(self, port_name: str) -> bool:
        """Returns True if this template has the given primitive port."""
        return port_name in self._prim_ports

    def prim_port_names_iter(self) -> Iterable[str]:
        """Iterates over primitive port names in this template.

        Yields
        ------
        port_name : str
            name of a primitive port in this template.
        """
        return self._prim_ports.keys()

    def new_template(self, temp_cls: Type[TemplateType], *, params: Optional[Dict[str, Any]] = None,
                     **kwargs: Any) -> TemplateType:
        """Create a new template.

        Parameters
        ----------
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.
        **kwargs : Any
            optional template parameters.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        kwargs['grid'] = self.grid
        return self.template_db.new_template(params=params, temp_cls=temp_cls, **kwargs)

    def add_instance(self,
                     master: TemplateBase,
                     *,
                     inst_name: str = '',
                     xform: Optional[Transform] = None,
                     nx: int = 1,
                     ny: int = 1,
                     spx: int = 0,
                     spy: int = 0,
                     commit: bool = True,
                     ) -> PyLayInstance:
        """Adds a new (arrayed) instance to layout.

        Parameters
        ----------
        master : TemplateBase
            the master template object.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        xform : Optional[Transform]
            the transformation object.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        inst : PyLayInstance
            the added instance.
        """
        if xform is None:
            xform = Transform()

        ref = self._layout.add_instance(master.layout_cellview, inst_name, xform, nx, ny,
                                        spx, spy, commit)
        return PyLayInstance(self, master, ref)

    def add_instance_primitive(self,
                               lib_name: str,
                               cell_name: str,
                               *,
                               xform: Optional[Transform] = None,
                               view_name: str = 'layout',
                               inst_name: str = '',
                               nx: int = 1,
                               ny: int = 1,
                               spx: int = 0,
                               spy: int = 0,
                               params: Optional[Dict[str, Any]] = None,
                               commit: bool = True,
                               **kwargs: Any,
                               ) -> PyLayInstRef:
        """Adds a new (arrayed) primitive instance to layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        xform : Optional[Transform]
            the transformation object.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.  Used for adding pcell instance.
        commit : bool
            True to commit the object immediately.
        **kwargs : Any
            additional arguments.  Usually implementation specific.

        Returns
        -------
        ref : PyLayInstRef
            A reference to the primitive instance.
        """
        if not params:
            params = kwargs
        else:
            params.update(kwargs)
        if xform is None:
            xform = Transform()

        # TODO: support pcells
        if params:
            raise ValueError("layout pcells not supported yet; see developer")

        return self._layout.add_prim_instance(lib_name, cell_name, view_name, inst_name, xform,
                                              nx, ny, spx, spy, commit)

    def is_horizontal(self, layer: str) -> bool:
        """Returns True if the given layer has no direction or is horizontal."""
        lay_id = self._grid.tech_info.get_layer_id(layer)
        return (lay_id is None) or self._grid.is_horizontal(lay_id)

    def add_rect(self, layer: str, purpose: str, bbox: BBox, commit: bool = True) -> PyRect:
        """Add a new rectangle.

        Parameters
        ----------
        layer: str
            the layer name.
        purpose: str
            the purpose name.
        bbox : BBox
            the rectangle bounding box.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        rect : PyRect
            the added rectangle.
        """
        return self._layout.add_rect(layer, purpose, bbox, commit)

    def add_rect_arr(self, layer: str, purpose: str, barr: BBoxArray) -> None:
        """Add a new rectangle array.

        Parameters
        ----------
        layer: str
            the layer name.
        purpose: str
            the purpose name.
        barr : BBoxArray
            the rectangle bounding box array.
        """
        self._layout.add_rect_arr(layer, purpose, barr)

    def add_res_metal(self, layer_id: int, bbox: BBox) -> None:
        """Add a new metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.
        bbox : BBox
            the resistor bounding box.
        """
        for lay, purp in self._grid.tech_info.get_res_metal_layers(layer_id):
            self._layout.add_rect(lay, purp, bbox, True)

    def add_path(self, layer: str, purpose: str, width: int, points: List[PointType],
                 start_style: PathStyle, *, join_style: PathStyle = PathStyle.round,
                 stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a new path.

        Parameters
        ----------
        layer : str
            the layer name.
        purpose : str
            the purpose name.
        width : int
            the path width.
        points : List[PointType]
            points defining this path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        half_width = width // 2
        return self._layout.add_path(layer, purpose, points, half_width, start_style,
                                     stop_style, join_style, commit)

    def add_path45_bus(self, layer: str, purpose: str, points: List[PointType], widths: List[int],
                       spaces: List[int], start_style: PathStyle, *,
                       join_style: PathStyle = PathStyle.round,
                       stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a path bus that only contains 45 degree turns.

        Parameters
        ----------
        layer : str
            the path layer.
        purpose : str
            the purpose name.
        points : List[PointType]
            points defining this path.
        widths : List[int]
            width of each path in the bus.
        spaces : List[int]
            space between each path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        return self._layout.add_path45_bus(layer, purpose, points, widths, spaces,
                                           start_style, stop_style, join_style, commit)

    def add_polygon(self, layer: str, purpose: str, points: List[PointType],
                    commit: bool = True) -> PyPolygon:
        """Add a new polygon.

        Parameters
        ----------
        layer : str
            the polygon layer.
        purpose: str
            the layer purpose.
        points : List[PointType]
            vertices of the polygon.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        polygon : PyPolygon
            the added polygon object.
        """
        return self._layout.add_poly(layer, purpose, points, commit)

    def add_blockage(self, layer: str, blk_type: BlockageType, points: List[PointType],
                     commit: bool = True) -> PyBlockage:
        """Add a new blockage object.

        Parameters
        ----------
        layer : str
            the layer name.
        blk_type : BlockageType
            the blockage type.
        points : List[PointType]
            vertices of the blockage object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        blockage : PyBlockage
            the added blockage object.
        """
        return self._layout.add_blockage(layer, blk_type, points, commit)

    def add_boundary(self, bnd_type: BoundaryType, points: List[PointType],
                     commit: bool = True) -> PyBoundary:
        """Add a new boundary.

        Parameters
        ----------
        bnd_type : str
            the boundary type.
        points : List[PointType]
            vertices of the boundary object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        boundary : PyBoundary
            the added boundary object.
        """
        return self._layout.add_boundary(bnd_type, points, commit)

    def reexport(self, port: Port, *,
                 net_name: str = '', label: str = '', show: bool = True) -> None:
        """Re-export the given port object.

        Add all geometries in the given port as pins with optional new name
        and label.

        Parameters
        ----------
        port : Port
            the Port object to re-export.
        net_name : str
            the new net name.  If not given, use the port's current net name.
        label : str
            the label.  If not given, use net_name.
        show : bool
            True to draw the pin in layout.
        """
        net_name = net_name or port.net_name
        if not label:
            if net_name != port.net_name:
                label = net_name
            else:
                label = port.label

        if net_name not in self._port_params:
            self._port_params[net_name] = dict(label=label, pins={}, show=show)

        port_params = self._port_params[net_name]
        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        # export all port geometries
        port_pins = port_params['pins']
        for wire_arr in port:
            layer_id = wire_arr.layer_id
            if layer_id not in port_pins:
                port_pins[layer_id] = [wire_arr]
            else:
                port_pins[layer_id].append(wire_arr)

    def add_pin_primitive(self, net_name: str, layer: str, bbox: BBox, *,
                          label: str = '', show: bool = True):
        """Add a primitive pin to the layout.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        layer : str
            the pin layer name.
        bbox : BBox
            the pin bounding box.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : bool
            True to draw the pin in layout.
        """
        label = label or net_name
        if net_name in self._prim_port_params:
            port_params = self._prim_port_params[net_name]
        else:
            port_params = self._prim_port_params[net_name] = dict(label=label, pins={}, show=show)

        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        port_pins = port_params['pins']

        if layer in port_pins:
            port_pins[layer].append(bbox)
        else:
            port_pins[layer] = [bbox]

    def add_label(self, label: str, layer: str, purpose: str, bbox: BBox) -> None:
        """Adds a label to the layout.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        layer : str
            the layer name.
        purpose : str
            the purpose name.
        bbox : BBox
            the label bounding box.
        """
        orient = Orientation.R90 if bbox.h > bbox.w else Orientation.R0
        xform = Transform(bbox.xm, bbox.ym, orient)
        self._layout.add_label(layer, purpose, xform, label)

    def add_pin(self, net_name: str, wire_arr_list: Union[WireArray, List[WireArray]],
                *, label: str = '', show: bool = True, edge_mode: int = 0) -> None:
        """Add new pin to the layout.

        If one or more pins with the same net name already exists,
        they'll be grouped under the same port.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        wire_arr_list : Union[WireArray, List[WireArray]]
            WireArrays representing the pin geometry.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : bool
            if True, draw the pin in layout.
        edge_mode : int
            If <0, draw the pin on the lower end of the WireArray.  If >0, draw the pin
            on the upper end.  If 0, draw the pin on the entire WireArray.
        """
        label = label or net_name

        if net_name not in self._port_params:
            self._port_params[net_name] = dict(label=label, pins={}, show=show)

        port_params = self._port_params[net_name]

        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        for warr in WireArray.wire_grp_iter(wire_arr_list):
            # add pin array to port_pins
            tid = warr.track_id
            layer_id = tid.layer_id
            if edge_mode != 0:
                # create new pin WireArray that's snapped to the edge
                cur_w = self.grid.get_wire_total_width(layer_id, tid.width)
                wl = warr.lower
                wu = warr.upper
                pin_len = min(cur_w * 2, wu - wl)
                if edge_mode < 0:
                    wu = wl + pin_len
                else:
                    wl = wu - pin_len
                warr = WireArray(tid, wl, wu)

            port_pins = port_params['pins']
            if layer_id not in port_pins:
                port_pins[layer_id] = [warr]
            else:
                port_pins[layer_id].append(warr)

    def add_via(self, bbox: BBox, bot_layer: str, top_layer: str, bot_dir: Orient2D, *,
                bot_purpose: str = '', top_purpose: str = '', extend: bool = True,
                top_dir: Optional[Orient2D] = None, add_layers: bool = False,
                commit: bool = True) -> PyVia:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        bbox : BBox
            the via bounding box, not including extensions.
        bot_layer : str
            the bottom layer name.
        top_layer : str
            the top layer name.
        bot_dir : Orient2D
            the bottom layer extension direction.
        bot_purpose : str
            bottom layer purpose.
        top_purpose : str
            top layer purpose.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.
        commit : bool
            True to commit via immediately.

        Returns
        -------
        via : PyVia
            the new via object.
        """
        tech_info = self._grid.tech_info
        via_info = tech_info.get_via_info(bbox, Direction.LOWER, bot_layer, top_layer,
                                          bot_dir, purpose=bot_purpose, adj_purpose=top_purpose,
                                          extend=extend, adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers ({}, {}) and ({}, {}) '
                             'with BBox: {}'.format(bot_layer, bot_purpose, top_layer, top_purpose,
                                                    bbox))

        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        return self._layout.add_via(xform, via_id, via_param, add_layers, commit)

    def add_via_arr(self, barr: BBoxArray, bot_layer: str, top_layer: str, bot_dir: Orient2D, *,
                    bot_purpose: str = '', top_purpose: str = '', extend: bool = True,
                    top_dir: Optional[Orient2D] = None, add_layers: bool = False) -> Dict[str, Any]:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        barr : BBoxArray
            the BBoxArray representing the via bounding boxes, not including extensions.
        bot_layer : str
            the bottom layer name.
        top_layer : str
            the top layer name.
        bot_dir : Orient2D
            the bottom layer extension direction.
        bot_purpose : str
            bottom layer purpose.
        top_purpose : str
            top layer purpose.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.

        Returns
        -------
        via_info : Dict[str, Any]
            the via information dictionary.
        """
        tech_info = self._grid.tech_info
        base_box = barr.base
        via_info = tech_info.get_via_info(base_box, Direction.LOWER, bot_layer, top_layer,
                                          bot_dir, purpose=bot_purpose, adj_purpose=top_purpose,
                                          extend=extend, adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers ({}, {}) and ({}, {}) '
                             'with BBox: {}'.format(bot_layer, bot_purpose, top_layer, top_purpose,
                                                    base_box))
        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        self._layout.add_via_arr(xform, via_id, via_param, add_layers, barr.nx, barr.ny,
                                 barr.spx, barr.spy)

        return via_info

    def add_via_primitive(self, via_type: str, xform: Transform, cut_width: int, cut_height: int,
                          *, num_rows: int = 1, num_cols: int = 1, sp_rows: int = 0,
                          sp_cols: int = 0, enc1: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          enc2: Tuple[int, int, int, int] = (0, 0, 0, 0), nx: int = 1, ny: int = 1,
                          spx: int = 0, spy: int = 0) -> None:
        """Adds via(s) by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        xform: Transform
            the transformation object.
        cut_width : CoordType
            via cut width.  This is used to create rectangle via.
        cut_height : CoordType
            via cut height.  This is used to create rectangle via.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : CoordType
            spacing between via cut rows.
        sp_cols : CoordType
            spacing between via cut columns.
        enc1 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on bottom layer.
            Defaults to all 0.
        enc2 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on top layer.
            Defaults to all 0.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : int
            column pitch.
        spy : int
            row pitch.
        """
        l1, r1, t1, b1 = enc1
        l2, r2, t2, b2 = enc2
        param = ViaParam(num_cols, num_rows, cut_width, cut_height, sp_cols, sp_rows,
                         l1, r1, t1, b1, l2, r2, t2, b2)
        self._layout.add_via_arr(xform, via_type, param, True, nx, ny, spx, spy)

    def add_via_on_grid(self, tid1: TrackID, tid2: TrackID, *, extend: bool = True) -> None:
        """Add a via on the routing grid.

        Parameters
        ----------
        tid1 : TrackID
            the first TrackID
        tid2 : TrackID
            the second TrackID
        extend : bool
            True to extend outside the via bounding box.
        """
        self._layout.add_via_on_intersection(WireArray(tid1, COORD_MIN, COORD_MAX),
                                             WireArray(tid2, COORD_MIN, COORD_MAX),
                                             extend, False)

    def extend_wires(self, warr_list: Union[WireArray, List[Optional[WireArray]]], *,
                     lower: Optional[int] = None, upper: Optional[int] = None,
                     min_len_mode: Optional[int] = None) -> List[Optional[WireArray]]:
        """Extend the given wires to the given coordinates.

        Parameters
        ----------
        warr_list : Union[WireArray, List[Optional[WireArray]]]
            the wires to extend.
        lower : Optional[int]
            the wire lower coordinate.
        upper : Optional[int]
            the wire upper coordinate.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.

        Returns
        -------
        warr_list : List[Optional[WireArray]]
            list of added wire arrays.
            If any elements in warr_list were None, they will be None in the return.
        """
        grid = self.grid

        new_warr_list = []
        for warr in WireArray.wire_grp_iter(warr_list):
            tid = warr.track_id
            if warr is None:
                new_warr_list.append(None)
            else:
                wlower = warr.lower
                wupper = warr.upper
                if lower is None:
                    cur_lower = wlower
                else:
                    cur_lower = min(lower, wlower)
                if upper is None:
                    cur_upper = wupper
                else:
                    cur_upper = max(upper, wupper)
                if min_len_mode is not None:
                    # extend track to meet minimum length
                    # make sure minimum length is even so that middle coordinate exists
                    min_len = grid.get_min_length(tid.layer_id, tid.width, even=True)
                    tr_len = cur_upper - cur_lower
                    if min_len > tr_len:
                        ext = min_len - tr_len
                        if min_len_mode < 0:
                            cur_lower -= ext
                        elif min_len_mode > 0:
                            cur_upper += ext
                        else:
                            cur_lower -= ext // 2
                            cur_upper = cur_lower + min_len

                new_warr = WireArray(tid, cur_lower, cur_upper)
                self._layout.add_warr(new_warr)
                new_warr_list.append(new_warr)

        return new_warr_list

    def add_wires(self, layer_id: int, track_idx: TrackType, lower: int, upper: int, *,
                  width: int = 1, num: int = 1, pitch: TrackType = 0) -> WireArray:
        """Add the given wire(s) to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        warr = WireArray(tid, lower, upper)
        self._layout.add_warr(warr)
        return warr

    def add_res_metal_warr(self, layer_id: int, track_idx: TrackType, lower: int, upper: int,
                           **kwargs: Any) -> WireArray:
        """Add metal resistor as WireArray to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        **kwargs :
            optional arguments to add_wires()

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        warr = self.add_wires(layer_id, track_idx, lower, upper, **kwargs)

        for _, _, box in warr.wire_iter(self.grid):
            self.add_res_metal(layer_id, box)

        return warr

    def add_mom_cap(self, cap_box: BBox, bot_layer: int, num_layer: int, *,
                    port_widths: Union[int, List[int], Dict[int, int]] = 1,
                    port_parity: Optional[Union[Tuple[int, int],
                                                Dict[int, Tuple[int, int]]]] = None,
                    array: bool = False,
                    **kwargs: Any) -> Dict[int, Tuple[List[WireArray], List[WireArray]]]:
        """Draw mom cap in the defined bounding box."""
        # TODO: port this method
        cap_rect_list = kwargs.get('return_cap_wires', None)
        cap_type = kwargs.get('cap_type', 'standard')

        if num_layer <= 1:
            raise ValueError('Must have at least 2 layers for MOM cap.')

        grid = self.grid
        tech_info = grid.tech_info

        mom_cap_dict = tech_info.tech_params['layout']['mom_cap'][cap_type]
        cap_margins = mom_cap_dict['margins']
        cap_info = mom_cap_dict['width_space']
        num_ports_on_edge = mom_cap_dict.get('num_ports_on_edge', {})
        port_widths_default = mom_cap_dict.get('port_widths_default', {})
        port_sp_min = mom_cap_dict.get('port_sp_min', {})

        top_layer = bot_layer + num_layer - 1

        if isinstance(port_widths, int):
            port_widths = {lay: port_widths for lay in range(bot_layer, top_layer + 1)}
        elif isinstance(port_widths, list) or isinstance(port_widths, tuple):
            if len(port_widths) != num_layer:
                raise ValueError('port_widths length != %d' % num_layer)
            port_widths = dict(zip(range(bot_layer, top_layer + 1), port_widths))
        else:
            port_widths = {lay: port_widths.get(lay, port_widths_default.get(lay, 1))
                           for lay in range(bot_layer, top_layer + 1)}

        if port_parity is None:
            port_parity = {lay: (0, 1) for lay in range(bot_layer, top_layer + 1)}
        elif isinstance(port_parity, tuple) or isinstance(port_parity, list):
            if len(port_parity) != 2:
                raise ValueError('port parity should be a tuple/list of 2 elements.')
            port_parity = {lay: port_parity for lay in range(bot_layer, top_layer + 1)}
        else:
            port_parity = {lay: port_parity.get(lay, (0, 1)) for lay in
                           range(bot_layer, top_layer + 1)}

        via_ext_dict = {lay: 0 for lay in range(bot_layer, top_layer + 1)}  # type: Dict[int, int]
        # get via extensions on each layer
        for vbot_layer in range(bot_layer, top_layer):
            vtop_layer = vbot_layer + 1
            bport_w = grid.get_track_width(vbot_layer, port_widths[vbot_layer])
            tport_w = grid.get_track_width(vtop_layer, port_widths[vtop_layer])
            bcap_w = cap_info[vbot_layer][0]
            tcap_w = cap_info[vtop_layer][0]

            # port-to-port via
            vbext1, vtext1 = grid.get_via_extensions_dim(vbot_layer, bport_w, tport_w)
            # cap-to-port via
            vbext2 = grid.get_via_extensions_dim(vbot_layer, bcap_w, tport_w)[0]
            # port-to-cap via
            vtext2 = grid.get_via_extensions_dim(vbot_layer, bport_w, tcap_w)[1]

            # record extension due to via
            via_ext_dict[vbot_layer] = max(via_ext_dict[vbot_layer], vbext1, vbext2)
            via_ext_dict[vtop_layer] = max(via_ext_dict[vtop_layer], vtext1, vtext2)

        # find port locations and cap boundaries.
        port_tracks = {}
        cap_bounds = {}
        cap_exts = {}
        for cur_layer in range(bot_layer, top_layer + 1):
            # mark bounding box as used.
            self.mark_bbox_used(cur_layer, cap_box)

            cur_num_ports = num_ports_on_edge.get(cur_layer, 1)
            cur_port_width = port_widths[cur_layer]
            cur_port_space = grid.get_num_space_tracks(cur_layer, cur_port_width,
                                                       half_space=True)
            dir_idx = grid.get_direction(cur_layer).value
            cur_lower, cur_upper = cap_box.get_interval(1 - dir_idx)
            # make sure adjacent layer via extension will not extend outside of cap bounding box.
            adj_via_ext = 0
            if cur_layer != bot_layer:
                adj_via_ext = via_ext_dict[cur_layer - 1]
            if cur_layer != top_layer:
                adj_via_ext = max(adj_via_ext, via_ext_dict[cur_layer + 1])
            # find track indices
            if array:
                tr_lower = grid.coord_to_track(cur_layer, cur_lower)
                tr_upper = grid.coord_to_track(cur_layer, cur_upper)
            else:
                tr_lower = grid.find_next_track(cur_layer, cur_lower + adj_via_ext,
                                                tr_width=cur_port_width,
                                                half_track=True, mode=1)
                tr_upper = grid.find_next_track(cur_layer, cur_upper - adj_via_ext,
                                                tr_width=cur_port_width,
                                                half_track=True, mode=-1)

            port_delta = cur_port_width + max(port_sp_min.get(cur_layer, 0), cur_port_space)
            if tr_lower + 2 * (cur_num_ports - 1) * port_delta >= tr_upper:
                raise ValueError('Cannot draw MOM cap; area too small.')

            ll0, lu0 = grid.get_wire_bounds(cur_layer, tr_lower, width=cur_port_width)
            ll1, lu1 = grid.get_wire_bounds(cur_layer,
                                            tr_lower + (cur_num_ports - 1) * port_delta,
                                            width=cur_port_width)
            ul0, uu0 = grid.get_wire_bounds(cur_layer,
                                            tr_upper - (cur_num_ports - 1) * port_delta,
                                            width=cur_port_width)
            ul1, uu1 = grid.get_wire_bounds(cur_layer, tr_upper, width=cur_port_width)

            # compute space from MOM cap wires to port wires
            port_w = lu0 - ll0
            lay_type = tech_info.get_layer_type_from_id(cur_layer)
            cur_margin = cap_margins[cur_layer]
            cur_margin = max(cur_margin, tech_info.get_min_space(lay_type, port_w))

            lower_tracks = [tr_lower + idx * port_delta for idx in range(cur_num_ports)]
            upper_tracks = [tr_upper - idx * port_delta for idx in range(cur_num_ports - 1, -1, -1)]
            port_tracks[cur_layer] = (lower_tracks, upper_tracks)
            cap_bounds[cur_layer] = (lu1 + cur_margin, ul0 - cur_margin)
            cap_exts[cur_layer] = (ll0, uu1)

        port_dict = {}
        cap_wire_dict = {}
        # draw ports/wires
        for cur_layer in range(bot_layer, top_layer + 1):
            cur_port_width = port_widths[cur_layer]
            # find port/cap wires lower/upper coordinates
            lower, upper = None, None
            if cur_layer != top_layer:
                lower, upper = cap_exts[cur_layer + 1]
            if cur_layer != bot_layer:
                tmpl, tmpu = cap_exts[cur_layer - 1]
                lower = tmpl if lower is None else min(lower, tmpl)
                upper = tmpu if upper is None else max(upper, tmpu)

            assert_msg = ('cur_layer is iterating and should never be equal to both '
                          'bot_layer and top_layer at the same time')
            assert lower is not None and upper is not None, assert_msg

            via_ext = via_ext_dict[cur_layer]
            lower -= via_ext
            upper += via_ext

            # draw lower and upper ports
            lower_tracks, upper_tracks = port_tracks[cur_layer]
            lower_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=cur_port_width)
                           for tr_idx in lower_tracks]
            upper_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=cur_port_width)
                           for tr_idx in upper_tracks]

            # assign port wires to positive/negative terminals
            lpar, upar = port_parity[cur_layer]
            if lpar == upar:
                raise ValueError('Port parity must be different.')
            elif lpar == 0:
                plist = upper_warrs
                nlist = lower_warrs
            else:
                plist = lower_warrs
                nlist = upper_warrs

            port_dict[cur_layer] = plist, nlist
            if cur_layer != bot_layer:
                # connect ports to layer below
                for clist, blist in zip((plist, nlist), port_dict[cur_layer - 1]):
                    if len(clist) == len(blist):
                        iter_list = zip(clist, blist)
                    else:
                        iter_list = product(clist, blist)

                    for cur_warr, bot_warr in iter_list:
                        cur_tid = cur_warr.track_id.base_index
                        cur_w = cur_warr.track_id.width
                        bot_tid = bot_warr.track_id.base_index
                        bot_w = bot_warr.track_id.width
                        self.add_via_on_grid(cur_layer - 1, bot_tid, cur_tid, bot_width=bot_w,
                                             top_width=cur_w)

            # draw cap wires
            cap_lower, cap_upper = cap_bounds[cur_layer]
            cap_tot_space = cap_upper - cap_lower
            cap_w, cap_sp = cap_info[cur_layer]
            cap_pitch = cap_w + cap_sp
            num_cap_wires = cap_tot_space // cap_pitch
            cap_lower += (cap_tot_space - (num_cap_wires * cap_pitch - cap_sp)) // 2

            cur_dir = grid.get_direction(cur_layer)
            wbox = BBox(cur_dir, lower, upper, cap_lower, cap_lower + cap_w)
            lay_purp_list = tech_info.get_lay_purp_list(cur_layer)

            # save cap wire information
            cur_rect_box = wbox
            cap_wire_dict[cur_layer] = (lpar, lay_purp_list, cur_rect_box, num_cap_wires, cap_pitch)

        # draw cap wires and connect to port
        for cur_layer in range(bot_layer, top_layer + 1):
            cur_rect_list = []
            lpar, lay_purp_list, cap_base_box, num_cap_wires, cap_pitch = cap_wire_dict[cur_layer]
            if cur_layer == bot_layer:
                prev_plist = prev_nlist = None
            else:
                prev_plist, prev_nlist = port_dict[cur_layer - 1]
            if cur_layer == top_layer:
                next_plist = next_nlist = None
            else:
                next_plist, next_nlist = port_dict[cur_layer + 1]

            cur_dir = grid.get_direction(cur_layer)
            next_dir = cur_dir.perpendicular()
            num_lay_purp = len(lay_purp_list)
            p_lists = (prev_plist, next_plist)
            n_lists = (prev_nlist, next_nlist)
            delta = 0
            for idx in range(num_cap_wires):
                # figure out the port wire to connect this cap wire to
                if idx % 2 == 0 and lpar == 0 or idx % 2 == 1 and lpar == 1:
                    ports_list = p_lists
                else:
                    ports_list = n_lists

                # draw the cap wire
                cap_lay, cap_purp = lay_purp_list[idx % num_lay_purp]
                cap_box = cap_base_box.get_move_by_orient(next_dir, delta)
                delta += cap_pitch
                rect = self.add_rect(cap_lay, cap_purp, cap_box)
                cur_rect_list.append(rect)

                # connect cap wire to port
                for pidx, port in enumerate(ports_list):
                    if port is not None:
                        port_warr = port[(idx // 2) % len(port)]
                        port_lay, port_purp = grid.get_layer_purpose(port_warr.layer_id,
                                                                     port_warr.track_id.base_index)
                        vbox = cap_box.intersect(port_warr.get_bbox_array(grid).base)
                        if pidx == 1:
                            self.add_via(vbox, cap_lay, port_lay, cur_dir,
                                         bot_purpose=cap_purp, top_purpose=port_purp)
                        else:
                            self.add_via(vbox, port_lay, cap_lay, next_dir,
                                         bot_purpose=port_purp, top_purpose=cap_purp)

            if cap_rect_list is not None:
                cap_rect_list.append(cur_rect_list)

        return port_dict

    def reserve_tracks(self, layer_id: int, track_idx: TrackType, *,
                       width: int = 1, num: int = 1, pitch: int = 0) -> None:
        """Reserve the given routing tracks so that power fill will not fill these tracks.

        Note: the size of this template should be set before calling this method.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.
        """
        # TODO: fix this method
        raise ValueError('Not implemented yet.')

    def connect_wires(self,  wire_arr_list: Union[WireArray, List[WireArray]], *,
                      lower: Optional[int] = None,
                      upper: Optional[int] = None,
                      debug: bool = False,
                      ) -> List[WireArray]:
        """Connect all given WireArrays together.

        all WireArrays must be on the same layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArr, List[WireArr]]
            WireArrays to connect together.
        lower : Optional[CoordType]
            if given, extend connection wires to this lower coordinate.
        upper : Optional[CoordType]
            if given, extend connection wires to this upper coordinate.
        debug : bool
            True to print debug messages.

        Returns
        -------
        conn_list : List[WireArray]
            list of connection wires created.
        """
        grid = self._grid

        # record all wire ranges
        layer_id = None
        intv_set = IntervalSet()
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):

            tid = wire_arr.track_id
            lay_id = tid.layer_id
            tr_w = tid.width
            if layer_id is None:
                layer_id = lay_id
            elif lay_id != layer_id:
                raise ValueError('WireArray layer ID != {}'.format(layer_id))

            cur_range = wire_arr.lower, wire_arr.upper
            for tidx in tid:
                intv = grid.get_wire_bounds(lay_id, tidx, width=tr_w)
                intv_rang_item = intv_set.get_first_overlap_item(intv)
                if intv_rang_item is None:
                    range_set = IntervalSet()
                    range_set.add(cur_range)
                    intv_set.add(intv, val=(range_set, tidx, tr_w))
                elif intv_rang_item[0] == intv:
                    tmp_rang_set: IntervalSet = intv_rang_item[1][0]
                    tmp_rang_set.add(cur_range, merge=True, abut=True)
                else:
                    raise ValueError('wire interval {} overlap existing wires.'.format(intv))

        # draw wires, group into arrays
        new_warr_list = []
        base_start = None  # type: Optional[int]
        base_end = None  # type: Optional[int]
        base_tidx = None  # type: Optional[HalfInt]
        base_width = None  # type: Optional[int]
        count = 0
        pitch = 0
        last_tidx = 0
        for set_item in intv_set.items():
            intv = set_item[0]
            range_set: IntervalSet = set_item[1][0]
            cur_tidx: HalfInt = set_item[1][1]
            cur_tr_w: int = set_item[1][2]
            cur_start = range_set.start
            cur_end = range_set.stop
            if lower is not None and lower < cur_start:
                cur_start = lower
            if upper is not None and upper > cur_end:
                cur_end = upper

            if debug:
                print('wires intv: %s, range: (%d, %d)' % (intv, cur_start, cur_end))
            if count == 0:
                base_tidx = cur_tidx
                base_start = cur_start
                base_end = cur_end
                base_width = cur_tr_w
                count = 1
                pitch = 0
            else:
                assert base_tidx is not None, "count == 0 should have set base_intv"
                assert base_width is not None, "count == 0 should have set base_width"
                assert base_start is not None, "count == 0 should have set base_start"
                assert base_end is not None, "count == 0 should have set base_end"
                if cur_start == base_start and cur_end == base_end and base_width == cur_tr_w:
                    # length and width matches
                    cur_pitch = cur_tidx - last_tidx
                    if count == 1:
                        # second wire, set half pitch
                        pitch = cur_pitch
                        count += 1
                    elif pitch == cur_pitch:
                        # pitch matches
                        count += 1
                    else:
                        # pitch does not match, add current wires and start anew
                        track_id = TrackID(layer_id, base_tidx, width=base_width,
                                           num=count, pitch=pitch)
                        warr = WireArray(track_id, base_start, base_end)
                        new_warr_list.append(warr)
                        self._layout.add_warr(warr)
                        base_tidx = cur_tidx
                        count = 1
                        pitch = 0
                else:
                    # length/width does not match, add cumulated wires and start anew
                    track_id = TrackID(layer_id, base_tidx, width=base_width,
                                       num=count, pitch=pitch)
                    warr = WireArray(track_id, base_start, base_end)
                    new_warr_list.append(warr)
                    self._layout.add_warr(warr)
                    base_start = cur_start
                    base_end = cur_end
                    base_tidx = cur_tidx
                    base_width = cur_tr_w
                    count = 1
                    pitch = 0

            # update last lower coordinate
            last_tidx = cur_tidx

        if base_tidx is None:
            # no wires given at all
            return []

        assert base_tidx is not None, "count == 0 should have set base_intv"
        assert base_start is not None, "count == 0 should have set base_start"
        assert base_end is not None, "count == 0 should have set base_end"

        # add last wires
        track_id = TrackID(layer_id, base_tidx, base_width, num=count, pitch=pitch)
        warr = WireArray(track_id, base_start, base_end)
        self._layout.add_warr(warr)
        new_warr_list.append(warr)
        return new_warr_list

    def connect_bbox_to_tracks(self, layer_dir: Direction, layer: str, purpose: str,
                               box_arr: Union[BBox, BBoxArray], track_id: TrackID, *,
                               track_lower: Optional[int] = None,
                               track_upper: Optional[int] = None,
                               min_len_mode: MinLenMode = MinLenMode.NONE,
                               wire_lower: Optional[int] = None,
                               wire_upper: Optional[int] = None) -> WireArray:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        layer_dir : Direction
            the primitive wire layer direction relative to the given tracks.  LOWER if
            the wires are below tracks, UPPER if the wires are above tracks.
        layer : str
            the primitive wire layer name.
        purpose : str
            the primitive wire purpose name.
        box_arr : Union[BBox, BBoxArray]
            bounding box of the wire(s) to connect to tracks.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            The minimum length extension mode.
        wire_lower : Optional[int]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[int]
            if given, extend wire(s) to this upper coordinate.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if isinstance(box_arr, BBox):
            box_arr = BBoxArray(box_arr)

        bnds = self._layout.connect_barr_to_tracks(layer_dir, layer, purpose, box_arr, track_id,
                                                   track_lower, track_upper, min_len_mode,
                                                   wire_lower, wire_upper)
        tr_idx = 1 - layer_dir.value
        return WireArray(track_id, bnds[tr_idx][0], bnds[tr_idx][1])

    def connect_bbox_to_differential_tracks(self, p_lay_dir: Direction, n_lay_dir: Direction,
                                            p_lay_purp: Tuple[str, str],
                                            n_lay_purp: Tuple[str, str],
                                            pbox: Union[BBox, BBoxArray],
                                            nbox: Union[BBox, BBoxArray], tr_layer_id: int,
                                            ptr_idx: TrackType, ntr_idx: TrackType, *,
                                            width: int = 1, track_lower: Optional[int] = None,
                                            track_upper: Optional[int] = None,
                                            min_len_mode: MinLenMode = MinLenMode.NONE
                                            ) -> DiffWarrType:
        """Connect the given differential primitive wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        p_lay_dir : Direction
            positive signal layer direction.
        n_lay_dir : Direction
            negative signal layer direction.
        p_lay_purp : Tuple[str, str]
            positive signal layer/purpose pair.
        n_lay_purp : Tuple[str, str]
            negative signal layer/purpose pair.
        pbox : Union[BBox, BBoxArray]
            positive signal wires to connect.
        nbox : Union[BBox, BBoxArray]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_bbox_to_matching_tracks([p_lay_dir, n_lay_dir],
                                                          [p_lay_purp, n_lay_purp], [pbox, nbox],
                                                          tr_layer_id, [ptr_idx, ntr_idx],
                                                          width=width, track_lower=track_lower,
                                                          track_upper=track_upper,
                                                          min_len_mode=min_len_mode)
        return track_list[0], track_list[1]

    def fix_track_min_length(self, tr_layer_id: int, width: int, track_lower: int, track_upper: int,
                             min_len_mode: MinLenMode) -> Tuple[int, int]:
        even = min_len_mode is MinLenMode.MIDDLE
        tr_len = max(track_upper - track_lower, self.grid.get_min_length(tr_layer_id, width,
                                                                         even=even))
        if min_len_mode is MinLenMode.LOWER:
            track_lower = track_upper - tr_len
        elif min_len_mode is MinLenMode.UPPER:
            track_upper = track_lower + tr_len
        elif min_len_mode is MinLenMode.MIDDLE:
            track_lower = (track_upper + track_lower - tr_len) // 2
            track_upper = track_lower + tr_len

        return track_lower, track_upper

    def connect_bbox_to_matching_tracks(self, lay_dir_list: List[Direction],
                                        lay_purp_list: List[Tuple[str, str]],
                                        box_arr_list: List[Union[BBox, BBoxArray]],
                                        tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                        width: int = 1, track_lower: Optional[int] = None,
                                        track_upper: Optional[int] = None,
                                        min_len_mode: MinLenMode = MinLenMode.NONE,
                                        ) -> List[Optional[WireArray]]:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        lay_dir_list : List[Direction]
            the primitive wire layer direction list.
        lay_purp_list : List[Tuple[str, str]]
            the primitive wire layer/purpose list.
        box_arr_list : List[Union[BBox, BBoxArray]]
            bounding box of the wire(s) to connect to tracks.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        Returns
        -------
        wire_arr : List[Optional[WireArray]]
            WireArrays representing the tracks created.
        """
        grid = self.grid
        tr_dir = grid.get_direction(tr_layer_id)
        w_dir = tr_dir.perpendicular()

        num = len(lay_dir_list)
        if len(lay_purp_list) != num or len(box_arr_list) != num or len(tr_idx_list) != num:
            raise ValueError('Connection list parameters have mismatch length.')
        if num == 0:
            raise ValueError('Connection lists are empty.')

        wl = None
        wu = None
        for lay_dir, (lay, purp), box_arr, tr_idx in zip(lay_dir_list, lay_purp_list,
                                                         box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)

            tid = TrackID(tr_layer_id, tr_idx, width=width)
            bnds = self._layout.connect_barr_to_tracks(lay_dir, lay, purp, box_arr, tid,
                                                       track_lower, track_upper, MinLenMode.NONE,
                                                       wl, wu)
            w_idx = lay_dir.value
            tr_idx = 1 - w_idx
            wl = bnds[w_idx][0]
            wu = bnds[w_idx][1]
            track_lower = bnds[tr_idx][0]
            track_upper = bnds[tr_idx][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for (lay, purp), box_arr, tr_idx in zip(lay_purp_list, box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)
            else:
                box_arr = BBoxArray(box_arr.base, tr_dir, nt=box_arr.get_num(tr_dir),
                                    spt=box_arr.get_sp(tr_dir))

            box_arr.set_interval(w_dir, wl, wu)
            self._layout.add_rect_arr(lay, purp, box_arr)

            warr = WireArray(TrackID(tr_layer_id, tr_idx, width=width), track_lower, track_upper)
            self._layout.add_warr(warr)
            ans.append(warr)

        return ans

    def connect_to_tracks(self, wire_arr_list: Union[WireArray, List[WireArray]],
                          track_id: TrackID, *, wire_lower: Optional[int] = None,
                          wire_upper: Optional[int] = None, track_lower: Optional[int] = None,
                          track_upper: Optional[int] = None, min_len_mode: MinLenMode = None,
                          ret_wire_list: Optional[List[WireArray]] = None,
                          debug: bool = False) -> Optional[WireArray]:
        """Connect all given WireArrays to the given track(s).

        All given wires should be on adjacent layers of the track.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        wire_lower : Optional[CoordType]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[CoordType]
            if given, extend wire(s) to this upper coordinate.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        ret_wire_list : Optional[List[WireArray]]
            If not none, extended wires that are created will be appended to this list.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Optional[WireArray]
            WireArray representing the tracks created.
        """
        # find min/max track Y coordinates
        tr_layer_id = track_id.layer_id
        tr_w = track_id.width

        # get top wire and bottom wire list
        warr_list_list = [[], []]
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):
            cur_layer_id = wire_arr.layer_id
            if cur_layer_id == tr_layer_id + 1:
                warr_list_list[1].append(wire_arr)
            elif cur_layer_id == tr_layer_id - 1:
                warr_list_list[0].append(wire_arr)
            else:
                raise ValueError(
                    'WireArray layer %d cannot connect to layer %d' % (cur_layer_id, tr_layer_id))

        if not warr_list_list[0] and not warr_list_list[1]:
            # no wires at all
            return None

        # connect wires together
        for warr in self.connect_wires(warr_list_list[0], lower=wire_lower, upper=wire_upper,
                                       debug=debug):
            bnds = self._layout.connect_warr_to_tracks(warr, track_id, None, None,
                                                       track_lower, track_upper)
            if ret_wire_list is not None:
                ret_wire_list.append(WireArray(warr.track_id, bnds[0][0], bnds[0][1]))
            track_lower = bnds[1][0]
            track_upper = bnds[1][1]
        for warr in self.connect_wires(warr_list_list[1], lower=wire_lower, upper=wire_upper,
                                       debug=debug):
            bnds = self._layout.connect_warr_to_tracks(warr, track_id, None, None,
                                                       track_lower, track_upper)
            if ret_wire_list is not None:
                ret_wire_list.append(WireArray(warr.track_id, bnds[1][0], bnds[1][1]))
            track_lower = bnds[0][0]
            track_upper = bnds[0][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, tr_w, track_lower,
                                                             track_upper, min_len_mode)
        result = WireArray(track_id, track_lower, track_upper)
        self._layout.add_warr(result)
        return result

    def connect_to_track_wires(self, wire_arr_list: Union[WireArray, List[WireArray]],
                               track_wires: Union[WireArray, List[WireArray]], *,
                               min_len_mode: Optional[int] = None,
                               debug: bool = False) -> Union[Optional[WireArray],
                                                             List[Optional[WireArray]]]:
        """Connect all given WireArrays to the given WireArrays on adjacent layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_wires : Union[WireArray, List[WireArray]]
            list of tracks as WireArrays.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Union[Optional[WireArray], List[Optional[WireArray]]]
            WireArrays representing the tracks created.  None if nothing to do.
        """
        ans = []  # type: List[Optional[WireArray]]
        for warr in WireArray.wire_grp_iter(track_wires):
            tr = self.connect_to_tracks(wire_arr_list, warr.track_id, track_lower=warr.lower,
                                        track_upper=warr.upper, min_len_mode=min_len_mode,
                                        debug=debug)
            ans.append(tr)

        if isinstance(track_wires, WireArray):
            return ans[0]
        return ans

    def strap_wires(self, warr: WireArray, targ_layer: int, tr_w_list: Optional[List[int]] = None,
                    min_len_mode_list: Optional[List[int]] = None) -> WireArray:
        """Strap the given WireArrays to the target routing layer.

        This method is used to connects wires on adjacent layers that has the same direction.
        The track locations must be valid on all routing layers for this method to work.

        Parameters
        ----------
        warr : WireArray
            the WireArrays to strap.
        targ_layer : int
            the final routing layer ID.
        tr_w_list : Optional[List[int]]
            the track widths to use on each layer.  If not specified, will determine automatically.
        min_len_mode_list : Optional[List[int]]
            minimum length mode flags on each layer.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.  None if nothing to do.
        """
        warr_layer = warr.layer_id

        if targ_layer == warr_layer:
            # no need to do anything
            return warr

        num_connections = abs(targ_layer - warr_layer)  # type: int

        # set default values
        if tr_w_list is None:
            tr_w_list = [-1] * num_connections
        elif len(tr_w_list) != num_connections:
            raise ValueError('tr_w_list must have exactly %d elements.' % num_connections)
        else:
            # create a copy of the given list, as this list may be modified later.
            tr_w_list = list(tr_w_list)

        if min_len_mode_list is None:
            min_len_mode_list_resolved = ([None] * num_connections)  # type: List[Optional[int]]
        else:
            # List[int] is a List[Optional[int]]
            min_len_mode_list_resolved = cast(List[Optional[int]], min_len_mode_list)

        if len(min_len_mode_list_resolved) != num_connections:
            raise ValueError('min_len_mode_list must have exactly %d elements.' % num_connections)

        layer_dir = 1 if targ_layer > warr_layer else -1
        for tr_w, mlen_mode in zip(tr_w_list, min_len_mode_list_resolved):
            warr = self._strap_wires_helper(warr, warr.layer_id + layer_dir, tr_w, mlen_mode)

        return warr

    def _strap_wires_helper(self, warr: WireArray, targ_layer: int, tr_w: int,
                            mlen_mode: Optional[int]) -> WireArray:
        """Helper method for strap_wires().  Connect one layer at a time."""
        # TODO: fix this
        grid = self._grid
        wire_tid = warr.track_id
        wire_layer = wire_tid.layer_id

        lower = warr.lower
        upper = warr.upper

        # error checking
        wdir = grid.get_direction(wire_layer)
        if wdir is not grid.get_direction(targ_layer):
            raise ValueError('Cannot strap wires with different directions.')

        # convert base track index
        base_coord = grid.track_to_coord(wire_layer, wire_tid.base_index)
        base_tid = grid.coord_to_track(targ_layer, base_coord)
        # convert pitch
        wire_pitch = grid.get_track_pitch(wire_layer)
        targ_pitch = grid.get_track_pitch(targ_layer)
        pitch_unit = wire_pitch * wire_tid.pitch
        if pitch_unit % (targ_pitch // 2) != 0:
            raise ValueError('Cannot strap wires on layers with mismatched pitch ')
        num_pitch = pitch_unit // targ_pitch
        # convert width
        if tr_w < 0:
            tmp_w = grid.get_track_width(wire_layer, wire_tid.width)
            tr_w = max(1, grid.get_track_width_inverse(targ_layer, tmp_w, mode=-1))

        # draw vias.  Update WireArray lower/upper
        new_lower = lower  # type: int
        new_upper = upper  # type: int
        w_lower = lower  # type: int
        w_upper = upper  # type: int
        for tid in wire_tid:
            coord = grid.track_to_coord(wire_layer, tid)
            tid2 = grid.coord_to_track(targ_layer, coord)
            wlay, wpurp = grid.get_layer_purpose(wire_layer, tid)
            tlay, tpurp = grid.get_layer_purpose(targ_layer, tid2)

            wlo, whi = grid.get_wire_bounds(wire_layer, tid, wire_tid.width)
            tlo, thi = grid.get_wire_bounds(targ_layer, tid2, tr_w)
            vbox = BBox(wdir, lower, upper, max(wlo, tlo), min(whi, thi))
            if wire_layer < targ_layer:
                via = self.add_via(vbox, wlay, tlay, wdir, bot_purpose=wpurp,
                                   top_purpose=tpurp, top_dir=wdir, extend=True)
                tbox, wbox = via.top_box, via.bottom_box
            else:
                via = self.add_via(vbox, tlay, wlay, wdir, bot_purpose=tpurp,
                                   top_purpose=wpurp, top_dir=wdir, extend=True)
                tbox, wbox = via.bottom_box, via.top_box

            new_lower = min(new_lower, tbox.get_coord(wdir, False))
            new_upper = max(new_upper, tbox.get_coord(wdir, True))
            w_lower = min(w_lower, wbox.get_coord(wdir, False))
            w_upper = max(w_upper, wbox.get_coord(wdir, True))

        # handle minimum length DRC rule
        min_len = grid.get_min_length(targ_layer, tr_w)
        ext = min_len - (new_upper - new_lower)
        if mlen_mode is not None and ext > 0:
            if mlen_mode < 0:
                new_lower -= ext
            elif mlen_mode > 0:
                new_upper += ext
            else:
                new_lower -= ext // 2
                new_upper += (ext - ext // 2)

        # add wires
        self.add_wires(wire_layer, wire_tid.base_index, w_lower, w_upper, width=wire_tid.width,
                       num=wire_tid.num, pitch=wire_tid.pitch)
        return self.add_wires(targ_layer, base_tid, new_lower, new_upper, width=tr_w,
                              num=wire_tid.num, pitch=num_pitch)

    def connect_differential_tracks(self, pwarr_list: Union[WireArray, List[WireArray]],
                                    nwarr_list: Union[WireArray, List[WireArray]],
                                    tr_layer_id: int, ptr_idx: TrackType, ntr_idx: TrackType, *,
                                    width: int = 1, track_lower: Optional[int] = None,
                                    track_upper: Optional[int] = None
                                    ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pwarr_list : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nwarr_list : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_matching_tracks([pwarr_list, nwarr_list], tr_layer_id,
                                                  [ptr_idx, ntr_idx], width=width,
                                                  track_lower=track_lower, track_upper=track_upper)
        return track_list[0], track_list[1]

    def connect_differential_wires(self, pin_warrs: Union[WireArray, List[WireArray]],
                                   nin_warrs: Union[WireArray, List[WireArray]],
                                   pout_warr: WireArray, nout_warr: WireArray, *,
                                   track_lower: Optional[int] = None,
                                   track_upper: Optional[int] = None
                                   ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two WireArrays symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pin_warrs : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nin_warrs : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        pout_warr : WireArray
            positive track wires.
        nout_warr : WireArray
            negative track wires.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        p_tid = pout_warr.track_id
        lay_id = p_tid.layer_id
        pidx = p_tid.base_index
        nidx = nout_warr.track_id.base_index
        width = p_tid.width

        if track_lower is None:
            tr_lower = pout_warr.lower
        else:
            tr_lower = min(track_lower, pout_warr.lower)
        if track_upper is None:
            tr_upper = pout_warr.upper
        else:
            tr_upper = max(track_upper, pout_warr.upper)

        return self.connect_differential_tracks(pin_warrs, nin_warrs, lay_id, pidx, nidx,
                                                width=width, track_lower=tr_lower,
                                                track_upper=tr_upper)

    def connect_matching_tracks(self, warr_list_list: List[Union[WireArray, List[WireArray]]],
                                tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                width: int = 1,
                                track_lower: Optional[int] = None,
                                track_upper: Optional[int] = None,
                                min_len_mode: MinLenMode = MinLenMode.NONE
                                ) -> List[Optional[WireArray]]:
        """Connect wires to tracks with optimal matching.

        This method connects the wires to tracks in a way that minimizes the parasitic mismatches.

        Parameters
        ----------
        warr_list_list : List[Union[WireArray, List[WireArray]]]
            list of signal wires to connect.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        track_list : List[WireArray]
            list of created tracks.
        """
        # simple error checking
        num_tracks = len(tr_idx_list)  # type: int
        if num_tracks != len(warr_list_list):
            raise ValueError('Connection list parameters have mismatch length.')
        if num_tracks == 0:
            raise ValueError('Connection lists are empty.')

        wbounds = [[None, None], [None, None]]
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            tid = TrackID(tr_layer_id, tr_idx, width=width)
            for warr in WireArray.wire_grp_iter(warr_list):
                cur_lay_id = warr.layer_id
                if cur_lay_id == tr_layer_id + 1:
                    wb_idx = 1
                elif cur_lay_id == tr_layer_id - 1:
                    wb_idx = 0
                else:
                    raise ValueError(
                        'WireArray layer {} cannot connect to layer {}'.format(cur_lay_id,
                                                                               tr_layer_id))

                bnds = self._layout.connect_warr_to_tracks(warr, tid, wbounds[wb_idx][0],
                                                           wbounds[wb_idx][1], track_lower,
                                                           track_upper)
                wbounds[wb_idx] = bnds[wb_idx]
                track_lower = bnds[1 - wb_idx][0]
                track_upper = bnds[1 - wb_idx][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            for warr in WireArray.wire_grp_iter(warr_list):
                wb_idx = (warr.layer_id - tr_layer_id + 1) // 2
                self._layout.add_warr(WireArray(warr.track_id, wbounds[wb_idx][0],
                                                wbounds[wb_idx][1]))

            warr = WireArray(TrackID(tr_layer_id, tr_idx, width=width), track_lower, track_upper)
            self._layout.add_warr(warr)
            ans.append(warr)

        return ans

    def draw_vias_on_intersections(self, bot_warr_list: Union[WireArray, List[WireArray]],
                                   top_warr_list: Union[WireArray, List[WireArray]]) -> None:
        """Draw vias on all intersections of the two given wire groups.

        Parameters
        ----------
        bot_warr_list : Union[WireArray, List[WireArray]]
            the bottom wires.
        top_warr_list : Union[WireArray, List[WireArray]]
            the top wires.
        """
        for bwarr in WireArray.wire_grp_iter(bot_warr_list):
            for twarr in WireArray.wire_grp_iter(top_warr_list):
                self._layout.add_via_on_intersection(bwarr, twarr, True, True)

    def has_blockage(self, layer_id: int, test_box: BBox, spx: int = 0, spy: int = 0) -> bool:
        """Returns true if there are blockage objects.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        test_box : BBox
            the BBox object.
        spx : int
            minimum horizontal spacing between objects and the given BBox.
        spy : int
            minimum vertical spacing between objects and the given BBox.

        Return
        ------
        has_blockage : bool
            True if some objects are too close to the given box.
        """
        # TODO: fix this
        return self._layout.has_blockage(layer_id, test_box, spx=spx, spy=spy)

    def blockage_iter(self, layer_id: int, test_box: BBox,
                      spx: int = 0, spy: int = 0) -> Iterable[GeoType]:
        """Returns all geometries that are too close to the given BBox.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        test_box : BBox
            the BBox object.
        spx : int
            minimum horizontal spacing between objects and the given BBox.
        spy : int
            minimum vertical spacing between objects and the given BBox.

        Yields
        ------
        obj : GeoType
            objects that are too close to the given BBox.
        """
        # TODO: fix this
        return self._layout.blockage_iter(layer_id, test_box, spx=spx, spy=spy)

    def is_track_available(self, layer_id: int, tr_idx: TrackType, lower: int, upper: int, *,
                           width: int = 1, sp: int = 0, sp_le: int = 0) -> bool:
        """Returns True if the given track is available.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the track ID.
        lower : int
            the lower track coordinate.
        upper : int
            the upper track coordinate.
        width : int
            the track width.
        sp : int
            required space around the track.
        sp_le : int
            required line-end space around the track.

        Returns
        -------
        available : bool
            True if the track is available.
        """
        # TODO: fix this
        grid = self._grid
        track_id = TrackID(layer_id, tr_idx, width=width)
        warr = WireArray(track_id, lower, upper)
        sp = max(sp, grid.get_space(layer_id, width))
        sp_le = max(sp_le, grid.get_line_end_space(layer_id, width))
        test_box = warr.get_bbox_array(grid).base
        wdir = grid.get_direction(layer_id)
        return self._layout.has_blockage_orient(layer_id, test_box, wdir, sp=sp, sp_le=sp_le)

    def mark_bbox_used(self, layer_id: int, bbox: BBox) -> None:
        """Marks the given bounding-box region as used in this Template."""
        # TODO: Fix this
        raise ValueError('Not implemented yet')

    def get_available_tracks(self, layer_id: int, tr_idx_list: List[Halfint], lower: int,
                             upper: int, *, width: int = 1, margin: int = 0) -> List[HalfInt]:
        """Returns empty tracks"""
        # TODO: fix this method
        raise ValueError('Not implemented yet')

    def do_power_fill(self, layer_id: int, space: int, space_le: int, *,
                      vdd_warrs: Optional[Union[WireArray, List[WireArray]]] = None,
                      vss_warrs: Optional[Union[WireArray, List[WireArray]]] = None,
                      bound_box: Optional[BBox] = None, fill_width: int = 1, fill_space: int = 0,
                      x_margin: int = 0, y_margin: int = 0, tr_offset: int = 0, min_len: int = 0,
                      flip: bool = False) -> Tuple[List[WireArray], List[WireArray]]:
        """Draw power fill on the given layer."""
        # TODO: fix this method
        raise ValueError('Not implemented yet')

    def do_max_space_fill(self, layer_id: int, bound_box: Optional[BBox] = None,
                          fill_pitch: TrackType = 1) -> None:
        """Draw density fill on the given layer."""
        # TODO: fix this method
        raise ValueError('Not implemented yet')


class BlackBoxTemplate(TemplateBase):
    """A black box template."""

    def __init__(self, temp_db: TemplateDB, params: Dict[str, Any], **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        self._sch_params = {}  # type: Dict[str, Any]

    @property
    def sch_params(self) -> Dict[str, Any]:
        return self._sch_params

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lib_name='The library name.',
            cell_name='The layout cell name.',
            top_layer='The top level layer.',
            size='The width/height of the cell, in resolution units.',
            ports='The port information dictionary.',
            show_pins='True to show pins.',
        )

    def get_layout_basename(self) -> str:
        return self.params['cell_name']

    def draw_layout(self) -> None:
        lib_name = self.params['lib_name']
        cell_name = self.params['cell_name']
        top_layer = self.params['top_layer']
        size = self.params['size']
        ports = self.params['ports']
        show_pins = self.params['show_pins']

        tech_info = self.grid.tech_info
        for term_name, pin_dict in ports.items():
            for lay, bbox_list in pin_dict.items():
                lay_id = tech_info.get_layer_id(lay)
                for xl, yb, xr, yt in bbox_list:
                    box = BBox(xl, yb, xr, yt)
                    self._register_pin(lay_id, lay, term_name, box, show_pins)

        self.add_instance_primitive(lib_name, cell_name)

        self.prim_top_layer = top_layer
        self.prim_bound_box = BBox(0, 0, size[0], size[1])

        for layer in range(1, top_layer + 1):
            self.mark_bbox_used(layer, self.prim_bound_box)

        self._sch_params = dict(
            lib_name=lib_name,
            cell_name=cell_name,
        )

    def _register_pin(self, lay_id: int, lay: str, term_name: str, box: BBox,
                      show_pins: bool) -> None:
        # TODO: fix this
        if lay_id is None:
            self.add_pin_primitive(term_name, lay, box, show=show_pins)
        else:
            dir_idx = self.grid.get_direction(lay_id).value
            dim = box.get_dim(1 - dir_idx)
            coord = box.get_center(1 - dir_idx)
            lower, upper = box.get_interval(dir_idx)

            try:
                tr_idx = self.grid.coord_to_track(lay_id, coord)
            except ValueError:
                self.add_pin_primitive(term_name, lay, box, show=show_pins)
                return

            width_ntr = self.grid.get_track_width_inverse(lay_id, dim)
            if self.grid.get_track_width(lay_id, width_ntr) == dim:
                track_id = TrackID(lay_id, tr_idx, width=width_ntr)
                warr = WireArray(track_id, lower, upper)
                self.add_pin(term_name, warr, show=show_pins)
            else:
                self.add_pin_primitive(term_name, lay, box, show=show_pins)
