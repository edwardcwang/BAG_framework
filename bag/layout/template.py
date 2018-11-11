# -*- coding: utf-8 -*-

"""This module defines layout template classes.
"""

from typing import TYPE_CHECKING, Union, Dict, Any, List, Set, TypeVar, Type, \
    Optional, Tuple, Iterable, Sequence, Callable, Generator, cast
from bag.typing import CoordType, LayerType, PointType

import abc
import copy
import time
from itertools import islice, product

import yaml

from bag.util.cache import DesignMaster, MasterDB
from bag.util.interval import IntervalSet
from ..io import get_encoding, open_file
from .routing.base import Port, TrackID, WireArray

# try to import cython modules
from pybag.enum import Orientation, PathStyle, BlockageType, BoundaryType, GeometryMode
from pybag.util.geometry import BBox, BBoxArray

if TYPE_CHECKING:
    from bag.core import BagProject
    from .routing.grid import RoutingGrid
    from bag.typing import TrackType, SizeType
    from pybag.layout import PyPath, PyBlockage, PyBoundary
    from pybag.layout import PyLayCellView, PyLayInstance, PyRect, PyVia
    from pybag.layout import PyPolygon90, PyPolygon45, PyPolygon

    GeoType = Union[PyRect, PyPolygon90, PyPolygon45, PyPolygon]
    TemplateType = TypeVar('TemplateType', bound='TemplateBase')

_io_encoding = get_encoding()


class TemplateDB(MasterDB):
    """A database of all templates.

    This class is responsible for keeping track of template libraries and
    creating new templates.

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
    gds_lay_file : str
        The GDS layer/purpose mapping file.
    flatten : bool
        True to compute flattened layout.
    **kwargs :
        additional arguments.
    """

    def __init__(self,  # type: TemplateDB
                 routing_grid,  # type: RoutingGrid
                 lib_name,  # type: str
                 prj=None,  # type: Optional[BagProject]
                 name_prefix='',  # type: str
                 name_suffix='',  # type: str
                 gds_lay_file='',  # type: str
                 flatten=False,  # type: bool
                 **kwargs):
        # type: (...) -> None
        MasterDB.__init__(self, lib_name, name_prefix=name_prefix, name_suffix=name_suffix)

        pure_oa = kwargs.get('pure_oa', False)

        self._prj = prj
        self._grid = routing_grid
        self._gds_lay_file = gds_lay_file
        self._flatten = flatten
        self._pure_oa = pure_oa

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[TemplateType], str, Dict[str, Any], Set[str], **Any) -> TemplateType
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[TemplateType]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs: Any
            optional arguments for the generator.

        Returns
        -------
        master : TemplateType
            the non-finalized generated instance.
        """
        # noinspection PyCallingNonCallable
        return gen_cls(self, lib_name, params, used_cell_names, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False, output='', **kwargs):
        # type: (str, Sequence[Any], bool, str, Any) -> None
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        # create layouts
        start = time.time()
        self._prj.instantiate_layout(lib_name, content_list)
        end = time.time()

        if debug:
            print('layout instantiation took %.4g seconds' % (end - start))

    @property
    def grid(self):
        # type: () -> RoutingGrid
        """Returns the default routing grid instance."""
        return self._grid

    def new_template(self, temp_cls, params=None, **kwargs):
        # type: (Type[TemplateType], Optional[Dict[str, Any]], **Any) -> TemplateType
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
        return self.new_master(temp_cls, params=params, **kwargs)

    def instantiate_layout(self, prj, template, top_cell_name=None, debug=False, rename_dict=None):
        # type: (BagProject, TemplateBase, Optional[str], bool, Optional[Dict[str, str]]) -> None
        """Instantiate the layout of the given :class:`~bag.layout.template.TemplateBase`.

        Parameters
        ----------
        prj : BagProject
            the :class:`~bag.BagProject` instance used to create layout.
        template : TemplateBase
            the :class:`~bag.layout.template.TemplateBase` to instantiate.
        top_cell_name : Optional[str]
            name of the top level cell.  If None, a default name is used.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        """
        self.batch_layout(prj, [template], [top_cell_name], debug=debug, rename_dict=rename_dict)

    def batch_layout(self,
                     prj,  # type: BagProject
                     template_list,  # type: Sequence[TemplateBase]
                     name_list=None,  # type: Optional[Sequence[Optional[str]]]
                     lib_name='',  # type: str
                     debug=False,  # type: bool
                     rename_dict=None,  # type: Optional[Dict[str, str]]
                     ):
        # type: (...) -> None
        """Instantiate all given templates.

        Parameters
        ----------
        prj : BagProject
            the :class:`~bag.BagProject` instance used to create layout.
        template_list : Sequence[TemplateBase]
            list of templates to instantiate.
        name_list : Optional[Sequence[Optional[str]]]
            list of template layout names.  If not given, default names will be used.
        lib_name : str
            Library to create the masters in.  If empty or None, use default library.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        """
        self._prj = prj
        self.instantiate_masters(template_list, name_list=name_list, lib_name=lib_name,
                                 debug=debug, rename_dict=rename_dict)


class TemplateBase(DesignMaster, metaclass=abc.ABCMeta):
    """The base template class.

    Parameters
    ----------
    temp_db : TemplateDB
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of the following optional parameters:

        grid : RoutingGrid
            the routing grid to use for this template.
        use_cybagoa : bool
            True to use cybagoa module to accelerate layout.

    Attributes
    ----------
    pins : dict
        the pins dictionary.
    children : List[str]
        a list of template cells this template uses.
    params : Dict[str, Any]
        the parameter values of this template.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None

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

        DesignMaster.__init__(self, temp_db, lib_name, params, used_names,
                              hidden_params=hidden_params)
        # update RoutingGrid
        fp_dict = self.params['flip_parity']
        if fp_dict is not None:
            self._grid.set_flip_parity(fp_dict)

        # create Cython wrapper object
        self._layout = PyLayCellView(self._grid.tech_info.pybag_tech, self.cell_name,
                                     get_encoding())

    @abc.abstractmethod
    def draw_layout(self):
        # type: () -> None
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        pass

    def populate_params(self, table, params_info, default_params, **kwargs):
        # type: (Dict[str, Any], Dict[str, str], Dict[str, Any], **Any) -> None
        """Fill params dictionary with values from table and default_params"""
        DesignMaster.populate_params(self, table, params_info, default_params, **kwargs)

        # add hidden parameters
        hidden_params = kwargs.get('hidden_params', {})
        for name, value in hidden_params.items():
            self.params[name] = table.get(name, value)

        # always add flip_parity parameter
        if 'flip_parity' not in self.params:
            self.params['flip_parity'] = table.get('flip_parity', None)
        # update RoutingGrid
        fp_dict = self.params['flip_parity']
        if fp_dict is not None:
            self._grid.set_flip_parity(fp_dict)

    def get_master_basename(self):
        # type: () -> str
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self.get_layout_basename()

    def get_layout_basename(self):
        # type: () -> str
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        return self.__class__.__name__

    def get_content(self, rename_fun):
        # type: (Callable[[str], str]) -> Tuple[str, PyLayCellView]
        """Returns the content of this master instance.

        Parameters
        ----------
        rename_fun : Callable[[str], str]
            a function that renames design masters.

        Returns
        -------
        content : Tuple[str, PyLayCellView]
            the content of this TemplateBase.
        """
        if not self.finalized:
            raise ValueError('This template is not finalized yet')

        return rename_fun(self._layout.cell_name), self._layout

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # create layout
        self.draw_layout()

        # finalize this template
        self.grid.tech_info.finalize_template(self)

        # construct port objects
        for net_name, port_params in self._port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for wire_arr_list in pin_dict.values():
                    for wire_arr in wire_arr_list:  # type: WireArray
                        for layer_name, bbox in wire_arr.wire_iter(self.grid):
                            self._layout.add_pin(net_name, layer_name, bbox, label)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # construct primitive port objects
        for net_name, port_params in self._prim_port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for layer_name, box_list in pin_dict.items():
                    for box in box_list:
                        self._layout.add_pin(net_name, layer_name, box, label)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # call super finalize routine
        DesignMaster.finalize(self)

    @property
    def template_db(self):
        # type: () -> TemplateDB
        """Returns the template database object"""
        # noinspection PyTypeChecker
        return self.master_db

    @property
    def is_empty(self):
        # type: () -> bool
        """Returns True if this template is empty."""
        return self._layout.is_empty

    @property
    def grid(self):
        # type: () -> RoutingGrid
        """Returns the RoutingGrid object"""
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        # type: (RoutingGrid) -> None
        """Change the RoutingGrid of this template."""
        if not self._finalized:
            self._grid = new_grid
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def array_box(self):
        # type: () -> Optional[BBox]
        """Returns the array/abutment bounding box of this template."""
        return self._array_box

    @array_box.setter
    def array_box(self, new_array_box):
        # type: (BBox) -> None
        """Sets the array/abutment bound box of this template."""
        if not self._finalized:
            self._array_box = new_array_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def fill_box(self):
        # type: () -> Optional[BBox]
        """Returns the dummy fill bounding box of this template."""
        return self._fill_box

    @fill_box.setter
    def fill_box(self, new_box):
        # type: (BBox) -> None
        """Sets the array/abutment bound box of this template."""
        if not self._finalized:
            self._fill_box = new_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def top_layer(self):
        # type: () -> int
        """Returns the top layer used in this template."""
        if self.size is None:
            if self.prim_top_layer is None:
                raise Exception('Both size and prim_top_layer are unset.')
            return self.prim_top_layer
        return self.size[0]

    @property
    def size(self):
        # type: () -> Optional[SizeType]
        """The size of this template, in (layer, num_x_block,  num_y_block) format."""
        return self._size

    @property
    def bound_box(self):
        # type: () -> Optional[BBox]
        """Returns the BBox with the size of this template.  None if size not set yet."""
        mysize = self.size
        if mysize is None:
            if self.prim_bound_box is None:
                raise ValueError('Both size and prim_bound_box are unset.')
            return self.prim_bound_box

        wblk, hblk = self.grid.get_size_dimension(mysize)
        return BBox(0, 0, wblk, hblk)

    @size.setter
    def size(self, new_size):
        # type: (SizeType) -> None
        """Sets the size of this template."""
        if not self._finalized:
            self._size = new_size
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def layout_cellview(self):
        # type: () -> PyLayCellView
        """Returns the internal layout object."""
        return self._layout

    def set_geometry_mode(self, mode):
        # type: (GeometryMode) -> None
        """Sets the geometry mode of this layout.

        Parameters
        ----------
        mode : GeometryMode
            the geometry mode.
        """
        self._layout.set_geometry_mode(mode.value)

    def get_rect_bbox(self, layer):
        # type: (LayerType) -> BBox
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.

        Parameters
        ----------
        layer : LayerType
            the layer name.

        Returns
        -------
        box : BBox
            the overall bounding box of the given layer.
        """
        return self._layout.get_rect_bbox(layer)

    def new_template_with(self, **kwargs):
        # type: (Any) -> TemplateBase
        """Create a new template with the given parameters.

        This method will update the parameter values with the given dictionary,
        then create a new template with those parameters and return it.

        Parameters
        ----------
        **kwargs
            a dictionary of new parameter values.
        """
        # get new parameter dictionary.
        new_params = copy.deepcopy(self.params)
        for key, val in kwargs.items():
            if key in new_params:
                new_params[key] = val

        return self.template_db.new_template(params=new_params, temp_cls=self.__class__,
                                             grid=self._parent_grid)

    def set_size_from_bound_box(self, top_layer_id, bbox, round_up=False,
                                half_blk_x=True, half_blk_y=True):
        # type: (int, BBox, bool, bool, bool) -> None
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

        if bbox.left_unit != 0 or bbox.bottom_unit != 0:
            raise ValueError('lower-left corner of overall bounding box must be (0, 0).')

        self.size = grid.get_size_tuple(top_layer_id, bbox.width_unit, bbox.height_unit,
                                        round_up=round_up, half_blk_x=half_blk_x,
                                        half_blk_y=half_blk_y)

    def set_size_from_array_box(self, top_layer_id):
        # type: (int) -> None
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

        dx = array_box.left_unit
        dy = array_box.bottom_unit
        if dx < 0 or dy < 0:
            raise ValueError('lower-left corner of array box must be in first quadrant.')

        self.size = grid.get_size_tuple(top_layer_id, 2 * dx + self.array_box.width_unit,
                                        2 * dy + self.array_box.height_unit)

    def write_summary_file(self, fname, lib_name, cell_name):
        # type: (str, str, str) -> None
        """Create a summary file for this template layout."""
        # get all pin information
        pin_dict = {}
        for port_name in self.port_names_iter():
            pin_cnt = 0
            port = self.get_port(port_name)
            for pin_warr in port:
                for layer_name, bbox in pin_warr.wire_iter(self.grid):
                    if pin_cnt == 0:
                        pin_name = port_name
                    else:
                        pin_name = '%s_%d' % (port_name, pin_cnt)
                    pin_cnt += 1
                    pin_dict[pin_name] = dict(
                        layer=[layer_name, self._grid.tech_info.pybag_tech.pin_purpose],
                        netname=port_name,
                        xy0=[bbox.left, bbox.bottom],
                        xy1=[bbox.right, bbox.top],
                    )

        # get size information
        bnd_box = self.bound_box
        if bnd_box is None:
            raise ValueError("bound_box is not set")
        res = self.grid.resolution
        info = {
            lib_name: {
                cell_name: dict(
                    pins=pin_dict,
                    xy0=[0.0, 0.0],
                    xy1=[bnd_box.width_unit * res, bnd_box.height_unit * res],
                ),
            },
        }

        with open_file(fname, 'w') as f:
            yaml.dump(info, f)

    def get_pin_name(self, name):
        # type: (str) -> str
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

    def get_port(self, name=''):
        # type: (str) -> Port
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

    def has_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this template has the given port."""
        return port_name in self._ports

    def port_names_iter(self):
        # type: () -> Iterable[str]
        """Iterates over port names in this template.

        Yields
        ------
        port_name : string
            name of a port in this template.
        """
        return self._ports.keys()

    def get_prim_port(self, name=''):
        # type: (str) -> Port
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

    def has_prim_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this template has the given primitive port."""
        return port_name in self._prim_ports

    def prim_port_names_iter(self):
        # type: () -> Iterable[str]
        """Iterates over primitive port names in this template.

        Yields
        ------
        port_name : str
            name of a primitive port in this template.
        """
        return self._prim_ports.keys()

    def new_template(self, params=None, temp_cls=None, debug=False, **kwargs):
        # type: (Dict[str, Any], Type[TemplateType], bool, **Any) -> TemplateType
        """Create a new template.

        Parameters
        ----------
        params : Dict[str, Any]
            the parameter dictionary.
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        debug : bool
            True to print debug messages.
        **kwargs : Any
            optional template parameters.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        kwargs['grid'] = self.grid
        return self.template_db.new_template(params=params, temp_cls=temp_cls, debug=debug,
                                             **kwargs)

    def move_all_by(self, dx=0, dy=0, unit_mode=True):
        # type: (int, int, bool) -> None
        """Move all layout objects Except pins in this layout by the given amount.

        Note: this method invalidates all WireArray objects and non-primitive pins.

        Parameters
        ----------
        dx : int
            the X shift.
        dy : int
            the Y shift.
        unit_mode : bool
            deprecated parameter.
        """
        # TODO: Implement this
        raise ValueError("Not implemented")

    def add_instance(self,  # type: TemplateBase
                     master,  # type: TemplateBase
                     inst_name=None,  # type: Optional[str]
                     loc=(0, 0),  # type: PointType
                     orient='R0',  # type: str
                     nx=1,  # type: int
                     ny=1,  # type: int
                     spx=0,  # type: CoordType
                     spy=0,  # type: CoordType
                     unit_mode=True,  # type: bool
                     commit=True,  # type: bool
                     ):
        # type: (...) -> PyLayInstance
        """Adds a new (arrayed) instance to layout.

        Parameters
        ----------
        master : TemplateBase
            the master template object.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        loc : PointType
            instance location.
        orient : str
            instance orientation.  Defaults to 'R0'.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        unit_mode : bool
            deprecated parameter.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        inst : PyLayInstance
            the added instance.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        return self._layout.add_instance(self.grid, self.children, master, self._lib_name,
                                         inst_name, loc[0], loc[1], Orientation[orient].value,
                                         nx, ny, spx, spy, commit=commit)

    def add_instance_primitive(self,  # type: TemplateBase
                               lib_name,  # type: str
                               cell_name,  # type: str
                               loc,  # type: PointType
                               view_name='layout',  # type: str
                               inst_name=None,  # type: Optional[str]
                               orient='R0',  # type: str
                               nx=1,  # type: int
                               ny=1,  # type: int
                               spx=0,  # type: CoordType
                               spy=0,  # type: CoordType
                               params=None,  # type: Optional[Dict[str, Any]]
                               **kwargs  # type: Any
                               ):
        # type: (...) -> None
        """Adds a new (arrayed) primitive instance to layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        loc : PointType
            instance location.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        orient : str
            instance orientation.  Defaults to "R0"
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
        **kwargs
            additional arguments.  Usually implementation specific.
        """
        if not params:
            params = kwargs
        else:
            params.update(kwargs)

        self._layout.add_prim_instance(lib_name, cell_name, view_name, inst_name, params, loc[0],
                                       loc[1], Orientation[orient].value, nx, ny, spx, spy)

    def is_horizontal(self, layer):
        # type: (LayerType) -> bool
        layer_name = layer if isinstance(layer, str) else layer[0]
        lay_id = self._grid.tech_info.get_layer_id(layer_name)
        return (lay_id is None) or self._grid.is_horizontal(lay_id)

    def add_rect(self, layer, bbox, commit=True):
        # type: (LayerType, BBox, bool) -> PyRect
        """Add a new rectangle.

        Parameters
        ----------
        layer: LayerType
            the layer name, or the (layer, purpose) pair.
        bbox : BBox
            the rectangle bounding box.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        rect : PyRect
            the added rectangle.
        """
        return self._layout.add_rect(layer, bbox, self.is_horizontal(layer), commit=commit)

    def add_rect_arr(self, layer, bbox, nx=1, ny=1, spx=0, spy=0):
        # type: (LayerType, BBox, int, int, CoordType, CoordType) -> None
        """Add a new rectangle.

        Parameters
        ----------
        layer: LayerType
            the layer name, or the (layer, purpose) pair.
        bbox : BBox
            the rectangle bounding box.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : int
            column pitch.
        spy : int
            row pitch.
        """
        self._layout.add_rect_arr(layer, bbox, self.is_horizontal(layer), nx, ny, spx, spy)

    def add_res_metal(self, layer_id, bbox, commit=True):
        # type: (int, BBox, bool) -> List[PyRect]
        """Add a new metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.
        bbox : BBox
            the resistor bounding box.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        rect_list : List[PyRect]
            list of rectangles defining the metal resistor.
        """
        is_horiz = self._grid.is_horizontal(layer_id)
        return [self._layout.add_rect(lay, bbox, is_horiz, commit=commit) for lay in
                self.grid.tech_info.get_res_metal_layers(layer_id)]

    def add_path(self, layer, width, points, start_style, join_style, stop_style='', commit=True):
        # type: (LayerType, int, Sequence[PointType], str, str, str, bool) -> PyPath
        """Add a new path.

        Parameters
        ----------
        layer : LayerType
            the path layer.
        width : int
            the path width.
        points : Sequence[PointType]
            points defining this path.
        start_style : str
            the path beginning style.
        join_style : str
            path style for the joints.
        stop_style : str
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        stop_style = stop_style or start_style
        return self._layout.add_path(layer, points, self.is_horizontal(layer), width,
                                     PathStyle[start_style].value, PathStyle[stop_style].value,
                                     PathStyle[join_style].value, commit=commit)

    def add_path45_bus(self,
                       layer,  # type: LayerType
                       points,  # type: Sequence[PointType]
                       widths,  # type: Sequence[int]
                       spaces,  # type: Sequence[int]
                       start_style,  # type: str
                       join_style='round',  # type: str
                       stop_style='',  # type: str
                       commit=True,  # type: bool
                       ):
        # type: (...) -> PyPath
        """Add a path bus that only contains 45 degree turns.

        Parameters
        ----------
        layer : LayerType
            the path layer.
        points : Sequence[PointType]
            points defining this path.
        widths : Sequence[int]
            width of each path in the bus.
        spaces : Sequence[int]
            space between each path.
        start_style : str
            the path beginning style.
        join_style : str
            path style for the joints.
        stop_style : str
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        stop_style = stop_style or start_style
        return self._layout.add_path45_bus(layer, points, widths, spaces,
                                           self.is_horizontal(layer), PathStyle[start_style].value,
                                           PathStyle[stop_style].value, PathStyle[join_style].value,
                                           commit=commit)

    def add_polygon(self, layer, points, commit=True):
        # type: (LayerType, Sequence[PointType], bool) -> PyPolygon
        """Add a new polygon.

        Parameters
        ----------
        layer : LayerType
            the polygon layer.
        points : Sequence[PointType]
            vertices of the polygon.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        polygon : PyPolygon
            the added polygon object.
        """
        return self._layout.add_polygon(layer, points, self.is_horizontal(layer), commit=commit)

    def add_blockage(self, layer, blk_type, points, commit=True):
        # type: (LayerType, str, Sequence[PointType], bool) -> PyBlockage
        """Add a new polygon.

        Parameters
        ----------
        layer : LayerType
            the path layer.
        blk_type : str
            the blockage type.
        points : Sequence[PointType]
            vertices of the blockage object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        blockage : PyBlockage
            the added blockage object.
        """
        return self._layout.add_blockage(layer, points, BlockageType[blk_type].value,
                                         commit=commit)

    def add_cell_boundary(self, box):
        # type: (BBox) -> None
        """Adds a cell boundary object to the this template.

        This is usually the PR boundary.

        Parameters
        ----------
        box : BBox
            the cell boundary bounding box.
        """
        self._grid.tech_info.add_cell_boundary(self, box)

    def add_boundary(self, bnd_type, points, commit=True):
        # type: (str, Sequence[PointType], bool) -> PyBoundary
        """Add a new boundary.

        Parameters
        ----------
        bnd_type : str
            the boundary type.
        points : Sequence[PointType]
            vertices of the boundary object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        boundary : PyBoundary
            the added boundary object.
        """
        return self._layout.add_boundary(points, BoundaryType[bnd_type].value, commit=commit)

    def reexport(self, port, net_name='', label='', show=True):
        # type: (Port, str, str, bool) -> None
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

    def add_pin_primitive(self, net_name, layer, bbox, label='', show=True):
        # type: (str, str, BBox, str, bool) -> None
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

    def add_label(self, label, layer, bbox):
        # type: (str, LayerType, BBox) -> None
        """Adds a label to the layout.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        layer : LayerType
            the pin layer name.
        bbox : BBox
            the pin bounding box.
        """
        # TODO: Implement this
        raise ValueError('Not implemented yet.')

    def add_pin(self, net_name, wire_arr_list, label='', show=True, edge_mode=0):
        # type: (str, Union[WireArray, List[WireArray]], str, bool, int) -> None
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
        edge_mode : int
            If <0, draw the pin on the lower end of the WireArray.  If >0, draw the pin
            on the upper end.  If 0, draw the pin on the entire WireArray.
        show : bool
            if True, draw the pin in layout.
        """
        if isinstance(wire_arr_list, WireArray):
            wire_arr_list = [wire_arr_list]
        else:
            pass

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

        for warr in wire_arr_list:
            # add pin array to port_pins
            layer_id = warr.track_id.layer_id
            if edge_mode != 0:
                cur_w = self.grid.get_track_width(layer_id, warr.track_id.width)
                wl = warr.lower_unit
                wu = warr.upper_unit
                pin_len = min(cur_w * 2, wu - wl)
                if edge_mode < 0:
                    wu = wl + pin_len
                else:
                    wl = wu - pin_len
                warr = WireArray(warr.track_id, wl, wu)

            port_pins = port_params['pins']
            if layer_id not in port_pins:
                port_pins[layer_id] = [warr]
            else:
                port_pins[layer_id].append(warr)

    def add_via(self,  # type: TemplateBase
                bbox,  # type: BBox
                bot_layer,  # type: LayerType
                top_layer,  # type: LayerType
                bot_dir,  # type: str
                nx=1,  # type: int
                ny=1,  # type: int
                spx=0,  # type: CoordType
                spy=0,  # type: CoordType
                extend=True,  # type: bool
                top_dir=None,  # type: Optional[str]
                add_layers=False,  # type: bool
                unit_mode=True,  # type: bool
                commit=True,  # type: bool
                ):
        # type: (...) -> PyVia
        """Adds a (arrayed) via object to the layout.

        Parameters
        ----------
        bbox : BBox
            the via bounding box, not including extensions.
        bot_layer : LayerType
            the bottom layer name, or a tuple of layer name and purpose name.
            If purpose name not given, use default purpose.
        top_layer : LayerType
            the top layer name, or a tuple of layer name and purpose name.
            If purpose name not given, use default purpose.
        bot_dir : str
            the bottom layer extension direction.  Either 'x' or 'y'.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : CoordType
            column pitch.
        spy : CoordType
            row pitch.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[str]
            top layer extension direction.  Can force to extend in same direction as bottom.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.
        unit_mode : bool
            deprecated parameter.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        via : PyVia
            the via object at position (0, 0) of the array.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        params = self._grid.tech_info.get_via_info(bbox, bot_layer, top_layer, bot_dir,
                                                   top_dir=top_dir, extend=extend)['params']
        loc = params['loc']
        orient = params['orient']
        cut_width = params['cut_width']
        cut_height = params['cut_height']
        cnx = params['num_cols']
        cny = params['num_rows']
        cspx = params['sp_cols']
        cspy = params['sp_rows']
        enc1 = params['enc1']
        enc2 = params['enc2']

        bot_horiz = self.is_horizontal(bot_layer)
        top_horiz = self.is_horizontal(top_layer)
        return self._layout.add_via_arr(params['id'], enc1, enc2, loc[0], loc[1],
                                        Orientation[orient].value, cut_width, cut_height,
                                        cnx, cny, cspx, cspy, nx, ny, spx, spy, add_layers,
                                        bot_horiz, top_horiz, commit=commit)

    def add_via_primitive(self, via_type,  # type: str
                          loc,  # type: PointType
                          cut_width,  # type: Optional[CoordType]
                          cut_height,  # type: Optional[CoordType]
                          num_rows=1,  # type: int
                          num_cols=1,  # type: int
                          sp_rows=0,  # type: CoordType
                          sp_cols=0,  # type: CoordType
                          enc1=None,  # type: Optional[List[CoordType]]
                          enc2=None,  # type: Optional[List[CoordType]]
                          orient='R0',  # type: str
                          nx=1,  # type: int
                          ny=1,  # type: int
                          spx=0,  # type: CoordType
                          spy=0,  # type: CoordType
                          unit_mode=True,  # type: bool
                          commit=True,  # type: bool
                          ):
        # type: (...) -> PyVia
        """Adds via(s) by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        loc : PointType
            the via location as a two-element list.
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
        orient : str
            orientation of the via.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : CoordType
            column pitch.
        spy : CoordType
            row pitch.
        unit_mode : bool
            Deprecated parameter.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        via : PyVia
            the via object at position (0, 0) of the array.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        default_enc = [0, 0, 0, 0]
        enc1 = enc1 or default_enc
        enc2 = enc2 or default_enc
        return self._layout.add_via_arr(via_type, enc1, enc2, loc[0], loc[1],
                                        Orientation[orient].value, cut_width, cut_height, num_cols,
                                        num_rows, sp_cols, sp_rows, nx, ny, spx, spy, False,
                                        False, False, commit=commit)

    def add_via_on_grid(self, bot_layer_id, bot_track, top_track, bot_width=1, top_width=1,
                        commit=True):
        # type: (int, TrackType, TrackType, int, int, bool) -> PyVia
        """Add a via on the routing grid.

        Parameters
        ----------
        bot_layer_id : int
            the bottom layer ID.
        bot_track : TrackType
            the bottom track index.
        top_track : TrackType
            the top track index.
        bot_width : int
            the bottom track width.
        top_width : int
            the top track width.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        via : PyVia
            the via object created.
        """
        grid = self.grid
        bl, bu = grid.get_wire_bounds(bot_layer_id, bot_track, width=bot_width)
        tl, tu = grid.get_wire_bounds(bot_layer_id + 1, top_track, width=top_width)
        bot_dir = grid.get_direction(bot_layer_id)
        if bot_dir == 'x':
            bbox = BBox(tl, bl, tu, bu)
        else:
            bbox = BBox(bl, tl, bu, tu)
        bname = grid.get_layer_name(bot_layer_id, bot_track)
        tname = grid.get_layer_name(bot_layer_id + 1, top_track)

        return self.add_via(bbox, bname, tname, bot_dir, commit=commit)

    def extend_wires(self,  # type: TemplateBase
                     warr_list,  # type: Union[WireArray, List[Optional[WireArray]]]
                     lower=None,  # type: Optional[CoordType]
                     upper=None,  # # type: Optional[CoordType]
                     unit_mode=True,  # type: bool
                     min_len_mode=None,  # type: Optional[int]
                     ):
        # type: (...) -> List[Optional[WireArray]]
        """Extend the given wires to the given coordinates.

        Parameters
        ----------
        warr_list : Union[WireArray, List[Optional[WireArray]]]
            the wires to extend.
        lower : Optional[CoordType]
            the wire lower coordinate.
        upper : Optional[CoordType]
            the wire upper coordinate.
        unit_mode: bool
            deprecated parameter
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.

        Returns
        -------
        warr_list : List[Optional[WireArray]]
            list of added wire arrays.
            If any elements in warr_list were None, they will be None in the return.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        if isinstance(warr_list, WireArray):
            warr_list = [warr_list]
        else:
            pass

        new_warr_list = []
        for warr in warr_list:
            if warr is None:
                new_warr_list.append(None)
            else:
                wlower = warr.lower_unit
                wupper = warr.upper_unit
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
                    min_len = self.grid.get_min_length(warr.layer_id, warr.track_id.width)
                    # make sure minimum length is even so that middle coordinate exists
                    min_len = -(-min_len // 2) * 2
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

                new_warr = WireArray(warr.track_id, cur_lower, cur_upper)
                for layer_name, bbox_arr in new_warr.wire_arr_iter(self.grid):
                    self.add_rect_arr(layer_name, bbox_arr.base, nx=bbox_arr.nx, ny=bbox_arr.ny,
                                      spx=bbox_arr.spx_unit, spy=bbox_arr.spy_unit)

                new_warr_list.append(new_warr)

        return new_warr_list

    def add_wires(self,  # type: TemplateBase
                  layer_id,  # type: int
                  track_idx,  # type: TrackType
                  lower,  # type: CoordType
                  upper,  # type: CoordType
                  width=1,  # type: int
                  num=1,  # type: int
                  pitch=0,  # type: TrackType
                  unit_mode=True  # type: bool
                  ):
        # type: (...) -> WireArray
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
        unit_mode: bool
            deprecated parameter.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        warr = WireArray(tid, lower, upper)

        for layer_name, bbox_arr in warr.wire_arr_iter(self.grid):
            self.add_rect_arr(layer_name, bbox_arr.base, nx=bbox_arr.nx, ny=bbox_arr.ny,
                              spx=bbox_arr.spx_unit, spy=bbox_arr.spy_unit)

        return warr

    def add_res_metal_warr(self,  # type: TemplateBase
                           layer_id,  # type: int
                           track_idx,  # type: TrackType
                           lower,  # type: CoordType
                           upper,  # type: CoordType
                           **kwargs):
        # type: (...) -> WireArray
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

        for _, bbox_arr in warr.wire_arr_iter(self.grid):
            for bbox in bbox_arr:
                self.add_res_metal(layer_id, bbox)

        return warr

    def add_mom_cap(self,  # type: TemplateBase
                    cap_box,  # type: BBox
                    bot_layer,  # type: int
                    num_layer,  # type: int
                    port_widths=1,  # type: Union[int, List[int], Dict[int, int]]
                    port_parity=None,
                    # type: Optional[Union[Tuple[int, int], Dict[int, Tuple[int, int]]]]
                    array=False,  # type: bool
<<<<<<< HEAD
                    **kwargs,  # type: Any
=======
                    **kwargs
>>>>>>> master
                    ):
        # type: (...) -> Any
        """Draw mom cap in the defined bounding box."""
        return_rect = kwargs.get('return_cap_wires', False)
        cap_type = kwargs.get('cap_type', 'standard')

        if num_layer <= 1:
            raise ValueError('Must have at least 2 layers for MOM cap.')

        tech_info = self.grid.tech_info

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
            bport_w = self.grid.get_track_width(vbot_layer, port_widths[vbot_layer])
            tport_w = self.grid.get_track_width(vtop_layer, port_widths[vtop_layer])
            bcap_w = cap_info[vbot_layer][0]
            tcap_w = cap_info[vtop_layer][0]

            # port-to-port via
            vbext1, vtext1 = self.grid.get_via_extensions_dim(vbot_layer, bport_w, tport_w)
            # cap-to-port via
            vbext2 = self.grid.get_via_extensions_dim(vbot_layer, bcap_w, tport_w)[0]
            # port-to-cap via
            vtext2 = self.grid.get_via_extensions_dim(vbot_layer, bport_w, tcap_w)[1]

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
            cur_port_space = self.grid.get_num_space_tracks(cur_layer, cur_port_width,
                                                            half_space=True)
            if self.grid.get_direction(cur_layer) == 'x':
                cur_lower, cur_upper = cap_box.bottom_unit, cap_box.top_unit
            else:
                cur_lower, cur_upper = cap_box.left_unit, cap_box.right_unit
            # make sure adjacent layer via extension will not extend outside of cap bounding box.
            adj_via_ext = 0
            if cur_layer != bot_layer:
                adj_via_ext = via_ext_dict[cur_layer - 1]
            if cur_layer != top_layer:
                adj_via_ext = max(adj_via_ext, via_ext_dict[cur_layer + 1])
            # find track indices
            if array:
                tr_lower = self.grid.coord_to_track(cur_layer, cur_lower)
                tr_upper = self.grid.coord_to_track(cur_layer, cur_upper)
            else:
                tr_lower = self.grid.find_next_track(cur_layer, cur_lower + adj_via_ext,
                                                     tr_width=cur_port_width,
                                                     half_track=True, mode=1)
                tr_upper = self.grid.find_next_track(cur_layer, cur_upper - adj_via_ext,
                                                     tr_width=cur_port_width,
                                                     half_track=True, mode=-1)

            port_delta = cur_port_width + max(port_sp_min.get(cur_layer, 0), cur_port_space)
            if tr_lower + 2 * (cur_num_ports - 1) * port_delta >= tr_upper:
                raise ValueError('Cannot draw MOM cap; area too small.')

            ll0, lu0 = self.grid.get_wire_bounds(cur_layer, tr_lower, width=cur_port_width)
            ll1, lu1 = self.grid.get_wire_bounds(cur_layer,
                                                 tr_lower + (cur_num_ports - 1) * port_delta,
                                                 width=cur_port_width)
            ul0, uu0 = self.grid.get_wire_bounds(cur_layer,
                                                 tr_upper - (cur_num_ports - 1) * port_delta,
                                                 width=cur_port_width)
            ul1, uu1 = self.grid.get_wire_bounds(cur_layer, tr_upper, width=cur_port_width)

            # compute space from MOM cap wires to port wires
            port_w = lu0 - ll0
            lay_name = tech_info.get_layer_name(cur_layer)
            if isinstance(lay_name, tuple) or isinstance(lay_name, list):
                lay_name = lay_name[0]
            lay_type = tech_info.get_layer_type(lay_name)
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

            is_horizontal = self.grid.is_horizontal(cur_layer)

            if is_horizontal:
                wbox = BBox(lower, cap_lower, upper, cap_lower + cap_w)
            else:
                wbox = BBox(cap_lower, lower, cap_lower + cap_w, upper)

            lay_name_list = tech_info.get_layer_name(cur_layer)
            if isinstance(lay_name_list, str):
                lay_name_list = [lay_name_list]

            # save cap wire information
            cur_rect_box = wbox
            cap_wire_dict[cur_layer] = (lpar, lay_name_list, cur_rect_box, num_cap_wires, cap_pitch)

        # draw cap wires and connect to port
        rect_list = []
        for cur_layer in range(bot_layer, top_layer + 1):
            cur_rect_list = []
            lpar, lay_name_list, cap_base_box, num_cap_wires, cap_pitch = cap_wire_dict[cur_layer]
            if cur_layer == bot_layer:
                prev_plist = prev_nlist = None
            else:
                prev_plist, prev_nlist = port_dict[cur_layer - 1]
            if cur_layer == top_layer:
                next_plist = next_nlist = None
            else:
                next_plist, next_nlist = port_dict[cur_layer + 1]

            cur_dir = self.grid.get_direction(cur_layer)
            is_horizontal = (cur_dir == 'x')
            next_dir = 'y' if is_horizontal else 'x'
            num_lay_names = len(lay_name_list)
            p_lists = (prev_plist, next_plist)
            n_lists = (prev_nlist, next_nlist)
            for idx in range(num_cap_wires):
                # figure out the port wire to connect this cap wire to
                if idx % 2 == 0 and lpar == 0 or idx % 2 == 1 and lpar == 1:
                    ports_list = p_lists
                else:
                    ports_list = n_lists

                # draw the cap wire
                cap_lay_name = lay_name_list[idx % num_lay_names]
                if is_horizontal:
                    cap_box = cap_base_box.move_by(dy=cap_pitch * idx)
                else:
                    cap_box = cap_base_box.move_by(dx=cap_pitch * idx)
                rect = self.add_rect(cap_lay_name, cap_box)
                cur_rect_list.append(rect)

                # connect cap wire to port
                for pidx, port in enumerate(ports_list):
                    if port is not None:
                        port_warr = port[(idx // 2) % len(port)]
                        port_lay_name = self.grid.get_layer_name(port_warr.layer_id,
                                                                 port_warr.track_id.base_index)
                        vbox = cap_box.intersect(port_warr.get_bbox_array(self.grid).base)
                        if pidx == 1:
                            self.add_via(vbox, cap_lay_name, port_lay_name, cur_dir)
                        else:
                            self.add_via(vbox, port_lay_name, cap_lay_name, next_dir)

            rect_list.append(cur_rect_list)

        if return_rect:
            return port_dict, rect_list
        else:
            return port_dict

    def reserve_tracks(self,  # type: TemplateBase
                       layer_id,  # type: int
                       track_idx,  # type: TrackType
                       width=1,  # type: int
                       num=1,  # type: int
                       pitch=0,  # type: TrackType
                       ):
        # type: (...) -> None
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
        bnd_box = self.bound_box
        if bnd_box is None:
            raise ValueError("bound_box is not set")

        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        if self.grid.get_direction(layer_id) == 'x':
            upper = bnd_box.width_unit
        else:
            upper = bnd_box.height_unit
        warr = WireArray(tid, 0, upper)

        lay_name = self.grid.get_layer_name(layer_id, track_idx)
        # self._used_tracks.record_rect(self.grid, lay_name, warr.get_bbox_array(self.grid))
        raise ValueError('Not implemented yet.')

    def connect_wires(self,  # type: TemplateBase
                      wire_arr_list,  # type: Union[WireArray, List[WireArray]]
                      lower=None,  # type: Optional[CoordType]
                      upper=None,  # type: Optional[CoordType]
                      debug=False,  # type: bool
                      unit_mode=True,  # type: bool
                      ):
        # type: (...) -> List[WireArray]
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
        unit_mode: bool
            deprecated parameter.

        Returns
        -------
        conn_list : List[WireArray]
            list of connection wires created.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        grid = self.grid

        if isinstance(wire_arr_list, WireArray):
            wire_arr_list = [wire_arr_list]
        else:
            pass

        if not wire_arr_list:
            # do nothing
            return []

        # record all wire ranges
        a = wire_arr_list[0]
        layer_id = a.layer_id
        direction = grid.get_direction(layer_id)
        is_horiz = direction == 'x'
        perp_dir = 'y' if direction == 'x' else 'x'
        htr_pitch = grid.get_track_pitch(layer_id) // 2
        intv_set = IntervalSet()
        for wire_arr in wire_arr_list:
            if wire_arr.layer_id != layer_id:
                raise ValueError('WireArray layer ID != %d' % layer_id)

            cur_range = wire_arr.lower_unit, wire_arr.upper_unit
            box_arr = wire_arr.get_bbox_array(grid)
            for box in box_arr:
                intv = box.get_interval(perp_dir)
                intv_rang_item = intv_set.get_first_overlap_item(intv)
                if intv_rang_item is None:
                    range_set = IntervalSet()
                    range_set.add(cur_range)
                    intv_set.add(intv, val=range_set)
                elif intv_rang_item[0] == intv:
                    intv_rang_item[1].add(cur_range, merge=True, abut=True)
                else:
                    raise ValueError('wire interval {} overlap existing wires.'.format(intv))

        # draw wires, group into arrays
        new_warr_list = []
        base_start = None  # type: Optional[int]
        base_end = None  # type: Optional[int]
        base_intv = None  # type: Optional[Tuple[int, int]]
        base_width = None  # type: Optional[int]
        count = 0
        hpitch = 0
        last_lower = 0
        for intv, range_set in intv_set.items():
            cur_start = range_set.get_start()  # type: int
            cur_end = range_set.get_end()  # type: int
            add = len(range_set) > 1
            if lower is not None and lower < cur_start:
                cur_start = lower
                add = True
            if upper is not None and upper > cur_end:
                cur_end = upper
                add = True

            cur_lower, cur_upper = intv
            if add:
                tr_id = grid.coord_to_track(layer_id, (cur_lower + cur_upper) // 2)
                layer_name = grid.get_layer_name(layer_id, tr_id)
                if is_horiz:
                    box = BBox(cur_start, cur_lower, cur_end, cur_upper)
                else:
                    box = BBox(cur_lower, cur_start, cur_upper, cur_end)
                self.add_rect(layer_name, box)

            if debug:
                print('wires intv: %s, range: (%d, %d)' % (intv, cur_start, cur_end))
            cur_width = cur_upper - cur_lower
            if count == 0:
                base_intv = intv
                base_start = cur_start
                base_end = cur_end
                base_width = cur_upper - cur_lower
                count = 1
                hpitch = 0
            else:
                assert base_intv is not None, "count == 0 should have set base_intv"
                assert base_width is not None, "count == 0 should have set base_width"
                assert base_start is not None, "count == 0 should have set base_start"
                assert base_end is not None, "count == 0 should have set base_end"
                if cur_start == base_start and cur_end == base_end and base_width == cur_width:
                    # length and width matches
                    cur_hpitch = (cur_lower - last_lower) // htr_pitch
                    if count == 1:
                        # second wire, set half pitch
                        hpitch = cur_hpitch
                        count += 1
                    elif hpitch == cur_hpitch:
                        # pitch matches
                        count += 1
                    else:
                        # pitch does not match, add current wires and start anew
                        tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv)
                        track_id = TrackID(layer_id, tr_idx, width=tr_width,
                                           num=count, pitch=hpitch / 2)
                        warr = WireArray(track_id, base_start, base_end)
                        new_warr_list.append(warr)
                        base_intv = intv
                        count = 1
                        hpitch = 0
                else:
                    # length/width does not match, add cumulated wires and start anew
                    tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv)
                    track_id = TrackID(layer_id, tr_idx, width=tr_width,
                                       num=count, pitch=hpitch / 2)
                    warr = WireArray(track_id, base_start, base_end)
                    new_warr_list.append(warr)
                    base_start = cur_start
                    base_end = cur_end
                    base_intv = intv
                    base_width = cur_width
                    count = 1
                    hpitch = 0

            # update last lower coordinate
            last_lower = cur_lower

        assert base_intv is not None, "count == 0 should have set base_intv"
        assert base_start is not None, "count == 0 should have set base_start"
        assert base_end is not None, "count == 0 should have set base_end"

        # add last wires
        tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv)
        track_id = TrackID(layer_id, tr_idx, tr_width, num=count, pitch=hpitch / 2)
        warr = WireArray(track_id, base_start, base_end)
        new_warr_list.append(warr)
        return new_warr_list

    def _draw_via_on_track(self, wlayer, box_arr, track_id, tl_unit=None,
                           tu_unit=None):
        # type: (str, BBoxArray, TrackID, Optional[int], Optional[int]) -> Tuple[int, int]
        """Helper method.  Draw vias on the intersection of the BBoxArray and TrackID."""
        grid = self.grid

        tr_layer_id = track_id.layer_id
        tr_width = track_id.width
        tr_dir = grid.get_direction(tr_layer_id)
        tr_pitch = grid.get_track_pitch(tr_layer_id)

        w_layer_id = grid.tech_info.get_layer_id(wlayer)
        w_dir = 'x' if tr_dir == 'y' else 'y'
        wbase = box_arr.base
        for sub_track_id in track_id.sub_tracks_iter(grid):
            base_idx = sub_track_id.base_index
            if w_layer_id > tr_layer_id:
                bot_layer = grid.get_layer_name(tr_layer_id, base_idx)
                top_layer = wlayer
                bot_dir = tr_dir
            else:
                bot_layer = wlayer
                top_layer = grid.get_layer_name(tr_layer_id, base_idx)
                bot_dir = w_dir
            # compute via bounding box
            tl, tu = grid.get_wire_bounds(tr_layer_id, base_idx, width=tr_width)
            if tr_dir == 'x':
                via_box = BBox(wbase.left_unit, tl, wbase.right_unit, tu)
                nx, ny = box_arr.nx, sub_track_id.num
                spx, spy = box_arr.spx_unit, sub_track_id.pitch * tr_pitch
                via = self.add_via(via_box, bot_layer, top_layer, bot_dir,
                                   nx=nx, ny=ny, spx=spx, spy=spy)
                vtbox = via.bottom_box if w_layer_id > tr_layer_id else via.top_box
                if tl_unit is None:
                    tl_unit = vtbox.left_unit
                else:
                    tl_unit = min(tl_unit, vtbox.left_unit)
                if tu_unit is None:
                    tu_unit = vtbox.right_unit + (nx - 1) * box_arr.spx_unit
                else:
                    tu_unit = max(tu_unit, vtbox.right_unit + (nx - 1) * box_arr.spx_unit)
            else:
                via_box = BBox(tl, wbase.bottom_unit, tu, wbase.top_unit)
                nx, ny = sub_track_id.num, box_arr.ny
                spx, spy = sub_track_id.pitch * tr_pitch, box_arr.spy_unit
                via = self.add_via(via_box, bot_layer, top_layer, bot_dir,
                                   nx=nx, ny=ny, spx=spx, spy=spy)
                vtbox = via.bottom_box if w_layer_id > tr_layer_id else via.top_box
                if tl_unit is None:
                    tl_unit = vtbox.bottom_unit
                else:
                    tl_unit = min(tl_unit, vtbox.bottom_unit)
                if tu_unit is None:
                    tu_unit = vtbox.top_unit + (ny - 1) * box_arr.spy_unit
                else:
                    tu_unit = max(tu_unit, vtbox.top_unit + (ny - 1) * box_arr.spy_unit)

        assert_msg = "for loop should have assigned tl_unit and tu_unit"
        assert tl_unit is not None and tu_unit is not None, assert_msg

        return tl_unit, tu_unit

    def connect_bbox_to_tracks(self,  # type: TemplateBase
                               layer_name,  # type: str
                               box_arr,  # type: Union[BBox, BBoxArray]
                               track_id,  # type: TrackID
                               track_lower=None,  # type: Optional[CoordType]
                               track_upper=None,  # type: Optional[CoordType]
                               unit_mode=True,  # type: bool
                               min_len_mode=None,  # type: Optional[int]
                               wire_lower=None,  # type: Optional[CoordType]
                               wire_upper=None,  # type: Optional[CoordType]
                               ):
        # type: (...) -> WireArray
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        layer_name : str
            the primitive wire layer name.
        box_arr : Union[BBox, BBoxArray]
            bounding box of the wire(s) to connect to tracks.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        unit_mode: bool
            deprecated parameter.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.
        wire_lower : Optional[CoordType]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[CoordType]
            if given, extend wire(s) to this upper coordinate.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        if isinstance(box_arr, BBox):
            box_arr = BBoxArray(box_arr)
        else:
            pass

        grid = self.grid

        # extend bounding boxes to tracks
        tl, tu = track_id.get_bounds(grid)
        if wire_lower is not None:
            tl = min(wire_lower, tl)
        if wire_upper is not None:
            tu = max(wire_upper, tu)

        tr_layer = track_id.layer_id
        tr_dir = grid.get_direction(tr_layer)
        base = box_arr.base
        if tr_dir == 'x':
            self.add_rect_arr(layer_name,
                              base.extend(y=tl).extend(y=tu),
                              nx=box_arr.nx, ny=box_arr.ny, spx=box_arr.spx_unit,
                              spy=box_arr.spy_unit)
        else:
            self.add_rect_arr(layer_name,
                              base.extend(x=tl).extend(x=tu),
                              nx=box_arr.nx, ny=box_arr.ny, spx=box_arr.spx_unit,
                              spy=box_arr.spy_unit)

        # draw vias
        tl_unit, tu_unit = self._draw_via_on_track(layer_name, box_arr, track_id,
                                                   tl_unit=track_lower, tu_unit=track_upper)

        # draw tracks
        if min_len_mode is not None:
            # extend track to meet minimum length
            min_len = grid.get_min_length(tr_layer, track_id.width)
            # make sure minimum length is even so that middle coordinate exists
            min_len = -(-min_len // 2) * 2
            tr_len = tu_unit - tl_unit
            if min_len > tr_len:
                ext = min_len - tr_len
                if min_len_mode < 0:
                    tl_unit -= ext
                elif min_len_mode > 0:
                    tu_unit += ext
                else:
                    tl_unit -= ext // 2
                    tu_unit = tl_unit + min_len
        result = WireArray(track_id, tl_unit, tu_unit)
        for layer_name, bbox_arr in result.wire_arr_iter(grid):
            self.add_rect_arr(layer_name, bbox_arr.base, nx=bbox_arr.nx, ny=bbox_arr.ny,
                              spx=bbox_arr.spx_unit, spy=bbox_arr.spy_unit)

        return result

    def connect_bbox_to_differential_tracks(self,  # type: TemplateBase
                                            layer_name,  # type: str
                                            pbox,  # type: Union[BBox, BBoxArray]
                                            nbox,  # type: Union[BBox, BBoxArray]
                                            tr_layer_id,  # type: int
                                            ptr_idx,  # type: TrackType
                                            ntr_idx,  # type: TrackType
                                            width=1,  # type: int
                                            track_lower=None,  # type: Optional[CoordType]
                                            track_upper=None,  # type: Optional[CoordType]
                                            unit_mode=True,  # type: bool
                                            ):
        # type: (...) -> Tuple[Optional[WireArray], Optional[WireArray]]
        """Connect the given differential primitive wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        layer_name : str
            the primitive wire layer name.
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
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        unit_mode: bool
            deprecated parameter.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_bbox_to_matching_tracks(layer_name, [pbox, nbox], tr_layer_id,
                                                          [ptr_idx, ntr_idx], width=width,
                                                          track_lower=track_lower,
                                                          track_upper=track_upper,
                                                          unit_mode=unit_mode)
        return track_list[0], track_list[1]

    def connect_bbox_to_matching_tracks(self,  # type: TemplateBase
                                        layer_name,  # type: str
                                        box_arr_list,  # type: List[Union[BBox, BBoxArray]]
                                        tr_layer_id,  # type: int
                                        tr_idx_list,  # type: List[TrackType]
                                        width=1,  # type: int
                                        track_lower=None,  # type: Optional[CoordType]
                                        track_upper=None,  # type: Optional[CoordType]
                                        unit_mode=True  # type: bool
                                        ):
        # type: (...) -> List[Optional[WireArray]]
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        layer_name : str
            the primitive wire layer name.
        box_arr_list : List[Union[BBox, BBoxArray]]
            bounding box of the wire(s) to connect to tracks.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        unit_mode: bool
            deprecated parameter.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')
        grid = self.grid

        num_tracks = len(tr_idx_list)
        if num_tracks != len(box_arr_list):
            raise ValueError('wire list length and track index list length mismatch.')
        if num_tracks == 0:
            raise ValueError('No tracks given')
        w_layer_id = grid.tech_info.get_layer_id(layer_name)
        if abs(w_layer_id - tr_layer_id) != 1:
            raise ValueError('Given primitive wires not adjacent to given track layer.')
        bot_layer_id = min(w_layer_id, tr_layer_id)

        # compute wire_lower/upper without via extension
        w_lower, w_upper = grid.get_wire_bounds(tr_layer_id, tr_idx_list[0], width=width)
        for tr_idx in islice(tr_idx_list, 1, None):
            cur_low, cur_up = grid.get_wire_bounds(tr_layer_id, tr_idx, width=width)
            w_lower = min(w_lower, cur_low)
            w_upper = max(w_upper, cur_up)

        # separate wire arrays into bottom/top tracks, compute wire/track lower/upper coordinates
        tr_width = grid.get_track_width(tr_layer_id, width)
        tr_dir = grid.get_direction(tr_layer_id)
        tr_horizontal = tr_dir == 'x'
        bbox_bounds = [None, None]  # type: List[int]
        for idx, box_arr in enumerate(box_arr_list):
            # convert to WireArray list
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)
            else:
                pass

            base = box_arr.base
            if w_layer_id < tr_layer_id:
                bot_dim = base.width_unit if tr_horizontal else base.height_unit
                top_dim = tr_width
                w_ext, tr_ext = grid.get_via_extensions_dim(bot_layer_id, bot_dim, top_dim)
            else:
                bot_dim = tr_width
                top_dim = base.width_unit if tr_horizontal else base.height_unit
                tr_ext, w_ext = grid.get_via_extensions_dim(bot_layer_id, bot_dim, top_dim)

            if bbox_bounds[0] is None:
                bbox_bounds = (w_lower - w_ext, w_upper + w_ext)
            else:
                bbox_bounds = (min(bbox_bounds[0], w_lower - w_ext),
                               max(bbox_bounds[1], w_upper + w_ext))

            # compute track lower/upper including via extension
            tr_bounds = box_arr.get_overall_bbox().get_interval(tr_dir)
            if track_lower is None:
                track_lower = tr_bounds[0] - tr_ext
            else:
                track_lower = min(track_lower, tr_bounds[0] - tr_ext)
            if track_upper is None:
                track_upper = tr_bounds[1] + tr_ext
            else:
                track_upper = max(track_upper, tr_bounds[1] + tr_ext)

        assert_msg = "track_lower/track_upper should be set above"
        assert track_lower is not None and track_upper is not None, assert_msg

        # draw tracks
        track_list = []  # type: List[Optional[WireArray]]
        for box_arr, tr_idx in zip(box_arr_list, tr_idx_list):
            track_list.append(self.add_wires(tr_layer_id, tr_idx, track_lower, track_upper,
                                             width=width))

            tr_id = TrackID(tr_layer_id, tr_idx, width=width)
            self.connect_bbox_to_tracks(layer_name, box_arr, tr_id, wire_lower=bbox_bounds[0],
                                        wire_upper=bbox_bounds[1])

        return track_list

    def connect_to_tracks(self,  # type: TemplateBase
                          wire_arr_list,  # type: Union[WireArray, List[WireArray]]
                          track_id,  # type: TrackID
                          wire_lower=None,  # type: Optional[CoordType]
                          wire_upper=None,  # type: Optional[CoordType]
                          track_lower=None,  # type: Optional[CoordType]
                          track_upper=None,  # type: Optional[CoordType]
                          unit_mode=True,  # type: bool
                          min_len_mode=None,  # type: Optional[int]
                          return_wires=False,  # type: bool
                          debug=False,  # type: bool
                          ):
        # type: (...) -> Union[Optional[WireArray], Tuple[Optional[WireArray], List[WireArray]]]
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
        unit_mode : bool
            deprecated parameter.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.
        return_wires : bool
            True to return the extended wires.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Union[Optional[WireArray], Tuple[Optional[WireArray], List[WireArray]]]
            WireArray representing the tracks/wires created.
            If return_wires is True, returns a Tuple[Optional[WireArray], List[WireArray]].
            If there was nothing to do, the first argument will be None.
            Otherwise, returns a WireArray.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        if isinstance(wire_arr_list, WireArray):
            # convert to list.
            wire_arr_list = [wire_arr_list]
        else:
            pass

        if not wire_arr_list:
            # do nothing
            if return_wires:
                return None, []
            return None

        grid = self.grid

        # find min/max track Y coordinates
        tr_layer_id = track_id.layer_id
        wl, wu = track_id.get_bounds(grid)
        if wire_lower is not None:
            wl = min(wire_lower, wl)

        if wire_upper is not None:
            wu = max(wire_upper, wu)

        # get top wire and bottom wire list
        top_list = []
        bot_list = []
        for wire_arr in wire_arr_list:
            cur_layer_id = wire_arr.layer_id
            if cur_layer_id == tr_layer_id + 1:
                top_list.append(wire_arr)
            elif cur_layer_id == tr_layer_id - 1:
                bot_list.append(wire_arr)
            else:
                raise ValueError(
                    'WireArray layer %d cannot connect to layer %d' % (cur_layer_id, tr_layer_id))

        # connect wires together
        top_wire_list = self.connect_wires(top_list, lower=wl, upper=wu, debug=debug)
        bot_wire_list = self.connect_wires(bot_list, lower=wl, upper=wu, debug=debug)

        # draw vias
        for w_layer_id, wire_list in ((tr_layer_id + 1, top_wire_list),
                                      (tr_layer_id - 1, bot_wire_list)):
            for wire_arr in wire_list:
                for wlayer, box_arr in wire_arr.wire_arr_iter(grid):
                    track_lower, track_upper = self._draw_via_on_track(wlayer, box_arr, track_id,
                                                                       tl_unit=track_lower,
                                                                       tu_unit=track_upper)
        assert_msg = "track_lower/track_upper should have been set just above"
        assert track_lower is not None and track_upper is not None, assert_msg

        if min_len_mode is not None:
            # extend track to meet minimum length
            min_len = grid.get_min_length(tr_layer_id, track_id.width)
            # make sure minimum length is even so that middle coordinate exists
            min_len = -(-min_len // 2) * 2
            tr_len = track_upper - track_lower
            if min_len > tr_len:
                ext = min_len - tr_len
                if min_len_mode < 0:
                    track_lower -= ext
                elif min_len_mode > 0:
                    track_upper += ext
                else:
                    track_lower -= ext // 2
                    track_upper = track_lower + min_len

        # draw tracks
        result = WireArray(track_id, track_lower, track_upper)
        for layer_name, bbox_arr in result.wire_arr_iter(grid):
            self.add_rect_arr(layer_name, bbox_arr.base, nx=bbox_arr.nx, ny=bbox_arr.ny,
                              spx=bbox_arr.spx_unit, spy=bbox_arr.spy_unit)

        if return_wires:
            top_wire_list.extend(bot_wire_list)
            return result, top_wire_list
        else:
            return result

    def connect_to_track_wires(self,  # type: TemplateBase
                               wire_arr_list,  # type: Union[WireArray, List[WireArray]]
                               track_wires,  # type: Union[WireArray, List[WireArray]]
                               min_len_mode=None,  # type: Optional[int]
                               debug=False,  # type: bool
                               ):
        # type: (...) -> Union[WireArray, List[WireArray]]
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
        wire_arr : Union[WireArray, List[WireArray]]
            WireArray representing the tracks created.  None if nothing to do.
        """
        ans = []
        if isinstance(track_wires, WireArray):
            ans_is_list = False
            track_wires = [track_wires]
        else:
            ans_is_list = True

        for warr in track_wires:
            tr = self.connect_to_tracks(wire_arr_list, warr.track_id,
                                        track_lower=warr.lower_unit, track_upper=warr.upper_unit,
                                        min_len_mode=min_len_mode, debug=debug)
            ans.append(tr)

        if not ans_is_list:
            return ans[0]
        return ans

    def connect_with_via_stack(self,  # type: TemplateBase
                               wire_array,  # type: Union[WireArray, List[WireArray]]
                               track_id,  # type: TrackID
                               tr_w_list=None,  # type: Optional[List[int]]
                               tr_mode_list=None,  # type: Optional[Union[int, List[int]]]
                               min_len_mode_list=None,  # type: Optional[Union[int, List[int]]]
                               debug=False,  # type: bool
                               ):
        # type: (...) -> List[WireArray]
        """Connect a single wire to the given track by using a via stack.

        This is a convenience function that draws via connections through several layers
        at once.  With optional parameters to control the track widths on each
        intermediate layers.

        Parameters
        ----------
        wire_array : Union[WireArray, List[WireArray]]
            the starting WireArray.
        track_id : TrackID
            the TrackID to connect to.
        tr_w_list : Optional[List[int]]
            the track widths to use on each layer.  If not specified, will compute automatically.
        tr_mode_list : Optional[Union[int, List[int]]]
            If tracks on intermediate layers do not line up nicely,
            the track mode flags determine whether to pick upper or lower tracks
        min_len_mode_list : Optional[Union[int, List[int]]]
            minimum length mode flags on each layer.
        debug : bool
            True to print debug messages.

        Returns
        -------
        warr_list : List[WireArray]
            List of created WireArrays.
        """
        if not isinstance(wire_array, WireArray):
            # error checking
            if len(wire_array) != 1:
                raise ValueError('connect_with_via_stack() only works on WireArray '
                                 'and TrackID with a single wire.')
            # convert to WireArray.
            wire_array = wire_array[0]

        # error checking
        warr_tid = wire_array.track_id
        warr_layer = warr_tid.layer_id
        tr_layer = track_id.layer_id
        tr_index = track_id.base_index
        if warr_tid.num != 1 or track_id.num != 1:
            raise ValueError('connect_with_via_stack() only works on WireArray '
                             'and TrackID with a single wire.')
        if tr_layer == warr_layer:
            raise ValueError('Cannot connect wire to track on the same layer.')

        num_connections = abs(tr_layer - warr_layer)

        # set default values
        if tr_w_list is None:
            tr_w_list = [-1] * num_connections
        elif len(tr_w_list) == num_connections - 1:
            # user might be inclined to not list the last track width, as it is included in
            # TrackID.  Allow for this exception
            tr_w_list = tr_w_list + [-1]
        elif len(tr_w_list) != num_connections:
            raise ValueError('tr_w_list must have exactly %d elements.' % num_connections)
        else:
            # create a copy of the given list, as this list may be modified later.
            tr_w_list = list(tr_w_list)

        if tr_mode_list is None:
            tr_mode_list = [0] * num_connections
        elif isinstance(tr_mode_list, int):
            tr_mode_list = [tr_mode_list] * num_connections
        elif len(tr_mode_list) != num_connections:
            raise ValueError('tr_mode_list must have exactly %d elements.' % num_connections)

        if min_len_mode_list is None:
            min_len_mode_list_resolved = [None] * num_connections  # type: List[Optional[int]]
        elif isinstance(min_len_mode_list, int):
            min_len_mode_list_resolved = [min_len_mode_list] * num_connections
        elif len(min_len_mode_list) != num_connections:
            raise ValueError('min_len_mode_list must have exactly %d elements.' % num_connections)
        else:
            min_len_mode_list_resolved = min_len_mode_list

        # determine via location
        grid = self.grid
        w_dir = grid.get_direction(warr_layer)
        t_dir = grid.get_direction(tr_layer)
        w_coord = grid.track_to_coord(warr_layer, warr_tid.base_index)
        t_coord = grid.track_to_coord(tr_layer, tr_index)
        if w_dir != t_dir:
            x0, y0 = (w_coord, t_coord) if w_dir == 'y' else (t_coord, w_coord)
        else:
            w_mid = wire_array.middle_unit
            x0, y0 = (w_coord, w_mid) if w_dir == 'y' else (w_mid, w_coord)

        # determine track width on each layer
        tr_w_list[num_connections - 1] = track_id.width
        if tr_layer > warr_layer:
            layer_dir = 1
            tr_w_prev = grid.get_track_width(tr_layer, tr_w_list[num_connections - 1])
            tr_w_idx_iter = range(num_connections - 2, -1, -1)
        else:
            layer_dir = -1
            tr_w_prev = grid.get_track_width(warr_layer, warr_tid.width)
            tr_w_idx_iter = range(0, num_connections - 1)
        for idx in tr_w_idx_iter:
            cur_layer = warr_layer + layer_dir * (idx + 1)
            if tr_w_list[idx] < 0:
                tr_w_list[idx] = max(1, grid.get_track_width_inverse(cur_layer, tr_w_prev))
            tr_w_prev = grid.get_track_width(cur_layer, tr_w_list[idx])

        # draw via stacks
        results = []  # type: List[WireArray]
        targ_layer = warr_layer
        for tr_w, tr_mode, min_len_mode in zip(tr_w_list, tr_mode_list, min_len_mode_list_resolved):
            targ_layer += layer_dir

            # determine track index to connect to
            if targ_layer == tr_layer:
                targ_index = tr_index
            else:
                targ_dir = grid.get_direction(targ_layer)
                coord = x0 if targ_dir == 'y' else y0
                targ_index = grid.coord_to_nearest_track(targ_layer, coord, half_track=True,
                                                         mode=tr_mode)

            targ_tid = TrackID(targ_layer, targ_index, width=tr_w)
            warr = self.connect_to_tracks(wire_array, targ_tid, min_len_mode=min_len_mode,
                                          debug=debug)
            results.append(warr)
            wire_array = warr

        return results

    def strap_wires(self,  # type: TemplateBase
                    warr,  # type: WireArray
                    targ_layer,  # type: int
                    tr_w_list=None,  # type: Optional[List[int]]
                    min_len_mode_list=None,  # type: Optional[List[int]]
                    ):
        # type: (...) -> WireArray
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

    def _strap_wires_helper(self,  # type: TemplateBase
                            warr,  # type: WireArray
                            targ_layer,  # type: int
                            tr_w,  # type: int
                            mlen_mode,  # type: Optional[int]
                            ):
        # type: (...) -> WireArray
        """Helper method for strap_wires().  Connect one layer at a time."""
        wire_tid = warr.track_id
        wire_layer = wire_tid.layer_id

        lower = warr.lower_unit
        upper = warr.upper_unit

        # error checking
        wdir = self.grid.get_direction(wire_layer)
        if wdir != self.grid.get_direction(targ_layer):
            raise ValueError('Cannot strap wires with different directions.')

        # convert base track index
        base_coord = self.grid.track_to_coord(wire_layer, wire_tid.base_index)
        base_tid = self.grid.coord_to_track(targ_layer, base_coord)
        # convert pitch
        wire_pitch = self.grid.get_track_pitch(wire_layer)
        targ_pitch = self.grid.get_track_pitch(targ_layer)
        pitch_unit = wire_pitch * wire_tid.pitch
        if pitch_unit % (targ_pitch // 2) != 0:
            raise ValueError('Cannot strap wires on layers with mismatched pitch ')
        num_pitch = pitch_unit // targ_pitch
        # convert width
        if tr_w < 0:
            width_unit = self.grid.get_track_width(wire_layer, wire_tid.width)
            tr_w = max(1, self.grid.get_track_width_inverse(targ_layer, width_unit, mode=-1))

        # draw vias.  Update WireArray lower/upper
        new_lower = lower  # type: int
        new_upper = upper  # type: int
        w_lower = lower  # type: int
        w_upper = upper  # type: int
        for tid in wire_tid:
            coord = self.grid.track_to_coord(wire_layer, tid)
            tid2 = self.grid.coord_to_track(targ_layer, coord)
            w_name = self.grid.get_layer_name(wire_layer, tid)
            t_name = self.grid.get_layer_name(targ_layer, tid2)

            w_yb, w_yt = self.grid.get_wire_bounds(wire_layer, tid, wire_tid.width)
            t_yb, t_yt = self.grid.get_wire_bounds(targ_layer, tid2, tr_w)
            vbox = BBox(lower, max(w_yb, t_yb), upper, min(w_yt, t_yt))
            if wdir == 'y':
                vbox = vbox.flip_xy()
            if wire_layer < targ_layer:
                via = self.add_via(vbox, w_name, t_name, wdir, extend=True, top_dir=wdir)
                tbox, wbox = via.top_box, via.bottom_box
            else:
                via = self.add_via(vbox, t_name, w_name, wdir, extend=True, top_dir=wdir)
                tbox, wbox = via.bottom_box, via.top_box

            if wdir == 'y':
                new_lower = min(new_lower, tbox.bottom_unit)
                new_upper = max(new_upper, tbox.top_unit)
                w_lower = min(w_lower, wbox.bottom_unit)
                w_upper = max(w_upper, wbox.top_unit)
            else:
                new_lower = min(new_lower, tbox.left_unit)
                new_upper = max(new_upper, tbox.right_unit)
                w_lower = min(w_lower, wbox.left_unit)
                w_upper = max(w_upper, wbox.top_unit)

        # handle minimum length DRC rule
        min_len = self.grid.get_min_length(targ_layer, tr_w)
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

    def connect_differential_tracks(self,  # type: TemplateBase
                                    pwarr_list,  # type: Union[WireArray, List[WireArray]]
                                    nwarr_list,  # type: Union[WireArray, List[WireArray]]
                                    tr_layer_id,  # type: int
                                    ptr_idx,  # type: TrackType
                                    ntr_idx,  # type: TrackType
                                    width=1,  # type: int
                                    track_lower=None,  # type: Optional[CoordType]
                                    track_upper=None,  # type: Optional[CoordType]
                                    unit_mode=True,  # type: bool
                                    debug=False  # type: bool
                                    ):
        # type: (...) -> Tuple[Optional[WireArray], Optional[WireArray]]
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
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        unit_mode: bool
            deprecated parameter.
        debug : bool
            True to print debug messages.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_matching_tracks([pwarr_list, nwarr_list], tr_layer_id,
                                                  [ptr_idx, ntr_idx], width=width,
                                                  track_lower=track_lower,
                                                  track_upper=track_upper,
                                                  unit_mode=unit_mode,
                                                  debug=debug)
        return track_list[0], track_list[1]

    def connect_differential_wires(self,  # type: TemplateBase
                                   pin_warrs,  # type: Union[WireArray, List[WireArray]]
                                   nin_warrs,  # type: Union[WireArray, List[WireArray]]
                                   pout_warr,  # type: WireArray
                                   nout_warr,  # type: WireArray
                                   track_lower=None,  # type: Optional[CoordType]
                                   track_upper=None,  # type: Optional[CoordType]
                                   unit_mode=True,  # type: bool
                                   debug=False  # type: bool
                                   ):
        # type: (...) -> Tuple[Optional[WireArray], Optional[WireArray]]
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        p_tid = pout_warr.track_id
        lay_id = p_tid.layer_id
        pidx = p_tid.base_index
        nidx = nout_warr.track_id.base_index
        width = p_tid.width

        if track_lower is None:
            tr_lower = pout_warr.lower_unit
        else:
            tr_lower = min(track_lower, pout_warr.lower_unit)
        if track_upper is None:
            tr_upper = pout_warr.upper_unit
        else:
            tr_upper = max(track_upper, pout_warr.upper_unit)

        return self.connect_differential_tracks(pin_warrs, nin_warrs, lay_id, pidx, nidx,
                                                width=width, track_lower=tr_lower,
                                                track_upper=tr_upper, debug=debug)

    def connect_matching_tracks(self,  # type: TemplateBase
                                warr_list_list,  # type: List[Union[WireArray, List[WireArray]]]
                                tr_layer_id,  # type: int
                                tr_idx_list,  # type: List[TrackType]
                                width=1,  # type: int
                                track_lower=None,  # type: Optional[CoordType]
                                track_upper=None,  # type: Optional[CoordType]
                                unit_mode=True,  # type: bool
                                debug=False  # type: bool
                                ):
        # type: (...) -> List[Optional[WireArray]]
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
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        unit_mode: bool
            deprecated parameter.
        debug : bool
            True to print debug messages.

        Returns
        -------
        track_list : List[WireArray]
            list of created tracks.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        grid = self.grid

        # simple error checking
        num_tracks = len(tr_idx_list)  # type: int
        if num_tracks != len(warr_list_list):
            raise ValueError('wire list length and track index list length mismatch.')
        if num_tracks == 0:
            raise ValueError('No tracks given')

        # compute wire_lower/upper without via extension
        w_lower, w_upper = grid.get_wire_bounds(tr_layer_id, tr_idx_list[0], width=width)
        for tr_idx in islice(tr_idx_list, 1, None):
            cur_low, cur_up = grid.get_wire_bounds(tr_layer_id, tr_idx, width=width)
            w_lower = min(w_lower, cur_low)
            w_upper = max(w_upper, cur_up)

        # separate wire arrays into bottom/top tracks, compute wire/track lower/upper coordinates
        bot_warrs = [[] for _ in range(num_tracks)]
        top_warrs = [[] for _ in range(num_tracks)]
        bot_bounds = [None, None]  # type: List[CoordType]
        top_bounds = [None, None]  # type: List[CoordType]
        for idx, warr_list in enumerate(warr_list_list):
            # convert to WireArray list
            if isinstance(warr_list, WireArray):
                warr_list = [warr_list]
            else:
                pass

            if not warr_list:
                raise ValueError('No wires found for track index %d' % idx)

            for warr in warr_list:
                warr_tid = warr.track_id
                cur_layer_id = warr_tid.layer_id
                cur_width = warr_tid.width
                if cur_layer_id == tr_layer_id + 1:
                    tr_ext, w_ext = grid.get_via_extensions(tr_layer_id, width, cur_width)
                    top_warrs[idx].append(warr)
                    cur_bounds = top_bounds
                elif cur_layer_id == tr_layer_id - 1:
                    w_ext, tr_ext = grid.get_via_extensions(cur_layer_id, cur_width, width)
                    bot_warrs[idx].append(warr)
                    cur_bounds = bot_bounds
                else:
                    raise ValueError('Cannot connect wire on layer %d '
                                     'to track on layer %d' % (cur_layer_id, tr_layer_id))

                # compute wire lower/upper including via extension
                if cur_bounds[0] is None:
                    cur_bounds[0] = w_lower - w_ext
                    cur_bounds[1] = w_upper + w_ext
                else:
                    cur_bounds[0] = min(cur_bounds[0], w_lower - w_ext)
                    cur_bounds[1] = max(cur_bounds[1], w_upper + w_ext)

                # compute track lower/upper including via extension
                warr_bounds = warr_tid.get_bounds(grid)
                if track_lower is None:
                    track_lower = warr_bounds[0] - tr_ext
                else:
                    track_lower = min(track_lower, warr_bounds[0] - tr_ext)
                if track_upper is None:
                    track_upper = warr_bounds[1] + tr_ext
                else:
                    track_upper = max(track_upper, warr_bounds[1] + tr_ext)

        assert_msg = "track_lower/track_upper should have been set above"
        assert track_lower is not None and track_upper is not None, assert_msg

        # draw tracks
        track_list = []  # type: List[Optional[WireArray]]
        for bwarr_list, twarr_list, tr_idx in zip(bot_warrs, top_warrs, tr_idx_list):
            track_list.append(self.add_wires(tr_layer_id, tr_idx, track_lower, track_upper,
                                             width=width))

            tr_id = TrackID(tr_layer_id, tr_idx, width=width)
            self.connect_to_tracks(bwarr_list, tr_id, wire_lower=bot_bounds[0],
                                   wire_upper=bot_bounds[1], min_len_mode=None, debug=debug)
            self.connect_to_tracks(twarr_list, tr_id, wire_lower=top_bounds[0],
                                   wire_upper=top_bounds[1], min_len_mode=None, debug=debug)

        return track_list

    def draw_vias_on_intersections(self, bot_warr_list, top_warr_list):
        # type: (Union[WireArray, List[WireArray]], Union[WireArray, List[WireArray]]) -> None
        """Draw vias on all intersections of the two given wire groups.

        Parameters
        ----------
        bot_warr_list : Union[WireArray, List[WireArray]]
            the bottom wires.
        top_warr_list : Union[WireArray, List[WireArray]]
            the top wires.
        """
        if isinstance(bot_warr_list, WireArray):
            bot_warr_list = [bot_warr_list]
        else:
            pass
        if isinstance(top_warr_list, WireArray):
            top_warr_list = [top_warr_list]
        else:
            pass

        grid = self.grid

        for bwarr in bot_warr_list:
            bot_tl = bwarr.lower_unit
            bot_tu = bwarr.upper_unit
            bot_track_idx = bwarr.track_id
            bot_layer_id = bot_track_idx.layer_id
            top_layer_id = bot_layer_id + 1
            bot_width = bot_track_idx.width
            bot_dir = self.grid.get_direction(bot_layer_id)
            bot_horizontal = (bot_dir == 'x')
            for bot_index in bot_track_idx:
                bot_lay_name = self.grid.get_layer_name(bot_layer_id, bot_index)
                btl, btu = grid.get_wire_bounds(bot_layer_id, bot_index, width=bot_width)
                for twarr in top_warr_list:
                    top_tl = twarr.lower_unit
                    top_tu = twarr.upper_unit
                    top_track_idx = twarr.track_id
                    top_width = top_track_idx.width
                    if top_tu >= btu and top_tl <= btl:
                        # top wire cuts bottom wire, possible intersection
                        for top_index in top_track_idx:
                            ttl, ttu = grid.get_wire_bounds(top_layer_id, top_index,
                                                            width=top_width)
                            if bot_tu >= ttu and bot_tl <= ttl:
                                # bottom wire cuts top wire, we have intersection.  Make bbox
                                if bot_horizontal:
                                    box = BBox(ttl, btl, ttu, btu)
                                else:
                                    box = BBox(btl, ttl, btu, ttu)
                                top_lay_name = self.grid.get_layer_name(top_layer_id, top_index)
                                self.add_via(box, bot_lay_name, top_lay_name, bot_dir)

    def has_blockage(self, layer_id, test_box, spx=0, spy=0):
        # type: (int, BBox, int, int) -> bool
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
        layer_name = self._grid.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, str):
            return self._layout.has_blockage(layer_name, test_box, spx=spx, spy=spy)
        else:
            for lay_name in layer_name:
                if self._layout.has_blockage(lay_name, test_box, spx=spx, spy=spy):
                    return True
            return False

    def blockage_iter(self, layer_id, test_box, spx=0, spy=0):
        # type: (int, BBox, int, int) -> Generator[GeoType, None, None]
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
        layer_name = self._grid.tech_info.get_layer_name(layer_id)
        if isinstance(layer_name, str):
            return self._layout.blockage_iter(layer_name, test_box, spx=spx, spy=spy)
        else:
            for lay_name in layer_name:
                yield from self._layout.blockage_iter(lay_name, test_box, spx=spx, spy=spy)

    def is_track_available(self,  # type: TemplateBase
                           layer_id,  # type: int
                           tr_idx,  # type: TrackType
                           lower,  # type: int
                           upper,  # type: int
                           width=1,  # type: int
                           sp=0,  # type: int
                           sp_le=0,  # type: int
                           unit_mode=True,  # type: bool
                           ):
        # type: (...) -> bool
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
        unit_mode : bool
            deprecated parameter.

        Returns
        -------
        available : bool
            True if the track is available.
        """
        if not unit_mode:
            raise ValueError('unit_mode = False not supported.')

        grid = self._grid
        is_x = grid.is_horizontal(layer_id)
        track_id = TrackID(layer_id, tr_idx, width=width)
        warr = WireArray(track_id, lower, upper)
        sp = max(sp, grid.get_space(layer_id, width))
        sp_le = max(sp_le, grid.get_line_end_space(layer_id, width))
        test_box = warr.get_bbox_array(grid).base
        if is_x:
            return not self.has_blockage(layer_id, test_box, spx=sp_le, spy=sp)
        else:
            return not self.has_blockage(layer_id, test_box, spx=sp, spy=sp_le)

    def mark_bbox_used(self, layer_id, bbox):
        # type: (int, BBox) -> None
        """Marks the given bounding-box region as used in this Template."""
        # TODO: fix this method
        layer_name = self.grid.get_layer_name(layer_id, 0)
        # self._used_tracks.record_rect(self.grid, layer_name, BBoxArray(bbox), dx=0, dy=0)
        raise ValueError('Not implemented yet')

    def get_available_tracks(self,  # type: TemplateBase
                             layer_id,  # type: int
                             tr_idx_list,  # type: List[int]
                             lower,  # type: CoordType
                             upper,  # type: CoordType
                             width=1,  # type: int
                             margin=0,  # type: CoordType
                             unit_mode=True,  # type: bool
                             ):
        # type: (...) -> List[int]
        """Returns empty tracks"""
        # TODO: fix this method
        raise ValueError('Not implemented yet')

    def do_power_fill(self,  # type: TemplateBase
                      layer_id,  # type: int
                      space,  # type: CoordType
                      space_le,  # type: CoordType
                      vdd_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                      vss_warrs=None,  # type: Optional[Union[WireArray, List[WireArray]]]
                      bound_box=None,  # type: Optional[BBox]
                      fill_width=1,  # type: int
                      fill_space=0,  # type: int
                      x_margin=0,  # type: CoordType
                      y_margin=0,  # type: CoordType
                      tr_offset=0,  # type: CoordType
                      min_len=0,  # type: CoordType
                      flip=False,  # type: bool
                      unit_mode=True,  # type: bool
                      ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw power fill on the given layer."""
        # TODO: fix this method
        raise ValueError('Not implemented yet')

    def do_max_space_fill(self,  # type: TemplateBase
                          layer_id,  # type: int
                          bound_box=None,  # type: Optional[BBox]
                          fill_pitch=1,  # type: TrackType
                          ):
        # type: (...) -> None
        """Draw density fill on the given layer."""
        # TODO: fix this method
        raise ValueError('Not implemented yet')


class BlackBoxTemplate(TemplateBase):
    """A black box template."""

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], Any) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = {}  # type: Dict[str, Any]

    @property
    def sch_params(self):
        # type: () -> Dict[str, Any]
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            lib_name='The library name.',
            cell_name='The layout cell name.',
            top_layer='The top level layer.',
            size='The width/height of the cell, in resolution units.',
            ports='The port information dictionary.',
            show_pins='True to show pins.',
        )

    def get_layout_basename(self):
        return self.params['cell_name']

    def draw_layout(self):
        # type: () -> None
        lib_name = self.params['lib_name']
        cell_name = self.params['cell_name']
        top_layer = self.params['top_layer']
        size = self.params['size']
        ports = self.params['ports']
        show_pins = self.params['show_pins']

        tech_info = self.grid.tech_info
        for term_name, pin_dict in ports.items():
            for lay_name, bbox_list in pin_dict.items():
                lay_id = tech_info.get_layer_id(lay_name)
                for xl, yb, xr, yt in bbox_list:
                    box = BBox(xl, yb, xr, yt)
                    self._register_pin(lay_id, lay_name, term_name, box, show_pins)

        self.add_instance_primitive(lib_name, cell_name, (0, 0))

        self.prim_top_layer = top_layer
        self.prim_bound_box = BBox(0, 0, size[0], size[1])

        for layer in range(1, top_layer + 1):
            self.mark_bbox_used(layer, self.prim_bound_box)

        self._sch_params = dict(
            lib_name=lib_name,
            cell_name=cell_name,
        )

    def _register_pin(self, lay_id, lay_name, term_name, box, show_pins):
        if lay_id is None:
            self.add_pin_primitive(term_name, lay_name, box, show=show_pins)
        else:
            if self.grid.get_direction(lay_id) == 'x':
                dim = box.height_unit
                coord = box.yc_unit
                lower = box.left_unit
                upper = box.right_unit
            else:
                dim = box.width_unit
                coord = box.xc_unit
                lower = box.bottom_unit
                upper = box.top_unit
            try:
                tr_idx = self.grid.coord_to_track(lay_id, coord)
            except ValueError:
                self.add_pin_primitive(term_name, lay_name, box, show=show_pins)
                return

            width_ntr = self.grid.get_track_width_inverse(lay_id, dim)
            if self.grid.get_track_width(lay_id, width_ntr) == dim:
                track_id = TrackID(lay_id, tr_idx, width=width_ntr)
                warr = WireArray(track_id, lower, upper)
                self.add_pin(term_name, warr, show=show_pins)
            else:
                self.add_pin_primitive(term_name, lay_name, box, show=show_pins)
