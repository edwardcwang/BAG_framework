# -*- coding: utf-8 -*-

"""This module defines layout template classes.
"""

from typing import TYPE_CHECKING, Union, Dict, Any, List, Set, TypeVar, Type, \
    Optional, Tuple, Iterable, Sequence, Callable

import os
import time
import abc
import copy
from itertools import chain, islice

import yaml

from bag.util.cache import DesignMaster, MasterDB
from bag.util.interval import IntervalSet
from .core import BagLayout
from .util import BBox, BBoxArray
from ..io import get_encoding, open_file
from .routing import Port, TrackID, WireArray
from .routing.fill import UsedTracks, get_power_fill_tracks, get_available_tracks
from .objects import Instance, Rect, Via, Path

if TYPE_CHECKING:
    from bag.core import BagProject
    from .objects import Polygon, Blockage, Boundary
    from .routing import RoutingGrid

# try to import cybagoa module
try:
    import cybagoa
except ImportError:
    cybagoa = None

TemplateType = TypeVar('TemplateType', bound='TemplateBase')


class TemplateDB(MasterDB):
    """A database of all templates.

    This class is responsible for keeping track of template libraries and
    creating new templates.

    Parameters
    ----------
    lib_defs : str
        path to the template library definition file.
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
    use_cybagoa : bool
        True to use cybagoa module to accelerate layout.
    flatten : bool
        True to compute flattened layout.
    **kwargs :
        additional arguments.
    """

    def __init__(self, lib_defs, routing_grid, lib_name, prj=None, name_prefix='', name_suffix='',
                 use_cybagoa=False, flatten=False, **kwargs):
        # type: (str, RoutingGrid, str, Optional[BagProject], str, str, bool, bool, **kwargs) -> None
        super(TemplateDB, self).__init__(lib_name, lib_defs=lib_defs,
                                         name_prefix=name_prefix, name_suffix=name_suffix)

        pure_oa = kwargs.get('pure_oa', False)

        if pure_oa:
            if cybagoa is None:
                raise ValueError('Cannot use pure OA mode when cybagoa is not found.')
            use_cybagoa = True

        self._prj = prj
        self._grid = routing_grid
        self._use_cybagoa = use_cybagoa and cybagoa is not None
        self._flatten = flatten
        self._pure_oa = pure_oa

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[TemplateType], str, Dict[str, Any], Set[str], **kwargs) -> TemplateType
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
        **kwargs
            optional arguments for the generator.

        Returns
        -------
        master : TemplateType
            the non-finalized generated instance.
        """
        # noinspection PyCallingNonCallable
        return gen_cls(self, lib_name, params, used_cell_names, **kwargs)

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
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        if self._use_cybagoa:
            # remove write locks from old layouts
            cell_view_list = [(item[0], 'layout') for item in content_list]
            if self._pure_oa:
                pass
            else:
                # create library if it does not exist
                self._prj.create_library(self._lib_name)
                self._prj.release_write_locks(self._lib_name, cell_view_list)

            if debug:
                print('Instantiating layout')
            # create OALayouts
            start = time.time()
            if 'CDSLIBPATH' in os.environ:
                cds_lib_path = os.path.abspath(os.path.join(os.environ['CDSLIBPATH'], 'cds.lib'))
            else:
                cds_lib_path = os.path.abspath('./cds.lib')
            with cybagoa.PyOALayoutLibrary(cds_lib_path, self._lib_name, self._prj.default_lib_path,
                                           self._prj.tech_info.via_tech_name, get_encoding()) as lib:
                lib.add_layer('prBoundary', 235)
                lib.add_purpose('drawing1', 241)
                lib.add_purpose('drawing2', 242)
                lib.add_purpose('drawing3', 243)
                lib.add_purpose('drawing4', 244)
                lib.add_purpose('drawing5', 245)
                lib.add_purpose('drawing6', 246)
                lib.add_purpose('drawing7', 247)
                lib.add_purpose('drawing8', 248)
                lib.add_purpose('drawing9', 249)
                lib.add_purpose('boundary', 250)
                lib.add_purpose('pin', 251)

                for cell_name, oa_layout in content_list:
                    lib.create_layout(cell_name, 'layout', oa_layout)
            end = time.time()
            if debug:
                print('layout instantiation took %.4g seconds' % (end - start))
        else:
            # create library if it does not exist
            self._prj.create_library(self._lib_name)

            if debug:
                print('Instantiating layout')
            via_tech_name = self._grid.tech_info.via_tech_name
            start = time.time()
            self._prj.instantiate_layout(self._lib_name, 'layout', via_tech_name, content_list)
            end = time.time()
            if debug:
                print('layout instantiation took %.4g seconds' % (end - start))

    @property
    def grid(self):
        # type: () -> RoutingGrid
        """Returns the default routing grid instance."""
        return self._grid

    def new_template(self, lib_name='', temp_name='', params=None, temp_cls=None, debug=False, **kwargs):
        # type: (str, str, Dict[str, Any], Type[TemplateType], bool, **kwargs) -> TemplateType
        """Create a new template.

        Parameters
        ----------
        lib_name : str
            template library name.
        temp_name : str
            template name
        params : Dict[str, Any]
            the parameter dictionary.
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        debug : bool
            True to print debug messages.
        **kwargs
            optional template parameters.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        kwargs['use_cybagoa'] = self._use_cybagoa
        master = self.new_master(lib_name=lib_name, cell_name=temp_name, params=params,
                                 gen_cls=temp_cls, debug=debug, **kwargs)

        return master

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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None

        # initialize template attributes
        self._grid = kwargs.get('grid', temp_db.grid).copy()
        self._layout = BagLayout(self._grid, use_cybagoa=kwargs.get('use_cybagoa', False),)
        self._size = None  # type: Tuple[int, int, int]
        self.pins = {}
        self._ports = {}
        self._port_params = {}
        self._array_box = None  # type: BBox
        self.prim_top_layer = None
        self.prim_bound_box = None
        self._used_tracks = UsedTracks(self._grid.resolution)
        self._added_inst_tracks = False

        # add hidden parameters
        if 'hidden_params' in kwargs:
            hidden_params = kwargs['hidden_params'].copy()
        else:
            hidden_params = {}
        hidden_params['flip_parity'] = None

        super(TemplateBase, self).__init__(temp_db, lib_name, params, used_names, hidden_params=hidden_params)
        # update RoutingGrid
        fp_dict = self.params['flip_parity']
        if fp_dict is not None:
            self._grid.set_flip_parity(fp_dict)

    @abc.abstractmethod
    def draw_layout(self):
        # type: () -> None
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        pass

    def populate_params(self, table, params_info, default_params, **kwargs):
        # type: (Dict[str, Any], Dict[str, str], Dict[str, Any], **kwargs) -> None
        """Fill params dictionary with values from table and default_params"""
        super(TemplateBase, self).populate_params(table, params_info, default_params, **kwargs)

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

    def get_content(self, lib_name, rename_fun):
        # type: (str, Callable[[str], str]) -> Union[List[Any], Tuple[str, 'cybagoa.PyOALayout']]
        """Returns the content of this master instance.

        Parameters
        ----------
        lib_name : str
            the library to create the design masters in.
        rename_fun : Callable[[str], str]
            a function that renames design masters.

        Returns
        -------
        content : Union[List[Any], Tuple[str, 'cybagoa.PyOALayout']]
            a list describing this layout, or PyOALayout if cybagoa is enabled.
        """
        if not self.finalized:
            raise ValueError('This template is not finalized yet')
        return self._layout.get_content(lib_name, self.cell_name, rename_fun)

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # create layout
        self.draw_layout()

        # finalize this template
        self.grid.tech_info.finalize_template(self)

        # update track parities of all instances
        if self.grid.tech_info.use_flip_parity():
            self._update_flip_parity()

        # update used tracks
        self._merge_inst_used_tracks()

        # construct port objects
        for net_name, port_params in self._port_params.items():
            pin_dict = port_params['pins']
            if port_params['show']:
                label = port_params['label']
                for wire_arr_list in pin_dict.values():
                    for wire_arr in wire_arr_list:  # type: WireArray
                        for layer_name, bbox in wire_arr.wire_iter(self.grid):
                            self._layout.add_pin(net_name, layer_name, bbox, label=label)
            self._ports[net_name] = Port(net_name, pin_dict)

        # finalize layout
        self._layout.finalize()
        # get set of children keys
        self.children = self._layout.get_masters_set()

        # call super finalize routine
        super(TemplateBase, self).finalize()

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
        # type: () -> BBox
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
        # type: () -> Optional[Tuple[int, int, int]]
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

        wblk, hblk = self.grid.get_size_dimension(mysize, unit_mode=True)
        return BBox(0, 0, wblk, hblk, self.grid.resolution, unit_mode=True)

    @size.setter
    def size(self, new_size):
        # type: (Tuple[int, int, int]) -> None
        """Sets the size of this template."""
        if not self._finalized:
            self._size = new_size
        else:
            raise RuntimeError('Template already finalized.')

    def _update_flip_parity(self):
        # type: () -> None
        """Update all instances in this template to have the correct track parity.
        """
        for inst in self._layout.inst_iter():
            top_layer = inst.master.top_layer
            bot_layer = self.grid.get_bot_common_layer(inst.master.grid, top_layer)
            loc = inst.location_unit
            fp_dict = self.grid.get_flip_parity_at(bot_layer, top_layer, loc,
                                                   inst.orientation, unit_mode=True)
            inst.new_master_with(flip_parity=fp_dict)

    def get_used_tracks(self):
        # type: () -> UsedTracks
        """Returns data structure of used tracks on the given layers."""
        return self._used_tracks

    def get_rect_bbox(self, layer):
        # type: (Union[str, Tuple[str, str]]) -> BBox
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.

        Parameters
        ----------
        layer : Union[str, Tuple[str, str]]
            the layer name.

        Returns
        -------
        box : BBox
            the overall bounding box of the given layer.
        """
        return self._layout.get_rect_bbox(layer)

    def new_template_with(self, **kwargs):
        # type: (**kwargs) -> TemplateBase
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

        return self.template_db.new_template(params=new_params, temp_cls=self.__class__, grid=self.grid)

    def set_size_from_bound_box(self, top_layer_id, bbox, round_up=False):
        # type: (int, BBox, bool) -> None
        """Compute the size from overall bounding box.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        bbox : BBox
            the overall bounding box
        round_up: bool
            True to round up bounding box if not quantized properly
        """
        grid = self.grid

        if bbox.left_unit != 0 or bbox.bottom_unit != 0:
            raise ValueError('lower-left corner of overall bounding box must be (0, 0).')

        self.size = grid.get_size_tuple(top_layer_id, bbox.width_unit, bbox.height_unit,
                                        round_up=round_up, unit_mode=True)

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

        dx = self.array_box.left_unit
        dy = self.array_box.bottom_unit
        if dx < 0 or dy < 0:
            raise ValueError('lower-left corner of array box must be in first quadrant.')

        self.size = grid.get_size_tuple(top_layer_id, 2 * dx + self.array_box.width_unit,
                                        2 * dy + self.array_box.height_unit, unit_mode=True)

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
                        layer=[layer_name, 'pin'],
                        netname=port_name,
                        xy0=[bbox.left, bbox.bottom],
                        xy1=[bbox.right, bbox.top],
                    )

        # get size information
        bnd_box = self.bound_box
        info = {
            lib_name: {
                cell_name: dict(
                    pins=pin_dict,
                    xy0=[0.0, 0.0],
                    xy1=[bnd_box.width, bnd_box.height],
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
            the port terminal name.  If None or empty, check if this template has only one port, then return it.

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

    def new_template(self, params=None, temp_cls=None, debug=False, **kwargs):
        # type: (Dict[str, Any], Type[TemplateType], bool, **kwargs) -> TemplateType
        """Create a new template.

        Parameters
        ----------
        params : Dict[str, Any]
            the parameter dictionary.
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        debug : bool
            True to print debug messages.
        **kwargs
            optional template parameters.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        kwargs['grid'] = self.grid
        return self.template_db.new_template(params=params, temp_cls=temp_cls, debug=debug, **kwargs)

    def move_all_by(self, dx=0.0, dy=0.0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> None
        """Move all layout objects Except pins in this layout by the given amount.

        primitive pins will be moved, but pins on routing grid will not.

        Parameters
        ----------
        dx : Union[float, int]
            the X shift.
        dy : Union[float, int]
            the Y shift.
        unit_mode : bool
            true if given shift values are in resolution units.
        """
        self._layout.move_all_by(dx=dx, dy=dy, unit_mode=unit_mode)

    def add_instance(self, master, inst_name=None, loc=(0, 0),
                     orient="R0", nx=1, ny=1, spx=0, spy=0, unit_mode=False):
        # type: (TemplateBase, Optional[str], Tuple[float, float], str, int, int, float, float) -> Instance
        """Adds a new (arrayed) instance to layout.

        Parameters
        ----------
        master : TemplateBase
            the master template object.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        loc : Tuple[float, float]
            instance location.
        orient : str
            instance orientation.  Defaults to "R0"
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : float
            column pitch.  Used for arraying given instance.
        spy : float
            row pitch.  Used for arraying given instance.
        unit_mode : bool
            True if dimensions are given in resolution units.

        Returns
        -------
        inst : Instance
            the added instance.
        """
        res = self.grid.resolution
        if not unit_mode:
            loc = int(round(loc[0] / res)), int(round(loc[1] / res))
            spx = int(round(spx / res))
            spy = int(round(spy / res))

        inst = Instance(self.grid, self._lib_name, master, loc=loc, orient=orient,
                        name=inst_name, nx=nx, ny=ny, spx=spx, spy=spy, unit_mode=True)

        self._layout.add_instance(inst)
        return inst

    def add_instance_primitive(self, lib_name,  # type: str
                               cell_name,  # type: str
                               loc,  # type: Tuple[float, float]
                               view_name='layout',  # type: str
                               inst_name=None,  # type: Optional[str]
                               orient="R0",  # type: str
                               nx=1,  # type: int
                               ny=1,  # type: int
                               spx=0.0,  # type: float
                               spy=0.0,  # type: float
                               params=None,  # type: Optional[Dict[str, Any]]
                               **kwargs  # type: **kwargs
                               ):
        # type: (...) -> None
        """Adds a new (arrayed) primitive instance to layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        loc : Tuple[float, float]
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
        spx : float
            column pitch.  Used for arraying given instance.
        spy : float
            row pitch.  Used for arraying given instance.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.  Used for adding pcell instance.
        **kwargs
            additional arguments.  Usually implementation specific.
        """
        self._layout.add_instance_primitive(lib_name, cell_name, loc,
                                            view_name=view_name, inst_name=inst_name,
                                            orient=orient, num_rows=ny, num_cols=nx,
                                            sp_rows=spy, sp_cols=spx,
                                            params=params, **kwargs)

    def add_rect(self, layer, bbox, nx=1, ny=1, spx=0.0, spy=0.0):
        # type: (Union[str, Tuple[str, str]], Union[BBox, BBoxArray], int, int, float, float) -> Rect
        """Add a new (arrayed) rectangle.

        Parameters
        ----------
        layer: Union[str, Tuple[str, str]]
            the layer name, or the (layer, purpose) pair.
        bbox : Union[BBox, BBoxArray]
            the rectangle bounding box.  If BBoxArray is given, its arraying parameters will be used instead.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : float
            column pitch.
        spy : float
            row pitch.

        Returns
        -------
        rect : Rect
            the added rectangle.
        """
        if isinstance(bbox, BBoxArray):
            nx, ny, spx, spy = bbox.nx, bbox.ny, bbox.spx, bbox.spy
            bbox = bbox.base
        else:
            pass

        rect = Rect(layer, bbox, nx=nx, ny=ny, spx=spx, spy=spy)
        self._layout.add_rect(rect)
        return rect

    def add_res_metal(self, layer_id, bbox, **kwargs):
        # type: (int, Union[BBox, BBoxArray], **kwargs) -> List[Rect]
        """Add a new metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.
        bbox : Union[BBox, BBoxArray]
            the resistor bounding box.  If BBoxArray is given, its arraying parameters will be used instead.
        **kwargs :
            optional arguments to add_rect()

        Returns
        -------
        rect_list : List[Rect]
            list of rectangles defining the metal resistor.
        """
        rect_list = []
        rect_layers = self.grid.tech_info.get_res_metal_layers(layer_id)
        for lay in rect_layers:
            rect_list.append(self.add_rect(lay, bbox, **kwargs))
        return rect_list

    def add_path(self, path):
        # type: (Path) -> Path
        """Add a new path.

        Parameters
        ----------
        path : Path
            the path to add.

        Returns
        -------
        path : Path
            the added path object.
        """
        self._layout.add_path(path)
        return path

    def add_polygon(self, polygon):
        # type: (Polygon) -> Polygon
        """Add a new polygon.

        Parameters
        ----------
        polygon : Polygon
            the blockage to add.

        Returns
        -------
        polygon : Polygon
            the added blockage object.
        """
        self._layout.add_polygon(polygon)
        return polygon

    def add_blockage(self, blockage):
        # type: (Blockage) -> Blockage
        """Add a new blockage.

        Parameters
        ----------
        blockage : Blockage
            the blockage to add.

        Returns
        -------
        blockage : Blockage
            the added blockage object.
        """
        self._layout.add_blockage(blockage)
        return blockage

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

    def add_boundary(self, boundary):
        # type: (Boundary) -> Boundary
        """Add a new boundary.

        Parameters
        ----------
        boundary : Boundary
            the boundary to add.

        Returns
        -------
        boundary : Boundary
            the added boundary object.
        """
        self._layout.add_boundary(boundary)
        return boundary

    def reexport(self, port, net_name='', label='', show=True,
                 fill_margin=0, fill_type='', unit_mode=False):
        # type: (Port, str, str, bool, Union[float, int], str, bool) -> None
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
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode : bool
            True if fill_margin is given in resolution units.
        """
        net_name = net_name or port.net_name
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

        # export all port geometries
        port_pins = port_params['pins']
        for wire_arr in port:
            self._used_tracks.add_wire_arrays(wire_arr, fill_margin=fill_margin, fill_type=fill_type,
                                              unit_mode=unit_mode)
            layer_id = wire_arr.layer_id
            if layer_id not in port_pins:
                port_pins[layer_id] = [wire_arr]
            else:
                port_pins[layer_id].append(wire_arr)

    def add_pin_primitive(self, net_name, layer, bbox, label=''):
        # type: (str, str, BBox, str) -> None
        """Add a primitive pin to the layout.

        A primitive pin will not show up as a port.  This is mainly used to add necessary
        label/pin for LVS purposes.

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
        """
        self._layout.add_pin(net_name, layer, bbox, label=label)

    def add_label(self, label, layer, bbox):
        # type: (str, Union[str, Tuple[str, str]], BBox) -> None
        """Adds a label to the layout.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        layer : Union[str, Tuple[str, str]]
            the pin layer name.
        bbox : BBox
            the pin bounding box.
        """
        self._layout.add_label(label, layer, bbox)

    def add_pin(self, net_name, wire_arr_list, label='', show=True):
        # type: (str, Union[WireArray, List[WireArray]], str, bool) -> None
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

        for wire_arr in wire_arr_list:
            # add pin array to port_pins
            layer_id = wire_arr.track_id.layer_id
            port_pins = port_params['pins']
            if layer_id not in port_pins:
                port_pins[layer_id] = [wire_arr]
            else:
                port_pins[layer_id].append(wire_arr)

    def add_via(self,  # type: TemplateBase
                bbox,  # type: BBox
                bot_layer,  # type: Union[str, Tuple[str, str]]
                top_layer,  # type: Union[str, Tuple[str, str]]
                bot_dir,  # type: str
                nx=1,  # type: int
                ny=1,  # type: int
                spx=0.0,  # type: Union[float, int]
                spy=0.0,  # type: Union[float, int]
                extend=True,  # type: bool
                top_dir=None,  # type: Optional[str]
                unit_mode=False,  # type: bool
                ):
        # type: (...) -> Via
        """Adds a (arrayed) via object to the layout.

        Parameters
        ----------
        bbox : BBox
            the via bounding box, not including extensions.
        bot_layer : Union[str, Tuple[str, str]]
            the bottom layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        top_layer : Union[str, Tuple[str, str]]
            the top layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        bot_dir : str
            the bottom layer extension direction.  Either 'x' or 'y'.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : Union[float, int]
            column pitch.
        spy : Union[float, int]
            row pitch.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[str]
            top layer extension direction.  Can force to extend in same direction as bottom.
        unit_mode : bool
            True if spx/spy are specified in resolution units.
        Returns
        -------
        via : Via
            the created via object.
        """
        via = Via(self.grid.tech_info, bbox, bot_layer, top_layer, bot_dir,
                  nx=nx, ny=ny, spx=spx, spy=spy, extend=extend, top_dir=top_dir, unit_mode=unit_mode)
        self._layout.add_via(via)

        return via

    def add_via_primitive(self, via_type,  # type: str
                          loc,  # type: List[float]
                          num_rows=1,  # type: int
                          num_cols=1,  # type: int
                          sp_rows=0.0,  # type: float
                          sp_cols=0.0,  # type: float
                          enc1=None,  # type: Optional[List[float]]
                          enc2=None,  # type: Optional[List[float]]
                          orient='R0',  # type: str
                          cut_width=None,  # type: Optional[float]
                          cut_height=None,  # type: Optional[float]
                          nx=1,  # type: int
                          ny=1,  # type: int
                          spx=0.0,  # type: float
                          spy=0.0,  # type: float
                          unit_mode=False,  # type: bool
                          ):
        # type: (...) -> None
        """Adds a via by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        loc : List[float]
            the via location as a two-element list.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : float
            spacing between via cut rows.
        sp_cols : float
            spacing between via cut columns.
        enc1 : Optional[List[float]]
            a list of left, right, top, and bottom enclosure values on bottom layer.  Defaults to all 0.
        enc2 : Optional[List[float]]
            a list of left, right, top, and bottom enclosure values on top layer.  Defaults. to all 0.
        orient : str
            orientation of the via.
        cut_width : Optional[float]
            via cut width.  This is used to create rectangle via.
        cut_height : Optional[float]
            via cut height.  This is used to create rectangle via.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : float
            column pitch.
        spy : float
            row pitch.
        unit_mode : bool
            True if all given dimensions are in resolution units.
        """
        if unit_mode:
            res = self.grid.resolution
            loc = (loc[0] * res, loc[1] * res)
            sp_rows *= res
            sp_cols *= res
            if enc1 is not None:
                enc1 = [v * res for v in enc1]
            if enc2 is not None:
                enc2 = [v * res for v in enc2]
            if cut_width is not None:
                cut_width *= res
            if cut_height is not None:
                cut_height *= res
            spx *= res
            spy *= res

        self._layout.add_via_primitive(via_type, loc, num_rows=num_rows, num_cols=num_cols,
                                       sp_rows=sp_rows, sp_cols=sp_cols,
                                       enc1=enc1, enc2=enc2, orient=orient,
                                       cut_width=cut_width, cut_height=cut_height,
                                       arr_nx=nx, arr_ny=ny, arr_spx=spx, arr_spy=spy)

    def add_via_on_grid(self, bot_layer_id, bot_track, top_track, bot_width=1, top_width=1):
        # type: (int, Union[float, int], Union[float, int], int, int) -> Via
        """Add a via on the routing grid.

        Parameters
        ----------
        bot_layer_id : int
            the bottom layer ID.
        bot_track : Union[float, int]
            the bottom track index.
        top_track : Union[float, int]
            the top track index.
        bot_width : int
            the bottom track width.
        top_width : int
            the top track width.
        """
        grid = self.grid
        res = grid.resolution
        bl, bu = grid.get_wire_bounds(bot_layer_id, bot_track, width=bot_width, unit_mode=True)
        tl, tu = grid.get_wire_bounds(bot_layer_id + 1, top_track, width=top_width, unit_mode=True)
        bot_dir = grid.get_direction(bot_layer_id)
        if bot_dir == 'x':
            bbox = BBox(tl, bl, tu, bu, res, unit_mode=True)
        else:
            bbox = BBox(bl, tl, bu, tu, res, unit_mode=True)
        bname = grid.get_layer_name(bot_layer_id, bot_track)
        tname = grid.get_layer_name(bot_layer_id + 1, top_track)

        return self.add_via(bbox, bname, tname, bot_dir)

    def extend_wires(self,  # type: TemplateBase
                     warr_list,  # type: Union[WireArray, List[WireArray]]
                     lower=None,  # type: Optional[Union[float, int]]
                     upper=None,  # # type: Optional[Union[float, int]]
                     fill_margin=0,  # type: Union[int, float]
                     fill_type='',  # type: str
                     unit_mode=False  # type: bool
                     ):
        # type: (...) -> List[WireArray]
        """Extend the given wires to the given coordinates.

        Parameters
        ----------
        warr_list : Union[WireArray, List[WireArray]]
            the wires to extend.
        lower : Optional[Union[float, int]]
            the wire lower coordinate.
        upper : Optional[Union[float, int]]
            the wire upper coordinate.
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if lower/upper/fill_margin is given in resolution units.

        Returns
        -------
        warr_list : List[WireArray]
            list of added wire arrays.
        """
        if isinstance(warr_list, WireArray):
            warr_list = [warr_list]
        else:
            pass

        res = self.grid.resolution
        if not unit_mode:
            if lower is not None:
                lower = int(round(lower / res))
            if upper is not None:
                upper = int(round(upper / res))
            fill_margin = int(round(fill_margin / res))

        new_warr_list = []
        for warr in warr_list:
            wlower = int(round(warr.lower / res))
            wupper = int(round(warr.upper / res))
            if lower is None:
                cur_lower = wlower
            else:
                cur_lower = min(lower, wlower)
            if upper is None:
                cur_upper = wupper
            else:
                cur_upper = max(upper, wupper)

            new_warr = WireArray(warr.track_id, cur_lower * res, cur_upper * res)
            for layer_name, bbox_arr in new_warr.wire_arr_iter(self.grid):
                self.add_rect(layer_name, bbox_arr)

            self._used_tracks.add_wire_arrays(new_warr, fill_margin=fill_margin, fill_type=fill_type,
                                              unit_mode=True)
            new_warr_list.append(new_warr)

        return new_warr_list

    def add_wires(self,  # type: TemplateBase
                  layer_id,  # type: int
                  track_idx,  # type: Union[float, int]
                  lower,  # type: Union[float, int]
                  upper,  # type: Union[float, int]
                  width=1,  # type: int
                  num=1,  # type: int
                  pitch=0,  # type: Union[float, int]
                  fill_margin=0,  # type: Union[int, float]
                  fill_type='',  # type: str
                  unit_mode=False  # type: bool
                  ):
        # type: (...) -> WireArray
        """Add the given wire(s) to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : Union[float, int]
            the smallest wire track index.
        lower : Union[float, int]
            the wire lower coordinate.
        upper : Union[float, int]
            the wire upper coordinate.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : Union[float, int]
            the wire pitch.
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if lower/upper/fill_margin is given in resolution units.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        res = self.grid.resolution
        if unit_mode:
            lower *= res
            upper *= res

        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        warr = WireArray(tid, lower, upper)

        for layer_name, bbox_arr in warr.wire_arr_iter(self.grid):
            self.add_rect(layer_name, bbox_arr)

        self._used_tracks.add_wire_arrays(warr, fill_margin=fill_margin, fill_type=fill_type,
                                          unit_mode=unit_mode)

        return warr

    def add_res_metal_warr(self,  # type: TemplateBase
                           layer_id,  # type: int
                           track_idx,  # type: Union[float, int]
                           lower,  # type: Union[float, int]
                           upper,  # type: Union[float, int]
                           **kwargs):
        # type: (...) -> WireArray
        """Add metal resistor as WireArray to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : Union[float, int]
            the smallest wire track index.
        lower : Union[float, int]
            the wire lower coordinate.
        upper : Union[float, int]
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
            self.add_res_metal(layer_id, bbox_arr)

        return warr

    def add_mom_cap(self,  # type: TemplateBase
                    cap_box,  # type: BBox
                    bot_layer,  # type: int
                    num_layer,  # type: int
                    port_widths=1,  # type: Union[int, List[int], Dict[int, int]
                    port_parity=None,  # type: Optional[Dict[int, Tuple[int, int]]]
                    fill_margin=0,  # type: Union[float, int]
                    fill_type='',  # type: str
                    unit_mode=False,  # type: bool
                    array=False,  # type: bool
                    ):
        # type: (...) -> Dict[int, Tuple[List[WireArray], List[WireArray]]]
        """Draw mom cap in the defined bounding box."""
        if num_layer <= 1:
            raise ValueError('Must have at least 2 layers for MOM cap.')

        top_layer = bot_layer + num_layer - 1

        if isinstance(port_widths, int):
            port_widths = {lay: port_widths for lay in range(bot_layer, top_layer + 1)}
        elif isinstance(port_widths, list):
            port_widths = dict(zip(range(bot_layer, top_layer + 1), port_widths))
        else:
            pass

        if port_parity is None:
            port_parity = {lay: (0, 1) for lay in range(bot_layer, top_layer + 1)}
        else:
            pass

        res = self.grid.resolution
        tech_info = self.grid.tech_info
        if not unit_mode:
            fill_margin = int(round(fill_margin / res))

        mom_cap_dict = tech_info.tech_params['layout']['mom_cap']
        cap_margins = mom_cap_dict['margins']
        cap_info = mom_cap_dict['width_space']

        via_ext_dict = {lay: 0 for lay in range(bot_layer, top_layer + 1)}
        # get via extensions on each layer
        for vbot_layer in range(bot_layer, top_layer):
            vtop_layer = vbot_layer + 1
            bport_w = self.grid.get_track_width(vbot_layer, port_widths[vbot_layer], unit_mode=True)
            tport_w = self.grid.get_track_width(vtop_layer, port_widths[vtop_layer], unit_mode=True)
            bcap_w = int(round(cap_info[vbot_layer][0] / res))
            tcap_w = int(round(cap_info[vtop_layer][0] / res))

            # port-to-port via
            vbext1, vtext1 = self.grid.get_via_extensions_dim(vbot_layer, bport_w, tport_w, unit_mode=True)
            # cap-to-port via
            vbext2 = self.grid.get_via_extensions_dim(vbot_layer, bcap_w, tport_w, unit_mode=True)[0]
            # port-to-cap via
            vtext2 = self.grid.get_via_extensions_dim(vbot_layer, bport_w, tcap_w, unit_mode=True)[1]

            # record extension due to via
            via_ext_dict[vbot_layer] = max(via_ext_dict[vbot_layer], vbext1, vbext2)
            via_ext_dict[vtop_layer] = max(via_ext_dict[vtop_layer], vtext1, vtext2)

        # find port locations and cap boundaries.
        port_tracks = {}
        cap_bounds = {}
        cap_exts = {}
        for cur_layer in range(bot_layer, top_layer + 1):
            cur_port_width = port_widths[cur_layer]
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
                tr_lower = self.grid.coord_to_track(cur_layer, cur_lower, unit_mode=True)
                tr_upper = self.grid.coord_to_track(cur_layer, cur_upper, unit_mode=True)
            else:
                tr_lower = self.grid.find_next_track(cur_layer, cur_lower + adj_via_ext, tr_width=cur_port_width,
                                                     half_track=True, mode=1, unit_mode=True)
                tr_upper = self.grid.find_next_track(cur_layer, cur_upper - adj_via_ext, tr_width=cur_port_width,
                                                     half_track=True, mode=-1, unit_mode=True)

            tll, tlu = self.grid.get_wire_bounds(cur_layer, tr_lower, width=cur_port_width, unit_mode=True)
            tul, tuu = self.grid.get_wire_bounds(cur_layer, tr_upper, width=cur_port_width, unit_mode=True)

            # compute space from MOM cap wires to port wires
            lay_name = tech_info.get_layer_name(cur_layer)
            if isinstance(lay_name, tuple) or isinstance(lay_name, list):
                lay_name = lay_name[0]
            lay_type = tech_info.get_layer_type(lay_name)
            cur_margin = int(round(cap_margins[cur_layer] / res))
            cur_margin = max(cur_margin, tech_info.get_min_space(lay_type, tlu - tll, unit_mode=True))

            port_tracks[cur_layer] = (tr_lower, tr_upper)
            cap_bounds[cur_layer] = (tlu + cur_margin, tul - cur_margin)
            cap_exts[cur_layer] = (tll, tuu)

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

            via_ext = via_ext_dict[cur_layer]
            lower -= via_ext
            upper += via_ext

            # draw lower and upper ports
            tr_lower, tr_upper = port_tracks[cur_layer]
            lwarr = self.add_wires(cur_layer, tr_lower, lower, upper, width=cur_port_width,
                                   fill_margin=fill_margin, fill_type=fill_type, unit_mode=True)
            uwarr = self.add_wires(cur_layer, tr_upper, lower, upper, width=cur_port_width,
                                   fill_margin=fill_margin, fill_type=fill_type, unit_mode=True)

            # assign port wires to positive/negative terminals
            plist, nlist = [], []
            lpar, upar = port_parity[cur_layer]
            cur_num_ports = 2 if lpar != upar else 1
            for par, warr in zip((lpar, upar), (lwarr, uwarr)):
                if par == 0:
                    plist.append(warr)
                else:
                    nlist.append(warr)
            port_dict[cur_layer] = plist, nlist
            if cur_layer != bot_layer:
                # connect ports to layer below
                for clist, blist in zip((plist, nlist), port_dict[cur_layer - 1]):
                    for cur_warr in clist:
                        cur_tid = cur_warr.track_id.base_index
                        cur_w = cur_warr.track_id.width
                        for bot_warr in blist:
                            bot_tid = bot_warr.track_id.base_index
                            bot_w = bot_warr.track_id.width
                            self.add_via_on_grid(cur_layer - 1, bot_tid, cur_tid, bot_width=bot_w,
                                                 top_width=cur_w)

            # mark all in-between tracks as used
            fake_warr = WireArray(TrackID(cur_layer, tr_lower + 1, num=int(tr_upper - tr_lower) + 1, pitch=1),
                                  lower * res, upper * res)
            self._used_tracks.add_wire_arrays(fake_warr)

            # draw cap wires
            cap_lower, cap_upper = cap_bounds[cur_layer]
            cap_tot_space = cap_upper - cap_lower
            cap_w, cap_sp = cap_info[cur_layer]
            cap_w = int(round(cap_w / res))
            cap_sp = int(round(cap_sp / res))
            cap_pitch = cap_w + cap_sp
            num_cap_wires = cap_tot_space // cap_pitch
            if (num_cap_wires + cur_num_ports) % 2 != 0:
                # number of cap wires and number of ports on this layer should have same parity
                num_cap_wires -= 1
            cap_lower += (cap_tot_space - (num_cap_wires * cap_pitch - cap_sp)) // 2

            is_horizontal = (self.grid.get_direction(cur_layer) == 'x')

            if is_horizontal:
                wbox = BBox(lower, cap_lower, upper, cap_lower + cap_w, res, unit_mode=True)
            else:
                wbox = BBox(cap_lower, lower, cap_lower + cap_w, upper, res, unit_mode=True)

            lay_name = tech_info.get_layer_name(cur_layer)
            num2 = num_cap_wires // 2
            num1 = num_cap_wires - num2
            pitch_capw = cap_pitch * 2 * res
            # find layer names for even/odd cap wires
            if isinstance(lay_name, tuple) or isinstance(lay_name, list):
                if len(lay_name) != 2:
                    # TODO: support triple+ patterning layers?
                    raise ValueError('Only double patterning is supported.')

                lay_name0, lay_name1 = lay_name
            else:
                lay_name0, lay_name1 = lay_name, lay_name

            # draw cap wires and assign to port
            if is_horizontal:
                wbox2 = wbox.move_by(dy=cap_pitch, unit_mode=True)
                self.add_rect(lay_name0, wbox, ny=num1, spy=pitch_capw)
                self.add_rect(lay_name1, wbox2, ny=num2, spy=pitch_capw)
            else:
                wbox2 = wbox.move_by(dx=cap_pitch, unit_mode=True)
                self.add_rect(lay_name0, wbox, nx=num1, spx=pitch_capw)
                self.add_rect(lay_name1, wbox2, nx=num2, spx=pitch_capw)
            # assign cap wires to ports
            if lpar == 1:
                cap_wire_dict[cur_layer] = [(lay_name0, wbox, num1),
                                            (lay_name1, wbox2, num2)], pitch_capw
            else:
                cap_wire_dict[cur_layer] = [(lay_name1, wbox2, num2),
                                            (lay_name0, wbox, num1)], pitch_capw

        # draw cap wires and vias
        for cur_layer in range(bot_layer, top_layer):
            cur_infos, cur_pitch = cap_wire_dict[cur_layer]
            next_infos, next_pitch = cap_wire_dict[cur_layer + 1]
            cur_ports = port_dict[cur_layer]
            next_ports = port_dict[cur_layer + 1]
            cur_dir = self.grid.get_direction(cur_layer)
            is_horizontal = (cur_dir == 'x')
            if is_horizontal:
                spx, spy = next_pitch, cur_pitch
            else:
                spx, spy = cur_pitch, next_pitch

            info_iter = zip(cur_infos, next_infos, cur_ports, next_ports)
            for (cur_name, cur_box, cur_num), (next_name, next_box, next_num), cplist, nplist in info_iter:
                # connect cap wire to cap wire
                vbox = cur_box.intersect(next_box)
                if is_horizontal:
                    nx, ny = next_num, cur_num
                else:
                    nx, ny = cur_num, next_num
                self.add_via(vbox, cur_name, next_name, cur_dir, nx=nx, ny=ny, spx=spx, spy=spy)
                # connect cap wire to port
                for npwarr in nplist:
                    port_lay_name = self.grid.get_layer_name(cur_layer + 1, npwarr.track_id.base_index)
                    vbox = cur_box.intersect(npwarr.get_bbox_array(self.grid).base)
                    if is_horizontal:
                        self.add_via(vbox, cur_name, port_lay_name, cur_dir, ny=ny, spy=spy)
                    else:
                        self.add_via(vbox, cur_name, port_lay_name, cur_dir, nx=nx, spx=spx)
                # connect port to cap wire
                for cpwarr in cplist:
                    port_lay_name = self.grid.get_layer_name(cur_layer, cpwarr.track_id.base_index)
                    vbox = next_box.intersect(cpwarr.get_bbox_array(self.grid).base)
                    if is_horizontal:
                        self.add_via(vbox, port_lay_name, next_name, cur_dir, nx=nx, spx=spx)
                    else:
                        self.add_via(vbox, port_lay_name, next_name, cur_dir, ny=ny, spy=spy)

        return port_dict

    def reserve_tracks(self,  # type: TemplateBase
                       layer_id,  # type: int
                       track_idx,  # type: Union[float, int]
                       width=1,  # type: int
                       num=1,  # type: int
                       pitch=0,  # type: Union[float, int]
                       fill_margin=0,  # type: Union[int, float]
                       fill_type='',  # type: str
                       unit_mode=False  # type: bool
                       ):
        # type: (...) -> None
        """Reserve the given routing tracks so that power fill will not fill these tracks.

        Note: the size of this template should be set before calling this method.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : Union[float, int]
            the smallest wire track index.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : Union[float, int]
            the wire pitch.
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if fill_margin is given in resolution units.
        """

        bnd_box = self.bound_box
        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        if self.grid.get_direction(layer_id) == 'x':
            upper = bnd_box.width
        else:
            upper = bnd_box.height
        warr = WireArray(tid, 0.0, upper)

        self._used_tracks.add_wire_arrays(warr, fill_margin=fill_margin, fill_type=fill_type,
                                          unit_mode=unit_mode)

    def connect_wires(self,  # type: TemplateBase
                      wire_arr_list,  # type: Union[WireArray, List[WireArray]]
                      lower=None,  # type: Optional[Union[int, float]]
                      upper=None,  # type: Optional[Union[int, float]]
                      debug=False,  # type: bool
                      fill_margin=0,  # type: Union[int, float]
                      fill_type='',  # type: str
                      unit_mode=False  # type: bool
                      ):
        # type: (...) -> List[WireArray]
        """Connect all given WireArrays together.

        all WireArrays must be on the same layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArr, List[WireArr]]
            WireArrays to connect together.
        lower : Optional[Union[int, float]]
            if given, extend connection wires to this lower coordinate.
        upper : Optional[Union[int, float]]
            if given, extend connection wires to this upper coordinate.
        debug : bool
            True to print debug messages.
        fill_margin : Union[float, int]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if lower/upper/fill_margin is given in resolution units.

        Returns
        -------
        conn_list : List[WireArray]
            list of connection wires created.
        """
        grid = self.grid
        res = grid.resolution

        if not unit_mode:
            fill_margin = int(round(fill_margin / res))
            if lower is not None:
                lower = int(round(lower / res))
            if upper is not None:
                upper = int(round(upper / res))

        if isinstance(wire_arr_list, WireArray):
            wire_arr_list = [wire_arr_list]
        else:
            pass

        if not wire_arr_list:
            # do nothing
            return []

        # calculate wire vertical coordinates
        a = wire_arr_list[0]
        layer_id = a.layer_id
        direction = grid.get_direction(layer_id)
        perp_dir = 'y' if direction == 'x' else 'x'
        htr_pitch = grid.get_track_pitch(layer_id, unit_mode=True) // 2
        intv_set = IntervalSet()

        for wire_arr in wire_arr_list:
            if wire_arr.layer_id != layer_id:
                raise ValueError('WireArray layer ID != %d' % layer_id)

            cur_range = (int(round(wire_arr.lower / res)),
                         int(round(wire_arr.upper / res)))
            if lower is not None:
                cur_range = (min(cur_range[0], lower), max(cur_range[1], lower))
            if upper is not None:
                cur_range = (min(cur_range[0], upper), max(cur_range[1], upper))

            box_arr = wire_arr.get_bbox_array(grid)
            for box in box_arr:
                intv = box.get_interval(perp_dir, unit_mode=True)
                try:
                    old_range = intv_set[intv]
                    intv_set[intv] = min(cur_range[0], old_range[0]), max(cur_range[1], old_range[1])
                except KeyError:
                    success = intv_set.add(intv, cur_range)
                    if not success:
                        raise ValueError('wire interval {} overlap existing wires.'.format(intv))

        # draw wires, group into arrays
        new_warr_list = []
        base_range = None
        base_intv = None
        base_width = None
        count = 0
        hpitch = 0
        last_lower = 0
        for intv, wrange in intv_set.items():
            if debug:
                print('wires intv: %s, range: %s' % (intv, wrange))
            cur_width = intv[1] - intv[0]
            cur_lower = intv[0]
            if count == 0:
                base_range = wrange
                base_intv = intv
                base_width = intv[1] - intv[0]
                count = 1
                hpitch = 0
            else:
                if wrange[0] == base_range[0] and \
                                wrange[1] == base_range[1] and \
                                base_width == cur_width:
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
                        tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv, unit_mode=True)
                        track_id = TrackID(layer_id, tr_idx, tr_width, num=count, pitch=hpitch / 2)
                        warr = WireArray(track_id, base_range[0] * res, base_range[1] * res)
                        for layer_name, bbox_arr in warr.wire_arr_iter(grid):
                            self.add_rect(layer_name, bbox_arr)
                        new_warr_list.append(warr)
                        base_range = wrange
                        base_intv = intv
                        base_width = cur_width
                        count = 1
                        hpitch = 0
                else:
                    # length/width does not match, add cumulated wires and start anew
                    tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv, unit_mode=True)
                    track_id = TrackID(layer_id, tr_idx, tr_width, num=count, pitch=hpitch / 2)
                    warr = WireArray(track_id, base_range[0] * res, base_range[1] * res)
                    for layer_name, bbox_arr in warr.wire_arr_iter(grid):
                        self.add_rect(layer_name, bbox_arr)
                    new_warr_list.append(warr)
                    base_range = wrange
                    base_intv = intv
                    base_width = cur_width
                    count = 1
                    hpitch = 0

            # update last lower coordinate
            last_lower = cur_lower

        # add last wires
        tr_idx, tr_width = grid.interval_to_track(layer_id, base_intv, unit_mode=True)
        track_id = TrackID(layer_id, tr_idx, tr_width, num=count, pitch=hpitch / 2)
        warr = WireArray(track_id, base_range[0] * res, base_range[1] * res)
        for layer_name, bbox_arr in warr.wire_arr_iter(grid):
            self.add_rect(layer_name, bbox_arr)
        new_warr_list.append(warr)

        self._used_tracks.add_wire_arrays(new_warr_list, fill_margin=fill_margin, fill_type=fill_type,
                                          unit_mode=True)
        return new_warr_list

    def _draw_via_on_track(self, wlayer, box_arr, track_id, tl_unit=None,
                           tu_unit=None):
        # type: (str, BBoxArray, TrackID, float, float) -> Tuple[float, float]
        """Helper method.  Draw vias on the intersection of the BBoxArray and TrackID."""
        grid = self.grid
        res = grid.resolution

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
            tl, tu = grid.get_wire_bounds(tr_layer_id, base_idx, width=tr_width, unit_mode=True)
            if tr_dir == 'x':
                via_box = BBox(wbase.left_unit, tl, wbase.right_unit, tu, res, unit_mode=True)
                nx, ny = box_arr.nx, sub_track_id.num
                spx, spy = box_arr.spx, sub_track_id.pitch * tr_pitch
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
                via_box = BBox(tl, wbase.bottom_unit, tu, wbase.top_unit, res, unit_mode=True)
                nx, ny = sub_track_id.num, box_arr.ny
                spx, spy = sub_track_id.pitch * tr_pitch, box_arr.spy
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

        return tl_unit, tu_unit

    def connect_bbox_to_tracks(self,  # type: TemplateBase
                               layer_name,  # type: str
                               box_arr,  # type: Union[BBox, BBoxArray]
                               track_id,  # type: TrackID
                               track_lower=None,  # type: Optional[Union[int, float]]
                               track_upper=None,  # type: Optional[Union[int, float]]
                               fill_margin=0,  # type: Union[int, float]
                               fill_type='',  # type: str
                               unit_mode=False  # type: bool
                               ):
        # type: (...) -> WireArray
        """Connect the given lower layer to given tracks.

        This method is used to connect layer below RoutingGrid to RoutingGrid.

        Parameters
        ----------
        layer_name : str
            the lower level layer name.
        box_arr : Union[BBox, BBoxArray]
            bounding box of the wire(s) to connect to tracks.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        track_lower : Optional[Union[int, float]]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[Union[int, float]]
            if given, extend track(s) to this upper coordinate.
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if track_lower/track_upper/fill_margin is given in resolution units.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if isinstance(box_arr, BBox):
            box_arr = BBoxArray(box_arr)
        else:
            pass

        res = self.grid.resolution

        # extend bounding boxes to tracks
        tl, tu = track_id.get_bounds(self.grid, unit_mode=True)
        tr_dir = self.grid.get_direction(track_id.layer_id)
        base = box_arr.base
        if tr_dir == 'x':
            self.add_rect(layer_name, base.extend(y=tl, unit_mode=True).extend(y=tu, unit_mode=True),
                          nx=box_arr.nx, ny=box_arr.ny, spx=box_arr.spx, spy=box_arr.spy)
        else:
            self.add_rect(layer_name, base.extend(x=tl, unit_mode=True).extend(x=tu, unit_mode=True),
                          nx=box_arr.nx, ny=box_arr.ny, spx=box_arr.spx, spy=box_arr.spy)

        # draw vias
        tl_unit = track_lower
        tu_unit = track_upper
        if not unit_mode:
            fill_margin = int(round(fill_margin / res))
            if track_lower is not None:
                tl_unit = int(round(track_lower / res))
            if track_upper is not None:
                tu_unit = int(round(track_upper / res))

        tl_unit, tu_unit = self._draw_via_on_track(layer_name, box_arr, track_id,
                                                   tl_unit=tl_unit, tu_unit=tu_unit)

        # draw tracks
        result = WireArray(track_id, tl_unit * res, tu_unit * res)
        for layer_name, bbox_arr in result.wire_arr_iter(self.grid):
            self.add_rect(layer_name, bbox_arr)

        self._used_tracks.add_wire_arrays(result, fill_margin=fill_margin, fill_type=fill_type,
                                          unit_mode=True)
        return result

    def connect_to_tracks(self,  # type: TemplateBase
                          wire_arr_list,  # type: Union[WireArray, List[WireArray]]
                          track_id,  # type: TrackID
                          wire_lower=None,  # type: Optional[Union[float, int]]
                          wire_upper=None,  # type: Optional[Union[float, int]]
                          track_lower=None,  # type: Optional[Union[float, int]]
                          track_upper=None,  # type: Optional[Union[float, int]]
                          fill_margin=0,  # type: Union[int, float]
                          fill_type='',  # type: str
                          unit_mode=False,  # type: bool
                          min_len_mode=None,  # type: Optional[int]
                          debug=False,  # type: bool
                          ):
        # type: (...) -> Optional[WireArray]
        """Connect all given WireArrays to the given track(s).

        All given wires should be on adjacent layers of the track.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        wire_lower : Optional[Union[float, int]]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[Union[float, int]]
            if given, extend wire(s) to this upper coordinate.
        track_lower : Optional[Union[float, int]]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[Union[float, int]]
            if given, extend track(s) to this upper coordinate.
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode : bool
            True if track_lower/track_upper/fill_margin is given in resolution units.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Optional[WireArray]
            WireArray representing the tracks created.  None if nothing to do.
        """
        if isinstance(wire_arr_list, WireArray):
            # convert to list.
            wire_arr_list = [wire_arr_list]
        else:
            pass

        if not wire_arr_list:
            # do nothing
            return None

        grid = self.grid
        res = grid.resolution

        if not unit_mode:
            fill_margin = int(round(fill_margin / res))
            if track_upper is not None:
                track_upper = int(round(track_upper / res))
            if track_lower is not None:
                track_lower = int(round(track_lower / res))

        # find min/max track Y coordinates
        tr_layer_id = track_id.layer_id
        wl, wu = track_id.get_bounds(grid, unit_mode=True)
        if wire_lower is not None:
            if not unit_mode:
                wire_lower = int(round(wire_lower / res))
            wl = min(wire_lower, wl)

        if wire_upper is not None:
            if not unit_mode:
                wire_upper = int(round(wire_upper / res))
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
                raise ValueError('WireArray layer %d cannot connect to layer %d' % (cur_layer_id, tr_layer_id))

        # connect wires together
        top_wire_list = self.connect_wires(top_list, lower=wl, upper=wu, fill_margin=fill_margin,
                                           fill_type=fill_type, unit_mode=True, debug=debug)
        bot_wire_list = self.connect_wires(bot_list, lower=wl, upper=wu, fill_margin=fill_margin,
                                           fill_type=fill_type, unit_mode=True, debug=debug)

        # draw vias
        for w_layer_id, wire_list in ((tr_layer_id + 1, top_wire_list), (tr_layer_id - 1, bot_wire_list)):
            for wire_arr in wire_list:
                for wlayer, box_arr in wire_arr.wire_arr_iter(grid):
                    track_lower, track_upper = self._draw_via_on_track(wlayer, box_arr, track_id,
                                                                       tl_unit=track_lower, tu_unit=track_upper)

        if min_len_mode is not None:
            # extend track to meet minimum length
            min_len = grid.get_min_length(tr_layer_id, track_id.width, unit_mode=True)
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
                    track_upper += (ext - ext // 2)

        # draw tracks
        result = WireArray(track_id, track_lower * res, track_upper * res)
        for layer_name, bbox_arr in result.wire_arr_iter(grid):
            self.add_rect(layer_name, bbox_arr)

        self._used_tracks.add_wire_arrays(result, fill_margin=fill_margin, fill_type=fill_type, unit_mode=True)
        return result

    def connect_with_via_stack(self,  # type: TemplateBase
                               wire_array,  # type: Union[WireArray, List[WireArray]]
                               track_id,  # type: TrackID
                               tr_w_list=None,  # type: Optional[List[int]]
                               tr_mode_list=None,  # type: Optional[List[int]]
                               min_len_mode_list=None,  # type: Optional[List[int]]
                               fill_margin=0,  # type: Union[int, float]
                               fill_type='',  # type: str
                               unit_mode=False,  # type: bool
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
        tr_mode_list : Optional[List[int]]
            If tracks on intermediate layers do not line up nicely,
            the track mode flags determine whether to pick upper or lower tracks
        min_len_mode_list : Optional[List[int]]
            minimum length mode flags on each layer.
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode : bool
            True if fill_margin is given in resolution units.
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
                raise ValueError('connect_with_via_stack() only works on WireArray and TrackID with a single wire.')
            # convert to WireArray.
            wire_array = wire_array[0]

        # error checking
        warr_tid = wire_array.track_id
        warr_layer = warr_tid.layer_id
        tr_layer = track_id.layer_id
        tr_index = track_id.base_index
        if warr_tid.num != 1 or track_id.num != 1:
            raise ValueError('connect_with_via_stack() only works on WireArray and TrackID with a single wire.')
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
        elif len(tr_mode_list) != num_connections:
            raise ValueError('tr_mode_list must have exactly %d elements.' % num_connections)

        if min_len_mode_list is None:
            min_len_mode_list = [None] * num_connections
        elif len(min_len_mode_list) != num_connections:
            raise ValueError('min_len_mode_list must have exactly %d elements.' % num_connections)

        # determine via location
        grid = self.grid
        w_dir = grid.get_direction(warr_layer)
        t_dir = grid.get_direction(tr_layer)
        w_coord = grid.track_to_coord(warr_layer, warr_tid.base_index, unit_mode=True)
        t_coord = grid.track_to_coord(tr_layer, tr_index, unit_mode=True)
        if w_dir != t_dir:
            x0, y0 = (w_coord, t_coord) if w_dir == 'y' else (t_coord, w_coord)
        else:
            w_mid = int(round(wire_array.middle / grid.resolution))
            x0, y0 = (w_coord, w_mid) if w_dir == 'y' else (w_mid, w_coord)

        # determine track width on each layer
        tr_w_list[num_connections - 1] = track_id.width
        if tr_layer > warr_layer:
            layer_dir = 1
            tr_w_prev = grid.get_track_width(tr_layer, tr_w_list[num_connections - 1], unit_mode=True)
            tr_w_idx_iter = range(num_connections - 2, -1, -1)
        else:
            layer_dir = -1
            tr_w_prev = grid.get_track_width(warr_layer, warr_tid.width, unit_mode=True)
            tr_w_idx_iter = range(0, num_connections - 1)
        for idx in tr_w_idx_iter:
            cur_layer = warr_layer + layer_dir * (idx + 1)
            if tr_w_list[idx] < 0:
                tr_w_list[idx] = grid.get_track_width_inverse(cur_layer, tr_w_prev, unit_mode=True)
            tr_w_prev = grid.get_track_width(cur_layer, tr_w_list[idx], unit_mode=True)

        # draw via stacks
        results = []
        targ_layer = warr_layer
        for tr_w, tr_mode, min_len_mode in zip(tr_w_list, tr_mode_list, min_len_mode_list):
            targ_layer += layer_dir

            # determine track index to connect to
            if targ_layer == tr_layer:
                targ_index = tr_index
            else:
                targ_dir = grid.get_direction(targ_layer)
                coord = x0 if targ_dir == 'y' else y0
                targ_index = grid.coord_to_nearest_track(targ_layer, coord, half_track=True,
                                                         mode=tr_mode, unit_mode=True)

            targ_tid = TrackID(targ_layer, targ_index, width=tr_w)
            warr = self.connect_to_tracks(wire_array, targ_tid, fill_margin=fill_margin, fill_type=fill_type,
                                          unit_mode=unit_mode, min_len_mode=min_len_mode, debug=debug)
            results.append(warr)
            wire_array = warr

        return results

    def strap_wires(self,  # type: TemplateBase
                    warr,  # type: WireArray
                    targ_layer,  # type: int
                    tr_w_list=None,  # type: Optional[List[int]]
                    min_len_mode_list=None,  # type: Optional[List[int]]
                    fill_margin=0,  # type: Union[int, float]
                    fill_type='',  # type: str
                    unit_mode=False,  # type: bool
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
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode : bool
            True if track_lower/track_upper/fill_margin is given in resolution units.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.  None if nothing to do.
        """
        warr_layer = warr.layer_id

        if targ_layer == warr_layer:
            # no need to do anything
            return warr

        num_connections = abs(targ_layer - warr_layer)

        # set default values
        if tr_w_list is None:
            tr_w_list = [-1] * num_connections
        elif len(tr_w_list) != num_connections:
            raise ValueError('tr_w_list must have exactly %d elements.' % num_connections)
        else:
            # create a copy of the given list, as this list may be modified later.
            tr_w_list = list(tr_w_list)

        if min_len_mode_list is None:
            min_len_mode_list = [None] * num_connections
        elif len(min_len_mode_list) != num_connections:
            raise ValueError('min_len_mode_list must have exactly %d elements.' % num_connections)

        layer_dir = 1 if targ_layer > warr_layer else -1
        for tr_w, mlen_mode in zip(tr_w_list, min_len_mode_list):
            warr = self._strap_wires_helper(warr, warr.layer_id + layer_dir, tr_w, mlen_mode,
                                            fill_margin, fill_type, unit_mode)

        return warr

    def _strap_wires_helper(self,  # type: TemplateBase
                            warr,  # type: WireArray
                            targ_layer,  # type: int
                            tr_w,  # type: int
                            mlen_mode,  # type: Optional[int]
                            fill_margin,  # type: Union[int, float]
                            fill_type,  # type: str
                            unit_mode,  # type: bool
                            ):
        # type: (...) -> WireArray
        """Helper method for strap_wires().  Connect one layer at a time."""
        wire_tid = warr.track_id
        wire_layer = wire_tid.layer_id

        res = self.grid.resolution
        lower = int(round(warr.lower / res))
        upper = int(round(warr.upper / res))
        if not unit_mode:
            fill_margin = int(round(fill_margin / res))

        # error checking
        wdir = self.grid.get_direction(wire_layer)
        if wdir != self.grid.get_direction(targ_layer):
            raise ValueError('Cannot strap wires with different directions.')

        # convert base track index
        base_coord = self.grid.track_to_coord(wire_layer, wire_tid.base_index, unit_mode=True)
        base_tid = self.grid.coord_to_track(targ_layer, base_coord, unit_mode=True)
        # convert pitch
        wire_pitch = self.grid.get_track_pitch(wire_layer, unit_mode=True)
        targ_pitch = self.grid.get_track_pitch(targ_layer, unit_mode=True)
        targ_pitch_half = targ_pitch // 2
        pitch_unit = int(round(wire_pitch * wire_tid.pitch))
        if pitch_unit % targ_pitch_half != 0:
            raise ValueError('Cannot strap wires on layers with mismatched pitch ')
        num_pitch = pitch_unit // targ_pitch_half
        if num_pitch % 2 == 0:
            num_pitch //= 2
        else:
            num_pitch /= 2
        # convert width
        if tr_w < 0:
            width_unit = self.grid.get_track_width(wire_layer, wire_tid.width, unit_mode=True)
            tr_w = max(1, self.grid.get_track_width_inverse(targ_layer, width_unit, mode=-1, unit_mode=True))

        # draw vias.  Update WireArray lower/upper
        new_lower, new_upper = lower, upper
        w_lower, w_upper = lower, upper
        for tid in wire_tid:
            coord = self.grid.track_to_coord(wire_layer, tid, unit_mode=True)
            tid2 = self.grid.coord_to_track(targ_layer, coord, unit_mode=True)
            w_name = self.grid.get_layer_name(wire_layer, tid)
            t_name = self.grid.get_layer_name(targ_layer, tid2)

            w_yb, w_yt = self.grid.get_wire_bounds(wire_layer, tid, wire_tid.width, unit_mode=True)
            t_yb, t_yt = self.grid.get_wire_bounds(targ_layer, tid2, tr_w, unit_mode=True)
            vbox = BBox(lower, max(w_yb, t_yb), upper, min(w_yt, t_yt), res, unit_mode=True)
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
        min_len = self.grid.get_min_length(targ_layer, tr_w, unit_mode=True)
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
                       num=wire_tid.num, pitch=wire_tid.pitch, fill_margin=fill_margin,
                       fill_type=fill_type, unit_mode=True)
        return self.add_wires(targ_layer, base_tid, new_lower, new_upper, width=tr_w,
                              num=wire_tid.num, pitch=num_pitch, fill_margin=fill_margin,
                              fill_type=fill_type, unit_mode=True)

    def connect_differential_tracks(self,  # type: TemplateBase
                                    pwarr_list,  # type: Union[WireArray, List[WireArray]]
                                    nwarr_list,  # type: Union[WireArray, List[WireArray]]
                                    tr_layer_id,  # type: int
                                    ptr_idx,  # type: Union[int, float]
                                    ntr_idx,  # type: Union[int, float]
                                    width=1,  # type: int
                                    track_lower=None,  # type: Optional[Union[float, int]]
                                    track_upper=None,  # type: Optional[Union[float, int]]
                                    fill_margin=0,  # type: Union[int, float]
                                    fill_type='',  # type: str
                                    unit_mode=False,  # type: bool
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
        ptr_idx : Union[int, float]
            positive track index.
        ntr_idx : Union[int, float]
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[Union[float, int]]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[Union[float, int]]
            if given, extend track(s) to this upper coordinate.
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if track_lower/track_upper/fill_margin is given in resolution units.
        debug : bool
            True to print debug messages.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_matching_tracks([pwarr_list, nwarr_list], tr_layer_id, [ptr_idx, ntr_idx],
                                                  width=width, track_lower=track_lower, track_upper=track_upper,
                                                  fill_margin=fill_margin, fill_type=fill_type, unit_mode=unit_mode,
                                                  debug=debug)
        return track_list[0], track_list[1]

    def connect_matching_tracks(self,  # type: TemplateBase
                                warr_list_list,  # type: List[Union[WireArray, List[WireArray]]]
                                tr_layer_id,  # type: int
                                tr_idx_list,  # type: List[Union[int, float]]
                                width=1,  # type: int
                                track_lower=None,  # type: Optional[Union[float, int]]
                                track_upper=None,  # type: Optional[Union[float, int]]
                                fill_margin=0,  # type: Union[int, float]
                                fill_type='',  # type: str
                                unit_mode=False,  # type: bool
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
        tr_idx_list : List[Union[int, float]]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[Union[float, int]]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[Union[float, int]]
            if given, extend track(s) to this upper coordinate.
        fill_margin : Union[int, float]
            minimum margin between wires and fill.
        fill_type : str
            fill connection type.  Either 'VDD' or 'VSS'.  Defaults to 'VSS'.
        unit_mode: bool
            True if track_lower/track_upper/fill_margin is given in resolution units.
        debug : bool
            True to print debug messages.

        Returns
        -------
        track_list : List[WireArray]
            list of created tracks.
        """
        grid = self.grid
        res = grid.resolution

        if not unit_mode:
            fill_margin = int(round(fill_margin / res))
            if track_lower is not None:
                track_lower = int(round(track_lower / res))
            if track_upper is not None:
                track_upper = int(round(track_upper / res))

        # simple error checking
        num_tracks = len(tr_idx_list)
        if num_tracks != len(warr_list_list):
            raise ValueError('wire list length and track index list length mismatch.')
        if num_tracks == 0:
            raise ValueError('No tracks given')

        # compute wire_lower/upper without via extension
        w_lower, w_upper = grid.get_wire_bounds(tr_layer_id, tr_idx_list[0], width=width, unit_mode=True)
        for tr_idx in islice(tr_idx_list, 1, None):
            cur_low, cur_up = grid.get_wire_bounds(tr_layer_id, tr_idx, width=width, unit_mode=True)
            w_lower = min(w_lower, cur_low)
            w_upper = max(w_upper, cur_up)

        # separate wire arrays into bottom/top tracks, and compute wire/track lower/upper coordinates
        bot_warrs = [[] for _ in range(num_tracks)]
        top_warrs = [[] for _ in range(num_tracks)]
        bot_bounds = [None, None]  # type: List[Union[float, int]]
        top_bounds = [None, None]  # type: List[Union[float, int]]
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
                    tr_ext, w_ext = self.grid.get_via_extensions(tr_layer_id, width, cur_width, unit_mode=True)
                    top_warrs[idx].append(warr)
                    cur_bounds = top_bounds
                elif cur_layer_id == tr_layer_id - 1:
                    w_ext, tr_ext = self.grid.get_via_extensions(cur_layer_id, cur_width, width, unit_mode=True)
                    bot_warrs[idx].append(warr)
                    cur_bounds = bot_bounds
                else:
                    raise ValueError('Cannot connect wire on layer %d '
                                     'to track on layer %d' % (cur_layer_id, tr_layer_id))

                # compute wire lower/upper including via extension
                if cur_bounds[0] is None:
                    cur_bounds[0] = w_lower - w_ext
                else:
                    cur_bounds[0] = min(cur_bounds[0], w_lower - w_ext)
                if cur_bounds[1] is None:
                    cur_bounds[1] = w_upper + w_ext
                else:
                    cur_bounds[1] = max(cur_bounds[1], w_upper + w_ext)

                # compute track lower/upper including via extension
                warr_bounds = warr_tid.get_bounds(grid, unit_mode=True)
                if track_lower is None:
                    track_lower = warr_bounds[0] - tr_ext
                else:
                    track_lower = min(track_lower, warr_bounds[0] - tr_ext)
                if track_upper is None:
                    track_upper = warr_bounds[1] + tr_ext
                else:
                    track_upper = max(track_upper, warr_bounds[1] + tr_ext)

        # draw tracks
        track_list = []
        for tr_idx in tr_idx_list:
            track_list.append(self.add_wires(tr_layer_id, tr_idx, track_lower, track_upper, width=width,
                                             fill_margin=fill_margin, fill_type=fill_type, unit_mode=True))

        # connect wires to tracks
        for bwarr_list, twarr_list, tr_idx in zip(bot_warrs, top_warrs, tr_idx_list):
            tr_id = TrackID(tr_layer_id, tr_idx, width=width)
            self.connect_to_tracks(bwarr_list, tr_id, wire_lower=bot_bounds[0], wire_upper=bot_bounds[1],
                                   fill_margin=fill_margin, fill_type=fill_type, unit_mode=True,
                                   min_len_mode=None, debug=debug)
            self.connect_to_tracks(twarr_list, tr_id, wire_lower=top_bounds[0], wire_upper=top_bounds[1],
                                   fill_margin=fill_margin, fill_type=fill_type, unit_mode=True,
                                   min_len_mode=None, debug=debug)

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
        res = grid.resolution

        for bwarr in bot_warr_list:
            bot_intv = int(round(bwarr.lower / res)), int(round(bwarr.upper / res))
            bot_track_idx = bwarr.track_id
            bot_layer_id = bot_track_idx.layer_id
            top_layer_id = bot_layer_id + 1
            bot_width = bot_track_idx.width
            bot_dir = self.grid.get_direction(bot_layer_id)
            for bot_index in bot_track_idx:
                bot_lay_name = self.grid.get_layer_name(bot_layer_id, bot_index)
                btl, btu = grid.get_wire_bounds(bot_layer_id, bot_index, width=bot_width, unit_mode=True)
                for twarr in top_warr_list:
                    top_intv = int(round(twarr.lower / res)), int(round(twarr.upper / res))
                    top_track_idx = twarr.track_id
                    top_width = top_track_idx.width
                    if top_intv[1] >= btu and top_intv[0] <= btl:
                        # top wire cuts bottom wire, possible intersection
                        for top_index in top_track_idx:
                            ttl, ttu = grid.get_wire_bounds(top_layer_id, top_index, width=top_width, unit_mode=True)
                            if bot_intv[1] >= ttu and bot_intv[0] <= ttl:
                                # bottom wire cuts top wire, we have intersection.  Make bbox
                                if bot_dir == 'x':
                                    box = BBox(ttl, btl, ttu, btu, res, unit_mode=True)
                                else:
                                    box = BBox(btl, ttl, btu, ttu, res, unit_mode=True)
                                top_lay_name = self.grid.get_layer_name(top_layer_id, top_index)
                                self.add_via(box, bot_lay_name, top_lay_name, bot_dir)

    def _merge_inst_used_tracks(self):
        template_bot_layer = self.grid.layers[0]
        if not self._added_inst_tracks:
            self._added_inst_tracks = True
            for inst in self._layout.inst_iter():
                top_layer = inst.master.top_layer
                bot_layer = self.grid.get_bot_common_layer(inst.master.grid, top_layer)
                for cidx in range(inst.nx):
                    for ridx in range(inst.ny):
                        # merge tracks on common layers
                        inst_used_tracks = inst.get_used_tracks(bot_layer, top_layer, row=ridx, col=cidx)
                        self._used_tracks.merge(inst_used_tracks)
                        # black out tracks on changed layers
                        if bot_layer > template_bot_layer:
                            inst_box = inst.get_bound_box_of(row=ridx, col=cidx)
                            for lay_id in range(template_bot_layer, bot_layer):
                                if lay_id in self.grid.layers:
                                    self._mark_bbox_used(lay_id, inst_box)

    def _mark_bbox_used(self, layer_id, bbox):
        if self.grid.get_direction(layer_id) == 'x':
            lower, upper = bbox.left, bbox.right
            tl, tu = bbox.bottom_unit, bbox.top_unit
        else:
            lower, upper = bbox.bottom, bbox.top
            tl, tu = bbox.left_unit, bbox.right_unit

        tr_w2 = self.grid.get_track_width(layer_id, 1, unit_mode=True) // 2
        tl -= tr_w2
        tu += tr_w2
        tidx0 = self.grid.coord_to_nearest_track(layer_id, tl, half_track=True,
                                                 mode=1, unit_mode=True)
        tidx1 = self.grid.coord_to_nearest_track(layer_id, tu, half_track=True,
                                                 mode=-1, unit_mode=True)
        htr0 = int(round(tidx0 * 2 + 1))
        htr1 = int(round(tidx1 * 2 + 1))
        warr = WireArray(TrackID(layer_id, tidx0, num=htr1 - htr0 + 1, pitch=0.5), lower, upper)
        self._used_tracks.add_wire_arrays(warr)

    def get_available_tracks(self,  # type: TemplateBase
                             layer_id,  # type: int
                             tr_idx_list,  # type: List[int]
                             lower,  # type: Union[float, int]
                             upper,  # type: Union[float, int]
                             width=1,  # type: int
                             margin=0,  # type: Union[float, int]
                             unit_mode=False,  # type: bool
                             ):
        # type: (...) -> List[int]
        """Returns empty tracks"""
        if not unit_mode:
            res = self.grid.resolution
            lower = int(round(lower / res))
            upper = int(round(upper / res))
            margin = int(round(margin / res))

        self._merge_inst_used_tracks()
        return get_available_tracks(self.grid, layer_id, tr_idx_list, lower, upper,
                                    width, margin, self._used_tracks.get_tracks_info(layer_id))

    def do_power_fill(self,  # type: TemplateBase
                      layer_id,  # type: int
                      vdd_warrs,  # type: Union[WireArray, List[WireArray]]
                      vss_warrs,  # type: Union[WireArray, List[WireArray]]
                      sup_width=1,  # type: int
                      fill_margin=0,  # type: Union[float, int]
                      edge_margin=0,  # type: Union[float, int]
                      unit_mode=False,  # type: bool
                      sup_spacing=-1,  # type: int
                      debug=False,  # type: bool
                      ):
        # type: (...) -> Tuple[List[WireArray], List[WireArray]]
        """Draw power fill on the given layer."""
        if not unit_mode:
            res = self.grid.resolution
            fill_margin = int(round(fill_margin / res))
            edge_margin = int(round(edge_margin / res))

        self._merge_inst_used_tracks()
        top_vdd, top_vss = get_power_fill_tracks(self.grid, self.bound_box, layer_id,
                                                 self._used_tracks.get_tracks_info(layer_id),
                                                 sup_width, fill_margin, edge_margin,
                                                 sup_spacing=sup_spacing, debug=debug)
        for warr in chain(top_vdd, top_vss):
            for layer, box_arr in warr.wire_arr_iter(self.grid):
                self.add_rect(layer, box_arr)
        self.draw_vias_on_intersections(vdd_warrs, top_vdd)
        self.draw_vias_on_intersections(vss_warrs, top_vss)

        return top_vdd, top_vss
