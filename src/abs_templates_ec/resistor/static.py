# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Tuple, Any, List, Optional, Union

from bag.layout.util import BBox
from bag.layout.template import TemplateBase
from bag.layout.routing import RoutingGrid

from .core import ResTech

if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig


class ResTechStatic(ResTech):
    """An implementation of ResTech that simply uses a static layout.

    Parameters
    ----------
    config : Dict[str, Any]
        the technology configuration dictionary.
    tech_info : TechInfo
        the TechInfo object.
    """

    def __init__(self, config, tech_info):
        # type: (Dict[str, Any], TechInfoConfig) -> None
        ResTech.__init__(self, config, tech_info)

    def get_min_res_core_size(self, l, w, res_type, sub_type, threshold, options):
        # type: (int, int, str, str, str, Dict[str, Any]) -> Tuple[int, int]
        return 1, 1

    def get_core_info(self,
                      grid,  # type: RoutingGrid
                      width,  # type: int
                      height,  # type: int
                      l,  # type: int
                      w,  # type: int
                      res_type,  # type: str
                      sub_type,  # type: str
                      threshold,  # type: str
                      track_widths,  # type: List[int]
                      track_spaces,  # type: List[Union[float, int]]
                      options,  # type: Dict[str, Any]
                      ):
        # type: (...) -> Optional[Dict[str, Any]]
        return None

    def get_lr_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         wedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        return None

    def get_tb_edge_info(self,
                         grid,  # type: RoutingGrid
                         core_info,  # type: Dict[str, Any]
                         hedge,  # type: int
                         l,  # type: int
                         w,  # type: int
                         res_type,  # type: str
                         sub_type,  # type: str
                         threshold,  # type: str
                         track_widths,  # type: List[int]
                         track_spaces,  # type: List[Union[float, int]]
                         options,  # type: Dict[str, Any]
                         ):
        # type: (...) -> Optional[Dict[str, Any]]
        return None

    def draw_res_core(self, template, layout_info):
        # type: (TemplateBase, Dict[str, Any]) -> None
        template_lib = self.res_config['template_lib']
        core_cell = self.res_config['core_cell']
        btr, ttr = self.res_config['port_tracks']
        xl, xr = self.res_config['port_coord']
        wcore, hcore = self.res_config['core_size']

        template.add_instance_primitive(template_lib, core_cell, (0, 0))
        port_layer = self.get_bot_layer()

        warr = template.add_wires(port_layer, btr, xl, xr, unit_mode=True)
        template.add_pin('bot', warr, show=False)
        warr = template.add_wires(port_layer, ttr, xl, xr, unit_mode=True)
        template.add_pin('top', warr, show=False)

        res = template.grid.resolution
        template.prim_bound_box = BBox(0, 0, wcore, hcore, res, unit_mode=True)
        template.array_box = template.prim_bound_box
        template.prim_top_layer = port_layer

    def draw_res_boundary(self, template, boundary_type, layout_info, end_mode):
        # type: (TemplateBase, str, Dict[str, Any]) -> None
        wcore, hcore = self.res_config['core_size']
        wedge, hedge = self.res_config['edge_size']
        template_lib = self.res_config['template_lib']
        core_cell = self.res_config['%s_cell' % boundary_type]

        template.add_instance_primitive(template_lib, core_cell, (0, 0))

        if boundary_type == 'corner':
            wbox, hbox = wedge, hedge
        elif boundary_type == 'lr':
            wbox, hbox = wedge, hcore
        else:
            wbox, hbox = wcore, hedge

        res = template.grid.resolution
        template.prim_bound_box = BBox(0, 0, wbox, hbox, res, unit_mode=True)
        template.array_box = template.prim_bound_box
        template.prim_top_layer = self.get_bot_layer()

    def get_res_info(self,  # type: ResTechStatic
                     grid,  # type: RoutingGrid
                     l,  # type: int
                     w,  # type: int
                     res_type,  # type: str
                     sub_type,  # type: str
                     threshold,  # type: str
                     min_tracks,  # type: Tuple[int, ...]
                     em_specs,  # type: Dict[str, Any]
                     ext_dir,  # type: Optional[str]
                     max_blk_ext=100,  # type: int
                     connect_up=False,  # type: bool
                     options=None,  # type: Optional[Dict[str, Any]]
                     ):
        # type: (...) -> Dict[str, Any]
        wcore, hcore = self.res_config['core_size']
        wedge, hedge = self.res_config['edge_size']
        well_xl = self.res_config.get('well_xl', 0)

        track_widths, track_spaces, _, _ = self.get_core_track_info(grid, min_tracks, em_specs,
                                                                    connect_up=connect_up)

        core_info = {}
        edge_lr_info = {}
        edge_tb_info = {}

        bot_layer = self.get_bot_layer()
        num_tracks = []
        num_corner_tracks = []
        for lay in range(bot_layer, bot_layer + len(min_tracks)):
            if grid.get_direction(lay) == 'y':
                dim = wcore
                dim_corner = wedge
            else:
                dim = hcore
                dim_corner = hedge

            pitch = grid.get_track_pitch(lay, unit_mode=True)
            if dim % pitch == 0:
                num_tracks.append(dim // pitch)
            else:
                num_tracks.append(dim / pitch)
            if dim_corner % pitch == 0:
                num_corner_tracks.append(dim_corner // pitch)
            else:
                num_corner_tracks.append(dim_corner / pitch)

        return dict(
            l=l,
            w=w,
            res_type=res_type,
            sub_type=sub_type,
            threshold=threshold,
            options=options,
            w_core=wcore,
            h_core=hcore,
            w_edge=wedge,
            h_edge=hedge,
            track_widths=track_widths,
            track_spaces=track_spaces,
            num_tracks=num_tracks,
            num_corner_tracks=num_corner_tracks,
            core_info=core_info,
            edge_lr_info=edge_lr_info,
            edge_tb_info=edge_tb_info,
            well_xl=well_xl,
        )
