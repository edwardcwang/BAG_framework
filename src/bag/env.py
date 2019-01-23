# -*- coding: utf-8 -*-

"""This module defines various methods to query information about the design environment.
"""

from typing import Tuple, Dict, Any, Optional

import os

from .io.file import read_file, read_yaml_env
from .layout.tech import TechInfo
from .layout.routing import RoutingGrid
from .util.importlib import import_class


def get_bag_work_dir() -> str:
    """Returns the BAG working directory."""
    work_dir = os.environ.get('BAG_WORK_DIR', '')
    if not work_dir:
        raise ValueError('Environment variable BAG_WORK_DIR not defined.')
    if not os.path.isdir(work_dir):
        raise ValueError('$BAG_WORK_DIR = "{}" is not a directory'.format(work_dir))

    return work_dir


def get_tech_dir() -> str:
    """Returns the technology directory."""
    tech_dir = os.environ.get('BAG_TECH_CONFIG_DIR', '')
    if not tech_dir:
        raise ValueError('Environment variable BAG_TECH_CONFIG_DIR not defined.')
    if not os.path.isdir(tech_dir):
        raise ValueError('BAG_TECH_CONFIG_DIR = "{}" is not a directory'.format(tech_dir))

    return tech_dir


def get_bag_config() -> Dict[str, Any]:
    """Returns the BAG configuration dictioanry."""
    bag_config_path = os.environ.get('BAG_CONFIG_PATH', '')
    if not bag_config_path:
        raise ValueError('Environment variable BAG_CONFIG_PATH not defined.')

    return read_yaml_env(bag_config_path)


def get_tech_params(bag_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Returns the technology parameters dictioanry.

    Parameters
    ----------
    bag_config : Optional[Dict[str, Any]]
        the BAG configuration dictionary.  If None, will try to read it from file.

    Returns
    -------
    tech_params : Dict[str, Any]
        the technology configuration dictionary.
    """
    if bag_config is None:
        bag_config = get_bag_config()

    fname = bag_config['tech_config_path']
    ans = read_yaml_env(bag_config['tech_config_path'])
    ans['tech_config_fname'] = fname
    return ans


def create_tech_info(bag_config: Optional[Dict[str, Any]] = None) -> TechInfo:
    """Create TechInfo object."""
    tech_params = get_tech_params(bag_config=bag_config)

    if 'class' in tech_params:
        tech_cls = import_class(tech_params['class'])
        tech_info = tech_cls(tech_params)
    else:
        # just make a default tech_info object as place holder.
        print('*WARNING*: No TechInfo class defined.  Using a dummy version.')
        tech_info = TechInfo(tech_params, {}, '')

    return tech_info


def create_routing_grid(tech_info: Optional[TechInfo] = None,
                        bag_config: Optional[Dict[str, Any]] = None) -> RoutingGrid:
    """Create RoutingGrid object."""
    if tech_info is None:
        tech_info = create_tech_info(bag_config=bag_config)
    return RoutingGrid(tech_info, tech_info.tech_params['tech_config_fname'])


def get_port_number(bag_config: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """Read the port number from the port file..

    Parameters
    ----------
    bag_config : Optional[Dict[str, Any]]
        the BAG configuration dictionary.  If None, will try to read it from file.

    Returns
    -------
    port : int
        the port number.  Negative on failure.
    msg : str
        Empty string on success, the error message on failure.
    """
    if bag_config is None:
        bag_config = get_bag_config()

    port_file = os.path.join(get_bag_work_dir(), bag_config['socket']['port_file'])
    try:
        port = int(read_file(port_file))
    except ValueError as err:
        return -1, str(err)
    except FileNotFoundError as err:
        return -1, str(err)

    return port, ''


def get_netlist_setup_file() -> str:
    """Returns the netlist setup file path."""
    ans = os.path.abspath(os.path.join(get_tech_dir(), 'netlist_setup', 'netlist_setup.yaml'))
    if not os.path.isfile(ans):
        raise ValueError(ans + ' is not a file.')
    return ans


def get_gds_layer_map() -> str:
    """Returns the GDs layer map file."""
    ans = os.path.abspath(os.path.join(get_tech_dir(), 'gds_setup', 'gds.layermap'))
    if not os.path.isfile(ans):
        raise ValueError(ans + ' is not a file.')
    return ans


def get_gds_object_map() -> str:
    """Returns the GDS object map file."""
    ans = os.path.abspath(os.path.join(get_tech_dir(), 'gds_setup', 'gds.objectmap'))
    if not os.path.isfile(ans):
        raise ValueError(ans + ' is not a file.')
    return ans
