# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any

import abc

from .tech import LaygoTech
from ..analog_mos.finfet import MOSTechFinfetBase
from ..analog_mos.finfet import ExtInfo, RowInfo, AdjRowInfo, EdgeInfo, FillInfo


if TYPE_CHECKING:
    from bag.layout.tech import TechInfoConfig


class LaygoTechFinfetBase(MOSTechFinfetBase, LaygoTech, metaclass=abc.ABCMeta):
    """Base class for implementations of LaygoTech in Finfet technologies.

    This class for now handles all DRC rules and drawings related to PO, OD, CPO,
    and MD. The rest needs to be implemented by subclasses.

    Parameters
    ----------
    config : Dict[str, Any]
        the technology configuration dictionary.
    tech_info : TechInfo
        the TechInfo object.
    mos_entry_name : str
        name of the entry that contains technology parameters for transistors in
        the given configuration dictionary.
    """

    def __init__(self, config, tech_info, mos_entry_name='mos'):
        # type: (Dict[str, Any], TechInfoConfig, str) -> None
        MOSTechFinfetBase.__init__(self, config, tech_info, mos_entry_name=mos_entry_name)
