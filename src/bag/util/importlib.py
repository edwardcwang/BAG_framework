# -*- coding: utf-8 -*-

"""This module defines various import helper methods.
"""

from typing import Type

import importlib


def import_class(class_str: str) -> Type:
    """Given a Python class string, import and return that Python class.

    Parameters
    ----------
    class_str : str
        a Python class string.

    Returns
    -------
    py_class : Type
        a Python class.
    """
    sections = class_str.split('.')

    module_str = '.'.join(sections[:-1])
    class_str = sections[-1]
    modul = importlib.import_module(module_str)
    return getattr(modul, class_str)
