# -*- coding: utf-8 -*-

"""This package contains LVS/RCX related verification methods.
"""

import importlib

from .base import Checker

__all__ = ['make_checker', 'Checker']


def make_checker(checker_cls, tmp_dir, **kwargs):
    """Returns a checker object.

    Parameters
    -----------
    checker_cls : str
        the Checker class absolute path name.
    tmp_dir : string
        directory to save temporary files in.
    kwargs : dict
        keyword arguments needed to create a Checker object.
    """
    sections = checker_cls.split('.')

    module_str = '.'.join(sections[:-1])
    class_str = sections[-1]
    module = importlib.import_module(module_str)
    return getattr(module, class_str)(tmp_dir, **kwargs)
