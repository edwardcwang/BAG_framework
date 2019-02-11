# -*- coding: utf-8 -*-

"""This module handles string related IO.
"""

from io import StringIO

from ruamel.yaml import YAML

yaml = YAML(typ='unsafe')


def read_yaml_str(content: str) -> object:
    """Parse the given yaml str and return the python object."""
    return yaml.load(content)


def to_yaml_str(obj: object) -> str:
    """Converts the given python object into a YAML string."""
    stream = StringIO()
    yaml.dump(obj, stream)
    return stream.getvalue()
