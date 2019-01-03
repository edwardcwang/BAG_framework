# -*- coding: utf-8 -*-

"""This module handles file related IO.
"""

from typing import TextIO, Any, Iterable

import os
import string
import tempfile
import time
import pkg_resources
import codecs

from ruamel.yaml import YAML

from .common import bag_encoding, bag_codec_error

yaml = YAML(typ='unsafe')


def open_file(fname: str, mode: str) -> TextIO:
    """Opens a file with the correct encoding interface.

    Use this method if you need to have a file handle.

    Parameters
    ----------
    fname : str
        the file name.
    mode : str
        the mode, either 'r', 'w', or 'a'.

    Returns
    -------
    file_obj : TextIO
        a file objects that reads/writes string with the BAG system encoding.
    """
    if mode != 'r' and mode != 'w' and mode != 'a':
        raise ValueError("Only supports 'r', 'w', or 'a' mode.")
    return open(fname, mode, encoding=bag_encoding, errors=bag_codec_error)


def read_file(fname: str) -> str:
    """Read the given file and return content as string.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : str
        the content as a unicode string.
    """
    with open_file(fname, 'r') as f:
        content = f.read()
    return content


def readlines_iter(fname: str) -> Iterable[str]:
    """Iterate over lines in a file.

    Parameters
    ----------
    fname : str
        the file name.

    Yields
    ------
    line : str
        a line in the file.
    """
    with open_file(fname, 'r') as f:
        for line in f:
            yield line


def read_yaml(fname: str) -> Any:
    """Read the given file using YAML.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open_file(fname, 'r') as f:
        content = yaml.load(f)

    return content


def read_yaml_env(fname: str) -> Any:
    """Parse YAML file with environment variable substitution.

    Parameters
    ----------
    fname : str
        yaml file name.

    Returns
    -------
    table : Any
        the object returned by YAML.
    """
    content = read_file(fname)
    # substitute environment variables
    content = string.Template(content).substitute(os.environ)
    return yaml.load(content)


def read_resource(package: str, fname: str) -> str:
    """Read the given resource file and return content as string.

    Parameters
    ----------
    package : str
        the package name.
    fname : str
        the resource file name.

    Returns
    -------
    content : str
        the content as a unicode string.
    """
    raw_content = pkg_resources.resource_string(package, fname)
    return raw_content.decode(encoding=bag_encoding, errors=bag_codec_error)


def write_file(fname: str, content: str, append: bool = False, mkdir: bool = True) -> None:
    """Writes the given content to file.

    Parameters
    ----------
    fname : str
        the file name.
    content : str
        the unicode string to write to file.
    append : bool
        True to append instead of overwrite.
    mkdir : bool
        If True, will create parent directories if they don't exist.
    """
    if mkdir:
        fname = os.path.abspath(fname)
        dname = os.path.dirname(fname)
        os.makedirs(dname, exist_ok=True)

    mode = 'a' if append else 'w'
    with open_file(fname, mode) as f:
        f.write(content)


def write_yaml(fname: str, obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using YAML format.

    Parameters
    ----------
    fname : str
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    if mkdir:
        fname = os.path.abspath(fname)
        dname = os.path.dirname(fname)
        os.makedirs(dname, exist_ok=True)

    with open_file(fname, 'w') as f:
        yaml.dump(obj, f)


def make_temp_dir(prefix: str, parent_dir: str = '') -> str:
    """Create a new temporary directory.

    Parameters
    ----------
    prefix : str
        the directory prefix.
    parent_dir : str
        the parent directory.

    Returns
    -------
    dir_name : str
        the temporary directory name.
    """
    prefix += time.strftime("_%Y%m%d_%H%M%S")
    parent_dir = parent_dir or tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=parent_dir)


def open_temp(**kwargs: Any) -> TextIO:
    """Opens a new temporary file for writing with unicode interface.

    Parameters
    ----------
    **kwargs : Any
        the tempfile keyword arguments.  See documentation for
        :func:`tempfile.NamedTemporaryFile`.

    Returns
    -------
    file : TextIO
        the opened file that accepts unicode input.
    """
    timestr = time.strftime("_%Y%m%d_%H%M%S")
    if 'prefix' in kwargs:
        kwargs['prefix'] += timestr
    else:
        kwargs['prefix'] = timestr
    temp = tempfile.NamedTemporaryFile(**kwargs)
    return codecs.getwriter(bag_encoding)(temp, errors=bag_codec_error)
