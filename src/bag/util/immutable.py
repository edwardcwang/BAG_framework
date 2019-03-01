# -*- coding: utf-8 -*-

"""This module defines various immutable and hashable data types.
"""

from __future__ import annotations

from typing import TypeVar, Any, Generic, Dict, Iterable, Tuple, Union, Optional, Mapping

import sys
import abc
import bisect
import collections


class Immutable(abc.ABC):
    """This is the abstract base class of all immutable data types."""

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool: ...

    @abc.abstractmethod
    def __hash__(self) -> int: ...


T = TypeVar('T')
U = TypeVar('U')
ImmutableType = Union[Immutable, Tuple[Immutable, ...]]


def combine_hash(a: int, b: int) -> int:
    """Combine the two given hash values.

    Parameter
    ---------
    a : int
        the first hash value.
    b : int
        the second hash value.

    Returns
    -------
    hash : int
        the combined hash value.
    """
    # algorithm taken from boost::hash_combine
    return sys.maxsize & (a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2)))


class ImmutableSortedDict(Immutable, collections.Mapping, Generic[T, U]):
    """An immutable dictionary with sorted keys."""

    def __init__(self,
                 table: Optional[Mapping[T, Any]] = None) -> None:
        if table is not None:
            if isinstance(table, ImmutableSortedDict):
                self._keys = table._keys
                self._vals = table._vals
                self._hash = table._hash
            else:
                self._keys = tuple(sorted(table.keys()))
                self._vals = tuple((to_immutable(table[k]) for k in self._keys))
                self._hash = combine_hash(hash(self._keys), hash(self._vals))
        else:
            self._keys = tuple()
            self._vals = tuple()
            self._hash = combine_hash(hash(self._keys), hash(self._vals))

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, ImmutableSortedDict) and
                self._hash == other._hash and
                self._keys == other._keys and
                self._vals == other._vals)

    def __hash__(self) -> int:
        return self._hash

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterable[T]:
        return iter(self._keys)

    def __getitem__(self, item: T) -> U:
        idx = bisect.bisect_left(self._keys, item)
        if idx == len(self._keys) or self._keys[idx] != item:
            raise KeyError('Key not found: {}'.format(item))
        return self._vals[idx]

    def get(self, item: T, default: Optional[U] = None) -> Optional[U]:
        idx = bisect.bisect_left(self._keys, item)
        if idx == len(self._keys) or self._keys[idx] != item:
            return default
        return self._vals[idx]

    def keys(self) -> Iterable[T]:
        return iter(self._keys)

    def values(self) -> Iterable[U]:
        return iter(self._vals)

    def items(self) -> Iterable[Tuple[T, U]]:
        return zip(self._keys, self._vals)

    def copy(self, append: Optional[Dict[T, Any]] = None) -> ImmutableSortedDict[T, U]:
        if append is None:
            return self.__class__(self)
        else:
            tmp = dict(zip(self._keys, self._vals))
            tmp.update(append)
            return self.__class__(tmp)


def to_immutable(obj: Any) -> ImmutableType:
    """Convert the given Python object into an immutable type."""
    if (obj is None or isinstance(obj, str) or isinstance(obj, int) or
            isinstance(obj, float) or isinstance(obj, complex) or
            isinstance(obj, Immutable)):
        return obj
    if isinstance(obj, tuple) or isinstance(obj, list):
        return tuple((to_immutable(v) for v in obj))
    if isinstance(obj, set):
        return tuple((to_immutable(v) for v in sorted(obj)))
    if isinstance(obj, dict):
        return ImmutableSortedDict(obj)

    raise ValueError('Cannot convert the following object to immutable type: {}'.format(obj))
