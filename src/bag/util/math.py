# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from math import trunc, ceil, floor
from numbers import Integral, Real


class HalfInt(Integral):
    """A class that represents a half integer."""

    def __init__(self, dbl_val: Any) -> None:
        if isinstance(dbl_val, Integral):
            self._val = int(dbl_val)
        else:
            raise ValueError('HafInt internal value must be an integer.')

    @classmethod
    def convert(cls, val: Any) -> HalfInt:
        if isinstance(val, HalfInt):
            return val
        elif isinstance(val, Integral):
            return HalfInt(2 * int(val))
        elif isinstance(val, Real):
            tmp = float(2 * val)
            if tmp.is_integer():
                return HalfInt(int(tmp))
        raise ValueError('Cannot convert {} type {} to HalfInt.'.format(val, type(val)))

    @property
    def value(self) -> float:
        q, r = divmod(self._val, 2)
        return q if r == 0 else q + 0.5

    @property
    def is_integer(self) -> bool:
        return self._val % 2 == 0

    def div2(self, round_up: bool = False) -> HalfInt:
        q, r = divmod(self._val, 2)
        if r or not round_up:
            self._val = q
        else:
            self._val = q + 1
        return self

    def to_string(self) -> str:
        q, r = divmod(self._val, 2)
        if r == 0:
            return '{:d}'.format(q)
        return '{:d}.5'.format(q)

    def up(self) -> None:
        self._val += 1

    def down(self) -> None:
        self._val -= 1

    def increment(self, other: Any) -> None:
        self._val += HalfInt.convert(other)._val

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'HalfInt({})'.format(self._val / 2)

    def __hash__(self):
        return hash(self._val)

    def __eq__(self, other):
        if isinstance(other, HalfInt):
            return self._val == other._val
        return self._val == 2 * other

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        if isinstance(other, HalfInt):
            return self._val <= other._val
        return self._val <= 2 * other

    def __lt__(self, other):
        if isinstance(other, HalfInt):
            return self._val < other._val
        return self._val < 2 * other

    def __ge__(self, other):
        return not (self < other)

    def __gt__(self, other):
        return not (self <= other)

    def __add__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val + other._val)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val * other._val, 2)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __truediv__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(2 * self._val, other._val)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __floordiv__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(2 * (self._val // other._val))

    def __mod__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val % other._val)

    def __divmod__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val, other._val)
        return HalfInt(2 * q), HalfInt(r)

    def __pow__(self, other, modulus=None):
        other = HalfInt.convert(other)
        if self.is_integer and other.is_integer:
            return HalfInt(2 * (self._val // 2)**(other._val // 2))
        raise ValueError('result is not a HalfInt.')

    def __lshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __and__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __xor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __or__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return HalfInt.convert(other) / self

    def __rfloordiv__(self, other):
        return HalfInt.convert(other) // self

    def __rmod__(self, other):
        return HalfInt.convert(other) % self

    def __rdivmod__(self, other):
        return HalfInt.convert(other).__divmod__(self)

    def __rpow__(self, other):
        return HalfInt.convert(other)**self

    def __rlshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rrshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __rand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __rxor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ror__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    def __imod__(self, other):
        return self % other

    def __ipow__(self, other):
        return self ** other

    def __ilshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __irshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __iand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __ixor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ior__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __neg__(self):
        return HalfInt(-self._val)

    def __pos__(self):
        return HalfInt(self._val)

    def __abs__(self):
        return HalfInt(abs(self._val))

    def __invert__(self):
        return -self

    def __complex__(self):
        raise TypeError('Cannot cast to complex')

    def __int__(self):
        if self._val % 2 == 1:
            raise ValueError('Not an integer.')
        return self._val // 2

    def __float__(self):
        return self._val / 2

    def __index__(self):
        return int(self)

    def __round__(self, ndigits=0):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(round(self._val / 2) * 2)

    def __trunc__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(trunc(self._val / 2) * 2)

    def __floor__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(floor(self._val / 2) * 2)

    def __ceil__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(ceil(self._val / 2) * 2)
