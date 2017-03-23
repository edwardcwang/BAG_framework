# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""This module defines various interpolation classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.interpolation as imag_interp

from ..math.dfun import DiffFunction

__author__ = 'erichang'
__all__ = ['interpolate_grid']


def interpolate_grid(scale_list, values, method='spline',
                     extrapolate=False, delta=1e-4, num_extrapolate=2):
    """Interpolates multidimensional data on a regular grid.

    returns an Interpolator for the given dataset.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing).
    values : numpy.array
        The output data in N dimensions.  The length in each dimension must
        be at least 2.
    method : str
        The interpolation method.  Either 'linear', or 'spline'.
        Defaults to 'spline'.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    delta : float
        the finite difference step size.  Finite difference is only used for
        linear interpolation and spline interpolation on 3D data or greater.
        Defaults to 1e-4 of the grid spacing.
    num_extrapolate: int
        If spline interpolation is selected on 3D data or greater, we linearly
        extrapolate the given data by this many points to fix behavior near
        input boundaries.

    Returns
    -------
    fun : bag.math.dfun.DiffFunction
        the interpolator function.
    """
    ndim = len(values.shape)
    if ndim == 1:
        return Interpolator1D(scale_list, values, method=method, extrapolate=extrapolate)
    elif method == 'linear':
        return LinearInterpolator(scale_list, values, delta=delta, extrapolate=extrapolate)
    elif method == 'spline':
        if ndim == 2:
            return Spline2D(scale_list, values, extrapolate=extrapolate)
        else:
            return MapCoordinateSpline(scale_list, values, delta=delta, extrapolate=extrapolate,
                                       num_extrapolate=num_extrapolate)
    else:
        raise ValueError('Unsupported interpolation method: %s' % method)


class LinearInterpolator(DiffFunction):
    """A linear interpolator on a regular grid for 2 or more dimensions.

    This class is backed by scipy.interpolate.RegularGridInterpolator.
    Derivatives are calculated using finite difference.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data in N dimensions.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    delta : float
        the finite difference step size.  Defaults to 1e-4 (relative to a spacing of 1).
    """

    def __init__(self, scale_list, values, extrapolate=False, delta=1e-4):
        ndim = len(values.shape)
        # error checking
        if ndim == 1:
            raise ValueError('This class only works for dimension >= 2.')
        elif ndim != len(scale_list):
            raise ValueError('input and output dimension mismatch.')

        # compute points and deltas
        points = []
        delta_list = []
        for idx in range(ndim):
            num_pts = values.shape[idx]  # type: int
            if num_pts < 2:
                raise ValueError('Every dimension must have at least 2 points.')
            offset, scale = scale_list[idx]
            points.append(np.linspace(offset, (num_pts - 1) * scale + offset, num_pts))
            delta_list.append(scale * delta)

        DiffFunction.__init__(self, ndim, delta_list=delta_list)
        # noinspection PyTypeChecker
        self.fun = interp.RegularGridInterpolator(points, values, method='linear',
                                                  bounds_error=not extrapolate,
                                                  fill_value=None)

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        return self.fun(xi)


class Interpolator1D(DiffFunction):
    """An interpolator on a regular grid for 1 dimensional data.

    This class is backed by scipy.interpolate.InterpolatedUnivariateSpline.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.  Must be 1 dimension.
    method : str
        extrapolation method.  Either 'linear' or 'spline'.  Defaults to spline.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    """

    def __init__(self, scale_list, values, method='spline', extrapolate=False):
        # error checking
        if len(values.shape) != 1:
            raise ValueError('This class only works for 1D data.')
        elif len(scale_list) != 1:
            raise ValueError('input and output dimension mismatch.')

        if method == 'linear':
            k = 1
        elif method == 'spline':
            k = 3
        else:
            raise ValueError('Unsuppoorted interpolation method: %s' % method)

        DiffFunction.__init__(self, 1, delta_list=None)
        offset, scale = scale_list[0]
        num_pts = values.shape[0]
        points = np.linspace(offset, (num_pts - 1) * scale + offset, num_pts)
        ext = 0 if extrapolate else 2
        self.fun = interp.InterpolatedUnivariateSpline(points, values, k=k, ext=ext)

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        return self.fun(xi)

    def deriv(self, xi, idx):
        """Calculate the derivative of the spline along the given index.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)
        idx : int
            The index to calculate the derivative on.

        Returns
        -------
        val : numpy.array
            The derivatives at the given coordinates.
        """
        if idx != 0:
            raise ValueError('Invalid derivative index: %d' % idx)

        return self.fun(xi, 1)


class Spline2D(DiffFunction):
    """A spline interpolator on a regular grid for 2D data.

    This class is backed by scipy.interpolate.RectBivariateSpline.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.  Must be 2D.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    """

    def __init__(self, scale_list, values, extrapolate=False):
        # error checking
        if len(values.shape) != 1:
            raise ValueError('This class only works for 2D data.')
        elif len(scale_list) != 1:
            raise ValueError('input and output dimension mismatch.')

        DiffFunction.__init__(self, 2, delta_list=None)

        nx, ny = values.shape
        offset, scale = scale_list[0]
        x = np.linspace(offset, (nx - 1) * scale + offset, nx)
        offset, scale = scale_list[1]
        y = np.linspace(offset, (ny - 1) * scale + offset, ny)

        self._min = x[0], y[0]
        self._max = x[-1], y[-1]
        self.fun = interp.RectBivariateSpline(x, y, values)
        self._extrapolate = extrapolate

    def _get_xy(self, xi):
        """Get X and Y array from given coordinates."""
        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != 2:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension 2" % (xi.shape[-1]))

        # check input within bounds.
        x = xi[..., 0]
        y = xi[..., 1]
        if not self._extrapolate and not np.all((self._min[0] <= x) & (x <= self._max[0]) &
                                                (self._min[1] <= y) & (y <= self._max[1])):
            raise ValueError('some inputs are out of bounds.')

        return x, y

    def __call__(self, xi):
        """Interpolate at the given coordinates.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        x, y = self._get_xy(xi)
        return self.fun(x, y, grid=False)

    def deriv(self, xi, idx):
        """Calculate the derivative of the spline along the given index.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)
        idx : int
            The index to calculate the derivative on.

        Returns
        -------
        val : numpy.array
            The derivatives at the given coordinates.
        """
        if idx < 0 or idx > 1:
            raise ValueError('Invalid derivative index: %d' % idx)

        x, y = self._get_xy(xi)
        if idx == 0:
            return self.fun(x, y, dx=1, grid=False)
        else:
            return self.fun(x, y, dy=1, grid=False)


class MapCoordinateSpline(DiffFunction):
    """A spline interpolator on a regular grid for multidimensional data.

    The spline interpolation is done using map_coordinate method in the
    scipy.ndimage.interpolation package.  The derivative is done using
    finite difference.

    if extrapolate is True, we use linear interpolation for values outside of
    bounds.

    Note: By default, map_coordinate uses the nearest value for all points
    outside the boundary.  This will cause undesired interpolation
    behavior near boundary points.  To solve this, we linearly
    extrapolates the given data for a fixed number of points.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.
    extrapolate : bool
        True to linearly extrapolate outside of bounds.
    num_extrapolate : int
        number of points to extrapolate in each dimension in each direction.
    delta : float
        the finite difference step size.  Defaults to 1e-4 (relative to a spacing of 1).
    """

    def __init__(self, scale_list, values, extrapolate=False, num_extrapolate=2,
                 delta=1e-4):
        shape = values.shape
        ndim = len(shape)

        # error checking
        if ndim < 3:
            raise ValueError('Data must have 3 or more dimensions.')
        elif ndim != len(scale_list):
            raise ValueError('input and output dimension mismatch.')

        self._scale_list = scale_list
        self._max = values.shape
        self._ext = num_extrapolate

        # linearly extrapolate given values
        self._extfun = LinearInterpolator([(0, 1)] * ndim, values, extrapolate=True)
        self._extrapolate = extrapolate
        swp_values = []
        ext_xi_shape = []
        delta_list = []
        for (offset, scale), n in zip(scale_list, shape):
            swp_values.append(np.arange(-num_extrapolate, n + num_extrapolate))
            ext_xi_shape.append(n + 2 * num_extrapolate)
            delta_list.append(scale * delta)

        ext_xi_shape.append(ndim)
        xi = np.empty(ext_xi_shape)
        xmat_list = np.meshgrid(*swp_values, indexing='ij', copy=False)
        for idx, xmat in enumerate(xmat_list):
            xi[..., idx] = xmat

        values_ext = self._extfun(xi)
        self._filt_values = imag_interp.spline_filter(values_ext)
        DiffFunction.__init__(self, ndim, delta_list=delta_list)

    def _normalize_inputs(self, xi):
        """Normalize the inputs."""
        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != self.ndim:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension %d" % (xi.shape[-1], self.ndim))

        xi = np.atleast_2d(xi.copy())
        need_extrapolate = False
        for idx, (offset, scale) in enumerate(self._scale_list):
            xi[..., idx] -= offset
            xi[..., idx] /= scale
            max_val = np.max(xi[..., idx])
            if max_val > self._max[idx]:
                if not self._extrapolate:
                    raise ValueError('Some inputs on dimension %d out of bounds.  Max value = %.4g' % (idx, max_val))
                else:
                    need_extrapolate = True
        min_val = np.min(xi)
        if min_val < 0.0:
            if not self._extrapolate:
                raise ValueError('Some inputs are negative.  Min Value = %.4g' % min_val)
            else:
                need_extrapolate = True
        # take extension input account.
        xi += self._ext

        return xi, need_extrapolate

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        xi, extrapolate = self._normalize_inputs(xi)
        if extrapolate:
            return self._extfun(xi)
        else:
            return imag_interp.map_coordinates(self._filt_values, xi.T, mode='nearest', prefilter=False).T
