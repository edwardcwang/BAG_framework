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


"""This module defines placement related classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import abc
import copy

import numpy as np
from .util import BBox, PortSpec, InstPin
from future.utils import with_metaclass


_transform_dict = {'R0': np.array([[1, 0], [0, 1]]),
                   'MX': np.array([[1, 0], [0, -1]]),
                   'MY': np.array([[-1, 0], [0, 1]]),
                   'R180': np.array([[-1, 0], [0, -1]]),
                   }


def _get_orientation(matrix):
    """Convert transformation matrix to orientation string.

    Parameters
    ----------
    matrix : np.multiarray.ndarray
        the transformation matrix.

    Returns
    -------
    orient : str
        the orientation string.
    """
    for orient, mat in _transform_dict.items():
        if np.allclose(mat, matrix):
            return orient
    raise Exception('Unsupported matrix %s' % matrix)


def _get_transform_matrix(orient):
    """Get the transformation matrix corresponding to the given orientation.

    Parameters
    ----------
    orient : string
        the orientation string.  Valid values are 'R0', 'MX', 'MY', or 'R180'.

    Returns
    -------
    mat : np.multiarray.ndarray
        the transformation matrix
    """
    if orient not in _transform_dict:
        raise Exception('Unsupported orientation %s' % orient)
    return _transform_dict[orient]


def _apply_transform(loc, orient, trans_loc, trans_orient):
    """Apply the given transformation to the given location and orientation.

    Translation is applied after rotation/mirroring.

    Parameters
    ----------
    loc : tuple[float]
        the location of the instance.
    orient : str
        the orientation of the instance.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
    trans_loc : tuple[float]
        the translation to apply, in (dx, dy) form.
    trans_orient : str
        the rotation/mirroring to apply.

    Returns
    -------
    new_loc : tuple[float]
        the transformed location.
    new_orient : str
        the transformed orientation.

    """
    m1 = _get_transform_matrix(orient)
    m2 = _get_transform_matrix(trans_orient)
    d1 = np.array(loc)
    d2 = np.array(trans_loc)

    df = np.dot(m2, d1) + d2
    mf = np.dot(m2, m1)

    return (df[0], df[1]), _get_orientation(mf)


class Array2D(object):
    """A dynamic 2D object array.

    This class is implemented using list of lists.  Unfilled indices contains None.
    """

    def __init__(self):
        self.arr = []

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise Exception('Index must be tuple')
        return self.arr[index[0]][index[1]]

    def __setitem__(self, index, value):
        if not isinstance(index, tuple):
            raise Exception('Index must be tuple')

        col, row = index
        if col < 0 or row < 0:
            raise Exception('Negative column/row index.')

        # initialize array if empty
        if len(self.arr) == 0:
            for cidx in range(col + 1):
                self.arr.append([None] * (row + 1))

        # expand array if needed
        ncol = len(self.arr)
        nrow = len(self.arr[0])
        if row >= nrow:
            diff = row + 1 - nrow
            for row_arr in self.arr:
                row_arr.extend([None] * diff)
            nrow = row + 1
        if col >= ncol:
            for cidx in range(col + 1 - ncol):
                self.arr.append([None] * nrow)

        # set item
        self.arr[col][row] = value

    def items_iter(self, row=None, col=None):
        """Returns an iterator over items.

        Parameters
        ----------
        row : int or None
            if not None, iterate over items in the given row.
        col : int or None
            if not None, iterator over items in the given column.

        Yields
        ------
        cidx : int
            the column index.
        ridx : int
            the row index.
        item :
            object in this 2D array.
        """
        if row is not None:
            for cidx, col_list in enumerate(self.arr):
                item = col_list[row]
                if item is not None:
                    yield cidx, row, item
        elif col is not None:
            for ridx, item in enumerate(self.arr[col]):
                if item is not None:
                    yield col, ridx, item
        else:
            for cidx, col_list in enumerate(self.arr):
                for ridx, item in enumerate(col_list):
                    if item is not None:
                        yield cidx, ridx, item

    def get_num_columns(self):
        """Returns the number of columns in this array.

        Returns
        -------
        num_col : int
            the number of columns in this array.
        """
        return len(self.arr)

    def get_num_rows(self):
        """Returns the number of rows in this array.

        Returns
        -------
        num_row : int
            the number of rows in this array.
        """
        if not self.arr:
            return 0
        return len(self.arr[0])


class Block(with_metaclass(abc.ABCMeta, object)):
    """The basic unit for cell placement.

    A block is either a single template instance or a group.  This class
    defines methods needed to arrange and place cells.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.

    Attributes
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    parent : bag.layout.placement.GridGroup
        the parent GridGroup containing this block.
    """

    def __init__(self, grid):
        self.grid = grid
        self.parent = None

    def get_top_layer(self):
        """Returns the top routing layer used by this block.

        Returns
        -------
        top_layer : int
            the top routing layer ID.
        """
        return self.get_min_size()[0]

    def get_dimension(self):
        """Returns the width and height of this block.

        Returns
        -------
        width : float
            the width of this block.
        height : float
            the height of this block.
        """
        return self.grid.get_grid_dimension(self.get_min_size())

    def move_by(self, dx, dy):
        """Move this block by the given amount.

        Parameters
        ----------
        dx : float
            the change in X coordinate.
        dy : float
            the change in Y coordinate.
        """
        loc = self.get_location()
        self.set_location(loc[0] + dx, loc[1] + dy)

    def block_changed(self, update_parent=True):
        """This method is called when the block is modified.

        This method informs any listeners that this block is changed and may need to be revalidated.

        Parameters
        ----------
        update_parent : bool
            True to tell parent container to validate.
        """
        if self.parent is not None and update_parent:
            self.parent.validate()

    def get_bbox(self):
        """Returns the bounding box of this Block.

        Returns
        -------
        bbox : bag.layout.util.BBox
            the bounding box of this block.
        """
        w, h = self.grid.get_grid_dimension(self.get_min_size())
        init_box = BBox(0, 0, w, h, self.grid.resolution)

        return init_box.transform(self.get_location(), self.get_orientation())

    @abc.abstractmethod
    def get_min_size(self, top_layer=None):
        """Returns the minimum size that encloses this block on the given layer.

        Parameters
        ----------
        top_layer : int or None
            if None, will use the block's top routing layer.

        Returns
        -------
        size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        return 0, 0, 0

    @abc.abstractmethod
    def get_orientation(self):
        """Returns the orientation of this block.

        Returns
        -------
        orient : str
            the orientation of this block.
        """
        return 'R0'

    @abc.abstractmethod
    def get_location(self):
        """Returns the location of this block.

        Returns
        -------
        loc : tuple[float]
            the location of this block.
        """
        return 0, 0

    @abc.abstractmethod
    def set_orientation(self, new_orient):
        """Change the orientation of this block.

        This method should only be called by the parent container.  If the orientation of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_orient : str
            the new orientation.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
        """
        pass

    @abc.abstractmethod
    def set_location(self, xc, yc):
        """Change the location of this block.

        This method should only be called by the parent container.  If the location of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        xc : float
            the new X coordinate.
        yc : float
            the new Y coordinate.
        """
        pass

    @abc.abstractmethod
    def resize(self, new_size):
        """Change the size of this block.

        This method should only be called by the parent container.  If the size of this block
        changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_size : tuple[int]
            the new (layer, num_x_block, num_y_block) size tuple.

        Returns
        -------
        msg : str
            empty string if the given size is valid.  Otherwise, msg is an error message.
        """
        pass


class EmptyBlock(Block):
    """A empty block used to add space between blocks.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    size : tuple[int]
        size of this block.
    loc : tuple[float]
        the location of this block.
    orient : str
        the orientation of this block.
    """
    def __init__(self, grid, size, loc=(0, 0), orient='R0'):
        Block.__init__(self, grid)

        self.size = size
        self.loc = loc
        self.orient = orient

    def get_min_size(self, top_layer=None):
        """Returns the minimum size that encloses this block on the given layer.

        Parameters
        ----------
        top_layer : int or None
            if None, will use the block's top routing layer.

        Returns
        -------
        size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        if top_layer is None or top_layer == self.size[0]:
            return self.size
        return self.grid.convert_size(self.size, top_layer=top_layer)

    def get_orientation(self):
        """Returns the orientation of this block.

        Returns
        -------
        orient : str
            the orientation of this block.
        """
        return self.orient

    def get_location(self):
        """Returns the location of this block.

        Returns
        -------
        loc : tuple[float]
            the location of this block.
        """
        return self.loc

    def set_orientation(self, new_orient):
        """Change the orientation of this block.

        This method should only be called by the parent container.  If the orientation of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_orient : str
            the new orientation.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
        """
        self.orient = new_orient

    def set_location(self, xc, yc):
        """Change the location of this block.

        This method should only be called by the parent container.  If the location of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        xc : float
            the new X coordinate.
        yc : float
            the new Y coordinate.
        """
        res = self.grid.resolution
        xc, yc = round(xc / res) * res, round(yc / res) * res
        self.loc = xc, yc

    def resize(self, new_size):
        """Change the size of this block.

        Parameters
        ----------
        new_size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        self.size = new_size


class TemplateInst(Block):
    """A block which represents a single template instance.

    Parameters
    ----------
    temp_db : bag.layout.template.TemplateDB
        the TemplateDB object.
    info : bag.layout.template.TemplateInfo
        the TemplateInfo object.
    inst_name : str
        the name of this particular instance.
    loc : tuple[float]
        the location of this block.
    orient : str
        the orientation of this block.

    Attributes
    ----------
    name : str
        name of this instance.

    """
    def __init__(self, temp_db, info, inst_name, loc=(0, 0), orient='R0'):
        Block.__init__(self, temp_db.grid)
        self.temp_db = temp_db
        self.info = info
        self.name = inst_name
        self.connections = []

        self.size = info.size
        self.size_offset = (0, 0)
        self.loc = loc
        self.orient = orient

    def register_connection(self, conn):
        self.connections.append(conn)

    def resize_pin(self, pin_name, num_tr):
        """Resize the given pin.

        Parameters
        ----------
        pin_name : str
            the pin to resize.
        num_tr : int
            the new number of tracks of this pin.
        """
        new_params = copy.deepcopy(self.info.params)
        port_specs = new_params['port_specs']
        port_specs[pin_name] = PortSpec(num_tr, port_specs[pin_name].idc)

        self.info = self.temp_db.new_template(self.info.lib_name, self.info.temp_name, self.name, new_params)
        self.size = self.info.size
        self.block_changed(update_parent=True)

    def block_changed(self, update_parent=True):
        """This method is called when the block is modified.

        This method informs any listeners that this block is changed and may need to be revalidated.

        Parameters
        ----------
        update_parent : bool
            True to tell parent container to validate.
        """
        for con in self.connections:
            con.invalidate()

        Block.block_changed(self, update_parent=update_parent)

    def get_inst_transform(self):
        """Return the transformation and shift matrices to convert instance coordinate to absolute coordinate.

        Returns
        -------
        transform : np.multiarray.ndarray
            the transformation matrix.
        shift : np.multiarray.ndarray
            the shift matrix.
        """
        res = self.grid.resolution
        mg = _get_transform_matrix(self.orient)
        dg = np.array(self.loc)
        ds = np.array(self.size_offset)
        da = np.dot(mg, ds) + dg

        da[0] = round(da[0] / res) * res
        da[1] = round(da[1] / res) * res

        return mg, da

    def get_pin(self, pin_name):
        """Returns an instance pin with the given name.

        Parameters
        ----------
        pin_name : str
            the pin name

        Returns
        -------
        inst_pin : bag.layout.util.InstPin
            the instance pin with the given name.
        """
        return InstPin(self, pin_name)

    def get_master_pin(self, pin_name):
        """Returns the master pin with the given name.

        Parameters
        ----------
        pin_name : str
            the pin name

        Returns
        -------
        pin : bag.layout.util.Pin or None
            the master pin if it exists, None otherwise.
        """
        return self.info.pins[pin_name]

    def get_instance_location(self):
        """Returns the location of the template instance.

        Returns
        -------
        xc : float
            the X coordinate of the template instance.
        yc : float
            the Y coordinate of the template instance.
        """
        _, da = self.get_inst_transform()
        return da[0], da[1]

    def get_min_size(self, top_layer=None):
        """Returns the minimum size that encloses this block on the given layer.

        Parameters
        ----------
        top_layer : int or None
            if None, will use the block's top routing layer.

        Returns
        -------
        size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        if top_layer is None or top_layer == self.size[0]:
            return self.size
        return self.grid.convert_size(self.size, top_layer=top_layer)

    def get_orientation(self):
        """Returns the orientation of this block.

        Returns
        -------
        orient : str
            the orientation of this block.
        """
        return self.orient

    def get_location(self):
        """Returns the location of this block.

        Returns
        -------
        loc : tuple[float]
            the location of this block.
        """
        return self.loc

    def set_orientation(self, new_orient):
        """Change the orientation of this block.

        This method should only be called by the parent container.  If the orientation of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_orient : str
            the new orientation.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
        """
        if new_orient != self.orient:
            self.orient = new_orient
            self.block_changed(update_parent=False)

    def set_location(self, xc, yc):
        """Change the location of this block.

        This method should only be called by the parent container.  If the location of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        xc : float
            the new X coordinate.
        yc : float
            the new Y coordinate.
        """
        res = self.grid.resolution
        xc, yc = round(xc / res) * res, round(yc / res) * res
        if xc != self.loc[0] or yc != self.loc[1]:
            self.loc = xc, yc
            self.block_changed(update_parent=False)

    def resize(self, new_size):
        """Change the size of this block.

        Parameters
        ----------
        new_size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        if new_size != self.size:
            inst_top_layer = self.info.size[0]
            if new_size[0] < inst_top_layer:
                msg = 'New top layer %d < template top layer %d' % (new_size[0], inst_top_layer)
                raise Exception(msg)

            inst_size = self.info.size
            new_size2 = self.grid.convert_size(new_size, inst_top_layer)
            if new_size2[1] < inst_size[1] or new_size2[2] < inst_size[2]:
                msg = 'new size %s < template size %s' % (str(new_size2), str(inst_size))
                raise Exception(msg)

            if self.info.is_primitive:
                new_params = copy.deepcopy(self.info.params)
                new_params['size'] = new_size2
                self.info = self.temp_db.new_template(self.info.lib_name, self.info.temp_name, self.name, new_params)
                self.size = self.info.size
            else:
                num_x_blk = (new_size2[1] - inst_size[1]) // 2
                num_y_blk = (new_size2[2] - inst_size[2]) // 2

                self.size_offset = self.grid.get_grid_dimension((inst_top_layer, num_x_blk, num_y_blk))
                self.size = new_size

            self.block_changed(update_parent=False)


class GridGroup(Block):
    """A container of multiple blocks that arranges them in a grid.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    blk_top_layer : int or None
        the top routing layer that defines the block pitch.  If None, will use the
        topmost layer of all the blocks.
    """

    def __init__(self, grid, blk_top_layer=None):
        Block.__init__(self, grid)
        self.block_grid = Array2D()

        if blk_top_layer is None:
            self.update_blk_top_layer = True
            blk_top_layer = 0
        else:
            self.update_blk_top_layer = False

        self.blk_top_layer = blk_top_layer
        self.col_widths = []
        self.row_heights = []
        self.row_bidx = []
        self.col_bidx = []

        self.size = (blk_top_layer, 0, 0)
        self.size_offset = (0, 0)
        self.loc = (0, 0)
        self.orient = 'R0'

    def validate(self):
        """Ensure every block in this group is aligned properly.

        This method recalculates all column widths and row heights, then resize and move all blocks
        to stay in the grid.
        """
        # get list of all items
        blk_list = list(self.block_grid.items_iter())
        grp_changed = False

        # get the top-most routing layer of all blocks
        blk_layer = 0
        for cidx, ridx, (blk, _) in blk_list:
            blk_layer = max(blk_layer, blk.get_top_layer())

        if blk_layer != self.blk_top_layer:
            if self.update_blk_top_layer:
                self.blk_top_layer = blk_layer
            elif blk_layer > self.blk_top_layer:
                msg = 'block routing layer (%d) higher than group block top layer (%d).'
                raise Exception(msg % (blk_layer, self.blk_top_layer))

        # recalculate column and row dimensions
        self.col_widths = [0] * self.block_grid.get_num_columns()
        self.row_heights = [0] * self.block_grid.get_num_rows()
        for cidx, ridx, (blk, _) in self.block_grid.items_iter():
            size = blk.get_min_size(top_layer=self.blk_top_layer)
            self.col_widths[cidx] = max(self.col_widths[cidx], size[1])
            self.row_heights[ridx] = max(self.row_heights[ridx], size[2])

        # calculate column/row coordinates
        self.row_bidx = np.cumsum(self.row_heights)
        self.row_bidx = np.insert(self.row_bidx, 0, [0])
        self.col_bidx = np.cumsum(self.col_widths)
        self.col_bidx = np.insert(self.col_bidx, 0, [0])

        # update group size
        new_size = self.blk_top_layer, self.col_bidx[-1], self.row_bidx[-1]
        if self.blk_top_layer > self.size[0]:
            # we have a higher top layer now, throw away old size.
            self.size = new_size
            self.size_offset = (0, 0)
            grp_changed = True
        else:
            # check if old size still encloses the current group.
            eq_size = self.grid.convert_size(self.size, self.blk_top_layer)
            if eq_size[1] < new_size[1] or eq_size[2] < new_size[2]:
                # current size exceed old size, so throw away old size
                self.size = new_size
                self.size_offset = (0, 0)
                grp_changed = True
            else:
                # old size still contains current group, so just recalculate size offset.
                num_x_blk = (eq_size[1] - new_size[1]) // 2
                num_y_blk = (eq_size[2] - new_size[2]) // 2
                self.size_offset = self.grid.get_grid_dimension((self.blk_top_layer, num_x_blk, num_y_blk))

        # resize all blocks
        for cidx, ridx, (blk, blk_orient) in blk_list:
            new_size = self.blk_top_layer, self.col_widths[cidx], self.row_heights[ridx]
            blk.resize(new_size)

        # arrange all blocks
        self._arrange_blocks()

        if grp_changed:
            self.block_changed(update_parent=True)

    def _get_block_location(self, cidx, ridx, blk_orient):
        """Returns the relative block location within this group.

        Parameters
        ----------
        cidx : int
            the block column index.
        ridx : int
            the block row index.
        blk_orient : str
            the block orientation

        Returns
        -------
        xc : float
            the block X coordinate.
        yc : float
            the block Y coordinate.
        """
        if blk_orient == 'R0':
            xc, yc = self.grid.get_grid_dimension((self.blk_top_layer, self.col_bidx[cidx],
                                                   self.row_bidx[ridx]))
        elif blk_orient == 'MY':
            xc, yc = self.grid.get_grid_dimension((self.blk_top_layer, self.col_bidx[cidx + 1],
                                                   self.row_bidx[ridx]))
        elif blk_orient == 'MX':
            xc, yc = self.grid.get_grid_dimension((self.blk_top_layer, self.col_bidx[cidx],
                                                   self.row_bidx[ridx + 1]))
        elif blk_orient == 'R180':
            xc, yc = self.grid.get_grid_dimension((self.blk_top_layer, self.col_bidx[cidx + 1],
                                                   self.row_bidx[ridx + 1]))
        else:
            raise Exception('Illegal orientation %s' % blk_orient)

        return xc + self.size_offset[0], yc + self.size_offset[1]

    def _arrange_blocks(self):
        """Make sure all blocks are at the right location.
        """
        mg = _get_transform_matrix(self.orient)
        dg = np.array(self.loc)
        for cidx, ridx, (blk, blk_orient) in self.block_grid.items_iter():
            # caluate block absolute and relative location/orientation
            mr = _get_transform_matrix(blk_orient)
            dr = np.array(self._get_block_location(cidx, ridx, blk_orient))
            ma = np.dot(mg, mr)
            da = np.dot(mg, dr) + dg

            # set new orientation and location
            abs_orient = _get_orientation(ma)
            blk.set_orientation(abs_orient)
            blk.set_location(da[0], da[1])

    def add_block(self, blk, cidx, ridx, orient='R0'):
        """Add a new block to this group at the given row and column.

        Note: this method simply add the block to this group and will not arrange it properly.  Call
        validate() to actually arrange the blocks.

        Parameters
        ----------
        blk : bag.layout.placement.Block
            the block to add.
        cidx : int
            the column index.
        ridx : int
            the row index.
        orient : str
            the orientation of the block.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
        """
        self.block_grid[cidx, ridx] = blk, orient
        blk.parent = self

    def add_space(self, top_layer, num_x, num_y, cidx, ridx):
        """Adds a new space block to this group at the given row and column.

        Parameters
        ----------
        top_layer : int
            the top routing layer defining the space block.
        num_x : int
            number of horizontal tracks in the space.
        num_y : int
            number of vertical tracks in the space.
        cidx : int
            the column index.
        ridx : int
            the row index.

        Returns
        -------
        space : bag.layout.placement.EmptyBlock
            the EmptyBlock instance representing the space.
        """
        size = self.grid.num_tracks_to_size(top_layer, num_x, num_y)
        space = EmptyBlock(self.grid, size)
        self.add_block(space, cidx, ridx)
        return space

    def get_min_size(self, top_layer=None):
        """Returns the minimum size that encloses this block on the given layer.

        Parameters
        ----------
        top_layer : int or None
            if None, will use the block's top routing layer.

        Returns
        -------
        size : tuple[int]
            the (layer, num_x_block, num_y_block) size tuple.
        """
        if top_layer is None or top_layer == self.size[0]:
            return self.size
        return self.grid.convert_size(self.size, top_layer=top_layer)

    def get_orientation(self):
        """Returns the orientation of this block.

        Returns
        -------
        orient : str
            the orientation of this block.
        """
        return self.orient

    def get_location(self):
        """Returns the location of this block.

        Returns
        -------
        loc : tuple[float]
            the location of this block.
        """
        return self.loc

    def set_orientation(self, new_orient):
        """Change the orientation of this block.

        This method should only be called by the parent container.  If the orientation of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_orient : str
            the new orientation.  Valid values are 'R0', 'MX', 'MY', or 'R180'.
        """
        if new_orient != self.orient:
            self.orient = new_orient
            self._arrange_blocks()
            self.block_changed(update_parent=False)

    def set_location(self, xc, yc):
        """Change the location of this block.

        This method should only be called by the parent container.  If the location of this block
        is changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        xc : float
            the new X coordinate.
        yc : float
            the new Y coordinate.
        """
        res = self.grid.resolution
        xc = round(xc / res) * res
        yc = round(yc / res) * res

        if abs(xc - self.loc[0]) >= res or abs(yc - self.loc[1]) >= res:
            self.loc = xc, yc
            self._arrange_blocks()
            self.block_changed(update_parent=False)

    def resize(self, new_size):
        """Change the size of this block.

        This method should only be called by the parent container.  If the size of this block
        changed, you must call block_changed(update_parent=False) at the end.

        Parameters
        ----------
        new_size : tuple[int]
            the new (layer, num_x_block, num_y_block) size tuple.

        Returns
        -------
        msg : str
            empty string if the given size is valid.  Otherwise, msg is an error message.
        """
        if new_size != self.size:
            if new_size[0] < self.blk_top_layer:
                msg = 'New top layer %d < group block top layer %d' % (new_size[0], self.blk_top_layer)
                raise Exception(msg)

            blk_size = self.blk_top_layer, self.col_bidx[-1], self.row_bidx[-1]
            new_size2 = self.grid.convert_size(new_size, self.blk_top_layer)
            if new_size2[1] < blk_size[1] or new_size2[2] < blk_size[2]:
                msg = 'new size %s < group size %s' % (str(new_size2), str(blk_size))
                raise Exception(msg)

            num_x_blk = (new_size2[1] - blk_size[1]) // 2
            num_y_blk = (new_size2[2] - blk_size[2]) // 2

            self.size_offset = self.grid.get_grid_dimension((self.blk_top_layer, num_x_blk, num_y_blk))
            self.size = new_size
            self._arrange_blocks()
            self.block_changed(update_parent=False)

    def get_inst_iter(self):
        """Returns an iterator over all TemplateInst in this group.

        Yields
        ------
        inst : bag.layout.placement.TemplateInst
            an instance in this group.
        """
        for _, _, (blk, _) in self.block_grid.items_iter():
            if isinstance(blk, GridGroup):
                for inst in blk.get_inst_iter():
                    yield inst
            elif isinstance(blk, TemplateInst):
                yield blk
            elif isinstance(blk, EmptyBlock):
                pass
            else:
                raise Exception('Unknown item: %s' % str(blk))
