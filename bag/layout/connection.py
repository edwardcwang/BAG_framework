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


"""This module defines connection template classes and port specifications.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import abc
import pprint
from future.utils import with_metaclass

from .util import BBox


def create_center_bottom_via(grid, bot_layer_name, box, min_num_tr, idc, pref_dir='upper'):
    """Create a via from sub-routing grid layer to the bottom routing layer.

    Tries to center the via on the bounding box.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    bot_layer_name : str
        name of the sub-routing grid layer.
    box : bag.layout.util.BBox
        the bounding box on the sub-routing grid layer.
    min_num_tr : int
        minimum number of via tracks
    idc : float
        DC current spec, in Amperes.
    pref_dir : str
        preference direction.

    Returns
    -------
    track : int
        center via track ID.
    num_tr : int
        number of via tracks.
    obj_list :
        list of objects.
    """
    bot_layer = grid.layers[0]
    tr = grid.to_nearest_track(box.xc, bot_layer, search_dir=pref_dir)
    max_num_tr = grid.get_max_num_tr(bot_layer)
    for ntr in range(min_num_tr, max_num_tr + 1):
        path = create_bottom_via(grid, bot_layer_name, box, tr, ntr, idc)
        if path:
            return tr, ntr, path

    raise Exception('EM spec cannot be met.')


def create_bottom_via(grid, bot_layer_name, box, track, num_tr, idc):
    """Create a via from sub-routing grid layer to the bottom routing layer.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    bot_layer_name : str
        name of the sub-routing grid layer.
    box : bag.layout.util.BBox
        the bounding box on the sub-routing grid layer.
    track : int
        the via center track ID.
    num_tr : int or None
        number of tracks of via.
    idc : float
        DC current spec, in Amperes.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if EM spec cannot be met.
    """
    via = grid.make_bottom_via(bot_layer_name, box, track, num_tr)
    metal = grid.make_bottom_metal(bot_layer_name, box.merge(via.bot_box))
    if metal.idc < idc:
        raise Exception('Bottom metal Idc = %.4g < %.4g' % (metal.idc, idc))
    if via.idc < idc:
        return []

    return [metal, via]


def create_via_stack(grid, layer1, box1, layer2, track2, idc, num_tr2=None, pref_dir='upper'):
    """Create a via stack from layer1 box1 to layer2 centered at track2 with num_tr tracks.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    layer1 : int
        layer 1 ID.
    box1 : bag.layout.util.BBox
        the starting bounding box.
    layer2 : int
        layer 2 ID.
    track2 : int
        center track ID on layer 2.
    idc : float
        required DC current, in amperes.
    num_tr2 : int or None
        number of tracks on layer 2.
    pref_dir : string
        preference direction of via stack.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if EM spec cannot be met.
    corrections : dict[int, int]
        a dictionary from point ID to new number of tracks needed to meet EM spec.
    """
    # basic error checking
    if layer1 == layer2:
        raise Exception('layer1 = layer2 = %d' % layer1)

    layer_inc = 1 if layer2 > layer1 else -1

    dir2 = grid.get_direction(layer2)
    targ_coord = grid.to_coordinate(track2, layer2)

    obj_list = []
    box_tr, box_num_tr, box_start, box_stop = grid.get_bbox_tracks(box1, layer1)
    opt_num_tr = grid.estimate_num_tracks(layer1, idc)

    # check given box and num_tr2 meet optimistic minimum.
    if box_num_tr < opt_num_tr:
        return [], {1: opt_num_tr}
    if num_tr2 is not None:
        opt_num_tr2 = grid.estimate_num_tracks(layer2, idc)
        if num_tr2 < opt_num_tr2:
            return [], {2: opt_num_tr2}

    param_list = [[layer1, box_tr, box_num_tr, box_start, box_stop, 1]]
    cur_idx = 0
    while param_list[-1][0] != layer2:
        cur_layer, cur_tr, cur_num_tr, cur_start, cur_stop, min_num_tr_next = param_list[cur_idx]
        next_layer = cur_layer + layer_inc
        next_dir = grid.get_direction(next_layer)
        # find the lowest and highest track on next layer inside the bounding box
        low_track = grid.to_nearest_track(cur_start, next_layer, search_dir='upper')
        high_track = grid.to_nearest_track(cur_stop, next_layer, search_dir='lower')
        # compute number of tracks on next layer inside the bounding box
        num_ovr_tr = max(0, high_track - low_track + 1)

        # find number of tracks needed on the next layer for EM spec.
        via = None
        via_tr = None
        num_tr_next = min_num_tr_next
        if next_layer == layer2 and num_tr2 is not None:
            # via to last level is fixed.
            via_tr = track2
            num_tr_next = num_tr2
            if cur_layer < next_layer:
                via = grid.make_via(cur_layer, cur_tr, cur_num_tr, via_tr, num_tr_next)
            else:
                via = grid.make_via(next_layer, via_tr, num_tr_next, cur_tr, cur_num_tr)
        else:
            max_num_tr = grid.get_max_num_tr(next_layer)
            for num_tr_next in range(min_num_tr_next, max_num_tr + 1, 1):
                # determine via middle track.
                if next_layer == layer2:
                    via_tr = track2
                elif next_dir == dir2:
                    # if next layer parallel to top layer, try to make via center on track2.
                    via_tr = grid.to_nearest_track(targ_coord, next_layer)
                else:
                    # if next layer perpendicular to top layer, try to make via overlap bounding box,
                    # but skew to pref_dir
                    if num_tr_next < num_ovr_tr:
                        offset = num_tr_next / 2
                    elif num_tr_next == num_ovr_tr:
                        offset = (num_tr_next - 1) / 2
                    else:
                        init = (num_tr_next - 1) / 2
                        diff = num_tr_next - num_ovr_tr
                        if num_ovr_tr % 2 == 1:
                            offset = init - (diff + 1) / 2
                        else:
                            offset = init - diff / 2

                    if pref_dir == 'upper':
                        via_tr = high_track - offset
                    else:
                        via_tr = low_track + offset

                # create via
                if cur_layer < next_layer:
                    via = grid.make_via(cur_layer, cur_tr, cur_num_tr, via_tr, num_tr_next)
                else:
                    via = grid.make_via(next_layer, via_tr, num_tr_next, cur_tr, cur_num_tr)

                if via.idc >= idc:
                    break
                else:
                    via = None

        if via is None:
            raise Exception('Cannot find via that pass EM spec idc = %.4g' % idc)

        if cur_layer < next_layer:
            vcur_box = via.bot_box
            vnext_box = via.top_box
        else:
            vcur_box = via.top_box
            vnext_box = via.bot_box

        # create metal to connect bounding box to via.
        if grid.get_direction(cur_layer) == 'x':
            mstart = min(cur_start, vcur_box.left)
            mstop = max(cur_stop, vcur_box.right)
            vstart = vnext_box.bottom
            vstop = vnext_box.top
        else:
            mstart = min(cur_start, vcur_box.bottom)
            mstop = max(cur_stop, vcur_box.top)
            vstart = vnext_box.left
            vstop = vnext_box.right

        metal = grid.make_metal(cur_layer, cur_tr, mstart, mstop, num_tr=cur_num_tr)
        if metal.idc < idc or via.idc < idc:
            # metal/via does not meet EM rule.  previous via needs to be larger
            if cur_idx == 0:
                if metal.idc < idc:
                    # port needs to be wider
                    return [], {1: box_num_tr + 1}
                else:
                    # via does not meet EM rule only if we're making we're only making
                    # one via, and num_tr2 is specified.
                    # in this case, increase the small of the two layers.
                    if box_num_tr <= num_tr2:
                        return [], {1: box_num_tr + 1}
                    else:
                        return [], {2: num_tr2 + 1}
            else:
                # remove previous via and metal, and redo it.
                obj_list = obj_list[:-2]
                param_list[cur_idx - 1][-1] += 1
                cur_idx -= 1
        else:
            # metal pass EM rule, we can move on to next layer
            obj_list.append(metal)
            obj_list.append(via)
            param_list[cur_idx][-1] = num_tr_next
            next_param = [next_layer, via_tr, num_tr_next, vstart, vstop, 1]
            if cur_idx + 1 >= len(param_list):
                param_list.append(next_param)
            else:
                param_list[cur_idx + 1] = next_param
            cur_idx += 1

    return obj_list, {}


def create_straight_line(grid, layer, box1, box2, idc):
    """Draw a straight line connecting the two given boxes.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    layer : int
        the box layer ID.
    box1 : bag.layout.util.BBox
        the starting bounding box.
    box2 : bag.layout.util.BBox
        the stopping bounding box.
    idc : float
        required DC current, in amperes.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if an error occured.
    corrections : dict[int, int]
        a dictionary from point ID to new number of tracks needed to meet EM spec.
    """
    tr1, num_tr1, start1, stop1 = grid.get_bbox_tracks(box1, layer)
    tr2, num_tr2, start2, stop2 = grid.get_bbox_tracks(box2, layer)

    # check trace ID and number of tracks matches.
    if num_tr1 != num_tr2:
        if num_tr1 < num_tr2:
            return [], {1: num_tr2}
        else:
            return [], {2: num_tr1}
    if tr1 != tr2:
        raise Exception('Boxes %s and %s not aligned' % (box1, box2))

    # make wire
    start = min(start1, start2)
    stop = max(stop1, stop2)
    metal = grid.make_metal(layer, tr1, start, stop, num_tr=num_tr1)
    # check EM spec.
    if metal.idc < idc:
        return [], {1: num_tr1 + 1, 2: num_tr1 + 1}
    else:
        return [metal], {}


def _create_l_bridge(grid, layer1, box1, layer2, box2, idc):
    """Helper method for create_bridge.

    draw a via from point1 to layer of point2, then connect with a straight
    line.  Looks like an "L" from vertical cross section.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    layer1 : int
        layer 1 ID.
    box1 : bag.layout.util.BBox
        the starting bounding box.
    layer2 : int
        layer 2 ID.
    box2 : bag.layout.util.BBox
        the stopping bounding box.
    idc : float
        required DC current, in amperes.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if an error occured.
    corrections : dict[int, int]
        a dictionary from point ID to new number of tracks needed to meet EM spec.
    """
    tr1, num_tr1, start1, stop1 = grid.get_bbox_tracks(box1, layer1)
    tr2, num_tr2, start2, stop2 = grid.get_bbox_tracks(box2, layer2)

    if grid.get_direction(layer2) == 'x':
        c1 = box1.xc
        c2 = box2.xc
    else:
        c1 = box1.yc
        c2 = box2.yc

    pref_dir = 'upper' if c1 < c2 else 'lower'
    via_list, corrections = create_via_stack(grid, layer1, box1, layer2, tr2, idc,
                                             num_tr2=num_tr2, pref_dir=pref_dir)
    if not via_list:
        # via failed EM spec.
        return [], corrections

    if layer2 > layer1:
        vbox = via_list[-1].top_box
    else:
        vbox = via_list[-1].bot_box

    obj_list, corrections = create_straight_line(grid, layer2, vbox, box2, idc)
    if not obj_list:
        # can't draw straight line; port2 not wide enough.
        return [], {2: num_tr1 + 1}
    via_list.extend(obj_list)
    return via_list, {}


def create_bridge(grid, bridge_layer, layer1, box1, layer2, box2, idc, pref_coord=None):
    """Create a bridge connection between two points.

    This method also may or may not draw via stacks, depending on the bridge layer
    relative to the layer of the two points.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    bridge_layer : int
        the bridge layer ID.
    layer1 : int
        layer 1 ID.
    box1 : bag.layout.util.BBox
        the starting bounding box.
    layer2 : int
        layer 2 ID.
    box2 : bag.layout.util.BBox
        the stopping bounding box.
    idc : float
        required DC current, in amperes.
    pref_coord : int or None
        preferred bridge wire center coordinate.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if an error occured.
    corrections : dict[int, int]
        a dictionary from point ID to new number of tracks needed to meet EM spec.
    """
    max_num_trb = grid.get_max_num_tr(bridge_layer)

    if bridge_layer == layer1 == layer2:
        # straight line connection.
        return create_straight_line(grid, layer1, box1, box2, idc)
    elif bridge_layer == layer1:
        # don't need to draw via at point 1
        obj_list, corrections = _create_l_bridge(grid, layer2, box2, layer1, box1, idc)

        # need to swap 1 and 2
        if not obj_list:
            new_corrections = {}
            if 1 in corrections:
                new_corrections[2] = corrections[1]
            if 2 in corrections:
                new_corrections[1] = corrections[2]
            return [], new_corrections

        # flip order
        obj_list.reverse()
        return obj_list, {}
    elif bridge_layer == layer2:
        # don't need to draw via at point 2
        obj_list, corrections = _create_l_bridge(grid, layer1, box1, layer2, box2, idc)
        return obj_list, corrections
    else:
        # need to draw via on both points.

        tr1, num_tr1, start1, stop1 = grid.get_bbox_tracks(box1, layer1)
        tr2, num_tr2, start2, stop2 = grid.get_bbox_tracks(box2, layer2)

        # find middle coordinate parallel to bridge direction.
        if grid.get_direction(bridge_layer) == 'x':
            c1 = box1.xc
            c2 = box2.xc
            mid = (box1.yc + box2.yc) / 2.0
        else:
            c1 = box1.yc
            c2 = box2.yc
            mid = (box1.xc + box2.xc) / 2.0

        if pref_coord is None:
            pref_coord = mid

        min_num_trb = grid.estimate_num_tracks(bridge_layer, idc)
        if c1 < c2:
            dir1 = 'upper'
            dir2 = 'lower'
        else:
            dir1 = 'lower'
            dir2 = 'upper'

        # iterate to find number of tracks needed for bridge.
        port1_fail = False
        port2_fail = False
        trb = grid.to_nearest_track(pref_coord, bridge_layer)
        for num_trb in range(min_num_trb, max_num_trb):
            # create vias
            via1_list, corrections = create_via_stack(grid, layer1, box1, bridge_layer, trb, idc,
                                                      num_tr2=num_trb, pref_dir=dir1)
            if not via1_list:
                port1_fail = True
                if abs(layer1 - bridge_layer) > 1:
                    # need wider port 1.
                    return [], {1: corrections[1]}
            else:
                port1_fail = False

            via2_list, corrections = create_via_stack(grid, layer2, box2, bridge_layer, trb, idc,
                                                      num_tr2=num_trb, pref_dir=dir2)
            if not via2_list:
                port2_fail = True
                if abs(layer2 - bridge_layer) > 1:
                    # need wider port 2.
                    return [], {2: corrections[1]}
            else:
                port2_fail = False

            if port1_fail or port2_fail:
                # if layer1 and layer2 are just 1 layer below bridge layer, we could potentially
                # meet EM spec by increasing num_trb
                continue

            # draw bridge
            if bridge_layer > layer1:
                vbox1 = via1_list[-1].top_box
            else:
                vbox1 = via1_list[-1].bot_box
            if bridge_layer > layer2:
                vbox2 = via2_list[-1].top_box
            else:
                vbox2 = via2_list[-1].bot_box

            obj_list, _ = create_straight_line(grid, bridge_layer, vbox1, vbox2, idc)
            if obj_list:
                # return bridge
                via1_list.extend(obj_list)
                via1_list.extend(reversed(via2_list))
                return via1_list, {}

        # EM spec is not satisfied for all choses of num_trb.  Need to increase port1/port2 size
        corrections = {}
        if port1_fail:
            corrections[1] = num_tr1 + 1
        if port2_fail:
            corrections[2] = num_tr2 + 1

        return [], corrections


def create_bar_connect(grid, bar_layer, bar_tr, bar_num_tr, lay_list, bbox_list, blay_list, idc_list, itot):
    """Draw a bar, then draw straight lines from given points to the bar.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the RoutingGrid instance.
    bar_layer : int
        the bar layer ID.
    bar_tr : int
        the bar track ID.
    bar_num_tr : int
        number of tracks for the bar.
    lay_list : list[int]
        list of point layers.
    bbox_list : list[bag.layout.util.BBox]
        list of point bounding boxes.
    blay_list : list[int]
        list of bridge layers.
    idc_list : list[float]
        list of DC current for each point connection.
    itot : float
        Dc current for the bar.

    Returns
    -------
    obj_list :
        list of objects, or an empty list if an error occured.
    corrections : dict[int, int]
        a dictionary from point ID to new number of tracks needed to meet EM spec.
    """
    bar_dir = grid.get_direction(bar_layer)

    # create bridges to bar.
    path = []
    overall_corrections = {}
    bar_bbox = BBox.get_invalid_bbox()
    bar_lay_str = grid.tech_info.get_layer_name(bar_layer)
    for idx, (lay, blay, bbox, idc) in enumerate(zip(lay_list, blay_list, bbox_list, idc_list)):
        if bar_dir == 'x':
            start = bbox.left
            stop = bbox.right
        else:
            start = bbox.bottom
            stop = bbox.top

        temp_box = grid.get_bbox(bar_layer, bar_tr, start, stop, bar_num_tr)
        obj_list, corrections = create_bridge(grid, blay, lay, bbox, bar_layer, temp_box, idc)
        if not obj_list:
            if 1 in corrections:
                overall_corrections[idx] = corrections[1]
            if 2 in corrections:
                val = corrections[2]
                if -1 in overall_corrections:
                    overall_corrections[-1] = max(overall_corrections[-1], val)
                else:
                    overall_corrections[-1] = val
        else:
            path.extend(obj_list)
            for obj in obj_list:
                bar_bbox = bar_bbox.merge(obj.get_bbox(bar_lay_str))

    # if any bridges fail, return corrections.
    if overall_corrections:
        return [], overall_corrections

    # create overall bar
    if bar_dir == 'x':
        start = bar_bbox.left
        stop = bar_bbox.right
    else:
        start = bar_bbox.bottom
        stop = bar_bbox.top

    for obj in path:
        print(obj)

    metal = grid.make_metal(bar_layer, bar_tr, start, stop, num_tr=bar_num_tr)
    if metal.idc < itot:
        return [], {-1: bar_num_tr + 1}
    else:
        path.append(metal)
        return path, {}


class Connection(with_metaclass(abc.ABCMeta, object)):
    """The connection base class.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the routing grid.

    Attributes
    ----------
    grid : bag.layout.routing.RoutingGrid
        the routing grid.
    path :
        a list of objects making the connection.
    is_valid : bool
        True if this connection is valid.

    """

    def __init__(self, grid):
        self.grid = grid
        self.path = None
        self.is_valid = False

    def invalidate(self):
        self.is_valid = False

    def get_geometries(self):
        return self.path

    def move_by(self, dx, dy):
        """Moves this connection by the given amount.

        Parameters
        ----------
        dx : float
            delta X.
        dy : float
            delta Y.
        """
        if self.path is not None:
            for item in self.path:
                item.move_by(dx, dy)

    def get_bbox(self):
        """Returns the bounding box of this connection.

        Returns
        -------
        bbox : bag.layout.util.BBox
            the bounding box.
        """
        box = BBox.get_invalid_bbox()

        if not self.path:
            return box
        else:
            for obj in self.path:
                box = box.merge(obj.get_bbox())

        return box

    @abc.abstractmethod
    def validate(self):
        return True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return pprint.pformat(self.path)


class StraightLine(Connection):
    """A straight line connection.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the routing grid.
    pin1 : bag.layout.util.InstPin
        the first pin.
    pin2 : bag.layout.util.InstPin
        the second pin.
    num_sp : int
        number of space tracks.
    idc : float
        maximum DC current, in Amperes.
    """
    def __init__(self, grid, pin1, pin2, num_sp=0, idc=0):
        Connection.__init__(self, grid)
        self.pin1 = pin1
        self.pin2 = pin2
        self.num_sp = num_sp
        self.idc = idc

        pin1.inst.register_connection(self)
        pin2.inst.register_connection(self)

    def validate(self):
        self.path = None
        self.is_valid = False

        if self.pin1.layer != self.pin2.layer:
            raise Exception('Ports on different layer: %d and %d' % (self.pin1.layer, self.pin2.layer))

        lay = self.pin1.layer

        obj_list, corrections = create_straight_line(self.grid, lay, self.pin1.bbox, self.pin2.bbox, self.idc)

        if not obj_list:
            if 1 in corrections:
                self.pin1.resize(corrections[1])
            if 2 in corrections:
                self.pin2.resize(corrections[2])
            return False

        self.path = obj_list
        self.is_valid = True
        return True


class BarConnect(Connection):
    """A bar connecting multiple ports together.

    Parameters
    ----------
    grid : bag.layout.routing.RoutingGrid
        the routing grid.
    bar_layer : int
        bar layer ID.
    bar_tr : int
        bar track ID.
    pin_list : list[bag.layout.util.InstPin]
        list of pins.
    pin_lay_list : list[int]
        list of pin connection layer IDs.
    ispec_list : list
        list of current specs.
    min_num_tr : int
        minimum number of bar tracks
    pref_dir : str
        Rounding direction for track number.  Valid values are 'upper' or 'lower'.

    """
    def __init__(self, grid, bar_layer, bar_tr, pin_list, pin_lay_list, ispec_list,
                 min_num_tr=1, pref_dir='upper'):
        Connection.__init__(self, grid)
        self.bar_layer = bar_layer
        self.bar_tr = bar_tr
        self.pin_list = pin_list
        self.pin_lay_list = pin_lay_list
        self.ispec_list = ispec_list
        self.min_num_tr = min_num_tr
        self.max_num_tr = grid.get_max_num_tr(bar_layer)
        self.pref_dir = pref_dir

        for pin in self.pin_list:
            pin.inst.register_connection(self)

    def validate(self):
        self.path = None
        self.is_valid = False

        lay_list = []
        bbox_list = []
        idc_list = []
        isink = 0.0
        isrc = 0.0
        for pin, ispec in zip(self.pin_list, self.ispec_list):
            lay_list.append(pin.layer)
            bbox_list.append(pin.bbox)
            idc_list.append(ispec[0])
            if ispec[1] > 0:
                isrc += ispec[0]
            else:
                isink += ispec[0]

        itot = max(isrc, isink)
        path = None
        for num_tr in range(self.min_num_tr, self.max_num_tr + 1):
            obj_list, corrections = create_bar_connect(self.grid, self.bar_layer, self.bar_tr, num_tr,
                                                       lay_list, bbox_list, self.pin_lay_list, idc_list, itot)
            if not obj_list:
                if -1 not in corrections:
                    for key, val in corrections.items():
                        self.pin_list[key].resize(val)
                    return False
            else:
                path = obj_list
                break

        if path is None:
            return False

        self.path = path
        self.is_valid = True
        return True
