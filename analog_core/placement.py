# -*- coding: utf-8 -*-

"""This module contains transistor row placement methods and data structures."""

from typing import TYPE_CHECKING, Optional, List, Union, Tuple

import bisect

if TYPE_CHECKING:
    from bag.layout.routing import TrackManager


class WireGroup(object):
    """A group of horizontal wires associated with a transistor row."""
    def __init__(self,
                 layer_id,  # type: int
                 num_tr,  # type: Union[float, int]
                 space=0,  # type: Union[float, int]
                 tr_manager=None,  # type: Optional[TrackManager]
                 name_list=None,  # type: Optional[List[str]]
                 ):
        # type: (...) -> None
        if num_tr < 1:
            raise ValueError('Cannot create WireGroup with < 1 track.')
        self.space = space
        self._layer = layer_id
        self._tr_manager = tr_manager
        if name_list is None:
            self._num_tr = num_tr
            self._locs = None
            self._names = None
        else:
            self._names = name_list
            self._num_tr, self._locs = tr_manager.place_wires(layer_id, name_list)

        self._tr_off = 0
        self._children = []

    @property
    def track_offset(self):
        # type: () -> Union[float, int]
        return self._tr_off

    @property
    def first_track(self):
        # type: () -> Tuple[Optional[str], Union[float, int], int]
        if self._names is None:
            return None, self._tr_off, 1

        name = self._names[0]
        idx = self._locs[0] + self._tr_off
        width = self._tr_manager.get_width(self._layer, name)
        return name, idx, width

    @property
    def last_track(self):
        # type: () -> Tuple[Optional[str], Union[float, int], int]
        if self._names is None:
            return None, self._tr_off + self._num_tr - 1, 1

        name = self._names[-1]
        idx = self._locs[-1] + self._tr_off
        width = self._tr_manager.get_width(self._layer, name)
        return name, idx, width

    def add_child(self, wire_grp):
        # type: (WireGroup) -> None
        self._children.append(wire_grp)

    def _get_space(self, wire_grp, name1, name2):
        # type: (WireGroup, str, str) -> Union[int, float]
        if name1 is None:
            space = self.space
            if name2 is None:
                space = max(space, wire_grp.space)
            else:
                space = max(space, self._tr_manager.get_space(self._layer, name2))
        else:
            if name2 is None:
                space = max(self._tr_manager.get_space(self._layer, name1), wire_grp.space)
            else:
                space = self._tr_manager.get_space(self._layer, (name1, name2))

        return space

    def get_mirror_space(self, wire_grp):
        # type: (WireGroup) -> Union[int, float]
        return self._get_space(wire_grp, self.first_track[0], wire_grp.first_track[0])

    def place_child(self, wire_grp):
        # type: (WireGroup) -> Union[int, float]
        sp = self._get_space(wire_grp, self.last_track[0], wire_grp.first_track[0])
        return self._tr_off + self._num_tr + sp

    def set_parents(self, parents):
        # type: (List[WireGroup]) -> None

        new_tr_off = -float('inf')
        new_parents = []
        for parent_wg in parents:
            cur_tr_off = parent_wg.place_child(self)
            if cur_tr_off > new_tr_off:
                del new_parents[:]
                new_tr_off = cur_tr_off
                new_parents.append(parent_wg)
            elif cur_tr_off == new_tr_off:
                new_parents.append(parent_wg)

        self._tr_off = new_tr_off
        for parent_wg in new_parents:
            parent_wg.add_child(self)

    def move_by(self, delta, propagate=True):
        # type: (Union[float, int], bool) -> None
        if delta != 0:
            self._tr_off += delta
            if propagate:
                for child in self._children:
                    cur_tr_off = child.track_offset
                    new_tr_off = self.place_child(child)
                    if new_tr_off > cur_tr_off:
                        child.move_propagate(new_tr_off - cur_tr_off, propagate=True)


class WireTree(object):
    def __init__(self, wire_groups, wire_id, mirror=False):
        # type: (List[WireGroup], Tuple[int, int], bool) -> None
        self._wire_list = [wire_groups]
        self._wire_ids = [wire_id]
        if mirror:
            for w1 in wire_groups:
                sp = 0
                for w2 in wire_groups:
                    sp = max(sp, w1.get_mirror_space(w2))

                sp2 = int(round(2 * sp))
                vtest = sp2 % 4
                if vtest == 0 or vtest == 3:
                    sp = (sp2 + 1) // 4
                else:
                    sp = ((sp2 + 1) // 2) / 2
                w1.move_by(sp)

    def add_wires(self, wire_groups, wire_id):
        # type: (List[WireGroup], Tuple[int, int]) -> None
        for wg in wire_groups:
            wg.set_parents(self._wire_list[-1])
        self._wire_ids.append(wire_id)
        self._wire_list.append(wire_groups)

    def get_wire_group(self, wire_id):
        # type: (Tuple[int, int]) -> Optional[List[WireGroup]]
        idx = bisect.bisect_left(self._wire_ids, wire_id)
        if self._wire_ids[idx] == wire_id:
            return self._wire_list[idx]
        else:
            return None
