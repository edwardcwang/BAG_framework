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
                 wire_type,  # type: str
                 num_tr=0,  # type: Union[float, int]
                 space=0,  # type: Union[float, int]
                 tr_manager=None,  # type: Optional[TrackManager]
                 name_list=None,  # type: Optional[List[str]]
                 track_offset=0,  # type: int
                 ):
        # type: (...) -> None
        self.space = space
        self._layer = layer_id
        self._wire_type = wire_type
        self._tr_manager = tr_manager
        if name_list is None:
            self._num_tr = num_tr
            self._locs = None
            self._names = None
        else:
            self._names = name_list
            self._num_tr, self._locs = tr_manager.place_wires(layer_id, name_list)

        if self._num_tr < 1:
            raise ValueError('Cannot create WireGroup with < 1 track.')
        self._tr_off = track_offset
        self._children = []

    def copy(self):
        """Returns a copy of this wire group.  Note: children will not be copied."""
        return WireGroup(self._layer, self._wire_type, num_tr=self._num_tr,
                         space=self.space, tr_manager=self._tr_manager, name_list=self._names,
                         track_offset=self._tr_off)

    @property
    def names(self):
        return self._names

    @property
    def locations(self):
        if self._locs is None:
            return None
        return [l + self._tr_off for l in self._locs]

    @property
    def type(self):
        return self._wire_type

    @property
    def interval(self):
        # type: () -> Tuple[Union[float, int], Union[float, int]]
        return self._tr_off, self._tr_off + self._num_tr

    @property
    def tr_manager(self):
        # type: () -> TrackManager
        return self._tr_manager

    @property
    def track_offset(self):
        # type: () -> Union[float, int]
        return self._tr_off

    @property
    def num_track(self):
        # type: () -> Union[float, int]
        return self._num_tr

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
            return None, self.last_used_track, 1

        name = self._names[-1]
        idx = self._locs[-1] + self._tr_off
        width = self._tr_manager.get_width(self._layer, name)
        return name, idx, width

    @property
    def last_used_track(self):
        # type: () -> Union[int, float]
        return self._tr_off + self._num_tr - 1

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
                space = max(space, wire_grp.tr_manager.get_space(self._layer, name2))
        else:
            if name2 is None:
                space = max(self._tr_manager.get_space(self._layer, name1), wire_grp.space)
            else:
                space = self._tr_manager.get_space(self._layer, (name1, name2))

        return space

    def get_mirror_space(self, wire_grp, first=True):
        # type: (WireGroup, bool) -> Union[int, float]
        if first:
            return self._get_space(wire_grp, self.first_track[0], wire_grp.first_track[0])
        return self._get_space(wire_grp, self.last_track[0], wire_grp.last_track[0])

    def place_child(self, wire_grp):
        # type: (WireGroup) -> Union[int, float]
        sp = self._get_space(wire_grp, self.last_track[0], wire_grp.first_track[0])
        return self._tr_off + self._num_tr + sp

    def set_parents(self, parents):
        # type: (List[WireGroup]) -> None

        new_tr_off = -float('inf')
        for parent_wg in parents:
            cur_tr_off = parent_wg.place_child(self)
            parent_wg.add_child(self)
            new_tr_off = max(new_tr_off, cur_tr_off)

        self._tr_off = new_tr_off

    def move_by(self, delta, propagate=True):
        # type: (Union[float, int], bool) -> None
        if delta != 0:
            self._tr_off += delta
            if propagate:
                for child in self._children:
                    cur_tr_off = child.track_offset
                    new_tr_off = self.place_child(child)
                    if new_tr_off > cur_tr_off:
                        child.move_by(new_tr_off - cur_tr_off, propagate=True)

    def move_up(self, delta_max=0):
        # type: (Union[float, int]) -> None
        delta = delta_max
        for child in self._children:
            child_idx = self.place_child(child)
            delta = min(delta, child.track_offset - child_idx)
        self._tr_off += delta


class WireTree(object):
    def __init__(self, mirror=False):
        # type: (bool) -> None
        self._wire_list = []
        self._wire_ids = []
        self._mirror = mirror

    def copy(self):
        new_tree = WireTree(mirror=self._mirror)
        for wg, wid in zip(self._wire_list, self._wire_ids):
            new_wg = [w.copy() for w in wg]
            new_tree.add_wires(new_wg, wid)
        return new_tree

    @classmethod
    def _get_half_space(cls, sp):
        # type: (Union[float, int]) -> Union[float, int]
        sp2 = int(round(2 * sp))
        vtest = sp2 % 4
        if vtest == 0 or vtest == 3:
            return (sp2 + 1) // 4
        else:
            return ((sp2 + 1) // 2) / 2

    def add_wires(self, wire_groups, wire_id):
        # type: (List[WireGroup], Tuple[int, int]) -> None
        if self._wire_list:
            last_wg = self._wire_list[-1]
            for wg in wire_groups:
                wg.set_parents(last_wg)
            self._wire_ids.append(wire_id)
            self._wire_list.append(wire_groups)
        else:
            self._wire_ids.append(wire_id)
            self._wire_list.append(wire_groups)
            if self._mirror:
                for w1 in wire_groups:
                    sp = 0
                    for w2 in wire_groups:
                        sp = max(sp, w1.get_mirror_space(w2))

                    w1.move_by(self._get_half_space(sp))

    def get_wire_groups(self, wire_id, get_next=False):
        # type: (Tuple[int, int]) -> Optional[List[WireGroup]]
        idx = bisect.bisect_left(self._wire_ids, wire_id)
        if idx == len(self._wire_ids):
            return None
        if self._wire_ids[idx] == wire_id or get_next:
            return self._wire_list[idx]
        else:
            return None

    def get_top_tr(self):
        # type: () -> Optional[Union[float, int]]
        top_tr = None
        if not self._wire_list:
            return top_tr
        wire_groups = self._wire_list[-1]
        if self._mirror:
            for w1 in wire_groups:
                sp = 0
                for w2 in wire_groups:
                    sp = max(sp, w1.get_mirror_space(w2, first=False))

                last_tr = w1.last_used_track + 0.5 + self._get_half_space(sp)
                if top_tr is None or last_tr > top_tr:
                    top_tr = last_tr
        else:
            for wg in self._wire_list[-1]:
                last_tr = wg.last_used_track + 0.5
                if top_tr is None or last_tr > top_tr:
                    top_tr = last_tr

        return top_tr
