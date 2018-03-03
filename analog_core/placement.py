# -*- coding: utf-8 -*-

"""This module contains transistor row placement methods and data structures."""

from typing import TYPE_CHECKING, Optional, List, Union, Tuple

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

    def place_child(self, wire_grp):
        # type: (WireGroup) -> Union[int, float]
        name1, idx1, _ = self.last_track
        name2, _, _ = wire_grp.first_track

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

        return self._tr_off + self._num_tr + space

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


class TrackCollection(object):
    pass
