from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

@dataclass
class Node:
    id: int
    x: float
    y: float
    demand_delivery: int = 0 
    demand_pickup: int = 0 
    is_depot: bool = False
    service_time: int = 0
    tw_open: Optional[int] = None 
    tw_close: Optional[int] = None
    cluster: Optional[int] = None


@dataclass
class Vehicle:
    id: int
    depot_id: int
    capacity: int
    start_time: int
    end_time: int
    fixed_cost: float
    var_cost_per_dist: float


class Problem:
    def __init__(
        self,
        nodes: Dict[int, Node],
        vehicles: List[Vehicle],
        tw_penalty_per_min: float = 1.0,
        speed_units_per_min: float = 60.0,
        prohibited: Optional[Set[Tuple[int, int]]] = None,
        pd_pairs: Optional[Dict[int, Tuple[int, int]]] = None,
        # pd_pairs: key = pickup_node_id, value = (delivery_node_id, quantity)
    ):
        self.nodes: Dict[int, Node] = nodes
        self.vehicles: List[Vehicle] = vehicles
        self.depots: List[int] = [nid for nid, n in nodes.items() if n.is_depot]
        self.customers: List[int] = [nid for nid, n in nodes.items() if not n.is_depot]

        self.tw_penalty_per_min: float = float(tw_penalty_per_min)
        self.speed_units_per_min: float = float(speed_units_per_min)

        self.prohibited: Set[Tuple[int, int]] = set(prohibited or set())

        self.pd_pairs: Dict[int, Tuple[int, int]] = dict(pd_pairs or {})
        self.pd_reverse: Dict[int, Tuple[int, int]] = {
            d: (p, qty) for p, (d, qty) in self.pd_pairs.items()
        }

        self.ids: List[int] = sorted(nodes.keys())
        self._id_index: Dict[int, int] = {nid: i for i, nid in enumerate(self.ids)}
        coords = np.array([[nodes[i].x, nodes[i].y] for i in self.ids], dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        self._dist = np.sqrt((diff ** 2).sum(axis=2))  # (N, N) euclidean

        eps = 1e-9
        self._time = self._dist / max(self.speed_units_per_min, eps)

        self.veh_by_id: Dict[int, Vehicle] = {v.id: v for v in vehicles}
        self.depot_of_vehicle: Dict[int, int] = {v.id: v.depot_id for v in vehicles}
        self.vehicles_of_depot: Dict[int, List[int]] = {}
        for v in vehicles:
            self.vehicles_of_depot.setdefault(v.depot_id, []).append(v.id)

    def d(self, i: int, j: int) -> float:
        """Khoảng cách giữa 2 id nút."""
        return float(self._dist[self._id_index[i], self._id_index[j]])

    def time(self, i: int, j: int) -> float:
        """Thời gian di chuyển."""
        return float(self._time[self._id_index[i], self._id_index[j]])

    def is_arc_prohibited(self, i: int, j: int) -> bool:
        """Kiểm tra cung (i->j) có bị cấm không."""
        return (i, j) in self.prohibited

    def vehicle_capacity(self, vehicle_id: int) -> int:
        return self.veh_by_id[vehicle_id].capacity

    def vehicle_time_window(self, vehicle_id: int) -> Tuple[int, int]:
        v = self.veh_by_id[vehicle_id]
        return v.start_time, v.end_time

    def depot_of(self, vehicle_id: int) -> int:
        return self.depot_of_vehicle[vehicle_id]

    def is_pickup(self, node_id: int) -> bool:
        return node_id in self.pd_pairs

    def is_delivery(self, node_id: int) -> bool:
        return node_id in self.pd_reverse

    def mate_of(self, node_id: int) -> Optional[int]:
        if node_id in self.pd_pairs:
            return self.pd_pairs[node_id][0]
        if node_id in self.pd_reverse:
            return self.pd_reverse[node_id][0]
        return None

    def qty_of_order(self, node_id: int) -> Optional[int]:
        if node_id in self.pd_pairs:
            return self.pd_pairs[node_id][1]
        if node_id in self.pd_reverse:
            return self.pd_reverse[node_id][1]
        return None
