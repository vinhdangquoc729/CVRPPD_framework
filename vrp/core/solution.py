from dataclasses import dataclass, field
from typing import List, Dict
from .problem import Node

@dataclass
class Route: # lộ trình 1 chuyến đơn lẻ
    vehicle_id: int 
    seq: List[int] # chuỗi các điểm: [Depot A, Khach 1, ..., Depot B]

@dataclass
class Solution:
    routes: List[Route] # danh sách phẳng tất cả các chuyến trong ngày

    @property
    def journeys(self) -> Dict[int, List[Route]]:
        j = {}
        for r in self.routes:
            j.setdefault(r.vehicle_id, []).append(r)
        return j

    def all_customers(self, nodes: Dict[int, "Node"]) -> List[int]:
        out = []
        for r in self.routes:
            out.extend([i for i in r.seq if not nodes[i].is_depot])
        return out