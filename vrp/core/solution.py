from dataclasses import dataclass, field
from typing import List, Dict
from .problem import Node

@dataclass
class Route: # lộ trình 1 xe
    vehicle_id: int 
    seq: List[int] # chuỗi các điểm

@dataclass
class Solution:
    routes: List[Route] # lời giải là các lộ trình

    def all_customers(self, nodes: Dict[int, "Node"]) -> List[int]:
        out = []
        for r in self.routes:
            out.extend([i for i in r.seq if not nodes[i].is_depot])
        return out