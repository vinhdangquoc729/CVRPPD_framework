# FILE: solution.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Route: 
    vehicle_id: int 
    seq: List[int] # Order ID

@dataclass
class Solution:
    routes: List[Route] 

    @property
    def journeys(self) -> Dict[int, List[Route]]:
        j = {}
        for r in self.routes:
            j.setdefault(r.vehicle_id, []).append(r)
        return j