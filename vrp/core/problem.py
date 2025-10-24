from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Node:
    id: int
    x: float
    y: float # toạ độ x, y
    demand_delivery: int = 0 # nhu cầu giao hàng
    demand_pickup: int = 0 # nhu cầu lấy hàng
    is_depot: bool = False # điểm depot
    service_time: int = 0
    tw_open: Optional[int] = None # thời gian bắt đầu time window (phút)
    tw_close: Optional[int] = None # thời gian kết thúc time window (phút)
    cluster: Optional[int] = None 


@dataclass
class Vehicle:
    id: int
    depot_id: int # thuộc depot nào
    capacity: int # sức chứa xe
    start_time: int # thời gian bắt đầu làm việc (phút)
    end_time: int # thời gian kết thúc làm việc (phút)
    fixed_cost: float # chi phí của xe
    var_cost_per_dist: float # chi phí theo quãng đường


class Problem:
    def __init__(self, nodes: Dict[int, Node], vehicles: List[Vehicle], tw_penalty_per_min: float = 1.0, speed_units_per_min: float = 50.0,
        prohibited: Optional[set] = None):
        self.nodes = nodes # danh sách nút
        self.vehicles = vehicles # danh sách xe
        self.depots = [nid for nid, n in nodes.items() if n.is_depot] # danh sách depot
        self.customers = [nid for nid, n in nodes.items() if not n.is_depot] # danh sách khách hàng
        self.tw_penalty_per_min = tw_penalty_per_min # phạt vi phạm time window
        self.speed_units_per_min = float(speed_units_per_min)
        self.prohibited = prohibited or set() # các tuyến đường bị cấm
        ids = sorted(nodes.keys()) 
        self._id_index = {nid:i for i, nid in enumerate(ids)}
        coords = np.array([[nodes[i].x, nodes[i].y] for i in ids], dtype=float) 
        d = coords[:,None,:] - coords[None,:,:]
        self.dist = np.sqrt((d**2).sum(axis=2)) # ma trận khoảng cách các điểm
        self.ids = ids

    def d(self, i: int, j: int) -> float: # hàm tính khoảng cách giữa 2 điểm i, j
        return float(self.dist[self._id_index[i], self._id_index[j]]) 