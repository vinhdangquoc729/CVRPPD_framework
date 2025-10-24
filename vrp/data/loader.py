import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Set
from ..core.problem import Problem, Node, Vehicle


def load_problem(folder: str) -> Problem:
    folder = Path(folder)
    nodes_df = pd.read_csv(folder / "nodes.csv") # file toạ độ các điểm
    veh_df = pd.read_csv(folder / "vehicles.csv") # file thông tin các xe
    meta = json.loads((folder / "meta.json").read_text()) 
    tw_pen = float(meta.get("tw_penalty_per_min", 1.0)) # phạt vi phạm time window
    speed = float(meta.get("travel_speed_units_per_min", 50.0))

    nodes: Dict[int, Node] = {} # danh sách các điểm
    for _, r in nodes_df.iterrows():
        nodes[int(r.id)] = Node(
        id=int(r.id), x=float(r.x), y=float(r.y),
        demand_delivery=int(r.demand_delivery), demand_pickup=int(r.demand_pickup),
        is_depot=bool(r.is_depot) if "is_depot" in r else (int(r.id) == 0),
        service_time=int(r.service_time) if "service_time" in r else 0,
        tw_open=None if pd.isna(r.get("tw_open", None)) else int(r.tw_open),
        tw_close=None if pd.isna(r.get("tw_close", None)) else int(r.tw_close),
        cluster=None if pd.isna(r.get("cluster", None)) else int(r.cluster)
    )

    vehicles: List[Vehicle] = [] # danh sách các xe
    for _, r in veh_df.iterrows():
        vehicles.append(Vehicle(
            id=int(r.vehicle_id), depot_id=int(r.depot_id), capacity=int(r.capacity),
            start_time=int(r.start_time), end_time=int(r.end_time),
            fixed_cost=float(r.fixed_cost), var_cost_per_dist=float(r.variable_cost_per_distance)
        ))

    prohib: Set[tuple] = set() # các tuyến đường bị cấm
    prohib_path = folder / "prohibited_arcs.csv"
    if prohib_path.exists():
        pr = pd.read_csv(prohib_path)
        for _, r in pr.iterrows():
            prohib.add((int(r.from_id), int(r.to_id)))

    return Problem(nodes=nodes, vehicles=vehicles, tw_penalty_per_min=tw_pen, speed_units_per_min=speed, prohibited=prohib)