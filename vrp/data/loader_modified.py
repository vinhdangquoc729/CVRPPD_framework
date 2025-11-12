# loader_modified.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
from ..core.problem_modified import Problem, Node, Vehicle


def _as_bool(x) -> bool:
    """Chuyển giá trị bất kỳ về bool, chấp nhận 1/0, 'true'/'false', 'yes'/'no'..."""
    if isinstance(x, bool):
        return x
    try:
        sx = str(x).strip().lower()
        if sx in ("1", "true", "t", "y", "yes"):
            return True
        if sx in ("0", "false", "f", "n", "no"):
            return False
    except Exception:
        pass
    return bool(x)


def _get_series_val(r: pd.Series, col: str, default=None):
    """Đọc trường từ một dòng DataFrame, trả default nếu thiếu hoặc NaN."""
    if col not in r.index:
        return default
    val = r[col]
    if pd.isna(val):
        return default
    return val


def load_problem_modified(folder: str | Path) -> Problem:
    """
    Load một instance từ thư mục chuẩn (nodes.csv, vehicles.csv, prohibited_arcs.csv?, orders.csv?, meta.json?)
    và trả về Problem (bản problem_modified).

    - Nếu có orders.csv: tạo pd_pairs {pickup_id: (delivery_id, quantity)}.
    - Nếu có prohibited_arcs.csv: tạo set cung cấm (i, j).
    - Đọc meta.json để lấy tw_penalty_per_min, travel_speed_units_per_min.
    """
    folder = Path(folder)

    nodes_df = pd.read_csv(folder / "nodes.csv")
    veh_df = pd.read_csv(folder / "vehicles.csv")

    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tw_pen = float(meta.get("tw_penalty_per_min", 1.0))
    speed = float(meta.get("travel_speed_units_per_min", 60.0))

    nodes: Dict[int, Node] = {}
    for _, r in nodes_df.iterrows():
        nid = int(r["id"])
        is_depot = _as_bool(_get_series_val(r, "is_depot", nid == 0))
        service_time = int(_get_series_val(r, "service_time", 0))

        tw_open = _get_series_val(r, "tw_open", None)
        tw_open = None if tw_open is None else int(tw_open)

        tw_close = _get_series_val(r, "tw_close", None)
        tw_close = None if tw_close is None else int(tw_close)

        cluster = _get_series_val(r, "cluster", None)
        cluster = None if cluster is None else int(cluster)

        nodes[nid] = Node(
            id=nid,
            x=float(r["x"]),
            y=float(r["y"]),
            demand_delivery=int(_get_series_val(r, "demand_delivery", 0)),
            demand_pickup=int(_get_series_val(r, "demand_pickup", 0)),
            is_depot=is_depot,
            service_time=service_time,
            tw_open=tw_open,
            tw_close=tw_close,
            cluster=cluster,
        )

    vehicles: List[Vehicle] = []
    for _, r in veh_df.iterrows():
        vehicles.append(
            Vehicle(
                id=int(r["vehicle_id"]),
                depot_id=int(r["depot_id"]),
                capacity=int(r["capacity"]),
                start_time=int(r["start_time"]),
                end_time=int(r["end_time"]),
                fixed_cost=float(r["fixed_cost"]),
                var_cost_per_dist=float(r["variable_cost_per_distance"]),
            )
        )

    prohib: Set[Tuple[int, int]] = set()
    prohib_path = folder / "prohibited_arcs.csv"
    if prohib_path.exists():
        pr = pd.read_csv(prohib_path)
        if not pr.empty and {"from_id", "to_id"}.issubset(pr.columns):
            for _, rr in pr.iterrows():
                prohib.add((int(rr["from_id"]), int(rr["to_id"])))

    pd_pairs: Dict[int, Tuple[int, int]] = {}
    orders_path = folder / "orders.csv"
    if orders_path.exists():
        od = pd.read_csv(orders_path)
        required = {"pickup_node_id", "delivery_node_id", "quantity"}
        if not od.empty and required.issubset(od.columns):
            for _, rr in od.iterrows():
                p = int(rr["pickup_node_id"])
                d = int(rr["delivery_node_id"])
                q = int(rr["quantity"])
                if p in nodes and d in nodes:
                    pd_pairs[p] = (d, q)

    prob = Problem(
        nodes=nodes,
        vehicles=vehicles,
        tw_penalty_per_min=tw_pen,
        speed_units_per_min=speed,
        prohibited=prohib,
        pd_pairs=pd_pairs if pd_pairs else None,
    )
    return prob

def load_problem_and_meta_modified(folder: str | Path) -> tuple[Problem, dict]:
    prob = load_problem_modified(folder)
    meta_path = Path(folder) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return prob, meta
