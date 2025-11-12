from typing import Tuple, Dict, Optional, Set
from .solution import Solution
from .problem_modified import Problem, Node
# from .problem import Problem, Node

def evaluate_modified(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    BIG = 1e6
    BIG_CAP = 1e5 
    speed = float(getattr(problem, "speed_units_per_min", 50.0))

    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed": 0.0,
        "tw_penalty": 0.0,
        "cap_violations": 0,
        "stockout_violations": 0,      # giao khi không đủ hàng trên xe
        "overflow_violations": 0,      # nhặt quá tải
        "overtime_routes": 0,
        "prohibited_uses": 0,
        "depot_passes": 0,
        "unserved_customers": 0,
        "pd_precedence_violations": 0, # giao trước khi pickup tương ứng
    }

    nodes: Dict[int, Node] = problem.nodes
    prohibited: Set[tuple] = getattr(problem, "prohibited", set())
    pd_reverse: Dict[int, tuple] = getattr(problem, "pd_reverse", {}) 
    pd_pairs: Dict[int, tuple] = getattr(problem, "pd_pairs", {}) 

    def add_leg(u: int, v: int, var_cost: float) -> float:
        nonlocal cost
        if (u, v) in prohibited:
            details["prohibited_uses"] += 1
            cost += BIG
        dist = problem.d(u, v)
        details["distance"] += dist
        cost += dist * var_cost
        return dist

    for route in sol.routes:
        veh = next(v for v in problem.vehicles if v.id == route.vehicle_id)
        Q = veh.capacity
        depot_id = veh.depot_id

        # chuẩn hoá route
        seq = list(route.seq)
        if not seq or seq[0] != depot_id:
            seq = [depot_id] + seq
        if seq[-1] != depot_id:
            seq = seq + [depot_id]

        has_customer = any(not nodes[n].is_depot for n in seq)
        if has_customer:
            details["fixed"] += veh.fixed_cost
            cost += veh.fixed_cost

        time_cur = veh.start_time
        load = 0
        visited_pickups = set()

        prev = seq[0]
        for i in range(1, len(seq)):
            cur = seq[i]
            nd = nodes[cur]

            # di chuyển
            dist = add_leg(prev, cur, veh.var_cost_per_dist)
            time_cur += dist / max(1e-9, speed)

            if nd.is_depot:
                if i not in (0, len(seq) - 1):
                    details["depot_passes"] += 1
            else:
                # time windows
                if (nd.tw_open is not None) and (time_cur < nd.tw_open):
                    time_cur = nd.tw_open
                if (nd.tw_close is not None) and (time_cur > nd.tw_close):
                    late = time_cur - nd.tw_close
                    details["tw_penalty"] += late * problem.tw_penalty_per_min
                    cost += late * problem.tw_penalty_per_min

                time_cur += nd.service_time

                if nd.demand_delivery > 0:
                    if cur in pd_reverse:
                        pickup_id, _qty = pd_reverse[cur]
                        if pickup_id not in visited_pickups:
                            details["pd_precedence_violations"] += 1
                            details["stockout_violations"] += 1
                            details["cap_violations"] += 1
                            cost += BIG_CAP
                        else:
                            load -= nd.demand_delivery
                            if load < 0:
                                details["cap_violations"] += 1
                                details["stockout_violations"] += 1
                                cost += BIG_CAP
                    else:
                        load -= nd.demand_delivery
                        if load < 0:
                            details["cap_violations"] += 1
                            details["stockout_violations"] += 1
                            cost += BIG_CAP

                if nd.demand_pickup > 0:
                    load += nd.demand_pickup
                    if load > Q:
                        details["cap_violations"] += 1
                        details["overflow_violations"] += 1
                        cost += BIG_CAP
                    if cur in pd_pairs or cur in getattr(problem, "pd_pairs", {}):
                        visited_pickups.add(cur)
                    else:
                        visited_pickups.add(cur)
            prev = cur
        if time_cur > veh.end_time:
            details["overtime_routes"] += 1
            cost += BIG_CAP

    visited_customers = set(sol.all_customers(nodes))
    unserved = [nid for nid, nd in nodes.items() if not nd.is_depot and nid not in visited_customers]
    details["unserved_customers"] = len(unserved)
    if unserved:
        cost += BIG * len(unserved)

    return (cost, details) if return_details else (cost, {})
