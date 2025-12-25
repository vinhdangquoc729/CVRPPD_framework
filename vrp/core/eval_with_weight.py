from typing import Tuple, Dict, List, Set
from collections import defaultdict
from .problem import Problem, Node
from .solution import Solution

def evaluate_with_weight(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    # --- Định nghĩa các hệ số trọng số ---
    W_TRAVEL = 1.0
    W_FIXED = 10.0
    W_PENALTY = 1.0

    BIG = 1e6      # phạt lỗi logic (đường cấm, không liên tục,...)
    BIG_CAP = 1e5  # phạt quá tải / quá giờ
    speed = getattr(problem, "speed_units_per_min", 50.0)

    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed": 0.0,
        "tw_penalty": 0.0,
        "cap_violations": 0,
        "overflow_violations": 0,
        "underload_violations": 0,
        "overtime_routes": 0,
        "prohibited_uses": 0,
        "unserved_customers": 0,
        "continuity_errors": 0,
    }

    nodes = problem.nodes

    # 1. Kiểm tra khách hàng được phục vụ
    all_customers: Set[int] = {nid for nid, nd in nodes.items() if not nd.is_depot}
    served_customers_global: Set[int] = set()
    for route in sol.routes:
        for nid in route.seq:
            nd = nodes[nid]
            if not nd.is_depot:
                served_customers_global.add(nid)
    
    unserved = all_customers - served_customers_global
    if unserved:
        details["unserved_customers"] = len(unserved)
        # Áp dụng hệ số phạt cho khách hàng không được phục vụ
        cost += W_PENALTY * (BIG * len(unserved))

    def add_leg(u: int, v: int, var_cost: float) -> float:
        nonlocal cost
        if (u, v) in problem.prohibited:
            details["prohibited_uses"] += 1
            cost += W_PENALTY * BIG
        
        dist = problem.d(u, v)
        details["distance"] += dist
        # Áp dụng hệ số chi phí di chuyển
        cost += W_TRAVEL * (dist * var_cost)
        return dist

    journeys = sol.journeys

    for veh_id, route_list in journeys.items():
        try:
            veh = next(v for v in problem.vehicles if v.id == veh_id)
        except StopIteration:
            continue 
        
        Q = veh.capacity
        has_customer_in_journey = False
        for r in route_list:
            if any(not nodes[n].is_depot for n in r.seq):
                has_customer_in_journey = True
                break
        if has_customer_in_journey:
            details["fixed"] += veh.fixed_cost * W_FIXED
            # Áp dụng hệ số chi phí cố định xe (x10)
            cost += W_FIXED * veh.fixed_cost

        current_time = veh.start_time
        current_loc = veh.depot_id 

        for r_idx, route in enumerate(route_list):
            seq = list(route.seq)
            if not seq: continue

            if not nodes[seq[0]].is_depot:
                seq = [current_loc] + seq
            if not nodes[seq[-1]].is_depot:
                seq = seq + [veh.depot_id]

            if seq[0] != current_loc:
                details["continuity_errors"] += 1
                cost += W_PENALTY * BIG 
                dist_deadhead = add_leg(current_loc, seq[0], veh.var_cost_per_dist)
                current_time += dist_deadhead / speed

            total_delivery_demand = sum(nodes[nid].demand_delivery for nid in seq if not nodes[nid].is_depot)
            load_deliv = min(Q, total_delivery_demand)
            load_pick = 0

            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i+1]
                node_v = nodes[v]
                current_time += add_leg(u, v, veh.var_cost_per_dist) / speed

                if node_v.is_depot:
                    load_pick = 0
                    current_time += node_v.service_time
                    continue

                load_deliv -= node_v.demand_delivery
                if load_deliv < 0:
                    details["cap_violations"] += 1
                    details["underload_violations"] += 1
                    cost += W_PENALTY * BIG_CAP
                    load_deliv = 0
                
                load_pick += node_v.demand_pickup
                if load_deliv + load_pick > Q:
                    details["cap_violations"] += 1
                    details["overflow_violations"] += 1
                    cost += W_PENALTY * BIG_CAP
                    overflow = (load_deliv + load_pick) - Q
                    load_pick = max(0, load_pick - overflow)

                if (node_v.tw_open is not None) and (current_time < node_v.tw_open):
                    current_time = node_v.tw_open
                
                if (node_v.tw_close is not None) and (current_time > node_v.tw_close):
                    late = current_time - node_v.tw_close
                    penalty_val = late * problem.tw_penalty_per_min
                    details["tw_penalty"] += penalty_val
                    # Áp dụng hệ số phạt cho vi phạm khung thời gian
                    cost += W_PENALTY * penalty_val

                current_time += node_v.service_time

            current_loc = seq[-1]

        if current_time > veh.end_time:
            details["overtime_routes"] += 1
            cost += W_PENALTY * BIG_CAP

    return (cost, details) if return_details else (cost, {})