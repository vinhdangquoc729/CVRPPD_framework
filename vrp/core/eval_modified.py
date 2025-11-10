from typing import Tuple, Dict
from .problem import Problem, Node
from .solution import Solution

def evaluate_modified(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    BIG = 1e6      # phạt đi vào đường cấm
    BIG_CAP = 1e5  # phạt tải trọng quá giờ
    speed = getattr(problem, "speed_units_per_min", 50.0) 

    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed": 0.0,
        "tw_penalty": 0.0,
        "cap_violations": 0,        # tổng số lần vi phạm tải
        "stockout_violations": 0,   # thiếu hàng để giao
        "overflow_violations": 0,   # quá tải khi nhặt
        "overtime_routes": 0,       # số route vượt quá thời gian   
        "prohibited_uses": 0,       # số lần đi đường cấm
        "depot_passes": 0,          # đi qua depot giữa chừng
        "unserved_customers": 0     # số khách không được phục vụ
    }

    nodes: Dict[int, Node] = problem.nodes

    def add_leg(u: int, v: int, var_cost: float) -> float:
        nonlocal cost
        if (u, v) in problem.prohibited:
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

        # chuẩn hoá route: phải bắt đầu/kết thúc ở depot đúng của xe
        seq = list(route.seq)
        if not seq or seq[0] != depot_id:
            seq = [depot_id] + seq
        if seq[-1] != depot_id:
            seq = seq + [depot_id]

        # chỉ cộng fixed_cost khi có ít nhất 1 khách trong route
        has_customer = any(not nodes[n].is_depot for n in seq)
        if has_customer:
            details["fixed"] += veh.fixed_cost
            cost += veh.fixed_cost

        time = veh.start_time
        load = 0  # Khởi đầu không mang hàng

        prev = seq[0]
        for i in range(1, len(seq)):
            cur = seq[i]
            nd = nodes[cur]

            time += add_leg(prev, cur, veh.var_cost_per_dist) / max(1e-9, speed)

            if nd.is_depot:
                # Depot giữa chừng: không làm gì cả
                if i not in (0, len(seq)-1):
                    details["depot_passes"] += 1
            else:
                # Time windows
                # Sớm
                if (nd.tw_open is not None) and (time < nd.tw_open):
                    time = nd.tw_open
                # Muộn
                if (nd.tw_close is not None) and (time > nd.tw_close):
                    late = time - nd.tw_close
                    details["tw_penalty"] += late * problem.tw_penalty_per_min
                    cost += late * problem.tw_penalty_per_min

                # Service time
                time += nd.service_time

                # Delivery
                if nd.demand_delivery > 0:
                    load -= nd.demand_delivery
                    if load < 0:
                        details["cap_violations"] += 1
                        details["stockout_violations"] += 1
                        cost += BIG_CAP
                        load = 0

                # Pickup
                if nd.demand_pickup > 0:
                    load += nd.demand_pickup
                    if load > Q:
                        details["cap_violations"] += 1
                        details["overflow_violations"] += 1
                        cost += BIG_CAP
                        load = Q 
            prev = cur

        # Phạt quá giờ
        if time > veh.end_time:
            details["overtime_routes"] += 1
            cost += BIG_CAP

    unserved = [n for n, nd in problem.nodes.items() if not nd.is_depot and n not in sol.served_customers()]
    details["unserved_customers"] = len(unserved)
    if unserved:
        cost += BIG * len(unserved)
    return (cost, details) if return_details else (cost, {})
