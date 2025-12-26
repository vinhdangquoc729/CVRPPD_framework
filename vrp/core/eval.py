from typing import Tuple, Dict, Set
from collections import defaultdict
from .problem import Problem, Vehicle, Order, ORDER_TYPE_PICKUP, ORDER_TYPE_DELIVERY
from .solution import Solution
def evaluate(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    # Trọng số phạt cho các ràng buộc cứng (Hard Constraints) không được vi phạm
    BIG_PENALTY = 1e9 
    
    total_transport_cost = 0.0
    details = {
        "distance": 0.0,
        "operating_cost": 0.0,
        "fixed_time_cost": 0.0,
        "penalty_cost": 0.0,  # Penalty do vi phạm Time Window (pc)
        "capacity_violations": 0,
        "volume_violations": 0,
        "incompatibility_violations": 0,
        "overtime_violations": 0, # Vi phạm giới hạn thời gian chạy của xe (vr_k)
        "unserved_orders": 0
    }

    # 1. Kiểm tra đơn hàng chưa được phục vụ
    all_order_ids = set(problem.orders_map.keys())
    served_order_ids = set()
    for route in sol.routes:
        served_order_ids.update(route.seq)
    
    unserved = all_order_ids - served_order_ids
    if unserved:
        details["unserved_orders"] = len(unserved)
        total_transport_cost += BIG_PENALTY * len(unserved)

    journeys = sol.journeys

    for veh_id, route_list in journeys.items():
        try:
            veh: Vehicle = next(v for v in problem.vehicles if v.id == veh_id)
        except StopIteration:
            continue
        
        # Nếu xe có thực hiện hành trình, cộng chi phí cố định (Time-based cost tc_k) 
        is_vehicle_used = any(r.seq for r in route_list)
        if is_vehicle_used:
            details["fixed_time_cost"] += veh.fixed_cost
            total_transport_cost += veh.fixed_cost

        current_time = veh.start_time
        current_loc_node = veh.start_depot_id 
        journey_distance = 0.0
        start_time_of_journey = veh.start_time
        
        for route in route_list:
            if not route.seq: continue
            
            # Kiểm tra tải trọng và thể tích đơn hàng trên route [cite: 253, 268]
            route_weight = sum(problem.orders_map[oid].weight for oid in route.seq)
            route_volume = sum(problem.orders_map[oid].capacity for oid in route.seq)
            
            if route_weight > veh.max_load_weight:
                details["capacity_violations"] += 1
                total_transport_cost += BIG_PENALTY
            
            if route_volume > veh.max_capacity:
                details["volume_violations"] += 1
                total_transport_cost += BIG_PENALTY

            # Duyệt qua các điểm trong route
            for oid in route.seq:
                order = problem.orders_map[oid]
                node = problem.nodes_map[order.node_id]

                # Tính chi phí vận hành (Operating cost oc_k) dựa trên quãng đường [cite: 224, 27]
                dist = problem.get_dist_node_to_node(current_loc_node, node.id)
                travel_time = problem.get_time_node_to_node(current_loc_node, node.id)
                
                journey_distance += dist
                current_time += travel_time
                
                op_cost = dist * veh.average_fee_transport
                details["operating_cost"] += op_cost
                total_transport_cost += op_cost

                # Kiểm tra vi phạm Time Window và tính Penalty Cost (pc) 
                # Công thức: pc = pc_k * delta_t
                if current_time > order.delivery_before_time:
                    delay_seconds = current_time - order.delivery_before_time
                    delay_minutes = delay_seconds / 60.0
                    # Theo báo: penalty_cost thường tính theo phút (VD: 1500 VND/phút) 
                    tw_penalty = delay_minutes * problem.tw_penalty_weight
                    details["penalty_cost"] += tw_penalty
                    total_transport_cost += tw_penalty
                
                # Nếu đến sớm hơn ES, xe phải chờ (không tính penalty nhưng tăng current_time) [cite: 177]
                if current_time < order.delivery_after_time:
                    current_time = order.delivery_after_time

                # Cộng thời gian phục vụ (sd_i) [cite: 189, 206]
                current_time += order.service_duration
                current_loc_node = node.id

            # Quay về Depot sau mỗi route [cite: 201]
            dist_to_depot = problem.get_dist_node_to_node(current_loc_node, veh.end_depot_id)
            time_to_depot = problem.get_time_node_to_node(current_loc_node, veh.end_depot_id)
            
            journey_distance += dist_to_depot
            current_time += time_to_depot
            
            op_cost_return = dist_to_depot * veh.average_fee_transport
            details["operating_cost"] += op_cost_return
            total_transport_cost += op_cost_return
            current_loc_node = veh.end_depot_id

        # Kiểm tra giới hạn thời gian chạy trong ngày của xe (vr_k) [cite: 155, 260]
        total_travel_time = current_time - start_time_of_journey
        if total_travel_time > veh.start_time + 28800: # Ví dụ 8 tiếng như trong báo [cite: 439]
            # Lưu ý: veh.vr_k nên được lấy từ problem model
            details["overtime_violations"] += 1
            total_transport_cost += BIG_PENALTY
            
        details["distance"] += journey_distance

    return (total_transport_cost, details) if return_details else (total_transport_cost, {})