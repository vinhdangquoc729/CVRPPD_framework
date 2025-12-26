from typing import Tuple, Dict, List, Set
from .problem import Problem, Node
from .solution import Solution

def evaluate(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    # --- CÁC HẰNG SỐ THEO BÀI BÁO ---
    # Giờ hoạt động: 6:00 (360') đến 15:00 (900') 
    SERVICE_START = 360 
    SERVICE_END = 900
    
    # Giờ cao điểm: 8:00 (480') đến 10:00 (600') [cite: 112, 161]
    PEAK_START = 480
    PEAK_END = 600
    # Hệ số tăng chi phí/thời gian trong giờ cao điểm (Peak factor) [cite: 113, 162, 388]
    PEAK_MULTIPLIER = 1.3 

    BIG = 1e7      # Phạt lỗi nặng (đường cấm, không phục vụ hết khách)
    BIG_CAP = 1e5  # Phạt vi phạm tải trọng/thời gian
    
    # Tốc độ di chuyển mặc định (tỷ lệ 1:1 giữa khoảng cách và thời gian như bài báo)
    speed = getattr(problem, "travel_speed_units_per_min", 300.0)

    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed": 0.0,
        "tw_penalty": 0.0,
        "peak_extra_cost": 0.0, # CHỈNH SỬA: Theo dõi chi phí phát sinh do giờ cao điểm
        "cap_violations": 0,
        "overflow_violations": 0,
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
            if nid in nodes and not nodes[nid].is_depot:
                served_customers_global.add(nid)
    
    unserved = all_customers - served_customers_global
    if unserved:
        details["unserved_customers"] = len(unserved)
        cost += BIG * len(unserved)

    # --- CHỈNH SỬA HÀM ADD_LEG ĐỂ TÍNH GIỜ CAO ĐIỂM ---
    def add_leg(u: int, v: int, var_cost: float, start_time: float) -> Tuple[float, float]:
        """Trả về (khoảng cách tính phí, thời gian di chuyển)"""
        nonlocal cost
        
        # Kiểm tra đường cấm [cite: 114, 182]
        if (u, v) in problem.prohibited:
            details["prohibited_uses"] += 1
            cost += BIG
            
        base_dist = problem.d(u, v) # Chi phí không đối xứng d(u,v) != d(v,u) 
        
        # Kiểm tra nếu khởi hành trong giờ cao điểm 
        is_peak = PEAK_START <= start_time <= PEAK_END
        multiplier = PEAK_MULTIPLIER if is_peak else 1.0
        
        actual_dist = base_dist * multiplier
        actual_time = actual_dist / speed
        
        if is_peak:
            details["peak_extra_cost"] += (actual_dist - base_dist) * var_cost

        details["distance"] += actual_dist
        cost += actual_dist * var_cost
        
        return actual_dist, actual_time

    journeys = sol.journeys

    for veh_id, route_list in journeys.items():
        try:
            veh = next(v for v in problem.vehicles if v.id == veh_id)
        except StopIteration:
            continue 
        
        Q = veh.capacity
        
        # CHỈNH SỬA: Giờ bắt đầu mặc định là 6:00 sáng theo bài báo 
        current_time = max(veh.start_time, SERVICE_START)
        current_loc = veh.depot_id 
        
        has_customer = any(any(not nodes[n].is_depot for n in r.seq) for r in route_list)
        if has_customer:
            details["fixed"] += veh.fixed_cost
            cost += veh.fixed_cost

        for route in route_list:
            seq = list(route.seq)
            if not seq: continue

            # Đảm bảo lộ trình bắt đầu/kết thúc tại Depot
            if not nodes[seq[0]].is_depot: seq = [current_loc] + seq
            if not nodes[seq[-1]].is_depot: seq = seq + [veh.depot_id]

            if seq[0] != current_loc:
                details["continuity_errors"] += 1
                cost += BIG
            
            # --- LOGIC GIAO-NHẬN ĐỒNG THỜI (SPDP) ---
            # z_ij: Lượng hàng cần giao cho các khách còn lại [cite: 195]
            # y_ij: Lượng hàng đã thu gom tích lũy [cite: 194]
            route_customers = [nid for nid in seq if not nodes[nid].is_depot]
            load_delivery = sum(nodes[nid].demand_delivery for nid in route_customers)
            load_pickup = 0

            # Kiểm tra tải trọng ngay lúc rời Depot (chỉ có hàng giao)
            if load_delivery > Q:
                details["cap_violations"] += 1
                cost += BIG_CAP

            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i+1]
                node_v = nodes[v]

                # Tính di chuyển (có tính đến giờ cao điểm)
                _, travel_time = add_leg(u, v, veh.var_cost_per_dist, current_time)
                current_time += travel_time

                if node_v.is_depot:
                    current_time += node_v.service_time
                    continue

                # Xử lý tại điểm khách 
                load_delivery -= node_v.demand_delivery
                load_pickup += node_v.demand_pickup
                
                # Ràng buộc tải trọng đồng thời: Hàng giao còn lại + Hàng thu gom <= Q 
                if load_delivery + load_pickup > Q:
                    details["cap_violations"] += 1
                    details["overflow_violations"] += 1
                    cost += BIG_CAP

                # Time Windows (Khung giờ phục vụ riêng của khách nếu có)
                if (node_v.tw_open is not None) and (current_time < node_v.tw_open):
                    current_time = node_v.tw_open
                
                if (node_v.tw_close is not None) and (current_time > node_v.tw_close):
                    late = current_time - node_v.tw_close
                    details["tw_penalty"] += late * problem.tw_penalty_per_min
                    cost += late * problem.tw_penalty_per_min

                current_time += node_v.service_time

            current_loc = seq[-1]

        # CHỈNH SỬA: Kiểm tra giờ về kho muộn hơn 15:00 
        max_end_time = min(veh.end_time, SERVICE_END)
        if current_time > max_end_time:
            details["overtime_routes"] += 1
            cost += BIG_CAP * (current_time - max_end_time)

    return (cost, details) if return_details else (cost, {})