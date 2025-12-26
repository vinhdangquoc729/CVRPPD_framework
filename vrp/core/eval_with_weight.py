from typing import Tuple, Dict, Set
from collections import defaultdict
from .problem import Problem, Vehicle, Order, ORDER_TYPE_PICKUP, ORDER_TYPE_DELIVERY
from .solution import Solution

def evaluate_with_weight(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    # --- CẤU HÌNH TRỌNG SỐ ---
    W_DIST = 1.0      # Trọng số cho chi phí di chuyển (theo quãng đường)
    W_VEH = 10.0      # Trọng số cho chi phí cố định của xe (việc sử dụng xe)
    W_PENALTY = 1.0   # Trọng số cho các loại phạt vi phạm ràng buộc

    # Trọng số phạt cơ bản (giữ nguyên logic gốc)
    BIG = 1e6      # Lỗi logic nghiêm trọng
    BIG_CAP = 1e5  # Quá tải, quá giờ
    
    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed_cost": 0.0,
        "tw_penalty": 0.0,
        "capacity_violations": 0,    # Quá tải tổng
        "stockout_violations": 0,    # Không đủ loại hàng cụ thể để giao
        "goods_incompatibility": 0,  # Vi phạm hàng kỵ nhau (1&4)
        "goods_not_allowed": 0,      # Xe chở hàng không được phép
        "overtime_violations": 0,
        "unserved_orders": 0
    }

    # 1. Kiểm tra unserved orders
    all_order_ids = set(problem.orders_map.keys())
    served_order_ids = set()
    for route in sol.routes:
        served_order_ids.update(route.seq)
    
    unserved = all_order_ids - served_order_ids
    if unserved:
        details["unserved_orders"] = len(unserved)
        # Áp dụng trọng số phạt
        cost += (BIG * len(unserved)) * W_PENALTY

    journeys = sol.journeys

    for veh_id, route_list in journeys.items():
        try:
            veh: Vehicle = next(v for v in problem.vehicles if v.id == veh_id)
        except StopIteration:
            continue
        
        if any(r.seq for r in route_list):
            details["fixed_cost"] += veh.fixed_cost
            # Áp dụng trọng số xe
            cost += veh.fixed_cost * W_VEH

        current_time = veh.start_time
        current_loc_node = veh.start_depot_id 
        total_dist_journey = 0.0
        
        for route in route_list:
            if not route.seq: continue

            # --- SETUP START LOAD (LOGIC MỚI) ---
            # Tính toán nhu cầu thực tế của route này
            needed_per_type = defaultdict(float)
            total_needed_weight = 0.0
            
            # Duyệt trước route để xem cần giao những gì
            for order_id in route.seq:
                o = problem.orders_map[order_id]
                if o.order_type == ORDER_TYPE_DELIVERY:
                    for item in o.goods:
                        # Chỉ tính nếu xe được phép chở loại này
                        if item.goods_type in veh.allowed_goods_types:
                            needed_per_type[item.goods_type] += item.weight
                            total_needed_weight += item.weight
            
            # Khởi tạo kho hàng trên xe
            goods_stock = defaultdict(float)
            current_total_load = 0.0
            
            if total_needed_weight <= veh.capacity:
                # Case 1: Đủ tải -> Lấy đúng số lượng cần thiết
                for g_type, w in needed_per_type.items():
                    goods_stock[g_type] = w
                    current_total_load += w
            else:
                # Case 2: Quá tải -> Lấy theo tỷ lệ (Proportional)
                scale_factor = veh.capacity / total_needed_weight if total_needed_weight > 0 else 0
                for g_type, w in needed_per_type.items():
                    scaled_w = w * scale_factor
                    goods_stock[g_type] = scaled_w
                    current_total_load += scaled_w
            
            # --- Bắt đầu di chuyển ---
            first_order = problem.orders_map[route.seq[0]]
            dist_leg = problem.get_dist_node_to_node(current_loc_node, first_order.node_id)
            total_dist_journey += dist_leg
            current_time += dist_leg / problem.speed
            
            # Áp dụng trọng số khoảng cách
            cost += (dist_leg * veh.var_cost_per_dist) * W_DIST
            
            current_loc_node = first_order.node_id
            
            # Duyệt các Order
            for i, order_id in enumerate(route.seq):
                order = problem.orders_map[order_id]
                node = problem.nodes_map[order.node_id]

                if i > 0:
                    prev_order = problem.orders_map[route.seq[i-1]]
                    d = problem.get_dist_order_to_order(prev_order.id, order.id)
                    total_dist_journey += d
                    current_time += d / problem.speed
                    
                    # Áp dụng trọng số khoảng cách
                    cost += (d * veh.var_cost_per_dist) * W_DIST
                    
                    current_loc_node = order.node_id

                # Time Window
                tw_open = max(order.tw_open, node.tw_open)
                tw_close = min(order.tw_close, node.tw_close)
                if current_time < tw_open:
                    current_time = tw_open
                elif current_time > tw_close:
                    late = current_time - tw_close
                    penalty_val = late * problem.tw_penalty
                    details["tw_penalty"] += penalty_val
                    # Áp dụng trọng số phạt
                    cost += penalty_val * W_PENALTY

                # --- LOGIC HÀNG HÓA ---
                
                # 1. Check hàng cho phép của xe
                if not order.contained_goods_types.issubset(veh.allowed_goods_types):
                    details["goods_not_allowed"] += 1
                    cost += BIG * W_PENALTY

                # 2. Xử lý Pickup/Delivery
                if order.order_type == ORDER_TYPE_DELIVERY:
                    # Giao hàng: Giảm tải.
                    for item in order.goods:
                        # Kiểm tra tồn kho loại hàng cụ thể
                        if goods_stock[item.goods_type] < item.weight - 1e-9:
                            details["stockout_violations"] += 1
                            cost += BIG * W_PENALTY
                        
                        goods_stock[item.goods_type] -= item.weight
                        current_total_load -= item.weight
                    
                    if current_total_load < -1e-9: 
                         pass 

                elif order.order_type == ORDER_TYPE_PICKUP:
                    # Nhận hàng: Tăng tải.
                    added_weight = order.total_weight
                    if current_total_load + added_weight > veh.capacity + 1e-9:
                        details["capacity_violations"] += 1
                        cost += BIG_CAP * W_PENALTY
                    
                    current_total_load += added_weight
                    for item in order.goods:
                        goods_stock[item.goods_type] += item.weight

                # 3. Check xung đột hàng (1 vs 4)
                types_present = {gt for gt, w in goods_stock.items() if w > 1e-9}
                
                for incompatible_set in problem.incompatible_goods_pairs:
                    if incompatible_set.issubset(types_present):
                        details["goods_incompatibility"] += 1
                        cost += BIG * W_PENALTY

                current_time += order.service_time

            # --- Kết thúc Route: Về Depot ---
            next_dest_id = veh.start_depot_id 
            if route == route_list[-1]: # Chuyến cuối
                next_dest_id = veh.end_depot_id
            
            dist_return = problem.get_dist_node_to_node(current_loc_node, next_dest_id)
            total_dist_journey += dist_return
            current_time += dist_return / problem.speed
            
            # Áp dụng trọng số khoảng cách
            cost += (dist_return * veh.var_cost_per_dist) * W_DIST
            
            current_loc_node = next_dest_id
            
        # --- Kết thúc Journey ---
        details["distance"] += total_dist_journey
        
        if current_time > veh.end_time:
            details["overtime_violations"] += 1
            cost += BIG_CAP * W_PENALTY

    return (cost, details) if return_details else (cost, {})