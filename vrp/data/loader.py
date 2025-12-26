# FILE: vrp/data/loader.py
import csv
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# Đảm bảo đường dẫn import đúng với cấu trúc thư mục của bạn
from vrp.core.problem import (
    Problem, Node, Order, Vehicle, GoodItem, 
    ORDER_TYPE_DELIVERY, ORDER_TYPE_PICKUP
)

def safe_float(val, default=0.0) -> float:
    if not val or str(val).lower() == 'null' or val == '':
        return default
    try:
        return float(val)
    except ValueError:
        return default

def safe_int(val, default=0) -> int:
    if not val or str(val).lower() == 'null' or val == '':
        return default
    try:
        return int(float(val))
    except ValueError:
        return default

def safe_str(val, default="") -> str:
    if not val or str(val).lower() == 'null':
        return default
    return str(val).strip()

def load_problem(csv_dir: str = "data_processed") -> Problem:
    """
    Khởi tạo đối tượng Problem đầy đủ nhất bằng cách đọc các file CSV 
    được chuyển đổi từ hệ quản trị CSDL trunglhavnist.
    """
    print(f"--- Đang tải dữ liệu logistics toàn phần từ: {csv_dir} ---")

    # 1. Tải bảng Products (Thông tin vật lý và nhóm hàng)
    product_data_map = {}
    products_path = os.path.join(csv_dir, 'products.csv')
    if os.path.exists(products_path):
        with open(products_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = safe_int(row['id'])
                product_data_map[pid] = {
                    'code': safe_str(row['code']),
                    'name': safe_str(row['name']),
                    'goods_group_id': safe_int(row['goods_group_id']),
                    'price': safe_float(row['price']),
                    'weight_unit': safe_float(row['weight']),
                    'capacity_unit': safe_float(row['capacity']),
                    'length': safe_float(row['length']),
                    'width': safe_float(row['width']),
                    'height': safe_float(row['height'])
                }

    # 2. Tải bảng Nodes (Depots & Customers)
    nodes_list: List[Node] = []
    
    # 2a. Đọc Depots (Kho hàng)
    depots_path = os.path.join(csv_dir, 'depots.csv')
    if os.path.exists(depots_path):
        with open(depots_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nodes_list.append(Node(
                    id=safe_int(row['id']),
                    code=safe_str(row['code']),
                    name=safe_str(row['name']),
                    address=safe_str(row['address']),
                    latitude=safe_float(row['latitude']),
                    longitude=safe_float(row['longitude']),
                    is_depot=True,
                    start_time=safe_float(row['start_time']),
                    end_time=safe_float(row['end_time']),
                    unloading_cost=safe_float(row['unloading_cost']),
                    dx_code=safe_str(row['dx_code']),
                    created_at=safe_int(row.get('created_at')),
                    updated_at=safe_int(row.get('updated_at'))
                ))

    # 2b. Đọc Customers (Khách hàng)
    customers_path = os.path.join(csv_dir, 'customers.csv')
    if os.path.exists(customers_path):
        with open(customers_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nodes_list.append(Node(
                    id=safe_int(row['id']),
                    code=safe_str(row['code']),
                    name=safe_str(row['name']),
                    address=safe_str(row['address']),
                    latitude=safe_float(row['latitude']),
                    longitude=safe_float(row['longitude']),
                    is_depot=False,
                    start_time=safe_float(row['start_time']),
                    end_time=safe_float(row['end_time']),
                    penalty_cost=safe_float(row['penalty_cost']),
                    dx_code=safe_str(row['dx_code']),
                    created_at=safe_int(row.get('created_at')),
                    updated_at=safe_int(row.get('updated_at'))
                ))

    # 3. Tải bảng Order Items (Chi tiết các mặt hàng trong đơn)
    order_items_map = defaultdict(list)
    items_path = os.path.join(csv_dir, 'order_items.csv')
    if os.path.exists(items_path):
        with open(items_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = safe_int(row['order_id'])
                pid = safe_int(row['product_id'])
                p_info = product_data_map.get(pid, {})
                
                item = GoodItem(
                    item_id=safe_int(row['id']),
                    product_id=pid,
                    product_code=p_info.get('code', ""),
                    product_name=p_info.get('name', ""),
                    goods_group_id=p_info.get('goods_group_id', 0),
                    quantity=safe_int(row['quantity']),
                    weight=safe_float(row['weight']),     # Trọng lượng tổng dòng này
                    capacity=safe_float(row['capacity']), # Thể tích tổng dòng này
                    price=safe_float(row['price']),
                    length=p_info.get('length', 0.0),
                    width=p_info.get('width', 0.0),
                    height=p_info.get('height', 0.0)
                )
                order_items_map[oid].append(item)

    # 4. Tải bảng Orders (Yêu cầu giao hàng)
    orders_list: List[Order] = []
    orders_path = os.path.join(csv_dir, 'orders.csv')
    if os.path.exists(orders_path):
        with open(orders_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = safe_int(row['id'])
                orders_list.append(Order(
                    id=oid,
                    code=safe_str(row['code']),
                    customer_id=safe_int(row['customer_id']),
                    depot_id=safe_int(row.get('depot_id')) if row.get('depot_id') else None,
                    weight=safe_float(row['weight']),
                    capacity=safe_float(row['capacity']),
                    order_value=safe_float(row['order_value']),
                    time_service=safe_float(row['time_service']),
                    time_loading=safe_float(row['time_loading']),
                    delivery_after_time=safe_float(row['delivery_after_time']),
                    delivery_before_time=safe_float(row['delivery_before_time']),
                    delivery_mode=safe_str(row['delivery_mode'], "STANDARD"),
                    order_type=ORDER_TYPE_DELIVERY,
                    goods=order_items_map.get(oid, []),
                    dx_code=safe_str(row['dx_code']),
                    created_at=safe_int(row.get('created_at')),
                    updated_at=safe_int(row.get('updated_at'))
                ))

    # 5. Tải bảng Vehicles & Vehicles_Products (Đội xe và giới hạn hàng hóa)
    # 5a. Giới hạn hàng hóa theo từng xe
    veh_prod_limits = defaultdict(dict)
    vp_path = os.path.join(csv_dir, 'vehicles_products.csv')
    if os.path.exists(vp_path):
        with open(vp_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = safe_int(row['vehicle_id'])
                pid = safe_int(row['product_id'])
                veh_prod_limits[vid][pid] = safe_int(row['maxNumber'])

    # 5b. Thông tin chi tiết đội xe
    vehicles_list: List[Vehicle] = []
    veh_path = os.path.join(csv_dir, 'vehicles.csv')
    if os.path.exists(veh_path):
        with open(veh_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = safe_int(row['id'])
                # SQL mẫu không chỉ định kho xuất phát cụ thể, giả định mặc định ID 1
                vehicles_list.append(Vehicle(
                    id=vid,
                    name=safe_str(row['name']),
                    type=safe_str(row['type'], "TRUCK"),
                    available=str(row['available']).lower() == 'true',
                    max_load_weight=safe_float(row['max_load_weight']),
                    max_capacity=safe_float(row['max_capacity']),
                    length=safe_float(row['length']),
                    width=safe_float(row['width']),
                    height=safe_float(row['height']),
                    average_velocity=safe_float(row['average_velocity']),
                    max_velocity=safe_float(row['max_velocity']),
                    min_velocity=safe_float(row['min_velocity']),
                    fixed_cost=safe_float(row['fixed_cost']),
                    average_fee_transport=safe_float(row['average_fee_transport']),
                    vehicle_cost=safe_float(row['vehicle_cost']),
                    average_gas_consume=safe_float(row['average_gas_consume']),
                    gas_price=safe_float(row['gas_price']),
                    start_depot_id=1, 
                    end_depot_id=1,
                    start_time=25200.0, # 07:00:00
                    end_time=75600.0,   # 21:00:00
                    product_quantity_limits=veh_prod_limits.get(vid, {}),
                    allowed_goods_groups={8, 9, 10, 11, 12, 13, 15, 16}, # Mặc định cho phép tất cả
                    dx_code=safe_str(row['dx_code']),
                    created_at=safe_int(row.get('created_at')),
                    updated_at=safe_int(row.get('updated_at'))
                ))

    # 6. Tải bảng Correlations (Ma trận khoảng cách và thời gian thực)
    dist_matrix = {}
    time_matrix = {}
    corr_path = os.path.join(csv_dir, 'correlations.csv')
    if os.path.exists(corr_path):
        with open(corr_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = safe_int(row['from_node_id'])
                v = safe_int(row['to_node_id'])
                dist_matrix[(u, v)] = safe_float(row['distance'])
                time_matrix[(u, v)] = safe_float(row['time'])

    # 7. Tải bảng Product_Exclude (Ràng buộc hàng hóa kỵ nhau)
    prod_incomp: Set[Tuple[int, int]] = set()
    excl_path = os.path.join(csv_dir, 'product_exclude.csv')
    if os.path.exists(excl_path):
        with open(excl_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p1 = safe_int(row['product_excluding_id'])
                p2 = safe_int(row['excluded_product_id'])
                prod_incomp.add((p1, p2))

    # 8. Hoàn tất đóng gói đối tượng Problem
    print(f"-> Hoàn tất: {len(nodes_list)} địa điểm, {len(orders_list)} đơn hàng, {len(vehicles_list)} xe.")
    
    return Problem(
        nodes=nodes_list,
        orders=orders_list,
        vehicles=vehicles_list,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        product_incompatibility=prod_incomp
    )

if __name__ == "__main__":
    prob = load_problem("data_processed")
    print("\n--- TEST KIỂM TRA DỮ LIỆU ---")
    print(len(prob.orders_map))
    if prob.orders_map:
        o = prob.orders_map
        print(f"Đơn hàng {o.code}: Nặng {o.weight}kg, Có {len(o.goods)} mặt hàng.")
        for item in o.goods:
            print(f"  - Sản phẩm {item.product_name} (Nhóm {item.goods_group_id})")
    
    if prob.vehicles:
        v = prob.vehicles[-1]
        print(f"Xe {v.name}: Giới hạn đặc biệt cho {len(v.product_quantity_limits)} sản phẩm.")