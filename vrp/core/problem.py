# FILE: vrp/core/problem.py
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import numpy as np

# Loại đơn hàng (Logic nội bộ)
ORDER_TYPE_DELIVERY = 0 
ORDER_TYPE_PICKUP = 1   

@dataclass
class Node:
    """
    Đại diện cho một điểm trên bản đồ (Depot hoặc Customer).
    Map từ bảng: 'depots' và 'customers'.
    """
    id: int
    code: str = ""
    name: str = ""
    address: str = ""
    
    # Tọa độ
    latitude: float = 0.0  # x
    longitude: float = 0.0 # y
    
    # Phân loại
    is_depot: bool = False
    dx_code: str = ""      # Mã định danh hệ thống ngoài
    
    # Thời gian hoạt động (giây)
    start_time: float = 0.0      # tw_open
    end_time: float = 86400.0    # tw_close
    
    # Chi phí đặc thù
    unloading_cost: float = 0.0  # Chỉ dành cho Depot
    penalty_cost: float = 0.0    # Chỉ dành cho Customer (phạt nếu vi phạm TW)
    
    # Meta data
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    @property
    def x(self): return self.latitude
    @property
    def y(self): return self.longitude
    @property
    def tw_open(self): return self.start_time
    @property
    def tw_close(self): return self.end_time


@dataclass
class GoodItem:
    """
    Chi tiết hàng hóa trong đơn hàng.
    Map từ bảng: 'order_items' JOIN 'products'.
    """
    product_id: int
    product_code: str = ""
    product_name: str = ""
    
    # Thuộc tính sản phẩm
    goods_group_id: int = 0      # Nhóm hàng
    weight: float = 0.0          # Tổng trọng lượng (quantity * unit_weight)
    capacity: float = 0.0        # Tổng thể tích (capacity trong DB)
    quantity: int = 1
    price: float = 0.0           # Giá trị hàng
    
    # Kích thước vật lý (từ bảng products)
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    # ID dòng trong order_items (nếu cần tracking)
    item_id: Optional[int] = None


@dataclass
class Order:
    """
    Yêu cầu vận chuyển.
    Map từ bảng: 'orders'.
    """
    id: int
    code: str = ""
    
    # Liên kết Node
    customer_id: int = 0   # node_id
    depot_id: Optional[int] = None # Kho chỉ định (nếu có)
    
    # Thuộc tính đơn hàng
    weight: float = 0.0    # Tổng trọng lượng
    capacity: float = 0.0  # Tổng thể tích
    order_value: float = 0.0
    
    # Thời gian (giây)
    time_service: float = 0.0  # Thời gian phục vụ
    time_loading: float = 0.0  # Thời gian bốc xếp
    
    # Khung giờ giao hàng yêu cầu
    delivery_after_time: float = 0.0   # tw_open của order
    delivery_before_time: float = 86400.0 # tw_close của order
    
    # Chế độ
    delivery_mode: str = "STANDARD"
    order_type: int = ORDER_TYPE_DELIVERY # Logic nội bộ (0: Giao, 1: Nhận)
    
    # Danh sách hàng hóa
    goods: List[GoodItem] = field(default_factory=list)
    
    dx_code: str = ""
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    @property
    def node_id(self): return self.customer_id
    @property
    def total_weight(self) -> float: return self.weight
    @property
    def total_volume(self) -> float: return self.capacity
    @property
    def service_duration(self) -> float: return self.time_service + self.time_loading
    @property
    def tw_open(self): return self.delivery_after_time
    @property
    def tw_close(self): return self.delivery_before_time
    @property
    def contained_product_ids(self) -> Set[int]:
        return {g.product_id for g in self.goods}


@dataclass
class Vehicle:
    """
    Phương tiện vận chuyển.
    Map từ bảng: 'vehicles'.
    """
    id: int
    name: str = "" # code/name
    type: str = "TRUCK" # TRUCK / BIKE
    driver_name: Optional[str] = None
    
    # Trạng thái
    available: bool = True
    
    # Tải trọng
    max_load_weight: float = 0.0 # capacity kg
    max_capacity: float = 0.0    # volume m3
    
    # Kích thước xe
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    # Vận tốc (km/h hoặc m/s tùy quy ước, DB thường là km/h)
    average_velocity: float = 0.0
    max_velocity: float = 0.0
    min_velocity: float = 0.0
    
    # Chi phí
    fixed_cost: float = 0.0             # Phí cố định
    average_fee_transport: float = 0.0  # Phí biến đổi theo khoảng cách (var_cost_per_dist)
    vehicle_cost: float = 0.0           # Giá trị xe/khấu hao
    
    # Nhiên liệu
    average_gas_consume: float = 0.0    # Lít/km?
    gas_price: float = 0.0              # Giá xăng
    
    # Logic lộ trình (Không có trong DB vehicles, cần set khi init problem)
    start_depot_id: int = 0
    end_depot_id: int = 0
    start_time: float = 0.0
    end_time: float = 86400.0
    
    dx_code: str = ""
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    # Ràng buộc
    allowed_goods_groups: Set[int] = field(default_factory=set)
    product_quantity_limits: Dict[int, int] = field(default_factory=dict) # Từ bảng vehicles_products

    @property
    def capacity(self): return self.max_load_weight
    @property
    def volume_capacity(self): return self.max_capacity
    @property
    def code(self): return self.name
    @property
    def var_cost_per_dist(self): return self.average_fee_transport


class Problem:
    def __init__(self, 
                 nodes: List[Node], 
                 orders: List[Order], 
                 vehicles: List[Vehicle],
                 distance_matrix: Dict[Tuple[int, int], float], 
                 time_matrix: Dict[Tuple[int, int], float],     
                 product_incompatibility: Set[Tuple[int, int]] = None 
                 ):
        
        self.nodes_map = {n.id: n for n in nodes}
        self.orders_map = {o.id: o for o in orders}
        self.vehicles = vehicles
        
        # Ma trận từ bảng correlations
        self.dist_matrix = distance_matrix 
        self.time_matrix = time_matrix
        
        # Cấu hình mặc định
        self.speed = 1.0 
        self.tw_penalty_weight = 1500.0 
        
        # Ràng buộc kỵ nhau từ bảng product_exclude
        self.product_incompatibility = product_incompatibility or set()

    def get_dist_node_to_node(self, u_node_id: int, v_node_id: int) -> float:
        return self.dist_matrix.get((u_node_id, v_node_id), float('inf'))

    def get_time_node_to_node(self, u_node_id: int, v_node_id: int) -> float:
        return self.time_matrix.get((u_node_id, v_node_id), float('inf'))

    def get_dist_order_to_order(self, u_order_id: int, v_order_id: int) -> float:
        u_node = self.orders_map[u_order_id].node_id
        v_node = self.orders_map[v_order_id].node_id
        return self.get_dist_node_to_node(u_node, v_node)
    
    def get_time_order_to_order(self, u_order_id: int, v_order_id: int) -> float:
        u_node = self.orders_map[u_order_id].node_id
        v_node = self.orders_map[v_order_id].node_id
        return self.get_time_node_to_node(u_node, v_node)