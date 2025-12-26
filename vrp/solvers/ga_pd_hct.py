# vrp/solvers/ga_pd_hct.py
from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from itertools import groupby
from collections import defaultdict

from .solver_base import Solver
from ..core.problem import Problem, Order
from ..core.solution import Solution, Route
from ..core.eval import evaluate

# =========================
#   Encoding: Head–Core–Tail (HCT)
# =========================

@dataclass
class Head:
    priority: List[int]             # Thứ tự ưu tiên xe
    routes_per_veh: List[int]       # Số lượng route mỗi xe
    nodes_per_route: List[int]      # Số lượng node mỗi route (bao gồm depot)
    orders_per_node: List[int]      # Số lượng order tại mỗi node ghé thăm

@dataclass
class EncodedSolution:
    head: Head
    core: List[int]                 # Danh sách Node ID (phẳng)
    tail: List[int]                 # Danh sách Order ID (phẳng)

# =========================
#   Encode / Decode Logic
# =========================

def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    id2vidx = {v.id: i for i, v in enumerate(P.vehicles)}
    nV = len(P.vehicles)
    
    assignments: List[List[List[Tuple[int, List[int]]]]] = [[] for _ in range(nV)]
    
    for r in sol.routes:
        v_idx = id2vidx.get(r.vehicle_id)
        if v_idx is None: continue
        
        veh = P.vehicles[v_idx]
        route_struct = []
        route_struct.append((veh.start_depot_id, [])) # Depot đầu
        
        if r.seq:
            node_order_pairs = []
            for oid in r.seq:
                if oid in P.orders_map:
                    node_order_pairs.append((P.orders_map[oid].node_id, oid))
            
            for node_id, group in groupby(node_order_pairs, key=lambda x: x[0]):
                orders = [x[1] for x in group]
                route_struct.append((node_id, orders))
        
        route_struct.append((veh.end_depot_id, [])) # Depot cuối
        assignments[v_idx].append(route_struct)

    veh_order_counts = []
    for v_idx in range(nV):
        total_orders = sum(len(item[1]) for route in assignments[v_idx] for item in route)
        veh_order_counts.append((v_idx, total_orders))
    
    priority = [x[0] for x in sorted(veh_order_counts, key=lambda t: (-t[1], t[0]))]
    
    routes_per_veh = [0] * nV
    nodes_per_route = []
    orders_per_node = []
    core = []
    tail = []
    
    for v_idx in priority:
        routes = assignments[v_idx]
        routes_per_veh[v_idx] = len(routes)
        
        for route in routes:
            nodes_per_route.append(len(route))
            for node_id, orders in route:
                core.append(node_id)
                orders_per_node.append(len(orders))
                tail.extend(orders)
                
    head = Head(priority, routes_per_veh, nodes_per_route, orders_per_node)
    return EncodedSolution(head, core, tail)

def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    routes: List[Route] = []
    ptr_core, ptr_tail, ptr_node_counts, ptr_route_counts = 0, 0, 0, 0
    
    for v_idx in enc.head.priority:
        veh = P.vehicles[v_idx]
        num_routes = enc.head.routes_per_veh[v_idx]
        
        for _ in range(num_routes):
            num_nodes = enc.head.nodes_per_route[ptr_route_counts]
            ptr_route_counts += 1
            route_order_seq = []
            
            for _ in range(num_nodes):
                ptr_core += 1 # Skip Node ID
                num_orders = enc.head.orders_per_node[ptr_node_counts]
                ptr_node_counts += 1
                
                if num_orders > 0:
                    orders = enc.tail[ptr_tail : ptr_tail + num_orders]
                    route_order_seq.extend(orders)
                    ptr_tail += num_orders
            
            if route_order_seq:
                routes.append(Route(vehicle_id=veh.id, seq=route_order_seq))
                
    return Solution(routes=routes)

# =========================
#   GA Solver (Features: Guided Init + Route Move + Strict Goods Check)
# =========================

class GAPD_HCT_Solver(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 50, elite_frac: float = 0.1,
                 tournament_k: int = 2, p_cx: float = 0.9, p_mut: float = 0.2, p_route_mut: float = 0.2,
                 max_generations: int = 500, patience: int = 50, power_k: float = 1.0, evaluator: callable = None):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.p_cx = p_cx
        self.p_mut = p_mut
        self.p_route_mut = p_route_mut
        self.max_generations = max_generations
        self.patience = patience
        self.power_k = max(1.0, power_k)
        self.evaluator = evaluator if evaluator is not None else evaluate

        # Cache distances and vehicles for fast lookup
        self.veh_by_depot = defaultdict(list)
        for v in self.problem.vehicles:
            self.veh_by_depot[v.start_depot_id].append(v)
        self.all_depots = list(self.veh_by_depot.keys())

    def _quick_copy_enc(self, enc: EncodedSolution) -> EncodedSolution:
        h = enc.head
        new_head = Head(h.priority[:], h.routes_per_veh[:], h.nodes_per_route[:], h.orders_per_node[:])
        return EncodedSolution(new_head, [c for c in enc.core], [t for t in enc.tail])

    def _fitness(self, enc: EncodedSolution) -> float:
        sol = decode_to_solution(self.problem, enc)
        cost, _ = self.evaluator(self.problem, sol, return_details=False)
        return cost
        
    def _fitness_from_costs(self, costs: List[float]) -> List[float]:
        Amax, Amin = max(costs), min(costs)
        if abs(Amax - Amin) < 1e-12: return [1.0] * len(costs)
        return [((Amax - c) ** self.power_k) for c in costs]

    # ---------- Initialization: GUIDED + STRICT GOODS ----------

    def _init_population(self) -> List[EncodedSolution]:
        pop = []
        print("GA HCT: Initializing with Guided (Nearest Valid Vehicle) strategy...")
        for _ in range(self.pop_size):
            s = self._build_initial_solution_guided_strict()
            pop.append(encode_from_solution(self.problem, s))
        return pop

    def _build_initial_solution_guided_strict(self) -> Solution:
        """
        Khởi tạo có định hướng VÀ tuân thủ ràng buộc hàng hóa:
        1. Duyệt từng Order.
        2. Tìm các xe thỏa mãn (issubset).
        3. Trong các xe thỏa mãn, chọn xe ở Depot gần Order nhất.
        4. Sau khi gán xong, sắp xếp thứ tự đơn trong mỗi xe theo Nearest Neighbor.
        """
        P = self.problem
        routes_map: Dict[int, List[int]] = {v.id: [] for v in P.vehicles}
        
        all_orders = list(P.orders_map.values())
        self.rng.shuffle(all_orders) # Shuffle để tạo đa dạng trong quần thể

        for order in all_orders:
            order_types = order.contained_goods_types
            best_dist = float('inf')
            best_veh = None
            
            # --- Logic tìm xe tốt nhất (Gần nhất + Chở được) ---
            # Để tối ưu, ta duyệt qua các depot trước
            # (Giả định tất cả xe ở cùng 1 depot có tọa độ xuất phát giống nhau)
            
            # Sort depot theo khoảng cách tới Order
            # (Làm nhanh bằng cách chỉ check 1 xe đại diện mỗi depot để lấy tọa độ)
            
            # 1. Tìm tập hợp xe hợp lệ trên toàn mạng lưới
            valid_vehicles = []
            for v in P.vehicles:
                if order_types.issubset(v.allowed_goods_types):
                    valid_vehicles.append(v)
            
            if not valid_vehicles:
                # Fallback: Không xe nào chở được (Lỗi dữ liệu?), gán xe gần nhất
                # để code không crash, eval sẽ phạt.
                best_veh = min(P.vehicles, key=lambda v: P.get_dist_node_to_node(v.start_depot_id, order.node_id))
            else:
                # 2. Trong các xe hợp lệ, tìm xe ở depot gần nhất
                # Nhóm xe theo depot để cân bằng tải (Round Robin)
                valid_by_depot = defaultdict(list)
                for v in valid_vehicles:
                    valid_by_depot[v.start_depot_id].append(v)
                
                sorted_depots = sorted(valid_by_depot.keys(), 
                                     key=lambda d: P.get_dist_node_to_node(d, order.node_id))
                
                # Chọn depot gần nhất có xe hợp lệ
                target_depot = sorted_depots[0]
                candidates = valid_by_depot[target_depot]
                
                # Chọn xe trong candidates. Để cân bằng, chọn ngẫu nhiên hoặc xe ít việc nhất.
                # Ở đây chọn random cho đơn giản và đa dạng GA.
                best_veh = self.rng.choice(candidates)

            if best_veh:
                routes_map[best_veh.id].append(order.id)

        # Tạo Routes và sắp xếp NN
        sol_routes = []
        for v_id, order_ids in routes_map.items():
            if not order_ids: continue
            
            veh = next(v for v in P.vehicles if v.id == v_id)
            
            # Sắp xếp Nearest Neighbor cho danh sách order này
            sorted_orders = self._sort_orders_nn(veh.start_depot_id, order_ids)
            sol_routes.append(Route(v_id, sorted_orders))
            
        return Solution(routes=sol_routes)

    def _sort_orders_nn(self, depot_id: int, order_ids: List[int]) -> List[int]:
        """Sắp xếp Nearest Neighbor dựa trên vị trí Node của Order."""
        P = self.problem
        if not order_ids: return []
        
        # Map: OrderID -> NodeID
        o2n = {oid: P.orders_map[oid].node_id for oid in order_ids}
        
        unvisited = set(order_ids)
        ordered = []
        curr_node = depot_id
        
        while unvisited:
            # Tìm order có node gần curr_node nhất
            nxt_order = min(unvisited, key=lambda oid: P.get_dist_node_to_node(curr_node, o2n[oid]))
            ordered.append(nxt_order)
            unvisited.remove(nxt_order)
            curr_node = o2n[nxt_order]
            
        return ordered

    # --- CROSSOVER (STRICT GOODS CHECK) ---
    def _crossover(self, parent1: EncodedSolution, parent2: EncodedSolution) -> EncodedSolution:
        sol1 = decode_to_solution(self.problem, parent1)
        sol2 = decode_to_solution(self.problem, parent2)
        
        active_routes = [r for r in sol2.routes if r.seq]
        if not active_routes: return self._quick_copy_enc(parent1)
        
        # Chọn route donor từ mẹ
        r_donor = self.rng.choice(active_routes)
        orders_to_move = set(r_donor.seq)
        
        # Xóa các order này khỏi cha (sol1)
        new_routes_1 = []
        for r in sol1.routes:
            new_seq = [oid for oid in r.seq if oid not in orders_to_move]
            new_routes_1.append(Route(r.vehicle_id, new_seq))
            
        # Chèn vào sol1
        # Cố gắng chèn vào đúng xe đó (nếu xe đó ở sol1 cũng valid)
        target_v_id = r_donor.vehicle_id
        target_veh = next((v for v in self.problem.vehicles if v.id == target_v_id), None)
        
        # Kiểm tra tính tương thích (Phòng trường hợp mutation làm xe donor bị đổi mà goods ko đổi)
        # Lấy mẫu 1 order để check
        sample_order = self.problem.orders_map[r_donor.seq[0]]
        is_compatible = target_veh and sample_order.contained_goods_types.issubset(target_veh.allowed_goods_types)
        
        if not is_compatible:
            # Nếu xe gốc không chở được (hiếm), phải tìm xe khác trong sol1
            # Fallback: Trả về bản sao parent1 (abort crossover này)
            return self._quick_copy_enc(parent1)

        inserted = False
        for r in new_routes_1:
            if r.vehicle_id == target_v_id:
                r.seq.extend(list(r_donor.seq))
                inserted = True
                break
        if not inserted:
            new_routes_1.append(Route(target_v_id, list(r_donor.seq)))
            
        return encode_from_solution(self.problem, Solution(new_routes_1))

    # --- MUTATION 1: Standard (Swap/Move Order - STRICT CHECK) ---
    def _mutate(self, enc: EncodedSolution) -> EncodedSolution:
        sol = decode_to_solution(self.problem, enc)
        if not sol.routes: return enc
        
        # 1. Intra-route swap (Luôn an toàn vì cùng xe)
        if self.rng.random() < 0.5:
            active = [r for r in sol.routes if len(r.seq) >= 2]
            if active:
                r = self.rng.choice(active)
                i, j = self.rng.sample(range(len(r.seq)), 2)
                r.seq[i], r.seq[j] = r.seq[j], r.seq[i]
        
        # 2. Inter-route move (CẦN CHECK GOODS)
        else:
            active = [r for r in sol.routes if r.seq]
            route_map = {r.vehicle_id: r for r in sol.routes}
            all_vehs = self.problem.vehicles
            
            if active:
                r_src = self.rng.choice(active)
                if r_src.seq:
                    oid = r_src.seq[self.rng.randrange(len(r_src.seq))]
                    order = self.problem.orders_map[oid]
                    
                    # Tìm các xe đích hợp lệ (khác xe nguồn)
                    valid_dest_vehs = [v for v in all_vehs 
                                       if v.id != r_src.vehicle_id and 
                                       order.contained_goods_types.issubset(v.allowed_goods_types)]
                    
                    if valid_dest_vehs:
                        dest_veh = self.rng.choice(valid_dest_vehs)
                        
                        # Get destination route
                        if dest_veh.id in route_map:
                            r_dst = route_map[dest_veh.id]
                        else:
                            r_dst = Route(dest_veh.id, [])
                            sol.routes.append(r_dst)
                            
                        # Move
                        r_src.seq.remove(oid)
                        if not r_dst.seq: r_dst.seq.append(oid)
                        else: r_dst.seq.insert(self.rng.randrange(len(r_dst.seq)), oid)

        return encode_from_solution(self.problem, sol)

    # --- MUTATION 2: Move Route (STRICT CHECK) ---
    def _mutation_move_route(self, enc: EncodedSolution) -> EncodedSolution:
        """
        Chuyển toàn bộ 1 Route từ xe này sang xe khác.
        Chỉ chuyển nếu xe đích CHỞ ĐƯỢC tất cả hàng trong route đó.
        """
        sol = decode_to_solution(self.problem, enc)
        active_routes = [r for r in sol.routes if r.seq]
        if not active_routes: return enc
        
        route_to_move = self.rng.choice(active_routes)
        
        # Kiểm tra loại hàng của route này (Union tất cả orders)
        route_goods_types = set()
        for oid in route_to_move.seq:
            route_goods_types.update(self.problem.orders_map[oid].contained_goods_types)
            
        # Tìm các xe đích hợp lệ
        all_vehs = self.problem.vehicles
        current_v_id = route_to_move.vehicle_id
        
        candidates = [v for v in all_vehs 
                      if v.id != current_v_id and route_goods_types.issubset(v.allowed_goods_types)]
        
        if not candidates: return enc
        
        new_veh = self.rng.choice(candidates)
        route_to_move.vehicle_id = new_veh.id
        
        return encode_from_solution(self.problem, sol)

    def _save_final_population_details(self, population: List[EncodedSolution], filename: str = "final_pop_details.csv"):
        import pandas as pd
        import os
        all_records = []
        for i, enc in enumerate(population): 
            sol = decode_to_solution(self.problem, enc)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {"individual_id": i, "total_cost": cost, **details, "solution_str": str(sol)}
            all_records.append(record)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pd.DataFrame(all_records).to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        t0 = time.time()
        print("GA HCT: Initializing...")
        pop = self._init_population()
        
        fitness_vals = [self._fitness(ind) for ind in pop]
        best_idx = min(range(len(pop)), key=lambda i: fitness_vals[i])
        best_enc = self._quick_copy_enc(pop[best_idx])
        best_cost = fitness_vals[best_idx]
        
        print(f"GA HCT Start. Initial Best Cost: {best_cost:.2f}")
        
        gen, patience_counter = 0, 0
        
        while (time.time() - t0) < time_limit_sec and gen < self.max_generations:
            gen += 1
            new_pop = []
            
            # Elitism
            elite_count = max(1, int(self.pop_size * self.elite_frac))
            sorted_indices = sorted(range(len(pop)), key=lambda i: fitness_vals[i])
            for i in sorted_indices[:elite_count]:
                new_pop.append(self._quick_copy_enc(pop[i]))
                
            # Evolution
            while len(new_pop) < self.pop_size:
                p1_idx = min(self.rng.sample(range(len(pop)), self.tournament_k), key=lambda i: fitness_vals[i])
                p2_idx = min(self.rng.sample(range(len(pop)), self.tournament_k), key=lambda i: fitness_vals[i])
                p1, p2 = pop[p1_idx], pop[p2_idx]
                
                # Crossover
                if self.rng.random() < self.p_cx:
                    child = self._crossover(p1, p2)
                else:
                    child = self._quick_copy_enc(p1)
                
                # Mutation 1: Standard
                if self.rng.random() < self.p_mut:
                    child = self._mutate(child)
                    
                # Mutation 2: Move Route
                if self.rng.random() < self.p_route_mut:
                    child = self._mutation_move_route(child)
                    
                new_pop.append(child)
                
            pop = new_pop
            fitness_vals = [self._fitness(ind) for ind in pop]
            
            curr_best_idx = min(range(len(pop)), key=lambda i: fitness_vals[i])
            curr_best_cost = fitness_vals[curr_best_idx]
            
            if curr_best_cost < best_cost - 1e-9:
                best_cost = curr_best_cost
                best_enc = self._quick_copy_enc(pop[curr_best_idx])
                patience_counter = 0
            else:
                patience_counter += 1
                
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Cost = {best_cost:.2f}", end='\r')
                
            if patience_counter >= self.patience:
                print(f"\nStopping early at gen {gen} due to patience.")
                break
                
        print(f"\nGA HCT Finished. Best Cost: {best_cost:.2f}")
        self._save_final_population_details(pop, filename=f"last_generation/ga_hct_final_pop_seed{self.seed}_{len(self.problem.nodes_map)}_{self.evaluator.__name__}.csv")
        return decode_to_solution(self.problem, best_enc)