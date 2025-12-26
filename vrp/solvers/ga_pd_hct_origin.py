# vrp/solvers/ga_pd_hct_origin.py
from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from itertools import groupby

from .solver_base import Solver
from ..core.problem import Problem, Order
from ..core.solution import Solution, Route
from ..core.eval import evaluate

# =========================
#   Encoding: Head–Core–Tail (Paper Table 2)
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
    core: List[int]                 # Danh sách Node ID (depot + customers)
    tail: List[int]                 # Danh sách Order ID

# =========================
#   Encode / Decode Logic
# =========================

def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    # 1. Map Vehicle ID -> Index
    id2vidx = {v.id: i for i, v in enumerate(P.vehicles)}
    nV = len(P.vehicles)
    
    # Cấu trúc tạm để build: assignments[v_idx] = [Route1_Nodes, Route2_Nodes, ...]
    # Trong đó Route_Nodes = list of (NodeID, [OrderIDs])
    assignments: List[List[List[Tuple[int, List[int]]]]] = [[] for _ in range(nV)]
    
    # 2. Duyệt qua các Route trong Solution
    for r in sol.routes:
        v_idx = id2vidx.get(r.vehicle_id)
        if v_idx is None: continue
        
        veh = P.vehicles[v_idx]
        start_depot = veh.start_depot_id
        end_depot = veh.end_depot_id
        
        route_struct = []
        route_struct.append((start_depot, [])) # Depot đầu
        
        if r.seq:
            node_order_pairs = []
            for oid in r.seq:
                # Map từ Order ID -> Node ID để nhóm lại
                if oid in P.orders_map:
                    node_order_pairs.append((P.orders_map[oid].node_id, oid))
            
            # Groupby theo NodeID
            for node_id, group in groupby(node_order_pairs, key=lambda x: x[0]):
                orders = [x[1] for x in group]
                route_struct.append((node_id, orders))
        
        route_struct.append((end_depot, [])) # Depot cuối
        assignments[v_idx].append(route_struct)

    # 3. Xây dựng Head - Core - Tail
    veh_order_counts = []
    for v_idx in range(nV):
        total_orders = sum(len(item[1]) for route in assignments[v_idx] for item in route)
        veh_order_counts.append((v_idx, total_orders))
    
    # Priority sort
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
                
    head = Head(
        priority=priority,
        routes_per_veh=routes_per_veh,
        nodes_per_route=nodes_per_route,
        orders_per_node=orders_per_node
    )
    
    return EncodedSolution(head, core, tail)

def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    routes: List[Route] = []
    
    ptr_core = 0
    ptr_tail = 0
    ptr_node_counts = 0 
    ptr_route_counts = 0 
    
    for v_idx in enc.head.priority:
        veh = P.vehicles[v_idx]
        num_routes = enc.head.routes_per_veh[v_idx]
        
        for _ in range(num_routes):
            num_nodes = enc.head.nodes_per_route[ptr_route_counts]
            ptr_route_counts += 1
            
            route_order_seq = []
            
            for _ in range(num_nodes):
                # node_id = enc.core[ptr_core]
                ptr_core += 1
                
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
#   GA (Random Valid Init)
# =========================

class GAPD_HCT_ORIGIN_Solver(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 50, elite_frac: float = 0.1,
                 tournament_k: int = 2, p_cx: float = 0.9, p_mut: float = 0.2, p_route_mut: float = 0.2,
                 max_generations: int = 500, patience: int = 50, power_k: float = 1.0, evaluator: callable = None):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size, self.elite_frac, self.tournament_k = pop_size, elite_frac, tournament_k
        self.p_cx, self.p_mut, self.p_route_mut = p_cx, p_mut, p_route_mut
        self.max_generations, self.patience, self.power_k = max_generations, patience, max(1.0, power_k)
        self.evaluator = evaluator if evaluator is not None else evaluate

    def _quick_copy_enc(self, enc: EncodedSolution) -> EncodedSolution:
        h = enc.head
        new_head = Head(h.priority[:], h.routes_per_vehicle[:], h.nodes_per_route[:], h.orders_per_node[:])
        return EncodedSolution(new_head, [r[:] for r in enc.core_routes], [t[:] for t in enc.tail_orders])

    def _fitness(self, enc: EncodedSolution) -> float:
        sol = decode_to_solution(self.problem, enc)
        cost, _ = self.evaluator(self.problem, sol, return_details=False)
        return cost

    # ---------- Initialization (UPDATED: RANDOM VALID) ----------

    def _init_population(self) -> List[EncodedSolution]:
        pop = []
        print("GA HCT: Initializing population with Random Valid Assignment...")
        for _ in range(self.pop_size):
            s = self._build_initial_solution_random_valid()
            pop.append(encode_from_solution(self.problem, s))
        return pop

    def _build_initial_solution_random_valid(self) -> Solution:
        """
        Khởi tạo NGẪU NHIÊN nhưng HỢP LỆ (về loại hàng):
        1. Duyệt từng đơn hàng (shuffle).
        2. Tìm TẤT CẢ các xe có thể chở được loại hàng đó.
        3. Chọn NGẪU NHIÊN 1 xe trong số các xe hợp lệ (bất kể depot nào).
        """
        P = self.problem
        routes_map: Dict[int, List[int]] = {v.id: [] for v in P.vehicles}
        all_vehicles = P.vehicles
        
        # 1. Lấy và xáo trộn đơn hàng
        all_orders = list(P.orders_map.values())
        self.rng.shuffle(all_orders)

        # 2. Gán đơn hàng
        for order in all_orders:
            order_types = order.contained_goods_types
            
            # --- Tìm tất cả xe hợp lệ ---
            valid_vehicles = [
                v for v in all_vehicles
                if order_types.issubset(v.allowed_goods_types)
            ]
            
            target_vehicle = None
            
            if valid_vehicles:
                # --- CHỌN NGẪU NHIÊN TRONG SỐ XE HỢP LỆ ---
                target_vehicle = self.rng.choice(valid_vehicles)
            else:
                # Fallback: Nếu không xe nào chở được, chọn đại random bất kỳ xe nào
                target_vehicle = self.rng.choice(all_vehicles)
            
            if target_vehicle:
                routes_map[target_vehicle.id].append(order.id)

        # 3. Tạo Solution
        sol_routes = []
        for v in P.vehicles:
            seq = routes_map.get(v.id, [])
            if seq:
                sol_routes.append(Route(v.id, seq))
        
        return Solution(routes=sol_routes)

    # --- CROSSOVER ---
    def _crossover(self, parent1: EncodedSolution, parent2: EncodedSolution) -> EncodedSolution:
        sol1 = decode_to_solution(self.problem, parent1)
        sol2 = decode_to_solution(self.problem, parent2)
        
        non_empty_routes_2 = [r for r in sol2.routes if r.seq]
        if not non_empty_routes_2:
            return encode_from_solution(self.problem, sol1)
            
        r_donor = self.rng.choice(non_empty_routes_2)
        orders_to_move = set(r_donor.seq)
        
        new_routes_1 = []
        for r in sol1.routes:
            new_seq = [oid for oid in r.seq if oid not in orders_to_move]
            new_routes_1.append(Route(r.vehicle_id, new_seq))
            
        target_v_id = r_donor.vehicle_id
        inserted = False
        for r in new_routes_1:
            if r.vehicle_id == target_v_id:
                r.seq.extend(list(r_donor.seq)) 
                inserted = True
                break
        
        if not inserted:
            new_routes_1.append(Route(target_v_id, list(r_donor.seq)))
            
        child_sol = Solution(new_routes_1)
        return encode_from_solution(self.problem, child_sol)

    # --- MUTATION ---
    def _mutate(self, enc: EncodedSolution) -> EncodedSolution:
        sol = decode_to_solution(self.problem, enc)
        if not sol.routes: return enc
        
        # 1. Intra-route swap
        if self.rng.random() < 0.5:
            active_routes = [r for r in sol.routes if len(r.seq) >= 2]
            if active_routes:
                r = self.rng.choice(active_routes)
                i, j = self.rng.sample(range(len(r.seq)), 2)
                r.seq[i], r.seq[j] = r.seq[j], r.seq[i]
                
        # 2. Inter-route move
        else:
            active_routes = [r for r in sol.routes if len(r.seq) > 0]
            all_vehicle_ids = [v.id for v in self.problem.vehicles]
            route_map = {r.vehicle_id: r for r in sol.routes}
            
            if active_routes:
                r1 = self.rng.choice(active_routes)
                dest_v_id = self.rng.choice(all_vehicle_ids)
                
                if dest_v_id not in route_map:
                    r2 = Route(dest_v_id, [])
                    sol.routes.append(r2)
                    route_map[dest_v_id] = r2
                else:
                    r2 = route_map[dest_v_id]
                
                if r1.seq:
                    idx = self.rng.randrange(len(r1.seq))
                    oid = r1.seq.pop(idx)
                    if not r2.seq: r2.seq.append(oid)
                    else: r2.seq.insert(self.rng.randint(0, len(r2.seq)), oid)
                    
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
        print("Initializing GA Population (HCT Encoding)...")
        pop = self._init_population()
        
        fitness_vals = [self._fitness(ind) for ind in pop]
        best_idx = min(range(len(pop)), key=lambda i: fitness_vals[i])
        best_enc = pop[best_idx] 
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
                new_pop.append(pop[i])
                
            # Evolution
            while len(new_pop) < self.pop_size:
                p1_idx = min(self.rng.sample(range(len(pop)), self.tournament_k), key=lambda i: fitness_vals[i])
                p2_idx = min(self.rng.sample(range(len(pop)), self.tournament_k), key=lambda i: fitness_vals[i])
                p1, p2 = pop[p1_idx], pop[p2_idx]
                
                if self.rng.random() < self.p_cx:
                    child = self._crossover(p1, p2)
                else:
                    child = p1 
                
                if self.rng.random() < self.p_mut:
                    child = self._mutate(child)
                    
                new_pop.append(child)
                
            pop = new_pop
            fitness_vals = [self._fitness(ind) for ind in pop]
            
            curr_best_idx = min(range(len(pop)), key=lambda i: fitness_vals[i])
            curr_best_cost = fitness_vals[curr_best_idx]
            
            if curr_best_cost < best_cost:
                best_cost = curr_best_cost
                best_enc = pop[curr_best_idx]
                patience_counter = 0
            else:
                patience_counter += 1
                
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Cost = {best_cost:.2f}", end='\r')
                
            if patience_counter >= self.patience:
                print(f"\nStopping early at gen {gen} due to patience.")
                break
                
        print(f"\nGA Finished. Best Cost: {best_cost:.2f}")
        self._save_final_population_details(pop, filename=f"last_generation/ga_hct_final_pop_seed{self.seed}_{len(self.problem.nodes_map)}_{self.evaluator.__name__}.csv")
        return decode_to_solution(self.problem, best_enc)