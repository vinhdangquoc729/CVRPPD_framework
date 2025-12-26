from __future__ import annotations
import random
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem import Problem, Node, Vehicle
from ..core.solution import Solution, Route
from ..core.eval import evaluate

@dataclass
class _VehState:
    """Trạng thái của xe khi xây dựng lộ trình trong Phase 1."""
    vehicle: Vehicle
    completed_routes: List[Route]
    current_seq: List[int]        # [depot, c1, c2...]
    current_load: float
    current_time: float
    current_node_id: int

class OmbukiGASolver(Solver):
    def __init__(
        self,
        problem: Problem,
        seed: int = 42,
        pop_size: int = 50,
        max_generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_k: int = 4,
        tournament_p_best: float = 0.8,
        weight_routes: float = 100.0,
        weight_distance: float = 0.001,
        evaluator: callable = None,
    ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.tournament_p_best = tournament_p_best

        self.weight_routes = weight_routes
        self.weight_distance = weight_distance
        self.evaluator = evaluator if evaluator is not None else evaluate

        # --- QUAN TRỌNG: Tự xây dựng map dữ liệu để đảm bảo chính xác ---
        self.orders_map = {}
        if hasattr(problem, "orders_map") and problem.orders_map:
            self.orders_map = problem.orders_map
        elif hasattr(problem, "orders") and problem.orders:
            self.orders_map = {o.id: o for o in problem.orders}
        else: raise ValueError("Problem must have orders_map or orders attribute.")
        self.nodes_map = {}
        if hasattr(problem, "nodes_map") and problem.nodes_map:
            self.nodes_map = problem.nodes_map
        elif hasattr(problem, "nodes"):
            # Nếu nodes là list
            if isinstance(problem.nodes, list):
                self.nodes_map = {n.id: n for n in problem.nodes}
            # Nếu nodes là dict
            elif isinstance(problem.nodes, dict):
                self.nodes_map = problem.nodes

        # Xác định Gene là Order ID hay Node ID
        if self.orders_map:
            self._customers = list(self.orders_map.keys()) # Order-based
        else:
            self._customers = [nid for nid, n in self.nodes_map.items() if not n.is_depot] # Legacy

        self._veh_by_id: Dict[int, Vehicle] = {v.id: v for v in problem.vehicles}

    # ============================================================
    # PHASE 1: Sequential Greedy Decoding
    # ============================================================

    def _check_goods_compatibility(self, veh: Vehicle, item_id: int) -> bool:
        """Kiểm tra xe có được phép chở đơn hàng này không (Strict)."""
        if item_id in self.orders_map:
            order = self.orders_map[item_id]
            # subset check: {1} issubset {1,2,3} -> True
            return order.contained_goods_types.issubset(veh.allowed_goods_types)
        return True # Legacy/Node-based coi như luôn đúng

    def _get_item_info(self, item_id: int) -> Tuple[float, float, float, float, float]:
        """Lấy thông tin (weight_change, service_time, tw_open, tw_close, node_location_id)"""
        if item_id in self.orders_map:
            order = self.orders_map[item_id]
            node = self.nodes_map[order.node_id]
            w_change = order.total_weight if order.order_type == 1 else -order.total_weight
            return w_change, order.service_time, order.tw_open, order.tw_close, node.id
        
        # Legacy
        node = self.nodes_map[item_id]
        w_change = node.demand_pickup - node.demand_delivery
        return w_change, node.service_time, node.tw_open, node.tw_close, node.id

    def _try_append_to_current_trip(self, state: _VehState, item_id: int, 
                                    w_change: float, s_time: float, 
                                    tw_op: float, tw_cl: float, node_loc: int) -> bool:
        """Kiểm tra logic Tải trọng và Thời gian."""
        P = self.problem
        veh = state.vehicle
        
        # 1. Load
        new_load = state.current_load + w_change
        if new_load > veh.capacity or new_load < 0: 
            return False
            
        # 2. Time (Travel -> Arrive -> Wait -> Service)
        dist = P.get_dist_node_to_node(state.current_node_id, node_loc)
        arrival = state.current_time + (dist / P.speed)
        
        start_svc = max(arrival, tw_op if tw_op else 0)
        
        # Hard TW Check
        # if tw_cl is not None and start_svc > tw_cl:
        #     return False
            
        finish_svc = start_svc + s_time
        
        # 3. Check Return to Depot (End Time)
        dist_home = P.get_dist_node_to_node(node_loc, veh.end_depot_id)
        time_home = dist_home / P.speed
        
        if finish_svc + time_home > veh.end_time:
            return False
            
        return True

    def _decode_chromosome(self, chrom: List[int]) -> Solution:
        """
        Phase 1: Duyệt đơn hàng theo thứ tự Chromosome.
                 Duyệt xe theo thứ tự cố định.
                 Gán ngay nếu thỏa mãn (Goods + Time + Cap).
                 Hỗ trợ Multi-trip nếu xe hết giờ/tải cho chuyến này nhưng còn giờ trong ngày.
        """
        P = self.problem
        
        # Reset trạng thái xe
        sorted_vehs = sorted(P.vehicles, key=lambda v: v.id)
        veh_states = [
            _VehState(
                vehicle=v,
                completed_routes=[],
                current_seq=[v.start_depot_id],
                current_load=0.0,
                current_time=v.start_time,
                current_node_id=v.start_depot_id
            ) for v in sorted_vehs
        ]
        
        for item_id in chrom:
            # Skip depot ids if mixed in list
            if item_id in self.nodes_map and self.nodes_map[item_id].is_depot: continue

            w_change, s_time, tw_op, tw_cl, node_loc = self._get_item_info(item_id)
            assigned = False
            
            for state in veh_states:
                veh = state.vehicle
                
                # --- [STRICT CHECK] Goods Compatibility ---
                if not self._check_goods_compatibility(veh, item_id):
                    continue # Xe này không chở được, bỏ qua ngay
                
                # --- Try Trip Current ---
                if self._try_append_to_current_trip(state, item_id, w_change, s_time, tw_op, tw_cl, node_loc):
                    # Commit update
                    dist = P.get_dist_node_to_node(state.current_node_id, node_loc)
                    arr = state.current_time + (dist / P.speed)
                    start = max(arr, tw_op if tw_op else 0)
                    
                    state.current_time = start + s_time
                    state.current_load += w_change
                    state.current_node_id = node_loc
                    state.current_seq.append(item_id)
                    assigned = True
                    break
                
                # --- Try Multi-trip (Về depot -> Đi chuyến mới) ---
                # Tính thời gian về depot từ vị trí hiện tại
                dist_back = P.get_dist_node_to_node(state.current_node_id, veh.start_depot_id)
                arr_depot = state.current_time + (dist_back / P.speed)
                
                # Thời gian bắt đầu chuyến mới
                new_start = max(arr_depot, veh.start_time)
                
                # Kiểm tra khả thi chuyến mới: Depot -> Khách -> Về
                dist_to = P.get_dist_node_to_node(veh.start_depot_id, node_loc)
                arr_cust = new_start + (dist_to / P.speed)
                start_cust = max(arr_cust, tw_op if tw_op else 0)
                
                # Check TW & End Time
                if tw_cl is not None and start_cust > tw_cl: continue
                
                finish_cust = start_cust + s_time
                dist_end = P.get_dist_node_to_node(node_loc, veh.end_depot_id)
                if finish_cust + (dist_end / P.speed) > veh.end_time: continue
                
                # Check Capacity cho chuyến mới (quan trọng!)
                # Nếu w_change > capacity thì ngay cả chuyến mới cũng ko chở được
                if w_change > veh.capacity: continue

                # OK -> Đóng chuyến cũ, mở chuyến mới
                if state.current_seq[-1] != veh.start_depot_id:
                    state.current_seq.append(veh.start_depot_id)
                state.completed_routes.append(Route(veh.id, state.current_seq))
                
                state.current_seq = [veh.start_depot_id, item_id]
                state.current_load = w_change
                state.current_time = finish_cust
                state.current_node_id = node_loc
                assigned = True
                break
            
            if not assigned:
                # Không xe nào nhận -> Đành để unserved (để tránh vi phạm cứng)
                pass

        # Finalize routes
        final_routes = []
        for state in veh_states:
            final_routes.extend(state.completed_routes)
            if len(state.current_seq) > 1:
                if state.current_seq[-1] != state.vehicle.end_depot_id:
                    state.current_seq.append(state.vehicle.end_depot_id)
                final_routes.append(Route(state.vehicle.id, state.current_seq))
                
        return Solution(final_routes)

    # ============================================================
    # PHASE 2: Local Search (Robust Check)
    # ============================================================

    def _check_route_strict(self, seq: List[int], veh: Vehicle) -> Tuple[bool, float]:
        """Check valid route & return distance cost."""
        P = self.problem
        if len(seq) < 2: return True, 0.0
        
        t = float(veh.start_time)
        load = 0.0
        dist = 0.0
        
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i+1]
            
            # Identify Location IDs
            u_loc = P.orders_map[u].node_id if u in self.orders_map else u
            v_loc = P.orders_map[v].node_id if v in self.orders_map else v
            
            d_leg = P.get_dist_node_to_node(u_loc, v_loc)
            dist += d_leg
            t += d_leg / P.speed
            
            # Check Node/Order Constraints
            if v in self.orders_map:
                o = self.orders_map[v]
                # Goods check
                if not o.contained_goods_types.issubset(veh.allowed_goods_types):
                    return False, float('inf')
                
                tw_op, tw_cl, s_t = o.tw_open, o.tw_close, o.service_time
                w = o.total_weight if o.order_type == 1 else -o.total_weight
            else:
                # Depot
                nd = self.nodes_map[v]
                tw_op, tw_cl, s_t, w = nd.tw_open, nd.tw_close, nd.service_time, 0.0
                if nd.is_depot and t > veh.end_time: return False, float('inf')

            load += w
            if load > veh.capacity or load < 0: return False, float('inf')
            
            start = max(t, tw_op if tw_op else 0)
            if tw_cl is not None and start > tw_cl: return False, float('inf')
            t = start + s_t
            
        return True, dist

    def _improve_phase2(self, sol: Solution) -> Solution:
        if len(sol.routes) < 2: return sol
        new_routes = [Route(r.vehicle_id, list(r.seq)) for r in sol.routes]
        
        for i in range(len(new_routes) - 1):
            r1, r2 = new_routes[i], new_routes[i+1]
            # Chỉ move nếu cùng xe (hoặc logic tùy chọn) để an toàn
            if r1.vehicle_id != r2.vehicle_id: continue
            if len(r1.seq) <= 3: continue
            
            cand = r1.seq[-2]
            veh = self._veh_by_id[r1.vehicle_id]
            
            # Tạo ứng viên
            s1_new = r1.seq[:-2] + [r1.seq[-1]]
            s2_new = [r2.seq[0]] + [cand] + r2.seq[1:]
            
            # Check
            ok1, d1 = self._check_route_strict(s1_new, veh)
            ok2, d2 = self._check_route_strict(s2_new, veh)
            
            if ok1 and ok2:
                # Check current cost
                _, d1_old = self._check_route_strict(r1.seq, veh)
                _, d2_old = self._check_route_strict(r2.seq, veh)
                
                if (d1 + d2) < (d1_old + d2_old) - 1e-6:
                    r1.seq = s1_new
                    r2.seq = s2_new
                    
        return Solution(new_routes)

    # ... (Giữ nguyên các hàm GA: init_pop, tournament, crossover, mutate, solve, save ...)
    # Chú ý: _evaluate_chromosome gọi _decode_chromosome -> _improve_phase2 -> eval
    # Nên mọi thứ sẽ đồng bộ logic strict.

    def _init_population(self) -> List[List[int]]:
        base = self._customers
        pop = []
        for _ in range(self.pop_size):
            chrom = base[:] 
            self.rng.shuffle(chrom)
            pop.append(chrom)
        return pop

    def _evaluate_chromosome(self, chrom: List[int]) -> float:
        sol_phase1 = self._decode_chromosome(chrom)
        sol = self._improve_phase2(sol_phase1)
        cost, _ = self.evaluator(self.problem, sol, return_details=False)
        return cost

    def _tournament_select(self, pop: List[List[int]], fitnesses: List[float]) -> List[int]:
        r = self.rng
        N = len(pop)
        idxs = [r.randrange(N) for _ in range(self.tournament_k)]
        idxs.sort(key=lambda i: fitnesses[i])
        if r.random() < self.tournament_p_best:
            return pop[idxs[0]][:]
        else:
            return pop[r.choice(idxs)][:]

    def _crossover_ox(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        r = self.rng
        n = len(p1)
        if n < 2: return p1[:], p2[:]
        i, j = sorted(r.sample(range(n), 2))
        c1 = [None] * n
        c2 = [None] * n
        c1[i:j + 1] = p1[i:j + 1]
        c2[i:j + 1] = p2[i:j + 1]
        
        def fill(child, parent):
            pos = (j + 1) % n
            for gene in parent:
                if gene not in child:
                    child[pos] = gene
                    pos = (pos + 1) % n
        fill(c1, p2)
        fill(c2, p1)
        return c1, c2 

    def _mutate_inversion(self, chrom: List[int]) -> None:
        r = self.rng
        n = len(chrom)
        if n < 2: return
        max_len = 3 if n >= 3 else 2
        seg_len = r.randint(2, max_len)
        i = r.randint(0, n - seg_len)
        chrom[i:i+seg_len] = reversed(chrom[i:i+seg_len])

    def _save_final_population_details(self, population: List[List[int]], filename: str = "ombuki_final_pop_details.csv"):
        import pandas as pd
        import os
        all_records = []
        for i, chrom in enumerate(population[:10]):
            sol_phase1 = self._decode_chromosome(chrom)
            sol = self._improve_phase2(sol_phase1)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {"individual_id": i, "total_cost": cost, **details, "solution_str": str(sol)}
            all_records.append(record)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pd.DataFrame(all_records).to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 30000.0) -> Solution:
        r = self.rng
        t0 = time.time()
        print("Initializing Ombuki GA (Strict Goods Constraint)...")
        pop = self._init_population()
        fitnesses = [self._evaluate_chromosome(c) for c in pop]

        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_chrom = pop[best_idx][:]
        best_cost = fitnesses[best_idx]
        
        print(f"Ombuki Start. Initial Best Cost: {best_cost:.2f}")

        gen = 0
        while gen < self.max_generations and (time.time() - t0) < time_limit_sec:
            gen += 1
            new_pop = [best_chrom[:]]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                
                if r.random() < self.crossover_rate:
                    c1, c2 = self._crossover_ox(p1, p2)
                    child = c1 if r.random() < 0.5 else c2
                else:
                    child = p1[:]
                
                if r.random() < self.mutation_rate:
                    self._mutate_inversion(child)
                
                new_pop.append(child)

            pop = new_pop
            fitnesses = [self._evaluate_chromosome(c) for c in pop]

            curr_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            if fitnesses[curr_best_idx] < best_cost:
                best_cost = fitnesses[curr_best_idx]
                best_chrom = pop[curr_best_idx][:]
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best={best_cost:.2f}", end="\r")

        print(f"\nOmbuki Finished. Best Cost: {best_cost:.2f}")
        self._save_final_population_details(pop, filename=f"last_generation/ombuki_final_pop_seed{self.seed}.csv")
        
        best_sol_phase1 = self._decode_chromosome(best_chrom)
        return self._improve_phase2(best_sol_phase1)