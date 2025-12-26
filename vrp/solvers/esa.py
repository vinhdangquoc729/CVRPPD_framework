from __future__ import annotations
import math, random, time
import copy
from typing import List, Tuple, Dict
from collections import defaultdict
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate

class ESASolver(Solver):
    def __init__(self, problem: Problem, seed: int = 42,
                 mu: int = 50,                  
                 elite_frac: float = 0.7,       
                 alpha: float = 0.95,           
                 trials_per_iter: int = 8,
                 patience_iters: int = 50,
                 max_generation: int = 500,
                 evaluator: callable = None,
                 ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.mu = mu
        self.elite_frac = elite_frac
        self.alpha = alpha
        self.trials_per_iter = trials_per_iter
        self.patience_iters = patience_iters
        self.max_generation = max_generation
        self.evaluator = evaluator if evaluator is not None else evaluate

    def _init_population(self) -> List[Solution]:
        P = self.problem
        r = self.rng
        
        # 1. Pre-process: Nhóm xe theo Depot
        veh_at_depot = defaultdict(list)
        for v in P.vehicles:
            veh_at_depot[v.start_depot_id].append(v)
            
        all_depots = list(veh_at_depot.keys())
        
        # Danh sách tất cả đơn hàng
        all_orders = list(P.orders_map.values())
        
        pop: List[Solution] = []
        
        for _ in range(self.mu):
            assignments = defaultdict(list)
            counters = {d: 0 for d in all_depots}
            
            # Xáo trộn đơn hàng
            shuffled_orders = all_orders[:]
            r.shuffle(shuffled_orders)
            
            for order in shuffled_orders:
                target_vehicle = None
                order_types = order.contained_goods_types
                
                # --- LOGIC MỚI: DUYỆT TỪ GẦN RA XA ---
                
                # 1. Tính khoảng cách từ Order đến TẤT CẢ các Depot
                # List tuple: (distance, depot_id)
                depot_distances = []
                for d_id in all_depots:
                    dist = P.get_dist_node_to_node(d_id, order.node_id)
                    depot_distances.append((dist, d_id))
                
                # 2. Sắp xếp các depot từ gần đến xa
                depot_distances.sort(key=lambda x: x[0])
                
                # 3. Duyệt tuần tự: Kho nào gần nhất mà có xe phù hợp thì CHỐT luôn
                for dist, d_id in depot_distances:
                    # Lấy các xe tại depot này
                    vehs = veh_at_depot[d_id]
                    
                    # Lọc xe tương thích hàng hóa
                    valid_vehs = [
                        v for v in vehs 
                        if order_types.issubset(v.allowed_goods_types)
                    ]
                    
                    if valid_vehs:
                        # Tìm thấy! Chọn xe theo Round-Robin để cân bằng tải
                        idx = counters[d_id] % len(valid_vehs)
                        target_vehicle = valid_vehs[idx]
                        counters[d_id] += 1
                        break # Dừng ngay, không tìm tiếp kho xa hơn
                
                # 4. FALLBACK: Nếu duyệt hết sạch các kho mà không xe nào chịu chở (rất hiếm)
                if target_vehicle is None:
                    # Quay lại kho gần nhất (đầu danh sách đã sort) để gán đại
                    # Chấp nhận bị phạt lỗi logic còn hơn là mất đơn hàng
                    nearest_d_id = depot_distances[0][1]
                    vehs = veh_at_depot[nearest_d_id]
                    if vehs:
                        idx = counters[nearest_d_id] % len(vehs)
                        target_vehicle = vehs[idx]
                        counters[nearest_d_id] += 1
                
                # Gán vào assignment
                if target_vehicle:
                    assignments[target_vehicle.id].append(order.id)
            
            # Tạo Solution
            routes = []
            for v in P.vehicles:
                seq = assignments.get(v.id, [])
                routes.append(Route(vehicle_id=v.id, seq=seq))
            
            pop.append(Solution(routes=routes))
            
        return pop

    def _neighbor(self, sol: Solution) -> Solution:
        """
        Tạo giải pháp hàng xóm:
        1. Move (Relocate): Chuyển 1 order từ route này sang route khác.
        2. Swap: Đổi chỗ 2 order.
        """
        r = self.rng
        s = copy.deepcopy(sol)
        
        if not s.routes: return s
            
        non_empty_routes = [rt for rt in s.routes if len(rt.seq) > 0]
        all_routes = s.routes
        
        if not non_empty_routes: return s

        # --- Chọn toán tử (50/50) ---
        
        # OPERATOR 1: INTRA-ROUTE (Di chuyển trong nội bộ 1 xe)
        if r.random() < 0.4:
            rt = r.choice(non_empty_routes)
            if len(rt.seq) < 2: return s
            
            i = r.randint(0, len(rt.seq) - 1)
            oid = rt.seq.pop(i)
            j = r.randint(0, len(rt.seq)) 
            rt.seq.insert(j, oid)
            return s
            
        # OPERATOR 2: INTER-ROUTE (Chuyển sang xe khác)
        else:
            rt_src = r.choice(non_empty_routes)
            rt_dst = r.choice(all_routes)
            
            if rt_src is rt_dst:
                if len(rt_src.seq) > 1: r.shuffle(rt_src.seq)
                return s
            
            i = r.randint(0, len(rt_src.seq) - 1)
            oid = rt_src.seq.pop(i)
            
            if len(rt_dst.seq) == 0: j = 0
            else: j = r.randint(0, len(rt_dst.seq))
            rt_dst.seq.insert(j, oid)
            
            return s

    def _save_final_population_details(self, population: List[Solution], filename: str = "final_pop_details_esa.csv"):
        import pandas as pd
        import os
        all_records = []
        top_k = min(len(population), 10)
        for i, sol in enumerate(population):
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {"individual_id": i, "total_cost": cost, **details, "solution_str": str(sol)}
            all_records.append(record)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = pd.DataFrame(all_records)
        df.to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        P = self.problem
        r = self.rng
        t0 = time.time()

        print("Initializing population (Smart Logic)...")
        pop = self._init_population()
        
        def cost_of(s: Solution) -> float:
            val, _ = self.evaluator(P, s, return_details=False)
            return val

        pop.sort(key=cost_of)
        
        f_best = cost_of(pop[0])
        f_worst = cost_of(pop[-1])
        sup_delta_f = f_worst - f_best
        
        p = 0.95
        T = (-sup_delta_f / math.log(p)) if (sup_delta_f > 0) else 1000.0
        if T == 0: T = 1.0

        best = pop[0]
        best_cost = f_best
        patience = 0
        it = 0

        print(f"Start ESA Loop. Initial Best: {best_cost:.2f}")

        while (time.time() - t0 < time_limit_sec) and (it < self.max_generation):
            it += 1
            if it % 10 == 0:
                print(f"Generation {it}/{self.max_generation}, Temp={T:.2f}, Best Cost={best_cost:.2f}", end='\r')
            
            new_pop: List[Solution] = []
            elite_k = int(self.elite_frac * self.mu)
            if elite_k < 1: elite_k = 1
            new_pop.extend(copy.deepcopy(pop[:elite_k]))

            while len(new_pop) < self.mu:
                parent = r.choice(pop[:elite_k])
                child = copy.deepcopy(parent)
                
                for _ in range(self.trials_per_iter):
                    cand = self._neighbor(child)
                    dE = cost_of(cand) - cost_of(child)
                    if dE <= 0 or r.random() < math.exp(-dE / max(1e-6, T)):
                        child = cand
                new_pop.append(child)

            pop = sorted(new_pop, key=cost_of)
            cur_best_cost = cost_of(pop[0])

            if cur_best_cost < best_cost:
                best = copy.deepcopy(pop[0])
                best_cost = cur_best_cost
                patience = 0
            else:
                patience += 1
                if patience >= self.patience_iters:
                    print(f"\nStopping early at gen {it} due to patience.")
                    break

            T *= self.alpha
            if T < 1e-6: T = 1e-6

        print(f"\nESA Finished. Final Best Cost: {best_cost:.2f}")
        self._save_final_population_details(pop, filename=f"last_generation/esa_final_pop_seed{self.seed}_{len(self.problem.nodes_map)}_{self.evaluator.__name__}.csv")
        return best