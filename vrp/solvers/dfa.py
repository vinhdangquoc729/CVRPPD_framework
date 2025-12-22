from __future__ import annotations
import math
import random
import time as _time
from typing import List, Dict, Tuple, Any

from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate

class DFASolver(Solver):
    def __init__(
        self,
        problem: Problem,
        seed: int = 42,
        pop_size: int = 50,
        gamma: float = 0.95,
    ):
        super().__init__(problem, seed)
        self.pop_size = pop_size
        self.gamma = gamma
        self.rng = random.Random(seed)
        # Pre-cache depot ID để tránh truy cập thuộc tính object trong vòng lặp
        self._veh2dep = {v.id: v.depot_id for v in self.problem.vehicles}

    # =========================
    # Helpers & Pre-processing
    # =========================

    def _vehicles_by_depot(self) -> Dict[int, List]:
        P = self.problem
        by_dep: Dict[int, List] = {}
        for v in P.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _customers_by_nearest_depot(self) -> Dict[int, List[int]]:
        P = self.problem
        depots = P.depots
        cust_by_dep: Dict[int, List[int]] = {d: [] for d in depots}
        for nid, nd in P.nodes.items():
            if nd.is_depot: continue
            best_dep, best_dist = None, float("inf")
            for d in depots:
                dist = P.d(nid, d)
                if dist < best_dist:
                    best_dist, best_dep = dist, d
            if best_dep is not None:
                cust_by_dep[best_dep].append(nid)
        return cust_by_dep

    def _get_flattened(self, sol: Solution) -> List[int]:
        """Cache chuỗi khách hàng để tính Hamming nhanh hơn."""
        nodes = self.problem.nodes
        return [nid for rt in sol.routes for nid in rt.seq if not nodes[nid].is_depot]

    # =========================
    # Khởi tạo quần thể
    # =========================

    def _init_population_data(self) -> List[Dict[str, Any]]:
        P, r = self.problem, self.rng
        veh_by_dep = self._vehicles_by_depot()
        cust_by_dep = self._customers_by_nearest_depot()
        pop_data = []

        for _ in range(self.pop_size):
            routes: List[Route] = []
            for dep, vehs in veh_by_dep.items():
                customers = cust_by_dep.get(dep, [])[:]
                if not customers or not vehs: continue
                r.shuffle(customers)

                buckets = [[] for _ in range(len(vehs))]
                for i, c in enumerate(customers):
                    buckets[i % len(vehs)].append(c)

                for k, v in enumerate(vehs):
                    custs_of_veh = buckets[k]
                    if not custs_of_veh:
                        routes.append(Route(vehicle_id=v.id, seq=[dep, dep]))
                        continue
                    
                    avg_len = r.randint(4, 8)
                    for start in range(0, len(custs_of_veh), avg_len):
                        segment = custs_of_veh[start : start + avg_len]
                        routes.append(Route(vehicle_id=v.id, seq=[dep] + segment + [dep]))
            
            sol = Solution(routes=routes)
            cost, _ = evaluate(P, sol, return_details=False)
            pop_data.append({'sol': sol, 'cost': cost, 'flat': self._get_flattened(sol)})
        return pop_data

    # =========================
    # Phép toán Di chuyển (Tối ưu Shallow Copy)
    # =========================

    def _fast_insertion_move(self, sol: Solution) -> Solution:
        r = self.rng
        if not sol.routes: return sol
        active_indices = [idx for idx, rt in enumerate(sol.routes) if len(rt.seq) > 2]
        if not active_indices: return sol

        new_routes = list(sol.routes)
        if r.random() < 0.5:
            idx = r.choice(active_indices)
            old_rt = sol.routes[idx]
            if len(old_rt.seq) <= 3: return sol
            new_seq = list(old_rt.seq)
            i, j = r.sample(range(1, len(new_seq) - 1), 2)
            node = new_seq.pop(i)
            new_seq.insert(j, node)
            new_routes[idx] = Route(vehicle_id=old_rt.vehicle_id, seq=new_seq)
        else:
            idx_src = r.choice(active_indices)
            rt_src = sol.routes[idx_src]
            dep_id = self._veh2dep[rt_src.vehicle_id]
            same_dep_indices = [idx for idx, rt in enumerate(sol.routes) 
                               if self._veh2dep[rt.vehicle_id] == dep_id and idx != idx_src]
            if not same_dep_indices: return sol
            idx_dst = r.choice(same_dep_indices)
            rt_dst = sol.routes[idx_dst]
            new_seq_src, new_seq_dst = list(rt_src.seq), list(rt_dst.seq)
            node = new_seq_src.pop(r.randint(1, len(new_seq_src) - 2))
            new_seq_dst.insert(r.randint(1, max(1, len(new_seq_dst) - 1)), node)
            new_routes[idx_src] = Route(vehicle_id=rt_src.vehicle_id, seq=new_seq_src)
            new_routes[idx_dst] = Route(vehicle_id=rt_dst.vehicle_id, seq=new_seq_dst)
        return Solution(routes=new_routes)

    # =========================
    # Giải thuật Discrete Firefly (Tối ưu Jumps)
    # =========================

    def solve(
        self,
        time_limit_sec: float = 10000.0,
        patience_iters: int = 50,
        max_generations: int | None = 500,
    ) -> Solution:
        P = self.problem
        pop_data = self._init_population_data()
        pop_data.sort(key=lambda x: x['cost'])
        best_entry = pop_data[0]

        gen, no_improve, t0 = 0, 0, _time.time()
        
        # Giới hạn số "hướng nhảy" tối đa mỗi cá thể quan sát
        MAX_JUMPS_PER_ENTITY = 5 

        while (_time.time() - t0) < time_limit_sec and (max_generations is None or gen < max_generations):
            # Luôn so sánh với các cá thể tốt hơn (targets)
            new_pop_data = []

            for i in range(len(pop_data)):
                xi_entry = pop_data[i]
                success_jumps = 0
                
                # Duyệt qua các cá thể sáng hơn xj (index j < i)
                for j in range(i):
                    target = pop_data[j]
                    
                    if target['cost'] < xi_entry['cost']:
                        # Tính rij nhanh bằng flat cache
                        A, B = xi_entry['flat'], target['flat']
                        L = min(len(A), len(B))
                        rij = sum(1 for k in range(L) if A[k] != B[k]) + abs(len(A) - len(B))
                        
                        # Thử cải thiện xi theo hướng target
                        step_max = max(1, int(rij * (self.gamma ** gen) * 0.1))
                        improved_this_direction = False
                        
                        for _ in range(step_max):
                            cand_sol = self._fast_insertion_move(xi_entry['sol'])
                            c_val, _ = evaluate(P, cand_sol, return_details=False)
                            
                            if c_val < xi_entry['cost']:
                                xi_entry = {
                                    'sol': cand_sol,
                                    'cost': c_val,
                                    'flat': self._get_flattened(cand_sol)
                                }
                                improved_this_direction = True
                                break # Early exit cho step_max
                        
                        if improved_this_direction:
                            success_jumps += 1
                            if success_jumps >= MAX_JUMPS_PER_ENTITY:
                                break # Đạt giới hạn 5 hướng nhảy thành công
                
                # Nếu không có hướng nhảy nào tốt (hoặc là con tốt nhất), thử đột biến nhẹ
                if success_jumps == 0:
                    mutated = self._fast_insertion_move(xi_entry['sol'])
                    m_cost, _ = evaluate(P, mutated, return_details=False)
                    if m_cost < xi_entry['cost']:
                        xi_entry = {'sol': mutated, 'cost': m_cost, 'flat': self._get_flattened(mutated)}

                new_pop_data.append(xi_entry)

            # Sắp xếp và chọn lọc thế hệ mới
            new_pop_data.sort(key=lambda x: x['cost'])
            pop_data = new_pop_data[:self.pop_size]

            if pop_data[0]['cost'] < best_entry['cost']:
                best_entry, no_improve = pop_data[0], 0
            else:
                no_improve += 1

            gen += 1
            print(f"Gen {gen} | Best: {best_entry['cost']:.2f} | Jumps/Ent: {MAX_JUMPS_PER_ENTITY}", end='\r')
            if patience_iters and no_improve >= patience_iters: break

        return best_entry['sol']