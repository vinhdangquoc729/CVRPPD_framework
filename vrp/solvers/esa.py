from __future__ import annotations
import math, random, time
from typing import List, Tuple, Dict
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate


class ESASolver(Solver):
    def __init__(self, problem: Problem, seed: int = 42,
                 mu: int = 20,                 # cỡ quần thể
                 elite_frac: float = 0.3,         # giữ elite
                 alpha: float = 0.95,             # cooling rate
                 trials_per_iter: int = 8,        # số láng giềng cho mỗi cá thể/iter
                 patience_iters: int = 25,        # early stop nếu không cải thiện
                 max_generation: int = 500,       # THÊM: giới hạn số thế hệ tối đa
                 ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.mu = mu
        self.elite_frac = elite_frac
        self.alpha = alpha
        self.trials_per_iter = trials_per_iter
        self.patience_iters = patience_iters
        self.max_generation = max_generation # Lưu tham số max_generation

    # ... [Các hàm _vehicles_by_depot, _customers_by_nearest_depot, _init_population giữ nguyên] ...

    def _vehicles_by_depot(self) -> Dict[int, List]:
        by_dep: Dict[int, List] = {}
        for v in self.problem.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _customers_by_nearest_depot(self) -> Dict[int, List[int]]:
        P = self.problem
        depot_ids = list({v.depot_id for v in P.vehicles})
        cust_by_dep: Dict[int, List[int]] = {d: [] for d in depot_ids}
        for nid, nd in P.nodes.items():
            if nd.is_depot:
                continue
            best_dep = None
            best_dist = float("inf")
            for d in depot_ids:
                dist = P.d(nid, d)
                if dist < best_dist:
                    best_dist = dist
                    best_dep = d
            if best_dep is not None:
                cust_by_dep[best_dep].append(nid)
        return cust_by_dep
    
    def _init_population(self) -> List[Solution]:
        P = self.problem
        r = self.rng
        veh_by_dep = self._vehicles_by_depot()
        cust_by_dep = self._customers_by_nearest_depot()
        pop: List[Solution] = []
        for _ in range(self.mu):
            routes: List[Route] = []
            for dep, vehs in veh_by_dep.items():
                customers = cust_by_dep.get(dep, [])[:]
                if not customers or not vehs:
                    for v in vehs:
                        routes.append(Route(vehicle_id=v.id, seq=[dep, dep]))
                    continue
                r.shuffle(customers)
                buckets: List[List[int]] = [[] for _ in range(len(vehs))]
                for i, c in enumerate(customers):
                    buckets[i % len(vehs)].append(c)
                for k, v in enumerate(vehs):
                    custs_of_veh = buckets[k]
                    routes.append(Route(vehicle_id=v.id, seq=[dep, dep]))
                    if not custs_of_veh:
                        continue
                    avg_len = r.randint(4, 8)
                    start = 0
                    while start < len(custs_of_veh):
                        end = min(len(custs_of_veh), start + avg_len)
                        segment = custs_of_veh[start:end]
                        seq = [dep] + segment + [dep]
                        routes.append(Route(vehicle_id=v.id, seq=seq))
                        start = end
            pop.append(Solution(routes=routes))
        return pop

    def _neighbor(self, sol: Solution) -> Solution:
        import copy
        P = self.problem
        r = self.rng
        s = copy.deepcopy(sol)
        if not s.routes:
            return s
        veh2dep = {v.id: v.depot_id for v in P.vehicles}
        routes_with_cust = [rt for rt in s.routes if len(rt.seq) > 2]
        if not routes_with_cust:
            return s
        if r.random() < 0.5:
            rt = r.choice(routes_with_cust)
            depot_id = veh2dep[rt.vehicle_id]
            inner_idx = list(range(1, len(rt.seq) - 1))
            if len(inner_idx) < 2:
                return s
            i, j = r.sample(inner_idx, 2)
            node = rt.seq.pop(i)
            if j >= len(rt.seq): 
                j = len(rt.seq) - 1
            rt.seq.insert(j, node)
            if rt.seq[0] != depot_id: 
                rt.seq = [depot_id] + [n for n in rt.seq if n != depot_id]
            if rt.seq[-1] != depot_id: 
                rt.seq = [n for n in rt.seq if n != depot_id] + [depot_id]
            return s
        rt_src = r.choice(routes_with_cust)
        dep_src = veh2dep[rt_src.vehicle_id]
        candidates = [rt for rt in s.routes if veh2dep[rt.vehicle_id] == dep_src]
        if len(candidates) < 2:
            return self._neighbor_intra_only(s)
        rt_dst = r.choice(candidates)
        while rt_dst is rt_src:
            rt_dst = r.choice(candidates)
        src_inner_idx = list(range(1, len(rt_src.seq) - 1))
        if not src_inner_idx:
            return s
        i = r.choice(src_inner_idx)
        node = rt_src.seq.pop(i)
        dst_inner_pos = list(range(1, len(rt_dst.seq)))
        j = r.choice(dst_inner_pos)
        rt_dst.seq.insert(j, node)
        dep_dst = veh2dep[rt_dst.vehicle_id]
        if rt_src.seq[0] != dep_src:
            rt_src.seq = [dep_src] + [n for n in rt_src.seq if n != dep_src]
        if rt_src.seq[-1] != dep_src:
            rt_src.seq = [n for n in rt_src.seq if n != dep_src] + [dep_src]
        if rt_dst.seq[0] != dep_dst:
            rt_dst.seq = [dep_dst] + [n for n in rt_dst.seq if n != dep_dst]
        if rt_dst.seq[-1] != dep_dst:
            rt_dst.seq = [n for n in rt_dst.seq if n != dep_dst] + [dep_dst]
        return s

    def _neighbor_intra_only(self, sol: Solution) -> Solution:
        import copy
        P = self.problem
        r = self.rng
        s = copy.deepcopy(sol)
        veh2dep = {v.id: v.depot_id for v in P.vehicles}
        routes_with_cust = [rt for rt in s.routes if len(rt.seq) > 2]
        if not routes_with_cust:
            return s
        rt = r.choice(routes_with_cust)
        depot_id = veh2dep[rt.vehicle_id]
        inner_idx = list(range(1, len(rt.seq) - 1))
        if len(inner_idx) < 2:
            return s
        i, j = r.sample(inner_idx, 2)
        node = rt.seq.pop(i)
        if j >= len(rt.seq):
            j = len(rt.seq) - 1
        rt.seq.insert(j, node)
        if rt.seq[0] != depot_id:
            rt.seq = [depot_id] + [n for n in rt.seq if n != depot_id]
        if rt.seq[-1] != depot_id:
            rt.seq = [n for n in rt.seq if n != depot_id] + [depot_id]
        return s

    # =========================
    # Hàm solve với max_generation
    # =========================
    def solve(self, time_limit_sec: float = 30.0) -> Solution:
        P = self.problem
        r = self.rng
        t0 = time.time()

        pop = self._init_population()

        cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}

        def key_of(s: Solution):
            return tuple(tuple(rt.seq) for rt in s.routes)

        def cost_of(s: Solution) -> float:
            k = key_of(s)
            v = cost_cache.get(k)
            if v is None:
                v, _ = evaluate(P, s, return_details=False)
                cost_cache[k] = v
            return v

        pop.sort(key=cost_of)
        best = pop[0]
        best_cost = cost_of(best)

        costs0 = [cost_of(s) for s in pop]
        spread = max(costs0) - min(costs0) if len(costs0) > 1 else max(1.0, abs(costs0[0]))
        T = spread / max(1.0, math.log(1 / 0.95))

        patience = 0
        it = 0

        # SỬA: Thêm điều kiện it < self.max_generation
        while (time.time() - t0 < time_limit_sec) and (it < self.max_generation):
            it += 1
            print(f"Iteration {it}: best_cost = {best_cost}", end="\r")
            new_pop: List[Solution] = []

            elite_k = max(1, int(self.elite_frac * self.mu))
            elites = pop[:elite_k]
            new_pop.extend(elites)

            while len(new_pop) < self.mu:
                if len(pop) >= 2:
                    a, b = r.sample(pop, 2)
                    parent = a if cost_of(a) <= cost_of(b) else b
                else:
                    parent = pop[0]
                child = parent
                for _ in range(self.trials_per_iter):
                    cand = self._neighbor(child)
                    dE = cost_of(cand) - cost_of(child)
                    if dE <= 0 or r.random() < math.exp(-dE / max(1e-6, T)):
                        child = cand
                new_pop.append(child)

            pop = sorted(new_pop, key=cost_of)
            cur_best = pop[0]
            cur_cost = cost_of(cur_best)

            if cur_cost < best_cost:
                best, best_cost = cur_best, cur_cost
                patience = 0
            else:
                patience += 1
                if patience >= self.patience_iters:
                    break

            T *= self.alpha
            if T < 1e-6:
                T = 1e-6

        return best