from __future__ import annotations
import math, random, time
from typing import List, Dict, Tuple
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified  # dùng evaluator PD mới

class ESASolverPD(Solver):
    def __init__(self, problem: Problem, seed: int = 42,
                 mu: int = 24,
                 elite_frac: float = 0.25,
                 trials_per_iter: int = 8,
                 alpha: float = 0.95, 
                 patience_iters: int = 50): 
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.mu = mu
        self.elite_frac = elite_frac
        self.trials_per_iter = trials_per_iter
        self.alpha = alpha
        self.patience_iters = patience_iters

    # các helper
    def _customers(self) -> List[int]:
        return [i for i, nd in self.problem.nodes.items() if not nd.is_depot]

    def _vehicles_by_depot(self) -> Dict[int, List]:
        by_dep: Dict[int, List] = {}
        for v in self.problem.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _nearest_depot(self, nid: int) -> int:
        P = self.problem
        return min(P.depots, key=lambda d: P.d(nid, d))

    # Khởi tạo quần thể
    def _init_population(self) -> List[Solution]:
        P = self.problem
        r = self.rng
        veh_by_dep = self._vehicles_by_depot()
        custs = self._customers()

        # gán khách về depot gần nhất
        home: Dict[int, List[int]] = {d: [] for d in P.depots}
        for c in custs:
            d = self._nearest_depot(c)
            home[d].append(c)

        pop: List[Solution] = []
        for _ in range(self.mu):
            routes: List[Route] = []
            for d, vehs in veh_by_dep.items():
                if not vehs:
                    continue
                bag = home[d][:]
                r.shuffle(bag)
                buckets = [[] for _ in range(len(vehs))]
                for i, c in enumerate(bag):
                    buckets[i % len(vehs)].append(c)
                for k, v in enumerate(vehs):
                    seq = [d] + buckets[k] + [d]
                    routes.append(Route(vehicle_id=v.id, seq=seq))
            pop.append(Solution(routes=routes))
        return pop

    # ---------- Tiện ích route ----------
    def _customers_of_route(self, r: Route) -> List[int]:
        P = self.problem
        return [i for i in r.seq if not P.nodes[i].is_depot]

    def _rebuild(self, r: Route, custs: List[int]) -> None:
        P = self.problem
        dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
        r.seq = [dep] + custs + [dep]

    # ---------- Láng giềng (cấp-khách) ----------
    def _mv_intra_insert(self, s: Solution) -> None:
        routes = [rt for rt in s.routes if len(self._customers_of_route(rt)) >= 2]
        if not routes: return
        r = self.rng.choice(routes)
        C = self._customers_of_route(r)
        i = self.rng.randrange(len(C))
        x = C.pop(i)
        j = self.rng.randrange(len(C)+1)
        C.insert(j, x)
        self._rebuild(r, C)

    def _mv_inter_relocate(self, s: Solution) -> None:
        rs = [rt for rt in s.routes if len(self._customers_of_route(rt)) >= 1]
        if not rs: return
        rA = self.rng.choice(rs)
        CA = self._customers_of_route(rA)
        i = self.rng.randrange(len(CA))
        x = CA.pop(i)
        rB = self.rng.choice(s.routes)
        if rB is rA:
            j = self.rng.randrange(len(CA)+1)
            CA.insert(j, x)
            self._rebuild(rA, CA)
        else:
            CB = self._customers_of_route(rB)
            j = self.rng.randrange(len(CB)+1)
            CB.insert(j, x)
            self._rebuild(rA, CA)
            self._rebuild(rB, CB)

    def _mv_inter_swap(self, s: Solution) -> None:
        rs = [rt for rt in s.routes if len(self._customers_of_route(rt)) >= 1]
        if len(rs) < 2: return
        rA, rB = self.rng.sample(rs, 2)
        CA = self._customers_of_route(rA)
        CB = self._customers_of_route(rB)
        i = self.rng.randrange(len(CA))
        j = self.rng.randrange(len(CB))
        CA[i], CB[j] = CB[j], CA[i]
        self._rebuild(rA, CA); self._rebuild(rB, CB)

    def _mv_two_opt(self, s: Solution) -> None:
        routes = [rt for rt in s.routes if len(self._customers_of_route(rt)) >= 3]
        if not routes: return
        r = self.rng.choice(routes)
        C = self._customers_of_route(r)
        i = self.rng.randrange(0, len(C)-1)
        j = self.rng.randrange(i+1, len(C))
        C[i:j] = reversed(C[i:j])
        self._rebuild(r, C)

    def _random_neighbor(self, s: Solution) -> None:
        p = self.rng.random()
        if p < 0.30:
            self._mv_intra_insert(s)
        elif p < 0.65:
            self._mv_inter_relocate(s)
        elif p < 0.85:
            self._mv_inter_swap(s)
        else:
            self._mv_two_opt(s)

    def solve(self, time_limit_sec: float = 30.0) -> Solution:
        P = self.problem
        r = self.rng
        t0 = time.time()

        pop = self._init_population()

        # cache chi phí
        cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}
        def key_of(s: Solution):
            return tuple(tuple(rt.seq) for rt in s.routes)
        def cost_of(s: Solution) -> float:
            k = key_of(s)
            v = cost_cache.get(k)
            if v is None:
                v, _ = evaluate_modified(P, s, return_details=False)
                cost_cache[k] = v
            return v

        pop.sort(key=cost_of)
        best = pop[0]; best_cost = cost_of(best)

        # đặt nhiệt độ
        costs0 = [cost_of(s) for s in pop]
        spread = max(costs0) - min(costs0) if len(costs0) > 1 else max(1.0, abs(costs0[0]))
        T = spread / max(1.0, math.log(1 / 0.95))  # p_accept = 0.95 cho Δ=spread

        patience = 0
        it = 0

        while time.time() - t0 < time_limit_sec:
            it += 1
            new_pop: List[Solution] = []

            # elitism
            elite_k = max(1, int(self.elite_frac * self.mu))
            elites = pop[:elite_k]
            new_pop.extend(elites)

            # sinh ứng viên từ tournament nhỏ
            while len(new_pop) < self.mu:
                # chọn bố mẹ
                if len(pop) >= 2:
                    a, b = r.sample(pop, 2)
                    parent = a if cost_of(a) <= cost_of(b) else b
                else:
                    parent = pop[0]
                child = parent
                # nhiều bước láng giềng 
                for _ in range(self.trials_per_iter):
                    import copy
                    cand = copy.deepcopy(child)
                    self._random_neighbor(cand)
                    dE = cost_of(cand) - cost_of(child)
                    if dE <= 0 or r.random() < math.exp(-dE / max(1e-6, T)):
                        child = cand
                new_pop.append(child)

            pop = sorted(new_pop, key=cost_of)
            cur_best = pop[0]; cur_cost = cost_of(cur_best)

            if cur_cost < best_cost:
                best, best_cost = cur_best, cur_cost
                patience = 0
            else:
                patience += 1
                if patience >= self.patience_iters:
                    break

            # làm nguội
            T *= self.alpha
            if T < 1e-6:
                T = 1e-6

        return best
