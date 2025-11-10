from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified as EVAL


def _sol_key(sol: Solution) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(r.seq) for r in sol.routes)


class GAPDSolver(Solver):
    """
    GA theo bài báo (điều chỉnh cho bộ evaluate_modified):
      - Khởi tạo: chia khách ngẫu nhiên cho các route (mỗi xe 1 route), có depot đầu/cuối.
      - Fitness: power-law scaling F = (Amax - Ai)^k - Amin (Ai cost của i; Amax/Amin trên quần thể).
      - Selection: tournament size=2.
      - Elitism: giữ top 'elite_frac' cho thế hệ kế.
      - Crossover: Best-cost Route Crossover (swap 1 route giữa 2 cha, chèn vị trí tối ưu ở con).
      - Mutation: đảo (reverse) 1 đoạn liên tục trong 1 route (nếu route có ≥2 khách).
      - Replacement: elitism + offspring đến đủ 'pop_size'.
      - Stop: max_generations, patience_generations (không cải thiện), time_limit_sec.
    """

    def __init__(self,
                 problem: Problem,
                 seed: int = 42,
                 pop_size: int = 40,
                 elite_frac: float = 0.10,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 power_k: float = 2.0,
                 max_generations: int = 1000,
                 patience_generations: int = 150):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.power_k = max(1.0, power_k)
        self.max_generations = max_generations
        self.patience_generations = patience_generations

        # Cache cost để tránh evaluate trùng
        self._cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}

    # ---------- tiện ích ----------
    def cost(self, s: Solution) -> float:
        k = _sol_key(s)
        v = self._cost_cache.get(k)
        if v is None:
            v, _ = EVAL(self.problem, s, return_details=False)
            self._cost_cache[k] = v
        return v

    def fitness_map(self, pop: List[Solution]) -> List[float]:
        """Power-law scaling: F = (Amax - Ai)^k - Amin"""
        costs = [self.cost(s) for s in pop]
        Amax = max(costs); Amin = min(costs)
        # Tránh NaN khi tất cả bằng nhau
        if abs(Amax - Amin) < 1e-12:
            return [1.0 for _ in pop]
        Fit = [((Amax - c) ** self.power_k) - Amin for c in costs]
        # Tối thiểu dương
        base = max(1e-9, max(Fit))  # chỉ dùng để scale không bắt buộc
        return [max(1e-9, f / base) for f in Fit]

    def tournament2(self, pop: List[Solution]) -> Solution:
        a, b = self.rng.sample(pop, 2)
        return a if self.cost(a) <= self.cost(b) else b

    # ---------- khởi tạo ----------
    def _init_population(self) -> List[Solution]:
        P = self.problem
        customers = [i for i, nd in P.nodes.items() if not nd.is_depot]
        pop = []
        for _ in range(self.pop_size):
            self.rng.shuffle(customers)
            routes: List[Route] = []
            # 1 route/vehicle (bạn có thể nâng cấp: chia thành nhiều route/xe & nhiều depot-pass; evaluate_modified vẫn xử lý)
            start = 0
            for v in P.vehicles:
                depot = v.depot_id
                # chia đều tương đối cho các xe
                take = max(0, round(len(customers) / max(1, len(P.vehicles))))
                chunk = customers[start:start + take]
                start += take
                seq = [depot] + chunk[:] + [depot]
                routes.append(Route(vehicle_id=v.id, seq=seq))
            # nếu còn khách (do làm tròn): rải vòng tròn
            idx = 0
            while start < len(customers):
                r = routes[idx % len(routes)]
                r.seq.insert(len(r.seq) - 1, customers[start])
                start += 1; idx += 1
            pop.append(Solution(routes=routes))
        return pop

    # ---------- thao tác trên route ----------
    def _routes_of(self, s: Solution) -> List[Tuple[int, int, List[int]]]:
        """Return list of (route_index, depot_id, customers_list)."""
        P = self.problem
        out = []
        for ridx, r in enumerate(s.routes):
            depot = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
            custs = [x for x in r.seq if not P.nodes[x].is_depot]
            out.append((ridx, depot, custs))
        return out

    def _apply_route_customers(self, s: Solution, ridx: int, depot: int, customers: List[int]) -> None:
        s.routes[ridx].seq = [depot] + customers[:] + [depot]

    def _erase_customers(self, s: Solution, cust_set: set) -> None:
        P = self.problem
        for r in s.routes:
            dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
            newcust = [x for x in r.seq if not P.nodes[x].is_depot and x not in cust_set]
            r.seq = [dep] + newcust + [dep]

    def _best_insert_block(self, s: Solution, block: List[int]) -> Solution:
        """Chèn cả block vào vị trí (route, index) có chi phí thấp nhất."""
        best_sol = None
        best_cost = math.inf
        P = self.problem
        for ridx, depot, custs in self._routes_of(s):
            for pos in range(len(custs) + 1):
                cand = copy.deepcopy(s)
                newcust = custs[:pos] + block + custs[pos:]
                self._apply_route_customers(cand, ridx, depot, newcust)
                c = self.cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_sol = cand
        # fallback (không xảy ra): trả lại bản sao cũ
        return best_sol if best_sol is not None else copy.deepcopy(s)

    # ---------- Crossover: best-cost route crossover ----------
    def crossover(self, p1: Solution, p2: Solution) -> Tuple[Solution, Solution]:
        if self.rng.random() > self.crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        # Chọn ngẫu nhiên 1 route (phần khách) ở mỗi bố/mẹ
        r1idx, d1, c1 = self.rng.choice(self._routes_of(p1))
        r2idx, d2, c2 = self.rng.choice(self._routes_of(p2))
        seg1 = c1[:]  # cả route (theo bài báo)
        seg2 = c2[:]

        # Con 1: lấy P1, xoá khách của seg2, rồi chèn seg2 vào best vị trí
        child1 = copy.deepcopy(p1)
        self._erase_customers(child1, set(seg2))
        child1 = self._best_insert_block(child1, seg2)

        # Con 2: tương tự với seg1
        child2 = copy.deepcopy(p2)
        self._erase_customers(child2, set(seg1))
        child2 = self._best_insert_block(child2, seg1)

        return child1, child2

    # ---------- Mutation: reverse một đoạn trong 1 route ----------
    def mutate(self, s: Solution) -> None:
        if self.rng.random() > self.mutation_rate:
            return
        P = self.problem
        # chọn route còn ≥2 khách
        routes = [(i, r) for i, r in enumerate(s.routes)
                  if sum(1 for x in r.seq if not P.nodes[x].is_depot) >= 2]
        if not routes:
            return
        ridx, r = self.rng.choice(routes)
        depot = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
        custs = [x for x in r.seq if not P.nodes[x].is_depot]
        i, j = sorted(self.rng.sample(range(len(custs)), 2))
        custs[i:j+1] = reversed(custs[i:j+1])
        self._apply_route_customers(s, ridx, depot, custs)

    # ---------- vòng chính ----------
    def solve(self,
              time_limit_sec: float = 30.0,
              on_progress=None) -> Solution:
        rng = self.rng
        t0 = time.time()

        pop = self._init_population()
        pop.sort(key=self.cost)
        best = pop[0]
        best_cost = self.cost(best)

        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best)

        no_improve = 0
        gen = 0

        while True:
            gen += 1
            # --- Elitism
            elite_k = max(1, int(self.elite_frac * self.pop_size))
            next_pop: List[Solution] = [copy.deepcopy(s) for s in pop[:elite_k]]

            # --- Tạo offspring
            while len(next_pop) < self.pop_size:
                p1 = self.tournament2(pop)
                p2 = self.tournament2(pop)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                next_pop.append(c1)
                if len(next_pop) < self.pop_size:
                    next_pop.append(c2)

            pop = sorted(next_pop, key=self.cost)
            cur = pop[0]
            cur_cost = self.cost(cur)

            if cur_cost + 1e-9 < best_cost:
                best, best_cost = cur, cur_cost
                no_improve = 0
            else:
                no_improve += 1

            if on_progress is not None:
                on_progress(gen, time.time() - t0, best_cost, best)

            # --- stopping conditions
            if gen >= self.max_generations: break
            if no_improve >= self.patience_generations: break
            if (time.time() - t0) >= time_limit_sec: break

        return best
