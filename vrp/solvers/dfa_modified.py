from __future__ import annotations
import random
from typing import List, Dict, Tuple
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified  # PD evaluator


class DFASolverPD(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 80, gamma: float = 0.95):
        super().__init__(problem, seed)
        self.pop_size = pop_size
        self.gamma = gamma
        random.seed(seed)

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

    def _init_population(self) -> List[Solution]:
        """
        - Gán mỗi khách cho depot gần nhất.
        - Với mỗi depot: xáo trộn khách và chia đều cho các xe của depot đó.
        """
        P = self.problem
        veh_by_dep = self._vehicles_by_depot()
        pop: List[Solution] = []

        custs = self._customers()

        home: Dict[int, List[int]] = {d: [] for d in P.depots}
        for c in custs:
            d = self._nearest_depot(c)
            home[d].append(c)

        for _ in range(self.pop_size):
            routes: List[Route] = []
            for d, vehs in veh_by_dep.items():
                if not vehs:
                    continue

                bag = home[d][:]
                random.shuffle(bag)

                buckets = [[] for _ in range(len(vehs))]
                for i, c in enumerate(bag):
                    buckets[i % len(vehs)].append(c)

                for k, v in enumerate(vehs):
                    seq = [d] + buckets[k] + [d]
                    routes.append(Route(vehicle_id=v.id, seq=seq))

            pop.append(Solution(routes=routes))
        return pop

    @staticmethod
    def _flatten_order(P: Problem, sol: Solution) -> List[int]:
        """Chuỗi khách theo thứ tự đi qua trong toàn bộ lời giải (bỏ depots)."""
        out = []
        for r in sol.routes:
            out.extend([i for i in r.seq if not P.nodes[i].is_depot])
        return out

    @staticmethod
    def _hamming_like(P: Problem, a: Solution, b: Solution) -> int:
        """Khoảng cách đơn giản: Hamming trên vị trí (sau khi làm phẳng)."""
        A, B = DFASolverPD._flatten_order(P, a), DFASolverPD._flatten_order(P, b)
        L = min(len(A), len(B))
        return sum(1 for i in range(L) if A[i] != B[i]) + abs(len(A) - len(B))

    def _customers_of_route(self, r: Route) -> List[int]:
        P = self.problem
        return [i for i in r.seq if not P.nodes[i].is_depot]

    def _rebuild(self, r: Route, new_customers: List[int]) -> None:
        """Cập nhật seq của route từ list khách, giữ depot đúng của xe."""
        P = self.problem
        dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
        r.seq = [dep] + new_customers + [dep]

    # ---- Atomic moves (customer level) ----
    def _move_intra_insertion(self, s: Solution) -> None:
        """Đổi vị trí 1 khách trong cùng route."""
        routes = [r for r in s.routes if len(self._customers_of_route(r)) >= 2]
        if not routes:
            return
        r = random.choice(routes)
        C = self._customers_of_route(r)
        i = random.randrange(len(C))
        x = C.pop(i)
        j = random.randrange(len(C) + 1)
        C.insert(j, x)
        self._rebuild(r, C)

    def _move_inter_relocate(self, s: Solution) -> None:
        """Chuyển 1 khách từ route A -> route B (có thể trùng A)."""
        rs = [r for r in s.routes if len(self._customers_of_route(r)) >= 1]
        if not rs:
            return
        rA = random.choice(rs)
        C_A = self._customers_of_route(rA)
        x = C_A.pop(random.randrange(len(C_A)))

        rB = random.choice(s.routes)  # có thể trùng A
        if rB is rA:
            j = random.randrange(len(C_A) + 1)
            C_A.insert(j, x)
            self._rebuild(rA, C_A)
        else:
            C_B = self._customers_of_route(rB)
            j = random.randrange(len(C_B) + 1)
            C_B.insert(j, x)
            self._rebuild(rA, C_A)
            self._rebuild(rB, C_B)

    def _move_inter_swap(self, s: Solution) -> None:
        """Hoán đổi 1 khách giữa 2 route."""
        rs = [r for r in s.routes if len(self._customers_of_route(r)) >= 1]
        if len(rs) < 2:
            return
        rA, rB = random.sample(rs, 2)
        C_A = self._customers_of_route(rA)
        C_B = self._customers_of_route(rB)
        i = random.randrange(len(C_A))
        j = random.randrange(len(C_B))
        C_A[i], C_B[j] = C_B[j], C_A[i]
        self._rebuild(rA, C_A)
        self._rebuild(rB, C_B)

    def _move_two_opt(self, s: Solution) -> None:
        """2-opt trong 1 route (đảo 1 đoạn khách)."""
        routes = [r for r in s.routes if len(self._customers_of_route(r)) >= 3]
        if not routes:
            return
        r = random.choice(routes)
        C = self._customers_of_route(r)
        i = random.randrange(0, len(C) - 1)
        j = random.randrange(i + 1, len(C))
        C[i:j] = reversed(C[i:j])
        self._rebuild(r, C)

    def _random_move(self, s: Solution) -> None:
        """Chọn ngẫu nhiên 1 move."""
        mv = random.random()
        if mv < 0.30:
            self._move_intra_insertion(s)
        elif mv < 0.65:
            self._move_inter_relocate(s)
        elif mv < 0.85:
            self._move_inter_swap(s)
        else:
            self._move_two_opt(s)

    def solve(
        self,
        time_limit_sec: float = 10000.0,
        patience_iters: int = 25,
        max_generations: int = 1000,
    ) -> Solution:
        import time as _time
        P = self.problem
        pop = self._init_population()

        # Cache chi phí để tránh tính lại nhiều lần
        cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}

        def key_of(s: Solution):
            return tuple(tuple(r.seq) for r in s.routes)

        def cost_of(s: Solution) -> float:
            k = key_of(s)
            c = cost_cache.get(k)
            if c is None:
                c, _ = evaluate_modified(P, s, return_details=False)
                cost_cache[k] = c
            return c

        # chi phí càng thấp, độ sáng càng cao
        pop.sort(key=lambda s: -cost_of(s))
        best = pop[0]
        best_cost = cost_of(best)

        gen = 0
        no_improve = 0
        t0 = _time.time()

        while (_time.time() - t0) < time_limit_sec and (max_generations is None or gen < max_generations):
            new_pop: List[Solution] = []
            for i in range(len(pop)):
                xi = pop[i]
                ci = cost_of(xi)

                # bay về các cá thể sáng hơn
                for j in range(len(pop)):
                    if j == i:
                        continue
                    xj = pop[j]
                    if -cost_of(xj) > -ci:  # xj sáng hơn xi
                        rij = self._hamming_like(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** gen)))
                        trials: List[Solution] = []
                        for _ in range(max(2, step_max)):
                            import copy
                            cand = copy.deepcopy(xi)
                            self._random_move(cand)
                            trials.append(cand)
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                new_pop.append(xi)

            # chọn top pop_size
            pop = sorted(new_pop, key=cost_of)[:self.pop_size]

            # cập nhật best
            cur = pop[0]
            cur_cost = cost_of(cur)
            if cur_cost < best_cost:
                best, best_cost = cur, cur_cost
                no_improve = 0
            else:
                no_improve += 1

            gen += 1
            if patience_iters and no_improve >= patience_iters:
                break

        return best
