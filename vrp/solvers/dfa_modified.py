# vrp/solvers/dfa_modified.py
from __future__ import annotations
import random, copy
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem_modified import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified


class DFASolverPD(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 40, gamma: float = 0.95):
        super().__init__(problem, seed)
        self.pop_size = pop_size
        self.gamma = gamma
        random.seed(seed)

    # ------------------------- Basic helpers -------------------------

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

    def _customers_of_route(self, r: Route) -> List[int]:
        P = self.problem
        return [i for i in r.seq if not P.nodes[i].is_depot]

    def _rebuild(self, r: Route, new_customers: List[int]) -> None:
        """Cập nhật seq của route từ list khách, giữ đúng depot của xe."""
        P = self.problem
        dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
        r.seq = [dep] + new_customers + [dep]

    def _cost(self, s: Solution) -> float:
        c, _ = evaluate_modified(self.problem, s, return_details=False)
        return c

    # ------------------------- PD helpers -------------------------

    def _pd_maps(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Return:
          p2d: pickup -> delivery
          d2p: delivery -> pickup
        """
        p2d: Dict[int, int] = {}
        d2p: Dict[int, int] = {}
        pd = getattr(self.problem, "pd_pairs", {})
        if isinstance(pd, dict):
            for p, tup in pd.items():
                if isinstance(tup, (tuple, list)) and len(tup) >= 1:
                    d = int(tup[0])
                    p2d[int(p)] = d
                    d2p[d] = int(p)
        return p2d, d2p

    def _find_route_and_index(self, s: Solution, node: int) -> Tuple[Optional[int], Optional[int]]:
        """Tìm (route_index, seq_index) của node (theo chỉ số trong seq, kể cả depot)."""
        P = self.problem
        for ridx, r in enumerate(s.routes):
            for j, x in enumerate(r.seq):
                if not P.nodes[x].is_depot and x == node:
                    return ridx, j
        return None, None

    def _cust_positions(self, seq: List[int]) -> List[int]:
        """Trả về danh sách index trong seq là khách (bỏ depot)."""
        return [i for i, x in enumerate(seq) if not self.problem.nodes[x].is_depot]

    # ------------------------- Initialization -------------------------

    def _init_population(self) -> List[Solution]:
        """
        - Gán mỗi khách cho depot gần nhất.
        - Với mỗi depot: xáo trộn khách và chia đều cho các xe của depot đó.
        - Sau đó ép same-vehicle + precedence cho tất cả cặp PD.
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

            sol = Solution(routes=routes)
            self._co_locate_pd_same_vehicle(sol)  # ép PD ngay khi khởi tạo
            pop.append(sol)

        return pop

    def _co_locate_pd_same_vehicle(self, s: Solution) -> None:
        """
        Đưa mọi cặp p–d về cùng route (route của p) và đảm bảo d sau p.
        Thực hiện theo hai pha:
          (A) Nếu p & d cùng route nhưng d đứng trước p -> đổi chỗ (không động tới depot).
          (B) Nếu p & d khác route -> cắt d và chèn về route của p (chèn sau p, chọn vị trí rẻ nhất).
        """
        P = self.problem
        p2d, d2p = self._pd_maps()
        if not p2d:
            return

        # (A) Sửa precedence trong các route đã cùng cặp
        for ridx, r in enumerate(s.routes):
            seq = r.seq
            cust_idx = self._cust_positions(seq)
            if not cust_idx:
                continue
            nodes_k = [seq[i] for i in cust_idx]
            pos_of = {nodes_k[i]: cust_idx[i] for i in range(len(nodes_k))}
            changed = False
            for p, d in p2d.items():
                if p in pos_of and d in pos_of and pos_of[d] < pos_of[p]:
                    i, j = pos_of[p], pos_of[d]
                    seq[i], seq[j] = seq[j], seq[i]
                    changed = True
                    # cập nhật pos_of tối thiểu cho các cặp kế tiếp
                    pos_of[p], pos_of[d] = j, i
            if changed:
                s.routes[ridx].seq = seq

        # (B) Co-locate về route của pickup
        for p, d in p2d.items():
            r_p, j_p = self._find_route_and_index(s, p)
            r_d, j_d = self._find_route_and_index(s, d)
            if r_p is None or r_d is None:
                continue
            if r_p == r_d:
                continue  # đã cùng route

            # cắt d khỏi r_d
            seq_d = s.routes[r_d].seq
            # j_d hiện tại đã đúng trên s.routes[r_d].seq
            cut_d = seq_d.pop(j_d)
            assert cut_d == d

            # chèn d vào r_p, ngay sau p (cheapest theo cost thực)
            seq_p = s.routes[r_p].seq
            # vị trí hợp lệ: (j_p+1) .. (len(seq_p)-1) (trước depot cuối)
            best_cost, best_pos = float("inf"), None
            for pos in range(j_p + 1, len(seq_p)):
                cand = copy.deepcopy(s)
                cand.routes[r_p].seq = seq_p[:pos] + [d] + seq_p[pos:]
                c = self._cost(cand)
                if c < best_cost:
                    best_cost, best_pos = c, pos
            if best_pos is None:
                best_pos = min(len(seq_p) - 1, j_p + 1)
            s.routes[r_p].seq = seq_p[:best_pos] + [d] + seq_p[best_pos:]

    # ------------------------- PD-safe moves -------------------------

    def _reversal_keeps_precedence(self, seq: List[int], i: int, j: int) -> bool:
        """
        Kiểm tra đảo đoạn seq[i:j] (end-exclusive) có vi phạm precedence không.
        """
        P = self.problem
        p2d, d2p = self._pd_maps()
        if not p2d:
            return True

        after = seq[:i] + list(reversed(seq[i:j])) + seq[j:]
        # Lấy thứ tự ghé bỏ depot
        order = [x for x in after if not P.nodes[x].is_depot]

        # Vị trí node -> index trong order
        pos: Dict[int, int] = {}
        for idx, x in enumerate(order):
            pos[x] = idx

        for p, d in p2d.items():
            if p in pos and d in pos and pos[d] < pos[p]:
                return False
        return True

    def _move_two_opt(self, s: Solution) -> None:
        """2-opt an toàn precedence trong 1 route."""
        routes = [r for r in s.routes if len(self._customers_of_route(r)) >= 3]
        if not routes:
            return
        r = random.choice(routes)
        seq = r.seq
        cust_idx = [i for i, x in enumerate(seq) if not self.problem.nodes[x].is_depot]
        if len(cust_idx) < 3:
            return
        a = random.randrange(0, len(cust_idx) - 1)
        b = random.randrange(a + 1, len(cust_idx))
        i, j = cust_idx[a], cust_idx[b] + 1  # end-exclusive
        if self._reversal_keeps_precedence(seq, i, j):
            r.seq = seq[:i] + list(reversed(seq[i:j])) + seq[j:]

    def _move_intra_insertion(self, s: Solution) -> None:
        """Đổi vị trí 1 khách trong cùng route (an toàn PD)."""
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
        # chốt an toàn
        self._co_locate_pd_same_vehicle(s)

    def _move_bundle_relocate(self, s: Solution) -> None:
        P = self.problem
        p2d, d2p = self._pd_maps()

        all_nodes = [x for r in s.routes for x in r.seq if not P.nodes[x].is_depot]
        if not all_nodes:
            return
        x = random.choice(all_nodes)

        if x in p2d:
            p, d = x, p2d[x]
        elif x in d2p:
            p, d = d2p[x], x
        else:
            self._move_intra_insertion(s)
            return

        r_p, j_p = self._find_route_and_index(s, p)
        r_d, j_d = self._find_route_and_index(s, d)
        if r_p is None or r_d is None:
            return

        # cắt d khỏi route d (nếu khác route p)
        if r_p != r_d:
            s.routes[r_d].seq.pop(j_d)
            # NOTE: không evaluate ở đây

        # chèn d NGAY SAU p trong chính route p (một vị trí duy nhất)
        # cập nhật j_p nếu trước đó đã pop trong cùng route
        # (nếu r_p == r_d và j_d < j_p, sau khi pop, j_p -= 1)
        if r_p == r_d and j_d < j_p:
            j_p -= 1

        seq_p = s.routes[r_p].seq
        insert_pos = min(j_p + 1, len(seq_p) - 1)  # trước depot cuối
        seq_p.insert(insert_pos, d)


    def _random_move(self, s: Solution) -> None:
        mv = random.random()
        if mv < 0.60:
            self._move_bundle_relocate(s)
        elif mv < 0.90:
            self._move_two_opt(s)
        else:
            self._move_intra_insertion(s)

    @staticmethod
    def _flatten_order(P: Problem, sol: Solution) -> List[int]:
        out = []
        for r in sol.routes:
            out.extend([i for i in r.seq if not P.nodes[i].is_depot])
        return out

    @staticmethod
    def _hamming_like(P: Problem, a: Solution, b: Solution) -> int:
        A, B = DFASolverPD._flatten_order(P, a), DFASolverPD._flatten_order(P, b)
        L = min(len(A), len(B))
        return sum(1 for i in range(L) if A[i] != B[i]) + abs(len(A) - len(B))

    def solve(
        self,
        time_limit_sec: float = 1000000.0,
        patience_iters: int = 80,
        max_generations: int = 500,
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

        # “Độ sáng” = -cost -> cost thấp sáng hơn
        pop.sort(key=lambda s: -cost_of(s))
        best = pop[0]
        best_cost = cost_of(best)

        gen = 0
        no_improve = 0
        t0 = _time.time()

        while (_time.time() - t0) < time_limit_sec and (max_generations is None or gen < max_generations):
            new_pop: List[Solution] = []
            for i in range(len(pop)):
                xi = copy.deepcopy(pop[i])
                ci = cost_of(xi)

                # bay về các cá thể sáng hơn
                for j in range(len(pop)):
                    if j == i:
                        continue
                    xj = pop[j]
                    if -cost_of(xj) > -ci:  # xj sáng hơn xi (tức cost thấp hơn)
                        rij = self._hamming_like(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** gen)))
                        trials: List[Solution] = []
                        for _ in range(max(2, step_max)):
                            cand = copy.deepcopy(xi)
                            self._random_move(cand)
                            # chốt an toàn PD sau mỗi thử
                            self._co_locate_pd_same_vehicle(cand)
                            trials.append(cand)
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                # chốt an toàn trước khi thêm vào quần thể mới
                self._co_locate_pd_same_vehicle(xi)
                new_pop.append(xi)

            # chọn top pop_size (cost thấp nhất)
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

        self._co_locate_pd_same_vehicle(best)
        return best
