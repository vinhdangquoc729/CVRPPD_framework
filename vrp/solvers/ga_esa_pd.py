# vrp/solvers/ga_esa_pd.py
from __future__ import annotations
import random, math, time
from typing import List, Tuple, Dict, Set
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified as EVAL   # dùng evaluator PD

class GAESAPDSolver(Solver):
    """
    GA + (ESA-style mutation) cho VRP-PD (pickup-only / delivery-only, TW mềm).
    - Biểu diễn: danh sách Route (mỗi route: depot ... depot).
    - Fitness: evaluate_modified(problem, sol).
    - Crossover: Route-Exchange Crossover (REX).
    - Mutation: relocate/swap/2opt intra/inter.
    - Repair: depot đầu/cuối; pickup-before-delivery để tránh prefix tải âm.
    """

    def __init__(self,
                 problem: Problem,
                 seed: int = 42,
                 pop_size: int = 500,
                 elite_frac: float = 0.25,
                 tour_size: int = 3,
                 mut_rate: float = 0.3,
                 mut_steps: int = 3,
                 patience_iters: int = 200):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.tour_size = max(2, tour_size)
        self.mut_rate = mut_rate
        self.mut_steps = mut_steps
        self.patience_iters = patience_iters

    # ------------- util -------------
    def _key(self, s: Solution) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(r.seq) for r in s.routes)

    def _cost(self, s: Solution, cache: Dict) -> float:
        k = self._key(s)
        v = cache.get(k)
        if v is None:
            v, _ = EVAL(self.problem, s, return_details=False)
            cache[k] = v
        return v

    # ------------- khởi tạo -------------
    def _init_population(self) -> List[Solution]:
        P = self.problem
        rng = self.rng

        # nhóm customer theo depot gần nhất để “seed” ban đầu
        def nearest_depot(nid: int) -> int:
            return min(P.depots, key=lambda d: P.d(nid, d))

        custs = [i for i, nd in P.nodes.items() if not nd.is_depot]
        by_dep: Dict[int, List[int]] = {}
        for i in custs:
            d = nearest_depot(i)
            by_dep.setdefault(d, []).append(i)

        # vehicles by depot
        veh_by_dep: Dict[int, List] = {}
        for v in P.vehicles:
            veh_by_dep.setdefault(v.depot_id, []).append(v)

        pop: List[Solution] = []
        for _ in range(self.pop_size):
            routes: List[Route] = []
            for dep, vs in veh_by_dep.items():
                rng.shuffle(vs)
                cc = by_dep.get(dep, [])[:]
                rng.shuffle(cc)
                if not vs:
                    continue
                # chia đều khách vào các xe
                buckets = [[] for _ in range(len(vs))]
                for idx, c in enumerate(cc):
                    buckets[idx % len(vs)].append(c)
                # mỗi route: depot + khách đã chia + depot
                for k, v in enumerate(vs):
                    seq = [v.depot_id] + self._pickup_first_order(buckets[k]) + [v.depot_id]
                    routes.append(Route(vehicle_id=v.id, seq=seq))
            sol = Solution(routes=routes)
            self._repair_full(sol)    # sửa depot & pickup-first
            pop.append(sol)
        return pop

    # ------------- pickup-first & repair tải prefix -------------
    def _pickup_first_order(self, custs: List[int]) -> List[int]:
        """Heuristic: ưu tiên pickup trước delivery để giảm prefix âm (tham lam theo nhu cầu)."""
        P = self.problem
        rng = self.rng
        if len(custs) <= 2:
            rng.shuffle(custs)
            return custs
        picks = [i for i in custs if P.nodes[i].demand_pickup > 0]
        dels  = [i for i in custs if P.nodes[i].demand_delivery > 0]
        # sắp xếp pick giảm dần nhu cầu, delivery tăng dần (giao ít trước để giảm rủi ro)
        picks.sort(key=lambda i: -P.nodes[i].demand_pickup)
        dels.sort(key=lambda i: P.nodes[i].demand_delivery)
        # xen kẽ nhẹ để giảm vòng vo theo khoảng cách
        order = []
        while picks or dels:
            if picks:
                order.append(picks.pop(0))
            if rng.random() < 0.3 and picks:  # thỉnh thoảng thêm pick tiếp
                order.append(picks.pop(0))
            if dels:
                order.append(dels.pop(0))
        return order

    def _repair_route_prefix(self, r: Route) -> None:
        """Nếu delivery gây prefix âm => đẩy các pickup chưa ghé lên trước vị trí đó (tham lam)."""
        P = self.problem
        seq = r.seq
        if len(seq) <= 3:
            return
        # bỏ depot đầu/cuối
        dep = seq[0]
        inner = [i for i in seq[1:-1]]
        load = 0
        i = 0
        changed = False
        while i < len(inner):
            nid = inner[i]
            nd = P.nodes[nid]
            if nd.demand_pickup > 0:
                load += nd.demand_pickup
            if nd.demand_delivery > 0:
                if load < nd.demand_delivery:
                    # cần pickup trước: tìm pickup sau i để kéo lên
                    need = nd.demand_delivery - load
                    j = i + 1
                    pulled = False
                    while j < len(inner):
                        nj = inner[j]
                        ndj = P.nodes[nj]
                        if ndj.demand_pickup > 0:
                            # move nj lên trước i
                            inner.insert(i, inner.pop(j))
                            load += ndj.demand_pickup
                            changed = True
                            pulled = True
                            break
                        j += 1
                    if not pulled:
                        # không tìm thấy pickup để sửa, thôi tiếp (để evaluator phạt)
                        pass
                # sau khi (có thể) sửa, thực hiện giao
                if load >= nd.demand_delivery:
                    load -= nd.demand_delivery
            i += 1
        if changed:
            r.seq = [dep] + inner + [dep]

    def _repair_full(self, s: Solution) -> None:
        """Depot đầu/cuối + sửa prefix cho mọi route."""
        P = self.problem
        for r in s.routes:
            dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
            body = [i for i in r.seq if not P.nodes[i].is_depot]
            # pickup-first sơ bộ
            body = self._pickup_first_order(body)
            r.seq = [dep] + body + [dep]
            self._repair_route_prefix(r)

    # ------------- mutation (PD moves) -------------
    def _customer_positions(self, r: Route) -> List[int]:
        P = self.problem
        return [idx for idx, x in enumerate(r.seq) if not P.nodes[x].is_depot]

    def _mutate_once(self, s: Solution) -> None:
        """intra/inter relocate/swap + 2-opt (cấp–khách)."""
        P = self.problem
        rng = self.rng
        routes = [r for r in s.routes if len(self._customer_positions(r)) > 0]
        if not routes:
            return
        move = rng.choice(["intra_reloc", "intra_swap", "inter_reloc", "inter_swap", "two_opt"])
        if move in ("intra_reloc", "intra_swap", "two_opt"):
            r = rng.choice(routes)
            pos = self._customer_positions(r)
            if len(pos) < 2:
                return
            if move == "intra_reloc":
                i = rng.choice(pos)
                node = r.seq.pop(i)
                pos2 = self._customer_positions(r)
                j = rng.choice(pos2 + [pos2[-1] + 1])
                r.seq.insert(j, node)
            elif move == "intra_swap":
                i, j = rng.sample(pos, 2)
                r.seq[i], r.seq[j] = r.seq[j], r.seq[i]
            else:  # two_opt
                i, j = sorted(rng.sample(pos, 2))
                r.seq[i:j+1] = reversed(r.seq[i:j+1])
            self._repair_route_prefix(r)
        elif move == "inter_reloc":
            if len(routes) < 2: return
            ra, rb = rng.sample(routes, 2)
            pa = self._customer_positions(ra)
            if not pa: return
            ia = rng.choice(pa)
            node = ra.seq.pop(ia)
            pb = self._customer_positions(rb)
            ins = rng.choice(pb + [pb[-1] + 1]) if pb else 1
            rb.seq.insert(ins, node)
            self._repair_route_prefix(ra); self._repair_route_prefix(rb)
        else:  # inter_swap
            if len(routes) < 2: return
            ra, rb = rng.sample(routes, 2)
            pa = self._customer_positions(ra); pb = self._customer_positions(rb)
            if not pa or not pb: return
            ia = rng.choice(pa); ib = rng.choice(pb)
            ra.seq[ia], rb.seq[ib] = rb.seq[ib], ra.seq[ia]
            self._repair_route_prefix(ra); self._repair_route_prefix(rb)

    # ------------- crossover: Route-Exchange (REX) -------------
    def _rex_crossover(self, A: Solution, B: Solution) -> Solution:
        """Chép ngẫu nhiên một số route từ A, lấp phần còn lại theo thứ tự đường đi ở B."""
        P = self.problem
        rng = self.rng

        # map: vehicle_id -> depot_id
        veh_dep = {v.id: v.depot_id for v in P.vehicles}

        # copy ngẫu nhiên ~1/2 số route của A
        chosen_mask = [rng.random() < 0.5 for _ in A.routes]
        chosen_routes: List[Route] = []
        taken: Set[int] = set()  # khách đã có trong child

        for flag, r in zip(chosen_mask, A.routes):
            if not flag: 
                continue
            body = [i for i in r.seq if not P.nodes[i].is_depot]
            chosen_routes.append(Route(vehicle_id=r.vehicle_id, seq=[veh_dep[r.vehicle_id]] + body[:] + [veh_dep[r.vehicle_id]]))
            taken.update(body)

        # tạo khung cho mọi vehicle có mặt ở A hoặc B
        all_veh_ids = {r.vehicle_id for r in A.routes} | {r.vehicle_id for r in B.routes}
        base: Dict[int, List[int]] = {vid: [] for vid in all_veh_ids}
        for rt in chosen_routes:
            base[rt.vehicle_id] = [i for i in rt.seq if not P.nodes[i].is_depot]

        # duyệt B và lấp khách chưa có
        for r in B.routes:
            for i in r.seq:
                if P.nodes[i].is_depot: 
                    continue
                if i in taken:
                    continue
                # gán vào route có cùng depot (ưu tiên), nếu chưa có thì gán bất kỳ
                dep = veh_dep[r.vehicle_id]
                # tìm route child có depot này và còn hiện hữu
                candidates = [vid for vid in base.keys() if veh_dep.get(vid, dep) == dep]
                if not candidates:
                    candidates = list(base.keys())
                vid = rng.choice(candidates)
                base[vid].append(i)
                taken.add(i)

        # build routes + repair
        child_routes: List[Route] = []
        for vid, body in base.items():
            dep = veh_dep[vid]
            body2 = self._pickup_first_order(body)
            rt = Route(vehicle_id=vid, seq=[dep] + body2 + [dep])
            self._repair_route_prefix(rt)
            child_routes.append(rt)

        return Solution(routes=child_routes)

    # ------------- selection -------------
    def _tournament(self, pop: List[Solution], k: int, cache: Dict) -> Solution:
        rng = self.rng
        cand = self.rng.sample(pop, k)
        cand.sort(key=lambda s: self._cost(s, cache))
        return cand[0]

    # ------------- solve -------------
    def solve(self, time_limit_sec: float = 30.0, on_progress=None) -> Solution:
        rng = self.rng
        P = self.problem
        cache: Dict = {}

        pop = self._init_population()
        pop.sort(key=lambda s: self._cost(s, cache))
        best = pop[0]; best_cost = self._cost(best, cache)

        t0 = time.time()
        it = 0
        no_imp = 0

        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best)

        while time.time() - t0 < time_limit_sec:
            it += 1
            # elitism
            elite_k = max(1, int(self.elite_frac * self.pop_size))
            elites = sorted(pop, key=lambda s: self._cost(s, cache))[:elite_k]

            # tạo thế hệ mới
            new_pop: List[Solution] = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, self.tour_size, cache)
                p2 = self._tournament(pop, self.tour_size, cache)
                child = self._rex_crossover(p1, p2)

                # mutation + repair
                if rng.random() < self.mut_rate:
                    for _ in range(self.mut_steps):
                        self._mutate_once(child)
                self._repair_full(child)

                new_pop.append(child)

            pop = sorted(new_pop, key=lambda s: self._cost(s, cache))
            cur = pop[0]; cur_cost = self._cost(cur, cache)
            if cur_cost + 1e-9 < best_cost:
                best, best_cost = cur, cur_cost
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience_iters:
                    break

            if on_progress is not None:
                on_progress(it, time.time() - t0, best_cost, best)

        return best
