# vrp/solvers/ga_pd_hct.py
from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified  # dùng evaluator PD của bạn

@dataclass
class Head:
    priority: List[int]                 # permutation các vehicle indices (theo id trong Problem.vehicles)
    routes_per_vehicle: List[int]       # len = |V|
    nodes_per_route: List[int]          # tổng các route
    orders_per_node: List[int]          # tổng khách của tất cả route (không gồm các depot)

@dataclass
class EncodedSolution:
    head: Head
    core_routes: List[List[int]]        # các route: [depot, ...customers..., depot]
    tail_orders: List[List[int]]        # song song core_routes, số đơn tại từng KH (mặc định 1)

# ---------- encode/decode ----------
def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    # gom route theo vehicle id
    by_vid: Dict[int, List[List[int]]] = {}
    for r in sol.routes:
        by_vid.setdefault(r.vehicle_id, []).append(list(r.seq))

    vids_all = [v.id for v in P.vehicles]
    # ưu tiên: xe được dùng (nhiều KH hơn) đứng trước, sau đó theo id
    def served_cnt(seq: List[int]) -> int:
        return sum(1 for i in seq if not P.nodes[i].is_depot)

    used = [(vid, sum(served_cnt(seq) for seq in by_vid.get(vid, []))) for vid in vids_all]
    priority = [vid for vid, _ in sorted(used, key=lambda t: (-t[1], t[0]))]

    routes_per_vehicle = [len(by_vid.get(vid, [])) for vid in vids_all]

    # nodes_per_route theo thứ tự duyệt vehicle = priority, sau đó từng route theo thứ tự hiện tại
    core_routes: List[List[int]] = []
    nodes_per_route: List[int] = []
    tail_orders: List[List[int]] = []
    orders_per_node_flat: List[int] = []

    for vid in priority:
        for seq in by_vid.get(vid, []):
            if not seq:
                continue
            core_routes.append(list(seq))
            nodes_per_route.append(len(seq))
            # tail: số đơn cho từng KH của route (mặc định 1)
            kh = [i for i in seq if not P.nodes[i].is_depot]
            tail_orders.append([1] * len(kh))
            orders_per_node_flat.extend([1] * len(kh))

    head = Head(
        priority=priority,
        routes_per_vehicle=[len(by_vid.get(vid, [])) for vid in vids_all],
        nodes_per_route=nodes_per_route,
        orders_per_node=orders_per_node_flat,
    )
    return EncodedSolution(head=head, core_routes=core_routes, tail_orders=tail_orders)


def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    # phân bổ các route trong core theo head.priority và routes_per_vehicle
    routes: List[Route] = []
    core_idx = 0
    # map: vị trí trong danh sách vehicles theo id -> index trong P.vehicles
    id2vehidx = {v.id: k for k, v in enumerate(P.vehicles)}
    for vid in enc.head.priority:
        num = enc.head.routes_per_vehicle[id2vehidx[vid]]
        for _ in range(num):
            if core_idx >= len(enc.core_routes):
                break
            seq = list(enc.core_routes[core_idx])
            # bảo đảm depot đầu/cuối đúng depot của vehicle
            dep = P.vehicles[id2vehidx[vid]].depot_id
            if not seq or seq[0] != dep:
                seq = [dep] + [x for x in seq if not P.nodes[x].is_depot] + [dep]
            if seq[-1] != dep:
                seq = [x for x in seq if not P.nodes[x].is_depot]
                seq = [dep] + seq + [dep]
            routes.append(Route(vehicle_id=vid, seq=seq))
            core_idx += 1

    # thêm những xe chưa có route: route rỗng [depot, depot]
    have_vids = {r.vehicle_id for r in routes}
    for v in P.vehicles:
        if v.id not in have_vids:
            dep = v.depot_id
            routes.append(Route(vehicle_id=v.id, seq=[dep, dep]))

    return Solution(routes=routes)

# =========================
#   GA (Head–Core–Tail)
# =========================

class GAPD_HCT_Solver(Solver):
    def __init__(self,
                 problem: Problem,
                 seed: int = 42,
                 pop_size: int = 80,
                 elite_frac: float = 0.10,
                 tournament_k: int = 2,
                 p_cx: float = 0.9,
                 p_mut: float = 0.2,
                 max_generations: int = 1000,
                 patience: int = 50,
                 power_k: float = 1.0):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.p_cx = p_cx
        self.p_mut = p_mut
        self.max_generations = max_generations
        self.patience = patience
        self.power_k = max(1.0, power_k)

    # ---------- utils ----------
    def _cost(self, s: Solution) -> float:
        c, _ = evaluate_modified(self.problem, s, return_details=False)
        return c

    def _fitness_from_costs(self, costs: List[float]) -> List[float]:
        # power-law scaling Fi = (Amax - Ai)^k - Amin  (đổi sign để Fi >= 0)
        Amax = max(costs)
        Amin = min(costs)
        k = self.power_k
        return [((Amax - Ai) ** k) - Amin for Ai in costs]

    # ---------- init ----------
    def _random_initial_solution(self) -> Solution:
        # không dùng nữa: giữ để tương thích, gọi bản guided
        return self._build_initial_solution_guided()

    def _init_population(self) -> List[EncodedSolution]:
        """
        Tạo quần thể ban đầu có định hướng (guided), nhưng vẫn có đa dạng nho nhỏ
        qua shuffle/2-opt-light và số route per depot dao động nhẹ.
        """
        pop: List[EncodedSolution] = []
        for _ in range(self.pop_size):
            s = self._build_initial_solution_guided()
            pop.append(encode_from_solution(self.problem, s))
        return pop
    def _vehicles_by_depot(self) -> Dict[int, List[int]]:
        by_dep: Dict[int, List[int]] = {}
        for v in self.problem.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v.id)
        return by_dep

    def _nearest_depot(self, nid: int) -> int:
        P = self.problem
        return min(P.depots, key=lambda d: P.d(nid, d))

    def _assign_customers_to_depots(self) -> Dict[int, List[int]]:
        """Gán mỗi khách về depot gần nhất (theo khoảng cách Euclid của P.d)."""
        P = self.problem
        cust_by_dep: Dict[int, List[int]] = {}
        for nid, nd in P.nodes.items():
            if not nd.is_depot:
                d = self._nearest_depot(nid)
                cust_by_dep.setdefault(d, []).append(nid)
        return cust_by_dep

    def _nearest_neighbor_order(self, depot_id: int, customers: List[int]) -> List[int]:
        """Sắp thứ tự khách theo NN rồi lắc rất nhẹ (đảo 1 đoạn nhỏ) để đa dạng."""
        P, rng = self.problem, self.rng
        if not customers:
            return [depot_id, depot_id]
        unvis = set(customers)
        # bắt đầu từ khách gần depot nhất
        cur = min(unvis, key=lambda j: P.d(depot_id, j))
        route = [depot_id, cur]
        unvis.remove(cur)
        while unvis:
            nxt = min(unvis, key=lambda j: P.d(route[-1], j))
            route.append(nxt)
            unvis.remove(nxt)
        route.append(depot_id)

        # lắc nhẹ: đảo một đoạn ngắn nếu có thể
        if len(route) > 5 and rng.random() < 0.4:
            i = rng.randint(1, len(route) - 3)
            j = rng.randint(i + 1, len(route) - 2)
            route[i:j+1] = reversed(route[i:j+1])

        # tối ưu 2-opt nhẹ
        return self._two_opt_light(route)

    def _two_opt_light(self, seq: List[int]) -> List[int]:
        """2-opt nhẹ chỉ trên phần khách (không động tới depot đầu/cuối)."""
        P = self.problem
        if len(seq) < 6:
            return seq[:]
        best = seq[:]
        improved = True
        # chỉ cho phép ~ vài chục thử để giữ nhanh
        tries = 0
        while improved and tries < 50:
            improved = False
            tries += 1
            # chỉ xét các cạnh trong vùng khách
            for a in range(1, len(best) - 3):
                for b in range(a + 1, len(best) - 1):
                    if b == a + 1:
                        continue
                    old = P.d(best[a-1], best[a]) + P.d(best[b], best[b+1])
                    new = P.d(best[a-1], best[b]) + P.d(best[a], best[b+1])
                    if new + 1e-9 < old:
                        best[a:b+1] = reversed(best[a:b+1])
                        improved = True
                        break
                if improved:
                    break
        return best

    def _balanced_split(self, items: List[int], k: int) -> List[List[int]]:
        """Chia items thành k phần (xấp xỉ cân bằng)."""
        k = max(1, k)
        n = len(items)
        if k >= n:
            return [[x] for x in items] + [[] for _ in range(k - n)]
        # round-robin để đều
        parts = [[] for _ in range(k)]
        for i, x in enumerate(items):
            parts[i % k].append(x)
        return parts

    def _build_initial_solution_guided(self) -> Solution:
        """
        Dựng nghiệm khởi tạo có định hướng:
        - Gán khách -> depot gần nhất
        - Với mỗi depot, chia đều khách cho số xe của depot (round-robin)
        - Mỗi nhóm khách tạo 1 route NN + 2-opt nhẹ
        """
        P, rng = self.problem, self.rng
        routes: List[Route] = []

        cust_by_dep = self._assign_customers_to_depots()
        veh_by_dep  = self._vehicles_by_depot()

        for dep_id, custs in cust_by_dep.items():
            vids = veh_by_dep.get(dep_id, [])
            if not vids:
                # nếu depot không có xe (hiếm), bỏ qua
                continue

            # một ít đa dạng: đôi khi dùng nhiều route hơn số xe (nhưng ≤ 2x)
            factor = 1 if len(custs) <= 12 else (2 if rng.random() < 0.35 else 1)
            k_routes = max(1, min(len(vids) * factor, len(custs)))

            # xáo nhẹ để đa dạng giữa các cá thể, nhưng vẫn theo depot
            tmp = custs[:]
            rng.shuffle(tmp)
            groups = self._balanced_split(tmp, k_routes)

            # round-robin gán route -> xe
            for i, g in enumerate(groups):
                vid = vids[i % len(vids)]
                seq = self._nearest_neighbor_order(dep_id, g)
                routes.append(Route(vehicle_id=vid, seq=seq))

        # thêm xe không có khách: route rỗng (depot, depot)
        used_vids = {r.vehicle_id for r in routes}
        for v in P.vehicles:
            if v.id not in used_vids:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))

        return Solution(routes=routes)

    # ---------- decode+evaluate ----------
    def _decode_cost(self, enc: EncodedSolution) -> float:
        return self._cost(decode_to_solution(self.problem, enc))

    # ---------- selection ----------
    def _tournament(self, population: List[EncodedSolution], costs: List[float]) -> EncodedSolution:
        rng = self.rng
        k = self.tournament_k
        idxs = rng.sample(range(len(population)), k)
        best = min(idxs, key=lambda i: costs[i])
        return copy.deepcopy(population[best])

    # ---------- crossover: Best-cost route crossover (xấp xỉ) ----------
    def _route_segments(self, enc: EncodedSolution) -> List[Tuple[int, List[int]]]:
        # trả về list (route_index, customer_list)
        P = self.problem
        segs = []
        for ridx, seq in enumerate(enc.core_routes):
            customers = [i for i in seq if not P.nodes[i].is_depot]
            if customers:
                segs.append((ridx, customers))
        return segs

    def _insert_segment_best(self, enc: EncodedSolution, route_from: int, seg_customers: List[int]) -> None:
        P = self.problem
        rng = self.rng

        # loại toàn bộ khách trong segment khỏi MỌI route (quan trọng!)
        self._remove_customers(enc, seg_customers)

        base_cost = self._decode_cost(enc)
        best_cost = math.inf
        best_place = None

        cand_routes = list(range(len(enc.core_routes)))
        rng.shuffle(cand_routes)
        cand_routes = cand_routes[:min(8, len(cand_routes))]

        for ridx in cand_routes:
            route = enc.core_routes[ridx]
            cust_idx = [i for i, x in enumerate(route) if not P.nodes[x].is_depot]
            try_positions = [1] if not cust_idx else list(range(cust_idx[0], cust_idx[-1] + 2))
            for pos in try_positions:
                tmp = copy.deepcopy(enc)
                tmp.core_routes[ridx] = route[:pos] + seg_customers + route[pos:]
                cost = self._decode_cost(tmp)
                if cost < best_cost:
                    best_cost, best_place = cost, (ridx, pos)

        if best_place is None:
            if not enc.core_routes: return
            ridx, pos = 0, 1
        else:
            ridx, pos = best_place

        enc.core_routes[ridx] = enc.core_routes[ridx][:pos] + seg_customers + enc.core_routes[ridx][pos:]

        
    # --- helpers: customers universe ---
    def _all_customers(self) -> List[int]:
        P = self.problem
        return [i for i, nd in P.nodes.items() if not nd.is_depot]

    def _remove_customers(self, enc: EncodedSolution, custs: List[int]) -> None:
        S = set(custs)
        P = self.problem
        for ridx, seq in enumerate(enc.core_routes):
            keep = []
            for x in seq:
                if P.nodes[x].is_depot or x not in S:
                    keep.append(x)
            # đảm bảo còn depot đầu/cuối
            if len(keep) < 2:
                keep = [seq[0], seq[-1]]
            enc.core_routes[ridx] = keep

    def _repair_uniqueness(self, enc: EncodedSolution) -> None:
        """Đảm bảo mỗi khách xuất hiện đúng 1 lần; chèn khách thiếu bằng cheapest-insertion."""
        P = self.problem
        universe = set(self._all_customers())
        seen = set()
        # 1) loại trùng trong từng route (giữ lần đầu)
        for ridx, seq in enumerate(enc.core_routes):
            new_seq = []
            for x in seq:
                if P.nodes[x].is_depot:
                    new_seq.append(x)
                else:
                    if x not in seen:
                        new_seq.append(x); seen.add(x)
            # bảo toàn depot
            if not new_seq or P.nodes[new_seq[0]].is_depot is False:
                new_seq = [seq[0]] + [k for k in new_seq if not P.nodes[k].is_depot]
            if P.nodes[new_seq[-1]].is_depot is False:
                new_seq = new_seq + [seq[-1]]
            enc.core_routes[ridx] = new_seq

        missing = list(universe - seen)
        if not missing:
            return

        # 2) cheapest insertion các khách thiếu
        for c in missing:
            best = (math.inf, None, None)  # (cost, ridx, pos)
            for ridx, route in enumerate(enc.core_routes):
                # vị trí chèn giữa depot đầu và depot cuối (trước depot cuối)
                cust_idx = [i for i, x in enumerate(route) if not P.nodes[x].is_depot]
                if not cust_idx:
                    try_positions = [1]
                else:
                    try_positions = list(range(cust_idx[0], cust_idx[-1] + 2))
                for pos in try_positions:
                    tmp = copy.deepcopy(enc)
                    tmp.core_routes[ridx] = route[:pos] + [c] + route[pos:]
                    cost = self._decode_cost(tmp)
                    if cost < best[0]:
                        best = (cost, ridx, pos)
            if best[1] is None:
                # fallback: nếu cá thể chưa có route, bỏ qua (hiếm)
                continue
            ridx, pos = best[1], best[2]
            enc.core_routes[ridx] = enc.core_routes[ridx][:pos] + [c] + enc.core_routes[ridx][pos:]

    def _crossover(self, A: EncodedSolution, B: EncodedSolution) -> Tuple[EncodedSolution, EncodedSolution]:
        rng = self.rng
        a = copy.deepcopy(A)
        b = copy.deepcopy(B)

        segs_a = self._route_segments(a)
        segs_b = self._route_segments(b)
        if not segs_a or not segs_b:
            return a, b

        ra, seg_a = rng.choice(segs_a)
        rb, seg_b = rng.choice(segs_b)

        # swap theo best-cost insertion
        self._insert_segment_best(a, ra, seg_b)
        self._insert_segment_best(b, rb, seg_a)

        return a, b

    # ---------- mutation: invert 1 đoạn trong 1 route ----------
    def _mutation(self, enc: EncodedSolution) -> None:
        rng = self.rng
        if not enc.core_routes:
            return
        ridx = rng.randrange(len(enc.core_routes))
        route = enc.core_routes[ridx]
        if len(route) <= 3:
            return
        # chỉ chọn trong phần khách
        cust_idx = [i for i, x in enumerate(route) if not self.problem.nodes[x].is_depot]
        if len(cust_idx) < 2:
            return
        i, j = sorted(rng.sample(cust_idx, 2))
        enc.core_routes[ridx] = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]

    # ---------- solve ----------
    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        rng = self.rng
        t0 = time.time()

        pop = self._init_population()
        costs = [self._decode_cost(ind) for ind in pop]
        best_idx = min(range(len(pop)), key=lambda i: costs[i])
        best_enc = copy.deepcopy(pop[best_idx])
        best_cost = costs[best_idx]

        patience = 0
        gen = 0

        while (time.time() - t0) < time_limit_sec and gen < self.max_generations and patience < self.patience:
            gen += 1

            # --- selection + elitism ---
            elite_k = max(1, int(self.elite_frac * self.pop_size))
            elites_idx = sorted(range(len(pop)), key=lambda i: costs[i])[:elite_k]
            new_pop: List[EncodedSolution] = [copy.deepcopy(pop[i]) for i in elites_idx]

            # fill rest
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament(pop, costs)
                parent2 = self._tournament(pop, costs)

                # crossover
                if rng.random() < self.p_cx:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # mutation
                if rng.random() < self.p_mut:
                    self._mutation(child1)
                if rng.random() < self.p_mut and len(new_pop) + 1 < self.pop_size:
                    self._mutation(child2)
                self._repair_uniqueness(child1)
                self._repair_uniqueness(child2)
                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(child2)

            pop = new_pop
            costs = [self._decode_cost(ind) for ind in pop]

            # update best
            cur_idx = min(range(len(pop)), key=lambda i: costs[i])
            cur_cost = costs[cur_idx]
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_enc = copy.deepcopy(pop[cur_idx])
                patience = 0
            else:
                patience += 1

        return decode_to_solution(self.problem, best_enc)
