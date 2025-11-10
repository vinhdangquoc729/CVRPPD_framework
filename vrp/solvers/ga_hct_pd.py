from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified

@dataclass
class Head:
    """
    Head:
    - priority: hoán vị các chỉ số xe (0..|V|-1) theo mức độ ưu tiên phân bổ route.
    - routes_per_vehicle: số route của từng xe (theo chỉ số xe).
    - nodes_per_route: số node (bao gồm depot đầu/cuối) của từng route trong core (tuần tự).
    - orders_per_node: số đơn trên từng KH (flatten theo thứ tự các route trong core; depot không có đơn).
    """
    priority: List[int] 
    routes_per_vehicle: List[int]
    nodes_per_route: List[int]
    orders_per_node: List[int]

@dataclass
class EncodedSolution:
    head: Head
    core_routes: List[List[int]]        # các route: [depot, ...customers..., depot]
    tail_orders: List[List[int]]        # song song core_routes, số đơn tại từng KH (mặc định 1)

def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    """
    Mã hoá H–C–T từ Solution:
    - Gom route theo chỉ số xe (index trong P.vehicles).
    - Head.priority = hoán vị chỉ số xe, sắp theo số khách phục vụ (giảm dần), rồi theo chỉ số xe (tăng dần).
    """
    # Map: index xe -> danh sách route.seq
    by_vidx: Dict[int, List[List[int]]] = {}
    id2vidx = {v.id: i for i, v in enumerate(P.vehicles)}
    for r in sol.routes:
        vidx = id2vidx[r.vehicle_id]
        by_vidx.setdefault(vidx, []).append(list(r.seq))

    nV = len(P.vehicles)

    def count_customers(seq: List[int]) -> int:
        return sum(1 for x in seq if not P.nodes[x].is_depot)

    used = [(vidx, sum(count_customers(seq) for seq in by_vidx.get(vidx, [])))
            for vidx in range(nV)]
    priority: List[int] = [vidx for vidx, _ in sorted(used, key=lambda t: (-t[1], t[0]))]

    routes_per_vehicle = [len(by_vidx.get(vidx, [])) for vidx in range(nV)]

    # duyệt theo priority
    core_routes: List[List[int]] = []
    nodes_per_route: List[int] = []
    tail_orders: List[List[int]] = []
    orders_per_node_flat: List[int] = []

    for vidx in priority:
        for seq in by_vidx.get(vidx, []):
            if not seq:
                continue
            core_routes.append(list(seq))
            nodes_per_route.append(len(seq))
            # tail: số đơn cho từng KH của route (mặc định 1)
            kh = [i for i in seq if not P.nodes[i].is_depot]
            tail_vec = [1] * len(kh)
            tail_orders.append(tail_vec)
            orders_per_node_flat.extend(tail_vec)

    head = Head(
        priority=priority,       
        routes_per_vehicle=routes_per_vehicle,
        nodes_per_route=nodes_per_route, 
        orders_per_node=orders_per_node_flat,
    )
    return EncodedSolution(head=head, core_routes=core_routes, tail_orders=tail_orders)


def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    """
    Giải mã EncodedSolution về Solution:
    - Dùng head.priority (chỉ số xe) + routes_per_vehicle để phân bổ các route trong core.
    - Dùng nodes_per_route để chuẩn hoá độ dài chuỗi (cắt/pad an toàn nếu cần).
    - Tail (orders) hiện vẫn là 1 đơn/KH; phần này chủ yếu để đồng bộ độ dài và mở đường
      cho các biến thể nhiều đơn trong tương lai.
    """
    routes: List[Route] = []
    core_idx = 0
    nV = len(P.vehicles)

    # Chuẩn hoá: nếu priority bị thiếu/thừa chỉ số, tự hiệu chỉnh tối thiểu
    seen = set(enc.head.priority)
    if len(enc.head.priority) != nV or any(v >= nV or v < 0 for v in enc.head.priority):
        enc.head.priority = list(range(nV))
    elif len(seen) != len(enc.head.priority):
        # nếu có lặp, thay bằng permutation mặc định
        enc.head.priority = list(range(nV))

    # Số route dự kiến theo xe (theo chỉ số xe)
    rpv = enc.head.routes_per_vehicle[:]
    if len(rpv) != nV:  # tự vá khi config không đồng bộ
        rpv = [0] * nV

    # Giới hạn vector nodes_per_route theo số route có thực
    npr = enc.head.nodes_per_route[:]
    if len(npr) != len(enc.core_routes):
        # fallback: khớp độ dài core_routes
        npr = [len(cr) for cr in enc.core_routes]

    # Duyệt theo thứ tự ưu tiên của xe (chỉ số xe)
    for vidx in enc.head.priority:
        need = rpv[vidx] if 0 <= vidx < nV else 0
        for _ in range(max(0, need)):
            if core_idx >= len(enc.core_routes):
                break
            seq = list(enc.core_routes[core_idx])
            # chuẩn hoá theo nodes_per_route (nếu có)
            target_len = npr[core_idx] if core_idx < len(npr) else len(seq)
            if target_len >= 2:
                # cắt/pad an toàn (giữ depot đầu/cuối đúng của xe)
                dep = P.vehicles[vidx].depot_id
                # lấy phần khách
                customers = [x for x in seq if not P.nodes[x].is_depot]
                # dựng lại theo target_len
                inner_need = max(0, target_len - 2)  # phần khách cần
                if inner_need < len(customers):
                    customers = customers[:inner_need]
                # nếu thiếu khách để đạt target_len -> cứ để nguyên, không pad KH ảo
                seq = [dep] + customers + [dep]
            else:
                # bảo đảm tối thiểu [dep, dep]
                dep = P.vehicles[vidx].depot_id
                seq = [dep, dep]

            # chắc chắn depot đúng của xe
            dep = P.vehicles[vidx].depot_id
            if not seq or seq[0] != dep:
                seq = [dep] + [x for x in seq if not P.nodes[x].is_depot] + [dep]
            if seq[-1] != dep:
                seq = [x for x in seq if not P.nodes[x].is_depot]
                seq = [dep] + seq + [dep]

            routes.append(Route(vehicle_id=P.vehicles[vidx].id, seq=seq))
            core_idx += 1

    # thêm các xe chưa có route -> route rỗng [dep, dep]
    have_vids = {r.vehicle_id for r in routes}
    for vidx, v in enumerate(P.vehicles):
        if v.id not in have_vids:
            routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))

    return Solution(routes=routes)

class GA_HCT_PD_Solver(Solver):
    """
    GA–HCT cho VRP-PD:
    - Khởi tạo guided (gán KH -> depot gần nhất; NN + 2-opt nhẹ; chia đều theo depot).
    - Crossover: hoán đổi đoạn route và chèn theo vị trí best-cost.
    - Mutation: đảo một đoạn trong một route.
    - Selection & Elitism: dùng FITNESS power-law (không so sánh cost thô).
    """
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
        """
        Power-law fitness:
           Fi = (Amax − Ai)^k − Amin
        - Cost càng thấp => Fitness càng cao.
        - Trả về vector fitness (cùng chiều với costs).
        """
        Amax = max(costs)
        Amin = min(costs)
        k = self.power_k
        # đảm bảo không NaN khi tất cả bằng nhau
        if abs(Amax - Amin) < 1e-12:
            return [1.0] * len(costs)
        return [((Amax - Ai) ** k) - Amin for Ai in costs]

    def _random_initial_solution(self) -> Solution:
        return self._build_initial_solution_guided()

    def _init_population(self) -> List[EncodedSolution]:
        pop: List[EncodedSolution] = []
        for _ in range(self.pop_size):
            s = self._build_initial_solution_guided()
            pop.append(encode_from_solution(self.problem, s))
        return pop

    def _vehicles_by_depot(self) -> Dict[int, List[int]]:
        by_dep: Dict[int, List[int]] = {}
        for vidx, v in enumerate(self.problem.vehicles):
            by_dep.setdefault(v.depot_id, []).append(vidx)   # LƯU CHỈ SỐ XE
        return by_dep

    def _nearest_depot(self, nid: int) -> int:
        P = self.problem
        return min(P.depots, key=lambda d: P.d(nid, d))

    def _assign_customers_to_depots(self) -> Dict[int, List[int]]:
        P = self.problem
        cust_by_dep: Dict[int, List[int]] = {}
        for nid, nd in P.nodes.items():
            if not nd.is_depot:
                d = self._nearest_depot(nid)
                cust_by_dep.setdefault(d, []).append(nid)
        return cust_by_dep

    def _nearest_neighbor_order(self, depot_id: int, customers: List[int]) -> List[int]:
        P, rng = self.problem, self.rng
        if not customers:
            return [depot_id, depot_id]
        unvis = set(customers)
        cur = min(unvis, key=lambda j: P.d(depot_id, j))
        route = [depot_id, cur]
        unvis.remove(cur)
        while unvis:
            nxt = min(unvis, key=lambda j: P.d(route[-1], j))
            route.append(nxt)
            unvis.remove(nxt)
        route.append(depot_id)
        if len(route) > 5 and rng.random() < 0.4:
            i = rng.randint(1, len(route) - 3)
            j = rng.randint(i + 1, len(route) - 2)
            route[i:j+1] = reversed(route[i:j+1])
        return self._two_opt_light(route)

    def _two_opt_light(self, seq: List[int]) -> List[int]:
        P = self.problem
        if len(seq) < 6:
            return seq[:]
        best = seq[:]
        improved, tries = True, 0
        while improved and tries < 50:
            improved = False
            tries += 1
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
        k = max(1, k)
        n = len(items)
        if k >= n:
            return [[x] for x in items] + [[] for _ in range(k - n)]
        parts = [[] for _ in range(k)]
        for i, x in enumerate(items):
            parts[i % k].append(x)
        return parts

    def _build_initial_solution_guided(self) -> Solution:
        """
        Gán khách hàng -> depot gần nhất, chia đều theo số xe tại depot (dùng CHỈ SỐ XE),
        sắp nearest neighbor + 2-opt.
        """
        P, rng = self.problem, self.rng
        routes: List[Route] = []

        cust_by_dep = self._assign_customers_to_depots()
        vidx_by_dep  = self._vehicles_by_depot()    # trả về CHỈ SỐ XE

        for dep_id, custs in cust_by_dep.items():
            vidxs = vidx_by_dep.get(dep_id, [])
            if not vidxs:
                continue

            factor = 1 if len(custs) <= 12 else (2 if rng.random() < 0.35 else 1)
            k_routes = max(1, min(len(vidxs) * factor, len(custs)))

            tmp = custs[:]
            rng.shuffle(tmp)
            groups = self._balanced_split(tmp, k_routes)

            # round-robin gán route -> CHỈ SỐ XE
            for i, g in enumerate(groups):
                vidx = vidxs[i % len(vidxs)]
                seq = self._nearest_neighbor_order(dep_id, g)
                routes.append(Route(vehicle_id=P.vehicles[vidx].id, seq=seq))

        # route rỗng cho xe chưa dùng
        used_vids = {r.vehicle_id for r in routes}
        for v in P.vehicles:
            if v.id not in used_vids:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))

        return Solution(routes=routes)

    def _decode_cost(self, enc: EncodedSolution) -> float:
        return self._cost(decode_to_solution(self.problem, enc))

    def _tournament(self, population: List[EncodedSolution], fitness: List[float]) -> EncodedSolution:
        """
        Tournament theo FITNESS (cao hơn tốt hơn), không dùng cost thô.
        """
        rng = self.rng
        k = self.tournament_k
        idxs = rng.sample(range(len(population)), k)
        best = max(idxs, key=lambda i: fitness[i])  # FITNESS lớn nhất thắng
        return copy.deepcopy(population[best])

    def _route_segments(self, enc: EncodedSolution) -> List[Tuple[int, List[int]]]:
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

        self._remove_customers(enc, seg_customers)

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
            if not enc.core_routes:
                return
            ridx, pos = 0, 1
        else:
            ridx, pos = best_place

        enc.core_routes[ridx] = enc.core_routes[ridx][:pos] + seg_customers + enc.core_routes[ridx][pos:]

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
            if len(keep) < 2:
                keep = [seq[0], seq[-1]]
            enc.core_routes[ridx] = keep

    def _repair_uniqueness(self, enc: EncodedSolution) -> None:
        """
        Đảm bảo mỗi khách xuất hiện đúng 1 lần; chèn khách thiếu bằng cheapest-insertion.
        """
        P = self.problem
        universe = set(self._all_customers())
        seen = set()

        # loại trùng + đồng bộ tail
        new_tail: List[List[int]] = []
        for ridx, seq in enumerate(enc.core_routes):
            new_seq = []
            for x in seq:
                if P.nodes[x].is_depot:
                    new_seq.append(x)
                else:
                    if x not in seen:
                        new_seq.append(x); seen.add(x)
            # depot đúng hai đầu
            if not new_seq or not P.nodes[new_seq[0]].is_depot:
                new_seq = [seq[0]] + [k for k in new_seq if not P.nodes[k].is_depot]
            if not P.nodes[new_seq[-1]].is_depot:
                new_seq = new_seq + [seq[-1]]

            enc.core_routes[ridx] = new_seq
            # tail cho KH của route này (mặc định 1)
            kh = [x for x in new_seq if not P.nodes[x].is_depot]
            new_tail.append([1] * len(kh))

        enc.tail_orders = new_tail

        missing = list(universe - seen)
        if not missing:
            return

        # cheapest-insertion cho khách thiếu
        for c in missing:
            best = (math.inf, None, None)  # (cost, ridx, pos)
            for ridx, route in enumerate(enc.core_routes):
                cust_idx = [i for i, x in enumerate(route) if not P.nodes[x].is_depot]
                try_positions = [1] if not cust_idx else list(range(cust_idx[0], cust_idx[-1] + 2))
                for pos in try_positions:
                    tmp = copy.deepcopy(enc)
                    tmp.core_routes[ridx] = route[:pos] + [c] + route[pos:]
                    cost = self._decode_cost(tmp)
                    if cost < best[0]:
                        best = (cost, ridx, pos)
            if best[1] is None:
                continue
            ridx, pos = best[1], best[2]
            enc.core_routes[ridx] = enc.core_routes[ridx][:pos] + [c] + enc.core_routes[ridx][pos:]
            route = enc.core_routes[ridx]
            kh = [x for x in route if not P.nodes[x].is_depot]
            enc.tail_orders[ridx] = [1] * len(kh)


    def _mutation(self, enc: EncodedSolution) -> None:
        rng = self.rng
        if not enc.core_routes:
            return
        ridx = rng.randrange(len(enc.core_routes))
        route = enc.core_routes[ridx]
        if len(route) <= 3:
            return
        cust_idx = [i for i, x in enumerate(route) if not self.problem.nodes[x].is_depot]
        if len(cust_idx) < 2:
            return
        i, j = sorted(rng.sample(cust_idx, 2))
        enc.core_routes[ridx] = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
        # đồng bộ tail theo số KH mới (vẫn toàn 1)
        kh = [x for x in enc.core_routes[ridx] if not self.problem.nodes[x].is_depot]
        enc.tail_orders[ridx] = [1] * len(kh)

    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        rng = self.rng
        t0 = time.time()

        pop = self._init_population()
        costs = [self._decode_cost(ind) for ind in pop]
        fitness = self._fitness_from_costs(costs)

        # best theo COST nhưng selection theo FITNESS
        best_idx = min(range(len(pop)), key=lambda i: costs[i])
        best_enc = copy.deepcopy(pop[best_idx])
        best_cost = costs[best_idx]

        patience = 0
        gen = 0

        while (time.time() - t0) < time_limit_sec and gen < self.max_generations and patience < self.patience:
            gen += 1

            elite_k = max(1, int(self.elite_frac * self.pop_size))
            elites_idx = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)[:elite_k]
            new_pop: List[EncodedSolution] = [copy.deepcopy(pop[i]) for i in elites_idx]

            while len(new_pop) < self.pop_size:
                parent1 = self._tournament(pop, fitness)
                parent2 = self._tournament(pop, fitness)

                if rng.random() < self.p_cx:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

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
            fitness = self._fitness_from_costs(costs)

            cur_idx = min(range(len(pop)), key=lambda i: costs[i])
            cur_cost = costs[cur_idx]
            if cur_cost + 1e-12 < best_cost:
                best_cost = cur_cost
                best_enc = copy.deepcopy(pop[cur_idx])
                patience = 0
            else:
                patience += 1

        return decode_to_solution(self.problem, best_enc)
