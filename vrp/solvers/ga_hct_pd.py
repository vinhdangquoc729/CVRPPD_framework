from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval_modified import evaluate_modified


@dataclass
class Head:
    """
    Head:
    - priority: hoán vị chỉ số xe (0..|V|-1) theo mức độ ưu tiên phân bổ route.
    - routes_per_vehicle: số route của từng xe (theo chỉ số xe).
    - nodes_per_route: số node (bao gồm depot đầu/cuối) của từng route trong core (tuần tự).
    - orders_per_node: số order trên từng KH (flatten theo thứ tự các route trong core; depot không có order).
    """
    priority: List[int]
    routes_per_vehicle: List[int]
    nodes_per_route: List[int]
    orders_per_node: List[int]


@dataclass
class EncodedSolution:
    head: Head
    core_routes: List[List[int]]       # mỗi route là dãy node: [depot, ...customers..., depot]
    tail_orders: List[List[List[int]]] # tail_orders[r][i] = list order_id ở khách thứ i (bỏ depot)


def _pd_maps(P: Problem) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Trả về:
      pickup_to_delivery: {pickup_node -> delivery_node}
      delivery_to_pickup: {delivery_node -> pickup_node}
    Nếu không có pd_pairs thì trả rỗng.
    """
    pickup_to_delivery: Dict[int, int] = {}
    delivery_to_pickup: Dict[int, int] = {}
    pd = getattr(P, "pd_pairs", {})
    if isinstance(pd, dict):
        for p, tup in pd.items():
            if isinstance(tup, (tuple, list)) and len(tup) >= 1:
                d = int(tup[0])
                pickup_to_delivery[int(p)] = d
                delivery_to_pickup[d] = int(p)
    return pickup_to_delivery, delivery_to_pickup


def _default_orders_for_route(P: Problem, seq: List[int]) -> List[List[int]]:
    """
    Lập danh sách order_id cho mỗi KH trong route:
      - order_id = pickup_node_id (quy ước).
      - tại pickup: [pickup_id]; tại delivery tương ứng: [pickup_id];
      - node không thuộc cặp PD -> [].
    Độ dài = số KH trong 'seq' (bỏ depot).
    """
    p2d, d2p = _pd_maps(P)
    out: List[List[int]] = []
    for n in seq:
        if P.nodes[n].is_depot:
            continue
        if n in p2d:
            out.append([n])
        elif n in d2p:
            out.append([d2p[n]])
        else:
            out.append([])
    return out


def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    """
    Mã hoá H–C–T theo đúng ý tưởng paper, nhưng an toàn với PD:
    - Gom route theo CHỈ SỐ XE.
    - Head.priority: xe phục vụ nhiều KH hơn đứng trước; tie-break theo chỉ số xe.
    - Tail: danh sách ID order tại từng KH (order_id = pickup_node_id).
    - Head.orders_per_node: flatten số order tại từng KH (metadata).
    """
    by_vidx: Dict[int, List[List[int]]] = {}
    id2vidx = {v.id: i for i, v in enumerate(P.vehicles)}
    for r in sol.routes:
        vidx = id2vidx[r.vehicle_id]
        by_vidx.setdefault(vidx, []).append(list(r.seq))

    nV = len(P.vehicles)

    def _count_customers(seq: List[int]) -> int:
        return sum(1 for x in seq if not P.nodes[x].is_depot)

    used = [(vidx, sum(_count_customers(seq) for seq in by_vidx.get(vidx, [])))
            for vidx in range(nV)]
    priority: List[int] = [vidx for vidx, _ in sorted(used, key=lambda t: (-t[1], t[0]))]
    routes_per_vehicle = [len(by_vidx.get(vidx, [])) for vidx in range(nV)]

    core_routes: List[List[int]] = []
    nodes_per_route: List[int] = []
    tail_orders: List[List[List[int]]] = []
    orders_per_node_flat: List[int] = []

    for vidx in priority:
        for seq in by_vidx.get(vidx, []):
            if not seq:
                continue
            core_routes.append(list(seq))
            nodes_per_route.append(len(seq))  # metadata
            tail_vec = _default_orders_for_route(P, seq)
            tail_orders.append(tail_vec)
            orders_per_node_flat.extend([len(ids) for ids in tail_vec])

    head = Head(
        priority=priority,
        routes_per_vehicle=routes_per_vehicle,
        nodes_per_route=nodes_per_route,
        orders_per_node=orders_per_node_flat,
    )
    enc = EncodedSolution(head=head, core_routes=core_routes, tail_orders=tail_orders)
    _sync_head_metadata(enc)  # đảm bảo nhất quán ngay khi tạo
    return enc


def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    """
    Giải mã:
      - Không dùng nodes_per_route để cắt khách (tránh bỏ khách).
      - Dựng lại route theo depot của xe & danh sách KH trong core_routes.
      - Tail đi kèm để giữ thông tin order, nhưng không nhân bản điểm dừng.
    """
    routes: List[Route] = []
    core_idx = 0
    nV = len(P.vehicles)

    # Chuẩn hoá priority
    seen = set(enc.head.priority)
    if len(enc.head.priority) != nV or any(v >= nV or v < 0 for v in enc.head.priority) or len(seen) != len(enc.head.priority):
        enc.head.priority = list(range(nV))

    # routes_per_vehicle
    rpv = enc.head.routes_per_vehicle[:]
    if len(rpv) != nV:
        rpv = [0] * nV

    # Duyệt theo thứ tự ưu tiên xe, dựng lại route trực tiếp từ core (NO TRIM)
    for vidx in enc.head.priority:
        need = rpv[vidx] if 0 <= vidx < nV else 0
        for _ in range(max(0, need)):
            if core_idx >= len(enc.core_routes):
                break
            dep = P.vehicles[vidx].depot_id
            # lấy đúng danh sách KH (bỏ depot), rồi ghép depot 2 đầu
            customers = [x for x in enc.core_routes[core_idx] if not P.nodes[x].is_depot]
            seq = [dep] + customers + [dep]
            # đảm bảo depot đúng 2 đầu
            if seq[0] != dep:
                seq = [dep] + [x for x in seq if not P.nodes[x].is_depot] + [dep]
            if seq[-1] != dep:
                seq = [x for x in seq if not P.nodes[x].is_depot]
                seq = [dep] + seq + [dep]
            routes.append(Route(vehicle_id=P.vehicles[vidx].id, seq=seq))
            core_idx += 1

    # route rỗng cho xe chưa dùng
    have_vids = {r.vehicle_id for r in routes}
    for v in P.vehicles:
        if v.id not in have_vids:
            routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))

    return Solution(routes=routes)


def _sync_head_metadata(enc: EncodedSolution) -> None:
    """
      - nodes_per_route: len(seq)
      - orders_per_node: flatten len(list order) tại mỗi khách
    """
    enc.head.nodes_per_route = [len(seq) for seq in enc.core_routes]
    new_flat: List[int] = []
    for route_tail in enc.tail_orders:
        for ids in route_tail:
            new_flat.append(len(ids))
    enc.head.orders_per_node = new_flat


class GA_HCT_PD_Solver(Solver):
    """
    - Init: gán khách vào depot gần nhất; NN (có đảo ngẫu nhiên nhẹ) và chia đều theo depot.
    - Crossover: Best-Cost Route Crossover (cắt đoạn KH + tail[order IDs], chèn cheapest).
    - Mutation: đảo 1 đoạn KH trong 1 route (đồng bộ tail).
    - Repair: uniqueness (mỗi KH đúng 1 lần) + PD precedence; tùy chọn ép same-vehicle (mặc định True).
    - Selection: tournament theo fitness power-law.
    """

    def __init__(self,
                 problem: Problem,
                 seed: int = 42,
                 pop_size: int = 80,
                 elite_frac: float = 0.10,
                 tournament_k: int = 2,
                 p_cx: float = 0.9,
                 p_mut: float = 0.2,
                 max_generations: int = 500,
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
        # Nếu dataset yêu cầu same-vehicle -> chắc chắn True; vẫn giúp giảm stockout/precedence
        self.enforce_same_vehicle = True

    def _cost(self, s: Solution) -> float:
        c, _ = evaluate_modified(self.problem, s, return_details=False)
        return c

    def _decode_cost(self, enc: EncodedSolution) -> float:
        # đồng bộ metadata trước khi chấm để tránh lệch
        _sync_head_metadata(enc)
        return self._cost(decode_to_solution(self.problem, enc))

    def _fitness_from_costs(self, costs: List[float]) -> List[float]:
        Amax = max(costs); Amin = min(costs); k = self.power_k
        if abs(Amax - Amin) < 1e-12: return [1.0] * len(costs)
        return [((Amax - Ai) ** k) - Amin for Ai in costs]

    def _vehicles_by_depot(self) -> Dict[int, List[int]]:
        by_dep: Dict[int, List[int]] = {}
        for vidx, v in enumerate(self.problem.vehicles):
            by_dep.setdefault(v.depot_id, []).append(vidx)
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
        route = [depot_id, cur]; unvis.remove(cur)
        while unvis:
            nxt = min(unvis, key=lambda j: P.d(route[-1], j))
            route.append(nxt); unvis.remove(nxt)
        route.append(depot_id)
        # đảo nhẹ để đa dạng
        if len(route) > 5 and rng.random() < 0.4:
            i = rng.randint(1, len(route)-3); j = rng.randint(i+1, len(route)-2)
            route[i:j+1] = reversed(route[i:j+1])
        return route

    def _balanced_split(self, items: List[int], k: int) -> List[List[int]]:
        k = max(1, k); n = len(items)
        if k >= n: return [[x] for x in items] + [[] for _ in range(k-n)]
        parts = [[] for _ in range(k)]
        for i, x in enumerate(items): parts[i % k].append(x)
        return parts

    def _build_initial_solution_guided(self) -> Solution:
        P, rng = self.problem, self.rng
        routes: List[Route] = []
        cust_by_dep = self._assign_customers_to_depots()
        vidx_by_dep = self._vehicles_by_depot()
        for dep_id, custs in cust_by_dep.items():
            vidxs = vidx_by_dep.get(dep_id, [])
            if not vidxs: continue
            factor = 1 if len(custs) <= 12 else (2 if rng.random() < 0.35 else 1)
            k_routes = max(1, min(len(vidxs)*factor, len(custs)))
            tmp = custs[:]; rng.shuffle(tmp)
            groups = self._balanced_split(tmp, k_routes)
            for i, g in enumerate(groups):
                vidx = vidxs[i % len(vidxs)]
                seq = self._nearest_neighbor_order(dep_id, g)
                routes.append(Route(vehicle_id=P.vehicles[vidx].id, seq=seq))
        # thêm route rỗng
        used_vids = {r.vehicle_id for r in routes}
        for v in P.vehicles:
            if v.id not in used_vids:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))
        return Solution(routes=routes)

    def _build_initial_solution_random(self) -> Solution:
        """
        - Lấy tất cả khách hàng, xáo trộn ngẫu nhiên.
        - Chia cho một số xe (random) rồi tạo route: depot -> khách... -> depot.
        - Mỗi xe còn lại có 1 route rỗng (depot, depot).
        """
        P, rng = self.problem, self.rng
        routes: List[Route] = []

        # Danh sách khách
        customers = [i for i, nd in P.nodes.items() if not nd.is_depot]
        rng.shuffle(customers)

        vehicles = list(P.vehicles)

        # Không có khách: tất cả xe đều route rỗng
        if not customers:
            for v in vehicles:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))
            return Solution(routes=routes)

        nV = len(vehicles)
        nC = len(customers)

        num_used = min(nV, nC)
        used_indices = rng.sample(list(range(nV)), k=num_used)

        buckets = self._balanced_split(customers, num_used)

        # Tạo route cho các xe được dùng
        for vidx, bucket in zip(used_indices, buckets):
            v = vehicles[vidx]
            if not bucket:
                # trường hợp hiếm khi bucket rỗng
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))
            else:
                # có thể shuffle lại một lần nữa cho random hơn
                rng.shuffle(bucket)
                seq = [v.depot_id] + bucket + [v.depot_id]
                routes.append(Route(vehicle_id=v.id, seq=seq))

        # Các xe không dùng -> route rỗng
        used_set = set(used_indices)
        for vidx, v in enumerate(vehicles):
            if vidx not in used_set:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))

        return Solution(routes=routes)


    def _init_population(self) -> List[EncodedSolution]:
        pop: List[EncodedSolution] = []
        for _ in range(self.pop_size):
            # s = self._build_initial_solution_guided()
            s = self._build_initial_solution_random()
            enc = encode_from_solution(self.problem, s)
            # Repair trước khi vào quần thể để tránh cá thể bẩn
            self._repair_uniqueness(enc)
            self._repair_pd_constraints(enc)
            _sync_head_metadata(enc)
            pop.append(enc)
        return pop

    # ---------- helpers cho tail/indices ----------

    def _cust_indices(self, seq: List[int]) -> List[int]:
        return [i for i, x in enumerate(seq) if not self.problem.nodes[x].is_depot]

    # ---------- remove / insert segment (đồng bộ tail) ----------

    def _remove_customers(self, enc: EncodedSolution, custs: Iterable[int]) -> None:
        """
        Xoá 1 lần mỗi khách trong danh sách khỏi core_routes và đồng bộ tail_orders.
        """
        S = list(custs)
        if not S: return
        P = self.problem
        for ridx, seq in enumerate(enc.core_routes):
            if not S: break
            new_seq: List[int] = []
            new_tail_k: List[List[int]] = []
            tail_k = enc.tail_orders[ridx] if ridx < len(enc.tail_orders) else []
            tpos = 0
            for node in seq:
                if P.nodes[node].is_depot:
                    new_seq.append(node)
                else:
                    if node in S:
                        S.remove(node)  # bỏ node + tail tương ứng
                    else:
                        new_seq.append(node)
                        ids_here = tail_k[tpos] if tpos < len(tail_k) else []
                        new_tail_k.append(list(ids_here))
                    tpos += 1
            # đảm bảo depot hai đầu
            if not new_seq or not P.nodes[new_seq[0]].is_depot:
                new_seq = [seq[0]] + [x for x in new_seq if not P.nodes[x].is_depot]
            if not P.nodes[new_seq[-1]].is_depot:
                new_seq = new_seq + [seq[-1]]
            enc.core_routes[ridx] = new_seq
            enc.tail_orders[ridx] = new_tail_k
        _sync_head_metadata(enc)

    def _insert_segment_best(self, enc: EncodedSolution,
                             seg_customers: List[int],
                             seg_tail_ids: List[List[int]]) -> None:
        P = self.problem; rng = self.rng
        if not enc.core_routes or not seg_customers:
            return
        best_cost = math.inf; best_place = None
        cand = list(range(len(enc.core_routes))); rng.shuffle(cand)
        cand = cand[:min(8, len(cand))]
        for ridx in cand:
            route = enc.core_routes[ridx]
            cust_idx = self._cust_indices(route)
            try_positions = [1] if not cust_idx else list(range(cust_idx[0], cust_idx[-1]+2))
            tail = enc.tail_orders[ridx] if ridx < len(enc.tail_orders) else []
            for pos in try_positions:
                tmp = copy.deepcopy(enc)
                tmp.core_routes[ridx] = route[:pos] + seg_customers + route[pos:]
                before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
                tmp.tail_orders[ridx] = tail[:before] + copy.deepcopy(seg_tail_ids) + tail[before:]
                _sync_head_metadata(tmp)
                cost = self._decode_cost(tmp)
                if cost < best_cost:
                    best_cost = cost; best_place = (ridx, pos)
        ridx, pos = best_place if best_place is not None else (0, 1)
        route = enc.core_routes[ridx]; tail = enc.tail_orders[ridx]
        before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
        enc.core_routes[ridx] = route[:pos] + seg_customers + route[pos:]
        enc.tail_orders[ridx] = tail[:before] + copy.deepcopy(seg_tail_ids) + tail[before:]
        _sync_head_metadata(enc)

    def _all_customers(self) -> List[int]:
        P = self.problem
        return [i for i, nd in P.nodes.items() if not nd.is_depot]

    def _repair_uniqueness(self, enc: EncodedSolution) -> None:
        """
        Sửa lỗi: mỗi khách đúng 1 lần trong toàn bộ core_routes.
        """
        P = self.problem
        universe = set(self._all_customers())
        seen: set[int] = set()

        new_all_tail: List[List[List[int]]] = []
        for ridx, seq in enumerate(enc.core_routes):
            new_seq: List[int] = []
            new_tail: List[List[int]] = []
            tail = enc.tail_orders[ridx] if ridx < len(enc.tail_orders) else []
            tpos = 0
            for node in seq:
                if P.nodes[node].is_depot:
                    new_seq.append(node)
                else:
                    ids_here = tail[tpos] if tpos < len(tail) else []
                    if node not in seen:
                        new_seq.append(node); new_tail.append(list(ids_here)); seen.add(node)
                    # nếu trùng -> bỏ
                    tpos += 1
            # depot đúng 2 đầu
            if not new_seq or not P.nodes[new_seq[0]].is_depot:
                new_seq = [seq[0]] + [k for k in new_seq if not P.nodes[k].is_depot]
            if not P.nodes[new_seq[-1]].is_depot:
                new_seq = new_seq + [seq[-1]]
            enc.core_routes[ridx] = new_seq
            new_all_tail.append(new_tail)
        enc.tail_orders = new_all_tail

        missing = list(universe - seen)
        if missing:
            for c in missing:
                best = (math.inf, None, None)  # cost, ridx, pos
                for ridx, route in enumerate(enc.core_routes):
                    cust_idx = self._cust_indices(route)
                    try_positions = [1] if not cust_idx else list(range(cust_idx[0], cust_idx[-1]+2))
                    tail = enc.tail_orders[ridx]
                    for pos in try_positions:
                        tmp = copy.deepcopy(enc)
                        tmp.core_routes[ridx] = route[:pos] + [c] + route[pos:]
                        before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
                        tmp.tail_orders[ridx] = tail[:before] + [[]] + tail[before:]
                        _sync_head_metadata(tmp)
                        cost = self._decode_cost(tmp)
                        if cost < best[0]:
                            best = (cost, ridx, pos)
                if best[1] is None:
                    continue
                ridx, pos = best[1], best[2]
                route = enc.core_routes[ridx]; tail = enc.tail_orders[ridx]
                before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
                enc.core_routes[ridx] = route[:pos] + [c] + route[pos:]
                enc.tail_orders[ridx] = tail[:before] + [[]] + tail[before:]

        _sync_head_metadata(enc)

    def _repair_pd_constraints(self, enc: EncodedSolution) -> None:
        """
        Sửa vi phạm PD:
          - Nếu delivery đứng trước pickup -> hoán đổi vị trí 2 khách (đồng bộ tail).
          - Nếu pickup delivery khác route, chuyển delivery về route của pickup
            và chỉ chèn ở vị trí sau pickup (cheapest-insertion bị ràng buộc).
        """
        P = self.problem
        p2d, d2p = _pd_maps(P)
        if not p2d:
            _sync_head_metadata(enc)
            return

        def build_loc() -> Dict[int, Tuple[int, int]]:
            loc: Dict[int, Tuple[int, int]] = {}
            for ridx, seq in enumerate(enc.core_routes):
                pos = 0
                for n in seq:
                    if P.nodes[n].is_depot: continue
                    loc[n] = (ridx, pos); pos += 1
            return loc

        loc = build_loc()
        for p, d in p2d.items():
            if p not in loc or d not in loc:
                continue
            r_p, ip = loc[p]; r_d, idd = loc[d]
            if r_p == r_d and idd < ip:
                # hoán đổi vị trí theo chỉ số trong SEQ (không tính depot)
                seq = enc.core_routes[r_p]
                cust_idx = self._cust_indices(seq)
                if ip < len(cust_idx) and idd < len(cust_idx):
                    i = cust_idx[ip]; j = cust_idx[idd]
                    seq[i], seq[j] = seq[j], seq[i]
                    enc.tail_orders[r_p][ip], enc.tail_orders[r_p][idd] = enc.tail_orders[r_p][idd], enc.tail_orders[r_p][ip]
                # rebuild loc vì đã đổi
                loc = build_loc()

        if not self.enforce_same_vehicle:
            _sync_head_metadata(enc)
            return

        for p, d in p2d.items():
            loc = build_loc()  # rebuild sau mỗi vòng
            if p not in loc or d not in loc:
                continue
            r_p, ip = loc[p]
            r_d, idd = loc[d]
            if r_p == r_d:
                continue  # đã cùng route, không làm gì thêm

            # (1) Remove d khỏi r_d (lấy chỉ số trực tiếp theo SEQ để an toàn)
            seq_d = enc.core_routes[r_d]
            try:
                j_seq = next(i for i, x in enumerate(seq_d) if x == d)
            except StopIteration:
                # d không còn trong seq (đã bị move ở chỗ nào trước đó) -> bỏ qua
                continue
            # tail-index tương ứng với số KH (non-depot) trước & gồm cả d, trừ 1
            k_tail = sum(1 for x in seq_d[:j_seq+1] if not P.nodes[x].is_depot) - 1
            if not (0 <= k_tail < len(enc.tail_orders[r_d])):  # guard
                # lệch dữ liệu (hiếm) -> rebuild tail cho route này rồi tính lại
                enc.tail_orders[r_d] = _default_orders_for_route(P, seq_d)
                k_tail = max(0, min(len(enc.tail_orders[r_d]) - 1, k_tail))
            node_d = d
            tail_d = enc.tail_orders[r_d][k_tail]
            enc.core_routes[r_d] = seq_d[:j_seq] + seq_d[j_seq+1:]
            enc.tail_orders[r_d].pop(k_tail)

            # (2) Insert d vào r_p CHỈ ở vị trí sau pickup p, trước depot cuối
            route = enc.core_routes[r_p]
            tailp = enc.tail_orders[r_p]
            try:
                jp_pick = next(i for i, x in enumerate(route) if x == p)
            except StopIteration:
                # pickup p biến mất? rebuild nhanh tail & bỏ cặp này
                enc.tail_orders[r_p] = _default_orders_for_route(P, route)
                continue

            try_positions = list(range(jp_pick + 1, len(route)))  # trước depot cuối
            if not try_positions:
                try_positions = [min(jp_pick + 1, len(route) - 1)]

            best = (math.inf, None)
            for pos in try_positions:
                tmp = copy.deepcopy(enc)
                tmp.core_routes[r_p] = route[:pos] + [node_d] + route[pos:]
                before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
                tmp.tail_orders[r_p] = tailp[:before] + [tail_d] + tailp[before:]
                _sync_head_metadata(tmp)
                cost = self._decode_cost(tmp)
                if cost < best[0]:
                    best = (cost, pos)

            pos = best[1] if best[1] is not None else (jp_pick + 1)
            before = sum(1 for x in route[:pos] if not P.nodes[x].is_depot)
            enc.core_routes[r_p] = route[:pos] + [node_d] + route[pos:]
            enc.tail_orders[r_p] = tailp[:before] + [tail_d] + tailp[before:]

        _sync_head_metadata(enc)

    def _mutation(self, enc: EncodedSolution) -> None:
        rng = self.rng
        if not enc.core_routes: return
        ridx = rng.randrange(len(enc.core_routes))
        route = enc.core_routes[ridx]
        cust_idx = self._cust_indices(route)
        if len(cust_idx) < 2: return
        tail = enc.tail_orders[ridx]

        i_pos, j_pos = sorted(rng.sample(range(len(cust_idx)), 2))
        i, j = cust_idx[i_pos], cust_idx[j_pos]

        route[i:j+1] = reversed(route[i:j+1])
        tail[i_pos:j_pos+1] = reversed(tail[i_pos:j_pos+1])
        _sync_head_metadata(enc)

    def _pick_segment(self, enc: EncodedSolution) -> Tuple[int, List[int], List[List[int]]]:
        P, rng = self.problem, self.rng
        candidates = [ridx for ridx, seq in enumerate(enc.core_routes)
                      if any(not P.nodes[x].is_depot for x in seq)]
        if not candidates: return -1, [], []
        ridx = rng.choice(candidates)
        seq = enc.core_routes[ridx]; tail = enc.tail_orders[ridx]
        cust_pos = self._cust_indices(seq)
        if len(cust_pos) == 1:
            i_pos = j_pos = 0
        else:
            i_pos, j_pos = sorted(rng.sample(range(len(cust_pos)), 2))
        i, j = cust_pos[i_pos], cust_pos[j_pos]
        seg_customers = [x for x in seq[i:j+1] if not self.problem.nodes[x].is_depot]
        seg_tail = copy.deepcopy(tail[i_pos:j_pos+1])
        return ridx, seg_customers, seg_tail

    def _crossover(self, p1: EncodedSolution, p2: EncodedSolution) -> Tuple[EncodedSolution, EncodedSolution]:
        c1 = copy.deepcopy(p1); c2 = copy.deepcopy(p2)
        # p1 -> c2
        _, seg1_cust, seg1_tail = self._pick_segment(p1)
        if seg1_cust:
            self._remove_customers(c2, seg1_cust)
            self._insert_segment_best(c2, seg1_cust, seg1_tail)
        # p2 -> c1
        _, seg2_cust, seg2_tail = self._pick_segment(p2)
        if seg2_cust:
            self._remove_customers(c1, seg2_cust)
            self._insert_segment_best(c1, seg2_cust, seg2_tail)

        self._repair_uniqueness(c1); self._repair_pd_constraints(c1); _sync_head_metadata(c1)
        self._repair_uniqueness(c2); self._repair_pd_constraints(c2); _sync_head_metadata(c2)
        return c1, c2

    def _tournament(self, population: List[EncodedSolution], fitness: List[float]) -> EncodedSolution:
        rng = self.rng; k = self.tournament_k
        idxs = rng.sample(range(len(population)), min(k, len(population)))
        best = max(idxs, key=lambda i: fitness[i]) 
        return copy.deepcopy(population[best])

    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        rng = self.rng; t0 = time.time()

        pop = self._init_population()
        for ind in pop:
            self._repair_uniqueness(ind)
            self._repair_pd_constraints(ind)
            _sync_head_metadata(ind)

        costs = [self._decode_cost(ind) for ind in pop]
        fitness = self._fitness_from_costs(costs)

        best_idx = min(range(len(pop)), key=lambda i: costs[i])
        best_enc = copy.deepcopy(pop[best_idx]); best_cost = costs[best_idx]
        patience = 0; gen = 0

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
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                if rng.random() < self.p_mut: self._mutation(child1)
                if rng.random() < self.p_mut and len(new_pop) + 1 < self.pop_size: self._mutation(child2)

                self._repair_uniqueness(child1); self._repair_pd_constraints(child1); _sync_head_metadata(child1)
                self._repair_uniqueness(child2); self._repair_pd_constraints(child2); _sync_head_metadata(child2)

                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(child2)

            pop = new_pop

            for ind in pop:
                self._repair_uniqueness(ind)
                self._repair_pd_constraints(ind)
                _sync_head_metadata(ind)

            costs = [self._decode_cost(ind) for ind in pop]
            fitness = self._fitness_from_costs(costs)

            cur_idx = min(range(len(pop)), key=lambda i: costs[i])
            cur_cost = costs[cur_idx]
            if cur_cost + 1e-12 < best_cost:
                best_cost = cur_cost; best_enc = copy.deepcopy(pop[cur_idx]); patience = 0
            else:
                patience += 1

        return decode_to_solution(self.problem, best_enc)
