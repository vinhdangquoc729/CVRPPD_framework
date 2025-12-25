# vrp/solvers/ga_pd_hct_origin.py
from __future__ import annotations
import random, math, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate

# =========================
#   Encoding: Head–Core–Tail
# =========================

@dataclass
class Head:
    priority: List[int]
    routes_per_vehicle: List[int]
    nodes_per_route: List[int]
    orders_per_node: List[int]

@dataclass
class EncodedSolution:
    head: Head
    core_routes: List[List[int]]
    tail_orders: List[List[int]]

def _sync_head_metadata(enc: EncodedSolution) -> None:
    enc.head.nodes_per_route = [len(seq) for seq in enc.core_routes]
    flat_orders: List[int] = []
    for route_tail in enc.tail_orders:
        flat_orders.extend(route_tail)
    enc.head.orders_per_node = flat_orders

# ---------- encode/decode ----------

def encode_from_solution(P: Problem, sol: Solution) -> EncodedSolution:
    nV = len(P.vehicles)
    id2vidx = {v.id: i for i, v in enumerate(P.vehicles)}

    by_vidx: Dict[int, List[List[int]]] = {}
    for r in sol.routes:
        vidx = id2vidx.get(r.vehicle_id, None)
        if vidx is None: continue
        by_vidx.setdefault(vidx, []).append(list(r.seq))

    def served_cnt(seq: List[int]) -> int:
        return sum(1 for i in seq if not P.nodes[i].is_depot)

    used = [(vidx, sum(served_cnt(seq) for seq in by_vidx.get(vidx, []))) for vidx in range(nV)]
    priority: List[int] = [vidx for vidx, _ in sorted(used, key=lambda t: (-t[1], t[0]))]
    routes_per_vehicle = [len(by_vidx.get(vidx, [])) for vidx in range(nV)]

    core_routes, tail_orders, flat_orders = [], [], []
    for vidx in priority:
        for seq in by_vidx.get(vidx, []):
            if not seq: continue
            core_routes.append(list(seq))
            num_cust = sum(1 for i in seq if not P.nodes[i].is_depot)
            tail_vec = [1] * num_cust
            tail_orders.append(tail_vec)
            flat_orders.extend(tail_vec)

    head = Head(priority=priority, routes_per_vehicle=routes_per_vehicle,
                nodes_per_route=[len(r) for r in core_routes], orders_per_node=flat_orders)
    return EncodedSolution(head=head, core_routes=core_routes, tail_orders=tail_orders)

def decode_to_solution(P: Problem, enc: EncodedSolution) -> Solution:
    routes: List[Route] = []
    core_idx = 0
    pri = enc.head.priority
    rpv = enc.head.routes_per_vehicle # Số lượng route trên mỗi xe [cite: 918]

    for vidx in pri:
        veh = P.vehicles[vidx]
        # Lấy đúng số lượng chuyến (trips) mà xe này được gán
        num_trips = rpv[vidx] 
        
        for _ in range(num_trips):
            if core_idx >= len(enc.core_routes): break
            seq_core = enc.core_routes[core_idx]
            
            # Lọc khách hàng, kẹp giữa depot của xe đó
            customers = [x for x in seq_core if not P.nodes[x].is_depot]
            if customers:
                routes.append(Route(vehicle_id=veh.id, seq=[veh.depot_id] + customers + [veh.depot_id]))
            core_idx += 1

    # Các xe không chạy chuyến nào thì tạo route rỗng [0, 0]
    used_vids = {r.vehicle_id for r in routes}
    for v in P.vehicles:
        if v.id not in used_vids:
            routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))
            
    return Solution(routes=routes)

# =========================
#   GA (Optimized)
# =========================

class GAPD_HCT_ORIGIN_Solver(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 50, elite_frac: float = 0.1,
                 tournament_k: int = 2, p_cx: float = 0.9, p_mut: float = 0.2, p_route_mut: float = 0.2,
                 max_generations: int = 500, patience: int = 50, power_k: float = 1.0, evaluator: callable = None):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size, self.elite_frac, self.tournament_k = pop_size, elite_frac, tournament_k
        self.p_cx, self.p_mut, self.p_route_mut = p_cx, p_mut, p_route_mut
        self.max_generations, self.patience, self.power_k = max_generations, patience, max(1.0, power_k)
        self.evaluator = evaluator if evaluator is not None else evaluate

    def _quick_copy_enc(self, enc: EncodedSolution) -> EncodedSolution:
        """Thay thế deepcopy bằng thủ công để tăng tốc."""
        h = enc.head
        new_head = Head(h.priority[:], h.routes_per_vehicle[:], h.nodes_per_route[:], h.orders_per_node[:])
        return EncodedSolution(new_head, [r[:] for r in enc.core_routes], [t[:] for t in enc.tail_orders])

    def _decode_cost(self, enc: EncodedSolution) -> float:
        cost, _ = self.evaluator(self.problem, decode_to_solution(self.problem, enc), return_details=False)
        return cost

    def _fitness_from_costs(self, costs: List[float]) -> List[float]:
        Amax, Amin = max(costs), min(costs)
        if abs(Amax - Amin) < 1e-12: return [1.0] * len(costs)
        return [((Amax - c) ** self.power_k) for c in costs]

    # ---------- Initialization ----------

    def _init_population(self) -> List[EncodedSolution]:
        pop = []
        for _ in range(self.pop_size):
            s = self._build_initial_solution_randomly()
            pop.append(encode_from_solution(self.problem, s))
        return pop

    def _build_initial_solution_guided(self) -> Solution:
        P, rng = self.problem, self.rng
        routes = []
        
        # Gom khách theo depot gần nhất (Giữ nguyên logic cũ)
        cust_by_dep = {}
        for nid, nd in P.nodes.items():
            if not nd.is_depot:
                d = min(P.depots, key=lambda dep: P.d(nid, dep))
                cust_by_dep.setdefault(d, []).append(nid)
        
        veh_by_dep = {}
        for v in P.vehicles: 
            veh_by_dep.setdefault(v.depot_id, []).append(v)

        for dep_id, custs in cust_by_dep.items():
            vehs = veh_by_dep.get(dep_id, [])
            if not vehs: continue
            
            # Lấy một xe đại diện để tính toán tải trọng (Giả sử xe đầu tiên)
            capacity = vehs[0].capacity
            
            tmp = custs[:]
            rng.shuffle(tmp)
            
            # Sắp xếp khách theo NN để có chuỗi logic trước khi ngắt
            full_seq = self._nearest_neighbor_order(dep_id, tmp)
            # Bỏ depot ở đầu và cuối chuỗi NN để lấy danh sách khách thuần túy
            customers_only = [n for n in full_seq if not P.nodes[n].is_depot]
            
            current_trip = []
            current_load = 0
            v_idx = 0 # Dùng để xoay vòng xe nếu có nhiều xe tại depot
            
            for cid in customers_only:
                demand = P.nodes[cid].demand_delivery # Ước lượng theo hàng giao
                
                if current_load + demand > capacity and current_trip:
                    # Nếu thêm khách này sẽ quá tải -> Đóng chuyến cũ
                    veh_id = vehs[v_idx % len(vehs)].id
                    routes.append(Route(vehicle_id=veh_id, seq=[dep_id] + current_trip + [dep_id]))
                    # Reset cho chuyến mới
                    current_trip = [cid]
                    current_load = demand
                    v_idx += 1 # Chuyển sang chuyến tiếp theo (có thể của xe khác hoặc cùng xe)
                else:
                    current_trip.append(cid)
                    current_load += demand
                    
            # Thêm chuyến cuối cùng
            if current_trip:
                veh_id = vehs[v_idx % len(vehs)].id
                routes.append(Route(vehicle_id=veh_id, seq=[dep_id] + current_trip + [dep_id]))

        return Solution(routes=routes)

    def _build_initial_solution_randomly(self) -> Solution:
        P, rng = self.problem, self.rng
        routes = []
        
        # 1. Lấy danh sách tất cả khách hàng
        all_customers = [nid for nid, nd in P.nodes.items() if not nd.is_depot]
        num_vehs = len(P.vehicles)
        
        # 2. Quyết định số lượng route sẽ tạo ra
        # Giả sử mỗi xe chạy từ 1 đến 2 chuyến để đảm bảo tính đa dạng
        num_routes_to_create = num_vehs * rng.randint(1, 2)
        
        # Tạo danh sách các nhóm khách hàng trống cho từng route
        customer_groups = [[] for _ in range(num_routes_to_create)]
        
        # 3. PHÂN BỔ NGẪU NHIÊN: Mỗi khách hàng vào một route bất kỳ
        for cid in all_customers:
            target_route = rng.randrange(num_routes_to_create)
            customer_groups[target_route].append(cid)
            
        # 4. Gán các nhóm khách này cho các xe (xoay vòng xe)
        for i, group in enumerate(customer_groups):
            if not group: 
                continue # Bỏ qua nếu route đó ngẫu nhiên không có khách nào
                
            # Chọn xe theo thứ tự xoay vòng (round-robin)
            veh = P.vehicles[i % num_vehs]
            dep_id = veh.depot_id
            
            # Tạo route: [Depot của xe, ...khách ngẫu nhiên..., Depot của xe]
            # Lưu ý: group lúc này đã chứa khách theo thứ tự ngẫu nhiên vì lấy từ vòng lặp
            routes.append(Route(vehicle_id=veh.id, seq=[dep_id] + group + [dep_id]))
            
        # 5. Đảm bảo những xe không được gán khách vẫn có route rỗng [depot, depot]
        used_vids = {r.vehicle_id for r in routes}
        for v in P.vehicles:
            if v.id not in used_vids:
                routes.append(Route(vehicle_id=v.id, seq=[v.depot_id, v.depot_id]))
                
        return Solution(routes=routes)

    def _balanced_split(self, items, k):
        parts = [[] for _ in range(k)]
        for i, x in enumerate(items): parts[i % k].append(x)
        return parts

    def _nearest_neighbor_order(self, depot_id, customers):
        P = self.problem
        if not customers: return [depot_id, depot_id]
        unvis = set(customers)
        cur = min(unvis, key=lambda j: P.d(depot_id, j))
        route = [depot_id, cur]
        unvis.remove(cur)
        while unvis:
            nxt = min(unvis, key=lambda j: P.d(route[-1], j))
            route.append(nxt)
            unvis.remove(nxt)
        route.append(depot_id)
        return self._two_opt_light(route)

    def _two_opt_light(self, seq):
        P = self.problem
        if len(seq) < 6: return seq[:]
        best = seq[:]
        for _ in range(30):
            improved = False
            for a in range(1, len(best) - 3):
                for b in range(a + 1, len(best) - 1):
                    old = P.d(best[a-1], best[a]) + P.d(best[b], best[b+1])
                    new = P.d(best[a-1], best[b]) + P.d(best[a], best[b+1])
                    if new < old - 1e-9:
                        best[a:b+1] = best[a:b+1][::-1]
                        improved = True
            if not improved: break
        return best

    # ---------- Genetic Operators ----------

    def _tournament(self, population, fitness):
        idxs = self.rng.sample(range(len(population)), min(self.tournament_k, len(population)))
        return population[max(idxs, key=lambda i: fitness[i])]

    def _insert_segment_best(self, enc: EncodedSolution, seg_customers: List[int]) -> None:
        """Tối ưu hóa: Sử dụng Delta Distance thay vì gọi full decode."""
        P = self.problem
        if not enc.core_routes or not seg_customers: return
        
        best_delta, best_place = math.inf, (0, 1)
        cand_routes = self.rng.sample(range(len(enc.core_routes)), min(8, len(enc.core_routes)))
        
        seg_dist = sum(P.d(seg_customers[i], seg_customers[i+1]) for i in range(len(seg_customers)-1))
        f, l = seg_customers[0], seg_customers[-1]

        for ridx in cand_routes:
            route = enc.core_routes[ridx]
            for pos in range(1, len(route)):
                delta = P.d(route[pos-1], f) + seg_dist + P.d(l, route[pos]) - P.d(route[pos-1], route[pos])
                if delta < best_delta:
                    best_delta, best_place = delta, (ridx, pos)
        
        ridx, pos = best_place
        enc.core_routes[ridx][pos:pos] = seg_customers # In-place insert
        # Cập nhật tail_orders tương ứng (mỗi KH 1 order)
        enc.tail_orders[ridx][pos-1:pos-1] = [1] * len(seg_customers)

    def _repair_uniqueness(self, enc: EncodedSolution) -> None:
        P = self.problem
        universe = set(i for i, nd in P.nodes.items() if not nd.is_depot)
        seen = set()
        for ridx in range(len(enc.core_routes)):
            new_r, new_t = [], []
            for i, x in enumerate(enc.core_routes[ridx]):
                if P.nodes[x].is_depot: new_r.append(x)
                elif x not in seen:
                    new_r.append(x); seen.add(x)
                    if i-1 < len(enc.tail_orders[ridx]): new_t.append(enc.tail_orders[ridx][i-1])
            if len(new_r) < 2: new_r = [enc.core_routes[ridx][0], enc.core_routes[ridx][-1]]
            elif not P.nodes[new_r[-1]].is_depot: new_r.append(enc.core_routes[ridx][-1])
            enc.core_routes[ridx], enc.tail_orders[ridx] = new_r, new_t

        missing = list(universe - seen)
        for c in missing:
            best_delta, best_pos = math.inf, (0, 1)
            for ridx, route in enumerate(enc.core_routes):
                for pos in range(1, len(route)):
                    d = P.d(route[pos-1], c) + P.d(c, route[pos]) - P.d(route[pos-1], route[pos])
                    if d < best_delta: best_delta, best_pos = d, (ridx, pos)
            r_idx, p = best_pos
            enc.core_routes[r_idx].insert(p, c)
            enc.tail_orders[r_idx].insert(p-1, 1)
        _sync_head_metadata(enc)

    def _crossover(self, A, B):
        a, b = self._quick_copy_enc(A), self._quick_copy_enc(B)
        def get_seg(enc):
            active = [(i, r) for i, r in enumerate(enc.core_routes) if len(r) > 2]
            if not active: return None
            ridx, r = self.rng.choice(active)
            custs = [x for x in r if not self.problem.nodes[x].is_depot]
            return custs
        
        seg_a, seg_b = get_seg(a), get_seg(b)
        if seg_a: self._insert_segment_best(b, seg_a)
        if seg_b: self._insert_segment_best(a, seg_b)
        return a, b

    def _mutation(self, enc):
        if not enc.core_routes: return
        ridx = self.rng.randrange(len(enc.core_routes))
        route = enc.core_routes[ridx]
        cust_idx = [i for i, x in enumerate(route) if not self.problem.nodes[x].is_depot]
        if len(cust_idx) < 2: return
        i, j = sorted(self.rng.sample(cust_idx, 2))
        route[i:j+1] = route[i:j+1][::-1]

    def _mutation_move_route(self, enc):
        rpv = enc.head.routes_per_vehicle
        donors = [i for i, c in enumerate(rpv) if c > 0]
        if not donors or len(rpv) < 2: return
        v_from = self.rng.choice(donors)
        v_to = self.rng.choice([i for i in range(len(rpv)) if i != v_from])
        rpv[v_from] -= 1; rpv[v_to] += 1

    # ---------- Main Solver ----------

    def solve(self, time_limit_sec: float = 60.0) -> Solution:
        t0 = time.time()
        pop = self._init_population()
        costs = [self._decode_cost(ind) for ind in pop]
        best_enc = self._quick_copy_enc(pop[min(range(len(pop)), key=lambda i: costs[i])])
        best_cost = min(costs)

        gen, patience = 0, 0
        while (time.time() - t0) < time_limit_sec and gen < self.max_generations and patience < self.patience:
            gen += 1

            fitness = self._fitness_from_costs(costs)
            
            # Elitism
            elite_k = max(1, int(self.elite_frac * self.pop_size))
            elites_idx = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)[:elite_k]
            new_pop = [self._quick_copy_enc(pop[i]) for i in elites_idx]

            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament(pop, fitness), self._tournament(pop, fitness)
                if self.rng.random() < self.p_cx: c1, c2 = self._crossover(p1, p2)
                else: c1, c2 = self._quick_copy_enc(p1), self._quick_copy_enc(p2)

                for c in [c1, c2]:
                    if self.rng.random() < self.p_mut: self._mutation(c)
                    # if self.rng.random() < self.p_route_mut: self._mutation_move_route(c)
                    self._repair_uniqueness(c)
                    if len(new_pop) < self.pop_size: new_pop.append(c)

            pop = new_pop
            costs = [self._decode_cost(ind) for ind in pop]
            cur_min = min(costs)
            if cur_min < best_cost - 1e-9:
                best_cost, patience = cur_min, 0
                best_enc = self._quick_copy_enc(pop[costs.index(cur_min)])
            else: patience += 1
            print(f"Gen {gen} | Best: {best_cost:.2f} | P: {patience}", end='\r')

        self._save_final_population_details(pop, filename=f"last_generation/ga_tcpvrp_origin_final_pop_seed{self.seed}.csv")
        return decode_to_solution(self.problem, best_enc)
    
    def _save_final_population_details(self, population: List[EncodedSolution], filename: str = "final_pop_details.csv"):
        import pandas as pd
        all_records = []        
        for i, enc in enumerate(population):
            # 1. Decode cá thể thành lời giải thực tế
            sol = decode_to_solution(self.problem, enc)
            
            # 2. Đánh giá chi tiết (lấy toàn bộ các trường trong details)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            
            # 3. Tạo bản ghi kết hợp
            record = {
                "individual_id": i,
                "total_cost": cost,
                **details, # Giải nén distance, fixed, penalty...
                "solution_str": str(sol) # Lưu cấu trúc route để tái hiện nếu cần
            }
            all_records.append(record)
        
        # Xuất ra file CSV
        df = pd.DataFrame(all_records)
        df.to_csv(filename, index=False)
