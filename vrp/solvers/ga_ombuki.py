from __future__ import annotations
import random
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem import Problem, Node, Vehicle
from ..core.solution import Solution, Route


@dataclass
class _RouteState:
    """
    Trạng thái tạm khi build 1 route trong Phase 1.
    """
    vehicle: Vehicle
    depot_id: int
    seq: List[int]         # [depot, c1, c2, ...]
    time: float            # thời gian hiện tại tại node cuối
    load: float            # tải hiện tại


class OmbukiGASolver(Solver):
    """
    GA Ombuki với cơ chế:
    - Phase 1: Greedy Split (Nới lỏng: chỉ cắt khi hết giờ làm việc).
    - Phase 2: Local Search (Move khách cuối).
    - Hỗ trợ Multi-trip (Hành trình nhiều chuyến).
    """

    def __init__(
        self,
        problem: Problem,
        seed: int = 42,
        pop_size: int = 50,
        max_generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_k: int = 4,
        tournament_p_best: float = 0.8,
        weight_routes: float = 100.0,
        weight_distance: float = 0.001,
    ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.tournament_p_best = tournament_p_best

        self.weight_routes = weight_routes
        self.weight_distance = weight_distance

        self._customers: List[int] = list(problem.customers)
        self._veh_by_depot: Dict[int, List[Vehicle]] = self._build_vehicles_by_depot()
        self._veh_by_id: Dict[int, Vehicle] = {v.id: v for v in problem.vehicles}

    # ============================================================
    # Helpers
    # ============================================================

    def _build_vehicles_by_depot(self) -> Dict[int, List[Vehicle]]:
        by_dep: Dict[int, List[Vehicle]] = {}
        for v in self.problem.vehicles:
            # Sử dụng depot_id cũ (từ problem.py chưa sửa)
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    # ============================================================
    # Routing scheme (Phase 1)
    # ============================================================

    def _choose_depot_for_customer(self, cust_id: int) -> int:
        P = self.problem
        best_dep: Optional[int] = None
        best_dist = float("inf")
        for d in P.depots:
            dist = P.d(d, cust_id)
            if dist < best_dist:
                best_dist = dist
                best_dep = d
        return best_dep

    def _get_next_vehicle_for_depot(
        self,
        depot_id: int,
        used_vehicles: Dict[int, int],
    ) -> Vehicle:
        """
        Cơ chế Round-Robin: Xoay vòng xe để tạo Multi-trip.
        """
        vehs = self._veh_by_depot.get(depot_id, [])
        if not vehs:
            return self.problem.vehicles[0]
        
        idx = used_vehicles.get(depot_id, 0)
        veh = vehs[idx % len(vehs)]
        used_vehicles[depot_id] = idx + 1
        return veh

    def _can_append_customer(
        self,
        rs: _RouteState,
        cust_id: int,
    ) -> bool:
        """
        Điều kiện cắt route (Phiên bản Nới lỏng):
        Chỉ trả về False (Cắt) khi xe KHÔNG kịp về kho trước giờ nghỉ (veh.end_time).
        Bỏ qua check Capacity và Time Window (chấp nhận bị phạt).
        """
        P = self.problem
        nodes = P.nodes
        speed = P.speed_units_per_min

        veh = rs.vehicle
        depot_id = rs.depot_id
        u = rs.seq[-1]
        node_c = nodes[cust_id]

        # 1. Tính thời gian đến khách
        travel = P.d(u, cust_id)
        arrival = rs.time + travel / speed

        # Đến sớm -> Chờ
        tw_open = node_c.tw_open if node_c.tw_open is not None else -math.inf
        start_service = max(arrival, tw_open)
        
        finish_service = start_service + node_c.service_time

        # --- BỎ CHECK TIME WINDOW ---
        # if node_c.tw_close is not None and finish_service > node_c.tw_close:
        #     return False

        # --- BỎ CHECK CAPACITY ---
        # new_load = rs.load + node_c.demand_pickup - node_c.demand_delivery
        # if new_load > veh.capacity: return False

        # 2. Kiểm tra điều kiện duy nhất: Kịp về kho trước end_time?
        travel_back = P.d(cust_id, depot_id)
        back_arrival = finish_service + travel_back / speed
        
        if back_arrival > veh.end_time:
            return False

        return True

    def _append_customer(self, rs: _RouteState, cust_id: int) -> None:
        """
        Thực hiện thêm khách vào route.
        """
        P = self.problem
        nodes = P.nodes
        speed = P.speed_units_per_min

        u = rs.seq[-1]
        node_c = nodes[cust_id]

        travel = P.d(u, cust_id)
        arrival = rs.time + travel / speed

        tw_open = node_c.tw_open if node_c.tw_open is not None else -math.inf
        start_service = max(arrival, tw_open)
        finish_service = start_service + node_c.service_time

        rs.time = finish_service
        rs.load = rs.load + node_c.demand_pickup - node_c.demand_delivery
        rs.seq.append(cust_id)

    def _open_new_route_for_customer(
        self,
        cust_id: int,
        used_vehicles: Dict[int, int],
        vehicle_availability: Dict[int, float]
    ) -> _RouteState:
        """
        Mở route mới. Hỗ trợ Multi-trip bằng cách lấy thời gian rảnh từ vehicle_availability.
        """
        P = self.problem
        nodes = P.nodes

        depot_id = self._choose_depot_for_customer(cust_id)
        veh = self._get_next_vehicle_for_depot(depot_id, used_vehicles)

        # Lấy thời gian bắt đầu thực tế (nếu xe đã chạy chuyến trước)
        start_t = vehicle_availability.get(veh.id, float(veh.start_time))
        
        # Nếu đây là chuyến thứ 2 trở đi, cộng thêm thời gian nạp hàng tại kho
        if start_t > veh.start_time:
            start_t += nodes[depot_id].service_time

        rs = _RouteState(
            vehicle=veh,
            depot_id=depot_id,
            seq=[depot_id],
            time=start_t,
            load=0.0,
        )

        # Cố gắng append khách đầu tiên
        if self._can_append_customer(rs, cust_id):
            self._append_customer(rs, cust_id)
        else:
            # Force append: Nếu ngay khách đầu tiên đã không kịp giờ (do chuyến trước về muộn)
            # ta vẫn thêm vào để đảm bảo tính hợp lệ (mọi khách phải có route),
            # dù route này sẽ bị phạt nặng về Time Window hoặc Overtime.
            speed = P.speed_units_per_min
            travel = P.d(depot_id, cust_id)
            arrival = rs.time + travel / speed
            node_c = nodes[cust_id]
            
            start_service = max(
                arrival,
                node_c.tw_open if node_c.tw_open is not None else -math.inf,
            )
            rs.time = start_service + node_c.service_time
            rs.load = node_c.demand_pickup - node_c.demand_delivery
            rs.seq.append(cust_id)

        return rs

    def _decode_chromosome(self, chrom: List[int]) -> Solution:
        """
        Giải mã nhiễm sắc thể thành Solution.
        """
        P = self.problem
        nodes = P.nodes

        routes: List[Route] = []
        used_vehicles: Dict[int, int] = {}
        
        # Dictionary lưu thời gian rảnh của xe: {vehicle_id: finish_time}
        vehicle_availability: Dict[int, float] = {}

        rs: Optional[_RouteState] = None

        for cust_id in chrom:
            if nodes[cust_id].is_depot:
                continue

            if rs is None:
                rs = self._open_new_route_for_customer(cust_id, used_vehicles, vehicle_availability)
                continue

            if self._can_append_customer(rs, cust_id):
                self._append_customer(rs, cust_id)
            else:
                # Đóng route cũ
                if rs.seq[-1] != rs.depot_id:
                    rs.seq.append(rs.depot_id)
                
                # Tính thời gian về kho và cập nhật availability
                t_back = P.d(rs.seq[-2], rs.depot_id) / P.speed_units_per_min
                finish_time = rs.time + t_back
                vehicle_availability[rs.vehicle.id] = finish_time

                routes.append(Route(vehicle_id=rs.vehicle.id, seq=rs.seq))

                # Mở route mới (có thể dùng lại xe vừa rồi nếu Round-Robin chọn trúng)
                rs = self._open_new_route_for_customer(cust_id, used_vehicles, vehicle_availability)

        # Đóng route cuối cùng
        if rs is not None:
            if rs.seq[-1] != rs.depot_id:
                rs.seq.append(rs.depot_id)
            
            t_back = P.d(rs.seq[-2], rs.depot_id) / P.speed_units_per_min
            finish_time = rs.time + t_back
            vehicle_availability[rs.vehicle.id] = finish_time
            
            routes.append(Route(vehicle_id=rs.vehicle.id, seq=rs.seq))

        return Solution(routes=routes)

    # ============================================================
    # Phase 2: Cải thiện (Nới lỏng check)
    # ============================================================

    def _check_route_and_distance(self, seq: List[int], veh: Vehicle) -> Tuple[bool, float]:
        """
        Kiểm tra feasibility và tính distance.
        Cũng NỚI LỎNG: Chỉ trả về False nếu vi phạm veh.end_time.
        (Capacity và TW violation không làm False, nhưng vẫn tính distance).
        """
        P = self.problem
        nodes = P.nodes
        speed = P.speed_units_per_min
        
        if not seq or len(seq) < 2:
            return True, 0.0

        depot_id = veh.depot_id 
        if seq[0] != depot_id or seq[-1] != depot_id:
            return False, float("inf")

        # Lưu ý: Reset time về start_time cho đơn giản hóa (chấp nhận xấp xỉ cho Local Search)
        time = float(veh.start_time)
        load = 0.0
        total_dist = 0.0

        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            dist_uv = P.d(u, v)
            total_dist += dist_uv
            arrival = time + dist_uv / speed
            node_v = nodes[v]

            if node_v.is_depot:
                if arrival > veh.end_time:
                    return False, float("inf")
                time = arrival
                continue

            tw_open = node_v.tw_open if node_v.tw_open is not None else -math.inf
            
            # BỎ CHECK TW CLOSE
            # tw_close = ...
            
            start_service = max(arrival, tw_open)
            finish_service = start_service + node_v.service_time
            
            # BỎ CHECK CAPACITY
            # load = ...

            time = finish_service

        return True, total_dist

    def _improve_phase2(self, sol: Solution) -> Solution:
        """
        Thử di chuyển khách cuối của route i sang route i+1 (cùng depot).
        """
        P = self.problem
        if len(sol.routes) < 2:
            return sol

        routes = [Route(vehicle_id=rt.vehicle_id, seq=list(rt.seq)) for rt in sol.routes]

        for idx in range(len(routes) - 1):
            r1 = routes[idx]
            r2 = routes[idx + 1]
            if len(r1.seq) <= 3:
                continue

            dep1 = r1.seq[0]
            dep2 = r2.seq[0]
            if dep1 != dep2:
                continue

            veh1 = self._veh_by_id[r1.vehicle_id]
            veh2 = self._veh_by_id[r2.vehicle_id]

            last_cust = r1.seq[-2]
            if P.nodes[last_cust].is_depot:
                continue

            # Check trạng thái hiện tại
            feas1_cur, dist1_cur = self._check_route_and_distance(r1.seq, veh1)
            feas2_cur, dist2_cur = self._check_route_and_distance(r2.seq, veh2)
            if not (feas1_cur and feas2_cur):
                continue
            best_pair_cost = dist1_cur + dist2_cur

            best_new_seq1 = None
            best_new_seq2 = None

            base_seq1 = r1.seq[:-2] + [dep1]
            
            # Thử chèn vào r2
            for pos in range(1, len(r2.seq)): 
                new_seq2 = r2.seq[:pos] + [last_cust] + r2.seq[pos:]
                
                feas1, dist1 = self._check_route_and_distance(base_seq1, veh1)
                feas2, dist2 = self._check_route_and_distance(new_seq2, veh2)
                
                if not (feas1 and feas2):
                    continue
                
                new_pair_cost = dist1 + dist2
                if new_pair_cost < best_pair_cost - 1e-9:
                    best_pair_cost = new_pair_cost
                    best_new_seq1 = base_seq1
                    best_new_seq2 = new_seq2

            if best_new_seq1 is not None and best_new_seq2 is not None:
                routes[idx].seq = best_new_seq1
                routes[idx + 1].seq = best_new_seq2

        return Solution(routes=routes)

    # ============================================================
    # GA components (Standard)
    # ============================================================

    def _init_population(self) -> List[List[int]]:
        base = self._customers
        pop: List[List[int]] = []
        for _ in range(self.pop_size):
            chrom = base[:] 
            self.rng.shuffle(chrom)
            pop.append(chrom)
        return pop

    def _evaluate_solution(self, sol: Solution) -> Tuple[float, int, float]:
        P = self.problem
        total_dist = 0.0
        for rt in sol.routes:
            seq = rt.seq
            for i in range(len(seq) - 1):
                total_dist += P.d(seq[i], seq[i + 1])
        num_routes = len(sol.routes)
        # Weighted sum cost
        cost = self.weight_routes * num_routes + self.weight_distance * total_dist
        return cost, num_routes, total_dist

    def _evaluate_chromosome(self, chrom: List[int]) -> Tuple[float, int, float]:
        sol_phase1 = self._decode_chromosome(chrom)
        sol = self._improve_phase2(sol_phase1)
        return self._evaluate_solution(sol)

    def _tournament_select(self, pop: List[List[int]], fitnesses: List[float]) -> List[int]:
        r = self.rng
        N = len(pop)
        idxs = [r.randrange(N) for _ in range(self.tournament_k)]
        idxs.sort(key=lambda i: fitnesses[i])
        if r.random() < self.tournament_p_best:
            return pop[idxs[0]][:]
        else:
            return pop[r.choice(idxs)][:]

    def _crossover_ox(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        r = self.rng
        n = len(p1)
        if n < 2:
            return p1[:], p2[:]
        i, j = sorted(r.sample(range(n), 2))
        c1 = [None] * n
        c2 = [None] * n
        
        # Copy đoạn [i, j]
        c1[i:j + 1] = p1[i:j + 1]
        c2[i:j + 1] = p2[i:j + 1]
        
        def fill_child(child, parent):
            pos = (j + 1) % n
            for gene in parent:
                if gene not in child:
                    child[pos] = gene
                    pos = (pos + 1) % n
        
        fill_child(c1, p2)
        fill_child(c2, p1)
        return c1, c2 

    def _mutate_inversion(self, chrom: List[int]) -> None:
        r = self.rng
        n = len(chrom)
        if n < 2: return
        max_len = 3 if n >= 3 else 2
        seg_len = r.randint(2, max_len)
        i = r.randint(0, n - seg_len)
        j = i + seg_len
        chrom[i:j] = reversed(chrom[i:j])

    def solve(self, time_limit_sec: float = 30000.0) -> Solution:
        r = self.rng
        t0 = time.time()
        pop = self._init_population()
        fitnesses: List[float] = []
        vectors: List[Tuple[int, float]] = []

        # Evaluate initial population
        for chrom in pop:
            f, n_route, dist = self._evaluate_chromosome(chrom)
            fitnesses.append(f)
            vectors.append((n_route, dist))

        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_chrom = pop[best_idx][:]
        best_cost, best_nroute, best_dist = fitnesses[best_idx], *vectors[best_idx]

        gen = 0
        while gen < self.max_generations and (time.time() - t0) < time_limit_sec:
            gen += 1
            print(f"Generation {gen}: best_cost = {best_cost}", end="\r")
            new_pop: List[List[int]] = []
            
            # Elitism
            elite = best_chrom[:]
            new_pop.append(elite)

            # Evolution loop
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament_select(pop, fitnesses)
                child = parent1[:]
                
                # Crossover
                if r.random() < self.crossover_rate:
                    parent2 = self._tournament_select(pop, fitnesses)
                    c1, c2 = self._crossover_ox(parent1, parent2)
                    child = c1 if r.random() < 0.5 else c2
                
                # Mutation
                if r.random() < self.mutation_rate:
                    self._mutate_inversion(child)
                
                new_pop.append(child)

            pop = new_pop
            fitnesses = []
            vectors = []
            
            # Re-evaluate
            for chrom in pop:
                f, n_route, dist = self._evaluate_chromosome(chrom)
                fitnesses.append(f)
                vectors.append((n_route, dist))

            # Update Best
            cur_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            cur_cost, cur_vec = fitnesses[cur_best_idx], vectors[cur_best_idx]

            if cur_cost < best_cost:
                best_cost = cur_cost
                best_chrom = pop[cur_best_idx][:]
                best_nroute, best_dist = cur_vec

        # Final decode
        best_sol_phase1 = self._decode_chromosome(best_chrom)
        best_solution = self._improve_phase2(best_sol_phase1)
        return best_solution