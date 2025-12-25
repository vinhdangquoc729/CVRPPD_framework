from __future__ import annotations
import random
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem import Problem, Node, Vehicle
from ..core.solution import Solution, Route
from ..core.eval import evaluate


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
        evaluator: callable = None,
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
        self.evaluator = evaluator if evaluator is not None else evaluate

        self._customers: List[int] = list(problem.customers)
        self._veh_by_depot: Dict[int, List[Vehicle]] = self._build_vehicles_by_depot()
        self._veh_by_id: Dict[int, Vehicle] = {v.id: v for v in problem.vehicles}

    # ============================================================
    # Helpers
    # ============================================================

    def _build_vehicles_by_depot(self) -> Dict[int, List[Vehicle]]:
        by_dep: Dict[int, List[Vehicle]] = {}
        for v in self.problem.vehicles:
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
        vehs = self._veh_by_depot.get(depot_id, [])
        if not vehs:
            return self.problem.vehicles[0]
        
        idx = used_vehicles.get(depot_id, 0)
        veh = vehs[idx % len(vehs)]
        used_vehicles[depot_id] = idx + 1
        return veh

    def _can_append_customer(self, rs: _RouteState, cust_id: int) -> bool:
        P = self.problem
        nodes = P.nodes
        speed = P.speed_units_per_min
        veh = rs.vehicle
        depot_id = rs.depot_id
        u = rs.seq[-1]
        node_c = nodes[cust_id]

        if (u, cust_id) in P.prohibited:
            return False
        if (cust_id, depot_id) in P.prohibited:
            return False

        new_load = rs.load + node_c.demand_pickup - node_c.demand_delivery
        if new_load > veh.capacity:
            return False

        travel = P.d(u, cust_id)
        arrival = rs.time + travel / speed
        tw_open = node_c.tw_open if node_c.tw_open is not None else -math.inf
        start_service = max(arrival, tw_open)
        finish_service = start_service + node_c.service_time
        
        travel_back = P.d(cust_id, depot_id)
        back_arrival = finish_service + travel_back / speed
        
        if back_arrival > veh.end_time:
            return False

        return True

    def _append_customer(self, rs: _RouteState, cust_id: int) -> None:
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
        P = self.problem
        nodes = P.nodes

        depot_id = self._choose_depot_for_customer(cust_id)
        veh = self._get_next_vehicle_for_depot(depot_id, used_vehicles)

        start_t = vehicle_availability.get(veh.id, float(veh.start_time))
        
        if start_t > veh.start_time:
            start_t += nodes[depot_id].service_time

        rs = _RouteState(
            vehicle=veh,
            depot_id=depot_id,
            seq=[depot_id],
            time=start_t,
            load=0.0,
        )

        if self._can_append_customer(rs, cust_id):
            self._append_customer(rs, cust_id)
        else:
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
        P = self.problem
        nodes = P.nodes

        routes: List[Route] = []
        used_vehicles: Dict[int, int] = {}
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
                if rs.seq[-1] != rs.depot_id:
                    rs.seq.append(rs.depot_id)
                
                t_back = P.d(rs.seq[-2], rs.depot_id) / P.speed_units_per_min
                finish_time = rs.time + t_back
                vehicle_availability[rs.vehicle.id] = finish_time

                routes.append(Route(vehicle_id=rs.vehicle.id, seq=rs.seq))
                rs = self._open_new_route_for_customer(cust_id, used_vehicles, vehicle_availability)

        if rs is not None:
            if rs.seq[-1] != rs.depot_id:
                rs.seq.append(rs.depot_id)
            
            t_back = P.d(rs.seq[-2], rs.depot_id) / P.speed_units_per_min
            finish_time = rs.time + t_back
            vehicle_availability[rs.vehicle.id] = finish_time
            
            routes.append(Route(vehicle_id=rs.vehicle.id, seq=rs.seq))

        return Solution(routes=routes)

    # ============================================================
    # Phase 2: Cải thiện
    # ============================================================

    def _check_route_and_distance(self, seq: List[int], veh: Vehicle) -> Tuple[bool, float]:
        P = self.problem
        nodes = P.nodes
        speed = P.speed_units_per_min
        
        if not seq or len(seq) < 2:
            return True, 0.0

        time = float(veh.start_time)
        load = 0.0
        total_dist = 0.0

        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if (u, v) in P.prohibited:
                return False, float("inf")
                
            dist_uv = P.d(u, v)
            total_dist += dist_uv
            arrival = time + dist_uv / speed
            node_v = nodes[v]

            if node_v.is_depot:
                if arrival > veh.end_time:
                    return False, float("inf")
                time = arrival
                continue

            load += (node_v.demand_pickup - node_v.demand_delivery)
            if load > veh.capacity:
                return False, float("inf")

            tw_open = node_v.tw_open if node_v.tw_open is not None else -math.inf
            start_service = max(arrival, tw_open)
            time = start_service + node_v.service_time

        return True, total_dist

    def _improve_phase2(self, sol: Solution) -> Solution:
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

            feas1_cur, dist1_cur = self._check_route_and_distance(r1.seq, veh1)
            feas2_cur, dist2_cur = self._check_route_and_distance(r2.seq, veh2)
            if not (feas1_cur and feas2_cur):
                continue
            best_pair_cost = dist1_cur + dist2_cur

            best_new_seq1 = None
            best_new_seq2 = None

            base_seq1 = r1.seq[:-2] + [dep1]
            
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
    # GA components
    # ============================================================

    def _init_population(self) -> List[List[int]]:
        base = self._customers
        pop: List[List[int]] = []
        for _ in range(self.pop_size):
            chrom = base[:] 
            self.rng.shuffle(chrom)
            pop.append(chrom)
        return pop

    def _evaluate_chromosome(self, chrom: List[int]) -> float:
        sol_phase1 = self._decode_chromosome(chrom)
        sol = self._improve_phase2(sol_phase1)
        cost, _ = self.evaluator(self.problem, sol, return_details=False)
        return cost

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

    def _save_final_population_details(self, population: List[List[int]], filename: str = "ombuki_final_pop_details.csv"):
        import pandas as pd
        all_records = []
        for i, chrom in enumerate(population):
            sol_phase1 = self._decode_chromosome(chrom)
            sol = self._improve_phase2(sol_phase1)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {
                "individual_id": i,
                "total_cost": cost,
                **details,
                "solution_str": str(sol)
            }
            all_records.append(record)
        df = pd.DataFrame(all_records)
        df.to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 30000.0) -> Solution:
        r = self.rng
        t0 = time.time()
        pop = self._init_population()
        fitnesses: List[float] = [self._evaluate_chromosome(chrom) for chrom in pop]

        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_chrom = pop[best_idx][:]
        best_cost = fitnesses[best_idx]

        gen = 0
        while gen < self.max_generations and (time.time() - t0) < time_limit_sec:
            gen += 1
            print(f"Generation {gen}: best_cost = {best_cost:.2f}", end="\r")
            new_pop: List[List[int]] = []
            
            new_pop.append(best_chrom[:])

            while len(new_pop) < self.pop_size:
                parent1 = self._tournament_select(pop, fitnesses)
                child = parent1[:]
                
                if r.random() < self.crossover_rate:
                    parent2 = self._tournament_select(pop, fitnesses)
                    c1, c2 = self._crossover_ox(parent1, parent2)
                    child = c1 if r.random() < 0.5 else c2
                
                if r.random() < self.mutation_rate:
                    self._mutate_inversion(child)
                
                new_pop.append(child)

            pop = new_pop
            fitnesses = [self._evaluate_chromosome(chrom) for chrom in pop]

            cur_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            if fitnesses[cur_best_idx] < best_cost:
                best_cost = fitnesses[cur_best_idx]
                best_chrom = pop[cur_best_idx][:]

        self._save_final_population_details(pop, filename=f"last_generation/ombuki_final_pop_seed{self.seed}.csv")
        best_sol_phase1 = self._decode_chromosome(best_chrom)
        return self._improve_phase2(best_sol_phase1)