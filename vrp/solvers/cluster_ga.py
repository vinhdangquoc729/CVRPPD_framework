from __future__ import annotations
import random
import math
import time as _time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from .solver_base import Solver
from ..core.problem import Problem, Order
from ..core.solution import Solution, Route
from ..core.eval import evaluate as default_evaluator

def _kmeans(points: List[Tuple[float, float]], k: int, rng: random.Random, max_iter: int = 50) -> List[int]:
    """K-Means clustering gom nhóm đơn hàng theo vị trí."""
    n = len(points)
    if n == 0: return []
    k = max(1, min(k, n))

    centroids: List[Tuple[float, float]] = [points[rng.randrange(n)]]
    for _ in range(1, k):
        d2 = []
        for (px, py) in points:
            best = min((px - cx) ** 2 + (py - cy) ** 2 for (cx, cy) in centroids)
            d2.append(best)
        s = sum(d2) or 1.0
        r = rng.random() * s
        acc = 0.0
        pick = 0
        for i, w in enumerate(d2):
            acc += w
            if acc >= r:
                pick = i
                break
        centroids.append(points[pick])

    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, (px, py) in enumerate(points):
            best_c, best_d = 0, float("inf")
            for c, (cx, cy) in enumerate(centroids):
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best_d, best_c = d, c
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True

        sx = [0.0] * k
        sy = [0.0] * k
        cnt = [0] * k
        for (px, py), c in zip(points, labels):
            sx[c] += px
            sy[c] += py
            cnt[c] += 1

        for c in range(k):
            if cnt[c] > 0:
                centroids[c] = (sx[c] / cnt[c], sy[c] / cnt[c])
            else:
                centroids[c] = points[rng.randrange(n)]
                changed = True

        if not changed:
            break
    return labels

@dataclass
class Chromosome:
    assignment: List[int]        # Cluster i ƯU TIÊN gán cho Vehicle j
    intra_orders: List[List[int]] # Thứ tự đơn hàng trong Cluster i
    cluster_order: List[int]     # Thứ tự xử lý các Cluster

class ClusterGASolver(Solver):
    def __init__(
        self,
        problem: Problem,
        seed: int = 42,
        avg_cluster_size: int = 5,
        pop_size: int = 50,
        generations: int = 500,
        cx_prob: float = 0.9,
        mut_prob: float = 0.2,
        elite_frac: float = 0.10,
        use_gene_inter_order: bool = True,
        evaluator: callable = None,
    ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.avg_cluster_size = max(3, avg_cluster_size)
        self.pop_size = max(10, pop_size)
        self.generations = max(1, generations)
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite = max(1, int(self.pop_size * elite_frac))
        self.use_gene_inter_order = use_gene_inter_order
        self.evaluator = evaluator if evaluator is not None else default_evaluator

        self.order_ids: List[int] = list(self.problem.orders_map.keys())
        self.vehicles = list(self.problem.vehicles)
        
        # 1. Gom cụm K-Means
        self.clusters: List[List[int]] = self._build_clusters()
        
        # Cache xe theo depot để tra cứu nhanh
        self.veh_by_depot = defaultdict(list)
        for vidx, v in enumerate(self.vehicles):
            self.veh_by_depot[v.start_depot_id].append(vidx)
        self.all_depot_ids = list(self.veh_by_depot.keys())

    def _build_clusters(self) -> List[List[int]]:
        P = self.problem
        n = len(self.order_ids)
        if n == 0: return []
        
        k = max(1, (n + self.avg_cluster_size - 1) // self.avg_cluster_size)
        pts = []
        for oid in self.order_ids:
            nd = P.nodes_map[P.orders_map[oid].node_id]
            pts.append((nd.x, nd.y))
            
        labels = _kmeans(pts, k, self.rng, max_iter=50)
        groups = [[] for _ in range(max(labels) + 1)]
        for oid, lab in zip(self.order_ids, labels):
            groups[lab].append(oid)
        return [g for g in groups if g]

    def _random_chromosome(self) -> Chromosome:
        """
        Khởi tạo chromosome. Gán cluster cho xe dựa trên:
        1. Khoảng cách gần nhất.
        2. Mức độ tương thích hàng hóa (cố gắng chọn xe chở được nhiều hàng nhất trong cụm).
        """
        P = self.problem
        assignment: List[int] = []
        
        for group in self.clusters:
            # Tính tâm cluster
            sum_x, sum_y = 0.0, 0.0
            cluster_goods = set()
            for oid in group:
                o = P.orders_map[oid]
                sum_x += P.nodes_map[o.node_id].x
                sum_y += P.nodes_map[o.node_id].y
                cluster_goods.update(o.contained_goods_types)
                
            cx = sum_x / len(group)
            cy = sum_y / len(group)
            
            # Sắp xếp depot theo khoảng cách
            sorted_depots = sorted(
                self.all_depot_ids, 
                key=lambda d: (P.nodes_map[d].x - cx)**2 + (P.nodes_map[d].y - cy)**2
            )
            
            picked_v_idx = None
            
            # Tìm xe phù hợp nhất
            for dep_id in sorted_depots:
                veh_indices = self.veh_by_depot.get(dep_id, [])
                # Ưu tiên xe chở được TOÀN BỘ loại hàng trong cụm
                perfect_vehs = [vi for vi in veh_indices if cluster_goods.issubset(self.vehicles[vi].allowed_goods_types)]
                
                if perfect_vehs:
                    picked_v_idx = self.rng.choice(perfect_vehs)
                    break
                
                # Nếu không có xe perfect, tìm xe chở được ÍT NHẤT 1 loại (để fallback)
                # Nhưng ưu tiên tiếp tục tìm ở depot xa hơn xem có xe perfect không
            
            # Nếu duyệt hết depot mà không có xe perfect, chọn xe bất kỳ ở depot gần nhất
            if picked_v_idx is None:
                nearest = sorted_depots[0]
                if self.veh_by_depot[nearest]:
                    picked_v_idx = self.rng.choice(self.veh_by_depot[nearest])
                else:
                    picked_v_idx = self.rng.randrange(len(self.vehicles))

            assignment.append(picked_v_idx)

        intra = [g[:] for g in self.clusters]
        for g in intra: self.rng.shuffle(g)
        
        c_order = list(range(len(self.clusters)))
        self.rng.shuffle(c_order)
        
        return Chromosome(assignment, intra, c_order)

    def _decode(self, chrom: Chromosome) -> Solution:
        """
        Giải mã:
        - Duyệt từng order trong cluster.
        - Kiểm tra tính tương thích với xe được gán (trong assignment).
        - Nếu tương thích -> Gán.
        - Nếu KHÔNG tương thích -> Đẩy vào danh sách 'orphans' để xử lý sau.
        """
        P = self.problem
        
        # 1. Tổ chức dữ liệu từ Chromosome
        # cluster_queue_by_veh[v_idx] = [cluster_idx, cluster_idx, ...]
        cluster_queue_by_veh = [[] for _ in self.vehicles]
        
        # Xác định thứ tự ưu tiên các cluster
        priority_map = {c: i for i, c in enumerate(chrom.cluster_order)}
        
        for c_idx, v_idx in enumerate(chrom.assignment):
            v_idx = max(0, min(v_idx, len(self.vehicles) - 1))
            cluster_queue_by_veh[v_idx].append(c_idx)
            
        # Sắp xếp cluster trong từng xe theo cluster_order
        for v_idx in range(len(self.vehicles)):
            cluster_queue_by_veh[v_idx].sort(key=lambda c: priority_map[c])

        # 2. Xây dựng lộ trình (Route) và lọc Orphans
        vehicle_routes_data = defaultdict(list) # v_id -> list[order_id]
        orphans = []
        
        for v_idx, veh in enumerate(self.vehicles):
            assigned_clusters = cluster_queue_by_veh[v_idx]
            
            for c_idx in assigned_clusters:
                # Duyệt từng đơn trong cụm
                for oid in chrom.intra_orders[c_idx]:
                    order = P.orders_map[oid]
                    # CHECK CỨNG: Xe có được chở loại hàng này không?
                    if order.contained_goods_types.issubset(veh.allowed_goods_types):
                        vehicle_routes_data[veh.id].append(oid)
                    else:
                        # Xe này không chở được -> Đẩy ra ngoài
                        orphans.append(oid)

        # 3. Xử lý Orphans (Những đơn hàng bị đá ra do xe không hợp lệ)
        # Gán orphans vào xe PHÙ HỢP NHẤT (Gần nhất + Đúng loại hàng)
        if orphans:
            # Shuffle orphans để ngẫu nhiên hóa việc chèn
            # self.rng.shuffle(orphans) # (Tuỳ chọn)
            
            for oid in orphans:
                order = P.orders_map[oid]
                best_veh = None
                best_dist = float('inf')
                
                # Duyệt qua TẤT CẢ xe để tìm cứu cánh
                for veh in self.vehicles:
                    # Điều kiện tiên quyết: Phải chở được hàng
                    if order.contained_goods_types.issubset(veh.allowed_goods_types):
                        # Tính khoảng cách từ Depot xe đó tới Order
                        d = P.get_dist_node_to_node(veh.start_depot_id, order.node_id)
                        if d < best_dist:
                            best_dist = d
                            best_veh = veh
                
                # Gán vào xe cứu cánh
                if best_veh:
                    # Chèn vào cuối lộ trình hiện tại của xe đó
                    vehicle_routes_data[best_veh.id].append(oid)
                else:
                    # Trường hợp cực đoan: Không xe nào trên hệ thống chở được
                    # Gán đại vào xe gần nhất (chấp nhận lỗi goods_not_allowed)
                    fallback_veh = min(self.vehicles, key=lambda v: P.get_dist_node_to_node(v.start_depot_id, order.node_id))
                    vehicle_routes_data[fallback_veh.id].append(oid)

        # 4. Tạo Solution Object và ngắt chuyến (Multi-trip logic)
        final_routes: List[Route] = []
        
        for v_idx, veh in enumerate(self.vehicles):
            seq = vehicle_routes_data.get(veh.id, [])
            if not seq: continue
            
            # Logic ngắt chuyến đơn giản dựa trên Capacity
            current_trip = []
            current_w = 0.0
            
            for oid in seq:
                w = P.orders_map[oid].total_weight
                # Nếu thêm vào mà quá tải (hệ số 1.2 cho phép du di nhẹ) -> Ngắt
                if (current_w + w > veh.capacity * 1.2) and current_trip:
                    final_routes.append(Route(veh.id, current_trip))
                    current_trip = []
                    current_w = 0.0
                
                current_trip.append(oid)
                current_w += w
            
            if current_trip:
                final_routes.append(Route(veh.id, current_trip))
                
        return Solution(final_routes)

    # ... (Các hàm _fitness, _cx, _mutate giữ nguyên như cũ) ...
    # Để tiết kiệm token, tôi chỉ paste lại những phần logic thay đổi ở trên.
    # Bạn hãy giữ nguyên các hàm phụ trợ (_order_clusters_nn, _fitness, lai ghép, đột biến)
    # từ phiên bản trước.
    
    def _order_clusters_nn(self, clist: List[int], depot_id: int) -> List[int]:
        if not clist: return []
        P = self.problem
        curr_node_id = depot_id
        remaining = set(clist)
        order = []
        while remaining:
            best_c = min(remaining, key=lambda c: self._approx_dist_to_cluster(P, curr_node_id, c))
            order.append(best_c)
            first_order_id = self.clusters[best_c][0]
            curr_node_id = P.orders_map[first_order_id].node_id
            remaining.remove(best_c)
        return order

    def _approx_dist_to_cluster(self, P: Problem, from_node_id: int, c_idx: int) -> float:
        return min(P.get_dist_node_to_node(from_node_id, P.orders_map[oid].node_id) for oid in self.clusters[c_idx])

    def _fitness(self, chrom: Chromosome) -> float:
        sol = self._decode(chrom)
        res = self.evaluator(self.problem, sol, return_details=False)
        return res[0] if isinstance(res, tuple) else float(res)

    def _cx_uniform_assignment(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        nC = len(self.clusters)
        ass1, ass2 = a.assignment[:], b.assignment[:]
        intra1, intra2 = [o[:] for o in a.intra_orders], [o[:] for o in b.intra_orders]
        for c in range(nC):
            if self.rng.random() < 0.5: ass1[c], ass2[c] = ass2[c], ass1[c]
            if self.rng.random() < 0.2: intra1[c], intra2[c] = intra2[c], intra1[c]
        ord1 = self._ox(a.cluster_order, b.cluster_order)
        ord2 = self._ox(b.cluster_order, a.cluster_order)
        return Chromosome(ass1, intra1, ord1), Chromosome(ass2, intra2, ord2)

    def _ox(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        if n <= 1: return p1[:]
        i, j = sorted(self.rng.sample(range(n), 2))
        child = [None] * n
        child[i:j] = p1[i:j]
        fill = [x for x in p2 if x not in child[i:j]]
        k = 0
        for t in range(n):
            if child[t] is None:
                child[t] = fill[k]
                k += 1
        return child

    def _mutate(self, x: Chromosome) -> None:
        nV, nC = len(self.vehicles), len(x.assignment)
        # Mutate assignment
        for _ in range(self.rng.randrange(1, max(1, nC // 20) + 1)):
            x.assignment[self.rng.randrange(nC)] = self.rng.randrange(nV)
        # Mutate intra
        for _ in range(self.rng.randrange(1, max(1, nC // 10) + 1)):
            c = self.rng.randrange(nC)
            seq = x.intra_orders[c]
            if len(seq) >= 2:
                i, j = self.rng.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
        # Mutate inter
        if nC >= 2 and self.rng.random() < 0.7:
            i, j = self.rng.sample(range(nC), 2)
            x.cluster_order[i], x.cluster_order[j] = x.cluster_order[j], x.cluster_order[i]

    def _save_final_population_details(self, population: List[Chromosome], filename: str = "cluster_ga_final_pop_details.csv"):
        import pandas as pd
        import os
        all_records = []
        for i, chrom in enumerate(population):
            sol = self._decode(chrom)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {"individual_id": i, "total_cost": cost, **details, "solution_str": str(sol)}
            all_records.append(record)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pd.DataFrame(all_records).to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 300.0, max_generations: Optional[int] = None, patience_gens: int = 50) -> Solution:
        if max_generations is None: max_generations = self.generations
        print("Initializing ClusterGA PD population...")
        pop = [self._random_chromosome() for _ in range(self.pop_size)]
        fit_cache: Dict[int, float] = {}

        def fitness(ch: Chromosome) -> float:
            key = id(ch)
            if key not in fit_cache: fit_cache[key] = self._fitness(ch)
            return fit_cache[key]

        pop.sort(key=fitness)
        best, best_cost, no_improve, gen, t0 = pop[0], fitness(pop[0]), 0, 0, _time.time()
        print(f"ClusterGA Start. Initial Best: {best_cost:.2f}")

        while gen < max_generations and (_time.time() - t0) < time_limit_sec:
            new_pop = pop[:min(self.elite, len(pop))]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.rng.sample(pop, 2), self.rng.sample(pop, 2)
                p1.sort(key=fitness); p2.sort(key=fitness)
                
                if self.rng.random() < self.cx_prob:
                    c1, c2 = self._cx_uniform_assignment(p1[0], p2[0])
                else:
                    c1 = Chromosome(p1[0].assignment[:], [o[:] for o in p1[0].intra_orders], p1[0].cluster_order[:])
                    c2 = Chromosome(p2[0].assignment[:], [o[:] for o in p2[0].intra_orders], p2[0].cluster_order[:])
                
                if self.rng.random() < self.mut_prob: self._mutate(c1)
                if self.rng.random() < self.mut_prob: self._mutate(c2)
                new_pop.extend([c1, c2])

            pop, fit_cache = new_pop[:self.pop_size], {}
            pop.sort(key=fitness)
            current_best = fitness(pop[0])
            
            if current_best < best_cost:
                best, best_cost, no_improve = pop[0], current_best, 0
            else:
                no_improve += 1
            
            gen += 1
            if gen % 10 == 0:
                print(f"Gen {gen}: {best_cost:.2f}", end="\r")
            
            if patience_gens and no_improve >= patience_gens:
                print(f"\nEarly stop at gen {gen}")
                break

        print(f"\nFinished. Best: {best_cost:.2f}")
        self._save_final_population_details(pop, filename=f"last_generation/cluster_ga_final_pop_seed{self.seed}_{len(self.problem.nodes_map)}_{self.evaluator.__name__}.csv")
        return self._decode(best)