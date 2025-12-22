from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .solver_base import Solver
from ..core.problem import Problem, Node
from ..core.solution import Solution, Route
from ..core.eval import evaluate as evaluator


def _kmeans(points: List[Tuple[float, float]], k: int, rng: random.Random, max_iter: int = 50) -> List[int]:
    n = len(points)
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

    # Lặp thuật toán kmeans
    labels = [0] * n
    for _ in range(max_iter):
        changed = False

        # Gán mỗi điểm về tâm gần nhất
        for i, (px, py) in enumerate(points):
            best_c, best_d = 0, float("inf")
            for c, (cx, cy) in enumerate(centroids):
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best_d, best_c = d, c
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True

        # Cập nhật lại tâm
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
                # cụm rỗng -> gieo lại ngẫu nhiên
                centroids[c] = points[rng.randrange(n)]
                changed = True

        if not changed:
            break

    return labels

@dataclass
class Chromosome:
    """
    Mã hoá lời giải ở cấp cụm:
      - assignment[c]: cụm c gán cho xe nào (chỉ số trong self.vehicles)
      - intra_orders[c]: thứ tự thăm các khách *bên trong* cụm c (list id khách)
      - cluster_order: hoán vị toàn cục các cụm (thứ tự thăm các cụm)
    """
    assignment: List[int]
    intra_orders: List[List[int]]
    cluster_order: List[int]

class ClusterGASolver(Solver):
    """
    1) Gom khách thành các cụm nhỏ (~avg_cluster_size khách/cụm) bằng k-means.
    2) GA hoạt động trên 3 gene: gán cụm->xe, thứ tự trong cụm, và hoán vị giữa các cụm.
    3) Giải mã: với mỗi xe, lấy các cụm được gán, sắp theo 'cluster_order' (hoặc NN),
      rồi ghép thứ tự khách trong từng cụm, kẹp giữa [depot ... depot].
    """

    def __init__(
        self,
        problem: Problem,
        seed: int = 42,
        avg_cluster_size: int = 5,   # số khách mục tiêu trong mỗi cụm
        pop_size: int = 80,
        generations: int = 2000,
        cx_prob: float = 0.9,
        mut_prob: float = 0.2,
        elite_frac: float = 0.10,
        use_gene_inter_order: bool = True,  # True: dùng cluster_order; False: dùng nearest neighbor giữa cụm
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

        # Danh sách id khách/xe để tiện truy xuất
        self.customers: List[int] = [i for i, nd in self.problem.nodes.items() if not nd.is_depot]
        self.vehicles = list(self.problem.vehicles)

        # Tiền xử lý: tạo cụm 1 lần
        self.clusters: List[List[int]] = self._build_clusters()

        # Ánh xạ: id khách -> chỉ số cụm
        self.cust2cluster: Dict[int, int] = {}
        for c_idx, group in enumerate(self.clusters):
            for cid in group:
                self.cust2cluster[cid] = c_idx

    def _build_clusters(self) -> List[List[int]]:
        """Chia cụm dựa trên toạ độ."""
        P = self.problem
        n = len(self.customers)
        if n == 0:
            return []
        k = max(1, (n + self.avg_cluster_size - 1) // self.avg_cluster_size)

        pts: List[Tuple[float, float]] = []
        for cid in self.customers:
            nd: Node = P.nodes[cid]
            pts.append((nd.x, nd.y))

        labels = _kmeans(pts, k, self.rng, max_iter=50)

        # gom khách theo nhãn cụm
        groups: List[List[int]] = [[] for _ in range(max(labels) + 1)]
        for cid, lab in zip(self.customers, labels):
            groups[lab].append(cid)

        # loại cụm rỗng (nếu có)
        groups = [g for g in groups if g]
        return groups

    def _random_chromosome(self) -> Chromosome:
        """
        Khởi tạo ngẫu nhiên:
        - Gán mỗi cụm cho một xe ngẫu nhiên thuộc depot gần trọng tâm cụm nhất.
        - Xáo trộn thứ tự khách bên trong từng cụm.
        - Sinh hoán vị ngẫu nhiên cho danh sách cụm.
        """
        P = self.problem

        veh_by_depot: Dict[int, List[int]] = {}
        for vidx, v in enumerate(self.vehicles):
            veh_by_depot.setdefault(v.depot_id, []).append(vidx)

        def nearest_depots_for_centroid(cx: float, cy: float) -> List[int]:
            """Trả về danh sách depot_id sắp theo khoảng cách tăng dần tới (cx, cy)."""
            return sorted(
                P.depots,
                key=lambda dep_id: (P.nodes[dep_id].x - cx) ** 2 + (P.nodes[dep_id].y - cy) ** 2
            )

        # Gene gán cụm -> xe
        assignment: List[int] = []
        for group in self.clusters:
            # trọng tâm cụm
            cx = sum(P.nodes[c].x for c in group) / len(group)
            cy = sum(P.nodes[c].y for c in group) / len(group)

            # duyệt các depot theo thứ tự gần - xa, chọn ngẫu nhiên 1 xe của depot đầu tiên có xe
            picked_v_idx: Optional[int] = None
            for dep_id in nearest_depots_for_centroid(cx, cy):
                veh_idxs = veh_by_depot.get(dep_id, [])
                if veh_idxs:
                    picked_v_idx = self.rng.choice(veh_idxs)  # chọn ngẫu nhiên trong đội xe của depot này
                    break
            if picked_v_idx is None:
                picked_v_idx = self.rng.randrange(len(self.vehicles)) if self.vehicles else 0

            assignment.append(picked_v_idx)

        # Gene thứ tự bên trong cụm (intra_orders)
        intra_orders: List[List[int]] = []
        for group in self.clusters:
            g = group[:]
            self.rng.shuffle(g)
            intra_orders.append(g)

        # Gene hoán vị cụm (cluster_order)
        cluster_order = list(range(len(self.clusters)))
        self.rng.shuffle(cluster_order)

        return Chromosome(assignment=assignment,
                          intra_orders=intra_orders,
                          cluster_order=cluster_order)


    def _decode(self, chrom: Chromosome) -> Solution:
        """
        Giải mã nhiễm sắc thể thành Solution hỗ trợ Multi-trip (Journey).
        Chiến thuật: Greedy Splitting (Tham lam ngắt chuyến).
        - Duyệt các cụm được gán cho xe.
        - Nếu cụm tiếp theo làm xe quá tải -> Về kho, lập chuyến mới.
        """
        P = self.problem
        
        # 1. Gom nhóm cụm theo xe
        clusters_by_vehicle: List[List[int]] = [[] for _ in self.vehicles]
        for c_idx, v_idx in enumerate(chrom.assignment):
            # Clip v_idx để an toàn
            v_idx = max(0, min(v_idx, len(self.vehicles) - 1))
            clusters_by_vehicle[v_idx].append(c_idx)

        routes: List[Route] = []

        # 2. Xử lý từng xe
        for v_idx, veh in enumerate(self.vehicles):
            assigned_clusters = clusters_by_vehicle[v_idx]
            if not assigned_clusters:
                # Nếu xe không được gán gì, tạo route rỗng để giữ chỗ (nếu muốn)
                # routes.append(Route(vehicle_id=veh.id, seq=[veh.depot_id, veh.depot_id]))
                continue

            # Sắp xếp thứ tự các cụm
            if self.use_gene_inter_order:
                # Dựa trên gene cluster_order
                pos = {c: i for i, c in enumerate(chrom.cluster_order)}
                ordered_clusters = sorted(assigned_clusters, key=lambda c: pos[c])
            else:
                # Dựa trên heuristic Nearest Neighbor
                ordered_clusters = self._order_clusters_nn(assigned_clusters, veh.depot_id)

            # --- LOGIC NGẮT CHUYẾN (SPLIT) ---
            current_trip_nodes: List[int] = []
            
            # Theo dõi tải trọng để quyết định khi nào về kho
            # (Heuristic đơn giản: tổng giao hoặc tổng nhận không vượt quá capacity)
            current_load_delivery = 0
            current_load_pickup = 0
            capacity = veh.capacity
            depot_id = veh.depot_id

            for c_idx in ordered_clusters:
                # Lấy danh sách khách trong cụm (theo thứ tự gene intra_orders)
                cluster_nodes = chrom.intra_orders[c_idx]
                
                # Tính tổng nhu cầu của cụm này
                c_del = sum(P.nodes[n].demand_delivery for n in cluster_nodes)
                c_pick = sum(P.nodes[n].demand_pickup for n in cluster_nodes)

                # Kiểm tra: Nếu thêm cụm này mà quá tải -> Ngắt chuyến cũ, tạo chuyến mới
                # (Đây là check đơn giản, eval.py sẽ check kỹ hơn về simultaneous)
                is_overload = (current_load_delivery + c_del > capacity) or \
                              (current_load_pickup + c_pick > capacity)

                if current_trip_nodes and is_overload:
                    # 1. Đóng gói chuyến cũ: Depot -> Khách -> Depot
                    seq = [depot_id] + current_trip_nodes + [depot_id]
                    routes.append(Route(vehicle_id=veh.id, seq=seq))
                    
                    # 2. Reset cho chuyến mới
                    current_trip_nodes = []
                    current_load_delivery = 0
                    current_load_pickup = 0

                # Thêm cụm vào chuyến hiện tại (hoặc chuyến mới vừa reset)
                current_trip_nodes.extend(cluster_nodes)
                current_load_delivery += c_del
                current_load_pickup += c_pick

            # Đừng quên lưu chuyến cuối cùng còn dang dở
            if current_trip_nodes:
                seq = [depot_id] + current_trip_nodes + [depot_id]
                routes.append(Route(vehicle_id=veh.id, seq=seq))

        return Solution(routes=routes)

    def _order_clusters_nn(self, clist: List[int], depot_id: int) -> List[int]:
        """
        Sắp thứ tự thăm các cụm bằng chiến lược hàng xóm gần nhất (NN):
        xuất phát từ depot -> mỗi bước chọn cụm có khách gần nhất.
        """
        if not clist:
            return []
        P = self.problem
        curr = depot_id
        remaining = set(clist)
        order: List[int] = []
        while remaining:
            best_c = None
            best_d = float("inf")
            for c in remaining:
                d = self._approx_dist_to_cluster(P, curr, c)
                if d < best_d:
                    best_d, best_c = d, c
            order.append(best_c)  # type: ignore
            curr = self._representative_of_cluster(best_c)  # type: ignore
            remaining.remove(best_c)  # type: ignore
        return order

    def _approx_dist_to_cluster(self, P: Problem, from_node: int, c_idx: int) -> float:
        """Khoảng cách xấp xỉ từ 1 đỉnh tới cụm = min khoảng cách tới bất kỳ khách nào trong cụm."""
        best = float("inf")
        for cid in self.clusters[c_idx]:
            d = P.d(from_node, cid)
            if d < best:
                best = d
        return best

    def _representative_of_cluster(self, c_idx: int) -> int:
        """Chọn đại diện cụm (đơn giản: khách đầu tiên trong danh sách)."""
        return self.clusters[c_idx][0]

    def _fitness(self, chrom: Chromosome) -> float:
        sol = self._decode(chrom)
        res = evaluator(self.problem, sol, return_details=False)
        # evaluator có thể trả float hoặc (float, details)
        return res[0] if isinstance(res, tuple) else float(res)

    def _cx_uniform_assignment(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        nC = len(self.clusters)

        # copy gene
        ass1, ass2 = a.assignment[:], b.assignment[:]
        intra1 = [ordr[:] for ordr in a.intra_orders]
        intra2 = [ordr[:] for ordr in b.intra_orders]

        # assignment + intra_orders
        for c in range(nC):
            if self.rng.random() < 0.5:
                ass1[c], ass2[c] = ass2[c], ass1[c]
            if self.rng.random() < 0.2:
                intra1[c], intra2[c] = intra2[c], intra1[c]

        # OX cho cluster_order (giữ cấu trúc hoán vị)
        ord1 = self._ox(a.cluster_order, b.cluster_order)
        ord2 = self._ox(b.cluster_order, a.cluster_order)

        return Chromosome(ass1, intra1, ord1), Chromosome(ass2, intra2, ord2)

    def _ox(self, p1: List[int], p2: List[int]) -> List[int]:
        """Order Crossover (OX) trên hoán vị."""
        n = len(p1)
        if n <= 1:
            return p1[:]
        i, j = sorted(self.rng.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n
        child[i:j] = p1[i:j]
        fill = [x for x in p2 if x not in child[i:j]]
        k = 0
        for t in range(n):
            if child[t] is None:
                child[t] = fill[k]
                k += 1
        return child 

    def _mutate(self, x: Chromosome) -> None:

        nV = len(self.vehicles)
        nC = len(x.assignment)

        # assignment: reassign cụm
        m = max(1, nC // 20)
        for _ in range(self.rng.randrange(1, m + 1)):
            c = self.rng.randrange(nC)
            x.assignment[c] = self.rng.randrange(nV)

        # intra_orders: đảo đoạn/hoán vị
        k = max(1, nC // 10)
        for _ in range(self.rng.randrange(1, k + 1)):
            c = self.rng.randrange(nC)
            seq = x.intra_orders[c]
            if len(seq) >= 3 and self.rng.random() < 0.5:
                i = self.rng.randrange(0, len(seq) - 1)
                j = self.rng.randrange(i + 1, len(seq))
                seq[i:j] = reversed(seq[i:j])
            elif len(seq) >= 2:
                i = self.rng.randrange(len(seq))
                j = self.rng.randrange(len(seq))
                seq[i], seq[j] = seq[j], seq[i]

        # cluster_order: swap 2 vị trí + đảo đoạn
        if nC >= 2 and self.rng.random() < 0.7:
            i, j = self.rng.randrange(nC), self.rng.randrange(nC)
            x.cluster_order[i], x.cluster_order[j] = x.cluster_order[j], x.cluster_order[i]
        if nC >= 3 and self.rng.random() < 0.3:
            i, j = sorted(self.rng.sample(range(nC), 2))
            x.cluster_order[i:j] = reversed(x.cluster_order[i:j])

    def solve(
        self,
        time_limit_sec: float = 300.0,
        max_generations: Optional[int] = None,
        patience_gens: int = 50,
    ) -> Solution:
        import time

        if max_generations is None:
            max_generations = self.generations

        # Khởi tạo quần thể
        pop: List[Chromosome] = [self._random_chromosome() for _ in range(self.pop_size)]
        fit_cache: Dict[int, float] = {}

        def fitness(ch: Chromosome) -> float:
            """Cache cost theo id(chromosome) để tránh decode/evaluate lặp lại."""
            key = id(ch)
            if key not in fit_cache:
                fit_cache[key] = self._fitness(ch)
            return fit_cache[key]

        # Đánh giá ban đầu + sắp xếp
        pop.sort(key=fitness)
        best = pop[0]
        best_cost = fitness(best)
        no_improve = 0
        gen = 0
        t0 = time.time()

        def tournament(k: int = 3) -> Chromosome:
            cand = self.rng.sample(pop, k if len(pop) >= k else max(2, len(pop)))
            cand.sort(key=fitness)
            return cand[0]

        # Tiến hoá
        while gen < max_generations and (time.time() - t0) < time_limit_sec:
            new_pop: List[Chromosome] = []

            elite_count = min(self.elite, len(pop))
            elites = pop[:elite_count]
            new_pop.extend(elites)

            while len(new_pop) < self.pop_size:
                p1 = tournament()
                p2 = tournament()

                if self.rng.random() < self.cx_prob:
                    c1, c2 = self._cx_uniform_assignment(p1, p2)
                else:
                    c1 = Chromosome(p1.assignment[:],
                                    [s[:] for s in p1.intra_orders],
                                    p1.cluster_order[:])
                    c2 = Chromosome(p2.assignment[:],
                                    [s[:] for s in p2.intra_orders],
                                    p2.cluster_order[:])

                if self.rng.random() < self.mut_prob:
                    self._mutate(c1)
                if self.rng.random() < self.mut_prob:
                    self._mutate(c2)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            fit_cache.clear()
            pop.sort(key=fitness)

            cur_best = pop[0]
            cur_cost = fitness(cur_best)
            if cur_cost < best_cost:
                best = cur_best
                best_cost = cur_cost
                no_improve = 0
            else:
                no_improve += 1

            if patience_gens and no_improve >= patience_gens:
                break

            gen += 1
            print(f"Generation {gen}: best_cost = {best_cost}", end="\r")

        return self._decode(best)
