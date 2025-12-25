from __future__ import annotations
import random
import math
import time as _time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .solver_base import Solver
from ..core.problem import Problem, Node
from ..core.solution import Solution, Route
from ..core.eval import evaluate as default_evaluator

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
    assignment: List[int]
    intra_orders: List[List[int]]
    cluster_order: List[int]

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

        self.customers: List[int] = [i for i, nd in self.problem.nodes.items() if not nd.is_depot]
        self.vehicles = list(self.problem.vehicles)
        self.clusters: List[List[int]] = self._build_clusters()

        self.cust2cluster: Dict[int, int] = {}
        for c_idx, group in enumerate(self.clusters):
            for cid in group:
                self.cust2cluster[cid] = c_idx

    def _build_clusters(self) -> List[List[int]]:
        P = self.problem
        n = len(self.customers)
        if n == 0: return []
        k = max(1, (n + self.avg_cluster_size - 1) // self.avg_cluster_size)
        pts: List[Tuple[float, float]] = []
        for cid in self.customers:
            nd: Node = P.nodes[cid]
            pts.append((nd.x, nd.y))
        labels = _kmeans(pts, k, self.rng, max_iter=50)
        groups: List[List[int]] = [[] for _ in range(max(labels) + 1)]
        for cid, lab in zip(self.customers, labels):
            groups[lab].append(cid)
        return [g for g in groups if g]

    def _random_chromosome(self) -> Chromosome:
        P = self.problem
        veh_by_depot: Dict[int, List[int]] = {}
        for vidx, v in enumerate(self.vehicles):
            veh_by_depot.setdefault(v.depot_id, []).append(vidx)

        def nearest_depots_for_centroid(cx: float, cy: float) -> List[int]:
            return sorted(P.depots, key=lambda d: (P.nodes[d].x - cx) ** 2 + (P.nodes[d].y - cy) ** 2)

        assignment: List[int] = []
        for group in self.clusters:
            cx = sum(P.nodes[c].x for c in group) / len(group)
            cy = sum(P.nodes[c].y for c in group) / len(group)
            picked_v_idx = None
            for dep_id in nearest_depots_for_centroid(cx, cy):
                veh_idxs = veh_by_depot.get(dep_id, [])
                if veh_idxs:
                    picked_v_idx = self.rng.choice(veh_idxs)
                    break
            if picked_v_idx is None:
                picked_v_idx = self.rng.randrange(len(self.vehicles)) if self.vehicles else 0
            assignment.append(picked_v_idx)

        intra_orders = []
        for group in self.clusters:
            g = group[:]
            self.rng.shuffle(g)
            intra_orders.append(g)

        cluster_order = list(range(len(self.clusters)))
        self.rng.shuffle(cluster_order)
        return Chromosome(assignment, intra_orders, cluster_order)

    def _decode(self, chrom: Chromosome) -> Solution:
        P = self.problem
        clusters_by_vehicle: List[List[int]] = [[] for _ in self.vehicles]
        for c_idx, v_idx in enumerate(chrom.assignment):
            v_idx = max(0, min(v_idx, len(self.vehicles) - 1))
            clusters_by_vehicle[v_idx].append(c_idx)

        routes: List[Route] = []
        for v_idx, veh in enumerate(self.vehicles):
            assigned_clusters = clusters_by_vehicle[v_idx]
            if not assigned_clusters: continue

            if self.use_gene_inter_order:
                pos = {c: i for i, c in enumerate(chrom.cluster_order)}
                ordered_clusters = sorted(assigned_clusters, key=lambda c: pos[c])
            else:
                ordered_clusters = self._order_clusters_nn(assigned_clusters, veh.depot_id)

            current_trip_nodes = []
            current_load_delivery = 0
            current_load_pickup = 0
            capacity = veh.capacity
            depot_id = veh.depot_id

            for c_idx in ordered_clusters:
                cluster_nodes = chrom.intra_orders[c_idx]
                c_del = sum(P.nodes[n].demand_delivery for n in cluster_nodes)
                c_pick = sum(P.nodes[n].demand_pickup for n in cluster_nodes)
                is_overload = (current_load_delivery + c_del > capacity) or (current_load_pickup + c_pick > capacity)

                if current_trip_nodes and is_overload:
                    routes.append(Route(veh.id, [depot_id] + current_trip_nodes + [depot_id]))
                    current_trip_nodes, current_load_delivery, current_load_pickup = [], 0, 0

                current_trip_nodes.extend(cluster_nodes)
                current_load_delivery += c_del
                current_load_pickup += c_pick

            if current_trip_nodes:
                routes.append(Route(veh.id, [depot_id] + current_trip_nodes + [depot_id]))
        return Solution(routes)

    def _order_clusters_nn(self, clist: List[int], depot_id: int) -> List[int]:
        if not clist: return []
        P, curr, remaining, order = self.problem, depot_id, set(clist), []
        while remaining:
            best_c = min(remaining, key=lambda c: self._approx_dist_to_cluster(P, curr, c))
            order.append(best_c)
            curr = self.clusters[best_c][0]
            remaining.remove(best_c)
        return order

    def _approx_dist_to_cluster(self, P: Problem, from_node: int, c_idx: int) -> float:
        return min(P.d(from_node, cid) for cid in self.clusters[c_idx])

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
        m = max(1, nC // 20)
        for _ in range(self.rng.randrange(1, m + 1)):
            x.assignment[self.rng.randrange(nC)] = self.rng.randrange(nV)
        k = max(1, nC // 10)
        for _ in range(self.rng.randrange(1, k + 1)):
            c = self.rng.randrange(nC)
            seq = x.intra_orders[c]
            if len(seq) >= 3 and self.rng.random() < 0.5:
                i, j = sorted(self.rng.sample(range(len(seq)), 2))
                seq[i:j] = reversed(seq[i:j])
            elif len(seq) >= 2:
                i, j = self.rng.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
        if nC >= 2 and self.rng.random() < 0.7:
            i, j = self.rng.sample(range(nC), 2)
            x.cluster_order[i], x.cluster_order[j] = x.cluster_order[j], x.cluster_order[i]
        if nC >= 3 and self.rng.random() < 0.3:
            i, j = sorted(self.rng.sample(range(nC), 2))
            x.cluster_order[i:j] = reversed(x.cluster_order[i:j])

    def _save_final_population_details(self, population: List[Chromosome], filename: str = "cluster_ga_final_pop_details.csv"):
        import pandas as pd
        all_records = []
        for i, chrom in enumerate(population):
            sol = self._decode(chrom)
            cost, details = self.evaluator(self.problem, sol, return_details=True)
            record = {"individual_id": i, "total_cost": cost, **details, "solution_str": str(sol)}
            all_records.append(record)
        pd.DataFrame(all_records).to_csv(filename, index=False)

    def solve(self, time_limit_sec: float = 300.0, max_generations: Optional[int] = None, patience_gens: int = 50) -> Solution:
        if max_generations is None: max_generations = self.generations
        pop = [self._random_chromosome() for _ in range(self.pop_size)]
        fit_cache: Dict[int, float] = {}

        def fitness(ch: Chromosome) -> float:
            key = id(ch)
            if key not in fit_cache: fit_cache[key] = self._fitness(ch)
            return fit_cache[key]

        pop.sort(key=fitness)
        best, best_cost, no_improve, gen, t0 = pop[0], fitness(pop[0]), 0, 0, _time.time()

        while gen < max_generations and (_time.time() - t0) < time_limit_sec:
            new_pop = pop[:min(self.elite, len(pop))]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.rng.sample(pop, 2), self.rng.sample(pop, 2)
                p1.sort(key=fitness); p2.sort(key=fitness)
                c1, c2 = self._cx_uniform_assignment(p1[0], p2[0]) if self.rng.random() < self.cx_prob else \
                         (Chromosome(p1[0].assignment[:], [o[:] for o in p1[0].intra_orders], p1[0].cluster_order[:]),
                          Chromosome(p2[0].assignment[:], [o[:] for o in p2[0].intra_orders], p2[0].cluster_order[:]))
                if self.rng.random() < self.mut_prob: self._mutate(c1)
                if self.rng.random() < self.mut_prob: self._mutate(c2)
                new_pop.extend([c1, c2])

            pop, fit_cache = new_pop[:self.pop_size], {}
            pop.sort(key=fitness)
            if fitness(pop[0]) < best_cost:
                best, best_cost, no_improve = pop[0], fitness(pop[0]), 0
            else: no_improve += 1
            if patience_gens and no_improve >= patience_gens: break
            gen += 1
            print(f"Generation {gen}: best_cost = {best_cost}", end="\r")

        self._save_final_population_details(pop, filename=f"last_generation/cluster_ga_final_pop_seed{self.seed}.csv")
        return self._decode(best)