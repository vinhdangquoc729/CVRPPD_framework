from __future__ import annotations
import math, random, time
from typing import List, Tuple, Dict
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate

class ESASolver(Solver):
    def __init__(self, problem: Problem, seed: int = 42,
                 mu: int = 20,                    # cỡ quần thể
                 elite_frac: float = 0.3,         # giữ elite
                 alpha: float = 0.95,             # cooling rate
                 trials_per_iter: int = 8,        # số láng giềng cho mỗi cá thể/iter
                 patience_iters: int = 50,        # early stop nếu không cải thiện
                 ):
        super().__init__(problem, seed)
        self.rng = random.Random(seed)
        self.mu = mu
        self.elite_frac = elite_frac
        self.alpha = alpha
        self.trials_per_iter = trials_per_iter
        self.patience_iters = patience_iters

    # các helper
    def _cluster_customers(self) -> Dict[int, List[int]]:
        P = self.problem
        clusters = {}
        for i, nd in P.nodes.items():
            if not nd.is_depot:
                clusters.setdefault(nd.cluster, []).append(i)
        return clusters

    def _vehicles_by_depot(self) -> Dict[int, List]:
        by_dep: Dict[int, List] = {}
        for v in self.problem.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _cluster_home_depot(self, members: List[int]) -> int:
        P = self.problem
        xs = [P.nodes[i].x for i in members]; ys = [P.nodes[i].y for i in members]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        best_d, best = None, 1e18
        for d in P.depots:
            dx = P.nodes[d].x - cx; dy = P.nodes[d].y - cy
            dist2 = dx*dx + dy*dy
            if dist2 < best: best, best_d = dist2, d
        return best_d

    def _route_as_blocks(self, route: Route):
        P = self.problem
        blocks = []
        cur_cid, cur_block = None, []
        for k in route.seq:
            if P.nodes[k].is_depot:
                continue
            cid = P.nodes[k].cluster
            if cur_cid is None or cid != cur_cid:
                if cur_block:
                    blocks.append((cur_cid, cur_block))
                cur_cid, cur_block = cid, [k]
            else:
                cur_block.append(k)
        if cur_block:
            blocks.append((cur_cid, cur_block))
        return blocks

    def _blocks_to_seq(self, depot_id: int, blocks) -> List[int]:
        seq = [depot_id]
        for _, block in blocks:
            seq.extend(block)
        seq.append(depot_id)
        return seq

    def _shuffle_within_cluster(self, block: List[int]) -> List[int]:
        P = self.problem
        r = self.rng
        if len(block) <= 2:
            out = block[:]
            r.shuffle(out)
            return out
        order = [block[0]]
        remain = set(block[1:])
        while remain:
            cur = order[-1]
            nxt = min(remain, key=lambda j: P.d(cur, j))
            order.append(nxt); remain.remove(nxt)
        if r.random() < 0.5:
            i = r.randint(0, len(order)-2)
            j = r.randint(i+1, len(order)-1)
            order[i:j] = reversed(order[i:j])
        return order

    # Khởi tạo quần thể
    def _init_population(self) -> List[Solution]:
        P = self.problem
        r = self.rng
        clusters = self._cluster_customers()
        veh_by_dep = self._vehicles_by_depot()

        # gán mỗi cụm về depot
        cluster_home = {cid: self._cluster_home_depot(members) for cid, members in clusters.items()}

        # nhóm cụm theo depot
        clusters_by_depot = {}
        for cid, dep in cluster_home.items():
            clusters_by_depot.setdefault(dep, []).append(cid)

        pop: List[Solution] = []
        for _ in range(self.mu):
            routes: List[Route] = []
            for dep, vehs in veh_by_dep.items():
                cids = clusters_by_depot.get(dep, [])[:]
                r.shuffle(cids)
                blocks = []
                for cid in cids:
                    custs = clusters[cid][:]
                    r.shuffle(custs)
                    blocks.append((cid, custs))
                if not vehs:
                    continue

                buckets = [[] for _ in range(len(vehs))]
                for i, b in enumerate(blocks):
                    buckets[i % len(vehs)].append(b)

                for k, v in enumerate(vehs):
                    if not buckets[k]:
                        continue
                    seq = self._blocks_to_seq(v.depot_id, buckets[k])
                    routes.append(Route(vehicle_id=v.id, seq=seq))
            pop.append(Solution(routes=routes))
        return pop

    def _neighbor(self, sol: Solution) -> Solution:
        import copy
        P = self.problem
        r = self.rng
        s = copy.deepcopy(sol)

        # chọn route có block
        routes = [rt for rt in s.routes if len(self._route_as_blocks(rt)) > 0]
        if not routes:
            return s
        r_src = r.choice(routes)
        dep_src = next(v.depot_id for v in P.vehicles if v.id == r_src.vehicle_id)
        blocks_src = self._route_as_blocks(r_src)

        # shuffle trong block
        if r.random() < 0.5:
            b_idx = r.randrange(len(blocks_src))
            cid, block = blocks_src[b_idx]
            blocks_src[b_idx] = (cid, self._shuffle_within_cluster(block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
            return s

        # relocate/swap giữa routes cùng depot
        same_dep_routes = [rt for rt in s.routes
                           if next(v.depot_id for v in P.vehicles if v.id == rt.vehicle_id) == dep_src]
        r_dst = r.choice(same_dep_routes)
        blocks_dst = self._route_as_blocks(r_dst)

        # pop 1 block
        b_idx = r.randrange(len(blocks_src))
        cid, block = blocks_src.pop(b_idx)

        if r_dst is r_src:
            # di chuyển vị trí block trong cùng route
            insert_pos = r.randrange(len(blocks_src)+1)
            blocks_src.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
        else:
            # chèn sang route đích
            insert_pos = r.randrange(len(blocks_dst)+1)
            blocks_dst.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
            dep_dst = next(v.depot_id for v in P.vehicles if v.id == r_dst.vehicle_id)
            r_dst.seq = self._blocks_to_seq(dep_dst, blocks_dst)

        return s

    def solve(self, time_limit_sec: float = 30.0) -> Solution:
        P = self.problem
        r = self.rng
        t0 = time.time()

        pop = self._init_population()

        # cache chi phí
        cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}
        def key_of(s: Solution):
            return tuple(tuple(rt.seq) for rt in s.routes)
        def cost_of(s: Solution) -> float:
            k = key_of(s)
            v = cost_cache.get(k)
            if v is None:
                v, _ = evaluate(P, s, return_details=False)
                cost_cache[k] = v
            return v

        pop.sort(key=cost_of)
        best = pop[0]; best_cost = cost_of(best)

        # đặt nhiệt độ
        costs0 = [cost_of(s) for s in pop]
        spread = max(costs0) - min(costs0) if len(costs0) > 1 else max(1.0, abs(costs0[0]))
        T = spread / max(1.0, math.log(1 / 0.95))

        patience = 0
        it = 0

        while time.time() - t0 < time_limit_sec:
            it += 1
            new_pop: List[Solution] = []

            # elitism
            elite_k = max(1, int(self.elite_frac * self.mu))
            elites = pop[:elite_k]
            new_pop.extend(elites)

            # sinh ứng viên từ tournament nhỏ
            while len(new_pop) < self.mu:
                # chọn bố mẹ
                if len(pop) >= 2:
                    a, b = r.sample(pop, 2)
                    parent = a if cost_of(a) <= cost_of(b) else b
                else:
                    parent = pop[0]
                child = parent
                # nhiều bước láng giềng
                for _ in range(self.trials_per_iter):
                    cand = self._neighbor(child)
                    dE = cost_of(cand) - cost_of(child)
                    if dE <= 0 or r.random() < math.exp(-dE / max(1e-6, T)):
                        child = cand
                new_pop.append(child)

            pop = sorted(new_pop, key=cost_of)
            cur_best = pop[0]; cur_cost = cost_of(cur_best)

            if cur_cost < best_cost:
                best, best_cost = cur_best, cur_cost
                patience = 0
            else:
                patience += 1
                if patience >= self.patience_iters:
                    break

            # làm nguội
            T *= self.alpha
            if T < 1e-6:
                T = 1e-6

        return best
