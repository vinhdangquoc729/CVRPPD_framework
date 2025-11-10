from __future__ import annotations
import math, random
from typing import List, Dict
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution, Route
from ..core.eval import evaluate

class DFASolver(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 50, gamma: float = 0.95):
        super().__init__(problem, seed)
        self.pop_size = pop_size
        self.gamma = gamma
        random.seed(seed)

    def _cluster_customers(self) -> Dict[int, list[int]]:
        P = self.problem
        clusters: Dict[int, list[int]] = {}
        for i, nd in P.nodes.items():
            if not nd.is_depot:
                clusters.setdefault(nd.cluster, []).append(i)
        return clusters

    def _vehicles_by_depot(self) -> Dict[int, list]:
        P = self.problem
        by_dep: Dict[int, list] = {}
        for v in P.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _nearest_depot(self, nid: int) -> int:
        P = self.problem
        return min(P.depots, key=lambda d: P.d(nid, d))

    def _cluster_home_depot(self, cid: int, members: list[int]) -> int:
        """Gán cụm về depot gần cụm nhất."""
        P = self.problem
        xs = [P.nodes[i].x for i in members]
        ys = [P.nodes[i].y for i in members]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        best_d, best = None, float("inf")
        for d in P.depots:
            dx = P.nodes[d].x - cx
            dy = P.nodes[d].y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best:
                best, best_d = dist, d
        return best_d  # type: ignore

    def _shuffle_within_cluster(self, cluster_seq: list[int]) -> list[int]:
        """
        Xáo trộn nhẹ + optional đảo đoạn ngắn (shake).
        """
        if len(cluster_seq) <= 2:
            return cluster_seq[:]
        P = self.problem
        order = [cluster_seq[0]]
        remain = set(cluster_seq[1:])
        while remain:
            cur = order[-1]
            nxt = min(remain, key=lambda j: P.d(cur, j))
            order.append(nxt)
            remain.remove(nxt)
        if random.random() < 0.3:
            i = random.randint(0, len(order) - 2)
            j = random.randint(i + 1, len(order) - 1)
            order[i:j] = reversed(order[i:j])
        return order

    def _route_as_blocks(self, route: Route):
        """Tách route thành danh sách block theo cụm: [(cluster_id, [customers...]), ...]."""
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

    def _blocks_to_seq(self, depot_id: int, blocks) -> list[int]:
        seq = [depot_id]
        for _, block in blocks:
            seq.extend(block)
        seq.append(depot_id)
        return seq

    def _repair_cluster_integrity(self, sol: Solution) -> Solution:
        """Đảm bảo mỗi cụm chỉ nằm trên một route."""
        from collections import defaultdict
        P = self.problem

        # gom vị trí khách theo cụm và route
        occur = defaultdict(lambda: defaultdict(list))  # cid -> route_idx -> [customers]
        for r_idx, r in enumerate(sol.routes):
            for i in r.seq:
                if not P.nodes[i].is_depot:
                    occur[P.nodes[i].cluster][r_idx].append(i)

        # với mỗi cụm, giữ route có nhiều khách nhất; move phần còn lại sang đó
        for cid, by_route in occur.items():
            if len(by_route) <= 1:
                continue
            target = max(by_route.items(), key=lambda kv: len(kv[1]))[0]

            # xoá các khách khỏi mọi route khác
            to_move = []
            for r_idx, custs in by_route.items():
                if r_idx == target:
                    continue
                r = sol.routes[r_idx]
                r.seq = [x for x in r.seq if (P.nodes[x].is_depot or P.nodes[x].cluster != cid)]
                to_move.extend(custs)

            # chèn vào route đích thành một block cuối
            r_t = sol.routes[target]
            blocks = self._route_as_blocks(r_t)
            to_move = self._shuffle_within_cluster(to_move)
            blocks.append((cid, to_move))
            dep_t = next(v.depot_id for v in P.vehicles if v.id == r_t.vehicle_id)
            r_t.seq = self._blocks_to_seq(dep_t, blocks)

        # chuẩn hoá depot đầu/cuối
        for r in sol.routes:
            dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
            body = [x for x in r.seq if not self.problem.nodes[x].is_depot]
            r.seq = [dep] + body + [dep]
        return sol

    def _init_population(self) -> List[Solution]:
        """
        - Gom khách theo cụm.
        - Gán cụm về depot nhà (trọng tâm cụm gần depot).
        - Với mỗi depot: xáo trộn block cụm và chia round-robin theo đội xe.
        """
        P = self.problem
        clusters = self._cluster_customers()
        veh_by_dep = self._vehicles_by_depot()

        # gán mỗi cụm về 1 depot (home)
        cluster_home: Dict[int, int] = {}
        for cid, members in clusters.items():
            cluster_home[cid] = self._cluster_home_depot(cid, members)

        # nhóm cụm theo depot
        clusters_by_depot: Dict[int, list[int]] = {}
        for cid, dep in cluster_home.items():
            clusters_by_depot.setdefault(dep, []).append(cid)

        pop: List[Solution] = []
        for _ in range(self.pop_size):
            routes: List[Route] = []

            for dep, vehs in veh_by_dep.items():
                if not vehs:
                    continue
                random.shuffle(vehs)

                cids = clusters_by_depot.get(dep, [])[:]
                random.shuffle(cids)

                # tạo block per cluster (xáo trộn nội bộ cụm)
                blocks = []
                for cid in cids:
                    block = self._shuffle_within_cluster(clusters[cid][:])
                    blocks.append((cid, block))

                # round-robin phân block cho xe
                rr = [[] for _ in range(len(vehs))]
                for i, b in enumerate(blocks):
                    rr[i % len(vehs)].append(b)

                # build route.seq cho từng xe
                for k, v in enumerate(vehs):
                    seq = self._blocks_to_seq(v.depot_id, rr[k])
                    routes.append(Route(vehicle_id=v.id, seq=seq))

            sol = Solution(routes=routes)
            pop.append(self._repair_cluster_integrity(sol))
        return pop

    @staticmethod
    def _hamming_by_cluster(P: Problem, a: Solution, b: Solution) -> int:
        """Khoảng cách Hamming sau khi làm phẳng"""
        def flatten(sol: Solution):
            out = []
            for r in sol.routes:
                out.extend([i for i in r.seq if not P.nodes[i].is_depot])
            return out

        A, B = flatten(a), flatten(b)
        L = min(len(A), len(B))
        return sum(1 for i in range(L) if A[i] != B[i]) + abs(len(A) - len(B))

    def _insertion_move(self, sol: Solution) -> Solution:
        import copy
        s = copy.deepcopy(sol)
        P = self.problem

        # chọn route nguồn có ít nhất 1 block
        routes = [r for r in s.routes if len(self._route_as_blocks(r)) > 0]
        if not routes:
            return s
        r_src = random.choice(routes)
        blocks_src = self._route_as_blocks(r_src)
        if not blocks_src:
            return s

        # intra-block shuffle
        if random.random() < 0.5:
            b_idx = random.randrange(len(blocks_src))
            cid, block = blocks_src[b_idx]
            blocks_src[b_idx] = (cid, self._shuffle_within_cluster(block))
            dep_src = next(v.depot_id for v in P.vehicles if v.id == r_src.vehicle_id)
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
            return self._repair_cluster_integrity(s)

        # relocate/swap block giữa các route cùng depot
        dep_src = next(v.depot_id for v in P.vehicles if v.id == r_src.vehicle_id)
        same_dep_routes = [r for r in s.routes
                           if next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id) == dep_src]
        if not same_dep_routes:
            return s

        r_dst = random.choice(same_dep_routes)
        blocks_dst = self._route_as_blocks(r_dst)

        # lấy 1 block ở src
        b_idx = random.randrange(len(blocks_src))
        cid, block = blocks_src.pop(b_idx)

        if r_dst is r_src:
            # di chuyển vị trí block trong cùng route
            insert_pos = random.randrange(len(blocks_src) + 1)
            blocks_src.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
        else:
            # chèn vào route đích ở vị trí bất kỳ
            insert_pos = random.randrange(len(blocks_dst) + 1)
            blocks_dst.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
            dep_dst = next(v.depot_id for v in P.vehicles if v.id == r_dst.vehicle_id)
            r_dst.seq = self._blocks_to_seq(dep_dst, blocks_dst)

        return self._repair_cluster_integrity(s)

    def solve(
        self,
        time_limit_sec: float = 10000.0,
        patience_iters: int = 25,
        max_generations: int | None = 10000, 
    ) -> Solution:
        import time as _time
        P = self.problem
        pop = self._init_population()

        # Cache chi phí để tránh tính lại nhiều lần
        cost_cache: Dict[tuple, float] = {}
        
        def key_of(s: Solution):
            return tuple(tuple(r.seq) for r in s.routes)
        
        def cost_of(s: Solution) -> float:
            key = key_of(s)
            c = cost_cache.get(key)
            if c is None:
                c, _ = evaluate(P, s, return_details=False)
                cost_cache[key] = c
            return c

        # chi phí càng thấp, độ sáng càng cao
        pop.sort(key=lambda s: -cost_of(s))
        best = pop[0]
        best_cost = cost_of(best)

        gen = 0
        no_improve = 0
        t0 = _time.time()

        # vòng lặp chính
        while (_time.time() - t0) < time_limit_sec and (max_generations is None or gen < max_generations):
            new_pop: List[Solution] = []
            for i in range(len(pop)):
                xi = pop[i]
                ci = cost_of(xi)

                # bay về các cá thể sáng hơn
                for j in range(len(pop)):
                    if j == i:
                        continue
                    xj = pop[j]
                    if -cost_of(xj) > -ci:
                        rij = self._hamming_by_cluster(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** gen)))   # <-- dùng gen
                        trials: List[Solution] = []
                        for _ in range(max(2, step_max)):
                            cand = self._insertion_move(xi)
                            trials.append(cand)
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                new_pop.append(xi)

            # chọn top pop_size
            pop = sorted(new_pop, key=cost_of)[:self.pop_size]

            # cập nhật best
            cur_cost = cost_of(pop[0])
            if cur_cost < best_cost:
                best = pop[0]
                best_cost = cur_cost
                no_improve = 0
            else:
                no_improve += 1

            gen += 1

            if patience_iters and no_improve >= patience_iters:
                break

        return best
