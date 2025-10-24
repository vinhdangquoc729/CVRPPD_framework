import math, random
from typing import List
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

    def _init_population(self) -> List[Solution]:
        import random
        P = self.problem
        clusters = self._cluster_customers()
        veh_by_dep = self._vehicles_by_depot()

        # gán mỗi cụm về 1 depot "nhà"
        cluster_home = {}
        for cid, members in clusters.items():
            cluster_home[cid] = self._cluster_home_depot(cid, members)

        # nhóm cụm theo depot
        clusters_by_depot = {}
        for cid, dep in cluster_home.items():
            clusters_by_depot.setdefault(dep, []).append(cid)

        pop = []
        for _ in range(self.pop_size):
            routes = []
            # cho mỗi depot, chia cụm thành các block cho đội xe của depot đó (round-robin)
            for dep, vehs in veh_by_dep.items():
                random.shuffle(vehs)
                cids = clusters_by_depot.get(dep, [])[:]
                random.shuffle(cids)

                # tạo danh sách block theo cụm (xáo trộn nội bộ cụm)
                blocks = []
                for cid in cids:
                    block = self._shuffle_within_cluster(clusters[cid][:])
                    blocks.append((cid, block))

                # phân block cho từng xe theo round-robin
                rr = [[] for _ in range(len(vehs))]
                for i, b in enumerate(blocks):
                    rr[i % len(vehs)].append(b)

                # lắp route.seq theo depot của xe
                for k, v in enumerate(vehs):
                    seq = self._blocks_to_seq(v.depot_id, rr[k])
                    routes.append(Route(vehicle_id=v.id, seq=seq))

            sol = Solution(routes=routes)
            pop.append(self._repair_cluster_integrity(sol))
        return pop

    @staticmethod
    def _hamming_by_cluster(P: Problem, a: Solution, b: Solution) -> int:
    # flatten per cluster sequences
        def flatten(sol: Solution):
            out = []
            for r in sol.routes:
                out.extend([i for i in r.seq if not P.nodes[i].is_depot])
            return out
        A, B = flatten(a), flatten(b)
        # naive Hamming on positions (can be improved by cluster segmentation)
        L = min(len(A), len(B))
        return sum(1 for i in range(L) if A[i] != B[i]) + abs(len(A)-len(B))


    def _insertion_move(self, sol: Solution) -> Solution:
        import copy, random
        P = self.problem
        s = copy.deepcopy(sol)

        # chọn route nguồn có ít nhất 1 block
        routes = [r for r in s.routes if len(self._route_as_blocks(r)) > 0]
        if not routes:
            return s
        r_src = random.choice(routes)
        blocks_src = self._route_as_blocks(r_src)
        if not blocks_src:
            return s

        # 50% intra-block shuffle
        if random.random() < 0.5:
            b_idx = random.randrange(len(blocks_src))
            cid, block = blocks_src[b_idx]
            blocks_src[b_idx] = (cid, self._shuffle_within_cluster(block))
            r_src.seq = self._blocks_to_seq(next(v.depot_id for v in P.vehicles if v.id == r_src.vehicle_id), blocks_src)
            return self._repair_cluster_integrity(s)

        # 50% relocate/swap block giữa các route cùng depot
        dep_src = next(v.depot_id for v in P.vehicles if v.id == r_src.vehicle_id)
        # chọn route đích cùng depot
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
            insert_pos = random.randrange(len(blocks_src)+1)
            blocks_src.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
        else:
            # chèn vào route đích ở vị trí bất kỳ
            insert_pos = random.randrange(len(blocks_dst)+1)
            blocks_dst.insert(insert_pos, (cid, block))
            r_src.seq = self._blocks_to_seq(dep_src, blocks_src)
            dep_dst = next(v.depot_id for v in P.vehicles if v.id == r_dst.vehicle_id)
            r_dst.seq = self._blocks_to_seq(dep_dst, blocks_dst)

        return self._repair_cluster_integrity(s)

    
    def solve(self, time_limit_sec: float = 10000.0, patience_iters: int = 20) -> Solution:
        import time as _time

        P = self.problem
        pop = self._init_population()

        # --- cache chi phí để tránh evaluate trùng ---
        cost_cache = {}
        def cost_of(s: Solution) -> float:
            key = tuple(tuple(r.seq) for r in s.routes)
            c = cost_cache.get(key)
            if c is None:
                c, _ = evaluate(P, s, return_details=False)
                cost_cache[key] = c
            return c

        # brightness = -cost
        pop.sort(key=lambda s: -cost_of(s))
        best = pop[0]
        best_cost = cost_of(best)

        it = 0
        no_improve = 0
        EPS = 1e-9
        t0 = _time.time()

        while it < 200 and (_time.time() - t0) < time_limit_sec:
            new_pop = []
            for i in range(len(pop)):
                xi = pop[i]
                ci = cost_of(xi)

                for j in range(len(pop)):
                    if j == i:
                        continue
                    xj = pop[j]
                    if -cost_of(xj) > -ci:
                        rij = self._hamming_by_cluster(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** it)))
                        trials = []
                        for _ in range(max(2, step_max)):
                            cand = self._insertion_move(xi)
                            trials.append(cand)
                        # chọn ứng viên tốt nhất
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                new_pop.append(xi)

            # chọn top pop_size
            pop = sorted(new_pop, key=cost_of)[:self.pop_size]

            # cập nhật best + bộ đếm kiên nhẫn
            cur_cost = cost_of(pop[0])
            if cur_cost + EPS < best_cost:
                best = pop[0]
                best_cost = cur_cost
                no_improve = 0
            else:
                no_improve += 1

            it += 1

            # early stop nếu không cải thiện trong 'patience_iters' vòng ---
            if patience_iters is not None and patience_iters > 0 and no_improve >= patience_iters:
                break

            # dừng theo thời gian ---
            if (_time.time() - t0) >= time_limit_sec:
                break

        return best

    def _cluster_customers(self):
        P = self.problem
        clusters = {}
        for i, nd in P.nodes.items():
            if not nd.is_depot:
                clusters.setdefault(nd.cluster, []).append(i)
        return clusters

    def _vehicles_by_depot(self):
        P = self.problem
        by_dep = {}
        for v in P.vehicles:
            by_dep.setdefault(v.depot_id, []).append(v)
        return by_dep

    def _nearest_depot(self, nid: int):
        P = self.problem
        return min(P.depots, key=lambda d: P.d(nid, d))

    def _cluster_home_depot(self, cid: int, members: list[int]):
        # gán cụm về depot gần “trọng tâm” cụm nhất
        P = self.problem
        # dùng trung bình toạ độ của cụm
        xs = [P.nodes[i].x for i in members]; ys = [P.nodes[i].y for i in members]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        # tìm depot gần (cx,cy)
        best_d, best = None, 1e18
        for d in P.depots:
            dx = P.nodes[d].x - cx; dy = P.nodes[d].y - cy
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < best: best, best_d = dist, d
        return best_d

    def _shuffle_within_cluster(self, cluster_seq: list[int]):
        # xáo trộn nhẹ theo NN để rút đường
        import random
        if len(cluster_seq) <= 2:
            return cluster_seq[:]
        order = [cluster_seq[0]]
        remain = set(cluster_seq[1:])
        P = self.problem
        while remain:
            cur = order[-1]
            nxt = min(remain, key=lambda j: P.d(cur, j))
            order.append(nxt); remain.remove(nxt)
        # random shake
        if random.random() < 0.3:
            i = random.randint(0, len(order)-2)
            j = random.randint(i+1, len(order)-1)
            order[i:j] = reversed(order[i:j])
        return order

    def _route_as_blocks(self, route):
        P = self.problem
        blocks = []
        cur_cid, cur_block = None, []
        for k in route.seq:
            if P.nodes[k].is_depot:  # bỏ depot
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

    def _blocks_to_seq(self, depot_id: int, blocks):
        seq = [depot_id]
        for _, block in blocks:
            seq.extend(block)
        seq.append(depot_id)
        return seq

    def _repair_cluster_integrity(self, sol):
        from collections import defaultdict
        P = self.problem
        # 1) gom vị trí khách theo cụm và route
        occur = defaultdict(lambda: defaultdict(list))  # cid -> route_idx -> [customers]
        for r_idx, r in enumerate(sol.routes):
            for i in r.seq:
                if not P.nodes[i].is_depot:
                    occur[P.nodes[i].cluster][r_idx].append(i)
        # 2) với mỗi cụm, giữ route có nhiều khách nhất; move phần còn lại sang đó
        for cid, by_route in occur.items():
            if len(by_route) <= 1:
                continue
            target = max(by_route.items(), key=lambda kv: len(kv[1]))[0]
            # xoá các khách của cid khỏi mọi route khác
            to_move = []
            for r_idx, custs in by_route.items():
                if r_idx == target: 
                    continue
                r = sol.routes[r_idx]
                r.seq = [x for x in r.seq if (P.nodes[x].is_depot or P.nodes[x].cluster != cid)]
                to_move.extend(custs)
            # chèn vào route đích thành một block (cuối danh sách block)
            r_t = sol.routes[target]
            blocks = self._route_as_blocks(r_t)
            # thêm block mới; trộn nhẹ trong block
            to_move = self._shuffle_within_cluster(to_move)
            blocks.append((cid, to_move))
            r_t.seq = self._blocks_to_seq(P.vehicles[target].depot_id, blocks)
        # 3) chuẩn hoá: đảm bảo mỗi route: depot đầu/cuối
        for r in sol.routes:
            dep = next(v.depot_id for v in P.vehicles if v.id == r.vehicle_id)
            if not r.seq or r.seq[0] != dep:
                r.seq = [dep] + [x for x in r.seq if not P.nodes[x].is_depot] + [dep]
            elif r.seq[-1] != dep:
                r.seq = [x for x in r.seq if not (x != r.seq[0] and P.nodes[x].is_depot)]
                r.seq.append(dep)
        return sol
