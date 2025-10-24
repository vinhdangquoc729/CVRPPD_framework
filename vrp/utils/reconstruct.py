from typing import List
from ..core.problem import Problem
from ..core.solution import Route, Solution

def reconstruct_with_refills(problem: Problem, sol: Solution) -> Solution:
    nodes = problem.nodes
    out_routes = []
    for r in sol.routes:
        veh = next(v for v in problem.vehicles if v.id == r.vehicle_id)
        depot = veh.depot_id
        seq = list(r.seq)
        if not seq or seq[0] != depot: seq = [depot] + seq
        if seq[-1] != depot: seq = seq + [depot]

        load_deliv = veh.capacity
        new_seq = [seq[0]]
        i = 0
        last_cluster = None
        while i < len(seq) - 1:
            u, v = new_seq[-1], seq[i+1]
            nv = nodes[v]
            # nếu bước vào cụm mới, tính NEED của block liên tiếp
            if not nv.is_depot:
                cur_cluster = nv.cluster
                first = (cur_cluster != last_cluster)
                if first:
                    need = 0; j = i+1
                    block = []
                    while j < len(seq):
                        w = seq[j]
                        if nodes[w].is_depot or nodes[w].cluster != cur_cluster: break
                        need += nodes[w].demand_delivery; block.append(w); j += 1
                    need = min(need, veh.capacity)
                    # nếu thiếu hàng và u != depot -> chèn depot
                    if load_deliv < need and u != depot:
                        new_seq.append(depot)      # u->depot
                        load_deliv = veh.capacity  # refill FULL
                        new_seq.append(v)          # depot->v
                        # đừng thêm cạnh gốc nữa
                        # cập nhật tải khi phục vụ v
                        load_deliv -= nv.demand_delivery
                        load_deliv = max(load_deliv, 0)
                        i += 1
                        last_cluster = cur_cluster
                        continue
                last_cluster = cur_cluster
            else:
                load_deliv = veh.capacity
                last_cluster = None

            # đi cạnh gốc u->v
            new_seq.append(v)
            # xử lý tải nếu là khách
            if not nv.is_depot:
                load_deliv -= nv.demand_delivery
                load_deliv = max(load_deliv, 0)

            i += 1

        out_routes.append(Route(vehicle_id=r.vehicle_id, seq=new_seq))
    return Solution(routes=out_routes)
