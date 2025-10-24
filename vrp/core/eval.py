from typing import Tuple, Dict, List, Set
from .problem import Problem, Node
from .solution import Solution

def evaluate(problem: Problem, sol: Solution, return_details=False) -> Tuple[float, dict]:
    BIG = 1e6      # phạt cứng cho vi phạm luật cụm / cung cấm
    BIG_CAP = 1e5  # phạt tải / overtime
    speed = getattr(problem, "speed_units_per_min", 50.0)

    cost = 0.0
    details = {
        "distance": 0.0,
        "fixed": 0.0,
        "tw_penalty": 0.0,
        "cap_violations": 0,
        "overtime_routes": 0,
        "prohibited_uses": 0,
        "cluster_refills": 0,
        "cluster_violation": 0,            # need > Q
        "cluster_revisit_violations": 0,   # vào–ra–vào lại trong cùng route
        "cluster_split_violations": 0,     # 1 cụm xuất hiện ở >1 route
        "cluster_incomplete_block": 0,     # route vào cụm nhưng block không chứa đủ toàn bộ khách của cụm
        "unserved_customers": 0
    }

    nodes = problem.nodes
    # --------- (A) Chuẩn bị tập khách theo cụm ----------
    cluster_to_all_customers: Dict[int, Set[int]] = {}
    for nid, nd in nodes.items():
        if not nd.is_depot:
            cluster_to_all_customers.setdefault(nd.cluster, set()).add(nid)

    # --------- (B) Ánh xạ cụm -> các route đang phục vụ ----------
    cluster_to_routes: Dict[int, List[int]] = {}
    served_customers_global: Set[int] = set()

    for r_idx, route in enumerate(sol.routes):
        for nid in route.seq:
            nd = nodes[nid]
            if nd.is_depot: 
                continue
            served_customers_global.add(nid)
            cluster_to_routes.setdefault(nd.cluster, []).append(r_idx)

    # 1) Không chia cụm: nếu 1 cụm xuất hiện ở >1 route -> vi phạm
    for c, routes_list in cluster_to_routes.items():
        if len(set(routes_list)) > 1:
            details["cluster_split_violations"] += 1
            cost += BIG

    # 2) Tất cả khách phải được phục vụ
    all_customers = set().union(*cluster_to_all_customers.values()) if cluster_to_all_customers else set()
    unserved = all_customers - served_customers_global
    if unserved:
        details["unserved_customers"] = len(unserved)
        # Tuỳ sách lược: phạt lớn để coi nghiệm không hợp lệ
        cost += BIG

    # --------- (C) Bắt đầu chấm chi phí, với SPD + refill FULL + contiguity ----------
    def add_leg(u: int, v: int, var_cost: float) -> float:
        nonlocal cost
        if (u, v) in problem.prohibited:
            details["prohibited_uses"] += 1
            cost += BIG
        dist = problem.d(u, v)
        details["distance"] += dist
        cost += dist * var_cost
        return dist  # 1 đơn vị = 1 phút (đổi nếu có speed)

    for route in sol.routes:
        veh = next(v for v in problem.vehicles if v.id == route.vehicle_id)
        Q = veh.capacity
        depot_id = veh.depot_id

        # Chi phí cố định
        details["fixed"] += veh.fixed_cost
        cost += veh.fixed_cost

        time = veh.start_time
        load_deliv = Q     # xuất phát ở depot -> FULL
        load_pick  = 0

        # Chuẩn hoá depot đầu/cuối
        seq = list(route.seq)
        if not seq or seq[0] != depot_id: seq = [depot_id] + seq
        if seq[-1] != depot_id: seq = seq + [depot_id]

        last_cluster = None
        closed_clusters: Set[int] = set()

        i = 0
        while i < len(seq) - 1:
            u, v = seq[i], seq[i+1]
            node_v = nodes[v]
            refill_done = False

            if not node_v.is_depot:
                cur_cluster = node_v.cluster

                # 3) Contiguity (revisit trong cùng route)
                if cur_cluster in closed_clusters:
                    details["cluster_revisit_violations"] += 1
                    cost += BIG

                first_of_cluster = (cur_cluster != last_cluster)
                if first_of_cluster:
                    # Đóng cụm trước nếu vừa chuyển cụm
                    if last_cluster is not None and last_cluster != cur_cluster:
                        closed_clusters.add(last_cluster)

                    # --- Block của cụm này trong route: phải chứa đủ toàn bộ khách của cụm ---
                    full_set = cluster_to_all_customers.get(cur_cluster, set())
                    block_set: Set[int] = set()
                    j = i + 1
                    while j < len(seq):
                        w = seq[j]
                        if nodes[w].is_depot or nodes[w].cluster != cur_cluster:
                            break
                        block_set.add(w); j += 1

                    if block_set != full_set:
                        # Route đã vào cụm nhưng block không chứa đủ tất cả khách của cụm (phần còn lại nằm route khác hoặc mất)
                        details["cluster_incomplete_block"] += 1
                        cost += BIG

                    # --- Nhu cầu giao của block ---
                    need = sum(nodes[k].demand_delivery for k in block_set)
                    if need > Q:
                        details["cluster_violation"] += 1
                        cost += BIG

                    # Nếu thiếu delivery và đang không ở depot -> quay depot refill FULL
                    if load_deliv < min(need, Q) and u != depot_id:
                        time += add_leg(u, depot_id, veh.var_cost_per_dist) / speed   # u->depot
                        load_pick = 0
                        load_deliv = Q
                        time += add_leg(depot_id, v, veh.var_cost_per_dist) / speed  # depot->v
                        details["cluster_refills"] += 1
                        refill_done = True

                last_cluster = cur_cluster
            else:
                # đến depot: bỏ hết & refill FULL
                load_pick = 0
                load_deliv = Q
                if last_cluster is not None:
                    closed_clusters.add(last_cluster)
                last_cluster = None

            # đi cạnh gốc nếu chưa vừa refill u->depot->v
            if not refill_done:
                time += add_leg(u, v, veh.var_cost_per_dist) / speed

            # xử lý tại node v
            if node_v.is_depot:
                pass
            else:
                # SPD: giao trước, nhặt sau
                load_deliv -= node_v.demand_delivery
                if load_deliv < 0:
                    details["cap_violations"] += 1
                    cost += BIG_CAP
                    load_deliv = 0
                load_pick += node_v.demand_pickup
                if load_deliv + load_pick > Q:
                    details["cap_violations"] += 1
                    cost += BIG_CAP
                    overflow = (load_deliv + load_pick) - Q
                    load_pick = max(0, load_pick - overflow)

                # TW: chờ miễn phí, phạt muộn
                if (node_v.tw_open is not None) and (time < node_v.tw_open):
                    time = node_v.tw_open
                if (node_v.tw_close is not None) and (time > node_v.tw_close):
                    late = time - node_v.tw_close
                    details["tw_penalty"] += late * problem.tw_penalty_per_min
                    cost += late * problem.tw_penalty_per_min

                time += node_v.service_time

            i += 1

        if time > veh.end_time:
            details["overtime_routes"] += 1
            cost += BIG_CAP

    return (cost, details) if return_details else (cost, {})
