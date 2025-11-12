#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vrp_pd_only.py — VRP datasets (Osaba-like) with PD exclusivity & paired PD for expansions.

- Base (single depot/vehicle) giữ nguyên số liệu gốc (không paired).
- BẢN MỞ RỘNG base_mdmv_tw_modified (PAIRED) **không sinh toạ độ mới**:
  dùng toạ độ gốc của file XML, depot cũng chọn từ các điểm gốc.
  Bảo đảm số khách chẵn để ghép pickup–delivery.
- Synthetic (PAIRED) có thể sinh mới như trước.

Usage:
  python generate_vrp_pd_only.py --in Osaba_50_1_1.xml --out out_dir --expand-base
  python generate_vrp_pd_only.py --in Osaba_50_1_1.xml --out out_dir --expand-base \
      --synth-sizes 200 500 1000 --del-mult-from-original 3.0 --pd-target-ratio 0.95 --lock-both-totals
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib

import numpy as np
import pandas as pd


# ------------------------ Args ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate VRP datasets (Osaba-like) with PD constraints.")
    p.add_argument("--in", dest="in_xml", required=True, help="Input Osaba XML path OR folder containing *.xml")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory")

    # Legacy PD-only (XOR) params (cho synthesize_instance_pd nếu cần)
    p.add_argument("--pd-target-ratio", type=float, default=0.95,
                   help="Target ratio: sum(delivery_new) ≈ ratio * sum(pickup_new). Default 0.95")
    p.add_argument("--pd-pickup-share", type=float, default=0.55,
                   help="Approx share of pickup-only customers initially. Default 0.55")
    p.add_argument("--pd-min-amount", type=int, default=1,
                   help="Minimum nonzero demand after scaling/rounding. Default 1")

    p.add_argument("--del-mult-from-original", type=float, default=2.0,
                   help="sum(delivery_new) = k * sum(delivery_original) (customers only). Default 2.0")

    p.add_argument("--lock-both-totals", action="store_true", default=False,
                   help="If set, scale quantities so sum(del_new)=k*sum(del_orig); pickup synced so totals match.")

    p.add_argument("--expand-base", action="store_true",
                   help="Create multi-depot, multi-vehicle, time-window expansion for the base XML")

    # Depots/vehicles/TW
    p.add_argument("--depots-extra-min", type=int, default=2)
    p.add_argument("--depots-extra-max", type=int, default=3)
    p.add_argument("--depots-min", type=int, default=3)
    p.add_argument("--depots-max", type=int, default=4)

    p.add_argument("--veh-per-depot-min", type=int, default=3)
    p.add_argument("--veh-per-depot-max", type=int, default=4)
    p.add_argument("--veh-capacities", type=int, nargs="+", default=[80, 100, 120])
    p.add_argument("--veh-start", type=int, default=7*60)
    p.add_argument("--veh-end", type=int, default=20*60)
    p.add_argument("--veh-fixed-cost", type=float, default=100.0,
                   help="[IGNORED] Fixed cost is automatically computed as 100 * capacity.")
    p.add_argument("--veh-var-cost", type=float, default=1.0)

    p.add_argument("--service-min", type=int, default=5)
    p.add_argument("--service-max", type=int, default=15)
    p.add_argument("--tw-open-min", type=int, default=8*60)
    p.add_argument("--tw-open-max", type=int, default=12*60)
    p.add_argument("--tw-close-min", type=int, default=13*60)
    p.add_argument("--tw-close-max", type=int, default=19*60)
    p.add_argument("--tw-min-width", type=int, default=60)
    p.add_argument("--tw-penalty-per-min", type=float, default=10.0)
    p.add_argument("--no-tw", action="store_true")

    # Synthetic controls
    p.add_argument("--synth-sizes", type=int, nargs="*", default=[])
    p.add_argument("--synth-clusters-min", type=int, default=20)
    p.add_argument("--synth-clusters-max", type=int, default=50)
    p.add_argument("--synth-area-jitter", type=float, default=1200.0)

    p.add_argument("--keep-prohibited-arcs", action="store_true",
                   help="If set, copy original prohibited arcs into expanded/synthetic instances.")

    # Quantity distributions
    p.add_argument("--dem-delivery-candidates", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--dem-delivery-probs", type=float, nargs="+", default=[0.4, 0.4, 0.15, 0.05])
    p.add_argument("--dem-pickup-candidates", type=int, nargs="+", default=[3, 5, 10, 15])
    p.add_argument("--dem-pickup-probs", type=float, nargs="+", default=[0.3, 0.4, 0.2, 0.1])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------ Helpers ------------------------
def assign_time_windows(n_customers, open_range, close_range, min_width):
    """Gán time window cho các điểm khách."""
    opens = np.random.randint(open_range[0], open_range[1] + 1, size=n_customers)
    closes = np.random.randint(close_range[0], close_range[1] + 1, size=n_customers)
    open_final = np.minimum(opens, closes - min_width)
    close_final = np.maximum(closes, open_final + min_width)
    open_final = np.clip(open_final, 0, 1440 - 1)
    close_final = np.clip(close_final, 0, 1440 - 1)
    return open_final, close_final


def choose_depots(coords, k):
    """Chọn k điểm làm depot dựa trên farthest-first."""
    N = coords.shape[0]
    first = np.random.randint(0, N)
    chosen = [first]
    for _ in range(1, k):
        dists = np.stack([np.linalg.norm(coords - coords[c], axis=1) for c in chosen], axis=1)
        d_near = np.min(dists, axis=1)
        nxt = int(np.argmax(d_near))
        chosen.append(nxt)
    return chosen


def gen_vehicle_table(depot_ids, per_depot_min, per_depot_max, cap_choices,
                      start_min, end_max, var_cost, multiple_fixed_cost=100.0):
    """fixed_cost = 100 * capacity."""
    rows, vid = [], 0
    for d in depot_ids:
        nveh = np.random.randint(per_depot_min, per_depot_max + 1)
        for _ in range(nveh):
            cap = int(np.random.choice(cap_choices))
            rows.append({
                "vehicle_id": vid,
                "depot_id": d,
                "capacity": cap,
                "start_time": start_min,
                "end_time": end_max,
                "fixed_cost": float(multiple_fixed_cost * cap),
                "variable_cost_per_distance": var_cost
            })
            vid += 1
    return pd.DataFrame(rows)


def with_service_and_tw(df, service_range, tw, tw_open_range, tw_close_range, tw_min_width):
    out = df.copy()
    if "is_depot" not in out.columns:
        out["is_depot"] = False
    out["service_time"] = 0
    cust_mask = ~out["is_depot"]
    out.loc[cust_mask, "service_time"] = np.random.randint(service_range[0], service_range[1] + 1, cust_mask.sum())
    if tw:
        open_min, close_min = assign_time_windows(cust_mask.sum(), tw_open_range, tw_close_range, tw_min_width)
        out.loc[cust_mask, "tw_open"] = open_min
        out.loc[cust_mask, "tw_close"] = close_min
    else:
        out["tw_open"] = np.nan
        out["tw_close"] = np.nan
    return out


def write_instance(folder, nodes_df, vehicles_df, prohib_df, meta, orders_df=None):
    folder.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(folder / "nodes.csv", index=False)
    vehicles_df.to_csv(folder / "vehicles.csv", index=False)
    (prohib_df if prohib_df is not None else pd.DataFrame(columns=["from_id", "to_id"])) \
        .to_csv(folder / "prohibited_arcs.csv", index=False)
    if orders_df is not None:
        orders_df.to_csv(folder / "orders.csv", index=False)
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _sample_order_sizes(rng, n_orders, candidates, probs, min_amount):
    q = rng.choice(candidates, size=n_orders, p=probs).astype(int)
    q[q < min_amount] = min_amount
    return q


def _build_pd_nodes_shuffled(rng, order_sizes, cluster_centers, area_jitter, start_node_id):
    """
    (Dùng cho synthetic) Tạo 2 node (P,D) cho mỗi đơn, rồi TRỘN trước khi gán id.
    """
    K = len(cluster_centers)
    tmp_nodes = []
    for oid, qty in enumerate(order_sizes):
        cidx = int(rng.randint(0, K))
        cx, cy = cluster_centers[cidx]
        # pickup
        px = float(rng.normal(loc=cx, scale=area_jitter))
        py = float(rng.normal(loc=cy, scale=area_jitter))
        tmp_nodes.append({
            "tmp_oid": oid, "role": "P",
            "addr": f"P.{oid}", "cluster": int(cidx) + 1,
            "demand_delivery": 0, "demand_pickup": int(qty),
            "x": px, "y": py, "is_depot": False,
        })
        # delivery (75% cùng cụm)
        didx = cidx if (rng.rand() < 0.75) else int(rng.randint(0, K))
        dx0, dy0 = cluster_centers[didx]
        dx = float(rng.normal(loc=dx0, scale=area_jitter))
        dy = float(rng.normal(loc=dy0, scale=area_jitter))
        tmp_nodes.append({
            "tmp_oid": oid, "role": "D",
            "addr": f"D.{oid}", "cluster": int(didx) + 1,
            "demand_delivery": int(qty), "demand_pickup": 0,
            "x": dx, "y": dy, "is_depot": False,
        })

    # shuffle & assign ids
    perm = rng.permutation(len(tmp_nodes))
    tmp_nodes = [tmp_nodes[i] for i in perm]

    rows, map_id, nid = [], {}, start_node_id
    for nd in tmp_nodes:
        nd_out = nd.copy()
        nd_out["id"] = nid
        rows.append(nd_out)
        map_id[(nd["tmp_oid"], nd["role"])] = nid
        nid += 1

    orders = []
    for oid, qty in enumerate(order_sizes):
        orders.append({
            "order_id": oid,
            "pickup_node_id": int(map_id[(oid, "P")]),
            "delivery_node_id": int(map_id[(oid, "D")]),
            "quantity": int(qty),
        })
    orders_df = pd.DataFrame(orders)
    return rows, orders_df, nid


def _ensure_pd_order_tw(nodes_df: pd.DataFrame, orders_df: pd.DataFrame,
                        travel_speed_units_per_min: float, tw_min_width: int) -> pd.DataFrame:
    """
    Nắn TW: tw_open(delivery) >= tw_open(pickup) + service_pickup + travel_min.
    Nếu cần, nới tw_close(delivery) để đảm bảo width.
    """
    df = nodes_df.set_index("id").copy()
    for _, r in orders_df.iterrows():
        pid, did = int(r["pickup_node_id"]), int(r["delivery_node_id"])
        px, py = float(df.at[pid, "x"]), float(df.at[pid, "y"])
        dx, dy = float(df.at[did, "x"]), float(df.at[did, "y"])
        dist = float(np.hypot(dx - px, dy - py))
        travel_min = max(1, int(dist / max(1e-9, travel_speed_units_per_min)))

        p_open = int(df.at[pid, "tw_open"]) if not np.isnan(df.at[pid, "tw_open"]) else 0
        p_serv = int(df.at[pid, "service_time"])
        d_open = int(df.at[did, "tw_open"]) if not np.isnan(df.at[did, "tw_open"]) else 0
        d_close = int(df.at[did, "tw_close"]) if not np.isnan(df.at[did, "tw_close"]) else 1439

        new_d_open = max(d_open, p_open + p_serv + travel_min)
        if new_d_open >= d_close:
            df.at[did, "tw_close"] = new_d_open + tw_min_width
        df.at[did, "tw_open"] = new_d_open
    return df.reset_index()


# ------------------------ Osaba loader ------------------------
def load_osaba_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes = []
    for node in root.findall("NodeP"):
        nodes.append({
            "id": int(node.findtext("id")),
            "addr": node.findtext("Addr"),
            "cluster": int(node.findtext("Cluster")),
            "demand_delivery": int(node.findtext("DemEnt")),
            "demand_pickup": int(node.findtext("DemRec")),
            "x": float(node.findtext("CoordX")),
            "y": float(node.findtext("CoordY")),
        })
    nodes_df = pd.DataFrame(nodes).sort_values("id").reset_index(drop=True)

    prohib = []
    for pr in root.findall("Prohibido"):
        a = int(pr.findtext("est1")); b = int(pr.findtext("est2"))
        prohib.append({"from_id": a, "to_id": b})
    prohib_df = pd.DataFrame(prohib)

    return nodes_df, prohib_df


# ------------------------ Legacy PD-only (XOR) synthesizer (giữ để dùng khi cần) ------------------------
def _apply_pd_exclusive_and_ratio_on_customers(
    df: pd.DataFrame,
    *, rng: np.random.RandomState,
    pickup_share: float,
    target_ratio: float,
    min_amount: int,
    deliv_candidates=None, deliv_probs=None,
    pick_candidates=None, pick_probs=None
) -> pd.DataFrame:
    out = df.copy()
    cust = out[~out["is_depot"]].copy()
    n = len(cust)
    if n == 0:
        return out

    roles = rng.rand(n) < pickup_share  # True: pickup-only
    cust["is_pickup_only"] = roles

    if (deliv_candidates is not None) and (deliv_probs is not None) and \
       (pick_candidates is not None) and (pick_probs is not None):
        cust["demand_delivery"] = 0
        cust["demand_pickup"] = 0
        dmask = ~cust["is_pickup_only"]
        if dmask.any():
            cust.loc[dmask, "demand_delivery"] = rng.choice(deliv_candidates, size=dmask.sum(), p=deliv_probs)
        pmask = cust["is_pickup_only"]
        if pmask.any():
            cust.loc[pmask, "demand_pickup"] = rng.choice(pick_candidates, size=pmask.sum(), p=pick_probs)
    else:
        pmask = roles; dmask = ~roles
        cust.loc[pmask, "demand_delivery"] = 0
        cust.loc[dmask, "demand_pickup"] = 0
        zero_both = (cust["demand_delivery"] <= 0) & (cust["demand_pickup"] <= 0)
        if zero_both.any():
            flip = rng.rand(zero_both.sum()) < pickup_share
            pick_idx = zero_both[zero_both].index[flip]
            del_idx  = zero_both[zero_both].index[~flip]
            cust.loc[pick_idx, "demand_pickup"] = max(min_amount, 1)
            cust.loc[del_idx,  "demand_delivery"] = max(min_amount, 1)

    sum_pick = float(cust["demand_pickup"].sum())
    sum_del  = float(cust["demand_delivery"].sum())
    if sum_del <= 0 and (~cust["is_pickup_only"]).any():
        cust.loc[~cust["is_pickup_only"], "demand_delivery"] = min_amount
        sum_del = float(cust["demand_delivery"].sum())
    if sum_pick <= 0 and (cust["is_pickup_only"]).any():
        cust.loc[cust["is_pickup_only"], "demand_pickup"] = min_amount
        sum_pick = float(cust["demand_pickup"].sum())

    if sum_del > 0 and sum_pick > 0:
        scale = (target_ratio * sum_pick) / sum_del
        del_vals = (cust["demand_delivery"].astype(float) * scale).round().astype(int)
        dmask = (~cust["is_pickup_only"]) & (del_vals <= 0)
        del_vals.loc[dmask] = min_amount
        cust["demand_delivery"] = del_vals

    cust = cust.drop(columns=["is_pickup_only"])
    out.loc[~out["is_depot"], ["demand_delivery", "demand_pickup"]] = cust[["demand_delivery", "demand_pickup"]].values
    return out


def _enforce_total_pickup(full_nodes: pd.DataFrame, pickup_target: float, min_amount: int) -> pd.DataFrame:
    out = full_nodes.copy()
    cust = out[~out["is_depot"]].copy()
    cur = float(cust["demand_pickup"].sum())
    if pickup_target is None or pickup_target <= 0 or cur <= 0:
        return out
    s = pickup_target / cur
    new_vals = (cust["demand_pickup"].astype(float) * s).round().astype(int)
    must_pos = (cust["demand_pickup"] > 0) & (new_vals <= 0)
    if must_pos.any():
        new_vals.loc[must_pos] = min_amount
    cust["demand_pickup"] = new_vals

    diff = int(round(pickup_target)) - int(cust["demand_pickup"].sum())
    if diff != 0:
        pos_idx = cust.index[cust["demand_pickup"] > 0].tolist()
        if pos_idx:
            step = 1 if diff > 0 else -1
            k = 0
            while diff != 0 and k < 100000:
                j = pos_idx[k % len(pos_idx)]
                if step < 0 and cust.at[j, "demand_pickup"] <= min_amount:
                    k += 1; continue
                cust.at[j, "demand_pickup"] += step
                diff -= step; k += 1

    out.loc[cust.index, "demand_pickup"] = cust["demand_pickup"]
    return out


def _enforce_total_delivery(full_nodes: pd.DataFrame, target_total: float, min_amount: int) -> pd.DataFrame:
    out = full_nodes.copy()
    cust = out[~out["is_depot"]].copy()
    if target_total is None or target_total <= 0:
        return out
    cur_sum = float(cust["demand_delivery"].sum())
    if cur_sum <= 0:
        m = max(1, int(0.1 * len(cust)))
        idx = cust.sample(n=m, random_state=0).index
        cust.loc[idx, "demand_delivery"] = min_amount
        cur_sum = float(cust["demand_delivery"].sum())

    s = target_total / cur_sum
    new_vals = (cust["demand_delivery"].astype(float) * s).round().astype(int)
    must_pos = (cust["demand_delivery"] > 0) & (new_vals <= 0)
    if must_pos.any():
        new_vals.loc[must_pos] = min_amount
    cust["demand_delivery"] = new_vals

    diff = int(round(target_total)) - int(cust["demand_delivery"].sum())
    if diff != 0:
        pos_idx = cust.index[cust["demand_delivery"] > 0].tolist()
        if pos_idx:
            step = 1 if diff > 0 else -1
            k = 0
            while diff != 0 and k < 100000:
                j = pos_idx[k % len(pos_idx)]
                if step < 0 and cust.at[j, "demand_delivery"] <= min_amount:
                    k += 1; continue
                cust.at[j, "demand_delivery"] += step
                diff -= step; k += 1

    out.loc[cust.index, "demand_delivery"] = cust["demand_delivery"]
    return out


def synthesize_instance_pd(
    nodes_df_base: pd.DataFrame,
    n_customers_target: int,
    n_clusters_min: int, n_clusters_max: int,
    area_jitter: float,
    depots_min: int, depots_max: int,
    service_range,
    tw_enabled: bool, tw_open_range, tw_close_range, tw_min_width: int,
    cap_choices,
    veh_start: int, veh_end: int, veh_var_cost: float,
    veh_per_depot_min: int, veh_per_depot_max: int,
    dem_deliv_candidates, dem_deliv_probs,
    dem_pick_candidates, dem_pick_probs,
    tw_penalty_per_min: float,
    pd_target_ratio: float,
    pd_pickup_share: float,
    pd_min_amount: int,
    rng: np.random.RandomState
):
    # Synthetic PD-only (XOR) — giữ nguyên như trước
    base = nodes_df_base[nodes_df_base["id"] != 0].copy()
    centers = base.groupby("cluster")[["x", "y"]].mean().reset_index()
    base_centers = centers[["x", "y"]].to_numpy()

    n_clusters = int(np.clip(np.random.randint(n_clusters_min, n_clusters_max + 1), n_clusters_min, n_clusters_max))
    chosen = rng.choice(len(base_centers), size=n_clusters, replace=True)
    cluster_centers = base_centers[chosen]

    proportions = rng.dirichlet(np.ones(n_clusters))
    cust_per_cluster = np.maximum((proportions * n_customers_target).astype(int), 1)
    diff = n_customers_target - cust_per_cluster.sum()
    if diff > 0: cust_per_cluster[:diff] += 1
    elif diff < 0: cust_per_cluster[:(-diff)] -= 1

    xs, ys, clusters_out, cid = [], [], [], 1
    for c_idx, cnt in enumerate(cust_per_cluster):
        cx, cy = cluster_centers[c_idx]
        x = rng.normal(loc=cx, scale=area_jitter, size=cnt)
        y = rng.normal(loc=cy, scale=area_jitter, size=cnt)
        xs.append(x); ys.append(y)
        clusters_out.extend([cid] * cnt); cid += 1
    xs = np.concatenate(xs); ys = np.concatenate(ys)

    coords = np.stack([xs, ys], axis=1)
    k_depots = int(np.random.randint(depots_min, depots_max + 1))
    depot_indices = choose_depots(coords, k_depots)

    depot_rows = []
    for i, idx in enumerate(depot_indices):
        depot_rows.append({
            "id": i, "addr": f"DEPOT.{i}", "cluster": 0,
            "demand_delivery": 0, "demand_pickup": 0,
            "x": float(coords[idx, 0]), "y": float(coords[idx, 1]),
            "is_depot": True
        })

    start_id = len(depot_rows)
    customers = [{
        "id": start_id + i, "addr": f"C.{i+1}",
        "cluster": int(clusters_out[i]),
        "demand_delivery": 0, "demand_pickup": 0,
        "x": float(xs[i]), "y": float(ys[i]), "is_depot": False
    } for i in range(n_customers_target)]
    full_nodes = pd.DataFrame(depot_rows + customers)

    full_nodes = with_service_and_tw(
        full_nodes, service_range=service_range, tw=tw_enabled,
        tw_open_range=tw_open_range, tw_close_range=tw_close_range, tw_min_width=tw_min_width
    )

    full_nodes = _apply_pd_exclusive_and_ratio_on_customers(
        full_nodes, rng=rng,
        pickup_share=pd_pickup_share, target_ratio=pd_target_ratio, min_amount=pd_min_amount,
        deliv_candidates=np.asarray(dem_deliv_candidates, dtype=int),
        deliv_probs=np.asarray(dem_deliv_probs, dtype=float),
        pick_candidates=np.asarray(dem_pickup_candidates, dtype=int),
        pick_probs=np.asarray(dem_pickup_probs, dtype=float),
    )

    depot_ids = full_nodes[full_nodes["is_depot"]]["id"].tolist()
    veh_df = gen_vehicle_table(
        depot_ids=depot_ids, per_depot_min=veh_per_depot_min, per_depot_max=veh_per_depot_max,
        cap_choices=cap_choices, start_min=veh_start, end_max=veh_end,
        var_cost=veh_var_cost
    )

    prohib_syn = pd.DataFrame(columns=["from_id", "to_id"])
    meta = {
        "source": "synthetic_from_Osaba_like_PD_only",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_customers": int(n_customers_target),
        "n_depots": int(len(depot_ids)),
        "n_vehicles": int(len(veh_df)),
        "type": "multi_depot_multi_vehicle_with_time_windows" if tw_enabled else "multi_depot_multi_vehicle",
        "time_window_units": "minutes_from_midnight",
        "tw_penalty_per_min": float(tw_penalty_per_min),
        "service_time_units": "minutes",
        "travel_speed_units_per_min": 60.0,
        "distance_units": "euclidean (coordinate units)",
        "clusters": int(len(np.unique(full_nodes.loc[~full_nodes['is_depot'], 'cluster']))),
        "pd_only": True,
        "pd_target_ratio": float(pd_target_ratio)
    }
    return full_nodes, veh_df, prohib_syn, meta


# ------------------------ Paired PD synthesizer (EXPANSION from ORIGINAL NODES) ------------------------
def _pick_k_depots_parity(coords: np.ndarray, k_min: int, k_max: int, n_customers_total: int, rng: np.random.RandomState) -> int:
    """
    Chọn số depot k in [k_min, k_max] sao cho (n_customers_total - k) là số chẵn.
    Ưu tiên k gần giữa dải, nếu không có thì fallback về k_min/k_max.
    """
    # ứng viên theo độ gần trung bình
    candidates = list(range(k_min, k_max + 1))
    # sắp xếp theo |k - mid|
    mid = 0.5 * (k_min + k_max)
    candidates.sort(key=lambda k: abs(k - mid))
    for k in candidates:
        if (n_customers_total - k) >= 2 and ((n_customers_total - k) % 2 == 0):
            return k
    # nếu không có k thoả, trả về ứng viên gần nhất rồi sau đó sẽ drop 1 khách (rất hiếm)
    return int(round(mid))


def synthesize_instance_pd_paired_from_original_nodes(
    nodes_df_base: pd.DataFrame,
    depots_min: int, depots_max: int,
    service_range,
    tw_enabled: bool, tw_open_range, tw_close_range, tw_min_width: int,
    cap_choices,
    veh_start: int, veh_end: int, veh_var_cost: float,
    veh_per_depot_min: int, veh_per_depot_max: int,
    order_qty_candidates, order_qty_probs,
    tw_penalty_per_min: float,
    rng: np.random.RandomState,
    travel_speed_units_per_min: float = 60.0,
):
    """
    Tạo instance PAIR từ các điểm GỐC:
    - Chọn depot từ các điểm gốc (id != 0) bằng farthest-first.
    - KHÁCH = phần còn lại sau khi lấy depot. Bảo đảm số khách chẵn (ưu tiên chỉnh k; nếu vẫn lẻ thì bỏ 1 khách).
    - Ghép 2*M khách thành M đơn: 1 pickup, 1 delivery (quantity bốc từ phân phối).
    - Không sinh toạ độ mới.
    """
    base = nodes_df_base[nodes_df_base["id"] != 0].copy().reset_index(drop=True)
    all_coords = base[["x", "y"]].to_numpy()
    n_total = len(base)
    if n_total < 2:
        raise ValueError("Not enough original customers to create paired PD instance.")

    # chọn k_depots sao cho số khách còn lại chẵn
    k_depots = _pick_k_depots_parity(all_coords, depots_min, depots_max, n_total, rng)

    # chọn vị trí depot (index trong base) bằng farthest-first
    depot_indices_in_base = choose_depots(all_coords, k_depots)
    depot_rows = []
    for i, idx in enumerate(depot_indices_in_base):
        row = base.iloc[idx]
        depot_rows.append({
            "id": i,
            "addr": f"DEPOT.{i}",
            "cluster": 0,
            "demand_delivery": 0,
            "demand_pickup": 0,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "is_depot": True
        })

    # khách = các điểm gốc còn lại
    mask = np.ones(n_total, dtype=bool)
    mask[depot_indices_in_base] = False
    customers_df = base.loc[mask].copy().reset_index(drop=True)

    # nếu số khách lẻ (trong trường hợp bất khả kháng), bỏ 1 khách cuối
    if len(customers_df) % 2 == 1:
        customers_df = customers_df.iloc[:-1, :].reset_index(drop=True)

    n_customers = len(customers_df)
    n_orders = n_customers // 2
    if n_orders == 0:
        raise ValueError("Paired construction failed: not enough customers after depot selection.")

    # shuffle khách và chia thành cặp (P,D)
    perm = rng.permutation(n_customers)
    customers_df = customers_df.iloc[perm].reset_index(drop=True)

    # lượng hàng per-order
    order_sizes = _sample_order_sizes(rng, n_orders, order_qty_candidates, order_qty_probs, min_amount=1)

    # ---------- REPLACE THIS WHOLE BLOCK ----------

    # lượng hàng per-order
    order_sizes = _sample_order_sizes(rng, n_orders, order_qty_candidates, order_qty_probs, min_amount=1)

    # TẠO DANH SÁCH NODE TẠM (2*n), mỗi đơn -> 1 P + 1 D, TOÀN BỘ DÙNG TỌA ĐỘ GỐC
    pd_tmp_nodes = []
    for oid in range(n_orders):
        rP = customers_df.iloc[2 * oid + 0]   # khách thứ 1 trong cặp
        rD = customers_df.iloc[2 * oid + 1]   # khách thứ 2 trong cặp
        qty = int(order_sizes[oid])

        # (tuỳ chọn) đảo vai P/D ngẫu nhiên để tránh bias
        if rng.rand() < 0.5:
            rP, rD = rD, rP

        pd_tmp_nodes.append({
            "tmp_oid": oid, "role": "P",
            "addr": f"P.{oid}",
            "cluster": int(rP.get("cluster", 0)),
            "demand_delivery": 0,
            "demand_pickup": qty,
            "x": float(rP["x"]), "y": float(rP["y"]),
            "is_depot": False,
        })
        pd_tmp_nodes.append({
            "tmp_oid": oid, "role": "D",
            "addr": f"D.{oid}",
            "cluster": int(rD.get("cluster", 0)),
            "demand_delivery": qty,
            "demand_pickup": 0,
            "x": float(rD["x"]), "y": float(rD["y"]),
            "is_depot": False,
        })

    # *** SHUFFLE TOÀN BỘ 2*n NODE TRƯỚC KHI GÁN ID ***
    rng.shuffle(pd_tmp_nodes)

    # GÁN ID: depots trước, rồi các node P/D đã shuffle
    next_id = len(depot_rows)
    rows, id_map = [], {}
    for nd in pd_tmp_nodes:
        nd_out = nd.copy()
        nd_out["id"] = next_id
        rows.append(nd_out)
        id_map[(nd["tmp_oid"], nd["role"])] = next_id
        next_id += 1

    # orders.csv: map lại theo id_map
    orders = []
    for oid in range(n_orders):
        orders.append({
            "order_id": oid,
            "pickup_node_id": int(id_map[(oid, "P")]),
            "delivery_node_id": int(id_map[(oid, "D")]),
            "quantity": int(order_sizes[oid]),
        })
    orders_df = pd.DataFrame(orders)

    # nodes để ghi ra: depots + toàn bộ rows (đã có demand_* đúng & ĐÃ SHUFFLE)
    full_nodes = pd.DataFrame(depot_rows + rows)

    # nodes & TW/service
    full_nodes = with_service_and_tw(
        full_nodes,
        service_range=service_range,
        tw=tw_enabled,
        tw_open_range=tw_open_range,
        tw_close_range=tw_close_range,
        tw_min_width=tw_min_width
    )
    if tw_enabled:
        full_nodes = _ensure_pd_order_tw(
            full_nodes, orders_df,
            travel_speed_units_per_min=travel_speed_units_per_min,
            tw_min_width=tw_min_width
        )

    # vehicles
    depot_ids = full_nodes[full_nodes["is_depot"]]["id"].tolist()
    veh_df = gen_vehicle_table(
        depot_ids=depot_ids,
        per_depot_min=veh_per_depot_min, per_depot_max=veh_per_depot_max,
        cap_choices=cap_choices, start_min=veh_start, end_max=veh_end,
        var_cost=veh_var_cost
    )

    prohib_syn = pd.DataFrame(columns=["from_id", "to_id"])  # caller có thể ghi đè bằng gốc
    meta = {
        "source": "expanded_from_original_nodes_PD_pairs",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_orders": int(n_orders),
        "n_nodes": int(len(full_nodes)),
        "n_customers": int(len(full_nodes) - len(depot_ids)),
        "n_depots": int(len(depot_ids)),
        "n_vehicles": int(len(veh_df)),
        "type": "multi_depot_multi_vehicle_with_time_windows" if tw_enabled else "multi_depot_multi_vehicle",
        "time_window_units": "minutes_from_midnight",
        "tw_penalty_per_min": float(tw_penalty_per_min),
        "service_time_units": "minutes",
        "travel_speed_units_per_min": float(travel_speed_units_per_min),
        "distance_units": "euclidean (coordinate units)",
        "pd_paired": True,
        "coords_from_original": True
    }
    return full_nodes, veh_df, prohib_syn, meta, orders_df


# ------------------------ Paired PD synthesizer (Synthetic) ------------------------
def synthesize_instance_pd_paired(
    nodes_df_base: pd.DataFrame,
    n_orders_target: int,
    n_clusters_min: int, n_clusters_max: int,
    area_jitter: float,
    depots_min: int, depots_max: int,
    service_range,
    tw_enabled: bool, tw_open_range, tw_close_range, tw_min_width: int,
    cap_choices,
    veh_start: int, veh_end: int, veh_var_cost: float,
    veh_per_depot_min: int, veh_per_depot_max: int,
    order_qty_candidates, order_qty_probs,
    tw_penalty_per_min: float,
    rng: np.random.RandomState,
    travel_speed_units_per_min: float = 60.0,
):
    """
    (Synthetic) Sinh n_orders_target đơn; mỗi đơn -> 2 node (P,D), có thể sinh mới toạ độ.
    """
    base = nodes_df_base[nodes_df_base["id"] != 0].copy()
    centers = base.groupby("cluster")[["x", "y"]].mean().reset_index()
    base_centers = centers[["x", "y"]].to_numpy()

    # Cụm khách
    n_clusters = int(np.clip(rng.randint(n_clusters_min, n_clusters_max + 1), n_clusters_min, n_clusters_max))
    chosen = rng.choice(len(base_centers), size=n_clusters, replace=True)
    cluster_centers = base_centers[chosen]

    # Depots từ gốc (không bắt buộc parity trong synthetic)
    all_coords = base[["x", "y"]].to_numpy()
    k_depots = int(rng.randint(depots_min, depots_max + 1))
    depot_indices = choose_depots(all_coords, k_depots)
    depot_rows = []
    for i, idx in enumerate(depot_indices):
        depot_rows.append({
            "id": i, "addr": f"DEPOT.{i}", "cluster": 0,
            "demand_delivery": 0, "demand_pickup": 0,
            "x": float(all_coords[idx, 0]), "y": float(all_coords[idx, 1]),
            "is_depot": True
        })

    # Orders & nodes (sinh mới quanh cluster)
    order_sizes = _sample_order_sizes(rng, n_orders_target, order_qty_candidates, order_qty_probs, min_amount=1)
    pd_rows, orders_df, next_id = _build_pd_nodes_shuffled(
        rng=rng,
        order_sizes=order_sizes,
        cluster_centers=cluster_centers,
        area_jitter=area_jitter,
        start_node_id=len(depot_rows),
    )
    full_nodes = pd.DataFrame(depot_rows + pd_rows)

    # Service & TW
    full_nodes = with_service_and_tw(
        full_nodes,
        service_range=service_range,
        tw=tw_enabled,
        tw_open_range=tw_open_range,
        tw_close_range=tw_close_range,
        tw_min_width=tw_min_width
    )
    if tw_enabled:
        full_nodes = _ensure_pd_order_tw(
            full_nodes, orders_df,
            travel_speed_units_per_min=travel_speed_units_per_min,
            tw_min_width=tw_min_width
        )

    # Vehicles
    depot_ids = full_nodes[full_nodes["is_depot"]]["id"].tolist()
    veh_df = gen_vehicle_table(
        depot_ids=depot_ids,
        per_depot_min=veh_per_depot_min, per_depot_max=veh_per_depot_max,
        cap_choices=cap_choices, start_min=veh_start, end_max=veh_end,
        var_cost=veh_var_cost
    )

    prohib_syn = pd.DataFrame(columns=["from_id", "to_id"])
    meta = {
        "source": "synthetic_from_Osaba_like_PD_pairs",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_orders": int(n_orders_target),
        "n_nodes": int(len(full_nodes)),
        "n_customers": int(len(full_nodes) - len(depot_ids)),
        "n_depots": int(len(depot_ids)),
        "n_vehicles": int(len(veh_df)),
        "type": "multi_depot_multi_vehicle_with_time_windows" if tw_enabled else "multi_depot_multi_vehicle",
        "time_window_units": "minutes_from_midnight",
        "tw_penalty_per_min": float(tw_penalty_per_min),
        "service_time_units": "minutes",
        "travel_speed_units_per_min": float(travel_speed_units_per_min),
        "distance_units": "euclidean (coordinate units)",
        "clusters": int(n_clusters),
        "pd_paired": True
    }
    return full_nodes, veh_df, prohib_syn, meta, orders_df


# ------------------------ Base normalization ------------------------
def _standardize_base_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    out = nodes_df.copy()
    out["is_depot"] = out["id"] == 0
    out["service_time"] = 0
    out["tw_open"] = 0
    out["tw_close"] = 1439
    return out


def _default_base_vehicle_df(var_cost: float) -> pd.DataFrame:
    cap = 240
    return pd.DataFrame([{
        "vehicle_id": 0, "depot_id": 0, "capacity": cap,
        "start_time": 0, "end_time": 1439, "fixed_cost": float(100 * cap),
        "variable_cost_per_distance": var_cost
    }])


# ------------------------ Pipeline ------------------------
def process_one(xml_path: Path, out_root: Path, args):
    h = hashlib.sha256(xml_path.stem.encode("utf-8")).hexdigest()
    offset = int(h[:8], 16) % 10_000_000
    rng = np.random.RandomState(args.seed + offset)
    np.random.seed(args.seed + offset)

    this_out = out_root / xml_path.stem
    this_out.mkdir(parents=True, exist_ok=True)

    nodes_df, prohib_df = load_osaba_xml(xml_path)
    if prohib_df is None or len(prohib_df) == 0:
        prohib_df = pd.DataFrame(columns=["from_id", "to_id"])

    # Totals gốc (customers only)
    orig_delivery_total = float(nodes_df.loc[nodes_df["id"] != 0, "demand_delivery"].sum())
    k = float(args.del_mult_from_original)
    ratio = float(args.pd_target_ratio)
    target_delivery_total = k * orig_delivery_total
    target_pickup_total = (target_delivery_total / ratio) if ratio > 0 else None

    # --- base_modified (schema unify only)
    base_dir = this_out / "base_modified"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_nodes_std = _standardize_base_nodes(nodes_df)
    base_vehicles = _default_base_vehicle_df(var_cost=float(args.veh_var_cost))
    base_meta = {
        "source": xml_path.name, "created_at": datetime.utcnow().isoformat() + "Z",
        "type": "single_depot_single_vehicle_with_time_windows",
        "time_window_units": "minutes_from_midnight", "tw_penalty_per_min": float(args.tw_penalty_per_min),
        "travel_speed_units_per_min": 60.0, "service_time_units": "minutes",
        "distance_units": "euclidean (coordinate units)", "n_depots": 1, "n_vehicles": 1,
        "note": "Base mirrors original amounts; PD-paired applies to expanded/synthetic sets."
    }
    write_instance(base_dir, base_nodes_std, base_vehicles, prohib_df, base_meta, orders_df=None)

    # --- base_mdmv_tw_modified (PAIRED, NO NEW COORDS)
    if args.expand_base:
        md_nodes, vehicles_df, prohib_used, meta, orders_df = synthesize_instance_pd_paired_from_original_nodes(
            nodes_df_base=nodes_df,
            depots_min=args.depots_min, depots_max=args.depots_max,
            service_range=(args.service_min, args.service_max),
            tw_enabled=(not args.no_tw),
            tw_open_range=(args.tw_open_min, args.tw_open_max),
            tw_close_range=(args.tw_close_min, args.tw_close_max),
            tw_min_width=args.tw_min_width,
            cap_choices=args.veh_capacities,
            veh_start=args.veh_start, veh_end=args.veh_end, veh_var_cost=args.veh_var_cost,
            veh_per_depot_min=args.veh_per_depot_min, veh_per_depot_max=args.veh_per_depot_max,
            order_qty_candidates=np.asarray(args.dem_delivery_candidates, dtype=int),  # dùng chung candidates cho size
            order_qty_probs=np.asarray(args.dem_delivery_probs, dtype=float),
            tw_penalty_per_min=args.tw_penalty_per_min,
            rng=np.random.RandomState(args.seed + offset + 1234),
            travel_speed_units_per_min=60.0,
        )

        if args.lock_both_totals:
            cur_total = float(md_nodes.loc[~md_nodes["is_depot"], "demand_delivery"].sum())
            if cur_total > 0 and target_delivery_total > 0:
                s = target_delivery_total / cur_total
                for col in ["demand_delivery", "demand_pickup"]:
                    vals = md_nodes[col].astype(float)
                    pos = (vals > 0)
                    vals.loc[pos] = np.maximum(np.round(vals.loc[pos] * s).astype(int), 1)
                    md_nodes[col] = vals.astype(int)
                # sync quantity trong orders_df
                id2qty = md_nodes.set_index("id")
                orders_df["quantity"] = [
                    int(max(id2qty.at[r["pickup_node_id"], "demand_pickup"],
                            id2qty.at[r["delivery_node_id"], "demand_delivery"]))
                    for _, r in orders_df.iterrows()
                ]

        base_exp_dir = this_out / "base_mdmv_tw_modified"
        meta.update({
            "source": xml_path.name,
            "original_total_delivery": float(nodes_df.loc[nodes_df["id"] != 0, "demand_delivery"].sum()),
            "target_delivery_total": target_delivery_total,
            "target_pickup_total": target_pickup_total,
            "delivery_mult_from_original": k,
            "lock_both_totals": bool(args.lock_both_totals),
            "coords_from_original": True
        })
        # GIỮ nguyên prohibited arcs gốc cho bản mở rộng (như yêu cầu trước)
        write_instance(base_exp_dir, md_nodes, vehicles_df, prohib_df, meta, orders_df=orders_df)

    # --- synthetic_*_mdmv_tw (PAIRED, có thể sinh mới)
    for n in args.synth_sizes:
        n_orders = max(1, int(n // 2))
        ndf, vdf, pdf_empty, meta_syn, orders_df = synthesize_instance_pd_paired(
            nodes_df_base=nodes_df, n_orders_target=int(n_orders),
            n_clusters_min=args.synth_clusters_min, n_clusters_max=args.synth_clusters_max,
            area_jitter=float(args.synth_area_jitter),
            depots_min=args.depots_min, depots_max=args.depots_max,
            service_range=(args.service_min, args.service_max),
            tw_enabled=(not args.no_tw),
            tw_open_range=(args.tw_open_min, args.tw_open_max),
            tw_close_range=(args.tw_close_min, args.tw_close_max),
            tw_min_width=args.tw_min_width,
            cap_choices=args.veh_capacities,
            veh_start=args.veh_start, veh_end=args.veh_end, veh_var_cost=args.veh_var_cost,
            veh_per_depot_min=args.veh_per_depot_min, veh_per_depot_max=args.veh_per_depot_max,
            order_qty_candidates=np.asarray(args.dem_delivery_candidates, dtype=int),
            order_qty_probs=np.asarray(args.dem_delivery_probs, dtype=float),
            tw_penalty_per_min=args.tw_penalty_per_min,
            rng=np.random.RandomState(args.seed + offset + int(n)),
            travel_speed_units_per_min=60.0,
        )

        if args.lock_both_totals:
            cur_total = float(ndf.loc[~ndf["is_depot"], "demand_delivery"].sum())
            if cur_total > 0 and target_delivery_total > 0:
                s = target_delivery_total / cur_total
                for col in ["demand_delivery", "demand_pickup"]:
                    vals = ndf[col].astype(float)
                    pos = (vals > 0)
                    vals.loc[pos] = np.maximum(np.round(vals.loc[pos] * s).astype(int), 1)
                    ndf[col] = vals.astype(int)
                id2qty = ndf.set_index("id")
                orders_df["quantity"] = [
                    int(max(id2qty.at[r["pickup_node_id"], "demand_pickup"],
                            id2qty.at[r["delivery_node_id"], "demand_delivery"]))
                    for _, r in orders_df.iterrows()
                ]

        meta_syn.update({
            "original_total_delivery": float(nodes_df.loc[nodes_df["id"] != 0, "demand_delivery"].sum()),
            "target_delivery_total": target_delivery_total,
            "target_pickup_total": target_pickup_total,
            "delivery_mult_from_original": k,
            "lock_both_totals": bool(args.lock_both_totals),
        })

        inst_dir = this_out / f"synthetic_{n}_mdmv_tw"
        prohib_to_write = prohib_df if args.keep_prohibited_arcs else pd.DataFrame(columns=["from_id", "to_id"])
        write_instance(inst_dir, ndf, vdf, prohib_to_write, meta_syn, orders_df=orders_df)

    # README
    readme = this_out / "README.md"
    readme.write_text(f"""# Osaba VRP — Normalized & Expanded for `{xml_path.name}` ({datetime.utcnow().isoformat()}Z)

- Base: giữ nguyên schema, không paired.
- Expanded (base_mdmv_tw_modified): **paired PD**, KHÔNG sinh toạ độ mới (dùng toạ độ gốc).
  Depot cũng lấy từ các điểm gốc; số khách được đảm bảo **chẵn** để ghép P↔D.
- Synthetic: **paired PD** (có thể sinh toạ độ mới), có `orders.csv`.
- Nếu --lock-both-totals: scale quantity toàn cục để đạt target_delivery_total (đồng bộ lại orders.csv).
- Vehicle fixed_cost = 100 × capacity (bỏ qua --veh-fixed-cost).
""", encoding="utf-8")

    print(f"[OK] Processed: {xml_path} -> {this_out}")


def main(args):
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.in_xml)
    if in_path.is_dir():
        targets = sorted(in_path.glob("*.xml"))
        if not targets:
            raise FileNotFoundError(f"No *.xml found in folder: {in_path}")
    else:
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")
        targets = [in_path]

    for xml in targets:
        process_one(xml, out_root, args)

    print("DONE. Output at:", str(out_root))


if __name__ == "__main__":
    args = parse_args()
    main(args)
