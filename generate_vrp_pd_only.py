#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vrp_pd_only.py — VRP datasets (Osaba-like) with PD exclusivity:
- mỗi khách chỉ pickup hoặc delivery (XOR)
- Khóa đồng thời 2 điều kiện tổng (nếu --lock-both-totals):
    (1) sum(delivery_new) = k * sum(delivery_original)   (k = --del-mult-from-original)
    (2) sum(delivery_new) = pd_target_ratio * sum(pickup_new)
  => scale PICKUP trước về sum_pickup_target = k * sum(del_orig) / ratio, rồi set DELIVERY về k * sum(del_orig)
- fixed_cost của xe luôn = 100 * capacity (tham số --veh-fixed-cost sẽ bị bỏ qua).

Usage:
  python generate_vrp_pd_only.py --in Osaba_50_1_1.xml --out out_dir --expand-base
  python generate_vrp_pd_only.py --in Osaba_50_1_1.xml --out out_dir --expand-base `
      --synth-sizes 200 500 1000 --del-mult-from-original 3.0 --pd-target-ratio 0.95 --lock-both-totals
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib

import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Generate VRP datasets (Osaba-like) with PD-only constraints (pickup XOR delivery).")
    p.add_argument("--in", dest="in_xml", required=True, help="Input Osaba XML path OR folder containing *.xml")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory")

    p.add_argument("--pd-target-ratio", type=float, default=0.95,
                   help="Target ratio: sum(delivery_new) ≈ ratio * sum(pickup_new). Default 0.95")
    p.add_argument("--pd-pickup-share", type=float, default=0.55,
                   help="Approx share of pickup-only customers initially. Default 0.55")
    p.add_argument("--pd-min-amount", type=int, default=1,
                   help="Minimum nonzero demand after scaling/rounding. Default 1")

    p.add_argument("--del-mult-from-original", type=float, default=2.0,
                   help="sum(delivery_new) = k * sum(delivery_original) (customers only). Default 2.0")

    p.add_argument("--lock-both-totals", action="store_true", default=True,
                   help="If set, enforce BOTH: sum(del_new)=k*sum(del_orig) and sum(del_new)=ratio*sum(pick_new) by scaling pickup first.")

    p.add_argument("--expand-base", action="store_true", help="Create multi-depot, multi-vehicle, time-window expansion for the base XML")

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

    p.add_argument("--synth-sizes", type=int, nargs="*", default=[])
    p.add_argument("--synth-clusters-min", type=int, default=20)
    p.add_argument("--synth-clusters-max", type=int, default=50)
    p.add_argument("--synth-area-jitter", type=float, default=1200.0)
    p.add_argument("--keep-prohibited-arcs", action="store_true")

    p.add_argument("--dem-delivery-candidates", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--dem-delivery-probs", type=float, nargs="+", default=[0.4, 0.4, 0.15, 0.05])
    p.add_argument("--dem-pickup-candidates", type=int, nargs="+", default=[3, 5, 10, 15])
    p.add_argument("--dem-pickup-probs", type=float, nargs="+", default=[0.3, 0.4, 0.2, 0.1])

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return args


def assign_time_windows(n_customers, open_range, close_range, min_width):
    """Gán time window cho các điểm khách"""
    opens = np.random.randint(open_range[0], open_range[1] + 1, size=n_customers)
    closes = np.random.randint(close_range[0], close_range[1] + 1, size=n_customers)
    open_final = np.minimum(opens, closes - min_width)
    close_final = np.maximum(closes, open_final + min_width)
    open_final = np.clip(open_final, 0, 1440 - 1)
    close_final = np.clip(close_final, 0, 1440 - 1)
    return open_final, close_final


def choose_depots(coords, k):
    """Chọn depot xa nhất"""
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
    """fixed_cost được tính = 100 * capacity."""
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


def write_instance(folder, nodes_df, vehicles_df, prohib_df, meta):
    folder.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(folder / "nodes.csv", index=False)
    vehicles_df.to_csv(folder / "vehicles.csv", index=False)
    (prohib_df if prohib_df is not None else pd.DataFrame(columns=["from_id", "to_id"])) \
        .to_csv(folder / "prohibited_arcs.csv", index=False)
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


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

def _apply_pd_exclusive_and_ratio_on_customers(
    df: pd.DataFrame,
    *, rng: np.random.RandomState,
    pickup_share: float,
    target_ratio: float,
    min_amount: int,
    deliv_candidates=None, deliv_probs=None,
    pick_candidates=None, pick_probs=None
) -> pd.DataFrame:
    """
    Ép mỗi khách chỉ pickup hoặc delivery và scale delivery để đạt ratio.
    """
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
    """Scale pickup-only để sum(pickup) = pickup_target."""
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

    # chỉnh sai số làm tròn
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
    """Scale delivery-only để sum(delivery) = target_total."""
    out = full_nodes.copy()
    cust = out[~out["is_depot"]].copy()

    if target_total is None or target_total <= 0:
        return out

    cur_sum = float(cust["demand_delivery"].sum())
    if cur_sum <= 0:
        # nếu không có delivery, gán min cho 10% khách để có base
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


def _standardize_base_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    out = nodes_df.copy()
    out["is_depot"] = out["id"] == 0
    out["service_time"] = 0
    out["tw_open"] = 0
    out["tw_close"] = 1439
    return out


def _default_base_vehicle_df(var_cost: float) -> pd.DataFrame:
    # capacity mặc định 240 -> fixed_cost = 100 * 240 = 24000
    cap = 240
    return pd.DataFrame([{
        "vehicle_id": 0, "depot_id": 0, "capacity": cap,
        "start_time": 0, "end_time": 1439, "fixed_cost": float(100 * cap),
        "variable_cost_per_distance": var_cost
    }])


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

    # original totals (customers only)
    orig_delivery_total = float(nodes_df.loc[nodes_df["id"] != 0, "demand_delivery"].sum())
    k = float(args.del_mult_from_original)
    ratio = float(args.pd_target_ratio)
    target_delivery_total = k * orig_delivery_total
    target_pickup_total = (target_delivery_total / ratio) if ratio > 0 else None

    # --- base_modified (schema unify only)
    base_dir = this_out / f"base_modified"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_nodes_std = _standardize_base_nodes(nodes_df)
    base_vehicles = _default_base_vehicle_df(var_cost=float(args.veh_var_cost))
    base_meta = {
        "source": xml_path.name, "created_at": datetime.utcnow().isoformat() + "Z",
        "type": "single_depot_single_vehicle_with_time_windows",
        "time_window_units": "minutes_from_midnight", "tw_penalty_per_min": float(args.tw_penalty_per_min),
        "travel_speed_units_per_min": 60.0, "service_time_units": "minutes",
        "distance_units": "euclidean (coordinate units)", "n_depots": 1, "n_vehicles": 1,
        "note": "Base mirrors original amounts; PD-only applies to expanded/synthetic sets."
    }
    write_instance(base_dir, base_nodes_std, base_vehicles, prohib_df, base_meta)

    if args.expand_base:
        base_customers = nodes_df[nodes_df["id"] != 0].copy()
        coords = base_customers[["x", "y"]].to_numpy()
        k_extra = int(np.random.randint(args.depots_extra_min, args.depots_extra_max + 1))
        extra_idxs = choose_depots(coords, k_extra)
        extra_depot_ids = base_customers.iloc[extra_idxs]["id"].tolist()
        multi_depots = [0] + extra_depot_ids

        md_nodes = nodes_df.copy()
        md_nodes["is_depot"] = md_nodes["id"].isin(multi_depots)
        md_nodes.loc[md_nodes["is_depot"], ["demand_delivery", "demand_pickup"]] = 0

        md_nodes = with_service_and_tw(
            md_nodes,
            service_range=(args.service_min, args.service_max),
            tw=(not args.no_tw),
            tw_open_range=(args.tw_open_min, args.tw_open_max),
            tw_close_range=(args.tw_close_min, args.tw_close_max),
            tw_min_width=args.tw_min_width,
        )

        md_nodes = _apply_pd_exclusive_and_ratio_on_customers(
            md_nodes, rng=rng,
            pickup_share=float(args.pd_pickup_share),
            target_ratio=float(args.pd_target_ratio),
            min_amount=int(args.pd_min_amount),
            deliv_candidates=None, deliv_probs=None,
            pick_candidates=None, pick_probs=None
        )

        if args.lock_both_totals and target_pickup_total is not None:
            md_nodes = _enforce_total_pickup(md_nodes, pickup_target=target_pickup_total, min_amount=int(args.pd_min_amount))
            md_nodes = _enforce_total_delivery(md_nodes, target_total=target_delivery_total, min_amount=int(args.pd_min_amount))
        else:
            # legacy: ratio first then absolute delivery
            md_nodes = _enforce_total_delivery(md_nodes, target_total=target_delivery_total, min_amount=int(args.pd_min_amount))

        vehicles_df = gen_vehicle_table(
            depot_ids=multi_depots,
            per_depot_min=args.veh_per_depot_min, per_depot_max=args.veh_per_depot_max,
            cap_choices=args.veh_capacities, start_min=args.veh_start, end_max=args.veh_end,
            var_cost=args.veh_var_cost,
        )

        base_exp_dir = this_out / "base_mdmv_tw_modified"
        meta = {
            "source": xml_path.name, "created_at": datetime.utcnow().isoformat() + "Z",
            "type": "multi_depot_multi_vehicle_with_time_windows" if not args.no_tw else "multi_depot_multi_vehicle",
            "time_window_units": "minutes_from_midnight", "tw_penalty_per_min": float(args.tw_penalty_per_min),
            "travel_speed_units_per_min": 60.0, "service_time_units": "minutes",
            "distance_units": "euclidean (coordinate units)", "n_depots": len(multi_depots),
            "n_vehicles": int(len(vehicles_df)), "pd_only": True, "pd_target_ratio": ratio,
            "original_total_delivery": orig_delivery_total, "target_delivery_total": target_delivery_total,
            "target_pickup_total": target_pickup_total, "delivery_mult_from_original": k,
            "lock_both_totals": bool(args.lock_both_totals)
        }
        # >>> LUÔN ghi giữ nguyên prohibited arcs gốc cho base_mdmv_tw_modified <<<
        write_instance(base_exp_dir, md_nodes, vehicles_df, prohib_df, meta)

    for n in args.synth_sizes:
        inst_dir = this_out / f"synthetic_{n}_mdmv_tw"
        ndf, vdf, pdf, meta = synthesize_instance_pd(
            nodes_df_base=nodes_df, n_customers_target=int(n),
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
            dem_deliv_candidates=args.dem_delivery_candidates, dem_deliv_probs=args.dem_delivery_probs,
            dem_pick_candidates=args.dem_pickup_candidates, dem_pick_probs=args.dem_pickup_probs,
            tw_penalty_per_min=args.tw_penalty_per_min,
            pd_target_ratio=float(args.pd_target_ratio),
            pd_pickup_share=float(args.pd_pickup_share),
            pd_min_amount=int(args.pd_min_amount),
            rng=np.random.RandomState(args.seed + offset + int(n))
        )

        if args.lock_both_totals and target_pickup_total is not None:
            ndf = _enforce_total_pickup(ndf, pickup_target=target_pickup_total, min_amount=int(args.pd_min_amount))
            ndf = _enforce_total_delivery(ndf, target_total=target_delivery_total, min_amount=int(args.pd_min_amount))
        else:
            ndf = _enforce_total_delivery(ndf, target_total=target_delivery_total, min_amount=int(args.pd_min_amount))

        meta.update({
            "original_total_delivery": orig_delivery_total,
            "target_delivery_total": target_delivery_total,
            "target_pickup_total": target_pickup_total,
            "delivery_mult_from_original": k,
            "lock_both_totals": bool(args.lock_both_totals),
        })
        # synthetic: vẫn giữ hành vi cũ — nếu không muốn, có thể chuyển sang dùng prohib_df gốc.
        write_instance(inst_dir, ndf, vdf, pdf, meta)

    # README
    readme = this_out / "README.md"
    readme.write_text(f"""# Osaba VRP (PD-only) — Normalized & Expanded for `{xml_path.name}` ({datetime.utcnow().isoformat()}Z)

- PD-only: mỗi khách chỉ P hoặc D.
- Nếu --lock-both-totals: 
    sum(del_new) = {k:.2f} × sum(del_orig) và sum(del_new) = {ratio:.2f} × sum(pick_new)
- Otherwise: áp ratio rồi ép absolute delivery (có thể lệch ratio).
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
