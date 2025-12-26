#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vrp_pd_only.py

FIXES:
1. Vehicles: Cập nhật allowed_goods_types tuân thủ nghiêm ngặt quy tắc "Không chở 1 & 4 đồng thời".
   Tập hợp lựa chọn: [1,2,3], [2,3,4], [2,3], [1,2], [1,3], [2,4], [3,4].
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Generate VRP datasets.")
    p.add_argument("--in", dest="in_xml", required=True)
    p.add_argument("--out", dest="out_dir", required=True)

    # Legacy params
    p.add_argument("--pd-target-ratio", type=float, default=0.95)
    p.add_argument("--pd-pickup-share", type=float, default=0.55)
    p.add_argument("--pd-min-amount", type=int, default=1)
    p.add_argument("--del-mult-from-original", type=float, default=2.0)
    p.add_argument("--lock-both-totals", action="store_true", default=False)
    p.add_argument("--expand-base", action="store_true")

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
    p.add_argument("--veh-fixed-cost", type=float, default=100.0)
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
    p.add_argument("--keep-prohibited-arcs", action="store_true")

    # Quantity distributions
    p.add_argument("--dem-delivery-candidates", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--dem-delivery-probs", type=float, nargs="+", default=[0.4, 0.4, 0.15, 0.05])
    p.add_argument("--dem-pickup-candidates", type=int, nargs="+", default=[3, 5, 10, 15])
    p.add_argument("--dem-pickup-probs", type=float, nargs="+", default=[0.3, 0.4, 0.2, 0.1])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------ Helpers ------------------------
def assign_time_windows(n_customers, open_range, close_range, min_width, rng=None):
    if rng is None: rng = np.random
    opens = rng.randint(open_range[0], open_range[1] + 1, size=n_customers)
    closes = rng.randint(close_range[0], close_range[1] + 1, size=n_customers)
    open_final = np.minimum(opens, closes - min_width)
    close_final = np.maximum(closes, open_final + min_width)
    open_final = np.clip(open_final, 0, 1440 - 1)
    close_final = np.clip(close_final, 0, 1440 - 1)
    return open_final, close_final


def choose_depots(coords, k):
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
                      start_min, end_max, var_cost, multiple_fixed_cost=100.0, rng=None):
    """
    Sinh bảng xe với cột allowed_goods_types.
    """
    if rng is None: rng = np.random.RandomState(42)
    
    rows, vid = [], 0
    
    # CÁC COMBO HÀNG HÓA HỢP LỆ (Không bao giờ chứa đồng thời 1 và 4)
    allowed_options = [
        [1, 2, 3],
        [2, 3, 4],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 4]
    ]
    
    for d in depot_ids:
        nveh = rng.randint(per_depot_min, per_depot_max + 1)
        for _ in range(nveh):
            cap = int(rng.choice(cap_choices))
            
            # Chọn ngẫu nhiên loại hàng cho phép
            opt_idx = rng.randint(0, len(allowed_options))
            allowed_goods = allowed_options[opt_idx]
            
            rows.append({
                "vehicle_id": vid,
                "depot_id": d,
                "capacity": cap,
                "start_time": start_min,
                "end_time": end_max,
                "fixed_cost": float(multiple_fixed_cost * cap),
                "variable_cost_per_distance": var_cost,
                "allowed_goods_types": json.dumps(allowed_goods)
            })
            vid += 1
    return pd.DataFrame(rows)


def with_service_and_tw(df, service_range, tw, tw_open_range, tw_close_range, tw_min_width, rng=None):
    """Gán Service Time và Time Window cho các Node không phải Depot."""
    if rng is None: rng = np.random.RandomState(42)
    
    out = df.copy()
    if "is_depot" not in out.columns:
        out["is_depot"] = False
        
    out["service_time"] = 0
    cust_mask = ~out["is_depot"]
    
    if cust_mask.sum() > 0:
        out.loc[cust_mask, "service_time"] = rng.randint(service_range[0], service_range[1] + 1, cust_mask.sum())
        
        if tw:
            open_min, close_min = assign_time_windows(cust_mask.sum(), tw_open_range, tw_close_range, tw_min_width, rng=rng)
            out.loc[cust_mask, "tw_open"] = open_min
            out.loc[cust_mask, "tw_close"] = close_min
        else:
            out["tw_open"] = np.nan
            out["tw_close"] = np.nan
            
    # Đảm bảo depot luôn mở full ngày
    out.loc[out["is_depot"], "tw_open"] = 0
    out.loc[out["is_depot"], "tw_close"] = 1439
    
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


# ------------------------ Goods & Order Logic ------------------------

def _split_integer(value: int, parts: int, rng: np.random.RandomState) -> list[int]:
    if parts <= 1:
        return [value]
    if value < parts: 
        res = [0] * parts
        for i in range(value): res[i] = 1
        return res
    
    points = sorted(rng.choice(range(1, value), parts - 1, replace=False))
    result = []
    prev = 0
    for p in points:
        result.append(p - prev)
        prev = p
    result.append(value - prev)
    return result

def _generate_goods_for_order(total_weight: int, rng: np.random.RandomState) -> list[dict]:
    if total_weight <= 0:
        return []
    
    num_items = rng.randint(1, 4) 
    weights = _split_integer(total_weight, num_items, rng)
    
    items = []
    # 0: none, 1: forbid 4, 4: forbid 1
    forbidden = None 
    available_types = [1, 2, 3, 4]
    
    for w in weights:
        if w <= 0: continue
        current_choices = list(available_types)
        if forbidden == 1 and 4 in current_choices:
            current_choices.remove(4)
        if forbidden == 4 and 1 in current_choices:
            current_choices.remove(1)
            
        g_type = rng.choice(current_choices)
        if g_type == 1: forbidden = 1
        if g_type == 4: forbidden = 4
        
        items.append({"goods_type": int(g_type), "weight": int(w)})
        
    return items

def _generate_complex_orders_for_base(nodes_df_with_demand: pd.DataFrame, depot_ids: list, rng: np.random.RandomState) -> pd.DataFrame:
    # Filter out DEPOTS from customers
    customers = nodes_df_with_demand[~nodes_df_with_demand["id"].isin(depot_ids)].copy()
    
    n_cust = len(customers)
    n_1 = int(0.5 * n_cust)
    n_2 = int(0.3 * n_cust)
    n_3 = n_cust - n_1 - n_2
    
    counts = [1]*n_1 + [2]*n_2 + [3]*n_3
    # Check if mismatch due to rounding
    if len(counts) < n_cust:
        counts.extend([1] * (n_cust - len(counts)))
    elif len(counts) > n_cust:
        counts = counts[:n_cust]
        
    rng.shuffle(counts)
    
    orders_data = []
    order_id_counter = 0
    
    for idx, (i, row) in enumerate(customers.iterrows()):
        node_id = int(row["id"])
        dem_del = int(row["demand_delivery"])
        dem_pick = int(row["demand_pickup"])
        
        target_n_orders = counts[idx]
        min_orders_needed = (1 if dem_del > 0 else 0) + (1 if dem_pick > 0 else 0)
        n_orders = max(target_n_orders, min_orders_needed)
        
        slots = [] 
        
        # Distribute Delivery
        if dem_del > 0:
            if dem_pick == 0:
                parts = _split_integer(dem_del, n_orders, rng)
                for p in parts: slots.append({"type": 0, "amount": p})
            else:
                n_del_slots = 1
                if n_orders > 2:
                    n_del_slots = rng.randint(1, n_orders)
                parts = _split_integer(dem_del, n_del_slots, rng)
                for p in parts: slots.append({"type": 0, "amount": p})

        # Distribute Pickup
        if dem_pick > 0:
            if dem_del == 0:
                parts = _split_integer(dem_pick, n_orders, rng)
                for p in parts: slots.append({"type": 1, "amount": p})
            else:
                n_pick_slots = n_orders - len(slots)
                if n_pick_slots < 1: n_pick_slots = 1
                parts = _split_integer(dem_pick, n_pick_slots, rng)
                for p in parts: slots.append({"type": 1, "amount": p})
                
        for slot in slots:
            goods = _generate_goods_for_order(slot["amount"], rng)
            orders_data.append({
                "order_id": order_id_counter,
                "node_id": node_id,
                "order_type": slot["type"], 
                "quantity": slot["amount"],
                "goods": json.dumps(goods)
            })
            order_id_counter += 1
            
    return pd.DataFrame(orders_data)


# ------------------------ Node Building Helpers ------------------------

def _build_pd_nodes_shuffled(rng, order_sizes, cluster_centers, area_jitter, start_node_id):
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


# ------------------------ Paired PD Logic ------------------------
def _pick_k_depots_parity(coords: np.ndarray, k_min: int, k_max: int, n_customers_total: int, rng: np.random.RandomState) -> int:
    candidates = list(range(k_min, k_max + 1))
    mid = 0.5 * (k_min + k_max)
    candidates.sort(key=lambda k: abs(k - mid))
    for k in candidates:
        if (n_customers_total - k) >= 2 and ((n_customers_total - k) % 2 == 0):
            return k
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
    base = nodes_df_base[nodes_df_base["id"] != 0].copy().reset_index(drop=True)
    all_coords = base[["x", "y"]].to_numpy()
    n_total = len(base)
    k_depots = _pick_k_depots_parity(all_coords, depots_min, depots_max, n_total, rng)
    depot_indices_in_base = choose_depots(all_coords, k_depots)
    
    depot_rows = []
    for i, idx in enumerate(depot_indices_in_base):
        row = base.iloc[idx]
        depot_rows.append({
            "id": i, "addr": f"DEPOT.{i}", "cluster": 0, "demand_delivery": 0, "demand_pickup": 0,
            "x": float(row["x"]), "y": float(row["y"]), "is_depot": True
        })

    mask = np.ones(n_total, dtype=bool)
    mask[depot_indices_in_base] = False
    customers_df = base.loc[mask].copy().reset_index(drop=True)
    if len(customers_df) % 2 == 1:
        customers_df = customers_df.iloc[:-1, :].reset_index(drop=True)

    n_customers = len(customers_df)
    n_orders = n_customers // 2
    perm = rng.permutation(n_customers)
    customers_df = customers_df.iloc[perm].reset_index(drop=True)
    order_sizes = _sample_order_sizes(rng, n_orders, order_qty_candidates, order_qty_probs, min_amount=1)

    pd_tmp_nodes = []
    for oid in range(n_orders):
        rP = customers_df.iloc[2 * oid + 0]
        rD = customers_df.iloc[2 * oid + 1]
        qty = int(order_sizes[oid])
        if rng.rand() < 0.5: rP, rD = rD, rP

        pd_tmp_nodes.append({
            "tmp_oid": oid, "role": "P", "addr": f"P.{oid}",
            "cluster": int(rP.get("cluster", 0)), "demand_delivery": 0, "demand_pickup": qty,
            "x": float(rP["x"]), "y": float(rP["y"]), "is_depot": False,
        })
        pd_tmp_nodes.append({
            "tmp_oid": oid, "role": "D", "addr": f"D.{oid}",
            "cluster": int(rD.get("cluster", 0)), "demand_delivery": qty, "demand_pickup": 0,
            "x": float(rD["x"]), "y": float(rD["y"]), "is_depot": False,
        })
    rng.shuffle(pd_tmp_nodes)

    next_id = len(depot_rows)
    rows, id_map = [], {}
    for nd in pd_tmp_nodes:
        nd_out = nd.copy()
        nd_out["id"] = next_id
        rows.append(nd_out)
        id_map[(nd["tmp_oid"], nd["role"])] = next_id
        next_id += 1

    orders = []
    for oid in range(n_orders):
        orders.append({
            "order_id": oid,
            "pickup_node_id": int(id_map[(oid, "P")]),
            "delivery_node_id": int(id_map[(oid, "D")]),
            "quantity": int(order_sizes[oid]),
        })
    orders_df = pd.DataFrame(orders)
    full_nodes = pd.DataFrame(depot_rows + rows)

    full_nodes = with_service_and_tw(full_nodes, service_range, tw_enabled, tw_open_range, tw_close_range, tw_min_width)
    if tw_enabled:
        full_nodes = _ensure_pd_order_tw(full_nodes, orders_df, travel_speed_units_per_min, tw_min_width)

    depot_ids = full_nodes[full_nodes["is_depot"]]["id"].tolist()
    veh_df = gen_vehicle_table(depot_ids, veh_per_depot_min, veh_per_depot_max, cap_choices, veh_start, veh_end, veh_var_cost)

    return full_nodes, veh_df, pd.DataFrame(columns=["from_id", "to_id"]), {}, orders_df


# ------------------------ Base normalization ------------------------
def _standardize_base_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Xóa cột demand để đảm bảo tính chất base_modified."""
    out = nodes_df.copy()
    # Các thông số TW sẽ được ghi đè bởi with_service_and_tw
    out["is_depot"] = out["id"] == 0
    out["service_time"] = 0
    
    # Drop cột demand
    cols_to_drop = [c for c in ["demand_delivery", "demand_pickup", "DemEnt", "DemRec"] if c in out.columns]
    out = out.drop(columns=cols_to_drop)
    return out


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

    # --- BASE MODIFIED: Tên folder là "base_modified" ---
    base_dir = this_out / "base_modified"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Base Nodes (với Logic Multi-Depot mới)
    all_coords = nodes_df[["x", "y"]].to_numpy()
    k_depots = rng.randint(args.depots_min, args.depots_max + 1)
    depot_indices = choose_depots(all_coords, k_depots)
    
    base_nodes_std = _standardize_base_nodes(nodes_df)
    
    # Set depot mới
    base_nodes_std["is_depot"] = False 
    base_nodes_std.loc[depot_indices, "is_depot"] = True
    
    base_depot_ids = base_nodes_std.loc[depot_indices, "id"].tolist()
    
    # --- FIXED: ÁP DỤNG TIME WINDOWS CHO BASE NODES ---
    base_nodes_std = with_service_and_tw(
        base_nodes_std,
        service_range=(args.service_min, args.service_max),
        tw=(not args.no_tw),
        tw_open_range=(args.tw_open_min, args.tw_open_max),
        tw_close_range=(args.tw_close_min, args.tw_close_max),
        tw_min_width=args.tw_min_width,
        rng=rng
    )
    
    # 2. Complex Orders 
    base_orders_df = _generate_complex_orders_for_base(nodes_df, base_depot_ids, rng)
    
    # 3. Vehicles (FIXED: có allowed_goods_types từ tập hợp hợp lệ)
    base_vehicles = gen_vehicle_table(
        depot_ids=base_depot_ids,
        per_depot_min=args.veh_per_depot_min,
        per_depot_max=args.veh_per_depot_max,
        cap_choices=args.veh_capacities,
        start_min=args.veh_start,
        end_max=args.veh_end,
        var_cost=float(args.veh_var_cost),
        rng=rng
    )

    base_meta = {
        "source": xml_path.name,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "type": "multi_depot_multi_vehicle_with_time_windows",
        "n_depots": len(base_depot_ids),
        "note": "Base Modified: Multi-Depot, Random TW applied, Goods Types in Vehicles (No 1 & 4 together)."
    }
    write_instance(base_dir, base_nodes_std, base_vehicles, prohib_df, base_meta, orders_df=base_orders_df)

    # --- EXPANSION (PAIRED) ---
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
            order_qty_candidates=np.asarray(args.dem_delivery_candidates, dtype=int),
            order_qty_probs=np.asarray(args.dem_delivery_probs, dtype=float),
            tw_penalty_per_min=args.tw_penalty_per_min,
            rng=np.random.RandomState(args.seed + offset + 1234),
            travel_speed_units_per_min=60.0,
        )

        base_exp_dir = this_out / "base_mdmv_tw_modified"
        meta.update({
            "source": xml_path.name,
            "coords_from_original": True
        })
        write_instance(base_exp_dir, md_nodes, vehicles_df, prohib_df, meta, orders_df=orders_df)

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