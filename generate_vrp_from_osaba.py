#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vrp_from_osaba.py  (multi-file capable, unified 4-file schema)

Usage (examples):
  # Basic: normalize and create a multi-depot/multi-vehicle (+TW) version of the base XML
  python generate_vrp_from_osaba.py --in /path/Osaba_50_1_1.xml --out out_dir --expand-base

  # Process all XMLs in a folder
  python generate_vrp_from_osaba.py --in /path/to/xml_folder --out out_dir --expand-base

  # Add synthetic instances 200,500,1000 customers with custom cluster and time-window settings
  python generate_vrp_from_osaba.py --in Osaba_50_1_1.xml --out out_dir \
    --expand-base \
    --synth-sizes 200 500 1000 \
    --synth-clusters-min 20 --synth-clusters-max 50 \
    --tw-open-min 480 --tw-open-max 720 --tw-close-min 780 --tw-close-max 1140 --tw-min-width 60 \
    --veh-per-depot-min 3 --veh-per-depot-max 4 \
    --veh-capacities 80 100 120 150 180 \
    --tw-penalty-per-min 1.0 \
    --seed 42

Outputs (per XML):
  out/<xml_stem>/
    base_50/                (ALWAYS 4 files: nodes/vehicles/prohibited_arcs/meta)
    base_50_mdmv_tw/        (if --expand-base)
    synthetic_{N}_mdmv_tw/  (if --synth-sizes ...)
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
    p = argparse.ArgumentParser(description="Generate VRP datasets from an Osaba-style XML (normalize + expand + synthesize).")
    p.add_argument("--in", dest="in_xml", required=True, help="Input Osaba XML path OR a folder containing *.xml")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory")

    # Expansion for base instance
    p.add_argument("--expand-base", action="store_true", help="Create multi-depot, multi-vehicle, time-window expansion for the base XML")

    # Depot selection
    p.add_argument("--depots-extra-min", type=int, default=2, help="Min additional depots for base expansion (besides original depot id=0)")
    p.add_argument("--depots-extra-max", type=int, default=3, help="Max additional depots for base expansion")
    p.add_argument("--depots-min", type=int, default=3, help="Min number of depots for synthetic instances")
    p.add_argument("--depots-max", type=int, default=4, help="Max number of depots for synthetic instances")

    # Vehicles
    p.add_argument("--veh-per-depot-min", type=int, default=3, help="Min vehicles per depot")
    p.add_argument("--veh-per-depot-max", type=int, default=4, help="Max vehicles per depot")
    p.add_argument("--veh-capacities", type=int, nargs="+", default=[80, 100, 120], help="Candidate vehicle capacities")
    p.add_argument("--veh-start", type=int, default=7*60, help="Vehicle start time (minutes from midnight)")
    p.add_argument("--veh-end", type=int, default=20*60, help="Vehicle end time (minutes from midnight)")
    p.add_argument("--veh-fixed-cost", type=float, default=100.0, help="Vehicle fixed cost")
    p.add_argument("--veh-var-cost", type=float, default=1.0, help="Vehicle variable cost per distance unit")

    # Service & time windows
    p.add_argument("--service-min", type=int, default=5, help="Min service time at customer (minutes)")
    p.add_argument("--service-max", type=int, default=15, help="Max service time at customer (minutes)")
    p.add_argument("--tw-open-min", type=int, default=8*60, help="TW open lower bound (minutes from midnight)")
    p.add_argument("--tw-open-max", type=int, default=12*60, help="TW open upper bound (inclusive)")
    p.add_argument("--tw-close-min", type=int, default=13*60, help="TW close lower bound (minutes from midnight)")
    p.add_argument("--tw-close-max", type=int, default=19*60, help="TW close upper bound (inclusive)")
    p.add_argument("--tw-min-width", type=int, default=60, help="Minimum TW width (minutes)")
    p.add_argument("--tw-penalty-per-min", type=float, default=1.0, help="Penalty per minute for violating time window")
    p.add_argument("--no-tw", action="store_true", help="Disable per-customer time windows (TW columns still present but NaN)")

    # Synthetic instance generation
    p.add_argument("--synth-sizes", type=int, nargs="*", default=[], help="List of synthetic customer sizes to generate (e.g., 200 500 1000)")
    p.add_argument("--synth-clusters-min", type=int, default=20, help="Min number of clusters for synthetic instances")
    p.add_argument("--synth-clusters-max", type=int, default=50, help="Max number of clusters for synthetic instances")
    p.add_argument("--synth-area-jitter", type=float, default=1200.0, help="Std-dev of Gaussian jitter around cluster centers")
    p.add_argument("--keep-prohibited-arcs", action="store_true", help="Copy prohibited arcs from XML into base_50_mdmv_tw (synthetic instances default to none)")

    # Demands (for synthetic)
    p.add_argument("--dem-delivery-candidates", type=int, nargs="+", default=[5, 10, 15, 20], help="Candidate delivery demands for synthetic")
    p.add_argument("--dem-delivery-probs", type=float, nargs="+", default=[0.4, 0.4, 0.15, 0.05], help="Probabilities for delivery demands")
    p.add_argument("--dem-pickup-candidates", type=int, nargs="+", default=[0, 3, 5, 10], help="Candidate pickup demands for synthetic")
    p.add_argument("--dem-pickup-probs", type=float, nargs="+", default=[0.5, 0.3, 0.15, 0.05], help="Probabilities for pickup demands")

    # Randomness
    p.add_argument("--seed", type=int, default=42, help="Random seed (base). When processing a folder, each XML is deterministically offset by its filename hash.")

    args = p.parse_args()
    return args


def assign_time_windows(n_customers, open_range, close_range, min_width):
    opens = np.random.randint(open_range[0], open_range[1] + 1, size=n_customers)
    closes = np.random.randint(close_range[0], close_range[1] + 1, size=n_customers)
    # ensure min width and ordering
    open_final = np.minimum(opens, closes - min_width)
    close_final = np.maximum(closes, open_final + min_width)
    open_final = np.clip(open_final, 0, 1440 - 1)
    close_final = np.clip(close_final, 0, 1440 - 1)
    return open_final, close_final


def choose_depots(coords, k):
    """Farthest-point sampling over customer coordinates."""
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
                      start_min, end_max, fixed_cost, var_cost):
    rows = []
    vid = 0
    for d in depot_ids:
        nveh = np.random.randint(per_depot_min, per_depot_max + 1)
        for _ in range(nveh):
            rows.append({
                "vehicle_id": vid,
                "depot_id": d,
                "capacity": int(np.random.choice(cap_choices)),
                "start_time": start_min,
                "end_time": end_max,
                "fixed_cost": fixed_cost,
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


def load_osaba_xml(xml_path):
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
            "x": int(node.findtext("CoordX")),
            "y": int(node.findtext("CoordY")),
        })
    nodes_df = pd.DataFrame(nodes).sort_values("id").reset_index(drop=True)

    prohib = []
    for pr in root.findall("Prohibido"):
        a = int(pr.findtext("est1"))
        b = int(pr.findtext("est2"))
        prohib.append({"from_id": a, "to_id": b})
    prohib_df = pd.DataFrame(prohib)

    return nodes_df, prohib_df


def synthesize_instance(nodes_df_base,
                        n_customers_target,
                        n_clusters_min, n_clusters_max,
                        area_jitter,
                        depots_min, depots_max,
                        service_range,
                        tw_enabled, tw_open_range, tw_close_range, tw_min_width,
                        cap_choices,
                        veh_start, veh_end, veh_fixed_cost, veh_var_cost,
                        veh_per_depot_min, veh_per_depot_max,
                        dem_deliv_candidates, dem_deliv_probs,
                        dem_pick_candidates, dem_pick_probs,
                        tw_penalty_per_min):

    base = nodes_df_base[nodes_df_base["id"] != 0].copy()
    centers = base.groupby("cluster")[["x", "y"]].mean().reset_index()
    base_centers = centers[["x", "y"]].to_numpy()

    # choose cluster centers with replacement
    n_clusters = int(np.clip(np.random.randint(n_clusters_min, n_clusters_max + 1), n_clusters_min, n_clusters_max))
    chosen = np.random.choice(len(base_centers), size=n_clusters, replace=True)
    cluster_centers = base_centers[chosen]

    # allocate customers to clusters via Dirichlet
    proportions = np.random.dirichlet(np.ones(n_clusters))
    cust_per_cluster = np.maximum((proportions * n_customers_target).astype(int), 1)
    diff = n_customers_target - cust_per_cluster.sum()
    if diff > 0:
        cust_per_cluster[:diff] += 1
    elif diff < 0:
        cust_per_cluster[:(-diff)] -= 1

    xs, ys, clusters_out = [], [], []
    cid = 1
    for c_idx, cnt in enumerate(cust_per_cluster):
        cx, cy = cluster_centers[c_idx]
        x = np.random.normal(loc=cx, scale=area_jitter, size=cnt)
        y = np.random.normal(loc=cy, scale=area_jitter, size=cnt)
        xs.append(x); ys.append(y)
        clusters_out.extend([cid] * cnt)
        cid += 1
    xs = np.concatenate(xs); ys = np.concatenate(ys)

    # demands
    dem_del = np.random.choice(dem_deliv_candidates, size=n_customers_target, p=dem_deliv_probs)
    dem_pick = np.random.choice(dem_pick_candidates, size=n_customers_target, p=dem_pick_probs)

    coords = np.stack([xs, ys], axis=1)
    k_depots = int(np.random.randint(depots_min, depots_max + 1))
    depot_indices = choose_depots(coords, k_depots)
    depot_rows = []
    for i, idx in enumerate(depot_indices):
        depot_rows.append({
            "id": i,
            "addr": f"DEPOT.{i}",
            "cluster": 0,
            "demand_delivery": 0,
            "demand_pickup": 0,
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "is_depot": True
        })

    start_id = len(depot_rows)
    records = []
    for i in range(n_customers_target):
        records.append({
            "id": start_id + i,
            "addr": f"C.{i+1}",
            "cluster": int(clusters_out[i]),
            "demand_delivery": int(dem_del[i]),
            "demand_pickup": int(dem_pick[i]),
            "x": float(xs[i]),
            "y": float(ys[i]),
            "is_depot": False
        })
    full_nodes = pd.DataFrame(depot_rows + records)

    full_nodes = with_service_and_tw(full_nodes,
                                     service_range=service_range,
                                     tw=tw_enabled,
                                     tw_open_range=tw_open_range,
                                     tw_close_range=tw_close_range,
                                     tw_min_width=tw_min_width)

    depot_ids = full_nodes[full_nodes["is_depot"]]["id"].tolist()
    veh_df = gen_vehicle_table(
        depot_ids=depot_ids,
        per_depot_min=veh_per_depot_min,
        per_depot_max=veh_per_depot_max,
        cap_choices=cap_choices,
        start_min=veh_start, end_max=veh_end,
        fixed_cost=veh_fixed_cost, var_cost=veh_var_cost
    )

    prohib_syn = pd.DataFrame(columns=["from_id", "to_id"])

    meta = {
        "source": "synthetic_from_Osaba_like",
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
    }
    return full_nodes, veh_df, prohib_syn, meta


def _standardize_base_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Make base_50 nodes match expanded schema: add is_depot, service_time, tw_open, tw_close."""
    out = nodes_df.copy()
    out["is_depot"] = out["id"] == 0
    out["service_time"] = 0
    out["tw_open"] = 0
    out["tw_close"] = 1439
    return out


def _default_base_vehicle_df(var_cost: float) -> pd.DataFrame:
    """Single-vehicle defaults for non-expanded base_50."""
    return pd.DataFrame([{
        "vehicle_id": 0,
        "depot_id": 0,                # assume Osaba depot node is id=0
        "capacity": 240,
        "start_time": 0,
        "end_time": 1439,
        "fixed_cost": 0.0,
        "variable_cost_per_distance": var_cost
    }])


def process_one(xml_path: Path, out_root: Path, args):
    # File-specific deterministic seed
    h = hashlib.sha256(xml_path.stem.encode("utf-8")).hexdigest()
    offset = int(h[:8], 16) % 10_000_000
    np.random.seed(args.seed + offset)

    this_out = out_root / xml_path.stem
    this_out.mkdir(parents=True, exist_ok=True)

    # 1) Load & normalize
    nodes_df, prohib_df = load_osaba_xml(xml_path)
    if prohib_df is None or len(prohib_df) == 0:
        prohib_df = pd.DataFrame(columns=["from_id", "to_id"])

    # --- base_50 (ALWAYS write 4 files with unified schema) ---
    base_dir = this_out / "base_50"
    base_dir.mkdir(parents=True, exist_ok=True)

    base_nodes_std = _standardize_base_nodes(nodes_df)
    base_vehicles = _default_base_vehicle_df(var_cost=float(args.veh_var_cost))

    base_meta = {
        "source": xml_path.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "type": "single_depot_single_vehicle_with_time_windows",
        "time_window_units": "minutes_from_midnight",
        "tw_penalty_per_min": float(args.tw_penalty_per_min),
        "travel_speed_units_per_min": 60.0,
        "service_time_units": "minutes",
        "distance_units": "euclidean (coordinate units)",
        "n_depots": 1,
        "n_vehicles": 1,
        "note": "Base (non-expanded): nodes carry TW 0-1439 and service_time=0; vehicles default to one vehicle with capacity=240, fixed_cost=0.",
    }

    write_instance(
        folder=base_dir,
        nodes_df=base_nodes_std,
        vehicles_df=base_vehicles,
        prohib_df=prohib_df,
        meta=base_meta
    )

    # 2) Optionally expand base to MD-MV (+TW)
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

        vehicles_df = gen_vehicle_table(
            depot_ids=multi_depots,
            per_depot_min=args.veh_per_depot_min,
            per_depot_max=args.veh_per_depot_max,
            cap_choices=args.veh_capacities,
            start_min=args.veh_start, end_max=args.veh_end,
            fixed_cost=args.veh_fixed_cost, var_cost=args.veh_var_cost,
        )

        base_exp_dir = this_out / "base_50_mdmv_tw"
        meta = {
            "source": xml_path.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "type": "multi_depot_multi_vehicle_with_time_windows" if not args.no_tw else "multi_depot_multi_vehicle",
            "time_window_units": "minutes_from_midnight",
            "tw_penalty_per_min": float(args.tw_penalty_per_min),
            "travel_speed_units_per_min": 60.0,
            "service_time_units": "minutes",
            "distance_units": "euclidean (coordinate units)",
            "note": "Depots chosen via farthest-point sampling among customers; original id=0 kept as depot.",
            "n_depots": len(multi_depots),
            "n_vehicles": int(len(vehicles_df)),
        }

        write_instance(
            folder=base_exp_dir,
            nodes_df=md_nodes,
            vehicles_df=vehicles_df,
            prohib_df=(prohib_df if args.keep_prohibited_arcs else pd.DataFrame(columns=["from_id", "to_id"])),
            meta=meta,
        )

    # 3) Synthetic instances
    for n in args.synth_sizes:
        inst_dir = this_out / f"synthetic_{n}_mdmv_tw"
        ndf, vdf, pdf, meta = synthesize_instance(
            nodes_df_base=nodes_df,
            n_customers_target=int(n),
            n_clusters_min=args.synth_clusters_min,
            n_clusters_max=args.synth_clusters_max,
            area_jitter=float(args.synth_area_jitter),
            depots_min=args.depots_min,
            depots_max=args.depots_max,
            service_range=(args.service_min, args.service_max),
            tw_enabled=(not args.no_tw),
            tw_open_range=(args.tw_open_min, args.tw_open_max),
            tw_close_range=(args.tw_close_min, args.tw_close_max),
            tw_min_width=args.tw_min_width,
            cap_choices=args.veh_capacities,
            veh_start=args.veh_start, veh_end=args.veh_end,
            veh_fixed_cost=args.veh_fixed_cost, veh_var_cost=args.veh_var_cost,
            veh_per_depot_min=args.veh_per_depot_min, veh_per_depot_max=args.veh_per_depot_max,
            dem_deliv_candidates=args.dem_delivery_candidates,
            dem_deliv_probs=args.dem_delivery_probs,
            dem_pick_candidates=args.dem_pickup_candidates,
            dem_pick_probs=args.dem_pickup_probs,
            tw_penalty_per_min=args.tw_penalty_per_min,
        )
        write_instance(inst_dir, ndf, vdf, pdf, meta)

    # 4) README cho file này
    readme = this_out / "README.md"
    readme.write_text(f"""# Osaba VRP — Normalized & Expanded for `{xml_path.name}` (Generated on {datetime.utcnow().isoformat()}Z)

## Base (unexpanded) — unified schema (4 files)
- `base_50/nodes.csv`: id, addr, cluster, demand_delivery, demand_pickup, x, y, is_depot, service_time=0, tw_open=0, tw_close=1439
- `base_50/vehicles.csv`: single vehicle (capacity=240, fixed_cost=0, start=0, end=1439)
- `base_50/prohibited_arcs.csv`: from_id, to_id
- `base_50/meta.json`: type=single_depot_single_vehicle_with_time_windows

## Expanded Base
- `base_50_mdmv_tw/` (if --expand-base): nodes + vehicles + prohibited_arcs + meta (MD-MV, ±TW)

## Synthetic Instances
- `synthetic_{{N}}_mdmv_tw/` for N in {args.synth_sizes}: same schema as expanded base.
""", encoding="utf-8")

    print(f"[OK] Processed: {xml_path} -> {this_out}")


def main(args):
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

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
    np.random.seed(args.seed)  # base seed (each file offsets deterministically)
    main(args)
