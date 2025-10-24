# -*- coding: utf-8 -*-
"""
Sinh dataset VRP:
- Chọn K trung tâm cụm (random hoặc farthest-first).
- Sinh khách xung quanh mỗi trung tâm (annulus [r_min, r_max]).
- Gán 'cluster' = id cụm theo trung tâm; KHÔNG gán depot cho cụm.
- Sinh D depot (độc lập) bằng farthest-first trên tập trung tâm (để tản đều).
- Xuất nodes.csv (schema: id,addr,cluster,demand_delivery,demand_pickup,x,y,is_depot,service_time,tw_open,tw_close)
  + vehicles.csv (mỗi xe có depot_id = id depot), meta.json.

Usage:
  python gen_clusters_around_centers.py `
    --out dataset_test/Osaba_200_synth/base_200 `
    --n 200 --k 12 `
    --depots 3 --veh_per_depot 4 `
    --capacity 80 100 120 150 `
    --bbox 493000 4785000 507000 4800000 `
    --r_min 200 --r_max 600 `
    --center_mode farthest --seed 42 --tw_mode cluster_stagger
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# ---------- helpers ----------
def farthest_first(points: np.ndarray, p: int, seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    n = len(points)
    if n == 0:
        return []
    chosen = [rng.randrange(n)]
    while len(chosen) < min(p, n):
        dmin = np.full(n, np.inf)
        for c in chosen:
            d = np.linalg.norm(points - points[c], axis=1)
            dmin = np.minimum(dmin, d)
        nxt = int(np.argmax(dmin))
        if nxt in chosen:  # fallback
            nxt = rng.randrange(n)
            if nxt in chosen:
                break
        chosen.append(nxt)
    return chosen

def sample_annulus(cx: float, cy: float, r_min: float, r_max: float, rng: random.Random) -> tuple[float,float]:
    # Uniform theo diện tích: r = sqrt(U*(r_max^2 - r_min^2) + r_min^2)
    u = rng.random()
    r = math.sqrt(u * (r_max*r_max - r_min*r_min) + r_min*r_min)
    a = rng.uniform(0.0, 2*math.pi)
    return cx + r*math.cos(a), cy + r*math.sin(a)

# ---------- main ----------
def generate_dataset(
    out_dir: str,
    N: int,
    K: int,
    D: int,
    veh_per_depot: int,
    capacities: List[int],
    bbox: Tuple[float,float,float,float],
    r_min: float,
    r_max: float,
    center_mode: str,
    seed: int,
    tw_mode: str,
):
    rng = random.Random(seed)
    np.random.seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox

    # 1) Chọn K trung tâm cụm trong bbox
    centers = np.column_stack([
        np.random.uniform(xmin, xmax, 4*K),     # over-sample rồi chọn farthest-first nếu cần
        np.random.uniform(ymin, ymax, 4*K)
    ])
    if center_mode == "farthest":
        idxs = farthest_first(centers, K, seed=seed)
        C = centers[idxs]
    else:
        C = centers[:K]

    # 2) Phân bổ số khách cho từng cụm (gần đều)
    base = N // K
    rem = N - base*K
    sizes = [base + (1 if i < rem else 0) for i in range(K)]

    # 3) Sinh khách xung quanh từng trung tâm (annulus [r_min, r_max])
    rows = []
    cust_coords = []
    cid_of = []
    for cid, (cx, cy) in enumerate(C):
        for _ in range(sizes[cid]):
            # rejection để giữ trong bbox; nếu fail nhiều lần thì clamp
            ok = False
            for _try in range(50):
                x, y = sample_annulus(float(cx), float(cy), r_min, r_max, rng)
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ok = True; break
            if not ok:
                x = min(max(cx, xmin), xmax)
                y = min(max(cy, ymin), ymax)
            cust_coords.append((float(x), float(y)))
            cid_of.append(cid)

    # 4) Sinh D depot độc lập: chọn từ tập trung tâm (farthest-first) để tản đều
    dep_ids = farthest_first(C, D, seed=seed)
    dep_coords = {d: (float(C[i,0]), float(C[i,1])) for d, i in enumerate(dep_ids)}

    # 5) nodes.csv (đúng schema, KHÔNG có depot_id ở khách)
    # depots
    nodes = []
    for d_id, (dx, dy) in dep_coords.items():
        nodes.append({
            "id": d_id,
            "addr": f"D{d_id}",
            "cluster": -1,
            "demand_delivery": 0,
            "demand_pickup": 0,
            "x": dx, "y": dy,
            "is_depot": 1,
            "service_time": 0,
            "tw_open": "", "tw_close": ""
        })
    # customers
    base_id = D
    day_start, day_end = 8*60, 20*60
    bands = min(K, 6)
    slot_w = (day_end - day_start) // max(1, bands)

    for i, ((x, y), cid) in enumerate(zip(cust_coords, cid_of)):
        dem_del = rng.randint(1, 10)
        dem_pick = rng.randint(0, 3)
        if tw_mode == "cluster_stagger":
            slot_id = cid % max(1, bands)
            base_open  = day_start + slot_w * slot_id
            base_close = base_open + slot_w
            jitter = rng.randint(-10, 10)
            tw_open  = max(day_start, base_open  + jitter)
            tw_close = min(day_end,   base_close + jitter)
        else:
            tw_open = tw_close = ""
        nodes.append({
            "id": base_id + i,
            "addr": f"C{base_id + i}",
            "cluster": int(cid),
            "demand_delivery": int(dem_del),
            "demand_pickup": int(dem_pick),
            "x": float(x), "y": float(y),
            "is_depot": 0,
            "service_time": rng.randint(2, 6),
            "tw_open": tw_open,
            "tw_close": tw_close
        })

    pd.DataFrame(nodes).to_csv(out / "nodes.csv", index=False)

    # 6) vehicles.csv — mỗi xe gắn với 1 depot (solver sẽ quyết cụm nào đi với depot nào)
    veh_rows = []
    vid = 0
    caps = capacities if capacities else [240]
    for d_id in range(D):
        for k in range(veh_per_depot):
            cap = int(caps[vid % len(caps)])  # xoay vòng nếu truyền nhiều capacity
            veh_rows.append({
                "vehicle_id": vid,
                "depot_id": d_id,
                "capacity": cap,
                "fixed_cost": 10.0,
                "variable_cost_per_distance": 1.0,
                "start_time": day_start,
                "end_time": day_end
            })
            vid += 1
    pd.DataFrame(veh_rows).to_csv(out / "vehicles.csv", index=False)

    # 7) meta.json
    meta = { "tw_penalty_per_min": 1.0, "travel_speed_units_per_min": 100.0 }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] saved to {out}")

# ---------- CLI ----------
def parse_bbox(v: List[float]) -> tuple[float,float,float,float]:
    assert len(v) == 4, "bbox cần 4 số: xmin ymin xmax ymax"
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="số khách")
    ap.add_argument("--k", type=int, required=True, help="số cụm (cluster)")
    ap.add_argument("--depots", type=int, default=3, help="số depot")
    ap.add_argument("--veh_per_depot", type=int, default=3)
    ap.add_argument("--capacity", nargs="+", type=int, default=[240],
                    help="capacity cho xe; truyền nhiều giá trị để xoay vòng")
    ap.add_argument("--bbox", nargs=4, type=float, default=[0,0,1000,1000])
    ap.add_argument("--r_min", type=float, default=100.0, help="bán kính trong (annulus)")
    ap.add_argument("--r_max", type=float, default=400.0, help="bán kính ngoài (annulus)")
    ap.add_argument("--center_mode", choices=["random","farthest"], default="farthest",
                    help="cách chọn trung tâm cụm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tw_mode", choices=["cluster_stagger","none"], default="cluster_stagger")
    args = ap.parse_args()

    generate_dataset(
        out_dir=args.out,
        N=args.n, K=args.k, D=args.depots,
        veh_per_depot=args.veh_per_depot,
        capacities=args.capacity,
        bbox=parse_bbox(args.bbox),
        r_min=args.r_min, r_max=args.r_max,
        center_mode=args.center_mode,
        seed=args.seed, tw_mode=args.tw_mode
    )

if __name__ == "__main__":
    main()
