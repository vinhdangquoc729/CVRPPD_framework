# -*- coding: utf-8 -*-
"""
Sinh dataset VRP: điểm ngẫu nhiên -> phân cụm (k-means) -> sinh depot (độc lập).
KHÔNG gán cluster cho depot trong CSV (để solver tự quyết).

Usage:
  python -m vrp.data.gen_clusters_from_points `
    --out dataset_test/Osaba_500_synth/base_500 `
    --n 200 --k 12 --depots 3 --veh_per_depot 4 `
    --capacity 80 100 120 150 `
    --bbox 493000 4785000 507000 4800000 `
    --seed 42 --tw_mode cluster_stagger
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# ---------- helpers ----------
def pairwise_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:,None,:] - C[None,:,:])**2).sum(axis=2))

def kmeans_plus_plus_init(X: np.ndarray, K: int, rng: random.Random) -> np.ndarray:
    N = X.shape[0]
    centers = [X[rng.randrange(N)].copy()]
    for _ in range(1, K):
        D2 = np.min(pairwise_dists(X, np.array(centers))**2, axis=1)
        probs = D2 / D2.sum()
        r = rng.random()
        centers.append(X[int(np.searchsorted(np.cumsum(probs), r))].copy())
    return np.array(centers)

def kmeans_lloyd(X: np.ndarray, K: int, seed: int = 42, iters: int = 50):
    rng = random.Random(seed)
    C = kmeans_plus_plus_init(X, K, rng)
    for _ in range(iters):
        D = pairwise_dists(X, C)
        A = np.argmin(D, axis=1)
        C_new = C.copy()
        for k in range(K):
            m = (A == k)
            if m.any():
                C_new[k] = X[m].mean(axis=0)
        if np.allclose(C_new, C):
            break
        C = C_new
    D = pairwise_dists(X, C)
    A = np.argmin(D, axis=1)
    return A, C

def balance_assignments(X: np.ndarray, A: np.ndarray, C: np.ndarray, K: int, max_ratio: float = 1.4):
    """Giảm cụm quá đông: chuyển điểm xa centroid nhất sang cụm gần khác đang thiếu."""
    target = len(A) / K
    D = pairwise_dists(X, C)
    changed = True
    while changed:
        changed = False
        sizes = np.bincount(A, minlength=K).astype(int)
        over = [k for k in range(K) if sizes[k] > math.ceil(target*max_ratio)]
        under = [k for k in range(K) if sizes[k] < math.floor(target*(2-max_ratio))]
        if not over or not under:
            break
        k_over = max(over, key=lambda k: sizes[k]-target)
        idxs = np.where(A == k_over)[0]
        idxs = idxs[np.argsort(-D[idxs, k_over])]  # xa nhất trước
        for i in idxs:
            k2 = min(under, key=lambda kk: D[i, kk])
            A[i] = k2
            changed = True
            sizes[k_over] -= 1
            sizes[k2] += 1
            if sizes[k_over] <= math.ceil(target*max_ratio):
                break
    return A

def farthest_first(points: np.ndarray, p: int, seed: int=42) -> List[int]:
    rng = random.Random(seed)
    n = len(points)
    chosen = [rng.randrange(n)]
    while len(chosen) < p:
        dmin = np.full(n, np.inf)
        for c in chosen:
            d = np.linalg.norm(points - points[c], axis=1)
            dmin = np.minimum(dmin, d)
        chosen.append(int(np.argmax(dmin)))
    return chosen

def kmedoids_one_swap(points: np.ndarray, p: int, seed: int=42, iters: int=50) -> List[int]:
    medoids = farthest_first(points, p, seed)
    def total_cost(meds):
        D = pairwise_dists(points, points[meds])
        return float(D.min(axis=1).sum())
    best = total_cost(medoids)
    improved = True; it=0
    while improved and it < iters:
        improved = False; it += 1
        non_m = [i for i in range(len(points)) if i not in medoids]
        for mi, m in enumerate(medoids):
            for j in non_m:
                trial = medoids.copy(); trial[mi] = j
                c = total_cost(trial)
                if c + 1e-9 < best:
                    best, medoids, improved = c, trial, True
                    break
            if improved: break
    return medoids

# ---------- main ----------
def generate_dataset(
    out_dir: str,
    N: int,
    K: int,
    D: int,
    veh_per_depot: int,
    capacities: List[int],
    bbox: Tuple[float,float,float,float],
    seed: int,
    tw_mode: str = "cluster_stagger",
):
    rng = random.Random(seed)
    np.random.seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = bbox
    # 1) random points
    X = np.column_stack([np.random.uniform(xmin, xmax, N),
                         np.random.uniform(ymin, ymax, N)])

    # 2) clusters (k-means)
    A, C = kmeans_lloyd(X, K, seed=seed, iters=60)
    A = balance_assignments(X, A, C, K, max_ratio=1.4)
    C = np.vstack([X[A==k].mean(axis=0) for k in range(K)])

    # 3) depots = k-medoids over ALL customers (độc lập cluster)
    depot_ids = kmedoids_one_swap(X, D, seed=seed)
    depot_coords = {d: (float(X[i,0]), float(X[i,1])) for d, i in enumerate(depot_ids)}

    # 4) nodes.csv — đúng schema bạn dùng, KHÔNG có depot_id ở khách
    rows = []
    # depots
    for d_id, (dx, dy) in depot_coords.items():
        rows.append({
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
    for i in range(N):
        cid = int(A[i])
        x, y = float(X[i,0]), float(X[i,1])
        dem_del  = rng.randint(1, 10)
        dem_pick = rng.randint(0, 3)
        if tw_mode == "cluster_stagger":
            bands = min(K, 6)
            slot_w = (day_end - day_start) // bands
            slot_id = cid % bands
            base_open = day_start + slot_w * slot_id
            base_close = base_open + slot_w
            jitter = rng.randint(-10, 10)
            tw_open  = max(day_start, base_open + jitter)
            tw_close = min(day_end,   base_close + jitter)
        else:
            tw_open = tw_close = ""
        rows.append({
            "id": base_id + i,
            "addr": f"C{base_id + i}",
            "cluster": cid,
            "demand_delivery": int(dem_del),
            "demand_pickup": int(dem_pick),
            "x": x, "y": y,
            "is_depot": 0,
            "service_time": rng.randint(2, 6),
            "tw_open": tw_open,
            "tw_close": tw_close,
        })
    pd.DataFrame(rows).to_csv(out / "nodes.csv", index=False)

    # 5) vehicles.csv — MỖI xe có depot_id là ID depot (0..D-1). Không động đến cluster.
    veh_rows = []
    vid = 0
    caps = capacities if capacities else [240]
    for d_id in range(D):
        for k in range(veh_per_depot):
            cap = int(caps[vid % len(caps)])  # xoay vòng nếu truyền nhiều capacity
            veh_rows.append({
                "id": vid,
                "depot_id": d_id,              # <-- chỉ nói xe này thuộc depot nào
                "capacity": cap,
                "fixed_cost": 10.0,
                "var_cost_per_dist": 1.0,
                "start_time": day_start,
                "end_time": day_end
            })
            vid += 1
    pd.DataFrame(veh_rows).to_csv(out / "vehicles.csv", index=False)

    # 6) meta.json
    meta = {"tw_penalty_per_min": 1.0, "travel_speed_units_per_min": 100.0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] saved nodes/vehicles/meta to: {out}")

# ---------- CLI ----------
def parse_bbox(v: List[float]) -> tuple[float,float,float,float]:
    assert len(v) == 4, "bbox cần 4 số: xmin ymin xmax ymax"
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="số khách")
    ap.add_argument("--k", type=int, required=True, help="số cụm (cluster)")
    ap.add_argument("--depots", type=int, default=3, help="số depot (D)")
    ap.add_argument("--veh_per_depot", type=int, default=3)
    ap.add_argument("--capacity", nargs="+", type=int, default=[240],
                    help="capacity cho xe; truyền nhiều giá trị để xoay vòng")
    ap.add_argument("--bbox", nargs=4, type=float, default=[0,0,1000,1000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tw_mode", choices=["cluster_stagger","none"], default="cluster_stagger")
    args = ap.parse_args()

    generate_dataset(
        out_dir=args.out,
        N=args.n, K=args.k, D=args.depots,
        veh_per_depot=args.veh_per_depot,
        capacities=args.capacity,
        bbox=parse_bbox(args.bbox),
        seed=args.seed,
        tw_mode=args.tw_mode,
    )

if __name__ == "__main__":
    main()
