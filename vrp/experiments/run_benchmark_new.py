from __future__ import annotations
import argparse
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import các thành phần từ project
try:
    from ..data.loader import load_problem
    from ..data.loader_modified import load_problem_modified
    from ..core.eval import evaluate
    from ..core.eval_modified import evaluate_modified
    from ..core.eval_org import evaluate as evaluate_org
    from ..core.eval_with_weight import evaluate_with_weight

    # Import các Solvers
    from ..solvers.dfa import DFASolver
    from ..solvers.ga_hct_pd import GA_HCT_PD_Solver
    from ..solvers.ga_pd_hct import GAPD_HCT_Solver
    from ..solvers.esa import ESASolver
    from ..solvers.dfa_modified import DFASolverPD
    from ..solvers.esa_modified import ESASolverPD
    from ..solvers.cluster_ga_modified import ClusterGASolverPD
    from ..solvers.cluster_ga import ClusterGASolver
    from ..solvers.ga_ombuki import OmbukiGASolver
    from ..solvers.ga_pd_hct_origin import GAPD_HCT_ORIGIN_Solver
except ImportError:
    print("Lỗi: Không thể import các module. Hãy đảm bảo chạy script bằng lệnh: python -m vrp.experiments.run_benchmark_new")
    sys.exit(1)

SOLVERS = {
    "dfa": DFASolver,
    "esa": ESASolver,
    "dfa_pd": DFASolverPD,
    "esa_pd": ESASolverPD,
    "ga_hct_pd": GA_HCT_PD_Solver,
    "ga_pd_hct": GAPD_HCT_Solver,
    "cluster_ga": ClusterGASolver,
    "cluster_ga_pd": ClusterGASolverPD,
    "ga_ombuki": OmbukiGASolver,
    "ga_pd_hct_origin": GAPD_HCT_ORIGIN_Solver,
}

# Danh sách các loại phạt (Penalty) dựa theo eval.py
PENALTY_KEYS = [
    'tw_penalty', 
    'cap_violations', 
    'overtime_routes', 
    'unserved_customers', 
    'prohibited_uses', 
    'continuity_errors'
]

def get_vehicles_used(sol) -> int:
    """Đếm số xe thực tế có phục vụ khách hàng (len > 2)."""
    if not sol or not sol.routes: return 0
    active_routes = [r for r in sol.routes if len(r.seq) > 2]
    return len(active_routes)

def run_single_instance(data_path: Path, solver_name: str, seed: int, 
                        time_limit: float, max_gen: int, evaluator_mode: str) -> Dict:
    instance_id = f"{data_path.parent.name}/{data_path.name}"
    res = {"instance": instance_id, "solver": solver_name, "seed": seed, "ok": False, "error": ""}

    try:
        # 1. Xác định loader và evaluator
        eval_func = None
        if evaluator_mode == "pd":
            eval_func = evaluate_modified
            loader_func = load_problem_modified
        elif evaluator_mode == "org":
            eval_func = evaluate_org
            loader_func = load_problem
        elif evaluator_mode == "with_weight":
            eval_func = evaluate_with_weight
            loader_func = load_problem
        else:
            eval_func = evaluate
            loader_func = load_problem
        prob = loader_func(str(data_path))
        
        # 2. Khởi tạo Solver
        SolverCls = SOLVERS[solver_name]
        # Thử truyền max_generation, nếu không hỗ trợ thì fallback
        try:
            solver = SolverCls(prob, seed=seed, max_generation=max_gen, evaluator=eval_func)
        except TypeError:
            solver = SolverCls(prob, seed=seed, evaluator=eval_func)

        # 3. Chạy Solver
        t0 = time.time()
        sol = solver.solve(time_limit_sec=time_limit)
        runtime = time.time() - t0

        # 4. Đánh giá kết quả
        cost, details = eval_func(prob, sol, return_details=True)
        
        # CHỈNH SỬA TẠI ĐÂY: Khớp với key "fixed" trong hàm evaluate của bạn
        fixed_val = details.get('fixed', 0) 

        res.update({
            "ok": True,
            "total_cost": cost,
            "distance": details.get('distance', 0),
            "fixed_cost": fixed_val,  # Lưu vào bảng với tên fixed_cost cho dễ hiểu
            "runtime": runtime,
            "vehicles_used": get_vehicles_used(sol),
            "routes_count": len([r for r in sol.routes if len(r.seq) > 2]),
            "solution": str(sol)
        })

        # 5. Lưu các loại penalty chi tiết
        total_p = 0
        for k in PENALTY_KEYS:
            val = details.get(k, 0)
            res[k] = val
            total_p += val
        res["total_penalty"] = total_p

    except Exception as e:
        res["error"] = str(e)
    return res

def main():
    parser = argparse.ArgumentParser(description="VRP Benchmark Runner - Fixed Cost Corrected")
    parser.add_argument("--data_glob", required=True, help="Glob pattern dataset")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--time", type=float, default=30.0)
    parser.add_argument("--max_gen", type=int, default=500)
    parser.add_argument("--out", type=str, default="benchmark_summary.csv")
    parser.add_argument("--out_raw", type=str, default="benchmark_raw_details.csv")
    parser.add_argument("--evaluator", choices=["orig", "pd", "org", "with_weight"], default=None)
    args = parser.parse_args()

    data_paths = sorted([p for p in Path(".").glob(args.data_glob) if p.is_dir()])
    if not data_paths: return

    raw_results = []
    for data_path in data_paths:
        for solver_name in args.solvers:
            for seed in args.seeds:
                print(f"Running {solver_name} on {data_path} (Seed {seed}) with evaluator {args.evaluator}...")
                res = run_single_instance(data_path, solver_name, seed, args.time, args.max_gen, args.evaluator)
                raw_results.append(res)

    df = pd.DataFrame(raw_results)
    df.to_csv(args.out_raw, index=False)

    df_success = df[df['ok'] == True].copy()
    if df_success.empty: return

    # Cấu hình tính toán Summary
    agg_dict = {
        'total_cost': ['min', 'mean', 'std'],
        'distance': 'mean',
        'fixed_cost': 'mean', # Tính Mean cho Fixed Cost chính xác
        'runtime': 'mean',
        'vehicles_used': 'mean',
        'total_penalty': 'mean'
    }
    for k in PENALTY_KEYS: agg_dict[k] = 'mean'

    summary = df_success.groupby(['instance', 'solver']).agg(agg_dict)
    summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns.values]
    summary = summary.reset_index()

    # Tính Gap dựa trên Best Cost tìm được của Instance đó
    best_instance_cost = df_success.groupby('instance')['total_cost'].min().to_dict()
    summary['Cost Gap (%)'] = summary.apply(
        lambda x: ((x['total_cost_min'] - best_instance_cost[x['instance']]) / best_instance_cost[x['instance']]) * 100, axis=1
    )

    summary = summary.round(2)
    summary.to_csv(args.out, index=False)
    print(f"\nĐã xuất báo cáo. Kiểm tra cột 'fixed_cost_mean' trong {args.out}")

if __name__ == "__main__":
    main()