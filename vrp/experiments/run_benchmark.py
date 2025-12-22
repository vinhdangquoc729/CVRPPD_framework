# vrp/experiments/run_benchmark.py
from __future__ import annotations
import argparse
import csv
import time
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Import các thành phần từ project
from ..data.loader import load_problem
from ..data.loader_modified import load_problem_modified
from ..core.eval import evaluate
from ..core.eval_modified import evaluate_modified

# Import các Solver
from ..solvers.dfa import DFASolver
from ..solvers.ga_hct_pd import GA_HCT_PD_Solver
from ..solvers.ga_pd_hct import GAPD_HCT_Solver
from ..solvers.esa import ESASolver
from ..solvers.dfa_modified import DFASolverPD
from ..solvers.esa_modified import ESASolverPD
from ..solvers.cluster_ga_modified import ClusterGASolverPD
from ..solvers.cluster_ga import ClusterGASolver
from ..solvers.ga_ombuki import OmbukiGASolver

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
}

def get_vehicles_used(sol) -> int:
    """
    Đếm số lượng xe thực tế được sử dụng.
    Chỉ tính các xe có ít nhất 1 route chứa khách hàng (len > 2).
    Route rỗng [Depot, Depot] (len=2) sẽ bị bỏ qua.
    """
    if not sol or not sol.routes:
        return 0
    
    active_vehicle_ids = set()
    for r in sol.routes:
        # Route có khách phải có dạng [Depot, Khách..., Depot] -> len >= 3
        if len(r.seq) > 2:
            active_vehicle_ids.add(r.vehicle_id)
            
    return len(active_vehicle_ids)

def run_single_instance(
    data_path: Path, 
    solver_name: str, 
    seed: int, 
    time_limit: float,
    evaluator_mode: str = None
) -> Dict[str, Any]:
    
    res = {
        "dataset": data_path.name,
        "solver": solver_name,
        "seed": seed,
        "ok": False,
        "total_cost": None,
        "runtime": None,
        "vehicles_used": 0,
        "details": "",
        "solution": "",
        "error": ""
    }

    try:
        # 1. Xác định Loader và Evaluator
        if evaluator_mode is None:
            use_modified = solver_name.endswith("_pd")
        else:
            use_modified = (evaluator_mode == "pd")

        if use_modified:
            loader_func = load_problem_modified
            eval_func = evaluate_modified
        else:
            loader_func = load_problem
            eval_func = evaluate

        # 2. Load bài toán
        prob = loader_func(str(data_path))

        # 3. Khởi tạo Solver
        SolverCls = SOLVERS[solver_name]
        solver = SolverCls(prob, seed=seed)

        # 4. Giải bài toán
        t0 = time.time()
        sol = solver.solve(time_limit_sec=time_limit)
        runtime = time.time() - t0

        # 5. Đánh giá kết quả
        cost, details = eval_func(prob, sol, return_details=True)
        
        # 6. Tính số xe (Logic mới: Bỏ qua route rỗng)
        n_vehicles = get_vehicles_used(sol)

        # 7. Ghi nhận kết quả
        res.update({
            "ok": True,
            "total_cost": cost,
            "runtime": round(runtime, 4),
            "vehicles_used": n_vehicles,
            "details": str(details), 
            "solution": str(sol)     
        })

    except Exception as e:
        res["error"] = str(e)
        # traceback.print_exc()

    return res

def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner for VRP Solvers")
    
    parser.add_argument("--data_glob", required=True, help="Glob pattern dataset (vd: 'data/test_set/*')")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS.keys()), choices=list(SOLVERS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--time", type=float, default=30.0)
    parser.add_argument("--out", type=str, default="benchmark_results.csv")
    parser.add_argument("--evaluator", choices=["orig", "pd"], default=None)

    args = parser.parse_args()

    # Tìm dataset
    # Lưu ý: Path(".").glob(...) tìm từ thư mục hiện tại.
    # Nếu bạn chạy từ root project, nó sẽ tìm đúng.
    data_paths = sorted(list(Path(".").glob(args.data_glob)))
    
    # Lọc chỉ lấy thư mục
    data_paths = [p for p in data_paths if p.is_dir()]

    if not data_paths:
        print(f"ERROR: Không tìm thấy dataset folder nào khớp với '{args.data_glob}'")
        return

    print(f"Found {len(data_paths)} datasets.")
    print(f"Solvers: {args.solvers}")
    print(f"Seeds: {args.seeds}")
    
    # Chuẩn bị CSV
    fieldnames = [
        "dataset", "solver", "seed", "ok", 
        "total_cost", "runtime", "vehicles_used", 
        "details", "solution", "error"
    ]
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Mở file CSV với chế độ 'w' (ghi mới) hoặc 'a' (nối đuôi) tùy bạn. 
    # Ở đây để 'w' để chạy lại từ đầu.
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        total_runs = len(data_paths) * len(args.solvers) * len(args.seeds)
        current_run = 0

        for data_path in data_paths:
            for solver_name in args.solvers:
                for seed in args.seeds:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] Running {solver_name} on {data_path.name} (seed={seed})...", end="", flush=True)
                    
                    result = run_single_instance(
                        data_path=data_path,
                        solver_name=solver_name,
                        seed=seed,
                        time_limit=args.time,
                        evaluator_mode=args.evaluator
                    )
                    
                    writer.writerow(result)
                    f.flush()
                    
                    if result["ok"]:
                        print(f" DONE. Cost={result['total_cost']:.2f}, Veh={result['vehicles_used']}")
                    else:
                        print(f" FAILED. Error: {result['error']}")

    print(f"\nHoàn tất benchmark. Kết quả lưu tại: {args.out}")

if __name__ == "__main__":
    main()