# vrp/experiments/run_benchmark_new.py
from __future__ import annotations
import argparse
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import modules
try:
    from ..data.loader import load_problem
    from ..data.loader_modified import load_problem_modified
    
    # Import evaluators
    from ..core.eval import evaluate
    from ..core.eval_modified import evaluate_modified
    from ..core.eval_org import evaluate as evaluate_org
    from ..core.eval_with_weight import evaluate_with_weight

    # Import Solvers
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
    print("Error: Could not import modules. Make sure to run as module: python -m vrp.experiments.run_benchmark_new")
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

# Updated Penalty Keys based on your evaluate function outputs
PENALTY_KEYS = [
    'tw_penalty', 
    'capacity_violations', 
    'stockout_violations', 
    'goods_incompatibility', 
    'goods_not_allowed', 
    'overtime_violations', 
    'unserved_orders'
]

def get_vehicles_used(sol) -> int:
    """Count vehicles with non-empty routes (len > 2 usually implies [depot, ..., depot])."""
    if not sol or not sol.routes: return 0
    # Assuming route structure is [start_depot, ... orders/nodes ..., end_depot]
    active_routes = [r for r in sol.routes if len(r.seq) > 2] 
    return len(active_routes)

def run_single_instance(data_path: Path, solver_name: str, seed: int, 
                        time_limit: float, max_gen: int, evaluator_mode: str) -> Dict:
    instance_id = f"{data_path.parent.name}/{data_path.name}"
    res = {"instance": instance_id, "solver": solver_name, "seed": seed, "ok": False, "error": ""}

    try:
        # 1. Determine Loader and Evaluator (Logic synced with run_experiment.py)
        eval_func = None
        loader_func = None
        
        # Default behavior matches run_experiment logic
        if evaluator_mode == "org":
            eval_func = evaluate_org
            loader_func = load_problem
        elif evaluator_mode == "pd" or (evaluator_mode is None and solver_name.endswith("_pd")):
            eval_func = evaluate_modified
            loader_func = load_problem_modified
        elif evaluator_mode == "orig":
            eval_func = evaluate
            loader_func = load_problem
        elif evaluator_mode == "with_weight":
            eval_func = evaluate_with_weight
            loader_func = load_problem
        else:
            # Default fallback
            eval_func = evaluate
            loader_func = load_problem

        # Load Problem
        prob = loader_func(str(data_path))
        
        # 2. Initialize Solver
        SolverCls = SOLVERS[solver_name]
        
        # Handle solver initialization arguments
        # Some solvers might not accept max_generation, so we try/except or inspect
        try:
            solver = SolverCls(prob, seed=seed, max_generation=max_gen, evaluator=eval_func)
        except TypeError:
            # Fallback for solvers that don't take max_generation in __init__
            solver = SolverCls(prob, seed=seed, evaluator=eval_func)

        # 3. Run Solver
        t0 = time.time()
        sol = solver.solve(time_limit_sec=time_limit)
        runtime = time.time() - t0

        # 4. Evaluate Result
        cost, details = eval_func(prob, sol, return_details=True)
        
        # Extract metrics safely
        dist_val = details.get('distance', 0)
        # Check for 'fixed_cost' or 'fixed' depending on specific eval implementation
        fixed_val = details.get('fixed_cost', details.get('fixed', 0))

        res.update({
            "ok": True,
            "total_cost": cost,
            "distance": dist_val,
            "fixed_cost": fixed_val,
            "runtime": runtime,
            "vehicles_used": get_vehicles_used(sol),
            "routes_count": len([r for r in sol.routes if len(r.seq) > 2]),
            # "solution": str(sol) # Uncomment if you want full solution string in CSV (can be large)
        })

        # 5. Record Penalties
        total_p = 0
        for k in PENALTY_KEYS:
            val = details.get(k, 0)
            res[k] = val
            total_p += val
        res["total_penalty_sum"] = total_p

    except Exception as e:
        res["error"] = str(e)
        # print(f"Error running {solver_name} on {instance_id}: {e}") # Debug print

    return res

def main():
    parser = argparse.ArgumentParser(description="VRP Benchmark Runner - Synced with run_experiment")
    parser.add_argument("--data_glob", required=True, help="Glob pattern for dataset folders (e.g. 'dataset/*')")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--time", type=float, default=30.0)
    parser.add_argument("--max_gen", type=int, default=500)
    parser.add_argument("--out", type=str, default="benchmark_summary.csv")
    parser.add_argument("--out_raw", type=str, default="benchmark_raw_details.csv")
    parser.add_argument("--evaluator", choices=["orig", "pd", "org", "with_weight"], default=None)
    args = parser.parse_args()

    # Find directories matching the glob pattern
    # Assuming user passes something like "dataset/Osaba*"
    # We look for directories inside the base path.
    # Adjust logic if passing exact paths.
    
    # Simple Glob expansion
    from glob import glob
    data_paths = sorted([Path(p) for p in glob(args.data_glob) if Path(p).is_dir()])
    
    if not data_paths:
        print(f"No data folders found matching: {args.data_glob}")
        return

    raw_results = []
    print(f"Found {len(data_paths)} instances. Solvers: {args.solvers}")

    for data_path in data_paths:
        for solver_name in args.solvers:
            if solver_name not in SOLVERS:
                print(f"Skipping unknown solver: {solver_name}")
                continue
                
            for seed in args.seeds:
                print(f">> Running {solver_name} on {data_path.name} (Seed {seed})...", end=" ", flush=True)
                res = run_single_instance(data_path, solver_name, seed, args.time, args.max_gen, args.evaluator)
                
                if res['ok']:
                    print(f"OK (Cost: {res['total_cost']:.2f})")
                else:
                    print(f"FAILED ({res['error']})")
                
                raw_results.append(res)

    df = pd.DataFrame(raw_results)
    df.to_csv(args.out_raw, index=False)

    df_success = df[df['ok'] == True].copy()
    if df_success.empty:
        print("No successful runs to summarize.")
        return

    # Configuration for Aggregation
    agg_dict = {
        'total_cost': ['min', 'mean', 'std'],
        'distance': 'mean',
        'fixed_cost': 'mean',
        'runtime': 'mean',
        'vehicles_used': 'mean',
        'total_penalty_sum': 'mean'
    }
    # Add specific penalties to aggregation if they exist in df
    for k in PENALTY_KEYS:
        if k in df_success.columns:
            agg_dict[k] = 'mean'

    summary = df_success.groupby(['instance', 'solver']).agg(agg_dict)
    
    # Flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Calculate Gap relative to best cost found for that instance (across all solvers/seeds)
    best_instance_cost = df_success.groupby('instance')['total_cost'].min().to_dict()
    summary['Gap_to_Best_%'] = summary.apply(
        lambda x: ((x['total_cost_min'] - best_instance_cost[x['instance']]) / best_instance_cost[x['instance']]) * 100 
        if best_instance_cost[x['instance']] != 0 else 0, 
        axis=1
    )

    summary = summary.round(2)
    summary.to_csv(args.out, index=False)
    print(f"\nBenchmark completed. Summary saved to {args.out}")
    print(f"Raw details saved to {args.out_raw}")

if __name__ == "__main__":
    main()