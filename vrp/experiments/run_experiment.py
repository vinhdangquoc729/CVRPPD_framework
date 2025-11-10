import argparse, yaml, time
from pathlib import Path
from ..data.loader import load_problem

from ..solvers.dfa import DFASolver
from ..solvers.ga_pd_hct import GAPD_HCT_Solver
from ..solvers.esa import ESASolver
from ..solvers.dfa_modified import DFASolverPD
from ..solvers.esa_modified import ESASolverPD
from ..solvers.cluster_ga import ClusterGASolver

from ..core.eval import evaluate
from ..core.eval_modified import evaluate_modified

from ..utils.visualize import draw_solution
from ..utils.reconstruct import reconstruct_with_refills

SOLVERS = {
    "dfa": DFASolver,
    "esa": ESASolver,
    "dfa_pd": DFASolverPD,
    "esa_pd": ESASolverPD,
    "ga_hct_pd": GAPD_HCT_Solver,
    "cluster_ga_pd": ClusterGASolver,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to instance folder (contains nodes.csv, vehicles.csv, ...)")
    ap.add_argument("--solver", choices=SOLVERS.keys(), default="dfa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time", type=float, default=30.0, help="time limit seconds (advisory)")
    ap.add_argument("--config", type=str, default=None, help="YAML config to pass into solver (e.g., pop_size, gamma)")
    ap.add_argument("--plot", action="store_true", help="Save a PNG visualization of the solution")
    ap.add_argument("--plot_path", type=str, default=None, help="Output path for PNG (default: <data>/solution.png)")
    ap.add_argument("--annotate", action="store_true", help="Annotate customer IDs on plot")

    # NEW: chọn evaluator
    ap.add_argument(
        "--evaluator",
        choices=["orig", "pd"],
        default=None,
        help="orig = evaluate (AC-VRP-SPDVCFP), pd = evaluate_modified (biến thể pickup/delivery). "
             "Nếu bỏ trống sẽ tự suy ra theo tên solver (_pd -> dùng pd)."
    )
    args = ap.parse_args()

    # chọn evaluator
    if args.evaluator is None:
        use_eval_modified = args.solver.endswith("_pd")
    else:
        use_eval_modified = (args.evaluator == "pd")
    EVAL = evaluate_modified if use_eval_modified else evaluate

    # load problem & config
    prob = load_problem(args.data)
    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    # Thiết lập solver
    SolverCls = SOLVERS[args.solver]
    solver = SolverCls(prob, seed=args.seed, **cfg)

    # Giải
    t0 = time.time()
    sol = solver.solve(time_limit_sec=args.time)
    elapsed = time.time() - t0
    cost, det = EVAL(prob, sol, return_details=True)

    # In kết quả
    print("solver:", args.solver)
    print("seed:", args.seed)
    print("time:", round(elapsed, 3), "s")
    print("total_cost:", cost)
    print("details:", det)
    print("solution:", sol)

    if args.plot:
        out = args.plot_path or str(Path(args.data) / f"solution_{args.solver}.png")
        viz_sol = reconstruct_with_refills(prob, sol)
        draw_solution(prob, viz_sol, save_path=out, show=False, annotate=args.annotate)
        print(f"Saved plot to: {out}")

if __name__ == "__main__":
    main()
