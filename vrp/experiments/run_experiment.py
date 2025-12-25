import argparse, time
from pathlib import Path
import yaml

from ..data.loader import load_problem
from ..data.loader_modified import load_problem_modified

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

from ..core.eval import evaluate
from ..core.eval_modified import evaluate_modified
from ..core.eval_org import evaluate as evaluate_org
from ..core.eval_with_weight import evaluate_with_weight

from ..utils.visualize import draw_solution
from ..utils.reconstruct import reconstruct_with_refills

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to instance folder (contains nodes.csv, vehicles.csv, ...)")
    ap.add_argument("--solver", choices=SOLVERS.keys(), default="dfa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time", type=float, default=30.0, help="time limit seconds (advisory)")
    ap.add_argument("--config", type=str, default=None, help="YAML to pass into solver (e.g., pop_size, gamma)")
    ap.add_argument("--plot", action="store_true", help="Save a PNG visualization of the solution")
    ap.add_argument("--plot_path", type=str, default=None, help="Output path for PNG (default: <data>/solution_<solver>.png)")
    ap.add_argument("--annotate", action="store_true", help="Annotate customer IDs on plot")
    ap.add_argument(
        "--evaluator",
        choices=["orig", "pd", "org", "with_weight"],
        default=None,
        help="orig = evaluate (AC-VRP-SPDVCFP), pd = evaluate_modified (VRPPD với precedence), org = evaluate_org (original). "
             "Nếu bỏ trống sẽ tự suy theo tên solver (_pd => pd)."
    )
    args = ap.parse_args()

    # Determine which evaluator to use
    if args.evaluator == "org":
        EVAL = evaluate_org
        prob = load_problem(args.data)
        chosen_loader = "org"
    elif args.evaluator == "pd" or (args.evaluator is None and args.solver.endswith("_pd")):
        EVAL = evaluate_modified
        prob = load_problem_modified(args.data)
        chosen_loader = "modified"
    elif args.evaluator == "orig":
        EVAL = evaluate
        prob = load_problem(args.data)
        chosen_loader = "orig"
    else:
        EVAL = evaluate_with_weight
        prob = load_problem(args.data)
        chosen_loader = "with_weight"

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    SolverCls = SOLVERS[args.solver]
    solver = SolverCls(prob, seed=args.seed, evaluator=EVAL, **cfg)

    t0 = time.time()
    sol = solver.solve(time_limit_sec=args.time)
    elapsed = time.time() - t0
    cost, det = EVAL(prob, sol, return_details=True)

    print("data_dir:", str(Path(args.data).resolve()))
    print("loader:", chosen_loader)
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
