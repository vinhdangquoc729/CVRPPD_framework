import argparse, yaml, time
from pathlib import Path
from ..data.loader import load_problem
# from ..solvers.greedy_baseline import GreedySolver
from ..solvers.dfa import DFASolver
from ..solvers.ga import GASolver
from ..solvers.esa import ESASolver
from ..core.eval import evaluate
from ..utils.visualize import draw_solution  
from ..utils.reconstruct import reconstruct_with_refills

SOLVERS = {
    "dfa": DFASolver,
    "ga": GASolver,
    "esa": ESASolver, 
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to instance folder (contains nodes.csv, vehicles.csv, ...)")
    ap.add_argument("--solver", choices=SOLVERS.keys(), default="greedy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time", type=float, default=30.0, help="time limit seconds (advisory)")
    ap.add_argument("--config", type=str, default=None, help="YAML config to pass into solver (e.g., pop_size, gamma)")
    ap.add_argument("--plot", action="store_true", help="Save a PNG visualization of the solution")
    ap.add_argument("--plot_path", type=str, default=None, help="Output path for PNG (default: <data>/solution.png)")
    ap.add_argument("--annotate", action="store_true", help="Annotate customer IDs on plot")
    args = ap.parse_args()

    prob = load_problem(args.data)
    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())

    SolverCls = SOLVERS[args.solver]
    solver = SolverCls(prob, seed=args.seed, **cfg)

    t0 = time.time()
    sol = solver.solve(time_limit_sec=args.time)
    elapsed = time.time() - t0
    cost, det = evaluate(prob, sol, return_details=True)

    print("solver:", args.solver)
    print("seed:", args.seed)
    print("time:", round(elapsed, 3), "s")
    print("total_cost:", cost)
    print("details:", det)
    print("solution:",sol)
    
    if args.plot:
        out = args.plot_path or str(Path(args.data) / "solution.png")
        viz_sol = reconstruct_with_refills(prob, sol)
        draw_solution(prob, viz_sol, save_path=out, show=False, annotate=args.annotate) 
        print(f"Saved plot to: {out}")

if __name__ == "__main__":
    main()