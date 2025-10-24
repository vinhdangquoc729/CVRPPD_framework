# vrp/experiments/run_experiment_logged.py
from __future__ import annotations
import argparse, time, json
from pathlib import Path
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

from ..data.loader import load_problem
from ..core.eval import evaluate
from ..core.solution import Solution
from ..utils.visualize import draw_solution
from ..utils.reconstruct import reconstruct_with_refills

# ---- import solvers gốc ----
from ..solvers.dfa import DFASolver as _DFASolver
from ..solvers.ga import GASolver as _GASolver

# ---------- Progress data structure ----------
class ProgressLog:
    def __init__(self):
        self.iters: List[int] = []
        self.times: List[float] = []
        self.costs: List[float] = []
        self.solutions: List[Solution] = []

    def record(self, it: int, t: float, cost: float, sol: Solution):
        self.iters.append(it)
        self.times.append(t)
        self.costs.append(cost)
        self.solutions.append(sol)

# ---------- DFASolver có logging (kế thừa từ bản hiện có) ----------
class LoggedDFASolver(_DFASolver):
    def solve(self,
              time_limit_sec: float = 30.0,
              patience_iters: int = 20,
              on_progress=None) -> Solution:
        import time as _time

        P = self.problem
        pop = self._init_population()

        # --- cache cost để tránh evaluate trùng ---
        cost_cache = {}
        def cost_of(s: Solution) -> float:
            key = tuple(tuple(r.seq) for r in s.routes)
            c = cost_cache.get(key)
            if c is None:
                c, _ = evaluate(P, s, return_details=False)
                cost_cache[key] = c
            return c

        # khởi tạo
        pop.sort(key=lambda s: -cost_of(s))   # brightness = -cost
        best = pop[0]
        best_cost = cost_of(best)

        it = 0
        no_improve = 0
        EPS = 1e-9
        t0 = _time.time()

        # log ban đầu
        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best)

        while it < 200 and (_time.time() - t0) < time_limit_sec:
            new_pop = []
            for i in range(len(pop)):
                xi = pop[i]
                ci = cost_of(xi)

                for j in range(len(pop)):
                    if j == i:
                        continue
                    xj = pop[j]
                    if -cost_of(xj) > -ci:
                        rij = self._hamming_by_cluster(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** it)))
                        trials = []
                        for _ in range(max(2, step_max)):
                            cand = self._insertion_move(xi)
                            trials.append(cand)
                        # chọn ứng viên tốt nhất theo cost
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                new_pop.append(xi)

            # chọn top pop_size
            pop = sorted(new_pop, key=cost_of)[:self.pop_size]

            # cập nhật best + patience
            cur_cost = cost_of(pop[0])
            if cur_cost + EPS < best_cost:
                best = pop[0]
                best_cost = cur_cost
                no_improve = 0
            else:
                no_improve += 1

            it += 1

            # callback tiến trình
            if on_progress is not None:
                elapsed = _time.time() - t0
                on_progress(it, elapsed, best_cost, best)

            # early stop theo kiên nhẫn
            if patience_iters is not None and patience_iters > 0 and no_improve >= patience_iters:
                break

            # dừng theo thời gian (phòng khi vòng lặp lâu)
            if (_time.time() - t0) >= time_limit_sec:
                break

        return best

# ---------- GA có logging (tuỳ chọn: nếu GA của bạn đã hỗ trợ callback thì dùng luôn) ----------
class LoggedGASolver(_GASolver):
    def solve(self, time_limit_sec: float = 30.0,
              on_progress: Callable[[int, float, float, Solution], None] | None = None) -> Solution:
        # Nếu GA của bạn có vòng lặp rõ, thêm hook tương tự như DFA ở trên.
        # Tạm thời fallback về solve gốc và chỉ log final.
        sol = super().solve(time_limit_sec=time_limit_sec)
        if on_progress is not None:
            on_progress(0, 0.0, evaluate(self.problem, sol, False)[0], sol)
        return sol

SOLVERS = {
    "dfa": LoggedDFASolver,
    "ga": LoggedGASolver,
}

# ---------- Plot helper ----------
def plot_cost_curves(out_png: str, iters: List[int], times: List[float], costs: List[float]):
    # vẽ 2 đồ thị (cost vs. iter) và (cost vs. time) xếp dọc riêng rẽ
    # theo yêu cầu: mỗi chart riêng - không subplots
    # Chart 1: cost vs iter
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=140)
    ax1.plot(iters, costs, linewidth=1.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best cost")
    ax1.set_title("Best cost vs. iteration")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fn1 = Path(out_png).with_name(Path(out_png).stem + "_cost_iter.png")
    fig1.savefig(fn1, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: cost vs time
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=140)
    ax2.plot(times, costs, linewidth=1.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Best cost")
    ax2.set_title("Best cost vs. time")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fn2 = Path(out_png).with_name(Path(out_png).stem + "_cost_time.png")
    fig2.savefig(fn2, bbox_inches="tight")
    plt.close(fig2)

    return str(fn1), str(fn2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path tới instance folder")
    ap.add_argument("--solver", choices=SOLVERS.keys(), default="dfa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--config", type=str, default=None, help="YAML config (pop_size, gamma, ...)")
    ap.add_argument("--snapshot_every", type=int, default=20,
                    help="số vòng lặp giữa 2 lần chụp ảnh lời giải")
    ap.add_argument("--snapshot_dir", type=str, default=None,
                    help="thư mục lưu ảnh snapshot (mặc định: <data>/snapshots)")
    ap.add_argument("--plot_final", action="store_true",
                    help="lưu ảnh solution cuối cùng (solution.png)")
    ap.add_argument("--annotate", action="store_true",
                    help="hiện id khách trên ảnh")
    args = ap.parse_args()

    # load data
    prob = load_problem(args.data)

    # load config (nếu có)
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    # chuẩn bị solver có logging
    SolverCls = SOLVERS[args.solver]
    solver = SolverCls(prob, seed=args.seed, **cfg)

    # logger
    log = ProgressLog()
    out_dir = Path(args.snapshot_dir or (Path(args.data) / "snapshots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(iter_idx: int, elapsed_sec: float, best_cost: float, best_sol: Solution):
        # ghi log
        log.record(iter_idx, elapsed_sec, best_cost, best_sol)

        # chụp snapshot mỗi N vòng lặp
        if args.snapshot_every > 0 and (iter_idx % args.snapshot_every == 0):
            viz_sol = reconstruct_with_refills(prob, best_sol)
            png_path = out_dir / f"sol_iter_{iter_idx:04d}.png"
            draw_solution(prob, viz_sol, save_path=str(png_path), show=False, annotate=args.annotate)
            print(f"[snapshot] iter={iter_idx} cost={best_cost:.3f} -> {png_path}")

    # chạy solver
    t0 = time.time()
    best_sol = solver.solve(time_limit_sec=args.time, on_progress=on_progress, patience_iters=20)

    elapsed = time.time() - t0
    total_cost, details = evaluate(prob, best_sol, return_details=True)

    # in kết quả
    print("solver:", args.solver)
    print("seed:", args.seed)
    print("time:", round(elapsed, 3), "s")
    print("total_cost:", total_cost)
    print("details:", details)
    print("solution:", best_sol)

    # plot solution cuối (tuỳ chọn)
    if args.plot_final:
        final_png = Path(args.data) / "solution.png"
        viz_sol = reconstruct_with_refills(prob, best_sol)
        draw_solution(prob, viz_sol, save_path=str(final_png), show=False, annotate=args.annotate)
        print(f"Saved final plot to: {final_png}")

    # vẽ đồ thị cost
    curve_png_base = out_dir / "progress.png"
    p1, p2 = plot_cost_curves(str(curve_png_base), log.iters, log.times, log.costs)
    print(f"Saved cost curves to: {p1} and {p2}")

    # lưu raw log để tái sử dụng
    log_path = out_dir / "progress.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "iters": log.iters,
            "times": log.times,
            "costs": log.costs,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved progress log to: {log_path}")

if __name__ == "__main__":
    main()
