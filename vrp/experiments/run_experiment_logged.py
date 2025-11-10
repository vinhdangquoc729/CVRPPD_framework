from __future__ import annotations
import argparse, time, json
from pathlib import Path
from typing import Callable, List, Tuple, Dict
import matplotlib.pyplot as plt

from ..data.loader import load_problem
from ..core.eval import evaluate
from ..core.eval_modified import evaluate_modified 
from ..core.solution import Solution
from ..utils.visualize import draw_solution
from ..utils.reconstruct import reconstruct_with_refills

# ---- import solvers gốc ----
from ..solvers.dfa import DFASolver as _DFASolver
from ..solvers.ga_pd import GAPDSolver as _GASolver
from ..solvers.dfa_modified import DFASolverPD
from ..solvers.esa_modified import ESASolverPD
from ..solvers.ga_pd_hct import GAPD_HCT_Solver as _GAHCTSolver, EncodedSolution, decode_to_solution
from ..solvers.cluster_ga import ClusterGASolver as _ClusterGASolver

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
class LoggedDFAPDSolver(DFASolverPD):
    """
    DFASolverPD có ghi log tiến trình qua callback on_progress(iter, elapsed, best_cost, best_solution).
    - brightness = -cost (cost do evaluate_modified trả về)
    - cache chi phí để tránh evaluate trùng
    - dừng sớm nếu không cải thiện 'patience_iters' vòng
    - bổ sung _random_neighbor và _hamming_like cho biến thể PD (không cụm)
    """

    # ---- helpers cho PD (không cụm) ----
    def _flatten_customers(self, s: Solution) -> list[int]:
        P = self.problem
        out = []
        for r in s.routes:
            out.extend([i for i in r.seq if not P.nodes[i].is_depot])
        return out

    def _hamming_like(self, P, a: Solution, b: Solution) -> int:
        A, B = self._flatten_customers(a), self._flatten_customers(b)
        L = min(len(A), len(B))
        return sum(1 for i in range(L) if A[i] != B[i]) + abs(len(A) - len(B))

    def _random_neighbor(self, s: Solution) -> None:
        """
        Sinh láng giềng cấp-khách cho PD:
        - intra-route relocate: lấy 1 khách, chèn vị trí khác trong cùng route
        - intra-route swap: hoán vị 2 khách trong cùng route
        - inter-route relocate: chuyển 1 khách từ route A sang B
        - inter-route swap: hoán vị 1-1 giữa A và B
        - 2-opt trong 1 route
        Chỉ chỉnh sửa 's' tại chỗ.
        """
        import random
        P = self.problem
        rng = random

        # chọn các route có >= 1 customer
        routes = [r for r in s.routes if sum(1 for x in r.seq if not P.nodes[x].is_depot) > 0]
        if not routes:
            return

        def customer_positions(r):
            """Trả về danh sách index trong r.seq tương ứng với KHÁCH (bỏ depot đầu/cuối)."""
            return [idx for idx, x in enumerate(r.seq) if not P.nodes[x].is_depot]

        move = rng.choice(["intra_reloc", "intra_swap", "inter_reloc", "inter_swap", "two_opt"])

        if move in ("intra_reloc", "intra_swap", "two_opt"):
            r = rng.choice(routes)
            pos = customer_positions(r)
            if len(pos) < 2:
                return

            if move == "intra_reloc":
                i = rng.choice(pos)
                node = r.seq.pop(i)
                pos2 = customer_positions(r)  # cập nhật sau pop
                j = rng.choice(pos2 + [pos2[-1] + 1])  # cho phép chèn cuối vùng khách
                r.seq.insert(j, node)

            elif move == "intra_swap":
                i, j = rng.sample(pos, 2)
                r.seq[i], r.seq[j] = r.seq[j], r.seq[i]

            elif move == "two_opt":
                # 2-opt chỉ trong phần khách của route (không đảo depot)
                i, j = sorted(rng.sample(pos, 2))
                r.seq[i:j+1] = reversed(r.seq[i:j+1])

        elif move == "inter_reloc":
            if len(routes) < 2:
                return
            r_src, r_dst = rng.sample(routes, 2)
            pos_src = customer_positions(r_src)
            if not pos_src:
                return
            i = rng.choice(pos_src)
            node = r_src.seq.pop(i)

            # chèn vào r_dst ở vị trí khách bất kỳ
            pos_dst = customer_positions(r_dst)
            insert_pos = rng.choice(pos_dst + [pos_dst[-1] + 1]) if pos_dst else 1  # nếu route trống khách, chèn sau depot
            r_dst.seq.insert(insert_pos, node)

        elif move == "inter_swap":
            if len(routes) < 2:
                return
            r_a, r_b = rng.sample(routes, 2)
            pos_a = customer_positions(r_a)
            pos_b = customer_positions(r_b)
            if not pos_a or not pos_b:
                return
            ia = rng.choice(pos_a); ib = rng.choice(pos_b)
            r_a.seq[ia], r_b.seq[ib] = r_b.seq[ib], r_a.seq[ia]

    # ---- solve có logging ----
    def solve(self,
              time_limit_sec: float = 30.0,
              patience_iters: int = 20,
              on_progress=None) -> Solution:

        import time as _time
        P = self.problem
        pop = self._init_population()

        # cache cost để tránh evaluate trùng
        cost_cache: Dict[tuple, float] = {}
        def key_of(s: Solution):
            return tuple(tuple(r.seq) for r in s.routes)
        def cost_of(s: Solution) -> float:
            k = key_of(s)
            c = cost_cache.get(k)
            if c is None:
                c, _ = evaluate_modified(P, s, return_details=False)
                cost_cache[k] = c
            return c

        pop.sort(key=cost_of)  # cost tăng dần
        best = pop[0]; best_cost = cost_of(best)

        it = 0
        no_improve = 0
        EPS = 1e-9
        t0 = _time.time()

        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best)

        while (_time.time() - t0) < time_limit_sec:
            new_pop: List[Solution] = []

            for i in range(len(pop)):
                xi = pop[i]
                ci = cost_of(xi)

                # bay về nghiệm sáng hơn
                for j in range(len(pop)):
                    if j == i: 
                        continue
                    xj = pop[j]
                    if cost_of(xj) + EPS < ci:
                        rij = self._hamming_like(P, xi, xj)
                        step_max = max(2, int(rij * (self.gamma ** it)))
                        # thử nhiều láng giềng và lấy tốt nhất
                        import copy
                        trials: List[Solution] = []
                        for _ in range(max(2, step_max)):
                            cand = copy.deepcopy(xi)
                            self._random_neighbor(cand)
                            trials.append(cand)
                        xi = min(trials, key=cost_of)
                        ci = cost_of(xi)

                new_pop.append(xi)

            pop = sorted(new_pop, key=cost_of)[:self.pop_size]

            cur_best = pop[0]; cur_cost = cost_of(cur_best)
            if cur_cost + EPS < best_cost:
                best, best_cost = cur_best, cur_cost
                no_improve = 0
            else:
                no_improve += 1

            it += 1
            if on_progress is not None:
                elapsed = _time.time() - t0
                on_progress(it, elapsed, best_cost, best)

            if patience_iters is not None and patience_iters > 0 and no_improve >= patience_iters:
                break
            if (_time.time() - t0) >= time_limit_sec:
                break

        return best

class LoggedESAPDSolver(ESASolverPD):
    """
    ESASolverPD có ghi log tiến trình qua on_progress(iter, elapsed, best_cost, best_solution).
    - Dùng evaluate_modified cho bài toán PD.
    - Cache cost để tránh evaluate trùng.
    - Early stop nếu không cải thiện trong 'patience_iters' vòng.
    """
    def _neighbor(self, sol: Solution) -> Solution:
        """
        Sinh láng giềng cấp-khách cho PD (trả về một bản sao đã thay đổi):
        - intra-route relocate: lấy 1 khách, chèn vị trí khác trong cùng route
        - intra-route swap: hoán vị 2 khách trong cùng route
        - inter-route relocate: chuyển 1 khách từ route A sang B
        - inter-route swap: hoán vị 1-1 giữa A và B
        - 2-opt trong 1 route (trên đoạn khách)
        """
        import copy, random
        P = self.problem
        s = copy.deepcopy(sol)

        # chọn các route có >= 1 khách
        routes = [r for r in s.routes if any(not P.nodes[x].is_depot for x in r.seq)]
        if not routes:
            return s

        def customer_positions(r):
            # chỉ số trong r.seq ứng với KHÁCH (bỏ depot)
            return [idx for idx, x in enumerate(r.seq) if not P.nodes[x].is_depot]

        move = random.choice(["intra_reloc", "intra_swap", "inter_reloc", "inter_swap", "two_opt"])

        if move in ("intra_reloc", "intra_swap", "two_opt"):
            r = random.choice(routes)
            pos = customer_positions(r)
            if len(pos) < 2:
                return s

            if move == "intra_reloc":
                i = random.choice(pos)
                node = r.seq.pop(i)
                pos2 = customer_positions(r)  # cập nhật sau pop
                if not pos2:
                    # route trở thành rỗng khách -> chèn lại ngay sau depot
                    r.seq.insert(1, node)
                else:
                    # cho phép chèn ở cuối vùng khách
                    j = random.choice(pos2 + [pos2[-1] + 1])
                    r.seq.insert(j, node)

            elif move == "intra_swap":
                i, j = random.sample(pos, 2)
                r.seq[i], r.seq[j] = r.seq[j], r.seq[i]

            elif move == "two_opt":
                i, j = sorted(random.sample(pos, 2))
                r.seq[i:j+1] = reversed(r.seq[i:j+1])

        elif move == "inter_reloc":
            if len(routes) < 2:
                return s
            r_src, r_dst = random.sample(routes, 2)
            pos_src = customer_positions(r_src)
            i = random.choice(pos_src) if pos_src else None
            if i is None:
                return s
            node = r_src.seq.pop(i)

            pos_dst = customer_positions(r_dst)
            insert_pos = (random.choice(pos_dst + [pos_dst[-1] + 1]) if pos_dst else 1)
            r_dst.seq.insert(insert_pos, node)

        elif move == "inter_swap":
            if len(routes) < 2:
                return s
            r_a, r_b = random.sample(routes, 2)
            pos_a = customer_positions(r_a)
            pos_b = customer_positions(r_b)
            if not pos_a or not pos_b:
                return s
            ia = random.choice(pos_a); ib = random.choice(pos_b)
            r_a.seq[ia], r_b.seq[ib] = r_b.seq[ib], r_a.seq[ia]

        return s

    def solve(self,
              time_limit_sec: float = 30.0,
              on_progress=None,
              patience_iters: int = 50) -> Solution:
        import time as _time, math

        P = self.problem
        r = self.rng
        t0 = _time.time()

        # --- khởi tạo quần thể ESA gốc ---
        pop = self._init_population()

        # --- cache chi phí ---
        cost_cache: Dict[tuple, float] = {}
        def key_of(s: Solution):
            return tuple(tuple(rt.seq) for rt in s.routes)

        def cost_of(s: Solution) -> float:
            k = key_of(s)
            v = cost_cache.get(k)
            if v is None:
                v, _ = evaluate_modified(P, s, return_details=False)
                cost_cache[k] = v
            return v

        # sắp xếp theo cost tăng
        pop.sort(key=cost_of)
        best = pop[0]
        best_cost = cost_of(best)

        # nhiệt độ khởi tạo như ESA gốc (dựa trên spread)
        costs0 = [cost_of(s) for s in pop]
        spread = (max(costs0) - min(costs0)) if len(costs0) > 1 else max(1.0, abs(costs0[0]))
        # tham số T bắt đầu: giống công thức trong ESA mẫu (-Δ/ln(p)), p=0.95
        T = spread / max(1.0, math.log(1/0.95))

        patience = 0
        it = 0

        # log ban đầu
        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best)

        while (_time.time() - t0) < time_limit_sec:
            it += 1

            # elitism
            elite_k = max(1, int(self.elite_frac * self.mu))
            elites = pop[:elite_k]
            new_pop: List[Solution] = list(elites)

            # sinh thêm cá thể qua tournament + SA acceptance
            while len(new_pop) < self.mu:
                # tournament nhỏ
                a, b = r.sample(pop, 2)
                parent = a if cost_of(a) <= cost_of(b) else b
                child = parent

                # nhiều bước láng giềng với SA
                for _ in range(self.trials_per_iter):
                    cand = self._neighbor(child)  # _neighbor trả về bản sao đã thay đổi
                    dE = cost_of(cand) - cost_of(child)
                    if dE <= 0 or r.random() < math.exp(-dE / max(1e-9, T)):
                        child = cand

                new_pop.append(child)

            # chọn lại quần thể theo cost
            pop = sorted(new_pop, key=cost_of)
            cur_best = pop[0]
            cur_cost = cost_of(cur_best)

            # cập nhật best + patience
            if cur_cost + 1e-9 < best_cost:
                best, best_cost = cur_best, cur_cost
                patience = 0
            else:
                patience += 1
                if patience_iters is not None and patience >= patience_iters:
                    # callback lần cuối trước khi thoát
                    if on_progress is not None:
                        on_progress(it, _time.time() - t0, best_cost, best)
                    break

            # callback tiến trình
            if on_progress is not None:
                on_progress(it, _time.time() - t0, best_cost, best)

            # làm nguội
            T *= self.alpha
            if T < 1e-6:
                T = 1e-6

            # kiểm tra thời gian
            if (_time.time() - t0) >= time_limit_sec:
                break

        return best
class LoggedGAHCTPDSolver(_GAHCTSolver):
    """
    GA Head–Core–Tail (PD) có logging và tương thích với run_experiment_logged (nhận patience_iters).
    """
    def solve(self,
              time_limit_sec: float = 60.0,
              on_progress=None,
              patience_iters: int | None = None,
              max_generations:   int | None = None) -> Solution:
        import time as _time, copy

        # dùng cấu hình mặc định từ solver nếu caller không override
        local_patience = self.patience if patience_iters is None else patience_iters
        local_max_gen  = self.max_generations if max_generations is None else max_generations

        t0 = _time.time()
        rng = self.rng

        # --- init population (HCT guided) ---
        pop = self._init_population()
        costs = [self._decode_cost(ind) for ind in pop]
        best_idx = min(range(len(pop)), key=lambda i: costs[i])
        best_enc = copy.deepcopy(pop[best_idx])
        best_cost = costs[best_idx]
        best_sol  = decode_to_solution(self.problem, best_enc)

        patience = 0
        gen = 0

        # log ban đầu
        if on_progress is not None:
            on_progress(0, 0.0, best_cost, best_sol)

        while (_time.time() - t0) < time_limit_sec and gen < local_max_gen and patience < local_patience:
            gen += 1

            # --- elitism ---
            elite_k = max(1, int(self.elite_frac * self.pop_size))
            elites_idx = sorted(range(len(pop)), key=lambda i: costs[i])[:elite_k]
            new_pop: list[EncodedSolution] = [copy.deepcopy(pop[i]) for i in elites_idx]

            # --- fill the rest via tournament + crossover + mutation ---
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament(pop, costs)
                parent2 = self._tournament(pop, costs)

                # crossover
                if rng.random() < self.p_cx:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # mutation
                if rng.random() < self.p_mut:
                    self._mutation(child1)
                if rng.random() < self.p_mut and len(new_pop) + 1 < self.pop_size:
                    self._mutation(child2)

                # repair uniqueness
                self._repair_uniqueness(child1)
                self._repair_uniqueness(child2)

                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(child2)

            pop = new_pop
            costs = [self._decode_cost(ind) for ind in pop]

            # --- update best ---
            cur_idx = min(range(len(pop)), key=lambda i: costs[i])
            cur_cost = costs[cur_idx]
            if cur_cost + 1e-9 < best_cost:
                best_cost = cur_cost
                best_enc  = copy.deepcopy(pop[cur_idx])
                best_sol  = decode_to_solution(self.problem, best_enc)
                patience  = 0
            else:
                patience += 1

            # --- logging ---
            if on_progress is not None:
                on_progress(gen, _time.time() - t0, best_cost, best_sol)

        return best_sol

class LoggedClusterGASolver(_ClusterGASolver):
    """
    Cluster-GA (PD) có logging tương thích run_experiment_logged.
    Solver gốc chưa có on_progress, nên ở đây log kết quả cuối.
    """
    def solve(self,
              time_limit_sec: float = 60.0,
              on_progress=None,
              patience_iters: int | None = None,
              max_generations: int | None = None) -> Solution:
        # Map tham số để tương thích signature solver gốc
        kwargs = {}
        if max_generations is not None:
            kwargs["max_generations"] = max_generations
        if patience_iters is not None:
            kwargs["patience_gens"] = patience_iters

        sol = super().solve(time_limit_sec=time_limit_sec, **kwargs)
        if on_progress is not None:
            # evaluate_modified vì bài PD
            cost, _ = evaluate_modified(self.problem, sol, return_details=False)
            on_progress(0, 0.0, cost, sol)
        return sol


SOLVERS = {
    "dfa": LoggedDFASolver,
    "ga": LoggedGASolver,
    "dfa_pd": LoggedDFAPDSolver,
    "esa_pd": LoggedESAPDSolver,   
    "ga_hct_pd": LoggedGAHCTPDSolver, 
    "cluster_ga": LoggedClusterGASolver,
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
    ap.add_argument("--evaluator", choices=["orig","pd"], default=None,
                help="Chọn evaluator: orig = evaluate (cũ), pd = evaluate_modified. Mặc định tự suy ra từ solver.")
    
    args = ap.parse_args()
    if args.evaluator is None:
        use_eval_modified = args.solver in ("dfa_pd", "esa_pd", "ga_hct_pd", "cluster_ga")
    else:
        use_eval_modified = (args.evaluator == "pd")

    EVAL = evaluate_modified if use_eval_modified else evaluate

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
            viz_sol = best_sol if use_eval_modified else reconstruct_with_refills(prob, best_sol)
            png_path = out_dir / f"sol_iter_{iter_idx:04d}.png"
            draw_solution(prob, viz_sol, save_path=str(png_path), show=False, annotate=args.annotate)
            print(f"[snapshot] iter={iter_idx} cost={best_cost:.3f} -> {png_path}")

    # chạy solver
    t0 = time.time()
    best_sol = solver.solve(time_limit_sec=args.time, on_progress=on_progress, patience_iters=20)

    elapsed = time.time() - t0
    total_cost, details = EVAL(prob, best_sol, return_details=True)

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
