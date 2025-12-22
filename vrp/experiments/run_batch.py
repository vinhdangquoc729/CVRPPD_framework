# vrp/experiments/run_batch.py
from __future__ import annotations
import argparse, csv, subprocess, time, ast, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

RUN_ONE: List[str] = [sys.executable, "-m", "vrp.experiments.run_experiment"]  # dùng python hiện tại

SOLVER_CHOICES = [
    "dfa", "esa",
    "dfa_pd", "esa_pd",
    "ga_hct_pd", "ga_pd_hct",
    "cluster_ga", "cluster_ga_pd",
    "ga_ombuki",
]

def build_cmd_args(
    data: str, solver: str, seed: int, tlimit: float,
    plot: bool, config: Optional[str],
    annotate: bool, plot_path: Optional[str], evaluator: Optional[str]
) -> List[str]:
    args = RUN_ONE + ["--data", data, "--solver", solver, "--seed", str(seed), "--time", str(tlimit)]
    if config:
        args += ["--config", config]
    if evaluator:
        args += ["--evaluator", evaluator]
    if plot:
        args += ["--plot"]
        if annotate:
            args += ["--annotate"]
        if plot_path:
            args += ["--plot_path", plot_path]
    return args

def run_once(cmd_args: List[str], cwd: Optional[str]=None) -> dict:
    """Chạy 1 experiment, parse stdout về dict kết quả."""
    t0 = time.time()
    proc = subprocess.run(cmd_args, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - t0

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    res = {
        "ok": proc.returncode == 0,
        "stdout": out,
        "stderr": err,
        "elapsed_wall": round(elapsed, 3),
        "returncode": proc.returncode,
        "cmd_pretty": " ".join(cmd_args),
    }
    if not res["ok"]:
        return res

    # Parse các dòng in từ run_experiment.py
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("solver:"):
            res["solver"] = line.split("solver:", 1)[1].strip()
        elif line.startswith("solution:"):
            res["solution"] = line.split("solution:", 1)[1].strip()
        elif line.startswith("seed:"):
            try:
                res["seed"] = int(line.split("seed:", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("time:"):
            # "time: 5.123 s"
            try:
                val = line.split("time:", 1)[1].strip().split()[0]
                res["time_printed"] = float(val)
            except Exception:
                pass
        elif line.startswith("total_cost:"):
            try:
                res["total_cost"] = float(line.split("total_cost:", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("details:"):
            try:
                det_str = line.split("details:", 1)[1].strip()
                res["details"] = ast.literal_eval(det_str)
            except Exception as e:
                res["details_parse_error"] = str(e)
    return res

def expand_data_glob(data_glob: str) -> List[str]:
    paths = sorted(str(p) for p in Path().glob(data_glob))
    if not paths:
        raise SystemExit(f"[run_batch] Không tìm thấy instance nào với glob: {data_glob}")
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True,
                    help='Glob tới thư mục instance (vd: "dataset_test/*/base_50")')
    ap.add_argument("--solvers", nargs="+", default=["dfa"],
                    choices=SOLVER_CHOICES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--annotate", action="store_true", help="Kèm nhãn ID khách trên hình")
    ap.add_argument("--plot_path", type=str, default=None, help="Ghi đè đường dẫn PNG")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--evaluator", choices=["orig", "pd"], default=None,
                    help="orig=evaluate, pd=evaluate_modified; mặc định suy theo tên solver")
    ap.add_argument("--jobs", type=int, default=1, help="số luồng song song (1 = tuần tự)")
    ap.add_argument("--out", type=str, default="batch_results.csv")
    ap.add_argument("--keep_logs", action="store_true", help="Lưu stdout/stderr vào CSV")
    args = ap.parse_args()

    datasets = expand_data_glob(args.data_glob)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "dataset","solver","seed","cmd","ok","return_total_cost","elapsed_wall","returncode"
    ]
    rows = []

    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for data in datasets:
            for solver in args.solvers:
                for seed in args.seeds:
                    cmd_args = build_cmd_args(
                        data=data,
                        solver=solver,
                        seed=seed,
                        tlimit=args.time,
                        plot=args.plot,
                        config=args.config,
                        annotate=args.annotate,
                        plot_path=args.plot_path,
                        evaluator=args.evaluator
                    )
                    fut = ex.submit(run_once, cmd_args)
                    fut._meta = {
                        "dataset": data, "solver": solver, "seed": seed,
                        "cmd": " ".join(cmd_args)
                    }
                    tasks.append(fut)

        for fut in as_completed(tasks):
            meta = fut._meta
            res = fut.result()
            row = {
                "dataset": meta["dataset"],
                "solver": meta["solver"],
                "seed": meta["seed"],
                "cmd": meta["cmd"],
                "ok": res.get("ok", False),
                "return_total_cost": res.get("total_cost"),
                "elapsed_wall": res.get("elapsed_wall"),
                "returncode": res.get("returncode"),
                "solution": res.get("solution")
            }
            det = res.get("details", {})
            for k, v in det.items():
                row[k] = v

            if args.keep_logs:
                row["stdout"] = res.get("stdout","")
                row["stderr"] = res.get("stderr","")

            rows.append(row)
            print(f"[DONE] {meta['solver']} seed={meta['seed']} data={meta['dataset']} "
                  f"-> ok={row['ok']} cost={row['return_total_cost']}")

    # Thu thập cột
    all_fields = list(base_fields)
    extra_keys = set(k for r in rows for k in r.keys() if k not in all_fields)
    all_fields += sorted(extra_keys)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[run_batch] Đã lưu kết quả: {out_path}  (n={len(rows)})")

if __name__ == "__main__":
    main()
