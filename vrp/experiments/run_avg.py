# vrp/experiments/run_avg.py
from __future__ import annotations
import argparse, csv, os, shlex, subprocess, sys, time, ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, pstdev

RUN_ONE = [sys.executable, "-m", "vrp.experiments.run_experiment"]  # dùng python hiện tại

def build_cmd_args(data: str, solver: str, seed: int, tlimit: float, plot: bool, config: str|None):
    args = RUN_ONE + ["--data", data, "--solver", solver, "--seed", str(seed), "--time", str(tlimit)]
    if config:
        args += ["--config", config]
    if plot:
        args += ["--plot"]
    return args

def run_once(cmd_args: list[str], cwd: str|None=None) -> dict:
    t0 = time.time()
    # Trên Windows: shell=False vẫn ổn vì dùng sys.executable -m
    proc = subprocess.run(cmd_args, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - t0

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    res = {
        "ok": proc.returncode == 0,
        "stdout": out,
        "stderr": err,
        "elapsed_wall": round(elapsed, 3),
        "total_cost": None,
        "details": None,
        "time_printed": None,
    }
    if not res["ok"]:
        return res

    # Parse các dòng in từ run_experiment.py
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("time:"):
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
            except Exception:
                pass
    return res

def expand_data_glob(data_glob: str) -> list[str]:
    paths = sorted(str(p) for p in Path().glob(data_glob))
    if not paths:
        raise SystemExit(f"[run_avg] Không tìm thấy instance nào với glob: {data_glob}")
    return paths

def summarize_costs(costs_ok: list[float]) -> dict:
    if not costs_ok:
        return {"avg": None, "std": None, "min": None, "max": None}
    if len(costs_ok) == 1:
        c = costs_ok[0]
        return {"avg": c, "std": 0.0, "min": c, "max": c}
    return {
        "avg": mean(costs_ok),
        "std": pstdev(costs_ok),  # std tổng thể (không phải mẫu)
        "min": min(costs_ok),
        "max": max(costs_ok),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True, help='Glob tới instance (vd: "dataset_test/*/base_50")')
    ap.add_argument("--solvers", nargs="+", default=["cluster_ga"],
                    choices=["dfa","ga","esa","dfa_pd","esa_pd","ga_hct_pd","cluster_ga"])
    ap.add_argument("--time", type=float, default=10.0, help="time limit mỗi run (giây)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=1, help="số luồng song song (1 = tuần tự)")
    ap.add_argument("--n_seeds", type=int, default=10, help="số seed cần chạy (mặc định 10)")
    ap.add_argument("--seed_start", type=int, default=42, help="seed bắt đầu (mặc định 42)")
    ap.add_argument("--out_csv", type=str, default="avg_results.csv", help="file CSV ghi tóm tắt")
    ap.add_argument("--out_raw", type=str, default=None, help="(tùy chọn) CSV ghi kết quả từng run")
    args = ap.parse_args()

    datasets = expand_data_glob(args.data_glob)
    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    out_rows_summary = []
    out_rows_raw = []

    # Chạy song song theo (dataset, solver, seed)
    jobs = max(1, args.jobs)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = []
        meta_map = {}
        for data in datasets:
            for solver in args.solvers:
                for seed in seeds:
                    cmd_args = build_cmd_args(data, solver, seed, args.time, args.plot, args.config)
                    fut = ex.submit(run_once, cmd_args)
                    meta = {"dataset": data, "solver": solver, "seed": seed, "cmd": " ".join(shlex.quote(x) for x in cmd_args)}
                    meta_map[fut] = meta
                    futs.append(fut)

        # Thu kết quả raw trước
        per_group_costs: dict[tuple[str,str], list[float]] = {}
        per_group_ok: dict[tuple[str,str], int] = {}
        per_group_runs: dict[tuple[str,str], int] = {}

        for fut in as_completed(futs):
            meta = meta_map[fut]
            res = fut.result()
            key = (meta["dataset"], meta["solver"])

            per_group_runs[key] = per_group_runs.get(key, 0) + 1
            if res.get("ok") and (res.get("total_cost") is not None):
                per_group_ok[key] = per_group_ok.get(key, 0) + 1
                per_group_costs.setdefault(key, []).append(res["total_cost"])
            else:
                per_group_ok[key] = per_group_ok.get(key, 0)

            # ghi raw row (nếu cần)
            out_rows_raw.append({
                "dataset": meta["dataset"],
                "solver": meta["solver"],
                "seed": meta["seed"],
                "cmd": meta["cmd"],
                "ok": res.get("ok", False),
                "total_cost": res.get("total_cost"),
                "elapsed_wall": res.get("elapsed_wall"),
                "time_printed": res.get("time_printed"),
                "stdout": res.get("stdout",""),
                "stderr": res.get("stderr",""),
            })

            print(f"[DONE] {meta['solver']} seed={meta['seed']} data={meta['dataset']} -> ok={res.get('ok')} cost={res.get('total_cost')}")

    # Tổng hợp trung bình cho mỗi (dataset, solver)
    for data in datasets:
        for solver in args.solvers:
            key = (data, solver)
            costs_ok = per_group_costs.get(key, [])
            stats = summarize_costs(costs_ok)
            n_run = per_group_runs.get(key, 0)
            n_ok = per_group_ok.get(key, 0)
            out_rows_summary.append({
                "dataset": data,
                "solver": solver,
                "n_runs": n_run,
                "n_ok": n_ok,
                "ok_rate": (n_ok / n_run) if n_run else None,
                "avg_cost": stats["avg"],
                "std_cost": stats["std"],
                "min_cost": stats["min"],
                "max_cost": stats["max"],
            })

    # Ghi CSV summary
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset","solver","n_runs","n_ok","ok_rate","avg_cost","std_cost","min_cost","max_cost"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows_summary:
            w.writerow(r)

    # (tùy chọn) ghi raw
    if args.out_raw:
        out_raw = Path(args.out_raw)
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        with out_raw.open("w", newline="", encoding="utf-8") as f:
            fields = ["dataset","solver","seed","cmd","ok","total_cost","elapsed_wall","time_printed","stdout","stderr"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in out_rows_raw:
                w.writerow(r)

    print(f"\n[run_avg] Đã lưu summary: {out_csv} (rows={len(out_rows_summary)})")
    if args.out_raw:
        print(f"[run_avg] Đã lưu raw: {args.out_raw} (rows={len(out_rows_raw)})")

if __name__ == "__main__":
    main()
