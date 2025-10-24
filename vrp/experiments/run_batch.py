# vrp/experiments/run_batch.py
from __future__ import annotations
import argparse, csv, os, sys, subprocess, shlex, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import sys

RUN_ONE = [sys.executable, "-m", "vrp.experiments.run_experiment"]  # dùng python hiện tại

def build_cmd_args(data: str, solver: str, seed: int, tlimit: float, plot: bool, config: str|None):
    args = RUN_ONE + ["--data", data, "--solver", solver, "--seed", str(seed), "--time", str(tlimit)]
    if config:
        args += ["--config", config]
    if plot:
        args += ["--plot"]
    return args

def build_cmd(data: str, solver: str, seed: int, tlimit: float, plot: bool, config: str|None):
    parts = [
        RUN_ONE,
        f"--data {shlex.quote(str(data))}",
        f"--solver {solver}",
        f"--seed {seed}",
        f"--time {tlimit}",
    ]
    if config:
        parts.append(f"--config {shlex.quote(str(config))}")
    if plot:
        parts.append("--plot")
    return " ".join(parts)

def run_once(cmd_args: str, cwd: str|None=None) -> dict:
    """Chạy 1 experiment, parse stdout về dict kết quả."""
    # Windows note: dùng shell=True để "-m" hoạt động trong cùng env python
    t0 = time.time()
    proc = subprocess.run(cmd_args, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - t0

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"returncode={proc.returncode}",
            "stdout": out,
            "stderr": err,
            "elapsed_wall": round(elapsed, 3),
        }

    # Parse các dòng in từ run_experiment.py
    # Kỳ vọng có các dòng:
    # solver: ...
    # seed: ...
    # time: XXX s
    # total_cost: NNN
    # details: {...}
    res = {"ok": True, "stdout": out, "stderr": err, "elapsed_wall": round(elapsed, 3)}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("solver:"):
            res["solver"] = line.split("solver:", 1)[1].strip()
        elif line.startswith("seed:"):
            res["seed"] = int(line.split("seed:", 1)[1].strip())
        elif line.startswith("time:"):
            # dạng: "time: 5.123 s"
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

def expand_data_glob(data_glob: str) -> list[str]:
    paths = sorted(str(p) for p in Path().glob(data_glob))
    if not paths:
        raise SystemExit(f"[run_batch] Không tìm thấy instance nào với glob: {data_glob}")
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True,
                    help='Glob tới thư mục instance (vd: "dataset_test/*/base_50")')
    ap.add_argument("--solvers", nargs="+", default=["dfa"], choices=["dfa","ga","esa"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=1, help="số luồng song song (1 = tuần tự)")
    ap.add_argument("--out", type=str, default="batch_results.csv")
    args = ap.parse_args()

    datasets = expand_data_glob(args.data_glob)

    # header CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames_base = [
        "dataset","solver","seed","cmd","ok","return_total_cost","elapsed_wall",
    ]
    # sẽ bung thêm các key trong details
    rows = []

    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for data in datasets:
            for solver in args.solvers:
                for seed in args.seeds:
                    cmd_args = build_cmd_args(data, solver, seed, args.time, args.plot, args.config)
                    fut = ex.submit(run_once, cmd_args)
                    fut._meta = {"dataset": data, "solver": solver, "seed": seed, "cmd": " ".join(cmd_args)}
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
            }
            det = res.get("details", {})
            # ghép chi tiết (distance, tw_penalty, ...)
            for k, v in det.items():
                row[k] = v
            # lưu stdout/stderr để dễ debug (tuỳ bạn muốn bỏ hay giữ)
            row["stdout"] = res.get("stdout","")
            row["stderr"] = res.get("stderr","")
            rows.append(row)
            print(f"[DONE] {meta['solver']} seed={meta['seed']} data={meta['dataset']} -> ok={row['ok']} cost={row['return_total_cost']}")

    # thu thập tất cả keys để ghi CSV đầy đủ cột
    all_fields = list(fieldnames_base)
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
