# run_all_xml.py
from pathlib import Path
import subprocess

IN_DIR  = r"instances_raw"          # folder chứa các .xml
OUT_DIR = r"dataset"           # folder output gốc
PYTHON  = "python"                        # hoặc đường dẫn python cụ thể

COMMON_ARGS = [
    "--expand-base",
    "--del-mult-from-original", "1.0",
    "--pd-target-ratio", "1.0",
    "--veh-per-depot-min", "3", "--veh-per-depot-max", "4",
    "--veh-capacities", "80", "100", "120", "150",
    # "--synth-sizes", "200", "500",
    # "--synth-clusters-min", "20", "--synth-clusters-max", "50",
    "--tw-open-min", "480", "--tw-open-max", "720",
    "--tw-close-min", "780", "--tw-close-max", "1200",
    "--tw-min-width", "60",
    "--seed", "42",
]

gen = Path("generate_vrp_pd_only.py").resolve()
for xml in sorted(Path(IN_DIR).glob("*.xml")):
    cmd = [PYTHON, str(gen), "--in", str(xml), "--out", OUT_DIR] + COMMON_ARGS
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
