from pathlib import Path
import subprocess
import sys

IN_DIR  = r"instances_raw"          # folder chứa các .xml
OUT_DIR = r"dataset"                # folder output gốc
PYTHON  = sys.executable            # Tự động lấy đường dẫn python đang chạy

COMMON_ARGS = [
    # --- QUAN TRỌNG: Thêm flag này để sinh ra bản mở rộng (base_mdmv_tw_modified) ---
    "--expand-base",

    # --- Các thông số giữ NGUYÊN bản gốc của bạn ---
    "--del-mult-from-original", "1.0",
    "--pd-target-ratio", "1.0",
    "--veh-per-depot-min", "3", "--veh-per-depot-max", "4",
    "--veh-capacities", "80", "100", "120", "150",
    
    # "--synth-sizes", "200", "500",
    # "--synth-clusters-min", "20", "--synth-clusters-max", "50",
    
    "--tw-open-min", "480", "--tw-open-max", "720",
    "--tw-close-min", "780", "--tw-close-max", "1140",
    "--tw-min-width", "420",
    "--seed", "42",
]

def main():
    gen = Path("generate_vrp_pd_only.py").resolve()
    input_path = Path(IN_DIR)
    
    # Kiểm tra file generate tồn tại không
    if not gen.exists():
        print(f"Lỗi: Không tìm thấy file {gen}")
        return

    # Lấy danh sách file xml
    xml_files = sorted(input_path.glob("*.xml"))
    if not xml_files:
        print(f"Không tìm thấy file .xml nào trong {IN_DIR}")
        return

    for xml in xml_files:
        cmd = [PYTHON, str(gen), "--in", str(xml), "--out", OUT_DIR] + COMMON_ARGS
        print(f">> Processing {xml.name}...")
        # print("   Command:", " ".join(cmd)) # Uncomment nếu muốn debug lệnh
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!! Lỗi khi chạy file {xml.name}")

if __name__ == "__main__":
    main()