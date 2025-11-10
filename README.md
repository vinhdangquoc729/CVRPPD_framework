# VRP Pickup and Delivery Framework

## 1. Tính năng chính
- Đọc và chuẩn hoá dữ liệu dạng file .xml của Osaba về dạng file .csv và .json
- Sinh dữ liệu simutaneous pickup and delivery, với time windows và multi-depots, multi-vehicles
- Cài đặt sẵn một số thuật toán như DFA, ESA, Cluster-GA..., cho phép chạy các thuật toán trên các bộ dữ liệu
- Ghi lại kết quả chạy để đánh giá các thuật toán, có hiển thị trực quan

## 2. Cấu trúc thư mục
```
configs/
└─ example.yaml               # Cấu hình thí nghiệm
instances_raw/                # Bộ dữ liệu gốc Osaba *.xml
dataset/                      # Nơi xuất dữ liệu đã chuẩn hoá / sinh tổng hợp
vrp/
├─ core/
│ ├─ eval.py                  # Hàm đánh giá chuẩn cho bài toán gốc (AC-VRP-SPDVCFP)
│ ├─ eval_modified.py         # Biến thể đánh giá cho bài toán simutaneous pickup and delivery mở rộng ràng buộc 
│ ├─ problem.py               
│ └─ solution.py 
│
├─ data/
│ └─ loader.py                # Load bài toán
│
├─ experiments/
│ ├─ run_experiment.py        # Chạy 1 thí nghiệm theo YAML
│ ├─ run_experiment_logged.py # Chạy & log các step
│ ├─ run_avg.py               
│ └─ run_batch.py            
│
├─ solvers/                   # Nơi đặt các solver
└─ utils/                     # Hàm tiện ích

generate_vrp_pd_only.py       # Script sinh dữ liệu PD‑only từ Osaba XML
run_all_xml.py                # Script chạy sinh dữ liệu hàng loạt
```

## 3. Hướng dẫn cài đặt
- Bước 1. Clone repository này về.
- Bước 2. Xây dựng và vào môi trường ảo
  ```
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Bước 3. Cài đặt các thư viện cần thiết
  ```
  pip install -r requirements.txt
  ```

## 4. Hướng dẫn sử dụng
### 4.1. Sinh dữ liệu
Ví dụ:
```
python generate_vrp_pd_only.py `
--in instances_raw/Osaba_50_1_1.xml `
--out dataset --expand-base
```

```
python generate_vrp_pd_only.py `
--in instances_raw/Osaba_50_1_1.xml `
--out dataset `
--expand-base `
--del-mult-from-original 1.0 `
--pd-target-ratio 1.0 `
--veh-per-depot-min 3 --veh-per-depot-max 4 `
--veh-capacities 80 100 120 150 `
--tw-open-min 480 --tw-open-max 720 `
--tw-close-min 780 --tw-close-max 1140 `
--tw-min-width 420 `
--seed 42
```

Chạy với tất cả các file trong folder raw: (các tham số chỉnh trực tiếp trong file)
```
python run_all_xml.py
```

### 4.2. Chạy thuật toán
- Chay 1 cấu hình
```
python -m vrp.experiments.run_experiment --data dataset/Osaba_100_1/base_mdmv_tw_modified --solver cluster_ga --seed 42 --time 30
```

- Chạy 1 cấu hình, có log cụ thể
```
python -m vrp.experiments.run_experiment_logged --data dataset/Osaba_100_1/base_mdmv_tw_modified --solver cluster_ga --seed 42 --time 30
```

---
Link bộ dữ liệu Osaba: https://github.com/Maldini32/AC-VRP-SPDVCFP
