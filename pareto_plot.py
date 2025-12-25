import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

def count_active_vehicles(sol_str):
    """
    Đếm số lượng xe thực sự có phục vụ khách hàng.
    Loại bỏ các route rỗng dạng seq=[X, X].
    """
    if pd.isna(sol_str): return 0
    
    # 1. Tìm tất cả các đoạn nội dung của từng Route
    # Pattern này bắt lấy toàn bộ cụm: Route(vehicle_id=X, seq=[...])
    route_patterns = re.findall(r"Route\(vehicle_id=(\d+), seq=\[([\d, ]+)\]\)", sol_str)
    
    active_vehicles = set()
    
    for v_id, seq_str in route_patterns:
        # 2. Chuyển chuỗi seq thành danh sách các ID nút
        seq = [node.strip() for node in seq_str.split(',')]
        
        # 3. CHỈ TÍNH những xe có seq dài hơn 2 (có khách hàng)
        if len(seq) > 2:
            active_vehicles.add(v_id)
            
    return len(active_vehicles)

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_pareto_fronts(algo_patterns):
    all_data = {}
    for algo_name, pattern in algo_patterns.items():
        files = glob.glob(pattern)
        if not files: continue
        
        df_list = [pd.read_csv(f) for f in files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['num_vehicles'] = combined_df['solution_str'].apply(count_active_vehicles)
        
        # Lọc lời giải khả thi dựa trên các tiêu chí từ eval.py
        combined_df = combined_df[
            (combined_df['cap_violations'] == 0) & 
            (combined_df['prohibited_uses'] == 0) &
            (combined_df['unserved_customers'] == 0)
        ]
        all_data[algo_name] = combined_df

    plots = [
        ("distance", "fixed", "Distance Cost", "Vehicle Cost"),
        ("distance", "num_vehicles", "Distance Cost", "Number of Vehicles"),
        ("tw_penalty", "fixed", "TW Penalty Cost", "Vehicle Cost"),
        ("tw_penalty", "num_vehicles", "TW Penalty Cost", "Number of Vehicles")
    ]

    # Tăng kích thước tổng thể của hình ảnh
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (x_col, y_col, x_label, y_label) in enumerate(plots):
        ax = axes[idx]
        
        for algo_name, df in all_data.items():
            if df.empty: continue
            points = df[[x_col, y_col]].values
            
            # Vẽ điểm mờ
            ax.scatter(df[x_col], df[y_col], alpha=0.1, s=20)
            
            # Vẽ Pareto Front
            pareto_mask = is_pareto_efficient(points)
            pareto_points = points[pareto_mask]
            pareto_points = pareto_points[pareto_points[:, 0].argsort()]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'o-', label=algo_name, linewidth=2)

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f"Pareto: {y_label} vs {x_label}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # CHỈNH SỬA TẠI ĐÂY: Thu nhỏ chú thích và đặt ở vị trí ít bị đè nhất
        ax.legend(fontsize='8', loc='best', framealpha=0.5)

    # Tăng khoảng cách giữa các biểu đồ con để tránh dính chữ
    plt.tight_layout(pad=5.0) 
    plt.show()

# Cấu hình các solver để khớp với tên file xuất ra
folder = "last_generation_weight_100/"
target_algos = {
    "GA_TCPVRP": f"{folder}ga_tcpvrp_final_pop_seed*.csv",
    "ESA": f"{folder}esa_final_pop_seed*.csv",
    "GA_TCPVRP_ORIGIN": f"{folder}ga_tcpvrp_origin_final_pop_seed*.csv",
    "Ombuki": f"{folder}ombuki_final_pop_seed*.csv",
    "ClusterGA": f"{folder}cluster_ga_final_pop_seed*.csv"
}

if __name__ == "__main__":
    plot_pareto_fronts(target_algos)