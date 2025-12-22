import pandas as pd
import glob

# 1. Cấu hình danh sách file (Bạn có thể liệt kê thủ công hoặc dùng glob)
# Giả sử các file của bạn có tên là solver1.csv, solver2.csv, ...
file_paths = ['result_dfa_org.csv', 'result_esa_org.csv', 'result_ga_hct_org.csv', 'result_cluster_ga_org.csv']

# Nếu các file nằm trong một thư mục, bạn có thể dùng:
# file_paths = glob.glob("path/to/your/csv/*.csv")

all_dfs = []

# 2. Đọc dữ liệu từ các file
for path in file_paths:
    df = pd.read_csv(path)
    # if path == 'result_cluster_ga.csv':
    #     df.iloc[:, 2:] = df.iloc[:, 2:] * 1.3
    # elif path == 'result_esa.csv':
    #     df.iloc[:, 2:] = df.iloc[:, 2:] * 1.2

    all_dfs.append(df)

# Gộp tất cả thành 1 DataFrame duy nhất
combined_df = pd.concat(all_dfs, ignore_index=True)

# 3. Tìm Best Cost cho mỗi instance (giá trị nhỏ nhất của total_cost_min giữa các solver)
# Chúng ta tạo một bảng phụ chứa giá trị nhỏ nhất của mỗi instance
best_costs = combined_df.groupby('instance')['total_cost_min'].transform('min')

# 4. Tính toán Cost Gap (%)
# Công thức: (Cost hiện tại - Best Cost) / Best Cost * 100
combined_df['Cost Gap (%)'] = ((combined_df['total_cost_min'] - best_costs) / best_costs) * 100

# Làm tròn kết quả để dễ quan sát (ví dụ 2 chữ số thập phân)
combined_df['Cost Gap (%)'] = combined_df['Cost Gap (%)'].round(4)

# 5. Sắp xếp lại theo instance để các solver của cùng 1 bộ dữ liệu nằm cạnh nhau
combined_df = combined_df.sort_values(by=['instance', 'solver']).reset_index(drop=True)

# 6. Lưu kết quả ra file mới
combined_df.to_csv('combined_results_org.csv', index=False)

print("Đã gộp thành công! File 'combined_results.csv' đã sẵn sàng.")
print(combined_df[['instance', 'solver', 'total_cost_min', 'Cost Gap (%)']].head(10))