import csv
import re
import os

# --- PHẦN 1: Dữ liệu SQL thô (Copy paste dữ liệu của bạn vào đây) ---
# Lưu ý: Tôi đã ghép các đoạn SQL bạn gửi thành một chuỗi lớn để demo.
# Bạn có thể đọc từ file text bằng cách thay biến này bằng open('file.sql').read()

raw_sql_data = """
insert into trunglhavnist.correlations (id, created_at, updated_at, distance, from_node_code, from_node_id, from_node_name, from_node_type, risk_probability, time, to_node_code, to_node_id, to_node_name, to_node_type)
values  (1, 1658894328761, null, 0, 'D1', 1, 'Kho hàng AEON Mall Hà Đông', 'DEPOT', 0.01, 0, 'D1', 1, 'Kho hàng AEON Mall Hà Đông', 'DEPOT'),
        (2, 1658894415511, null, 7498, 'D2', 2, 'Kho hàng Big C THĂNG LONG', 'DEPOT', 0.01, 1183, 'D1', 1, 'Kho hàng AEON Mall Hà Đông', 'DEPOT'),
        (3, 1658894415524, null, 0, 'D2', 2, 'Kho hàng Big C THĂNG LONG', 'DEPOT', 0.01, 0, 'D2', 2, 'Kho hàng Big C THĂNG LONG', 'DEPOT'),
        (4, 1658894435695, null, 6426, 'D1', 1, 'Kho hàng AEON Mall Hà Đông', 'DEPOT', 0.01, 938, 'D2', 2, 'Kho hàng Big C THĂNG LONG', 'DEPOT'),
        (5, 1658894579864, null, 15799, 'D3', 3, 'Kho hàng MM Mega Market Hoàng Mai', 'DEPOT', 0.01, 1760, 'D1', 1, 'Kho hàng AEON Mall Hà Đông', 'DEPOT');

insert into trunglhavnist.customers (id, created_at, updated_at, address, code, end_time, latitude, longitude, name, start_time, penalty_cost, dx_code)
values  (1, 1658895466038, 1658895466072, 'Tầng B1, TTTM Vincom Center Liễu Giai, Số 29 Liễu Giai, P. Ngọc Khánh, Q. Ba Đình, Hà Nội', 'C1', 75600, 21.0311, 105.8136648, 'VinMart Liễu Giai', 28800, 1000, '62e0bc682c553306621a180e'),
        (2, 1658895500465, 1658895500505, '163A Đại La, Phường Đồng Tâm, quận Hai Bà Trưng, Hà Nội', 'C2', 75600, 20.9964605, 105.8446364, 'VinMart Đại La', 28800, 1000, '62e0bc8c2c553306621a1821'),
        (3, 1658895522261, 1658895522280, 'Tòa nhà Nam Đô Complex, 609 Trương Định, Phường Thịnh Liệt, Quận Hoàng Mai, Hà Nội', 'C3', 75600, 20.9794457, 105.8449844, 'VinMart Trương Định', 28800, 1000, '62e0bca12c553306621a1834'),
        (4, 1658895551541, 1658895551559, 'TTTM Vincom Nguyễn Chí Thanh, Sô 54A Nguyễn Chí Thanh, Phường Láng Thượng, Quận Đống Đa, Hà Nội', 'C4', 75600, 21.0234901, 105.8093995, 'VinMart Nguyễn Chí Thanh', 28800, 1000, '62e0bcbf2c553306621a1847'),
        (5, 1658895662824, 1658895662855, 'T1, Toà 170 La Thành, P. Ô Chợ Dừa, Q. Đống Đa, Hà Nội', 'C5', 75600, 21.0207133, 105.8275735, 'VinMart La Thành', 28800, 1000, '62e0bd2e2c553306621a185a');

insert into trunglhavnist.depots (id, created_at, updated_at, address, code, end_time, latitude, longitude, name, start_time, unloading_cost, dx_code)
values  (1, 1658894328274, 1658894328371, 'AEON Mall Hà Đông, Dương Nội, Hà Đông, Hà Nội, Việt Nam', 'D1', 75600, 20.9897037, 105.7517169, 'Kho hàng AEON Mall Hà Đông', 25200, 0, '62e0b7f52c553306621a1430'),
        (2, 1658894415241, 1658894415271, 'Big C THĂNG LONG', 'D2', 75600, 21.0074192, 105.793188, 'Kho hàng Big C THĂNG LONG', 25200, 0, '62e0b84e2c553306621a1444'),
        (3, 1658894579619, 1658894579638, '126 Đ. Tam Trinh, Yên Sở, Hoàng Mai, Hà Nội, Việt Nam', 'D3', 75600, 20.970357, 105.8667081, 'Kho hàng MM Mega Market Hoàng Mai', 25200, 0, '62e0b8f32c553306621a1458'),
        (4, 1658894633663, 1658894633697, 'Trung tâm thương mại Aeon Mall Long Biên 27 Đ. Cổ Linh, Long Biên, Hà Nội', 'D4', 75600, 21.0272741, 105.8993753, 'Kho hàng AEON Mall Long Biên', 25200, 0, '62e0b9282c553306621a146c'),
        (5, 1658894716574, 1658894716624, 'Số 29, ngõ 47 Nguyễn Khá Trạc, Mai Dịch, Cầu Giấy, Hà Nội', 'D5', 75600, 21.0451697, 105.7773353, 'Công ty TNHH nước giải khát BlueSea (nước giải khát và đồ uống không cồn)', 25200, 0, '62e0b97c2c553306621a1480');

insert into trunglhavnist.depots_products (depot_id, product_id)
values  (1, 6), (1, 22), (1, 27), (1, 31), (1, 73), (1, 80), (2, 6), (2, 22), (2, 27), (2, 31);

insert into trunglhavnist.good_groups (id, created_at, updated_at, detail, name, dx_code)
values  (13, 1644981056507, null, 'Không có yêu cầu đặc biệt', 'Không xác định', '62ddc913a033170dbbae09e3'),
        (8, 1644833220700, null, 'Tránh ẩm', 'Tránh ẩm', '62ddc925a033170dbbae09ed'),
        (9, 1644833241068, null, 'Hàng dễ hỏng', 'Hàng dễ hỏng', '62ddc939a033170dbbae09f7'),
        (10, 1644833255783, null, 'Tránh nhiệt độ cao', 'Tránh nhiệt độ cao', '62ddc949a033170dbbae0a01'),
        (11, 1644833291004, null, 'Hàng hóa chất', 'Hóa chất', '62ddc95ea033170dbbae0a0b'),
        (12, 1644833313870, null, 'Thực phẩm rau củ và đồ khô', 'Rau củ đồ khô', '62ddc970a033170dbbae0a15'),
        (15, 1659232167717, null, 'Tránh va đập mạnh', 'Đồ nội thất', '62e5dfa74ffe4928a6b042b8'),
        (16, 1659232214452, null, 'Tránh va đập mạnh', 'Đồ điện gia dụng', '62e5dfd64ffe4928a6b042c2');

insert into trunglhavnist.order_items (id, created_at, updated_at, capacity, price, quantity, weight, order_id, product_id, return_order_id)
values  (1, 1658898151576, null, 0.6, 1267500, 15, 19.5, 1, 27, null),
        (2, 1658898151704, null, 0.6, 5400000, 15, 240, 1, 73, null),
        (3, 1658898151835, null, 1.4, 6800000, 20, 200, 1, 31, null),
        (4, 1658898151961, null, 0.2, 3100000, 20, 76, 1, 22, null),
        (5, 1658898152086, null, 3.6, 8990000, 10, 68, 1, 80, null),
        (6, 1658898162271, null, 0.6, 1267500, 15, 19.5, 2, 27, null),
        (7, 1658898162398, null, 0.6, 5400000, 15, 240, 2, 73, null),
        (8, 1658898162524, null, 1.4, 6800000, 20, 200, 2, 31, null);

insert into trunglhavnist.orders (id, created_at, updated_at, capacity, code, delivery_after_time, delivery_before_time, delivery_mode, intend_receive_time, order_value, time_service, time_loading, weight, customer_id, depot_id, dx_code)
values  (1, 1660186077665, 1660186078343, 0.68, 'O1', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 1, null, '62f46ddd075eaf223f75d802'),
        (2, 1660186093294, 1660186094011, 0.68, 'O2', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 2, null, '62f46ded075eaf223f75d816'),
        (3, 1660186101063, 1660186101751, 0.68, 'O3', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 3, null, '62f46df4075eaf223f75d82a'),
        (4, 1660186108594, 1660186109266, 0.68, 'O4', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 4, null, '62f46dfc075eaf223f75d83e'),
        (5, 1660186116183, 1660186116869, 0.68, 'O5', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 5, null, '62f46e03075eaf223f75d852'),
        (6, 1660186125393, 1660186126064, 0.68, 'O6', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 6, null, '62f46e0d075eaf223f75d866'),
        (7, 1660186133023, 1660186133736, 0.68, 'O7', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 8, null, '62f46e14075eaf223f75d87a'),
        (8, 1660186140455, 1660186141113, 0.68, 'O8', 28800, 75600, 'STANDARD', null, 5920000, 300, 300, 122, 7, null, '62f46e1c075eaf223f75d88e');

insert into trunglhavnist.product_exclude (product_excluding_id, excluded_product_id)
values  (100, 80), (100, 27), (22, 100), (22, 93), (27, 93), (31, 93), (33, 93), (95, 31), (80, 95), (22, 95), (41, 93), (42, 93), (43, 93);

insert into trunglhavnist.products (id, created_at, updated_at, capacity, code, height, length, name, price, weight, width, goods_group_id, dx_code)
values  (22, 1644847377712, 1644847377754, 0.01, 'P22', 0.29, 0.26, 'Nước Giặt Ariel Matic Dạng Túi Đậm Đặc 3.5kg', 155000, 3.8, 0.16, 11, '62ddcfc0a033170dbbae0e73'),
        (27, 1644848063278, 1644848063282, 0.04, 'P27', 0.17, 0.6, 'Lốc 10 Cuộn Giấy Vệ Sinh Cao Cấp Bless You À La Vie 2 Lớp', 84500, 1.3, 0.4, 8, '62ddcddfa033170dbbae0bd7'),
        (31, 1644848743850, 1644848743857, 0.07, 'P31', 0.3, 0.6, 'Thùng 48 hộp sữa Milo 180ml Nestle', 340000, 8, 0.4, 10, '62ddcf6ba033170dbbae0e69'),
        (103, 1659234286653, 1659234286664, 0.07, 'P103', 0.6, 0.6, 'Thùng 24 Lon Nước Tăng Lực Sting Dâu (320ml/Lon)', 198000, 5, 0.4, 10, '62e5e7ee4ffe4928a6b044a9'),
        (107, 1659234699374, 1659234699383, 0.07, 'P107', 0.3, 0.6, 'Sữa Tươi Tiệt Trùng Ít Đường TH true MILK 180 ml', 210000, 8, 0.4, 10, '62e5e98b4ffe4928a6b044d1'),
        (73, 1644945642117, 1644945642124, 0.04, 'P73', 0.34, 0.4, 'BIA TRÚC BẠCH (CHAI) 24 chai/két', 360000, 6, 0.3, 9, '62ddce80a033170dbbae0beb'),
        (100, 1659233331423, 1659233331428, 0.01, 'P100', 0.08, 0.05, 'Sữa đặc ông thọ', 22000, 0.38, 0.1, 10, '62e5e4334ffe4928a6b0448b'),
        (95, 1659232784710, 1659232784718, 0.07, 'P95', 0.6, 0.3, 'THÙNG 48 HỘP SỮA TƯƠI TIỆT TRÙNG VINAMILK 100% CÓ ĐƯỜNG 180ML', 350000, 8, 0.4, 10, '62e5e2104ffe4928a6b04459'),
        (80, 1644946931941, 1644946931947, 0.36, 'P80', 0.3, 0.4, 'Nước Cam ép Twister 24 chai/két -Rỗng', 899000, 6.8, 3, 13, '62ddd0f9a033170dbbae0ea1');

insert into trunglhavnist.vehicles (id, created_at, updated_at, available, average_fee_transport, average_gas_consume, average_velocity, fixed_cost, driver_name, gas_price, height, length, max_capacity, max_load_weight, max_velocity, min_velocity, name, type, width, dx_code, vehicle_cost)
values  (1, 1658744924151, 1658856123218, true, 1560, 0.06, 65, 1200, null, 30000, 1.73, 3.17, 8.993923999999998, 1250, 120, 10, 'XT1 HYUNDAI NEW PORTER 150 Kín Cánh Dơi', 'TRUCK', 1.64, '62de705b89785b2c61a385bd', 146000),
        (2, 1658745064510, 1658856123342, true, 1560, 0.06, 65, 1200, null, 30000, 1.73, 3.17, 8.993923999999998, 1250, 120, 10, 'XT2 - HYUNDAI NEW PORTER 150 Kín Cánh Dơi', 'TRUCK', 1.64, '62de70e889785b2c61a38801', 146000),
        (3, 1658745123855, 1658856123463, true, 1560, 0.06, 60, 1200, null, 30000, 1.7, 3, 8.16, 1000, 80, 40, 'Xe tải 1 tấn - 01', 'TRUCK', 1.6, '62de712389785b2c61a38825', 134000),
        (23, 1659235883168, 1659236416781, true, 520, 0.02, 45, 1200, null, 30000, 0.8, 1, 0.6400000000000001, 120, 70, 20, 'Xe máy wave alpha 110cc - 01', 'BIKE', 0.8, '62e5ee2b4ffe4928a6b04780', 53500);

insert into trunglhavnist.vehicles_products (id, created_at, updated_at, maxNumber, product_id, vehicle_id)
values  (1224, 1659235883236, null, 8, 107, 24),
        (1220, 1659235883234, null, 4, 103, 24),
        (1217, 1659235883233, null, 8, 31, 24),
        (1216, 1659235883233, null, 15, 27, 24),
        (1215, 1659235883233, null, 53, 22, 24),
        (1213, 1659235883188, null, 1, 80, 23);
"""

# --- PHẦN 2: Hàm xử lý chính ---

def parse_sql_values_to_csv(sql_text, output_dir="output_csvs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Tách các câu lệnh INSERT INTO
    # Regex này tìm từ khóa 'insert into', lấy tên bảng, cột và phần values
    # (?si) là cờ để dot (.) match newline và case-insensitive
    statements = re.split(r'insert\s+into\s+', sql_text, flags=re.IGNORECASE)

    for stmt in statements:
        if not stmt.strip():
            continue
            
        # Regex phân tích từng block:
        # Group 1: Tên bảng (có thể có schema.)
        # Group 2: Danh sách cột trong (...)
        # Group 3: Phần giá trị sau chữ VALUES
        match = re.match(r'([^\s\(]+)\s*\((.*?)\)\s*values\s*([\s\S]+)', stmt, re.IGNORECASE | re.DOTALL)
        
        if not match:
            continue

        full_table_name = match.group(1).strip()
        columns_str = match.group(2)
        values_str = match.group(3)

        # Xử lý tên file (bỏ schema trunglhavnist.)
        if '.' in full_table_name:
            table_name = full_table_name.split('.')[-1]
        else:
            table_name = full_table_name

        # Xử lý headers
        headers = [c.strip() for c in columns_str.split(',')]

        # Xử lý values
        # Phần này khó nhất vì giá trị có thể chứa dấu phẩy bên trong string (VD: 'Hà Nội, Việt Nam')
        # Ta dùng một hàm generator đơn giản để duyệt từng ký tự
        rows = []
        current_row = []
        current_val = []
        in_quote = False
        in_parenthesis = False
        
        # Làm sạch chuỗi values (bỏ dấu chấm phẩy cuối hoặc từ khóa insert tiếp theo nếu dính vào)
        # Cắt tại ký tự ';' hoặc chữ 'insert' nếu regex split chưa sạch
        end_idx = len(values_str)
        if ';' in values_str:
             end_idx = min(end_idx, values_str.find(';'))
        values_str = values_str[:end_idx].strip()

        # Parser thủ công để đảm bảo chính xác với các dấu phẩy
        i = 0
        while i < len(values_str):
            char = values_str[i]
            
            if char == '(':
                if not in_quote:
                    in_parenthesis = True
                    current_row = []
                    current_val = []
                else:
                    current_val.append(char)
            
            elif char == ')':
                if not in_quote:
                    if in_parenthesis:
                        # Kết thúc 1 row
                        # Add giá trị cuối cùng vào row
                        val_str = "".join(current_val).strip()
                        current_row.append(clean_sql_value(val_str))
                        rows.append(current_row)
                        in_parenthesis = False
                else:
                    current_val.append(char)
            
            elif char == "'":
                in_quote = not in_quote
                # Không append dấu quote vào giá trị cuối cùng để CSV sạch hơn
                
            elif char == ',':
                if not in_quote:
                    if in_parenthesis:
                        # Kết thúc 1 cell
                        val_str = "".join(current_val).strip()
                        current_row.append(clean_sql_value(val_str))
                        current_val = []
                    else:
                        # Dấu phẩy giữa các row ( ), ( ) -> bỏ qua
                        pass
                else:
                    current_val.append(char)
            
            else:
                if in_parenthesis:
                    current_val.append(char)
            
            i += 1

        # Ghi ra file CSV
        output_file = os.path.join(output_dir, f"{table_name}.csv")
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"Đã tạo file: {output_file} ({len(rows)} dòng)")

def clean_sql_value(val):
    """Làm sạch giá trị SQL thô thành giá trị Python/CSV"""
    if val.upper() == 'NULL':
        return None # hoặc "" tùy bạn chọn
    if val.upper() == 'TRUE':
        return True
    if val.upper() == 'FALSE':
        return False
    # Nếu là số, trả về số (để Excel hiểu)
    try:
        if '.' in val:
            return float(val)
        return int(val)
    except ValueError:
        return val # Trả về chuỗi nguyên gốc

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    folder_name = "data_raw"  # Thư mục chứa các file .sql hoặc .txt
    output_dir = "data_processed" # Thư mục chứa file csv đầu ra

    # Kiểm tra thư mục đầu vào có tồn tại không
    if not os.path.exists(folder_name):
        print(f"Lỗi: Không tìm thấy thư mục '{folder_name}'. Hãy tạo thư mục và bỏ các file SQL vào đó.")
    else:
        print(f"Đang đọc dữ liệu từ thư mục: {folder_name}...")
        
        for file_name in os.listdir(folder_name):
            full_path = os.path.join(folder_name, file_name)
            
            if os.path.isfile(full_path):
                print(f"--> Đang xử lý file: {file_name}")
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        raw_sql_data = f.read()
                        # Gọi hàm xử lý và lưu vào output_dir
                        parse_sql_values_to_csv(raw_sql_data, output_dir=output_dir)
                except Exception as e:
                    print(f"    Lỗi khi đọc file {file_name}: {e}")

        print("--- Hoàn tất ---")