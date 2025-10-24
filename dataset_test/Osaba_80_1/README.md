[E(x, y) = -\sum_{c_i \in C} f_i(c_i)]

[P(X = x, Y = y) = \frac{1}{Z} e^{-E(x, y)}]

[Z = \sum_{x \in \Omega_{x, y} \in \Omega_y} e^{-E(x, y)}]

Trong đó C là tập tất cả các nhóm đầy đủ của đồ thị G (một nhóm đầy đủ trong đồ thị vô hướng là một tập hợp các đỉnh mà giữa tất cả các cặp đỉnh trong tập hợp đó đều tồn tại một cạnh), (f_i) là hàm năng lượng của cụm (c_i) chỉ ra khả năng xảy ra các mối quan hệ trong cụm. Z là hằng số chuẩn hóa để tạo phân phối xác suất hợp lệ (<1). (E(x, y)) là hàm năng lượng được sử dụng để đánh giá mức độ "tốt" của một cặp giá trị ((x, y)) cụ thể của các biến ngẫu nhiên X, Y. Cặp giá trị ((x, y)) có (E(x, y)) thấp hơn được coi là tốt hơn.

Dựa vào công thức trên kết hợp định lý Bayes, ta suy ra phân phối của chuỗi nhãn Y khi biết X có dạng sau:

[ P(Y = y | X = x) = \frac{P(Y = y, X = x)}{P(X = x)} = \frac{\frac{e^{-E(x, y)}}{Z}}{\frac{\sum_{y' \in \Omega_y} e^{-E(x, y')}}{Z}} \quad = \frac{e^{-E(x, y)}}{Z(x)} \quad = \frac{exp\left(\sum_{c_i \in C} f_i(c_i)\right)}{Z(x)} ]