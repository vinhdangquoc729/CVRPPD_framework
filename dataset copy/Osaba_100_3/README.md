# Osaba VRP — Normalized & Expanded for `Osaba_100_3.xml` (2025-11-12T05:20:41.151948Z)

- Base: giữ nguyên schema, không paired.
- Expanded (base_mdmv_tw_modified): **paired PD**, KHÔNG sinh toạ độ mới (dùng toạ độ gốc).
  Depot cũng lấy từ các điểm gốc; số khách được đảm bảo **chẵn** để ghép P↔D.
- Synthetic: **paired PD** (có thể sinh toạ độ mới), có `orders.csv`.
- Nếu --lock-both-totals: scale quantity toàn cục để đạt target_delivery_total (đồng bộ lại orders.csv).
- Vehicle fixed_cost = 100 × capacity (bỏ qua --veh-fixed-cost).
