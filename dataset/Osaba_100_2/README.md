# Osaba VRP (PD-only) — Normalized & Expanded for `Osaba_100_2.xml` (2025-11-11T03:27:16.383610Z)

- PD-only: mỗi khách chỉ P hoặc D.
- Nếu --lock-both-totals: 
    sum(del_new) = 1.00 × sum(del_orig) và sum(del_new) = 1.00 × sum(pick_new)
- Otherwise: áp ratio rồi ép absolute delivery (có thể lệch ratio).
- Vehicle fixed_cost = 100 × capacity (bỏ qua --veh-fixed-cost).
