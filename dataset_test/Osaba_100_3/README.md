# Osaba VRP — Normalized & Expanded for `Osaba_100_3.xml` (Generated on 2025-10-22T16:47:21.682295Z)

## Base (unexpanded) — unified schema (4 files)
- `base_50/nodes.csv`: id, addr, cluster, demand_delivery, demand_pickup, x, y, is_depot, service_time=0, tw_open=0, tw_close=1439
- `base_50/vehicles.csv`: single vehicle (capacity=240, fixed_cost=0, start=0, end=1439)
- `base_50/prohibited_arcs.csv`: from_id, to_id
- `base_50/meta.json`: type=single_depot_single_vehicle_with_time_windows

## Expanded Base
- `base_50_mdmv_tw/` (if --expand-base): nodes + vehicles + prohibited_arcs + meta (MD-MV, ±TW)

## Synthetic Instances
- `synthetic_{N}_mdmv_tw/` for N in []: same schema as expanded base.
