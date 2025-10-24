# Osaba VRP — Normalized & Expanded (Generated on 2025-10-20T11:46:03.351790Z)

## Base (from XML)
- `base_50/nodes.csv`: id, addr, cluster, demand_delivery, demand_pickup, x, y
- `base_50/prohibited_arcs.csv`: from_id, to_id

## Expanded Base
- `base_50_mdmv_tw/` (if --expand-base):
  - `nodes.csv`: columns above + is_depot, service_time, tw_open, tw_close
  - `vehicles.csv`: vehicle_id, depot_id, capacity, start_time, end_time, fixed_cost, variable_cost_per_distance
  - `prohibited_arcs.csv`
  - `meta.json`

## Synthetic Instances
- `synthetic_{N}_mdmv_tw/` for N in [200, 500, 1000]:
  - Schema identical to expanded base.
  - 3–4 depots (configurable), 3–4 vehicles per depot (configurable), per-customer TW (unless --no-tw).
  - Clusters: 20–50; customers jittered around sampled cluster centers from the original data.

### Time Window Semantics
- Times are minutes from midnight. Example: 8:00 -> 480, 18:00 -> 1080.
- Penalty for arriving outside TW: `tw_penalty_per_min` in `meta.json` (linear).

### Notes
- Capacities are per-vehicle. Demands use (delivery, pickup) convention typical for CVRPPD.
- Distances are Euclidean on the provided coordinate plane.
- Randomness: seeded with 42 for reproducibility.
