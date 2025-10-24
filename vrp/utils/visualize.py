# vrp/utils/visualize.py
from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt

def draw_solution(problem, sol, save_path: Optional[str] = None, show: bool = False,
                  annotate: bool = False, dpi: int = 140):
    """
    Vẽ lời giải: mỗi route là một polyline qua các node trong Route.seq.
    - Depot: marker '*' lớn; Khách: marker 'o' nhỏ
    - Có thể annotate id khách (annotate=True)
    """
    nodes = problem.nodes
    veh_map = {v.id: v for v in problem.vehicles}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # vẽ tất cả khách (mờ) để thấy nền
    xs_c, ys_c = [], []
    for nid, nd in nodes.items():
        if not nd.is_depot:
            xs_c.append(nd.x); ys_c.append(nd.y)
    ax.scatter(xs_c, ys_c, s=10, alpha=0.25, label="customers (all)")

    # vẽ depot (tô đậm)
    xs_d, ys_d, lbl_d = [], [], []
    for nid, nd in nodes.items():
        if nd.is_depot:
            xs_d.append(nd.x); ys_d.append(nd.y); lbl_d.append(nid)
    dep_sc = ax.scatter(xs_d, ys_d, marker="*", s=180, zorder=5, label="depots")

    # vẽ từng route
    for r in sol.routes:
        v = veh_map[r.vehicle_id]
        dep = v.depot_id
        # polyline theo seq
        xs, ys = [], []
        for nid in r.seq:
            nd = nodes[nid]
            xs.append(nd.x); ys.append(nd.y)
        line = ax.plot(xs, ys, linewidth=1.5, alpha=0.9, label=f"veh {r.vehicle_id} @ D{dep}")
        # chấm các khách trên route (đậm hơn)
        cx, cy, ids = [], [], []
        for nid in r.seq:
            if not nodes[nid].is_depot:
                cx.append(nodes[nid].x); cy.append(nodes[nid].y); ids.append(nid)
        ax.scatter(cx, cy, s=16, alpha=0.9)

        if annotate:
            for (x, y, nid) in zip(cx, cy, ids):
                ax.text(x, y, str(nid), fontsize=7, ha="center", va="center")

    ax.set_title("VRP Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
