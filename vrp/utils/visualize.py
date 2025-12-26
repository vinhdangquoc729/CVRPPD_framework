# vrp/utils/visualize.py
from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt

def draw_solution(problem, sol, save_path: Optional[str] = None, show: bool = False,
                  annotate: bool = False, dpi: int = 140):
    """
    Vẽ lời giải: Hỗ trợ cả bài toán gốc (Node-based) và bài toán mới (Order-based).
    """
    
    # --- SỬA LỖI Ở ĐÂY ---
    # Kiểm tra xem problem dùng 'nodes' (bản cũ) hay 'nodes_map' (bản mới)
    if hasattr(problem, "nodes_map"):
        nodes = problem.nodes_map
    else:
        nodes = problem.nodes
    # ---------------------
    
    # Kiểm tra xem đây là bài toán Order-based hay Node-based
    # (Bài toán mới có thuộc tính orders_map)
    is_order_based = hasattr(problem, "orders_map")
    
    # Map vehicle để lấy depot
    veh_map = {v.id: v for v in problem.vehicles}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # 1. Vẽ nền: Tất cả các điểm khách hàng (mờ)
    xs_c, ys_c = [], []
    for nid, nd in nodes.items():
        if not nd.is_depot:
            xs_c.append(nd.x); ys_c.append(nd.y)
    ax.scatter(xs_c, ys_c, s=10, alpha=0.25, c='gray', label="Nodes")

    # 2. Vẽ Depot (tô đậm, hình sao)
    xs_d, ys_d = [], []
    for nid, nd in nodes.items():
        if nd.is_depot:
            xs_d.append(nd.x); ys_d.append(nd.y)
    ax.scatter(xs_d, ys_d, marker="*", s=200, c='red', zorder=10, label="Depots")

    # 3. Vẽ từng Route
    # Dùng colormap để mỗi xe 1 màu
    colors = plt.cm.get_cmap('tab10', len(sol.routes))

    for idx, r in enumerate(sol.routes):
        if not r.seq: continue
        
        veh = veh_map.get(r.vehicle_id)
        if not veh: continue

        # Xác định điểm đầu và điểm cuối của xe
        # Logic mới: Vehicle có start_depot_id / end_depot_id
        # Logic cũ: Vehicle có depot_id
        start_node_id = getattr(veh, 'start_depot_id', getattr(veh, 'depot_id', 0))
        end_node_id = getattr(veh, 'end_depot_id', getattr(veh, 'depot_id', 0))

        # Xây dựng danh sách toạ độ cho route này
        xs, ys = [], []
        
        # -- Thêm điểm xuất phát --
        if start_node_id in nodes:
            start_node = nodes[start_node_id]
            xs.append(start_node.x)
            ys.append(start_node.y)

        # -- Duyệt qua sequence (Order hoặc Node) --
        for item_id in r.seq:
            node_id = -1
            
            if is_order_based:
                # Nếu là Order-based, item_id là order_id -> tìm node_id tương ứng
                if hasattr(problem, "orders_map"):
                    order = problem.orders_map.get(item_id)
                    if order:
                        node_id = order.node_id
            else:
                # Nếu là Node-based, item_id chính là node_id
                node_id = item_id
            
            if node_id != -1 and node_id in nodes:
                nd = nodes[node_id]
                xs.append(nd.x)
                ys.append(nd.y)

        # -- Thêm điểm kết thúc --
        if end_node_id in nodes:
            end_node = nodes[end_node_id]
            xs.append(end_node.x)
            ys.append(end_node.y)

        # Vẽ đường nối (Line)
        color = colors(idx % 10)
        ax.plot(xs, ys, linewidth=1.5, alpha=0.8, color=color, label=f"V{r.vehicle_id}")
        
        # Vẽ điểm trên route (Scatter) - bỏ điểm đầu cuối vì đã vẽ depot
        if len(xs) > 2:
            ax.scatter(xs[1:-1], ys[1:-1], s=20, color=color, zorder=5)

            # Annotate (nếu cần)
            if annotate:
                # Chỉ annotate điểm giữa
                for i in range(1, len(xs)-1):
                    # Hiển thị Order ID hoặc Node ID tuỳ bài toán
                    txt = str(r.seq[i-1]) 
                    ax.text(xs[i], ys[i], txt, fontsize=6, color='black', ha='center', va='bottom')

    ax.set_title(f"Solution Visualization ({'Order-based' if is_order_based else 'Node-based'})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Legend gọn hơn
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8, ncol=2)
    
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)