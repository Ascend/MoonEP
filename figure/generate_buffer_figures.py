"""Generate the README figures explaining MoonEP's weight/grad buffer layout.

Outputs:
  figure/weight_buffer.png  - compact per-rank weight and prefetch views
  figure/grad_buffer.png    - compact per-rank grad and reduce views
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


YELLOW = "#FBE7A1"   # local experts / local grad
PINK = "#F3D6DA"     # prefetch slots / reduce buffer
RED = "#C05046"      # shared physical pool
EDGE = "#333333"
ARROW = "#555555"
TEXT = "#222222"

R = 4
LOCAL_W = 1.9
BUFFER_W = 1.9
BAR_H = 0.55
PAD_W = 0.38
PAD = "#E8EAED"


def arrow(ax, x1, y1, x2, y2, rad=0.0, style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=10,
                                 linewidth=lw, linestyle=ls, color=ARROW))


def padding(ax, x, y, width=PAD_W):
    ax.add_patch(Rectangle((x, y), width, BAR_H, facecolor=PAD,
                           edgecolor="#89919B", hatch="///", linewidth=0.8))


def draw_rank_bar(ax, x0, y, local_text, buffer_text, fontsize=8):
    # Local payload ends at a block boundary; the pool starts there.
    padding(ax, x0 - PAD_W, y)
    padding(ax, x0 + LOCAL_W + BUFFER_W, y)
    ax.add_patch(Rectangle((x0, y), LOCAL_W, BAR_H,
                           facecolor=YELLOW, edgecolor=EDGE, linewidth=1.0))
    ax.text(x0 + LOCAL_W / 2, y + BAR_H / 2, local_text,
            ha="center", va="center", fontsize=fontsize, color=TEXT)
    ax.add_patch(Rectangle((x0 + LOCAL_W, y), BUFFER_W, BAR_H,
                           facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
    ax.text(x0 + LOCAL_W + BUFFER_W / 2, y + BAR_H / 2, buffer_text,
            ha="center", va="center", fontsize=fontsize, color=TEXT)


def draw_rank_rows(ax, x0, y_top, pitch, title, local_text, buffer_text,
                   show_rank_labels):
    bar_w = LOCAL_W + BUFFER_W
    ax.text(x0 + bar_w / 2, y_top + BAR_H + 0.42, title,
            ha="center", va="center", fontsize=11, color=TEXT)
    centers = []
    for rank in range(R):
        y = y_top - rank * pitch
        centers.append(y + BAR_H / 2)
        if show_rank_labels:
            ax.text(x0 - PAD_W - 0.15, y + BAR_H / 2, f"EP rank {rank}",
                    ha="right", va="center", fontsize=9, color=TEXT)
        draw_rank_bar(ax, x0, y, local_text, buffer_text)

    tail_right = x0 + bar_w
    bus_x = tail_right + PAD_W + 0.3
    for center in centers:
        ax.plot([tail_right + PAD_W, bus_x], [center, center], color=ARROW, lw=1.1)
    ax.plot([bus_x, bus_x], [centers[-1], centers[0]], color=ARROW, lw=1.1)
    ax.plot([bus_x] * R, centers, "o", ms=3.5, color=ARROW)
    bottom = y_top - (R - 1) * pitch
    ax.annotate("", xy=(x0, bottom - 0.14),
                xytext=(x0 + bar_w, bottom - 0.14),
                arrowprops=dict(arrowstyle="|-|", color=ARROW, lw=1))
    ax.text(x0 + bar_w / 2, bottom - 0.36,
            "Contiguous GEMM view [2*epn, H, H']", ha="center", fontsize=8)
    return bus_x, centers


def draw_all_rank_view(ax, x0, y, title, buffer_text, width=1.7):
    ax.text(x0 + R * width / 2, y + BAR_H + 0.28, title,
            ha="center", va="center", fontsize=10, color=TEXT)
    for rank in range(R):
        payload_w = width - PAD_W
        ax.add_patch(Rectangle((x0 + rank * width, y), payload_w, BAR_H,
                               facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
        padding(ax, x0 + rank * width + payload_w, y)
        ax.text(x0 + rank * width + payload_w / 2, y + BAR_H / 2,
                f"rank {rank}\n{buffer_text}", ha="center", va="center",
                fontsize=8, color=TEXT)
    ax.annotate("", xy=(x0, y - 0.2), xytext=(x0 + width, y - 0.2),
                arrowprops=dict(arrowstyle="<->", color=ARROW, lw=1))
    ax.text(x0 + width / 2, y - 0.47, "rank stride", ha="center", fontsize=8)
    ax.text(x0 + R * width / 2, y - 0.83,
            "stride(0) = aligned rank bytes / element size; each rank payload stays contiguous",
            ha="center", fontsize=8.5, color=TEXT)


def buffer_figure(path, buffer_name, local_name, notes):
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0.1, 9.4)
    ax.axis("off")

    physical_x, physical_y, physical_w, physical_h = 5.2, 7.95, 4.6, 0.55
    ax.add_patch(Rectangle((physical_x, physical_y), physical_w, physical_h,
                           facecolor=RED, edgecolor=EDGE, linewidth=1.0))
    ax.text(physical_x + physical_w / 2, physical_y + physical_h / 2,
            f"{buffer_name} physical pool, per rank\nshared across layers", ha="center", va="center",
            fontsize=9, color="white", weight="bold")

    y_top = 6.65
    pitch = 0.82
    panels = ((1.65, "layer ℓ"), (8.15, "layer ℓ+1"))
    buses = []
    for panel_idx, (x0, title) in enumerate(panels):
        bus_x, centers = draw_rank_rows(
            ax, x0, y_top, pitch,
            title,
            f"{local_name}\n[epn, H, H']",
            f"{buffer_name} buffer\n[epn, H, H']",
            show_rank_labels=panel_idx == 0,
        )
        buses.append((bus_x, centers))
        arrow(ax,
              physical_x + (0.25 if panel_idx == 0 else 0.75) * physical_w,
              physical_y,
              x0 + LOCAL_W + BUFFER_W / 2,
              y_top + BAR_H,
              rad=-0.08 if panel_idx == 0 else 0.08)

    view_x, view_y, view_w = 3.7, 2.6, 1.9
    draw_all_rank_view(
        ax, view_x, view_y,
        f"{buffer_name}-buffer view on every rank  [R, epn, H, H']",
        f"{buffer_name} buffer",
        width=view_w,
    )
    arrow(ax, buses[0][0], buses[0][1][-1],
          view_x + 0.2, view_y + BAR_H, rad=-0.12)
    arrow(ax, buses[1][0], buses[1][1][-1],
          view_x + R * view_w - 0.2, view_y + BAR_H, rad=0.12)

    padding(ax, 4.0, 8.82)
    ax.text(4.55, 9.07, "VMM padding (outside logical tensor views)",
            va="center", fontsize=9, color=TEXT)

    for y, note in zip((1.15, 0.65), notes, strict=True):
        ax.text(7.5, y, note,
                ha="center", va="center", fontsize=8.5, color=TEXT)

    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def weight_buffer_figure(path):
    buffer_figure(
        path, "prefetch", "local experts",
        (
            "Local weights occupy the end of their VMM block; prefetch payload starts at the next block boundary.",
            "The GEMM view skips prefix/tail padding. The all-rank pool view skips padding via its rank stride.",
        ),
    )


def grad_buffer_figure(path):
    buffer_figure(
        path, "reduce", "local grad",
        (
            "One aligned physical block per full fp32 local grad; grad payload ends where the reduce payload starts.",
            "The GEMM grad view has no internal padding; only logical parameter grads enter framework reduction.",
        ),
    )


if __name__ == "__main__":
    weight_buffer_figure("figure/weight_buffer.png")
    grad_buffer_figure("figure/grad_buffer.png")
