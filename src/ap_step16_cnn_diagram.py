"""Step 16: layered (feature map style) diagram of our Wide CNN (top) and the Reimplemented CNN, the Fig. 7b model of the paper (bottom).
Parameter counts checked with sum(p.numel()) on the models in ap_step18_same_as_original.py:
  Wide: conv 7,744 + dense 130 = 7,874.   Reimplemented: conv 656 + conv 2,064 + dense 64,016 + dense 34 = 66,770.
Usage: python3 -I ap_step16_cnn_diagram.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).parent.parent
BLUE, LIGHT, ORANGE = "#4C72B0", "#C9D6EA", "#DD8452"
RED, LRED = "#C44E52", "#EBCACB"

fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 112); ax.set_ylim(12, 112); ax.set_aspect("equal"); ax.axis("off")
D = 0.9
H = 20


def slab(x, y, w, n, fc, ec):
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * D, y + i * D), w, H, fc=fc, ec=ec, lw=1.1))


def label(x, y, text, size=20):
    ax.text(x, y, text, ha="center", va="top", fontsize=size, linespacing=1.35)


def arrow(a, b, y):
    ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="->", lw=1.3))


def kernel(ix, y0, n_in, cx, n_out, cw, kw, kt, color):
    kx, ky = ix + (n_in - 1) * D + 6, y0 + (n_in - 1) * D + 9
    ax.add_patch(Rectangle((kx, ky), kw, 4.5, fc="#F6D9B8", ec=ORANGE, lw=1.4))
    ox, oy = cx + (n_out - 1) * D + 5, y0 + (n_out - 1) * D + 11
    ax.add_patch(Rectangle((ox, oy), 1.2, 1.2, fc="#9FB6DA", ec=color, lw=1.4))
    for (px, py), (qx, qy) in (((kx + kw, ky + 4.5), (ox, oy + 1.2)), ((kx + kw, ky), (ox, oy))):
        ax.plot([px, qx], [py, qy], color="gray", lw=0.8, ls="--")
    ax.text(kx + kw / 2, ky + 5.6, kt, fontsize=17, ha="center", color=ORANGE)


def units(x, y0, n, color, size=0.95):
    ys = [y0 + 3 + i * 2.2 for i in range(n)]
    for yy in ys:
        ax.add_patch(Circle((x, yy), size, fc="#9FB6DA", ec=color, lw=1.1))
    ax.text(x, y0 + 3 + n * 2.2 + 0.4, "...", fontsize=24, ha="center", va="center")
    return ys


def outputs(x, y0, ins, color):
    cy = y0 + H / 2 + 2
    ys = [cy - 2.5, cy + 2.5]
    for yy in ys:
        ax.add_patch(Circle((x, yy), 1.5, fc=ORANGE, ec="black", lw=1.2, zorder=3))
        for sx, sy in ins:
            ax.plot([sx, x - 1.4], [sy, yy], color="#999999", lw=0.5)
    ax.text(x, ys[1] + 2.4, "Clean or\nTrojan", fontsize=18, ha="center", va="bottom", linespacing=1.2)


def wide_row(y0):
    color, light = BLUE, LIGHT
    ax.text(55, y0 + H + 9, "Our Wide CNN: Total Trainable Parameters = 7,874", fontsize=23, ha="center", fontweight="bold", color=color)
    cy = y0 + H / 2 + 2
    ix, iw, cx, cw, rx = 4, 13, 33, 13, 56
    slab(ix, y0, iw, 5, "#F1E3D3", ORANGE); label(ix + iw / 2 + 1.8, y0 - 1.5, "Input\n5 x 1000\none-hot\nParams: 0")
    slab(cx, y0, cw, 8, light, color); label(cx + cw / 2 + 3, y0 - 1.5, "Conv1D\n64 filters\n64 x 977\nParams: 7,744")
    kernel(ix, y0, 5, cx, 8, cw, 2.2, "width 24", color)
    slab(rx, y0, cw, 8, light, color); label(rx + cw / 2 + 3, y0 - 1.5, "ReLU\n64 x 977\nParams: 0")
    arrow(ix + iw + 4 * D + 1, cx - 1, cy); arrow(cx + cw + 7 * D + 1, rx - 1, cy)
    gx, oxx = 80, 103
    ys = units(gx, y0 + 2, 8, color)
    for yy in ys:
        ax.plot([rx + cw + 7 * D + 0.5, gx - 1], [cy + 1, yy], color="#BBBBBB", lw=0.6, zorder=0)
    label(gx, y0 - 1.5, "Global\nMax-Pool\n64 units\nParams: 0")
    outputs(oxx, y0, [(gx + 1, yy) for yy in ys], color)
    label(oxx, y0 - 1.5, "Dense 2\n64 to 2\nParams: 130")


def reimpl_row(y0):
    color, light = RED, LRED
    ax.text(55, y0 + H + 9, "Reimplemented CNN: Total Trainable Parameters = 66,770", fontsize=23, ha="center", fontweight="bold", color=color)
    cy = y0 + H / 2 + 2
    ix, iw, cw = 3, 10, 10
    c1, c2, mp, dn, oxx = 21, 40, 59, 80, 103
    slab(ix, y0, iw, 5, "#F1E3D3", ORANGE); label(ix + iw / 2 + 1.8, y0 - 1.5, "Input\n5 x 1000\nParams: 0", 19)
    slab(c1, y0, cw, 5, light, color); label(c1 + cw / 2 + 2, y0 - 1.5, "Conv1D\nReLU\n16 x 1000\nParams: 656", 19)
    kernel(ix, y0, 5, c1, 5, cw, 1.6, "width 8", color)
    slab(c2, y0, cw, 5, light, color); label(c2 + cw / 2 + 2, y0 - 1.5, "Conv1D\nReLU\n16 x 1000\nParams: 2,064", 19)
    slab(mp, y0, 5, 5, light, color); label(mp + 4.5, y0 - 1.5, "Max-Pool 4\n16 x 250\nParams: 0", 19)
    arrow(ix + iw + 4 * D + 1, c1 - 1, cy); arrow(c1 + cw + 4 * D + 1, c2 - 1, cy); arrow(c2 + cw + 4 * D + 1, mp - 1, cy)
    ys = units(dn, y0 + 2, 8, color)
    for yy in ys:
        ax.plot([mp + 5 + 4 * D + 0.5, dn - 1], [cy + 1, yy], color="#BBBBBB", lw=0.6, zorder=0)
    label(dn, y0 - 1.5, "Dense 16\n4,000 to 16\nParams: 64,016", 19)
    outputs(oxx, y0, [(dn + 1, yy) for yy in ys], color)
    label(oxx, y0 - 1.5, "Dense 2\n16 to 2\nParams: 34", 19)


wide_row(80)
reimpl_row(28)
plt.rcParams.update({"pdf.fonttype": 42})
fig.tight_layout()
for e in ("png", "pdf", "svg"):
    fig.savefig(ROOT / f"figures/step16_cnn_architecture.{e}", dpi=170, bbox_inches="tight")
print("saved")
