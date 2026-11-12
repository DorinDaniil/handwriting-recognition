from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

LINE_IMAGE = Path(r"C:\Users\dim\AppData\Local\Temp\claude\95d346d48683c58d"
                  r"\52e29266-84b2-491f-8479-277d13cb7377\images\1.png")
LINE_CROP = (10, 4, 1110, 85)

INK = "#374151"
TEXT = "#1f2937"
GREY = "#6b7280"
CELL = "#eef0f3"
BLUE, BLUE_FILL = "#1e73b7", "#dceaf6"
ORANGE, ORANGE_FILL = "#d97400", "#fdebd3"

SANS = ["Segoe UI", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

plt.rcParams.update({"font.family": SANS, "svg.fonttype": "none"})

W, H = 1152, 532.8
fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def text(x, y, s, size=13.5, color=TEXT, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def note(x, y, s, ha="center"):
    return text(x, y, s, size=13, color=GREY, ha=ha, style="italic")


def mono(x, y, s, size=13, ha="left", **kw):
    return text(x, y, s, size=size, ha=ha, family=MONO, **kw)


def right_edge(t):
    fig.canvas.draw()
    box = t.get_window_extent(fig.canvas.get_renderer())
    return ax.transData.inverted().transform((box.x1, box.y0))[0]


def rounded(x, y, w, h, fc="white", ec=INK, lw=1.4, ls="-", r=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def block(cx, cy, w, h, title, subtitle=None, color=TEXT, fc="white", ec=INK):
    rounded(cx - w / 2, cy - h / 2, w, h, fc=fc, ec=ec)
    if subtitle is None:
        text(cx, cy, title, size=17, color=color, weight="bold")
    else:
        text(cx, cy + 11, title, size=17, color=color, weight="bold")
        text(cx, cy - 13, subtitle, size=13.5, color=GREY)


def pill(x, y, w, label, color, fill):
    rounded(x, y - 14, w, 28, fc=fill, ec=color, lw=1.2, r=5)
    text(x + w / 2, y, label, size=13.5, color=color, weight="bold")


def arrow(p, q, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=7,head_width=3.5",
                                 color=color, lw=1.4, shrinkA=0, shrinkB=0,
                                 capstyle="round", joinstyle="miter"))


def line(xs, ys):
    ax.plot(xs, ys, color=INK, lw=1.4, solid_capstyle="round")


def frames(x, cy, n=10, w=12, h=22):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w, cy - h / 2), w, h, fc=CELL, ec=INK, lw=1.2))
    return x + n * w


def handwriting(x, cy, width):
    img = np.asarray(Image.open(LINE_IMAGE).convert("L").crop(LINE_CROP))
    h = width * img.shape[0] / img.shape[1]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(x, x + width, cy - h / 2, cy + h / 2),
              interpolation="lanczos", zorder=1)
    ax.add_patch(Rectangle((x, cy - h / 2), width, h, fc="none", ec="#8b93a1", lw=0.8, zorder=2))
    return h


text(20, 492, "Модель учится по двум функциям потерь", size=15, ha="left")
pill(20, 455, 60, "CTC", BLUE, BLUE_FILL)
pill(92, 455, 186, "по последовательности", ORANGE, ORANGE_FILL)

ROW = 275
img_h = handwriting(16, ROW, 220)
text(126, ROW - img_h / 2 - 18, "Строка", size=15)
note(126, ROW - img_h / 2 - 39, "высота 96 px, ширина по аспекту")

arrow((242, ROW), (263, ROW))
block(344, ROW, 162, 64, "Свёрточный stem", "ResNet-блоки")
note(344, ROW - 48, "ширина ÷ 8,")
note(344, ROW - 64, "высота свёрнута")

arrow((425, ROW), (442, ROW))
cells_end = frames(444, ROW)
text(504, ROW + 25, "кадры", size=14, weight="bold")
note(504, ROW - 27, "один на 8 px ширины,")
note(504, ROW - 43, "вектор 384")

arrow((cells_end + 2, ROW), (588, ROW))
block(665, ROW, 150, 80, "Энкодер", "transformer, 12 слоёв")
note(665, ROW - 56, "общий для обеих голов")

CTC_Y, DEC_Y, SPINE = 360, 195, 760
line([740, SPINE], [ROW, ROW])
line([SPINE, SPINE], [DEC_Y, CTC_Y])
arrow((SPINE, CTC_Y), (782, CTC_Y))
arrow((SPINE, DEC_Y), (782, DEC_Y))

block(852, CTC_Y, 140, 50, "CTC-голова", color=BLUE, fc=BLUE_FILL, ec=BLUE)
note(852, CTC_Y - 41, "символ или пусто")
note(852, CTC_Y - 57, "на каждый кадр")

ANN = 950
text(ANN, 400, "Чтение за один проход", size=14, color=BLUE, weight="bold", ha="left")
mono(ANN, 373, "п п _ р и _ в в е т")
glue = text(ANN, 347, "склейка", color=GREY, ha="left")
gx = right_edge(glue) + 6
arrow((gx, 347), (gx + 18, 347))
mono(gx + 24, 347, "привет", size=14, weight="bold")

block(852, DEC_Y, 140, 64, "Декодер", "transformer, 4 слоя", color=ORANGE, fc=ORANGE_FILL, ec=ORANGE)
note(842, DEC_Y - 48, "смотрит на кадры", ha="right")
note(842, DEC_Y - 64, "и на уже написанное", ha="right")

text(ANN, 240, "Следующий символ", size=14, color=ORANGE, weight="bold", ha="left")
steps = mono(ANN, 212, "п → р → и →")
qx = right_edge(steps) + 6
rounded(qx, 201, 26, 22, fc=ORANGE_FILL, ec=ORANGE, lw=1.2, ls=(0, (4, 3)), r=3)
mono(qx + 13, 212, "?", size=14, ha="center", color=ORANGE, weight="bold")
text(ANN, 184, "буква за буквой", color=GREY, ha="left")

arrow((852, DEC_Y - 32), (852, 107), color=ORANGE)
rounded(727, 65, 250, 42, fc=ORANGE_FILL, ec="none", r=6)
mono(852, 86, "п → р → и → в → е → т → [END]", size=14, ha="center")

text(20, 44, "35.3M параметров: stem 4.4M · энкодер 21.3M · декодер 9.5M", color=GREY, ha="left")
text(20, 22, "алфавит: 152 символа, русский и английский", color=GREY, ha="left")

out = Path(__file__).resolve().parent
fig.savefig(out / "model_scheme.svg", facecolor="white")
fig.savefig(out / "model_scheme.png", dpi=200, facecolor="white")
