"""Generate a clean pipeline overview figure for the LaTeX report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "figures" / "pipeline_overview.png"

STEPS = [
    "FNSPID news\n+ price histories",
    "Filtering +\ncleaning",
    "ESG keyword\nscreen",
    "FinBERT /\nClimateBERT scoring",
    "Weekly ticker\naggregation",
    "Market feature\nengineering",
    "Future risk\ntargets",
    "Time-aware training\n+ validation tuning",
    "Risk forecasts\n+ evaluation",
]


def _draw_box(ax, x, y, text, width=1.75, height=0.72):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#2f5597",
        facecolor="#dbe8f7",
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#1f1f1f", linespacing=1.15)
    return {
        "top": (x, y + height / 2),
        "bottom": (x, y - height / 2),
        "left": (x - width / 2, y),
        "right": (x + width / 2, y),
    }


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            color="#333333",
            shrinkA=0,
            shrinkB=0,
        )
    )


def _path_arrow(ax, points):
    path = MplPath(points, [MplPath.MOVETO, *([MplPath.LINETO] * (len(points) - 2)), MplPath.LINETO])
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            color="#333333",
            shrinkA=0,
            shrinkB=0,
        )
    )


def _connect_rows(ax, from_box, to_box):
    x_from, y_from = from_box["bottom"]
    x_to, y_to = to_box["top"]
    mid_y = (y_from + y_to) / 2
    _path_arrow(
        ax,
        [
            (x_from, y_from - 0.03),
            (x_from, mid_y),
            (x_to, mid_y),
            (x_to, y_to + 0.03),
        ],
    )


def main() -> None:
    box_w, box_h = 1.75, 0.72
    col_gap = 0.48
    row_gap = 1.28
    xs = [1.05 + i * (box_w + col_gap) for i in range(3)]
    ys = [3.05, 3.05 - row_gap, 3.05 - 2 * row_gap]
    canvas_w = xs[-1] + box_w / 2 + 0.55
    canvas_h = ys[0] + box_h / 2 + 0.45

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h))
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.axis("off")

    positions = [(xs[c], ys[r]) for r in range(3) for c in range(3)]
    boxes = [_draw_box(ax, x, y, label, width=box_w, height=box_h) for (x, y), label in zip(positions, STEPS)]

    h_gap = 0.20
    for row in range(3):
        base = row * 3
        y = ys[row]
        for col in range(2):
            left = boxes[base + col]
            right = boxes[base + col + 1]
            _arrow(ax, (left["right"][0] + h_gap, y), (right["left"][0] - h_gap, y))

    _connect_rows(ax, boxes[2], boxes[3])
    _connect_rows(ax, boxes[5], boxes[6])

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
