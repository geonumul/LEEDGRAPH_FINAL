"""Generate Figure1_pipeline.png (portrait layout, matplotlib-only, Korean labels)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results" / "figures" / "Figure1_pipeline.png"

# ── Korean font (auto-detect an installed CJK font) ─────────────────────────
FONT_CANDIDATES = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _name in FONT_CANDIDATES:
    if any(_name in inst for inst in _installed):
        plt.rcParams["font.family"] = _name
        print(f"[font] using: {_name}")
        break
else:
    print("[font] WARNING: no Korean font found - text may render as boxes. "
          "Install one of: " + ", ".join(FONT_CANDIDATES))
plt.rcParams["axes.unicode_minus"] = False

TITLE_FS = 18
NODE_TITLE_FS = 13
NODE_BODY_FS = 10
TRACK_FS = 11

COLORS = {
    "input":    ("#D6E4F5", "#7FA6D9"),
    "parsing":  ("#DAE8FC", "#6B8FC9"),
    "rule":     ("#D5E8D4", "#82B366"),
    "hallu":    ("#FFF2CC", "#D6B656"),
    "output":   ("#E8E8E8", "#9E9E9E"),
    "analysis": ("#FFE6CC", "#D79B00"),
    "result":   ("#F8CECC", "#B85450"),
}


def add_box(ax, x, y, w, h, title, subtitle, color_key, title_fs=NODE_TITLE_FS):
    face, edge = COLORS[color_key]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h * 0.70,
        title,
        ha="center", va="center",
        fontsize=title_fs, fontweight="bold",
        color="#222222",
    )
    ax.text(
        x + w / 2, y + h * 0.30,
        subtitle,
        ha="center", va="center",
        fontsize=NODE_BODY_FS,
        color="#333333",
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#555555",
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    fig.suptitle(
        "LEEDGRAPH 파이프라인 구조\n"
        "한국 LEED 460건 | v2.0~v4.1 → v5 표준화",
        fontsize=TITLE_FS, fontweight="bold", y=0.985,
    )

    box_w = 5.6
    box_h = 1.15
    cx = 5.0

    layout = [
        ("input",   "Step 1 · 입력", "460건 LEED 스코어카드 (PDF)\nv2.0~v4.1, 한국",       12.10),
        ("parsing", "Step 2 · 파싱", "PDF 파싱 + CSV 매칭\nloader.py, pdfplumber",          10.40),
    ]
    for color_key, title, subtitle, y_center in layout:
        add_box(ax, cx - box_w / 2, y_center - box_h / 2, box_w, box_h,
                title, subtitle, color_key)

    add_arrow(ax, cx, 12.10 - box_h / 2, cx, 10.40 + box_h / 2)

    # ── Step 3: parallel split ──────────────────────────────────────────────
    split_w = 4.2
    split_h = 1.85
    left_x = 0.4
    right_x = 10.0 - 0.4 - split_w
    split_y = 8.55 - split_h / 2

    def draw_track_box(x, color_key, track_label, title, body):
        face, edge = COLORS[color_key]
        box = FancyBboxPatch(
            (x, split_y), split_w, split_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8, facecolor=face, edgecolor=edge,
        )
        ax.add_patch(box)
        ax.text(x + split_w / 2, split_y + split_h * 0.83, track_label,
                ha="center", va="center", fontsize=TRACK_FS,
                fontweight="bold", color=edge)
        ax.text(x + split_w / 2, split_y + split_h * 0.55, title,
                ha="center", va="center", fontsize=NODE_TITLE_FS,
                fontweight="bold", color="#222222")
        ax.text(x + split_w / 2, split_y + split_h * 0.22, body,
                ha="center", va="center", fontsize=NODE_BODY_FS, color="#333333")

    draw_track_box(
        left_x, "rule",
        "경로 1: 규칙 기반 (93.0%)",
        "Step 3a · 규칙 매핑",
        "v2.0~v4.1 → v5\n107개 매핑 규칙",
    )
    draw_track_box(
        right_x, "hallu",
        "경로 2: 검증 경로 (7.0%)",
        "Step 3b · 할루시네이션 검사",
        "달성률 drift < 20%p\n5개 수학 제약",
    )

    add_arrow(ax, cx, 10.40 - box_h / 2, left_x + split_w / 2, split_y + split_h)
    add_arrow(ax, cx, 10.40 - box_h / 2, right_x + split_w / 2, split_y + split_h)

    # ── Steps 4-6 ───────────────────────────────────────────────────────────
    rest = [
        ("output",   "Step 4 · 출력", "v5 표준화 점수\nproject_features.parquet (N=460)", 5.95),
        ("analysis", "Step 5 · 분석", "Random Forest + SHAP\n5-Fold CV, n=460",            3.95),
        ("result",   "Step 6 · 결과", "등급 결정 요인\nEA > EQ > WE",                       1.95),
    ]
    for color_key, title, subtitle, y_center in rest:
        add_box(ax, cx - box_w / 2, y_center - box_h / 2, box_w, box_h,
                title, subtitle, color_key)

    add_arrow(ax, left_x + split_w / 2,  split_y, cx, 5.95 + box_h / 2)
    add_arrow(ax, right_x + split_w / 2, split_y, cx, 5.95 + box_h / 2)
    add_arrow(ax, cx, 5.95 - box_h / 2, cx, 3.95 + box_h / 2)
    add_arrow(ax, cx, 3.95 - box_h / 2, cx, 1.95 + box_h / 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
