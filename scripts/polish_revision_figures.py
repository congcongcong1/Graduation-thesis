from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUBJECTIVE_DIR = ROOT / "data" / "主观图片"
FIGURE_DIR = ROOT / "data" / "figures_clean"


def chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return font_manager.FontProperties(family=name)
    return None


ZH = chinese_font()


def zh_font(size):
    if ZH is None:
        return None
    prop = ZH.copy()
    prop.set_size(size)
    return prop


def imread_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def save_ycbcr_panel():
    rows = [
        (
            "Y 通道（亮度）",
            "处理前Y通道图片.png",
            "处理后Y通道图片.png",
            3,
        ),
        (
            "Cb 通道（蓝色差）",
            "处理前Cb通道图片.png",
            "处理后Cb通道图片.png",
            4,
        ),
        (
            "Cr 通道（红色差）",
            "处理前Cr通道图片.png",
            "处理后Cr通道图片.png",
            3,
        ),
    ]

    fig = plt.figure(figsize=(19.2, 12.8), dpi=220)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.055,
        hspace=0.10,
        left=0.205,
        right=0.975,
        top=0.895,
        bottom=0.045,
    )

    titles = [
        "Baseline\n（NVENC 编码重建）",
        "RAPNet\n（前处理 + NVENC 编码重建）",
        "差异图（暖=RAPNet 增亮，冷=减暗）\n99 pct 归一化",
    ]
    title_font = zh_font(22)
    row_font = zh_font(29)

    for r, (label, before_name, after_name, vmax) in enumerate(rows):
        before = imread_gray(SUBJECTIVE_DIR / before_name)
        after = imread_gray(SUBJECTIVE_DIR / after_name)
        h = min(before.shape[0], after.shape[0])
        w = min(before.shape[1], after.shape[1])
        before = before[:h, :w]
        after = after[:h, :w]
        diff = after - before
        p99 = np.percentile(np.abs(diff), 99)
        if p99 < 1e-6:
            p99 = 1.0
        diff_norm = np.clip(diff / p99 * vmax, -vmax, vmax)

        axes = [fig.add_subplot(gs[r, c]) for c in range(3)]
        cax = fig.add_subplot(gs[r, 3])
        axes[0].imshow(before, cmap="gray", vmin=0, vmax=255)
        axes[1].imshow(after, cmap="gray", vmin=0, vmax=255)
        im = axes[2].imshow(diff_norm, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        for c, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if r == 0:
                ax.set_title(titles[c], fontproperties=title_font, pad=22)

        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=24, length=5)

        y = axes[0].get_position().y0 + axes[0].get_position().height / 2
        fig.text(
            0.192,
            y,
            label,
            ha="right",
            va="center",
            fontproperties=row_font,
            bbox=dict(facecolor="#f1f1f1", edgecolor="none", pad=9),
        )

    out = SUBJECTIVE_DIR / "subjective_ycbcr_panel_large_labels.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def crop_plot(src_name, dst_name, box):
    src = ROOT / "data" / src_name
    dst = FIGURE_DIR / dst_name
    img = Image.open(src)
    img.crop(box).save(dst)


def crop_plot_with_inner_legend(src_name, dst_name, plot_box, legend_box, legend_pos, legend_scale=0.78):
    src = ROOT / "data" / src_name
    dst = FIGURE_DIR / dst_name
    img = Image.open(src).convert("RGBA")
    plot = img.crop(plot_box)
    legend = img.crop(legend_box)
    if legend_scale != 1.0:
        new_size = (
            max(1, int(legend.width * legend_scale)),
            max(1, int(legend.height * legend_scale)),
        )
        legend = legend.resize(new_size, Image.Resampling.LANCZOS)
    plot.alpha_composite(legend, dest=legend_pos)
    plot.convert("RGB").save(dst)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_ycbcr_panel()
    crop_plot("图片Preset对比实验.png", "preset_compare_enlarged.png", (105, 70, 1910, 1145))
    crop_plot_with_inner_legend(
        "图片下采样算法.png",
        "downsample_compare_enlarged.png",
        plot_box=(0, 18, 1544, 1530),
        legend_box=(1588, 68, 2025, 365),
        legend_pos=(150, 88),
    )


if __name__ == "__main__":
    main()
