"""
生成研究背景章节动机图：
展示NVENC低码率编码块效应 vs RAPNet前处理后的改善效果
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import matplotlib.font_manager as fm
import subprocess

# ── 中文字体 ──────────────────────────────────────────
result = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True)
font_path = None
for line in result.stdout.splitlines():
    if 'WenQuanYi' in line or 'wenquanyi' in line.lower():
        path = line.split(':')[0].strip()
        if path.endswith('.ttf') or path.endswith('.otf'):
            font_path = path
            break
if font_path is None:
    for line in result.stdout.splitlines():
        path = line.split(':')[0].strip()
        if path.endswith('.ttf') or path.endswith('.otf'):
            font_path = path
            break
zh_font = fm.FontProperties(fname=font_path) if font_path else None

def T(s):
    return s  # already unicode

# ── 载入图片 ──────────────────────────────────────────
img_base = Image.open('/home/user/Graduation-thesis/data/主观图片/主观对比图原始.png').convert('RGB')
img_rap  = Image.open('/home/user/Graduation-thesis/data/主观图片/主观对比图处理后.png').convert('RGB')
W, H = img_base.size
img_rap_r = img_rap.resize((W, H), Image.LANCZOS)

base = np.array(img_base)
rap  = np.array(img_rap_r)

# ── 两个局部放大区域 (手工微调使可读性最佳) ─────────────
# Zoom 1: 礁石与水体交界（高差异区）
z1 = (480, 270, 600, 370)   # (x0, y0, x1, y1)
# Zoom 2: 平坦水面（低方差，块效应更明显）
z2 = (160, 250, 280, 350)

ZOOM_SCALE = 2.8   # 放大倍数

def crop_zoom(arr, box, scale):
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(patch).resize((new_w, new_h), Image.LANCZOS))

z1_base = crop_zoom(base, z1, ZOOM_SCALE)
z1_rap  = crop_zoom(rap,  z1, ZOOM_SCALE)
z2_base = crop_zoom(base, z2, ZOOM_SCALE)
z2_rap  = crop_zoom(rap,  z2, ZOOM_SCALE)

# ── 颜色方案 ──────────────────────────────────────────
COL_BASE = '#e05555'    # 红色框 = Baseline (有块效应)
COL_RAP  = '#4a9eda'    # 蓝色框 = RAPNet

# ── 布局 ──────────────────────────────────────────────
# 行1: Baseline全图 | RAPNet全图  (各占一半)
# 行2: Zoom1-Base | Zoom1-RAP | Zoom2-Base | Zoom2-RAP
fig = plt.figure(figsize=(14, 9))

gs = fig.add_gridspec(
    2, 4,
    height_ratios=[1.8, 1],
    hspace=0.06,
    wspace=0.04,
    left=0.02, right=0.98,
    top=0.91, bottom=0.05
)

ax_bl = fig.add_subplot(gs[0, :2])   # Baseline全图 (左2列)
ax_rp = fig.add_subplot(gs[0, 2:])   # RAPNet全图 (右2列)

ax_z1b = fig.add_subplot(gs[1, 0])   # Zoom1 Baseline
ax_z1r = fig.add_subplot(gs[1, 1])   # Zoom1 RAPNet
ax_z2b = fig.add_subplot(gs[1, 2])   # Zoom2 Baseline
ax_z2r = fig.add_subplot(gs[1, 3])   # Zoom2 RAPNet

def show_img(ax, arr, title, title_color='black', zoom_boxes=None):
    ax.imshow(arr)
    ax.set_title(title, color=title_color, fontsize=11,
                 fontproperties=zh_font, pad=4, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if zoom_boxes:
        for (box, ec, label) in zoom_boxes:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=2, edgecolor=ec, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x0+2, y0-4, label, color=ec, fontsize=8,
                    fontproperties=zh_font, fontweight='bold')

def show_zoom(ax, arr, ec, label):
    ax.imshow(arr)
    ax.set_title(label, color=ec, fontsize=9,
                 fontproperties=zh_font, pad=3, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(ec)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

# 全图 + 框标注
show_img(ax_bl, base,
         'Baseline（NVENC直接编码，QP=45）',
         title_color=COL_BASE,
         zoom_boxes=[
             (z1, COL_BASE, 'Zoom 1'),
             (z2, '#e09530', 'Zoom 2'),
         ])
show_img(ax_rp, rap,
         'RAPNet前处理后编码（QP=45）',
         title_color=COL_RAP,
         zoom_boxes=[
             (z1, COL_BASE, 'Zoom 1'),
             (z2, '#e09530', 'Zoom 2'),
         ])

# 分隔线（中间加一条细竖线）
line = matplotlib.lines.Line2D([0.5, 0.5], [0.05, 0.97],
                                transform=fig.transFigure,
                                color='#cccccc', linewidth=1, linestyle='--')
fig.add_artist(line)

# Zoom 放大图
show_zoom(ax_z1b, z1_base, COL_BASE, 'Zoom 1 — Baseline')
show_zoom(ax_z1r, z1_rap,  COL_RAP,  'Zoom 1 — RAPNet')
show_zoom(ax_z2b, z2_base, '#e09530', 'Zoom 2 — Baseline')
show_zoom(ax_z2r, z2_rap,  '#4ab87a', 'Zoom 2 — RAPNet')

# 总标题说明
fig.text(0.5, 0.955,
         'NVENC H.265低码率编码重建帧对比（BQTerrace序列，QP=45）',
         ha='center', fontsize=12.5,
         fontproperties=zh_font, fontweight='bold', color='#333333')
fig.text(0.5, 0.026,
         '注：两帧均为编码后解码重建帧，非原始帧。左侧可见平坦区块效应与边缘振铃；右侧经RAPNet前处理后感知质量改善。',
         ha='center', fontsize=8.5,
         fontproperties=zh_font, color='#555555')

out_path = '/home/user/Graduation-thesis/data/主观图片/background_blocking_comparison.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print('Saved:', out_path)
