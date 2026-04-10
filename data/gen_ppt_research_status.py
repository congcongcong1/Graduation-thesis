import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import subprocess, numpy as np

# ── 中文字体 ────────────────────────────────────────────
result = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True)
font_path = None
for line in result.stdout.splitlines():
    if 'WenQuanYi' in line:
        p = line.split(':')[0].strip()
        if p.endswith('.ttf') or p.endswith('.otf'):
            font_path = p; break
if not font_path:
    for line in result.stdout.splitlines():
        p = line.split(':')[0].strip()
        if p.endswith('.ttf') or p.endswith('.otf'):
            font_path = p; break
ZH = fm.FontProperties(fname=font_path)

C_DARK   = '#1e4d3c'
C_MID    = '#2e7d5e'
C_TEAL   = '#3d9e80'
C_WHITE  = '#ffffff'
C_GRAY   = '#f2f2f0'
C_TEXT   = '#333333'
C_BORDER = '#5bb89a'

fig = plt.figure(figsize=(16, 9), facecolor=C_GRAY)

# 顶部进度条
ax_top = fig.add_axes([0, 0.935, 1, 0.065], facecolor=C_MID)
ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1); ax_top.axis('off')
ax_top.add_patch(FancyBboxPatch((0, 0), 0.28, 1, boxstyle='square', fc=C_DARK, ec='none', zorder=2))
ax_top.text(0.14, 0.5, '1.2  国内外研究现状',
    ha='center', va='center', fontsize=15, color=C_WHITE,
    fontproperties=ZH, fontweight='bold', zorder=3)
ax_top.add_patch(FancyBboxPatch((0.28, 0.65), 0.30, 0.22, boxstyle='square', fc=C_TEAL, ec='none', alpha=0.6))

# 底部条
ax_bot = fig.add_axes([0, 0, 1, 0.055], facecolor=C_MID)
ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1); ax_bot.axis('off')
ax_bot.text(0.5, 0.5, '编码器感知前处理方法演进',
    ha='center', va='center', fontsize=13, color=C_WHITE,
    fontproperties=ZH, fontweight='bold')

# ── 左侧：时间轴演进图 ──────────────────────────────────
ax = fig.add_axes([0.02, 0.07, 0.62, 0.86], facecolor=C_GRAY)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

methods = [
    ('2011', 'Down-sampling+SR',  '间接码率约束，无编码器感知',    False, '#b0bec5'),
    ('2019', 'RR-DnCNN v2.0',    '联合优化误差，端到端重训练',    False, '#90a4ae'),
    ('2023', 'EA-MCTF',          '提取QP等特征，传统自适应滤波',  True,  '#78909c'),
    ('2023', 'RPP (bilibili)',   '代理网络间接感知，百万级CNN',   True,  '#4db6ac'),
    ('2024', 'RAPNet',           '真实编码集交互，<1M参数，极快', True,  '#26a69a'),
    ('本文', '本文方案',          'NVENC真实反馈，113K / 5.4ms',  True,  C_DARK),
]

TL_X   = 1.3
Y0     = 9.0
YSTEP  = 1.42

# 时间轴竖线
ax.annotate('', xy=(TL_X, 0.5), xytext=(TL_X, Y0 + 0.4),
    arrowprops=dict(arrowstyle='->', color=C_MID, lw=2.2))

for i, (yr, name, desc, aware, col) in enumerate(methods):
    y = Y0 - i * YSTEP
    is_last = (i == len(methods) - 1)

    # 圆点
    ax.plot(TL_X, y, 'o', markersize=12, color=col, zorder=5)
    ax.plot(TL_X, y, 'o', markersize=7,  color=C_WHITE, zorder=6)

    # 横接线
    ax.plot([TL_X, TL_X + 0.4], [y, y], color=col, lw=1.5)

    # 卡片框
    fc = col if is_last else C_WHITE
    tc = C_WHITE if is_last else C_DARK
    card = FancyBboxPatch((TL_X + 0.42, y - 0.52), 5.9, 1.06,
        boxstyle='round,pad=0.07',
        fc=fc, ec=col, linewidth=2.2 if is_last else 1.4, zorder=4)
    ax.add_patch(card)

    # 年份
    yr_col = C_TEAL if not is_last else '#a5d6c8'
    ax.text(TL_X + 0.78, y + 0.24, yr,
        fontsize=9.5, color=yr_col, fontproperties=ZH, fontweight='bold', va='center')

    # 方法名
    ax.text(TL_X + 0.78, y - 0.12, name,
        fontsize=12, color=tc, fontproperties=ZH, fontweight='bold', va='center')

    # 描述
    ax.text(TL_X + 4.2, y, desc,
        fontsize=10, color=tc, fontproperties=ZH, va='center', ha='center')

    # 编码器感知标签
    if aware:
        tag_c = C_TEAL if not is_last else '#a5d6c8'
        ax.text(TL_X + 6.05, y, '编码器感知',
            fontsize=9, color=tag_c, fontproperties=ZH,
            va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.28', fc='none', ec=tag_c, lw=1.3))

# 本文标注箭头
last_y = Y0 - 5 * YSTEP
ax.annotate('',
    xy=(TL_X + 0.42, last_y),
    xytext=(TL_X - 0.4, last_y - 0.55),
    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8,
                    connectionstyle='arc3,rad=-0.25'))
ax.text(TL_X - 0.42, last_y - 0.82, '本文切入点',
    fontsize=9, color=C_DARK, fontproperties=ZH, ha='center', fontweight='bold')

# ── 右侧标注框（3要点）──────────────────────────────────
ax_r = fig.add_axes([0.655, 0.10, 0.328, 0.82], facecolor='none')
ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis('off')

# 外框
outer = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle='round,pad=0.03', fc=C_WHITE, ec=C_BORDER, linewidth=2.5)
ax_r.add_patch(outer)

# 左侧三角指针（用多边形模拟）
tri_pts = np.array([[-0.04, 0.83], [0.0, 0.77], [0.0, 0.89]])
tri = plt.Polygon(tri_pts, fc=C_BORDER, ec=C_BORDER, zorder=5, transform=ax_r.transData)
ax_r.add_patch(tri)
tri2_pts = np.array([[-0.018, 0.83], [0.015, 0.785], [0.015, 0.875]])
tri2 = plt.Polygon(tri2_pts, fc=C_WHITE, ec=C_WHITE, zorder=6, transform=ax_r.transData)
ax_r.add_patch(tri2)

points = [
    ('1.', '两条研究路线分化',
     '全AI端到端架构追求极致压缩效率，\n但算力庞大难以在边缘设备部署；\nAI增强协处理架构兼容现有编码标准，\n正成为工业界与学术界研究热点。'),
    ('2.', '主流大模型算力过重',
     'VRT/RVRT等主流模型参数数千万、\n推理时延超过200ms，无法满足\nNVENC等实时硬件编码链路的\n部署约束，实用化存在明显瓶颈。'),
    ('3.', '轻量化编码器感知成趋势',
     '前处理正从离线复原转向与编码器\n紧密耦合的在线优化；参数<1M、\n推理<10ms的轻量方案是工程落地\n的关键窗口，也是本文的切入方向。'),
]

starts = [0.665, 0.350, 0.035]
block_h = 0.30

for idx, (num, title, body) in enumerate(points):
    y0 = starts[idx]
    # 序号
    ax_r.text(0.075, y0 + block_h * 0.72, num,
        fontsize=19, color=C_TEAL, fontproperties=ZH,
        fontweight='bold', va='center', ha='center')
    # 标题
    ax_r.text(0.19, y0 + block_h * 0.72, title,
        fontsize=11.5, color=C_TEAL, fontproperties=ZH,
        fontweight='bold', va='center')
    # 分隔线
    ax_r.plot([0.06, 0.94], [y0 + block_h * 0.52]*2, color='#c8e6d8', lw=0.9)
    # 正文
    ax_r.text(0.08, y0 + block_h * 0.22, body,
        fontsize=9.2, color=C_TEXT, fontproperties=ZH,
        va='center', linespacing=1.55)

out = '/home/user/Graduation-thesis/data/ppt_research_status.png'
fig.savefig(out, dpi=220, bbox_inches='tight', facecolor=C_GRAY)
print('Saved:', out)
