#!/usr/bin/env python
# coding: utf-8

# # Flare Conjunction Ratio Decay Curves by Class
#
# This notebook plots the **Conjunction Ratio (%)** as a function of the angular half-window $w$ (1°–50°) for solar flares of different classes. Five curves are shown:
#
# | Curve | Data Source | Color |
# |-------|-----------|-------|
# | Total | `sf_decay_boundary.csv` | Black |
# | C-Class | `sf_c_class_decay_boundary.csv` | Red |
# | B-Class | `sf_b_class_decay_boundary.csv` | Blue |
# | M-Class | `sf_m_class_decay_boundary.csv` | Green |
# | X-Class | `sf_x_class_decay_boundary.csv` | Purple |
#
# Key annotations:
# - **Horizontal dashed line** at 100% marks the random baseline.
# - **Vertical dashed lines** mark the decay boundaries: $w=21$ (C-Class) and $w=27$ (Total).


import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, figsize_double, save_dual

apply_acta_style("double")

# ── 路径: 兼容从项目根目录或 notebook 目录启动 ──
def resolve_project_root():
    candidates = [
        os.getcwd(),
        os.path.abspath(os.path.join(os.getcwd(), '..', '..')),
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, 'data', 'ready')):
            return cand
    raise FileNotFoundError('无法定位项目根目录: 缺少 data/ready')

BASE = resolve_project_root()
data_dir = os.path.join(BASE, 'results', '04_asymmetric', 'sf')
out_dir = os.path.join(BASE, 'results', '04_asymmetric')
fdr_path = os.path.join(BASE, 'results', '05_multidimensional', 'fdr_audit', 'decay_boundary_fdr.csv')

# ── 读取 5 个 CSV ──
files = {
    'Total':   'sf_decay_boundary.csv',
    'B-Class': 'sf_b_class_decay_boundary.csv',
    'C-Class': 'sf_c_class_decay_boundary.csv',
    'M-Class': 'sf_m_class_decay_boundary.csv',
    'X-Class': 'sf_x_class_decay_boundary.csv',
}
colors = {
    'Total': 'black',
    'B-Class': '#1F4ED8',
    'C-Class': '#D62728',
    'M-Class': '#187A2A',
    'X-Class': '#7A1F8A',
}

data = {}
for label, fname in files.items():
    df = pd.read_csv(os.path.join(data_dir, fname))
    df['Asym'] = df['Conj_Ratio'] - df['Opp_Ratio']
    data[label] = df

# ── 读取 FDR 校正数据 ──
fdr_df = pd.read_csv(fdr_path)

# ── 保存汇总坐标数据 CSV ──
df_list = []
for label, df in data.items():
    temp_df = df[['Window', 'Conj_Ratio', 'Opp_Ratio', 'Asym', 'Conj_Z']].copy()
    temp_df['Class'] = label
    df_list.append(temp_df)

csv_out = pd.concat(df_list, ignore_index=True)
cols = ['Class', 'Window', 'Conj_Ratio', 'Opp_Ratio', 'Asym', 'Conj_Z']
csv_out = csv_out[cols].round(4)
csv_out.to_csv(os.path.join(out_dir, 'Fig04_plot_data.csv'), index=False)
print(f"已保存坐标数据: Fig04_plot_data.csv  ({len(csv_out)} 行)")

# ── 轴范围（固定） ──
left_lo, left_hi = 90, 110
right_lo, right_hi = -10, 10
z_lo, z_hi = -1, 3.5
ratio_ticks = [90, 95, 100, 105, 110]
asym_ticks = [-10, -5, 0, 5, 10]
z_ticks = [-1, 0, 1, 2, 3]

# ── 绘图：论文全页宽度，比例 10x6 ──
fig, axes = plt.subplots(2, 3, figsize=figsize_double(aspect=0.55))
axes_flat = axes.flatten()

labels_order = ['Total', 'B-Class', 'C-Class', 'M-Class', 'X-Class']
panel_letters = ['a', 'b', 'c', 'd', 'e']
panel_axes = {}

for i, label in enumerate(labels_order):
    ax = axes_flat[i]
    panel_axes[label] = ax
    df = data[label]
    w = df['Window']
    color = colors[label]

    # 右轴：Asym 柱状图（底层背景）
    ax2 = ax.twinx()
    ax2.bar(w, df['Asym'], width=0.8, color='#D4A766', zorder=1)
    ax2.axhline(y=0, color='gray', linewidth=0.4, linestyle='-', zorder=0)
    ax2.set_ylim(right_lo, right_hi)
    ax2.set_yticks(asym_ticks)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 逻辑：仅右上角子图 (i=2, C-Class) 保留右侧刻度和标签
    if i == 2:
        ax2.tick_params(axis='y', labelcolor='#996600', labelsize=6.5)
    else:
        ax2.tick_params(axis='y', labelright=False) # 隐藏其他右轴刻度数字

    # 左轴：Ratio 曲线（顶层前景）
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(w, df['Conj_Ratio'], '-', color=color, linewidth=1.4,
            label='Conj Ratio', zorder=5)
    ax.plot(w, df['Opp_Ratio'], '--', color=color, linewidth=1.0,
            label='Opp Ratio', zorder=5)
    ax.axhline(y=100, color='gray', linewidth=0.6, linestyle='-', zorder=4)
    ax.set_ylim(left_lo, left_hi)
    ax.set_xlim(0.5, 50.5)
    ax.set_yticks(ratio_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_title(f'({panel_letters[i]}) {label}', loc='left', color=color)

    # 逻辑：仅左侧边缘 (i=0, 3) 保留左侧刻度数字
    if i not in [0, 3]:
        ax.tick_params(axis='y', labelleft=False)

    # 逻辑：仅底边 (i=3, 4) 保留 X 轴刻度数字
    if i not in [3, 4]:
        ax.tick_params(axis='x', labelbottom=False)

def annotate_bh_fdr(label, source_file, xytext, color):
    row = fdr_df[
        (fdr_df['Source_File'] == source_file)
        & (fdr_df['Window'] == 2)
    ]
    if len(row) != 1:
        return
    q_val = row.iloc[0]['Conj_q_file']
    rc_val = row.iloc[0]['Conj_Ratio']
    panel_axes[label].annotate(
        f'BH-FDR $q={q_val + 1e-9:.3f}$',
        xy=(2, rc_val), xytext=xytext,
        fontsize=6.2, color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=0.8, shrinkB=3),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#D6D6D6', linewidth=0.35),
    )

# ── FDR 标注: Total 支撑窗口选择; C-Class 支撑信号来源 ──
annotate_bh_fdr('Total', 'sf_decay_boundary.csv', (12, 105.0), '#922b21')
annotate_bh_fdr('C-Class', 'sf_c_class_decay_boundary.csv', (13, 105.8), '#922b21')

# ── (a) Total 加 Conj/Opp 线型小图例(全图唯一一处,(b)-(e) 共享同约定) ──
ratio_handles = [
    Line2D([], [], color='gray', lw=1.4,          label=r'$R_{\mathrm{C}}$'),
    Line2D([], [], color='gray', lw=1.0, ls='--', label=r'$R_{\mathrm{O}}$'),
]
leg_a = panel_axes['Total'].legend(
    handles=ratio_handles, loc='lower left',
    fontsize=6.5, handlelength=1.4, handletextpad=0.4,
    borderpad=0.3, labelspacing=0.2,
)
leg_a.get_frame().set_edgecolor('#D6D6D6')
leg_a.get_frame().set_linewidth(0.35)

# ── 子图 6：汇总 Z-score ──
ax6 = axes_flat[5]
for label in labels_order:
    df = data[label]
    ax6.plot(df['Window'], df['Conj_Z'], '-', color=colors[label],
             linewidth=1.2)
ax6.axhline(y=1.96, color='gray', linewidth=0.8, linestyle='--')
ax6.text(49, 2.05, '95% threshold\n($Z=1.96$)',
         ha='right', va='bottom', multialignment='right',
         fontsize=6.0, color='#7A7A7A')
ax6.set_xlim(0.5, 50.5)
ax6.set_ylim(z_lo, z_hi)
ax6.set_yticks(z_ticks)
ax6.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# 布局逻辑：底边和右边
ax6.yaxis.tick_right()                                  # 将刻度移至右侧
ax6.yaxis.set_label_position("right")                   # 将标签移至右侧
ax6.set_title(r'(f) Conj $Z$-score (all classes)', loc='left')

# 共享轴标签，避免小面板重复占用绘图区
fig.text(0.5, 0.045, r'Window half-width, $w/(^\circ)$',
         ha='center', va='center')
fig.text(0.025, 0.52, r'$(R_{\mathrm{C}}, R_{\mathrm{O}})/\mathrm{\%}$',
         ha='center', va='center', rotation='vertical')
fig.text(0.955, 0.705, r'$\mathrm{Asym}/\mathrm{\%}$',
         ha='center', va='center', rotation='vertical', color='#996600')
fig.text(0.955, 0.32, r'$Z_{\mathrm{C}}$',
         ha='center', va='center', rotation='vertical')

# ── 紧凑布局：留出共享坐标轴，让每个子图更宽 ──
plt.subplots_adjust(left=0.09, right=0.91, bottom=0.13, top=0.91,
                    wspace=0.16, hspace=0.32)

# ── 保存 EPS ──
eps_path = os.path.join(out_dir, 'Fig04_flare_decay.eps')
save_dual(fig, eps_path)
print(f"已保存图形: {eps_path}")

plt.show()
