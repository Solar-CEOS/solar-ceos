#!/usr/bin/env python
# coding: utf-8


"""
Fig05 黑子群信号稀释 — V4 双热力图版
  (a) R_C 热力图：Lifecycle + Area × w=1~5
  (b) A = R_C - R_O 不对称度热力图
  (c) 单行星 R_C 排名对比
  (d) ROC 可预测性对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, figsize_double, save_dual

apply_acta_style("double")

# ── 路径配置 ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
result_dir = str(PROJECT_ROOT / 'results' / '04_asymmetric')
sg_dir = os.path.join(result_dir, 'sg')
sf_dir = os.path.join(result_dir, 'sf')

COLOR_BASELINE = 'gray'
FRAME_EDGE = '#D6D6D6'

# ============================================================
# 数据读取
# ============================================================

algo1 = pd.read_csv(os.path.join(sg_dir, 'sg_algo1_total_pairs.csv'))

# --- 热力图数据：9 类别 × w=1~5, 分别取 R_C 和 R_O ---
categories_lifecycle = ['Daily', 'Onset', 'Dissipation', 'Duration']
area_groups_full = ['Small <100', 'Medium 100-500', 'Large 500-2000', 'XLarge >2000']
area_short_keys = ['Small', 'Medium', 'Large', 'XLarge']
display_labels = ['Daily', 'Onset', 'Dissip.', 'Dur.', 'All', 'Sm', 'Med', 'Lg', 'XL']
n_cats = len(display_labels)
windows = [1, 2, 3, 4, 5]

hm_rc = np.full((len(windows), n_cats), np.nan)  # R_C
hm_ro = np.full((len(windows), n_cats), np.nan)  # R_O
n_records = {}

# Lifecycle stages (cols 0-3)
for j, stage in enumerate(categories_lifecycle):
    for i, w in enumerate(windows):
        rc = algo1[(algo1['Stage'] == stage) & (algo1['Group'] == 'Total') &
                   (algo1['Window'] == w) & (algo1['Type'] == 'Conjunction')]
        ro = algo1[(algo1['Stage'] == stage) & (algo1['Group'] == 'Total') &
                   (algo1['Window'] == w) & (algo1['Type'] == 'Opposition')]
        if len(rc) > 0: hm_rc[i, j] = rc.iloc[0]['Ratio']
        if len(ro) > 0: hm_ro[i, j] = ro.iloc[0]['Ratio']
        if w == 1 and len(rc) > 0:
            n_records[stage] = int(rc.iloc[0]['N_Records'])

# All Total (col 4)
for i, w in enumerate(windows):
    rc = algo1[(algo1['Stage'] == 'All') & (algo1['Group'] == 'Total') &
               (algo1['Window'] == w) & (algo1['Type'] == 'Conjunction')]
    ro = algo1[(algo1['Stage'] == 'All') & (algo1['Group'] == 'Total') &
               (algo1['Window'] == w) & (algo1['Type'] == 'Opposition')]
    if len(rc) > 0: hm_rc[i, 4] = rc.iloc[0]['Ratio']
    if len(ro) > 0: hm_ro[i, 4] = ro.iloc[0]['Ratio']
    if w == 1 and len(rc) > 0:
        n_records['All'] = int(rc.iloc[0]['N_Records'])

# Area groups (cols 5-8)
for k, (grp, key) in enumerate(zip(area_groups_full, area_short_keys)):
    j = 5 + k
    for i, w in enumerate(windows):
        rc = algo1[(algo1['Stage'] == 'All') & (algo1['Group'] == grp) &
                   (algo1['Window'] == w) & (algo1['Type'] == 'Conjunction')]
        ro = algo1[(algo1['Stage'] == 'All') & (algo1['Group'] == grp) &
                   (algo1['Window'] == w) & (algo1['Type'] == 'Opposition')]
        if len(rc) > 0: hm_rc[i, j] = rc.iloc[0]['Ratio']
        if len(ro) > 0: hm_ro[i, j] = ro.iloc[0]['Ratio']
        if w == 1 and len(rc) > 0:
            n_records[key] = int(rc.iloc[0]['N_Records'])

# A = R_C - R_O
hm_a = hm_rc - hm_ro

# --- 单行星排名 ---
planet_masks = {1: 'Mercury', 2: 'Venus', 8: 'Mars',
                16: 'Jupiter', 32: 'Saturn', 64: 'Uranus', 128: 'Neptune'}

sf_cclass = pd.read_csv(os.path.join(sf_dir, 'sf_C_Class_subset_scan_no_earth.csv'))
flare_planet = sf_cclass[(sf_cclass['N_Planets'] == 1) &
                         (sf_cclass['Has_Earth'] == False) &
                         (sf_cclass['Window'] == 2)].copy()
flare_planet['Planet'] = flare_planet['Mask'].map(planet_masks)

sg_all = pd.read_csv(os.path.join(sg_dir, 'sg_subset_scan_no_earth.csv'))
sg_planet = sg_all[(sg_all['N_Planets'] == 1) & (sg_all['Has_Earth'] == False) &
                   (sg_all['Window'] == 1)].copy()
sg_planet['Planet'] = sg_planet['Mask'].map(planet_masks)

sg_daily_sub = pd.read_csv(os.path.join(sg_dir, 'sg_daily_subset_scan_no_earth.csv'))
sg_daily_planet = sg_daily_sub[(sg_daily_sub['N_Planets'] == 1) &
                               (sg_daily_sub['Has_Earth'] == False) &
                               (sg_daily_sub['Window'] == 1)].copy()
sg_daily_planet['Planet'] = sg_daily_planet['Mask'].map(planet_masks)

# --- ROC 数据 ---
roc_path = os.path.join(result_dir, '08_best_roc_data.json')
if not os.path.exists(roc_path):
    roc_path = os.path.join(result_dir, '00_exploratory/05_predictability_contrast/best_roc_data.json')
with open(roc_path) as f:
    roc_data = json.load(f)

print("数据读取完成")

# ============================================================
# 绘图：2 × 2
# ============================================================
# 比 figstyle_double() 宽 0.8" — 4 个 panel 信息密度大(双 heatmap + 7 行星
# 三色柱 + 6 行 ROC 注释),双栏 7.2" 容量边缘。EPS 8.0" 经 \includegraphics
# scale=0.89 渲染为 7.12",仍在双栏内,且印刷字号比 7.2" 版略大。
fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2 * 0.7))

# ── 共用函数：绘制热力图 ──
def format_count(n):
    if n >= 100_000:
        return f'{n / 1000:.0f}k'
    if n >= 10_000:
        return f'{n / 1000:.1f}k'
    if n >= 1_000:
        return f'{n / 1000:.1f}k'
    return f'{n}'


def draw_heatmap(ax, data, title, cmap_name, vmin, vmax, vcenter, cbar_label, fmt='{:.1f}'):
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    def contrast_text_color(value):
        mapped = float(np.clip(norm(value), 0, 1))
        red, green, blue, _ = cmap(mapped)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return 'white' if luminance < 0.50 else 'black'

    for i in range(len(windows)):
        for j in range(n_cats):
            val = data[i, j]
            if not np.isnan(val):
                color = contrast_text_color(val)
                ax.text(j, i, fmt.format(val), ha='center', va='center',
                        fontsize=5.7, color=color, fontweight='bold')

    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(len(windows)))
    ax.set_yticklabels([f'$w$={w}°' for w in windows], fontsize=7)
    ax.set_title(title, loc='left')

    # 分隔线
    ax.axvline(x=3.5, color='white', linewidth=1.5)
    ax.axvline(x=4.5, color='white', linewidth=1.5)

    # x 轴标签：名称 + N
    key_map = {'Daily': 'Daily', 'Onset': 'Onset', 'Dissip.': 'Dissipation',
               'Dur.': 'Duration', 'All': 'All', 'Sm': 'Small', 'Med': 'Medium',
               'Lg': 'Large', 'XL': 'XLarge'}
    xtick_labels = []
    for lbl in display_labels:
        key = key_map.get(lbl, lbl)
        n = n_records.get(key, None)
        if n is not None:
            xtick_labels.append(f'{lbl}\n({format_count(n)})')
        else:
            xtick_labels.append(lbl)
    # 8.0in figsize 下每 panel ~3.5in,9 列约 0.39in/列,fontsize 6 安全
    ax.set_xticklabels(xtick_labels, fontsize=5.7)
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False)

    # 分组标注
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, -0.2, 'Lifecycle', transform=trans,
            fontsize=5.8, ha='center', color='#C0392B', fontstyle='italic')
    ax.text(6.5, -0.2, 'Area Group', transform=trans,
            fontsize=5.8, ha='center', color='#3498DB', fontstyle='italic')

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    return im

# ────────────────────────────────────────────────────────────
# (a) R_C 热力图
# ────────────────────────────────────────────────────────────
draw_heatmap(axes[0, 0], hm_rc,
             r'(a) Conjunction ratio $R_{\mathrm{C}}/\mathrm{\%}$',
             'RdYlBu_r', 96, 112, 100, r'$R_{\mathrm{C}}/\mathrm{\%}$')

# ────────────────────────────────────────────────────────────
# (b) A = R_C - R_O 不对称度热力图
# ────────────────────────────────────────────────────────────
draw_heatmap(axes[0, 1], hm_a,
             r'(b) $\mathrm{Asym} = R_{\mathrm{C}} - R_{\mathrm{O}}$',
             'RdBu_r', -15, 15, 0, r'$\mathrm{Asym}/\mathrm{\%}$',
             fmt='{:+.1f}')

# ────────────────────────────────────────────────────────────
# (c) 单行星 R_C 排名
# ────────────────────────────────────────────────────────────
ax = axes[1, 0]

flare_sorted = flare_planet.sort_values('Conj_Ratio', ascending=False)
planet_order = flare_sorted['Planet'].tolist()

sg_dict = dict(zip(sg_planet['Planet'], sg_planet['Conj_Ratio']))
sg_daily_dict = dict(zip(sg_daily_planet['Planet'], sg_daily_planet['Conj_Ratio']))

x = np.arange(len(planet_order))
bar_width = 0.25

bars1 = ax.bar(x - bar_width, [flare_sorted[flare_sorted['Planet']==p]['Conj_Ratio'].values[0]
                                for p in planet_order],
               bar_width, color='#EA6759', label=r'Flare C ($w=2^\circ$)',
               edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax.bar(x, [sg_daily_dict.get(p, 100) for p in planet_order],
               bar_width, color='#52A7E0', label=r'SG Daily ($w=1^\circ$)',
               edgecolor='white', linewidth=0.5, zorder=3)
bars3 = ax.bar(x + bar_width, [sg_dict.get(p, 100) for p in planet_order],
               bar_width, color='#9FB4B8', label=r'SG All ($w=1^\circ$)',
               edgecolor='white', linewidth=0.5, zorder=3)

ax.axhline(y=100, color=COLOR_BASELINE, linewidth=0.8, linestyle='--', zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(planet_order, fontsize=6.5)
ax.set_ylabel(r'$R_{\mathrm{C}}/\mathrm{\%}$')
ax.set_ylim(88, 142)
ax.set_title(r'(c) Single-planet $R_{\mathrm{C}}$ ranking', loc='left')
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                ncol=3, framealpha=1.0, edgecolor=FRAME_EDGE, fontsize=6.1,
                handlelength=1.2, columnspacing=0.7, handletextpad=0.35)
leg.get_frame().set_linewidth(0.35)

# ────────────────────────────────────────────────────────────
# (d) ROC 可预测性对比
# ────────────────────────────────────────────────────────────
ax = axes[1, 1]

roc_cfg = roc_data['roc']
fpr_sfi = np.array(roc_cfg['sfi']['fpr'])
tpr_sfi = np.array(roc_cfg['sfi']['tpr'])
auc_sfi = roc_cfg['sfi']['auc']

fpr_ssn = np.array(roc_cfg['ssn']['fpr'])
tpr_ssn = np.array(roc_cfg['ssn']['tpr'])
auc_ssn = roc_cfg['ssn']['auc']

# 从 JSON 读取置换检验 p 值
perm_info = roc_data.get('permutation', {})
p_sfi = perm_info.get('sfi', {}).get('p', None)
p_ssn = perm_info.get('ssn', {}).get('p', None)

ax.plot(fpr_sfi, tpr_sfi, color='#E74C3C', linewidth=1.5,
        label=f'SFI (Flares) AUC={auc_sfi:.3f}', zorder=4)
ax.plot(fpr_ssn, tpr_ssn, color='#3498DB', linewidth=1.5,
        label=f'SSN (Sunspots) AUC={auc_ssn:.3f}', zorder=4)
ax.plot([0, 1], [0, 1], '--', color='lightgray', linewidth=0.8,
        label='Random (0.500)', zorder=2)

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_title('(d) Predictability: Planet Position → Burst', loc='left')
# Swap legend & annotation: ROC 曲线肩部贴近左上,原位置文字压数据;
# 改为图例占左上(只 3 条简短),注释挪到右下空白区(对角线下方)
leg = ax.legend(loc='upper left', framealpha=1.0,
                edgecolor=FRAME_EDGE, fontsize=6.2)
leg.get_frame().set_linewidth(0.35)

# 动态构建注释文字 — feature_set 与维度从 best_roc.json 实际配置读出
# (避免硬编码 "Top-3 + Angles (15D)" 与 best 选中的实际配置不一致)
_cfg = roc_data.get('config', {})
_FEATSET_LABELS = {
    'A_top3_pos':  ('Top-3 planets, positions only', 9),
    'B_all7_pos':  ('All 7 planets, positions only', 21),
    'C_top3_full': ('Top-3 planets + mutual angles', 15),
    'D_all7_full': ('All 7 planets + mutual angles', 63),
}
# Burst threshold tag (e.g. '15s' / '2s') is upstream code shorthand for
# the σ multiplier with the decimal point removed: '15s' → 1.5σ, '2s' → 2σ.
# Translate to a math-mode label so the figure does not read like a
# 15-second time threshold.
_BURST_LABELS = {'2s': r'$2\sigma$', '15s': r'$1.5\sigma$'}
_sfi_set = _cfg.get('sfi', {}).get('feature_set', '?')
_sfi_burst_raw = _cfg.get('sfi', {}).get('burst_threshold', '?')
_sfi_burst = _BURST_LABELS.get(_sfi_burst_raw, _sfi_burst_raw)
_label, _dim = _FEATSET_LABELS.get(_sfi_set, (_sfi_set, '?'))
feat_line = f'{_label} ({_dim}D); burst: {_sfi_burst}'

perm_line = ''
if p_sfi is not None and p_ssn is not None:
    perm_line = f'\nPerm $p$: SFI={p_sfi:.3f}, SSN={p_ssn:.3f}'
elif p_sfi is not None:
    perm_line = f'\nPerm $p$ = {p_sfi:.3f}'

ax.text(0.97, 0.04, f'LightGBM (CV-selected)\n{feat_line}\nTrain/Test: 1976–2010 / 2011–2024{perm_line}',
        transform=ax.transAxes, fontsize=5.7, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.24', facecolor='white',
                  edgecolor=FRAME_EDGE, linewidth=0.35))

plt.tight_layout(w_pad=2.0, h_pad=2.5)

# ============================================================
# 保存
# ============================================================
out_name = 'Fig05_sunspot_dilution'
eps_path = os.path.join(result_dir, f'{out_name}.eps')
save_dual(fig, eps_path)
print(f"\n已保存图片: {eps_path}")

# plt.show()

# ============================================================
# 保存图表背后的数据 (推荐使用 Excel 多 Sheet 格式)
# ============================================================
excel_path = os.path.join(result_dir, f'{out_name}.xlsx')

# 1. 整理 (a) 和 (b) 的热力图数据
df_hm_rc = pd.DataFrame(hm_rc, index=[f'w={w}°' for w in windows], columns=display_labels)
df_hm_a = pd.DataFrame(hm_a, index=[f'w={w}°' for w in windows], columns=display_labels)

# 2. 整理 (c) 单行星排名的柱状图数据
df_ranking = pd.DataFrame({
    'Planet': planet_order,
    'Flare_C (w=2)': [flare_sorted[flare_sorted['Planet']==p]['Conj_Ratio'].values[0] for p in planet_order],
    'SG_Daily (w=1)': [sg_daily_dict.get(p, 100) for p in planet_order],
    'SG_All (w=1)': [sg_dict.get(p, 100) for p in planet_order]
})

# 3. 整理 (d) ROC 曲线数据
df_roc = pd.concat([
    pd.Series(fpr_sfi, name='FPR_SFI'),
    pd.Series(tpr_sfi, name='TPR_SFI'),
    pd.Series(fpr_ssn, name='FPR_SSN'),
    pd.Series(tpr_ssn, name='TPR_SSN')
], axis=1)

# 4. 写入同一个 Excel 文件的不同 Sheet 中
try:
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_hm_rc.to_excel(writer, sheet_name='(a) RC_Heatmap')
        df_hm_a.to_excel(writer, sheet_name='(b) A_Heatmap')
        df_ranking.to_excel(writer, sheet_name='(c) Planet_Ranking', index=False)
        df_roc.to_excel(writer, sheet_name='(d) ROC_Curves', index=False)
    print(f"已保存图表数据: {excel_path}")
except ModuleNotFoundError:
    print("保存 Excel 失败: 缺少 openpyxl 库。请在终端运行 `pip install openpyxl` 后重试。")
