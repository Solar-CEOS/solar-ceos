#!/usr/bin/env python3
"""
tidal_correlation.py — 潮汐力 vs CEOS 增强比率
================================================
分析 7 颗行星的 CEOS Conj Ratio 与平均潮汐力 M/r³ 的
相关性，验证潮汐触发机制假说。

用法: python tidal_correlation.py
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '04_asymmetric'))
import ceos_engine as ce

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'results', '05_multidimensional', '10_robustness_tests')
W = 2  # 窗口宽度

# IAU 行星参数
# 质量 (kg), 平均日心距 (AU), 质量比 (相对太阳)
PLANET_DATA = {
    'Mercury':  {'mass_kg': 3.301e23, 'mean_r_au': 0.387},
    'Venus':    {'mass_kg': 4.867e24, 'mean_r_au': 0.723},
    'Mars':     {'mass_kg': 6.417e23, 'mean_r_au': 1.524},
    'Jupiter':  {'mass_kg': 1.898e27, 'mean_r_au': 5.203},
    'Saturn':   {'mass_kg': 5.683e26, 'mean_r_au': 9.537},
    'Uranus':   {'mass_kg': 8.681e25, 'mean_r_au': 19.19},
    'Neptune':  {'mass_kg': 1.024e26, 'mean_r_au': 30.07},
}

PLANET_MASKS = {
    'Mercury': 1, 'Venus': 2, 'Mars': 8,
    'Jupiter': 16, 'Saturn': 32, 'Uranus': 64, 'Neptune': 128
}


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    # 读取子集扫描结果
    sf_csv = os.path.join(ce.DEFAULT_OUTPUT_SF, 'sf_subset_scan_no_earth.csv')
    df = pd.read_csv(sf_csv)
    w2 = df[df['Window'] == W]

    rows = []
    for name, mask in PLANET_MASKS.items():
        r = w2[w2['Mask'] == mask]
        if len(r) == 0:
            continue
        r = r.iloc[0]

        pd_info = PLANET_DATA[name]
        au_to_m = 1.496e11
        r_m = pd_info['mean_r_au'] * au_to_m
        tidal = pd_info['mass_kg'] / (r_m ** 3)  # M/r³

        rows.append({
            'Planet': name,
            'Mass_kg': pd_info['mass_kg'],
            'Mean_r_AU': pd_info['mean_r_au'],
            'Tidal_M_r3': tidal,
            'Log_Tidal': np.log10(tidal),
            'Conj_Ratio': r['Conj_Ratio'],
            'Conj_p': r['Conj_p'],
            'Opp_Ratio': r['Opp_Ratio'],
            'Asym': r['Asym_Amp'],
        })

    df_tidal = pd.DataFrame(rows)
    df_tidal = df_tidal.sort_values('Tidal_M_r3', ascending=False)

    print("=== 潮汐力 vs CEOS Conj Ratio ===")
    print(df_tidal[['Planet', 'Mean_r_AU', 'Log_Tidal', 'Conj_Ratio', 'Conj_p']].to_string(index=False))

    # 相关分析
    x = df_tidal['Log_Tidal'].values
    y = df_tidal['Conj_Ratio'].values

    spearman_r, spearman_p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)

    print(f"\nSpearman: r={spearman_r:.3f}, p={spearman_p:.4f}")
    print(f"Pearson:  r={pearson_r:.3f}, p={pearson_p:.4f}")

    # 保存
    df_tidal.to_csv(os.path.join(OUTPUT_DIR, 'tidal_correlation.csv'), index=False)

    # ============================
    # 绘图
    # ============================
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'Venus': '#e74c3c', 'Mars': '#e67e22', 'Jupiter': '#3498db',
              'Saturn': '#9b59b6', 'Mercury': '#1abc9c', 'Uranus': '#2ecc71',
              'Neptune': '#34495e'}

    for _, r in df_tidal.iterrows():
        c = colors.get(r['Planet'], 'gray')
        marker = '★' if r['Conj_p'] < 0.05 else 'o'
        size = 200 if r['Conj_p'] < 0.05 else 100
        edge = 'black' if r['Conj_p'] < 0.05 else 'gray'
        ax.scatter(r['Log_Tidal'], r['Conj_Ratio'], s=size, c=c,
                   edgecolors=edge, linewidth=2, zorder=5)
        ax.annotate(r['Planet'],
                    xy=(r['Log_Tidal'], r['Conj_Ratio']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=c)

    # 拟合线
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, '--', color='red', alpha=0.5, linewidth=1.5,
            label=f'Linear fit (slope={slope:.1f})')

    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='Baseline 100%')

    ax.set_xlabel('log₁₀(M/r³)  [kg/m³]', fontsize=13)
    ax.set_ylabel(f'CEOS Conjunction Ratio (%, w={W}°)', fontsize=13)
    ax.set_title(f'Tidal Force vs Conjunction Enhancement Ratio\n'
                 f'Spearman r={spearman_r:.3f} (p={spearman_p:.3f}), '
                 f'Pearson r={pearson_r:.3f} (p={pearson_p:.3f})',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figures', 'tidal_correlation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图: {fig_path}")


if __name__ == '__main__':
    main()
