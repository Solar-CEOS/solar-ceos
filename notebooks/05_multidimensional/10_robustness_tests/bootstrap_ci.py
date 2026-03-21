#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap 置信区间
======================================
为关键行星组合的 CEOS Conj Ratio 计算
Bootstrap 95% 置信区间。

用法: python bootstrap_ci.py
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '04_asymmetric'))
import ceos_engine as ce

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'results', '05_multidimensional', '10_robustness_tests')
CACHE_SF = ce.DEFAULT_CACHE_SF
CACHE_SG = ce.DEFAULT_CACHE_SG

N_BOOT = 10000
W = 2
SEED = 42

# 行星组合 (名称, mask)
COMBOS = [
    ('Venus', 2),
    ('Mars', 8),
    ('Jupiter', 16),
    ('Saturn', 32),
    ('Ven+Mar', 10),
    ('Ven+Mar+Jup', 26),
    ('Ven+Mar+Jup+Sat', 58),
    ('All 5', 59),
]


def bootstrap_ratio(sun_lons, sun_idxs, planet_matrix, ephem_8p, w, mask, n_boot, rng):
    """Bootstrap: 有放回重采样事件索引，重新计算 Conj/Opp Ratio。"""
    N = len(sun_lons)
    T = ephem_8p.shape[0]
    P = 8

    # 预计算所有事件的 bit vectors
    bits_c_all = np.zeros(N, dtype=np.uint16)
    bits_o_all = np.zeros(N, dtype=np.uint16)
    for i in range(P):
        planet_lon = planet_matrix[:, i]
        delta = np.mod(sun_lons - planet_lon + 180, 360) - 180
        bits_c_all |= (np.abs(delta) <= w).astype(np.uint16) << i
        bits_o_all |= (np.abs(np.abs(delta) - 180) <= w).astype(np.uint16) << i

    mask_bits = np.uint16(mask)

    # 观测值
    k_obs_c = int(np.sum((bits_c_all & mask_bits) != 0))
    k_obs_o = int(np.sum((bits_o_all & mask_bits) != 0))

    # 用 CTS 均值作为 k_exp (从 subset scan CSV 读取更准确)
    sf_csv = os.path.join(ce.DEFAULT_OUTPUT_SF, 'sf_subset_scan_no_earth.csv')
    df_scan = pd.read_csv(sf_csv)
    row = df_scan[(df_scan['Window'] == w) & (df_scan['Mask'] == mask)]
    if len(row) > 0:
        k_exp_c = row.iloc[0]['Conj_k_exp']
        k_exp_o = row.iloc[0]['Opp_k_exp']
    else:
        k_exp_c = k_obs_c  # fallback
        k_exp_o = k_obs_o

    # Bootstrap
    boot_ratio_c = np.zeros(n_boot)
    boot_ratio_o = np.zeros(n_boot)

    for b in range(n_boot):
        # 有放回重采样事件索引
        idx = rng.choice(N, size=N, replace=True)
        bc = bits_c_all[idx]
        bo = bits_o_all[idx]
        kc = int(np.sum((bc & mask_bits) != 0))
        ko = int(np.sum((bo & mask_bits) != 0))
        boot_ratio_c[b] = kc / k_exp_c * 100 if k_exp_c > 0 else 100
        boot_ratio_o[b] = ko / k_exp_o * 100 if k_exp_o > 0 else 100

    obs_ratio_c = k_obs_c / k_exp_c * 100 if k_exp_c > 0 else 100
    obs_ratio_o = k_obs_o / k_exp_o * 100 if k_exp_o > 0 else 100

    ci_c = np.percentile(boot_ratio_c, [2.5, 97.5])
    ci_o = np.percentile(boot_ratio_o, [2.5, 97.5])

    return {
        'Conj_Ratio': round(obs_ratio_c, 2),
        'Conj_CI_lo': round(ci_c[0], 2),
        'Conj_CI_hi': round(ci_c[1], 2),
        'Opp_Ratio': round(obs_ratio_o, 2),
        'Opp_CI_lo': round(ci_o[0], 2),
        'Opp_CI_hi': round(ci_o[1], 2),
    }


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("加载耀斑数据 ...")
    df, ephem_daily = ce.load_flare_data(CACHE_SF, CACHE_SG)
    sun_lons = df['hme_lon'].values.astype(np.float64)
    sun_idxs = df['ephem_idx_daily'].values.astype(int)
    planet_matrix = df[ce.PLANET_COLS].values.astype(np.float64)

    results = []
    t0 = time.time()
    for name, mask in COMBOS:
        print(f"  Bootstrap {name} (mask={mask}) ...")
        res = bootstrap_ratio(sun_lons, sun_idxs, planet_matrix, ephem_daily,
                              W, mask, N_BOOT, rng)
        res['Combo'] = name
        res['Mask'] = mask
        results.append(res)
        print(f"    Conj: {res['Conj_Ratio']}% [{res['Conj_CI_lo']}, {res['Conj_CI_hi']}]")

    elapsed = time.time() - t0
    print(f"\nBootstrap 完成 ({elapsed:.1f}s)")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_ci.csv')
    df_results.to_csv(csv_path, index=False)

    # ============================
    # 绘图: 森林图 (Forest Plot)
    # ============================
    fig, ax = plt.subplots(figsize=(10, 6))

    n = len(results)
    y_pos = np.arange(n)

    for i, r in enumerate(results):
        ci_lo = r['Conj_CI_lo']
        ci_hi = r['Conj_CI_hi']
        ratio = r['Conj_Ratio']
        color = '#e74c3c' if ci_lo > 100 else '#3498db'

        ax.errorbar(ratio, i, xerr=[[ratio - ci_lo], [ci_hi - ratio]],
                    fmt='o', color=color, markersize=8, capsize=5,
                    linewidth=2, markeredgecolor='black', markeredgewidth=1)

    ax.axvline(100, color='gray', linestyle='--', linewidth=1, label='Baseline 100%')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['Combo'] for r in results], fontsize=11)
    ax.set_xlabel(f'Conjunction Ratio (%, w={W}°)', fontsize=13)
    ax.set_title(f'Bootstrap 95% CI (N_boot={N_BOOT})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)

    # 添加数值标注
    for i, r in enumerate(results):
        ax.annotate(f'{r["Conj_Ratio"]:.1f}% [{r["Conj_CI_lo"]:.1f}, {r["Conj_CI_hi"]:.1f}]',
                    xy=(r['Conj_CI_hi'] + 1, i), fontsize=9, va='center')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figures', 'bootstrap_forest_plot.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图: {fig_path}")

    # 打印摘要
    print(f"\n{'='*70}")
    print(f"Bootstrap 95% CI 摘要 (w={W}°, N_boot={N_BOOT}):")
    print(f"{'Combo':<20} {'Conj Ratio':>10} {'95% CI':>20} {'显著?':>6}")
    print("-" * 60)
    for r in results:
        sig = '✅' if r['Conj_CI_lo'] > 100 else '  '
        print(f'{r["Combo"]:<20} {r["Conj_Ratio"]:>9.1f}% '
              f'[{r["Conj_CI_lo"]:.1f}, {r["Conj_CI_hi"]:.1f}] {sig:>6}')


if __name__ == '__main__':
    main()
