#!/usr/bin/env python3
"""
null_planet_test.py — 随机行星零检验
=====================================
生成 N_FAKE=100 颗虚拟行星（随机轨道周期），运行 CEOS
单体分析并与真实行星对比，验证真实行星的显著性不是
多重比较偶然。

用法: python null_planet_test.py
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 添加 ceos 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '04_asymmetric'))
import ceos_engine as ce

# ============================
# 配置
# ============================
N_FAKE = 100          # 虚拟行星数量
N_SIM = 1000          # 每颗假行星的 CTS 模拟次数 (快速)
W = 2                 # 窗口宽度
SEED = 42
CACHE_SF = ce.DEFAULT_CACHE_SF
CACHE_SG = ce.DEFAULT_CACHE_SG
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'results', '05_multidimensional', '10_robustness_tests')

# 真实行星参数 (名称, 平均周期天数)
REAL_PLANETS = {
    'Mercury': 87.97, 'Venus': 224.70, 'Mars': 686.97,
    'Jupiter': 4332.59, 'Saturn': 10759.22,
    'Uranus': 30688.5, 'Neptune': 60182.0
}
# 对应的子集扫描 mask (不含 Earth bit=4)
PLANET_MASKS = {
    'Mercury': 1, 'Venus': 2, 'Mars': 8,
    'Jupiter': 16, 'Saturn': 32, 'Uranus': 64, 'Neptune': 128
}


def compute_fake_ceos(sun_lons, sun_idxs, ephem_daily, period_days, w, n_sim, rng):
    """对一颗假行星做 CEOS 单体分析。"""
    T = ephem_daily.shape[0]
    N = len(sun_lons)

    # 生成假行星经度: L_fake(t) = (360 * t / P) mod 360
    t_indices = sun_idxs.astype(np.float64)
    fake_lon = np.mod(360.0 * t_indices / period_days, 360.0).astype(np.float64)

    # 计算相位差
    delta = np.mod(sun_lons - fake_lon + 180, 360) - 180

    # 观测 k_obs
    conj_obs = int(np.sum(np.abs(delta) <= w))
    opp_obs = int(np.sum(np.abs(np.abs(delta) - 180) <= w))

    # CTS 模拟
    conj_sims = np.zeros(n_sim)
    opp_sims = np.zeros(n_sim)
    for i in range(n_sim):
        shift = rng.integers(0, T)
        si = (sun_idxs + shift) % T
        fake_lon_shifted = np.mod(360.0 * si.astype(np.float64) / period_days, 360.0)
        d = np.mod(sun_lons - fake_lon_shifted + 180, 360) - 180
        conj_sims[i] = np.sum(np.abs(d) <= w)
        opp_sims[i] = np.sum(np.abs(np.abs(d) - 180) <= w)

    # p 值
    def _pval(k_obs, k_sims):
        pl = (np.sum(k_sims <= k_obs) + 1) / (n_sim + 1)
        pr = (np.sum(k_sims >= k_obs) + 1) / (n_sim + 1)
        return min(2 * min(pl, pr), 1.0)

    conj_exp = float(conj_sims.mean())
    opp_exp = float(opp_sims.mean())
    conj_ratio = conj_obs / conj_exp * 100 if conj_exp > 0 else 100
    opp_ratio = opp_obs / opp_exp * 100 if opp_exp > 0 else 100
    conj_p = _pval(conj_obs, conj_sims)
    opp_p = _pval(opp_obs, opp_sims)

    return {
        'Period_days': period_days,
        'Conj_Ratio': round(conj_ratio, 2),
        'Conj_p': round(conj_p, 4),
        'Opp_Ratio': round(opp_ratio, 2),
        'Opp_p': round(opp_p, 4),
        'Asym': round(conj_ratio - opp_ratio, 2),
    }


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 加载耀斑数据
    print("加载耀斑数据 ...")
    df, ephem_daily = ce.load_flare_data(CACHE_SF, CACHE_SG)
    sun_lons = df['hme_lon'].values.astype(np.float64)
    sun_idxs = df['ephem_idx_daily'].values.astype(int)

    # 真实行星结果 (从 subset scan CSV 读取)
    sf_csv = os.path.join(ce.DEFAULT_OUTPUT_SF, 'sf_subset_scan_no_earth.csv')
    real_df = pd.read_csv(sf_csv)
    real_w2 = real_df[real_df['Window'] == W]

    real_results = []
    for name, mask in PLANET_MASKS.items():
        row = real_w2[real_w2['Mask'] == mask]
        if len(row) > 0:
            r = row.iloc[0]
            real_results.append({
                'Name': name, 'Period_days': REAL_PLANETS[name],
                'Conj_Ratio': r['Conj_Ratio'], 'Conj_p': r['Conj_p'],
                'Opp_Ratio': r['Opp_Ratio'], 'Opp_p': r['Opp_p'],
                'Asym': r['Asym_Amp'],
            })

    # 生成假行星并计算 CEOS
    print(f"生成 {N_FAKE} 颗假行星 (周期 50-5000 天)，CTS N={N_SIM} ...")
    fake_periods = rng.uniform(50, 5000, size=N_FAKE)
    fake_results = []
    t0 = time.time()
    for i, period in enumerate(fake_periods):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{N_FAKE}] 已完成 ({elapsed:.1f}s)")
        res = compute_fake_ceos(sun_lons, sun_idxs, ephem_daily,
                                period, W, N_SIM, rng)
        res['Name'] = f'Fake_{i+1:03d}'
        fake_results.append(res)

    elapsed = time.time() - t0
    print(f"假行星计算完成 ({elapsed:.1f}s)")

    # 保存 CSV
    df_fake = pd.DataFrame(fake_results)
    df_real = pd.DataFrame(real_results)
    df_fake.to_csv(os.path.join(OUTPUT_DIR, 'null_planet_fake_results.csv'), index=False)
    df_real.to_csv(os.path.join(OUTPUT_DIR, 'null_planet_real_results.csv'), index=False)

    # 统计
    n_sig_fake = (df_fake['Conj_p'] < 0.05).sum()
    print(f"\n假行星中 p<0.05: {n_sig_fake}/{N_FAKE} ({n_sig_fake/N_FAKE*100:.1f}%)")
    print(f"预期 (随机): ~5%")

    # ============================
    # 绘图
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: Conj p 值直方图
    ax = axes[0]
    ax.hist(df_fake['Conj_p'], bins=20, range=(0, 1), color='#7f8c8d',
            alpha=0.7, edgecolor='white', label=f'Fake planets (N={N_FAKE})')
    ax.axhline(N_FAKE / 20, color='red', linestyle='--', alpha=0.5,
               label='Uniform expectation')
    ax.axvline(0.05, color='orange', linestyle=':', linewidth=2,
               label='p=0.05 threshold')

    # 标注真实行星
    for _, r in df_real.iterrows():
        color = 'red' if r['Conj_p'] < 0.05 else 'blue'
        ax.axvline(r['Conj_p'], color=color, linewidth=2, alpha=0.8)
        ax.annotate(r['Name'], xy=(r['Conj_p'], ax.get_ylim()[1] * 0.9),
                    fontsize=8, rotation=45, ha='left', color=color, fontweight='bold')

    ax.set_xlabel('Conjunction p-value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Null Planet Test: Conjunction p-values (w={W}°)', fontsize=13)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 右图: Conj Ratio 分布
    ax = axes[1]
    ax.hist(df_fake['Conj_Ratio'], bins=25, color='#7f8c8d',
            alpha=0.7, edgecolor='white', label=f'Fake planets (N={N_FAKE})')
    ax.axvline(100, color='red', linestyle='--', alpha=0.5, label='Baseline 100%')

    for _, r in df_real.iterrows():
        color = 'red' if r['Conj_p'] < 0.05 else 'blue'
        ax.axvline(r['Conj_Ratio'], color=color, linewidth=2, alpha=0.8)
        ax.annotate(r['Name'], xy=(r['Conj_Ratio'], ax.get_ylim()[1] * 0.9),
                    fontsize=8, rotation=45, ha='left', color=color, fontweight='bold')

    ax.set_xlabel('Conjunction Ratio (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Null Planet Test: Conjunction Ratios (w={W}°)', fontsize=13)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figures', 'null_planet_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图: {fig_path}")

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"零检验结论:")
    print(f"  假行星 Conj p<0.05: {n_sig_fake}/{N_FAKE} ({n_sig_fake/N_FAKE*100:.1f}%)")
    print(f"  真实行星 Conj p<0.05:")
    for _, r in df_real.iterrows():
        flag = '✅' if r['Conj_p'] < 0.05 else '  '
        print(f"    {flag} {r['Name']}: R={r['Conj_Ratio']:.1f}%, p={r['Conj_p']:.4f}")


if __name__ == '__main__':
    main()
