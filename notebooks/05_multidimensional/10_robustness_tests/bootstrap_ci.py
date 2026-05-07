#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap 置信区间(v3 修订:block bootstrap + k_exp 同步重抽)
==============================================================================
为关键行星组合的 CEOS Conj Ratio 计算 bootstrap 95% 置信区间。

v3 修订(F 项):
  1. 由 i.i.d. 重抽改为 *block bootstrap*(块长 14 天,对应 active region 寿命),
     保留耀斑事件在时间维上的强自相关。
  2. k_exp 不再固定从主 CSV 读取,而是按 Kepler 解析 per-event 概率随每次
     bootstrap 重抽的事件子样本同步重算 → CI 不再因分母固定而偏窄。

用法: python bootstrap_ci.py
"""

import sys, os, time, pickle
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
BLOCK_DAYS = 14   # active region 典型寿命

# 行星组合 (名称, mask) — 8P 编码:Mer=1, Ven=2, Earth=4 (排除), Mar=8, Jup=16, Sat=32, Ura=64, Nep=128
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


def _per_event_p_conj_opp(sun_lons, sun_idxs, prob_maps, w):
    """对每个事件,计算每颗行星出现在 Conj/Opp 窗口内的 Kepler 解析概率。

    Returns:
        p_conj: (N, P) ndarray, 每个事件每颗行星的 conj 概率(width 2w°)
        p_opp:  (N, P)
    """
    P = len(ce.PLANET_COLS)
    N = len(sun_lons)
    sun_int = np.floor(sun_lons).astype(int) % 360

    p_conj = np.zeros((N, P), dtype=np.float64)
    p_opp = np.zeros((N, P), dtype=np.float64)
    for j, col in enumerate(ce.PLANET_COLS):
        hist = np.asarray(prob_maps[col])  # (360,) density per degree
        # Match ceos_engine Kepler baseline convention: 2w one-degree bins.
        for d in range(-w, w):
            idx_c = (sun_int + d) % 360
            idx_o = (sun_int + 180 + d) % 360
            p_conj[:, j] += hist[idx_c]
            p_opp[:, j] += hist[idx_o]
    return p_conj, p_opp


def _mask_event_prob(p_per_planet, mask):
    """对 'at least one of mask planets' 语义,计算每事件的总概率。

    p_event = 1 - prod_{j in mask}(1 - p_planet_j)
    """
    P = p_per_planet.shape[1]
    keep = np.zeros_like(p_per_planet[:, 0])
    log1m = np.zeros_like(keep)
    for j in range(P):
        if mask & (1 << j):
            log1m += np.log1p(-np.clip(p_per_planet[:, j], 0.0, 0.999999))
    return 1.0 - np.exp(log1m)


def _build_blocks(dates, block_days):
    """按 block_days 将事件 index 按日期块分组。

    Returns: list[ndarray of event indices], 长度 = 块数
    """
    d0 = dates.min()
    block_id = ((dates - d0).dt.days // block_days).values
    order = np.argsort(block_id, kind='stable')
    sorted_block = block_id[order]
    # split points where block_id changes
    diffs = np.where(np.diff(sorted_block) != 0)[0] + 1
    splits = np.split(order, diffs)
    return splits


def block_bootstrap_ratio(df, ephem_8p, prob_maps, w, mask, n_boot, block_days, rng):
    """Block bootstrap: 按时间块重抽事件,k_exp 同步用 Kepler per-event 概率重算。

    Returns: dict with Conj/Opp Ratio + 95% CI lo/hi.
    """
    sun_lons = df['hme_lon'].values.astype(np.float64)
    sun_idxs = df['ephem_idx_daily'].values.astype(int)
    planet_matrix = df[ce.PLANET_COLS].values.astype(np.float64)
    N = len(sun_lons)
    P = 8

    # Bit vectors for observed conj/opp
    bits_c_all = np.zeros(N, dtype=np.uint16)
    bits_o_all = np.zeros(N, dtype=np.uint16)
    for i in range(P):
        delta = np.mod(sun_lons - planet_matrix[:, i] + 180, 360) - 180
        bits_c_all |= (np.abs(delta) <= w).astype(np.uint16) << i
        bits_o_all |= (np.abs(np.abs(delta) - 180) <= w).astype(np.uint16) << i

    mask_bits = np.uint16(mask)

    # Per-event Kepler analytic conj/opp probabilities (mask-specific)
    p_conj_planet, p_opp_planet = _per_event_p_conj_opp(sun_lons, sun_idxs, prob_maps, w)
    p_event_conj = _mask_event_prob(p_conj_planet, mask)   # (N,)
    p_event_opp = _mask_event_prob(p_opp_planet, mask)

    # Observed counts and analytic expectations on full sample
    k_obs_c = int(np.sum((bits_c_all & mask_bits) != 0))
    k_obs_o = int(np.sum((bits_o_all & mask_bits) != 0))
    k_exp_c_full = float(np.sum(p_event_conj))
    k_exp_o_full = float(np.sum(p_event_opp))

    obs_ratio_c = k_obs_c / k_exp_c_full * 100 if k_exp_c_full > 0 else 100.0
    obs_ratio_o = k_obs_o / k_exp_o_full * 100 if k_exp_o_full > 0 else 100.0

    # Build time blocks
    blocks = _build_blocks(df['date'], block_days)
    n_blocks = len(blocks)

    # Bootstrap loop
    boot_ratio_c = np.empty(n_boot, dtype=np.float64)
    boot_ratio_o = np.empty(n_boot, dtype=np.float64)
    bits_c_hit = ((bits_c_all & mask_bits) != 0).astype(np.int64)
    bits_o_hit = ((bits_o_all & mask_bits) != 0).astype(np.int64)

    for b in range(n_boot):
        # Sample blocks with replacement
        sel = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[s] for s in sel])

        kc = int(bits_c_hit[idx].sum())
        ko = int(bits_o_hit[idx].sum())
        kec = float(p_event_conj[idx].sum())
        keo = float(p_event_opp[idx].sum())

        boot_ratio_c[b] = (kc / kec * 100) if kec > 0 else 100.0
        boot_ratio_o[b] = (ko / keo * 100) if keo > 0 else 100.0

    ci_c = np.percentile(boot_ratio_c, [2.5, 97.5])
    ci_o = np.percentile(boot_ratio_o, [2.5, 97.5])

    return {
        'Conj_Ratio': round(obs_ratio_c, 2),
        'Conj_CI_lo': round(float(ci_c[0]), 2),
        'Conj_CI_hi': round(float(ci_c[1]), 2),
        'Opp_Ratio': round(obs_ratio_o, 2),
        'Opp_CI_lo': round(float(ci_o[0]), 2),
        'Opp_CI_hi': round(float(ci_o[1]), 2),
    }


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    print("加载耀斑数据 ...")
    df, ephem_daily = ce.load_flare_data(CACHE_SF, CACHE_SG)

    # Load Kepler prob_maps (per-planet ephemeris density)
    pkl_path = os.path.join(CACHE_SF, 'kepler_prob_maps.pkl')
    with open(pkl_path, 'rb') as f:
        prob_maps = pickle.load(f)
    print(f"加载 Kepler prob_maps: {len(prob_maps)} 颗行星")

    print(f"Block bootstrap: block_days={BLOCK_DAYS}, n_boot={N_BOOT}")
    rng = np.random.default_rng(SEED)

    results = []
    t0 = time.time()
    for name, mask in COMBOS:
        print(f"  Bootstrap {name} (mask={mask}) ...")
        res = block_bootstrap_ratio(df, ephem_daily, prob_maps,
                                    W, mask, N_BOOT, BLOCK_DAYS, rng)
        res['Combo'] = name
        res['Mask'] = mask
        results.append(res)
        print(f"    Conj: {res['Conj_Ratio']}% [{res['Conj_CI_lo']}, {res['Conj_CI_hi']}]")

    elapsed = time.time() - t0
    print(f"\nBootstrap 完成 ({elapsed:.1f}s)")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_ci.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    # Forest plot
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
    ax.set_title(f'Block Bootstrap 95% CI '
                 f'(N_boot={N_BOOT}, block={BLOCK_DAYS}d)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)

    for i, r in enumerate(results):
        ax.annotate(f'{r["Conj_Ratio"]:.1f}% [{r["Conj_CI_lo"]:.1f}, {r["Conj_CI_hi"]:.1f}]',
                    xy=(r['Conj_CI_hi'] + 1, i), fontsize=9, va='center')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figures', 'bootstrap_forest_plot.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图: {fig_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Block Bootstrap 95% CI 摘要 (w={W}°, N_boot={N_BOOT}, block={BLOCK_DAYS}d):")
    print(f"{'Combo':<20} {'Conj Ratio':>10} {'95% CI':>20} {'显著?':>6}")
    print("-" * 60)
    for r in results:
        sig = '✅' if r['Conj_CI_lo'] > 100 else '  '
        print(f'{r["Combo"]:<20} {r["Conj_Ratio"]:>9.1f}% '
              f'[{r["Conj_CI_lo"]:.1f}, {r["Conj_CI_hi"]:.1f}] {sig:>6}')


if __name__ == '__main__':
    main()
