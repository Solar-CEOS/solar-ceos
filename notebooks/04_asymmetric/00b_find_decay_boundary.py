#!/usr/bin/env python3
"""
00b_find_decay_boundary.py — CEOS 信号衰减边界探测
====================================================
对耀斑/黑子分别扫描 w=1~W_MAX, 用 CTS 模拟计算 Conjunction/Opposition
的 Ratio, Z-score, p 值, 找出信号消失的临界窗口。

输出:
  results/04_asymmetric/sf/sf_decay_boundary.csv              (耀斑 Total)
  results/04_asymmetric/sf/sf_b_class_decay_boundary.csv      (耀斑 B-Class)
  results/04_asymmetric/sf/sf_c_class_decay_boundary.csv      (耀斑 C-Class)
  results/04_asymmetric/sf/sf_m_class_decay_boundary.csv      (耀斑 M-Class)
  results/04_asymmetric/sf/sf_x_class_decay_boundary.csv      (耀斑 X-Class)
  results/04_asymmetric/sg/sg_decay_boundary.csv              (黑子 Total)
  results/04_asymmetric/sg/sg_small_lt100_decay_boundary.csv  (黑子 Small)
  results/04_asymmetric/sg/sg_medium_100-500_decay_boundary.csv (黑子 Medium)
  results/04_asymmetric/sg/sg_large_500-2000_decay_boundary.csv (黑子 Large)
  results/04_asymmetric/sg/sg_xlarge_gt2000_decay_boundary.csv  (黑子 XLarge)

用法:
  ~/miniforge3/envs/py313_tian_env/bin/python 00b_find_decay_boundary.py
"""

import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algo_workers

# 配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE_SG = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sg', 'cache_data')
BASE_SF = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sf', 'cache_data')
OUT_SG  = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sg')
OUT_SF  = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sf')
PLANET_COLS_7P = ['199_lon','299_lon','499_lon','599_lon','699_lon','799_lon','899_lon']

W_MAX_SF = 50     # 耀斑扫描到 w=50 (100°, ~7.6 天)
W_MAX_SG = 10     # 黑子扫描到 w=10 (无信号, 快速确认)
N_SIM = 10000     # 高精度, p_val 精度 ±0.01, 定位衰减边界
SOLAR_ROT = 13.2  # 太阳自转速率 (°/天)


def scan_decay(label, df_path, ephem_path, w_max, out_dir, prefix,
               df=None, ephem_8p=None):
    """对指定数据集扫描 w=1~w_max, 输出 CSV
    
    参数:
        df: 可选, 预加载的 DataFrame (避免重复读文件)
        ephem_8p: 可选, 预加载的 8P 星历矩阵
    """
    print(f"\n{'='*70}")
    print(f"  {label}: w=1-{w_max}, N_SIM={N_SIM}")
    print(f"{'='*70}")
    
    if df is None:
        df = pd.read_parquet(df_path)
    if ephem_8p is None:
        ephem_8p = np.load(ephem_path)
    
    # 7P: 去掉地球 (index 2)
    col_indices = [0,1,3,4,5,6,7]
    ephem_7p = ephem_8p[:, col_indices]
    
    sun_lons = df['hme_lon'].values.astype(np.float64)
    sun_idxs = df['ephem_idx_daily'].values.astype(int)
    planet_matrix = df[PLANET_COLS_7P].values.astype(np.float64)
    N = len(sun_lons)
    T = ephem_7p.shape[0]
    
    print(f"  数据量 N={N}, 星历天数 T={T}")
    print(f"\n  {'w':>3s} {'窗口':>6s} {'~天数':>6s} | {'Conj_Ratio':>10s} {'Z':>6s} {'p':>8s} | {'Opp_Ratio':>10s} {'Z':>6s} {'p':>8s}")
    print(f"  {'-'*3} {'-'*6} {'-'*6} | {'-'*10} {'-'*6} {'-'*8} | {'-'*10} {'-'*6} {'-'*8}")
    
    rng = np.random.default_rng(42)
    rows = []
    
    for w in range(1, w_max + 1):
        row = {'Window': w, 'Deg_Window': 2*w, 'Days': round(2*w/SOLAR_ROT, 2)}
        
        for etype in ['Conjunction', 'Opposition']:
            k_obs = algo_workers.count_events_vectorized(sun_lons, planet_matrix, w, etype)
            
            # 批量 CTS 模拟 (GPU 优先, 自动退回 CPU)
            k_sims = algo_workers.run_cts_simulation(
                sun_lons, ephem_7p, sun_idxs, w, etype,
                N_SIM, algo_type='algo1', n_workers=1, seed=42+w*100+( 0 if etype=='Conjunction' else 1))
            
            k_exp = k_sims.mean()
            ratio = k_obs / k_exp * 100 if k_exp > 0 else 0
            std = k_sims.std()
            z = (k_obs - k_exp) / std if std > 0 else 0
            pl = (np.sum(k_sims <= k_obs) + 1) / (N_SIM + 1)
            pr = (np.sum(k_sims >= k_obs) + 1) / (N_SIM + 1)
            p = min(2 * min(pl, pr), 1.0)
            
            pfx = 'Conj' if etype == 'Conjunction' else 'Opp'
            row[f'{pfx}_k_obs'] = int(k_obs)
            row[f'{pfx}_k_exp'] = round(k_exp, 1)
            row[f'{pfx}_Ratio'] = round(ratio, 2)
            row[f'{pfx}_Z'] = round(z, 2)
            row[f'{pfx}_p'] = round(p, 4)
        
        rows.append(row)
        
        c_sig = '**' if row['Conj_p'] < 0.01 else ('*' if row['Conj_p'] < 0.05 else '')
        o_sig = '**' if row['Opp_p'] < 0.01 else ('*' if row['Opp_p'] < 0.05 else '')
        print(f"  {w:>3d} {row['Deg_Window']:>5d}° {row['Days']:>5.1f}d | "
              f"{row['Conj_Ratio']:>8.1f}%  {row['Conj_Z']:>+5.1f} {row['Conj_p']:>7.4f}{c_sig:2s} | "
              f"{row['Opp_Ratio']:>8.1f}%  {row['Opp_Z']:>+5.1f} {row['Opp_p']:>7.4f}{o_sig:2s}")
    
    # 保存 CSV
    out_file = os.path.join(out_dir, f'{prefix}_decay_boundary.csv')
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"\n  保存: {out_file}")
    
    # 找显著区间和衰减临界点
    sig_windows = [r for r in rows if r['Conj_p'] < 0.05]
    if not sig_windows:
        print(f"  ⚠️  Conjunction 在 w=1-{w_max} 全程不显著, 无 CEOS 信号")
    else:
        w_first_sig = sig_windows[0]['Window']
        w_last_sig = sig_windows[-1]['Window']
        # 找连续显著区间结束后的首次不显著
        decay_w = None
        for r in rows:
            if r['Window'] > w_last_sig and r['Conj_p'] > 0.05:
                decay_w = r
                break
        if w_first_sig == 1:
            if decay_w:
                print(f"  📍 Conjunction 信号 w=1-{w_last_sig} 显著, 在 w={decay_w['Window']} ({decay_w['Deg_Window']}°, ~{decay_w['Days']}天) 首次不显著 (p={decay_w['Conj_p']:.4f})")
            else:
                print(f"  ✅ Conjunction 信号在 w=1-{w_max} 全程显著 (p<0.05)")
        else:
            if decay_w:
                print(f"  📍 Conjunction 信号 w={w_first_sig}-{w_last_sig} 显著 (w=1 不显著 p={rows[0]['Conj_p']:.4f}), 在 w={decay_w['Window']} ({decay_w['Deg_Window']}°, ~{decay_w['Days']}天) 衰减")
            else:
                print(f"  📍 Conjunction 信号 w={w_first_sig}-{w_max} 显著 (w=1 不显著 p={rows[0]['Conj_p']:.4f})")
    
    return rows


def main():
    t0 = time.time()
    
    # ── 耀斑 Total ──
    df_sf = pd.read_parquet(os.path.join(BASE_SF, 'ready_Flare_All.parquet'))
    ephem_8p_sf = np.load(os.path.join(BASE_SG, 'ephem_matrix_8p.npy'))
    
    scan_decay("Flare Total (SF)",
               None, None,
               W_MAX_SF, OUT_SF, 'sf',
               df=df_sf, ephem_8p=ephem_8p_sf)
    
    # ── 耀斑 按 Class 分组 ──
    if 'Group' in df_sf.columns:
        for class_name in ['B-Class', 'C-Class', 'M-Class', 'X-Class']:
            mask = df_sf['Group'] == class_name
            n_class = mask.sum()
            if n_class < 30:
                print(f"\n  [跳过] {class_name}: 仅 {n_class} 条, 不足 30")
                continue
            df_sub = df_sf[mask].copy()
            safe_name = class_name.replace('-', '_').lower()
            scan_decay(f"Flare {class_name} (N={n_class})",
                       None, None,
                       W_MAX_SF, OUT_SF, f'sf_{safe_name}',
                       df=df_sub, ephem_8p=ephem_8p_sf)
    
    # ── 黑子 Total ──
    df_sg = pd.read_parquet(os.path.join(BASE_SG, 'ready_All.parquet'))
    ephem_8p_sg = np.load(os.path.join(BASE_SG, 'ephem_matrix_8p.npy'))
    
    scan_decay("Sunspot Total (SG)",
               None, None,
               W_MAX_SG, OUT_SG, 'sg',
               df=df_sg, ephem_8p=ephem_8p_sg)
    
    # ── 黑子 按面积分组 ──
    if 'Group' in df_sg.columns:
        for group_name in ['Small <100', 'Medium 100-500', 'Large 500-2000', 'XLarge >2000']:
            mask = df_sg['Group'] == group_name
            n_group = mask.sum()
            if n_group < 30:
                print(f"\n  [跳过] {group_name}: 仅 {n_group} 条, 不足 30")
                continue
            df_sub = df_sg[mask].copy()
            safe_name = group_name.replace(' ', '_').replace('<', 'lt').replace('>', 'gt').lower()
            scan_decay(f"Sunspot {group_name} (N={n_group})",
                       None, None,
                       W_MAX_SG, OUT_SG, f'sg_{safe_name}',
                       df=df_sub, ephem_8p=ephem_8p_sg)
    
    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()

