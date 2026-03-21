#!/usr/bin/env python3
"""
02_compute_sf.py — 耀斑 CEOS 全量计算
=====================================
使用 results/04_asymmetric/sf/cache_data/ 中的预处理数据，计算:
  1. Algo 1 (Total Pairs): Flare_All × 分组(B/C/M/X/Total) × w=1-30 × 冲合
  2. Algo 2 (At Least One): 同上
  3. Algo 3 (单体 781):    w=1-10 × 冲合
  4. Kuiper 检验
  5. 255 子集扫描: 5a All Total + 5b 按 Class (B/C/M/X)
  6. 太阳周分段: 6a All Total + 6b 按 Class (B/C/M/X)

用法:
  ~/miniforge3/envs/py313_tian_env/bin/python 02_compute_sf.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceos_engine as ce

# =====================================================================
# 配置参数
# =====================================================================

# 正式版参数
N_SIM_ALGO12 = 50000     # Algo 1/2 CTS 模拟次数 (提高精度, 解决临界 p 值)
N_SIM_ALGO3 = 100        # ⚠️ 仅用于信息展示，Algo 3 使用解析二项检验，此值不影响结果
N_SIM_SUBSET = 50000     # 子集扫描 CTS 模拟次数 (与 Algo1/2 统一)

# 窗口范围 (耀斑信号持续到 w~30, Algo3 到 w=10)
THRESHOLDS_ALGO12 = list(range(1, 31))   # w=1-30
THRESHOLDS_ALGO3  = list(range(1, 11))   # w=1-10 (781天体)
THRESHOLDS_SUBSET = list(range(1, 6))    # w=1-5 (子集扫描)
THRESHOLDS_CYCLE  = list(range(1, 6))    # w=1-5 (太阳周)

# 并行
N_WORKERS = max(1, os.cpu_count() - 2)

# 路径
CACHE_DIR  = ce.DEFAULT_CACHE_SF
OUTPUT_DIR = ce.DEFAULT_OUTPUT_SF

# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 70)
    print("耀斑 CEOS 全量计算")
    print(f"模拟次数: Algo1/2={N_SIM_ALGO12}, Algo3={N_SIM_ALGO3}, Subset={N_SIM_SUBSET}")
    print(f"窗口: Algo1/2=1-30, Algo3=1-10, Subset/Cycle=1-5")
    print(f"并行核心数: {N_WORKERS}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()
    
    # 耀斑分组排序键 (B < C < M < X)
    flare_sort = lambda x: ('BCMX'.find(x[0]) if x[0] in 'BCMX' else 99, x)
    
    # 发现缓存文件
    stage_files = sorted([f for f in os.listdir(CACHE_DIR) 
                          if f.startswith('ready_') and f.endswith('.parquet')])
    print(f"\n找到 {len(stage_files)} 个阶段文件: {[f.replace('ready_','').replace('.parquet','') for f in stage_files]}")
    
    # --- Step 1: Algo 1 (Total Pairs) ---
    print("\n" + "=" * 60)
    print("Step 1: Algo 1 (Total Pairs) w=1-30")
    print("=" * 60)
    ce.run_algo12(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO12, N_SIM_ALGO12,
                  N_WORKERS, stage_files, flare_sort, algo_type='algo1')
    
    # --- Step 2: Algo 2 (At Least One) ---
    print("\n" + "=" * 60)
    print("Step 2: Algo 2 (At Least One) w=1-30")
    print("=" * 60)
    ce.run_algo12(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO12, N_SIM_ALGO12,
                  N_WORKERS, stage_files, flare_sort, algo_type='algo2')
    
    # --- Step 3: Algo 3 (单体 781) ---
    print("\n" + "=" * 60)
    print("Step 3: Algo 3 (单体 781 天体) w=1-10")
    print("=" * 60)
    algo3_file = ce.run_algo3(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO3,
                               stage_files, prefix='sf')
    if algo3_file and os.path.exists(algo3_file):
        print("\n  应用 FDR 校正...")
        ce.apply_fdr_to_algo3(algo3_file)
    
    # --- Step 4: Kuiper 检验 ---
    print("\n" + "=" * 60)
    print("Step 4: Kuiper 检验")
    print("=" * 60)
    ce.run_kuiper_test(CACHE_DIR, OUTPUT_DIR, stage_files, prefix='sf')
    
    # --- Step 5: 255 子集扫描 (分组) ---
    print("\n" + "=" * 60)
    print("Step 5: 255 子集扫描 (含/不含地球)")
    print("=" * 60)
    # 加载耀斑数据（使用 sg 星历矩阵）
    df_all, ephem_daily = ce.load_flare_data(CACHE_DIR, ce.DEFAULT_CACHE_SG)
    sun_lons = df_all['hme_lon'].values.astype(np.float64)
    sun_idxs = df_all['ephem_idx_daily'].values.astype(int)
    planet_matrix = df_all[ce.PLANET_COLS].values.astype(np.float64)
    
    # 5a. All Total
    print("  [5a] All Total ...")
    ce.run_subset_scan(sun_lons, sun_idxs, planet_matrix, ephem_daily,
                       OUTPUT_DIR, 'sf', THRESHOLDS_SUBSET, N_SIM_SUBSET)
    
    # 5b. 按 Class 分组扫描 (B/C/M/X-Class)
    for class_name in ['B-Class', 'C-Class', 'M-Class', 'X-Class']:
        print(f"  [5b] {class_name} ...")
        mask = df_all['Group'] == class_name
        if mask.sum() < 30:
            print(f"    [跳过] {class_name}: 仅 {mask.sum()} 条, 不足 30")
            continue
        g_lons = df_all.loc[mask, 'hme_lon'].values.astype(np.float64)
        g_idxs = df_all.loc[mask, 'ephem_idx_daily'].values.astype(int)
        g_planets = df_all.loc[mask, ce.PLANET_COLS].values.astype(np.float64)
        safe_name = class_name.replace('-', '_')
        ce.run_subset_scan(g_lons, g_idxs, g_planets, ephem_daily,
                           OUTPUT_DIR, f'sf_{safe_name}',
                           THRESHOLDS_SUBSET, N_SIM_SUBSET)
    
    # --- Step 6: 太阳周分段 (分组) ---
    print("\n" + "=" * 60)
    print("Step 6: 太阳周分段 (SC21-SC24)")
    print("=" * 60)
    cycles = ce.load_solar_cycles()
    # 筛选 SC21-SC24 (覆盖耀斑数据 1975-2017)
    cycles_sf = [(sc, s, e) for sc, s, e in cycles if int(sc[2:]) >= 21 and int(sc[2:]) <= 24]
    print(f"  覆盖太阳周: {[c[0] for c in cycles_sf]}")
    
    # 6a. All Total
    print("  [6a] All Total ...")
    ce.run_solar_cycle_analysis(df_all, ephem_daily, cycles_sf,
                                 OUTPUT_DIR, 'sf', THRESHOLDS_CYCLE)
    
    # 6b. 按 Class 分组分段 (B/C/M/X-Class)
    for class_name in ['B-Class', 'C-Class', 'M-Class', 'X-Class']:
        print(f"  [6b] {class_name} ...")
        mask = df_all['Group'] == class_name
        if mask.sum() < 30:
            print(f"    [跳过] {class_name}: 仅 {mask.sum()} 条, 不足 30")
            continue
        df_class = df_all[mask].copy()
        safe_name = class_name.replace('-', '_')
        ce.run_solar_cycle_analysis(df_class, ephem_daily, cycles_sf,
                                     OUTPUT_DIR, f'sf_{safe_name}',
                                     THRESHOLDS_CYCLE)
    
    # --- Step 7: 太阳周 × 255 子集扫描 ---
    print("\n" + "=" * 60)
    print("Step 7: 太阳周 × 255 子集扫描 (CTS)")
    print("=" * 60)
    
    # 7a. All Total
    print("  [7a] All Total ...")
    ce.run_solar_cycle_subset_scan(df_all, ephem_daily, cycles_sf,
                                    OUTPUT_DIR, 'sf',
                                    thresholds_subset=[1,2,3], n_sim=5000)
    
    # 7b. C-Class (核心信号来源)
    print("  [7b] C-Class ...")
    mask_c = df_all['Group'] == 'C-Class'
    if mask_c.sum() >= 100:
        df_c = df_all[mask_c].copy()
        ce.run_solar_cycle_subset_scan(df_c, ephem_daily, cycles_sf,
                                        OUTPUT_DIR, 'sf_C_Class',
                                        thresholds_subset=[1,2,3], n_sim=5000)
    
    # 完成
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"全部计算完成! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print("=" * 70)
    
    # 列出输出文件
    print("\n输出文件清单:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
